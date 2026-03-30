"""
FastAPI application for the video processing pipeline.
Input: YouTube URL or Google Drive link → Output: PDF report (transcript + enhanced diagrams + summary).

Supports:
- Synchronous: POST /process-youtube (single long request; can hit tunnel/timeouts).
- Async poll: POST /process-youtube/submit → GET /job/{id} → GET /job/{id}/result (requires n8n timeout >= pipeline time).
- Async callback: POST /process-youtube/submit with callback_url → FastAPI POSTs result to that URL when done (no n8n timeout).
"""
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
import urllib.request
import ssl
import base64
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

# Import pipeline and downloader after env is loaded
import config

from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Logging (production-safe: no sensitive data in messages)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("lecture-pipeline")

# ---------------------------------------------------------------------------
# Job store for async mode (in-memory; single process)
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# Microsoft Graph / SharePoint configuration (for uploading result ZIP)
# ---------------------------------------------------------------------------

GRAPH_TENANT_ID = os.getenv("MS_TENANT_ID")
GRAPH_CLIENT_ID = os.getenv("MS_CLIENT_ID")
GRAPH_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
GRAPH_SITE_ID = os.getenv("MS_SITE_ID")
GRAPH_DRIVE_ID = os.getenv("MS_DRIVE_ID")
GRAPH_SENDER_EMAIL = os.getenv("MS_SENDER_EMAIL")

GRAPH_TOKEN_URL = (
    f"https://login.microsoftonline.com/{GRAPH_TENANT_ID}/oauth2/v2.0/token"
    if GRAPH_TENANT_ID
    else None
)


def _get_graph_access_token() -> str:
    """Get an app-only access token for Microsoft Graph using client credentials."""
    if not (GRAPH_TENANT_ID and GRAPH_CLIENT_ID and GRAPH_CLIENT_SECRET):
        raise RuntimeError(
            "Microsoft Graph client credentials are not fully configured. "
            "Expected MS_TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET environment variables."
        )
    if not GRAPH_TOKEN_URL:
        raise RuntimeError("GRAPH_TOKEN_URL is not configured.")

    data = {
        "client_id": GRAPH_CLIENT_ID,
        "client_secret": GRAPH_CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    resp = requests.post(GRAPH_TOKEN_URL, data=data)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to obtain Graph access token: {_sanitize_error_message(str(e))}") from e
    body = resp.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError("Graph token response missing access_token")
    return token


def _upload_zip_to_sharepoint(zip_bytes: bytes, file_name: str) -> str:
    """Upload the ZIP bytes to the configured SharePoint drive under Resources/ and return the webUrl."""
    if not (GRAPH_SITE_ID and GRAPH_DRIVE_ID):
        raise RuntimeError(
            "SharePoint site/drive not configured. Expected MS_SITE_ID and MS_DRIVE_ID environment variables."
        )

    token = _get_graph_access_token()

    # Upload to /Resources/{file_name}
    upload_url = (
        f"https://graph.microsoft.com/v1.0/"
        f"sites/{GRAPH_SITE_ID}/drives/{GRAPH_DRIVE_ID}/root:/Resources/{file_name}:/content"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/zip",
    }
    resp = requests.put(upload_url, headers=headers, data=zip_bytes)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to upload ZIP to SharePoint: {_sanitize_error_message(str(e))}") from e

    item = resp.json()
    web_url = item.get("webUrl")
    if not web_url:
        raise RuntimeError("SharePoint response missing webUrl")
    return web_url


def _send_completion_email(recipient_email: str, subject: str, body_html: str) -> None:
    """Send a simple completion email via Microsoft Graph. Never raises; logs on failure."""
    if not recipient_email or "@" not in recipient_email:
        return
    if not GRAPH_SENDER_EMAIL:
        logger.warning("MS_SENDER_EMAIL not set; skipping completion email to %s", recipient_email)
        return
    try:
        token = _get_graph_access_token()
    except Exception as e:
        logger.warning("Could not get Graph token for email: %s", e)
        return

    send_url = f"https://graph.microsoft.com/v1.0/users/{GRAPH_SENDER_EMAIL}/sendMail"
    message = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": body_html,
            },
            "toRecipients": [
                {"emailAddress": {"address": recipient_email}},
            ],
        },
        "saveToSentItems": False,
    }
    try:
        resp = requests.post(
            send_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=message,
            timeout=30,
        )
        resp.raise_for_status()
        logger.info("Completion email sent to %s", recipient_email)
    except Exception as e:
        logger.warning("Failed to send completion email to %s: %s", recipient_email, e)


def _is_youtube_url(s: str) -> bool:
    raw = (s or "").strip()
    return "youtube.com" in raw or "youtu.be" in raw


def _is_drive_url(s: str) -> bool:
    raw = (s or "").strip()
    if "drive.google.com" in raw:
        return True
    if raw and len(raw) >= 20 and "/" not in raw and " " not in raw:
        return True
    return False


def _is_onedrive_sharepoint_url(s: str) -> bool:
    """Recognize OneDrive/SharePoint share links."""
    raw = (s or "").strip().lower()
    return any(
        token in raw
        for token in (
            "1drv.ms/",
            "onedrive.live.com/",
            ".sharepoint.com/",
        )
    )


def _download_onedrive_sharepoint_video(
    share_url: str,
    output_dir: str,
    output_filename: str = "video",
) -> tuple[str, str]:
    """
    Download a video from OneDrive/SharePoint share link using Microsoft Graph /shares API.

    Returns:
        (video_path, video_title)
    """
    raw = (share_url or "").strip()
    if not raw:
        raise ValueError("OneDrive/SharePoint URL is required")

    os.makedirs(output_dir, exist_ok=True)
    token = _get_graph_access_token()
    auth_headers = {"Authorization": f"Bearer {token}"}

    # Graph shared-link format: u!{base64url(raw_url_without_padding)}
    share_id = "u!" + base64.urlsafe_b64encode(raw.encode("utf-8")).decode("utf-8").rstrip("=")
    item_url = f"https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem"
    item_resp = requests.get(item_url, headers=auth_headers, timeout=30)
    try:
        item_resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve OneDrive/SharePoint link via Graph: {_sanitize_error_message(str(e))}"
        ) from e

    item = item_resp.json() or {}
    original_name = (item.get("name") or output_filename or "video").strip()
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", original_name).strip().strip(". ") or f"{output_filename}.mp4"
    title = os.path.splitext(safe_name)[0] or output_filename
    save_path = os.path.abspath(os.path.join(output_dir, safe_name))

    # Prefer pre-authenticated download URL if Graph provides it.
    download_url = item.get("@microsoft.graph.downloadUrl")
    if download_url:
        dl_resp = requests.get(download_url, timeout=600, stream=True)
    else:
        content_url = f"https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem/content"
        dl_resp = requests.get(content_url, headers=auth_headers, timeout=600, stream=True)
    try:
        dl_resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download OneDrive/SharePoint file: {_sanitize_error_message(str(e))}"
        ) from e

    with open(save_path, "wb") as f:
        for chunk in dl_resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if not os.path.isfile(save_path) or os.path.getsize(save_path) <= 0:
        raise RuntimeError("OneDrive/SharePoint download produced an empty file")

    return save_path, title


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load env and ensure temp directory exists. Do not crash on startup errors."""
    try:
        logger.info("Pipeline API starting")
    except Exception as e:
        logger.exception("Startup warning: %s", e)
    yield
    try:
        logger.info("Pipeline API shutting down")
    except Exception:
        pass


app = FastAPI(
    title="Lecture Video Pipeline API",
    description="Download a video from YouTube or Google Drive, extract diagrams, transcribe, enhance, generate summary, and return a PDF.",
    version="1.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handlers (prevent crashes, return safe error responses)
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Return 422 with clear validation errors; do not leak internal structure."""
    errors = exc.errors()
    details = [f"{e.get('loc', [])}: {e.get('msg', 'invalid')}" for e in errors[:5]]
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": details},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    """Pass through HTTP exceptions as-is."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail} if isinstance(exc.detail, str) else {"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch any unhandled exception; log it; return 500 with a safe message (no stack trace)."""
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again or contact support.",
            "error_id": str(uuid.uuid4())[:8],
        },
    )


class ProcessYouTubeRequest(BaseModel):
    """Request body for /process-youtube and /process-youtube/submit."""

    youtube_url: Optional[str] = None
    drive_url: Optional[str] = None
    video_url: Optional[str] = None
    title: Optional[str] = None
    user_email: Optional[str] = None


class SubmitRequest(ProcessYouTubeRequest):
    """Submit body; optional callback_url for long-running jobs (avoids n8n execution timeout)."""

    callback_url: Optional[str] = None
    """When set, FastAPI POSTs job result (PDF or error) to this URL when done. Use an n8n webhook URL."""


def _sanitize_error_message(msg: str, max_len: int = 500) -> str:
    """Remove ANSI codes and truncate for safe storage/response."""
    if not msg:
        return "Unknown error"
    import re
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", str(msg))
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3] + "..."
    return cleaned


def _run_pipeline(video_path: str, output_dir: str, video_title: Optional[str] = None) -> dict:
    """Run full pipeline; returns dict with transcript_html, explanation_html, transcript_doc, explanation_doc, enhanced_diagrams_dir. Raises on failure."""
    try:
        from main import process_video_with_transcription
    except ImportError as e:
        raise RuntimeError(f"Pipeline module unavailable: {_sanitize_error_message(str(e))}") from e

    try:
        result = process_video_with_transcription(
            video_path,
            output_dir=output_dir,
            enable_transcription=True,
            enable_enhancement=True,
            enable_summary=False,
            enable_output_docs=True,
            skip_diagram_extraction=False,
            video_title=video_title,
        )
        return result
    except Exception as e:
        logger.exception("Pipeline failed for %s", video_path)
        raise RuntimeError(_sanitize_error_message(str(e))) from e


def _send_callback(
    callback_url: str,
    job_id: str,
    status: str,
    error: Optional[str] = None,
    pdf_transcript_path: Optional[str] = None,
    pdf_explanation_path: Optional[str] = None,
) -> None:
    """POST job result to callback_url as JSON. Never raises; logs on failure."""
    import json
    import base64
    try:
        url = callback_url.replace("/webhook-test/", "/webhook/", 1)
        payload = {"job_id": job_id, "status": status, "error": error or ""}
        if pdf_transcript_path and os.path.isfile(pdf_transcript_path):
            with open(pdf_transcript_path, "rb") as f:
                payload["pdf_transcript_base64"] = base64.b64encode(f.read()).decode("ascii")
        if pdf_explanation_path and os.path.isfile(pdf_explanation_path):
            with open(pdf_explanation_path, "rb") as f:
                payload["pdf_explanation_base64"] = base64.b64encode(f.read()).decode("ascii")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            resp.read()
        logger.info("Callback POST %s -> %s", url, resp.status)
    except Exception as e:
        logger.warning("Callback failed for job %s: %s", job_id, e)


def _job_worker(job_id: str, video_url: str, callback_url: Optional[str] = None) -> None:
    """Background thread: download video, run pipeline, update job state. Never raises."""
    work_dir = None
    try:
        work_dir = tempfile.mkdtemp(prefix="pipeline_")
        with _jobs_lock:
            if job_id not in _jobs:
                return
            _jobs[job_id]["work_dir"] = work_dir
            _jobs[job_id]["status"] = JOB_STATUS_RUNNING

        try:
            # Prefer an explicit title from the job record (submitted by frontend)
            # and do NOT override it with downloader-derived titles. This guarantees
            # that the ZIP and all document names match the user-provided title.
            with _jobs_lock:
                job_meta = (_jobs.get(job_id, {}) or {})
                explicit_title = job_meta.get("title")
                user_email = job_meta.get("user_email")

            video_title: Optional[str] = explicit_title
            if _is_youtube_url(video_url):
                from youtube_downloader import download_youtube_video

                video_path, _ = download_youtube_video(
                    video_url,
                    output_dir=work_dir,
                    output_filename="video",
                    best_quality=True,
                )
            elif _is_drive_url(video_url):
                from drive_downloader import download_drive_video

                video_path, _ = download_drive_video(
                    video_url,
                    output_dir=work_dir,
                    output_filename="video",
                )
            elif _is_onedrive_sharepoint_url(video_url):
                video_path, _ = _download_onedrive_sharepoint_video(
                    video_url,
                    output_dir=work_dir,
                    output_filename="video",
                )
            else:
                raise ValueError("URL must be YouTube, Google Drive, OneDrive, or SharePoint")

            if not video_path or not os.path.isfile(video_path):
                raise RuntimeError("Download did not produce a video file")

            pipeline_result = _run_pipeline(video_path, work_dir, video_title=video_title)
            do_upload = False
            pr_for_upload = None
            with _jobs_lock:
                if job_id not in _jobs:
                    return
                has_output = any(
                    pipeline_result.get(k) and os.path.isfile(pipeline_result[k])
                    for k in ("transcript_html", "transcript_doc", "explanation_html", "explanation_doc")
                )
                if has_output:
                    _jobs[job_id]["status"] = JOB_STATUS_DONE
                    _jobs[job_id]["pipeline_result"] = pipeline_result
                    _jobs[job_id]["pdf_path"] = pipeline_result.get("transcript_doc") or pipeline_result.get("transcript_html")
                    _jobs[job_id]["pdf_explanation_path"] = pipeline_result.get("explanation_doc") or pipeline_result.get("explanation_html")
                    do_upload = True
                    pr_for_upload = pipeline_result
                else:
                    _jobs[job_id]["status"] = JOB_STATUS_FAILED
                    _jobs[job_id]["error"] = "Pipeline did not produce output files"
            # Upload to SharePoint as soon as the process is finished (outside lock to avoid blocking).
            if do_upload and pr_for_upload:
                try:
                    # Prefer a human-friendly name based on the video title.
                    base_title = (video_title or "lecture_output").strip()
                    base_title = re.sub(r'[<>:"/\\|?*]', "_", base_title)[:200] or "lecture_output"
                    file_name = f"{base_title}.zip"
                    zip_bytes = _make_output_zip(pr_for_upload)
                    sharepoint_url = _upload_zip_to_sharepoint(zip_bytes, file_name)
                    with _jobs_lock:
                        if job_id in _jobs:
                            _jobs[job_id]["sharepoint_url"] = sharepoint_url
                            _jobs[job_id]["video_title"] = video_title
                    logger.info("Job %s: ZIP uploaded to SharePoint (Resources/%s)", job_id, file_name)
                    # Notify the user by email if we have an address.
                    if user_email:
                        subject = f"Your lecture processing is complete: {base_title}"
                        body_html = (
                            f"<p>Dear user,</p>"
                            f"<p>Your lecture processing job has finished successfully.</p>"
                            f"<p><strong>Title:</strong> {base_title}</p>"
                            f"<p>You can download the ZIP from SharePoint here:</p>"
                            f'<p><a href="{sharepoint_url}">{sharepoint_url}</a></p>'
                            f"<p>Best regards,<br/>Taraz Lecture Processing System</p>"
                        )
                        _send_completion_email(user_email, subject, body_html)
                except Exception as e:
                    logger.exception("SharePoint upload after pipeline failed for job %s: %s", job_id, e)
        except Exception as e:
            err_msg = _sanitize_error_message(str(e))
            logger.warning("Job %s failed: %s", job_id, err_msg)
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["status"] = JOB_STATUS_FAILED
                    _jobs[job_id]["error"] = err_msg
    except Exception as e:
        logger.exception("Job worker crashed for %s: %s", job_id, e)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = JOB_STATUS_FAILED
                _jobs[job_id]["error"] = _sanitize_error_message(str(e))
    finally:
        cb_url = None
        with _jobs_lock:
            cb_url = _jobs.get(job_id, {}).get("callback_url")
            job = _jobs.get(job_id) if job_id in _jobs else None
        if cb_url and job:
            status = job.get("status")
            err = job.get("error")
            pt = job.get("pdf_path") if status == JOB_STATUS_DONE else None
            pe = job.get("pdf_explanation_path") if status == JOB_STATUS_DONE else None
            _send_callback(cb_url, job_id, status or JOB_STATUS_FAILED, error=err, pdf_transcript_path=pt, pdf_explanation_path=pe)
        if work_dir and os.path.isdir(work_dir):
            with _jobs_lock:
                j = _jobs.get(job_id)
            if j and j.get("status") == JOB_STATUS_FAILED:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness/readiness."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Async job API (for ngrok / Cloudflare: each request &lt; 100s)
# ---------------------------------------------------------------------------


@app.post("/process-youtube/submit")
async def submit_job(body: SubmitRequest):
    """
    Start a pipeline job and return immediately.

    - Without callback_url: poll GET /job/{job_id} until status is 'done', then GET /job/{job_id}/result for the PDF.
      (Requires n8n execution timeout to be >= pipeline duration, e.g. 1–2 hours.)
    - With callback_url: FastAPI will POST the result (PDF or error) to that URL when done. No polling; no n8n timeout.
      Use an n8n webhook URL as callback_url so the PDF is delivered to a second workflow.
    """
    video_url = (body.video_url or body.youtube_url or body.drive_url or "").strip()
    if not video_url:
        raise HTTPException(
            status_code=400,
            detail="One of video_url, youtube_url, or drive_url is required",
        )
    if not _is_youtube_url(video_url) and not _is_drive_url(video_url) and not _is_onedrive_sharepoint_url(video_url):
        raise HTTPException(
            status_code=400,
            detail="URL must be a YouTube, Google Drive, OneDrive, or SharePoint link",
        )

    callback_url = (body.callback_url or "").strip() or None

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": JOB_STATUS_PENDING,
            "work_dir": None,
            "pipeline_result": None,
            "pdf_path": None,
            "pdf_explanation_path": None,
            "error": None,
            "callback_url": callback_url,
            "user_email": (body.user_email or "").strip() or None,
            "title": (body.title or "").strip() or None,
        }

    t = threading.Thread(target=_job_worker, args=(job_id, video_url), kwargs={"callback_url": callback_url}, daemon=True)
    t.start()

    out = {
        "job_id": job_id,
        "result_path": f"/job/{job_id}/result",
        "message": None,
    }
    if callback_url:
        out["message"] = "Job submitted. Result will be POSTed to callback_url when done (may take up to an hour)."
    else:
        out["message"] = "Job submitted. Poll GET /job/{job_id} until status is 'done', then GET result_path for the ZIP."
    out["download_note"] = "When status is 'done', use result_path (or GET /job/{job_id}/result) to download one ZIP containing 2 HTML, 2 DOC, and enhanced_diagrams folder. Do not use /result/transcript or /result/explanation (deprecated)."
    return out


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    """Return job status: pending, running, done, or failed."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    out = {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
    }
    if job["status"] == JOB_STATUS_DONE:
        out["result_url"] = f"/job/{job_id}/result"
    return out


def _make_output_zip(pipeline_result: dict) -> bytes:
    """Build one ZIP with 2 HTML, 2 DOC, and enhanced_diagrams/ folder. Raises on IO error."""
    import zipfile
    buf = BytesIO()
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key in ("transcript_html", "explanation_html", "transcript_doc", "explanation_doc"):
                path = pipeline_result.get(key)
                if path and os.path.isfile(path):
                    zf.write(path, os.path.basename(path))
            enhanced_dir = pipeline_result.get("enhanced_diagrams_dir")
            if enhanced_dir and os.path.isdir(enhanced_dir):
                for name in os.listdir(enhanced_dir):
                    full = os.path.join(enhanced_dir, name)
                    if os.path.isfile(full):
                        zf.write(full, os.path.join("enhanced_diagrams", name))
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.exception("Failed to build ZIP")
        raise RuntimeError(f"Could not build result archive: {_sanitize_error_message(str(e))}") from e


@app.get("/job/{job_id}/result")
async def job_result(job_id: str):
    """Build the result ZIP and upload it to SharePoint when status is 'done'. Cleans up after upload.

    Response: JSON with a SharePoint download URL instead of raw ZIP bytes.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JOB_STATUS_DONE:
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready (status: {job['status']}). Poll GET /job/{job_id} until status is 'done'.",
        )
    pipeline_result = job.get("pipeline_result") or {}
    has_any = any(
        pipeline_result.get(k) and os.path.isfile(pipeline_result[k])
        for k in ("transcript_html", "transcript_doc", "explanation_html", "explanation_doc")
    )
    if not has_any:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = JOB_STATUS_FAILED
                _jobs[job_id]["error"] = "Output file missing"
        raise HTTPException(status_code=500, detail="Output file missing")

    # Use SharePoint URL from worker upload if present; otherwise upload on demand.
    sharepoint_url: Optional[str] = job.get("sharepoint_url")
    if not sharepoint_url:
        try:
            zip_bytes = _make_output_zip(pipeline_result)
            file_name = f"lecture_output_{job_id}.zip"
            sharepoint_url = _upload_zip_to_sharepoint(zip_bytes, file_name)
        except Exception as e:
            logger.exception("Job %s result build/upload failed", job_id)
            raise HTTPException(status_code=500, detail=_sanitize_error_message(str(e)))

    work_dir = job.get("work_dir")
    try:
        return JSONResponse(
            {
                "job_id": job_id,
                "sharepoint_url": sharepoint_url,
                "message": "Result ZIP uploaded to SharePoint under Resources/.",
            }
        )
    finally:
        with _jobs_lock:
            if job_id in _jobs:
                del _jobs[job_id]
        if work_dir and os.path.isdir(work_dir):
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass


@app.get("/job/{job_id}/result/transcript")
async def job_result_transcript(job_id: str):
    """Deprecated. Use GET /job/{job_id}/result to download the full ZIP (2 HTML, 2 DOC, enhanced_diagrams)."""
    raise HTTPException(
        status_code=410,
        detail="Use GET /job/{job_id}/result to download a single ZIP containing 2 HTML, 2 DOC, and enhanced_diagrams folder.",
    )


@app.get("/job/{job_id}/result/explanation")
async def job_result_explanation(job_id: str):
    """Deprecated. Use GET /job/{job_id}/result to download the full ZIP (2 HTML, 2 DOC, enhanced_diagrams)."""
    raise HTTPException(
        status_code=410,
        detail="Use GET /job/{job_id}/result to download a single ZIP containing 2 HTML, 2 DOC, and enhanced_diagrams folder.",
    )


# ---------------------------------------------------------------------------
# Synchronous API (single long request; may timeout behind Cloudflare/ngrok)
# ---------------------------------------------------------------------------


@app.post(
    "/process-youtube",
    responses={
        200: {"content": {"application/zip": {}}, "description": "ZIP containing lecture_transcript.pdf and lecture_explanation.pdf"},
        400: {"description": "Invalid URL or request"},
        500: {"description": "Pipeline or download error"},
    },
)
async def process_youtube(body: ProcessYouTubeRequest):
    """
    Download video, run pipeline, and return a ZIP of both PDFs (transcript + explanation) in one request.
    Can timeout behind Cloudflare/ngrok; use POST /process-youtube/submit + GET /job/{id}/result for tunnels.
    """
    video_url = (body.video_url or body.youtube_url or body.drive_url or "").strip()
    if not video_url:
        raise HTTPException(
            status_code=400,
            detail="One of video_url, youtube_url, or drive_url is required",
        )

    work_dir = tempfile.mkdtemp(prefix="pipeline_")
    try:
        # Prefer explicit title from the request body when provided.
        video_title: Optional[str] = (body.title or "").strip() or None
        user_email: Optional[str] = (body.user_email or "").strip() or None
        if _is_youtube_url(video_url):
            from youtube_downloader import download_youtube_video

            video_path, _ = download_youtube_video(
                video_url,
                output_dir=work_dir,
                output_filename="video",
                best_quality=True,
            )
        elif _is_drive_url(video_url):
            from drive_downloader import download_drive_video

            video_path, _ = download_drive_video(
                video_url,
                output_dir=work_dir,
                output_filename="video",
            )
        elif _is_onedrive_sharepoint_url(video_url):
            video_path, _ = _download_onedrive_sharepoint_video(
                video_url,
                output_dir=work_dir,
                output_filename="video",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="URL must be a YouTube, Google Drive, OneDrive, or SharePoint link",
            )

        if not video_path or not os.path.isfile(video_path):
            raise HTTPException(
                status_code=500,
                detail="Download did not produce a video file",
            )

        pipeline_result = _run_pipeline(video_path, work_dir, video_title=video_title)
        has_any = any(
            pipeline_result.get(k) and os.path.isfile(pipeline_result.get(k))
            for k in ("transcript_html", "transcript_doc", "explanation_html", "explanation_doc")
        )
        if not has_any:
            raise HTTPException(
                status_code=500,
                detail="Pipeline did not produce output files",
            )

        zip_bytes = _make_output_zip(pipeline_result)
        # Name the ZIP based on the video title when available, otherwise fall back.
        base_title = (video_title or "lecture_output").strip()
        import re as _re_zip
        base_title = _re_zip.sub(r'[<>:"/\\|?*]', "_", base_title).strip().strip(". ") or "lecture_output"
        zip_name = f"{base_title}.zip"

        # In sync mode we can also optionally send a completion email.
        if user_email:
            # For sync, there is no SharePoint link yet; user will download directly.
            subject = f"Your lecture processing is complete: {base_title}"
            body_html = (
                f"<p>Dear user,</p>"
                f"<p>Your lecture processing job has finished successfully.</p>"
                f"<p><strong>Title:</strong> {base_title}</p>"
                f"<p>You downloaded the ZIP directly from the API.</p>"
                f"<p>Best regards,<br/>Taraz Lecture Processing System</p>"
            )
            _send_completion_email(user_email, subject, body_html)

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename=\"{zip_name}\"'},
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=_sanitize_error_message(str(e)))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=_sanitize_error_message(str(e)))
    except Exception as e:
        logger.exception("Sync pipeline failed")
        raise HTTPException(status_code=500, detail=_sanitize_error_message(str(e)))
    finally:
        if work_dir and os.path.isdir(work_dir):
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
