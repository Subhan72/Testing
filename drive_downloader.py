"""
Google Drive video downloader using gdown.
Downloads a video from a Google Drive share link to a local path.
"""
import os
import re
from typing import Optional

# Lazy import to avoid requiring gdown at import time
_gdown = None


def _get_gdown():
    global _gdown
    if _gdown is None:
        try:
            import gdown
            _gdown = gdown
        except ImportError:
            raise RuntimeError(
                "gdown is required for Google Drive downloads. Install with: pip install gdown"
            )
    return _gdown


def _extract_file_id(url_or_id: str) -> Optional[str]:
    """Extract Google Drive file ID from URL or return as-is if already an ID."""
    s = (url_or_id or "").strip()
    if not s:
        return None
    # https://drive.google.com/file/d/FILE_ID/view
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    # https://drive.google.com/open?id=FILE_ID
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    # https://drive.google.com/uc?id=FILE_ID
    m = re.search(r"uc\?id=([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    # Raw ID (alphanumeric, dash, underscore; typically 33 chars for Drive)
    if re.match(r"^[a-zA-Z0-9_-]{20,}$", s):
        return s
    return None


def download_drive_video(
    drive_url: str,
    output_dir: str,
    output_filename: str = "video",
) -> tuple[str, str]:
    """
    Download a video from Google Drive to a local file.

    Args:
        drive_url: Google Drive share link (e.g. https://drive.google.com/file/d/ID/view)
                   or file ID.
        output_dir: Directory to save the video.
        output_filename: Base filename (without extension). Defaults to "video".

    Returns:
        (video_path, video_title) where title is derived from the Drive file name.

    Raises:
        ValueError: If URL/ID is invalid.
        RuntimeError: If download fails.
    """
    if not drive_url or not str(drive_url).strip():
        raise ValueError("drive_url is required")

    raw = str(drive_url).strip()
    file_id = _extract_file_id(raw)
    if not file_id:
        raise ValueError(
            "Invalid Google Drive URL. Use a share link like "
            "https://drive.google.com/file/d/FILE_ID/view or https://drive.google.com/open?id=FILE_ID"
        )

    gdown_module = _get_gdown()

    os.makedirs(output_dir, exist_ok=True)

    # gdown expects url like https://drive.google.com/uc?id=FILE_ID
    url = f"https://drive.google.com/uc?id={file_id}"

    # Download into output_dir so Drive's original filename is preserved.
    # When output is a directory, gdown will place the file inside using the Drive title,
    # which we can then use as our video_title.
    out_path = output_dir

    try:
        path = gdown_module.download(url, output=out_path, quiet=True, fuzzy=True)
        if path is None:
            raise RuntimeError(
                "Google Drive download failed (file not found, not shared, or quota exceeded). "
                "Ensure the link is shared with 'Anyone with the link'."
            )
        path = os.path.abspath(path)
        # gdown may return file path or folder path
        if os.path.isfile(path):
            if os.path.getsize(path) > 0:
                base = os.path.splitext(os.path.basename(path))[0]
                return path, base
            raise RuntimeError("Downloaded file is empty.")
        if os.path.isdir(path):
            for name in os.listdir(path):
                p = os.path.join(path, name)
                if os.path.isfile(p) and os.path.getsize(p) > 0:
                    video_path = os.path.abspath(p)
                    base = os.path.splitext(os.path.basename(video_path))[0]
                    return video_path, base
            raise RuntimeError("Download produced a folder but no file inside.")
        # Fallback: find any new video-like file in output_dir
        video_extensions = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
        for name in os.listdir(output_dir):
            p = os.path.join(output_dir, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0 and name.lower().endswith(video_extensions):
                video_path = os.path.abspath(p)
                base = os.path.splitext(os.path.basename(video_path))[0]
                return video_path, base
        raise RuntimeError("Google Drive download did not produce a video file.")
    except ValueError as e:
        raise
    except RuntimeError as e:
        raise
    except Exception as e:
        raise RuntimeError(f"Google Drive download failed: {e}") from e
