"""
YouTube video downloader using yt-dlp
Downloads YouTube videos in best available quality to a local path.
Converts output to H.264 MP4 when needed so OpenCV can read frames (diagram extraction).
"""
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

# Lazy import to avoid requiring yt-dlp at import time
_ydl = None


def _ensure_opencv_compatible(video_path: str, output_dir: str, output_filename: str) -> str:
    """
    Convert video to H.264 MP4 so OpenCV (cv2.VideoCapture) can read frames.
    YouTube often serves VP9/WebM which some OpenCV builds don't decode; re-encoding to H.264 fixes diagram extraction.
    """
    if not video_path or not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
        return video_path
    ext = os.path.splitext(video_path)[1].lower()
    # If already .mp4 we could still have VP9 inside; convert to H.264 for reliability
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return video_path
    out_path = os.path.join(output_dir, f"{output_filename}_opencv.mp4")
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-movflags", "+faststart",
                "-y",
                out_path,
            ],
            check=True,
            timeout=3600,
            capture_output=True,
        )
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return os.path.abspath(out_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return video_path


def _get_ydl():
    global _ydl
    if _ydl is None:
        try:
            import yt_dlp
            _ydl = yt_dlp
        except ImportError:
            raise RuntimeError(
                "yt-dlp is required for YouTube downloads. Install with: pip install yt-dlp"
            )
    return _ydl


def download_youtube_video(
    youtube_url: str,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    best_quality: bool = True,
) -> tuple[str, str]:
    """
    Download a YouTube video to a local file.

    Args:
        youtube_url: YouTube video URL (e.g. https://www.youtube.com/watch?v=...)
        output_dir: Directory to save the video. Defaults to temp directory.
        output_filename: Base filename (without extension). Defaults to "video".
        best_quality: If True, download best quality. If False, use best up to 1080p.

    Returns:
        (video_path, video_title)

    Raises:
        ValueError: If URL is invalid or not a YouTube URL.
        RuntimeError: If download fails.
    """
    if not youtube_url or not str(youtube_url).strip():
        raise ValueError("youtube_url is required")

    url = str(youtube_url).strip()
    if "youtube.com" not in url and "youtu.be" not in url:
        raise ValueError("Invalid YouTube URL")

    ydl_module = _get_ydl()

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_pipeline_")
    os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        output_filename = "video"

    # Sanitize filename
    output_filename = re.sub(r'[<>:"/\\|?*]', "_", output_filename)[:200]
    outtmpl = os.path.join(output_dir, output_filename + ".%(ext)s")

    # Match working CLI: yt-dlp -f "bv*[height<=720]+ba" --merge-output-format mp4 --no-cache-dir
    format_string = "bv*[height<=720]+ba/bv*+ba/best" if best_quality else "bv*[height<=720]+ba"

    # Browser-like User-Agent to reduce 403 Forbidden from YouTube
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    opts = {
        "outtmpl": outtmpl,
        "format": format_string,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "cachedir": None,  # --no-cache-dir: avoid stale/corrupt cache
        "http_headers": {"User-Agent": user_agent},
    }

    # Cookies: use YT_DLP_COOKIES_PATH if set, else youtube_cookies.txt in project dir or cwd
    cookies_path = os.environ.get("YT_DLP_COOKIES_PATH", "").strip()
    if not cookies_path or not os.path.isfile(cookies_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_cookies = os.path.join(script_dir, "youtube_cookies.txt")
        if os.path.isfile(default_cookies):
            cookies_path = default_cookies
        elif os.path.isfile("youtube_cookies.txt"):
            cookies_path = os.path.abspath("youtube_cookies.txt")
        else:
            cookies_path = None
    if cookies_path:
        opts["cookiefile"] = cookies_path

    # Retry with different player clients; "no longer supported" means YouTube rejected that client
    client_lists = [
        ["mweb", "android", "tv_embedded", "web"] if cookies_path else ["mweb", "android", "tv_embedded"],
        ["android", "mweb"],
        ["ios", "android", "mweb"],
    ]
    last_error = None
    for client_list in client_lists:
        opts["extractor_args"] = {"youtube": {"player_client": client_list}}
        try:
            with ydl_module.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError("Failed to extract video info")
                # Prefer the original YouTube title; fallback to output_filename.
                raw_title = (info.get("title") or output_filename or "video").strip()
                # Simple filename-safe sanitization (same invalid chars as Windows)
                import re as _re  # local import to avoid global dependency
                safe_title = _re.sub(r'[<>:"/\\|?*]', "_", raw_title)[:200] or "video"
                ext = (info.get("ext") or "mp4").split("/")[0]
                result_path = os.path.normpath(os.path.join(output_dir, f"{output_filename}.{ext}"))
                if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                    path = os.path.abspath(result_path)
                    return _ensure_opencv_compatible(path, output_dir, output_filename), safe_title
                for name in os.listdir(output_dir):
                    p = os.path.join(output_dir, name)
                    if os.path.isfile(p) and os.path.getsize(p) > 0:
                        if output_filename in name or name.endswith((".mp4", ".mkv", ".webm")):
                            path = os.path.abspath(p)
                            # Title is already derived; reuse safe_title.
                            return _ensure_opencv_compatible(path, output_dir, output_filename), safe_title
                raise RuntimeError(
                    "Download produced an empty or missing file. "
                    "Install ffmpeg for best quality; the video may also be region/age restricted."
                )
        except RuntimeError:
            raise
        except Exception as e:
            last_error = e
            err_msg = str(e).strip().lower()
            if "no longer supported" in err_msg or "not supported" in err_msg:
                continue
            # Other error: handle and re-raise
            err_msg_full = str(e).strip()
            if "empty" in err_msg_full.lower():
                raise RuntimeError(
                    "YouTube download produced an empty file (merge failed). "
                    "Install ffmpeg and ensure it is on PATH, or the video may be restricted."
                ) from e
            if "403" in err_msg_full or "Forbidden" in err_msg_full:
                raise RuntimeError(
                    f"YouTube download failed: {e}. "
                    "Try: pip install -U yt-dlp. Use youtube_cookies.txt if needed."
                ) from e
            if "yt_dlp" in type(e).__module__ or "youtube_dl" in type(e).__module__:
                raise RuntimeError(f"YouTube download failed: {e}") from e
            raise
    # All client lists failed with "no longer supported"
    raise RuntimeError(
        f"YouTube download failed: {last_error}. "
        "Try: pip install -U yt-dlp (latest version). Ensure youtube_cookies.txt is fresh (re-export from browser)."
    ) from last_error
