"""
!download — save videos, audio and images from X, Facebook, Instagram and YouTube.

    !download <url>                 video (falls back to images for photo posts)
    !download <url> audio           audio only, as mp3          (needs ffmpeg)
    !download <url> images          photos/carousels via gallery-dl
    !download <url> video           video only, no image fallback
    !download <url> q=480           cap the quality at 480p     (or q=<yt-dlp format>)
    !download <url1> <url2> ...     several links at once
    !download batch urls.txt        one link per line (# for comments)
    !download status                progress of your downloads
    !download list                  most recent files
    !download folder | clear

Downloads run in the background, so EMLII stays responsive — check on them with
!download status. Files go to downloads/media/ next to EMLII.

Needs:  pip install yt-dlp     (videos/audio)
        pip install gallery-dl (image posts, optional)
        ffmpeg on PATH         (audio extraction / merging, optional)
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

COMMAND = "download"
HELP = "Download video/audio/images from a link: !download <url> [audio|images]"



class _OutDir:
    """Behaves like a Path but follows Settings → Folders → Downloads (EMLII_DOWNLOADS_DIR) live."""
    def _p(self):
        root = os.environ.get("EMLII_DOWNLOADS_DIR")
        return Path(root) / "media" if root else Path("downloads") / "media"
    def __getattr__(self, name): return getattr(self._p(), name)
    def __truediv__(self, other): return self._p() / other
    def __str__(self): return str(self._p())
    __fspath__ = __str__


OUT_DIR = _OutDir()
MAX_PARALLEL = 2
MAX_JOBS_KEPT = 25
MAX_BATCH = 50

PLATFORM_MAP = {
    "twitter.com": "x", "x.com": "x",
    "facebook.com": "facebook", "fb.watch": "facebook",
    "instagram.com": "instagram",
    "youtube.com": "youtube", "youtu.be": "youtube",
}

_jobs = {}
_lock = threading.Lock()
_next_id = 1
_slots = threading.Semaphore(MAX_PARALLEL)


def detect_platform(url):
    host = urlparse(url).netloc.lower().replace("www.", "")
    for key, name in PLATFORM_MAP.items():
        if key in host:
            return name
    return "web"


# ── workers (ported from media_downloader.py; they update a job dict, not stdout) ──
class _Log:
    """yt-dlp logger that keeps the last error instead of printing it."""
    def __init__(self, job): self.job = job
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): self.job["last_error"] = str(m)


def _ytdlp(job, quality, audio_only):
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed — run: pip install yt-dlp")
    if audio_only and not shutil.which("ffmpeg"):
        raise RuntimeError("audio extraction needs ffmpeg on your PATH")

    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                job["pct"] = d.get("downloaded_bytes", 0) * 100.0 / total
            job["msg"] = (d.get("_speed_str") or "").strip()
        elif d.get("status") == "finished":
            job["pct"], job["msg"] = 100.0, "processing…"

    opts = {
        "outtmpl": str(OUT_DIR / "%(title).120s.%(ext)s"),
        "format": "bestaudio/best" if audio_only else quality,
        "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
        "ignoreerrors": False, "logger": _Log(job), "progress_hooks": [hook],
    }
    if audio_only:
        opts["postprocessors"] = [{"key": "FFmpegExtractAudio",
                                   "preferredcodec": "mp3", "preferredquality": "192"}]
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(job["url"], download=True)
        except yt_dlp.utils.DownloadError as e:
            raise RuntimeError(str(e).replace("ERROR: ", "").strip().splitlines()[-1][:160])
        name = ydl.prepare_filename(info)
        if audio_only:
            name = name.rsplit(".", 1)[0] + ".mp3"
    job["title"] = info.get("title") or job.get("title")
    job["files"].append(name)


def _gallerydl(job):
    exe = shutil.which("gallery-dl")
    cmd = [exe] if exe else None
    if cmd is None and not getattr(sys, "frozen", False):
        try:
            import gallery_dl  # noqa: F401
            cmd = [sys.executable, "-m", "gallery_dl"]
        except ImportError:
            pass
    if cmd is None:
        raise RuntimeError("gallery-dl is not installed — run: pip install gallery-dl")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0     # no console flash in the .exe
    r = subprocess.run(cmd + ["-d", str(OUT_DIR), job["url"]], capture_output=True,
                       text=True, timeout=900, creationflags=flags)
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        raise RuntimeError(err[-1][:160] if err else f"gallery-dl exited with code {r.returncode}")
    job["files"] += [ln.lstrip("# ").strip() for ln in r.stdout.splitlines() if ln.strip()]
    if not job["files"]:
        job["files"].append(str(OUT_DIR))


def _work(job, quality, audio_only, images_only, video_only):
    with _slots:
        job["state"], job["msg"] = "running", ""
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if images_only:
                _gallerydl(job)
            elif video_only or audio_only:
                _ytdlp(job, quality, audio_only)
            else:
                try:
                    _ytdlp(job, quality, False)
                except Exception as first:
                    if job["platform"] == "youtube":
                        raise
                    job["msg"] = "no video — trying images…"
                    try:
                        _gallerydl(job)
                    except Exception as second:
                        raise RuntimeError(f"{first}  |  images: {second}")
            job["state"], job["pct"], job["msg"] = "done", 100.0, ""
        except Exception as e:
            job["state"], job["msg"] = "failed", str(e)[:200]
        job["finished"] = time.time()


def start_job(url, quality="best", audio_only=False, images_only=False, video_only=False):
    global _next_id
    with _lock:
        job = {"id": _next_id, "url": url, "platform": detect_platform(url),
               "mode": "audio" if audio_only else "images" if images_only else "video" if video_only else "auto",
               "state": "queued", "pct": None, "msg": "", "files": [], "title": None,
               "last_error": None, "started": time.time(), "finished": None}
        _jobs[_next_id] = job
        _next_id += 1
        for old in sorted(_jobs)[:-MAX_JOBS_KEPT]:                 # keep the table small
            if _jobs[old]["state"] in ("done", "failed"):
                del _jobs[old]
    threading.Thread(target=_work, args=(job, quality, audio_only, images_only, video_only),
                     daemon=True).start()
    return job


# ── presentation ─────────────────────────────────────────────────────────────
def _short(text, n):
    text = str(text or "")
    return text if len(text) <= n else text[:n - 1] + "…"


def _status():
    with _lock:
        jobs = [dict(j) for j in sorted(_jobs.values(), key=lambda j: j["id"], reverse=True)]
    if not jobs:
        return "No downloads yet. Try:  !download <link>"
    out = [f"Downloads → {OUT_DIR}"]
    for j in jobs[:12]:
        head = f"#{j['id']:<3}{j['platform']:<10}"
        if j["state"] == "done":
            f = j["files"][0] if j["files"] else ""
            out.append(f"{head}✓ {_short(Path(f).name or j['title'], 30)}")
        elif j["state"] == "failed":
            out.append(f"{head}✗ {_short(j['msg'], 40)}")
        elif j["state"] == "running":
            pct = f"{j['pct']:.0f}%" if j["pct"] is not None else "…"
            out.append(f"{head}▶ {pct} {_short(j['msg'] or j['title'] or j['url'], 28)}")
        else:
            out.append(f"{head}· queued")
    return "\n".join(out)


def _recent_files(n=10):
    if not OUT_DIR.exists():
        return "Nothing downloaded yet."
    files = sorted((p for p in OUT_DIR.rglob("*") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    if not files:
        return "Nothing downloaded yet."
    return "Recent files\n" + "\n".join(
        f"  {_short(p.name, 36):<36} {p.stat().st_size / 1_048_576:>7.1f} MB" for p in files)


USAGE = """Download media from X, Facebook, Instagram, YouTube
  !download <url>            video (images for photo posts)
  !download <url> audio      mp3 only
  !download <url> images     photos / carousels
  !download <url> q=480      limit quality
  !download batch <file>     one link per line
  !download status | list | folder | clear"""


def _reply(text):
    return {"type": "reply", "text": text}


def run(arg, ctx):
    tokens = (arg or "").split()
    if not tokens:
        return _reply(USAGE)
    sub = tokens[0].lower()

    if sub in ("status", "jobs", "s"):
        return _reply(_status())
    if sub in ("list", "files", "ls"):
        return _reply(_recent_files())
    if sub in ("folder", "dir", "path"):
        return _reply(f"Downloads are saved to:\n{OUT_DIR.resolve()}")
    if sub == "clear":
        with _lock:
            for k in [k for k, j in _jobs.items() if j["state"] in ("done", "failed")]:
                del _jobs[k]
        return _reply("Cleared finished downloads from the list (files are untouched).")
    if sub in ("help", "?"):
        return _reply(USAGE)

    urls, quality = [], "best"
    audio = images = video = False
    if sub == "batch":
        path = arg.split(None, 1)[1].strip().strip('"') if len(tokens) > 1 else ""
        if not path:
            return _reply("Usage: !download batch <file with one link per line>")
        try:
            with open(path, "r", encoding="utf-8") as f:
                urls = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        except OSError as e:
            return _reply(f"Couldn't read '{path}': {e}")
        if not urls:
            return _reply("No links found in that file.")
        urls = urls[:MAX_BATCH]
    else:
        for t in tokens:
            low = t.lower()
            if low.startswith(("http://", "https://")):
                urls.append(t)
            elif low in ("audio", "mp3", "-a", "--audio-only"):
                audio = True
            elif low in ("images", "image", "photos", "-i", "--images-only"):
                images = True
            elif low in ("video", "-v", "--video-only"):
                video = True
            elif low.startswith(("q=", "quality=")):
                val = t.split("=", 1)[1]
                quality = f"best[height<={val}]" if val.isdigit() else (val or "best")
            else:
                return _reply(f"Don't know '{t}'.\n\n{USAGE}")
        if not urls:
            return _reply("Give me a link starting with http:// or https://\n\n" + USAGE)
    if images and video:
        return _reply("Pick one of: images / video.")
    if audio and images:
        return _reply("Audio only works for video links — pick audio or images.")

    jobs = [start_job(u, quality, audio, images, video) for u in urls]
    if len(jobs) == 1:
        j = jobs[0]
        return _reply(f"Started #{j['id']} · {j['platform']} · {j['mode']}\n"
                      f"Saving to {OUT_DIR}\nCheck progress:  !download status")
    return _reply(f"Started {len(jobs)} downloads (#{jobs[0]['id']}–#{jobs[-1]['id']}), "
                  f"{MAX_PARALLEL} at a time.\nCheck progress:  !download status")
