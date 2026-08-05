"""Download library + CLI.

Handles the source types intake feeds it:
  - Google Drive folders  -> gdown.download_folder
  - Google Drive files    -> yt-dlp transcoded preview stream (<= max_source_height),
                             or the raw file via gdown when --original is set
  - VODs (Kick/YouTube/Twitch) -> yt-dlp (optionally with browser cookies)
  - direct http(s) files  -> yt-dlp (generic)
  - local paths           -> copied in

Big Drive VODs (tens of GB) are not downloaded raw by default: Drive serves
transcoded 360p/720p preview streams that yt-dlp can list and fetch. We pick the best
stream at or below `max_source_height` (default 720). If no transcoded stream exists we
FAIL LOUD with the original file size rather than silently pulling a 33GB file; pass
--original (or original=True) to force the raw download.

Routing is BY FILE TYPE, decided per downloaded file (not per source): video
extensions -> campaign/footage/, image extensions -> campaign/assets/. A Drive folder
that mixes footage and watermark PNGs is split correctly, so nothing non-video ever
counts as footage or reaches index.py.

Per-source download errors raise DownloadError (catchable) so a failed OPTIONAL source
(e.g. a Kick VOD behind a 403) can be logged and skipped without aborting intake.
Environment errors (missing yt-dlp/gdown) still fail loud.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".flv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DOC_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".rtf", ".csv",
            ".tsv", ".xlsx", ".xls", ".gdoc", ".gsheet", ".gslides", ".json"}


class DownloadError(Exception):
    """A single source failed to download (recoverable — skip it, keep going)."""


def classify_file(path):
    """Route by extension: 'footage' (video) / 'asset' (image) / 'doc' (readable
    document) / 'other' (kept but unhandled, so coverage can flag it). Never None —
    nothing is silently dropped."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in VIDEO_EXTS:
        return "footage"
    if ext in IMAGE_EXTS:
        return "asset"
    if ext in DOC_EXTS:
        return "doc"
    return "other"


def is_url(s):
    return urlparse(s).scheme in ("http", "https")


def is_drive_folder(url):
    low = url.lower()
    return "drive.google.com" in low and ("/folders/" in low or "folder" in low)


def is_drive_file(url):
    """A Google Drive *file* link (not a folder): /file/d/<id>/… or ?id=<id>."""
    low = url.lower()
    if "drive.google.com" not in low or is_drive_folder(url):
        return False
    return "/file/d/" in low or "id=" in low


def _human_size(n):
    if not n:
        return "unknown size"
    f, i, units = float(n), 0, ["B", "KB", "MB", "GB", "TB"]
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}"


def _dest_dir(kind):
    return {"asset": C.ASSETS, "doc": C.DOCS, "other": C.OTHER}.get(kind, C.FOOTAGE)


def _unique_dest(path):
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


# --- fetch into a staging dir (returns list of Paths) --------------------------
def _fetch_local(url, staging):
    src = Path(url).expanduser()
    if not src.exists():
        raise DownloadError(f"local source does not exist: {url}")
    out = staging / src.name
    shutil.copy2(src, out)
    return [out]


def _fetch_drive(url, staging):
    try:
        import gdown
    except ImportError:
        raise DownloadError("gdown not installed (pip install -r requirements.txt)")
    C.log(f"gdown folder: {url}")
    try:
        gdown.download_folder(url=url, output=str(staging), quiet=False, use_cookies=False)
    except Exception as e:
        raise DownloadError(f"gdown failed: {e}")
    files = [p for p in staging.rglob("*") if p.is_file()]
    if not files:
        raise DownloadError("Drive folder produced no files")
    return files


class _CollectingLogger:
    """yt-dlp logger that keeps error/warning lines so a swallowed ffmpeg
    'Postprocessing: Conversion failed!' can be surfaced with the real ffmpeg cause.

    yt-dlp routes the failing ffmpeg output line through logger.error, but only prints
    the generic 'Conversion failed!' summary. We retain everything and attach it to the
    DownloadError so mux failures are actually diagnosable."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def debug(self, msg):    # yt-dlp sends screen/info messages here too; ignore.
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        self.warnings.append(str(msg))

    def error(self, msg):
        self.errors.append(str(msg))

    def detail(self):
        lines = self.errors + self.warnings
        return "\n".join(f"  {ln}" for ln in lines if ln.strip())


def _ytdlp_workdir():
    """yt-dlp temp + cache dir INSIDE the project, so the mux step doesn't land on a
    full system-temp volume and its scratch/cache stays local and inspectable."""
    d = C.ROOT / ".ytdlp"
    (d / "temp").mkdir(parents=True, exist_ok=True)
    (d / "cache").mkdir(parents=True, exist_ok=True)
    return d


def _check_free_space(path, needed_bytes, what):
    """Fail loud (recoverable) if the volume holding `path` can't fit `needed_bytes`,
    naming the space actually needed — rather than letting ffmpeg die mid-mux with a
    cryptic 'Conversion failed'."""
    if not needed_bytes:
        return
    try:
        free = shutil.disk_usage(str(path)).free
    except OSError:
        return  # can't determine free space; don't block the download
    if free < needed_bytes:
        raise DownloadError(
            f"not enough free disk to {what}: need ~{_human_size(needed_bytes)} "
            f"(with mux headroom), only {_human_size(free)} free on the volume at "
            f"{path}. Free up space or lower --max-source-height.")


def _fetch_ytdlp(url, staging, cookies_from_browser=None, format_id=None,
                 merge_output_format="mp4"):
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise DownloadError("yt-dlp not installed (pip install -r requirements.txt)")
    workdir = _ytdlp_workdir()
    logger = _CollectingLogger()
    opts = {"outtmpl": str(staging / "%(title).80s-%(id)s.%(ext)s"),
            "no_warnings": True, "noprogress": True,
            "logger": logger,
            # Keep yt-dlp scratch + cache inside the project (see _ytdlp_workdir).
            "paths": {"temp": str(workdir / "temp")},
            "cachedir": str(workdir / "cache")}
    if merge_output_format:
        # Explicit container + force ffmpeg as the merger so DASH video+audio muxes
        # deterministically instead of guessing an extension.
        opts["merge_output_format"] = merge_output_format
    if format_id:
        opts["format"] = format_id
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
        C.log(f"yt-dlp using cookies from browser: {cookies_from_browser}")
    C.log(f"yt-dlp: {url}")
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        detail = logger.detail()
        msg = f"yt-dlp failed: {e}"
        if detail:
            msg += f"\nyt-dlp/ffmpeg output:\n{detail}"
        raise DownloadError(msg)
    files = [p for p in staging.rglob("*") if p.is_file()]
    if not files:
        raise DownloadError("yt-dlp produced no files")
    return files


def _drive_file_info(url, cookies_from_browser=None):
    """Probe a Drive file with yt-dlp (the -F equivalent) and return its info dict."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise DownloadError("yt-dlp not installed (pip install -r requirements.txt)")
    opts = {"quiet": True, "no_warnings": True}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as e:
        raise DownloadError(f"could not list Drive formats via yt-dlp: {e}")


def _original_size(info, formats):
    """Best estimate of the raw file's size, in bytes (or None)."""
    src = next((f for f in formats if f.get("format_id") == "source"), None)
    for cand in (src, info):
        if cand:
            s = cand.get("filesize") or cand.get("filesize_approx")
            if s:
                return s
    sizes = [f.get("filesize") or f.get("filesize_approx") or 0 for f in formats]
    return max(sizes) if sizes else None


def _size_of(f):
    return f.get("filesize") or f.get("filesize_approx") or 0


def _has_video(f):
    return bool(f.get("height")) and f.get("vcodec") not in (None, "none")


def _has_acodec(f):
    return f.get("acodec") not in (None, "none")


def _has_audio_stream(path):
    """True if the media file contains at least one audio stream (via ffprobe)."""
    C.require_exe("ffprobe")
    proc = C.run_cmd(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture=True,
    )
    return bool((proc.stdout or "").strip())


def _require_audio(files):
    """Fail loud if any downloaded video file has no audio track — a video-only
    transcode that slipped through would break audio extraction downstream."""
    for p in files:
        if classify_file(p) == "footage" and not _has_audio_stream(p):
            raise DownloadError(
                f"downloaded Drive stream has no audio track: {Path(p).name}. "
                f"The transcode was video-only and audio muxing failed. "
                f"Re-run with --original to fetch the full file with audio.")


def _fetch_drive_file(url, staging, max_source_height=720, original=False,
                      cookies_from_browser=None):
    """Fetch a single Drive file. By default download the best transcoded preview
    stream at or below max_source_height; --original forces the raw file via gdown."""
    if original:
        try:
            import gdown
        except ImportError:
            raise DownloadError("gdown not installed (pip install -r requirements.txt)")
        C.log(f"Drive file (--original): downloading raw file via gdown: {url}")
        try:
            gdown.download(url=url, output=str(staging) + os.sep, quiet=False, fuzzy=True)
        except Exception as e:
            raise DownloadError(f"gdown failed: {e}")
        files = [p for p in staging.rglob("*") if p.is_file()]
        if not files:
            raise DownloadError("gdown produced no files")
        return files

    info = _drive_file_info(url, cookies_from_browser=cookies_from_browser)
    formats = info.get("formats") or []
    orig = _human_size(_original_size(info, formats))
    # Transcoded preview streams are the ones with a real (video) height.
    transcoded = [f for f in formats if _has_video(f)]
    if not transcoded:
        raise DownloadError(
            f"no transcoded preview stream available for this Drive file "
            f"(original is {orig}). Refusing to download the raw file. "
            f"Re-run with --original to force the full download.")

    eligible = [f for f in transcoded if f["height"] <= max_source_height]
    if not eligible:
        heights = ", ".join(f"{h}p" for h in sorted({f['height'] for f in transcoded}))
        raise DownloadError(
            f"no transcoded stream at or below {max_source_height}p "
            f"(available: {heights}; original is {orig}). "
            f"Raise config max_source_height or re-run with --original.")

    # DASH-style Drive transcodes split video and audio (e.g. format 136 is
    # video-only). We MUST end up with audio: either a combined stream (vcodec AND
    # acodec, like format 22) or a video-only stream muxed with a separate audio
    # track. Fail loud if neither is possible.
    combined = [f for f in eligible if _has_acodec(f)]
    video_only = [f for f in eligible if not _has_acodec(f)]
    audio_only = [f for f in formats if _has_acodec(f) and not f.get("height")]
    if not combined and not audio_only:
        heights = ", ".join(f"{h}p" for h in sorted({f['height'] for f in eligible}))
        raise DownloadError(
            f"transcoded video exists ({heights}) but no audio track is available to "
            f"mux (original is {orig}). Re-run with --original for the full file.")

    # A pre-muxed progressive stream (format 18 = 360p mp4, or any combined stream ≤
    # cap) needs no post-processing — it's the safe fallback when the ffmpeg mux breaks.
    progressive_fmt = (f"18/b[height<={max_source_height}][vcodec!=none][acodec!=none]"
                       f"/b[height<={max_source_height}]")

    # yt-dlp resolves + muxes natively; ffmpeg does the merge. Prefer best video-only
    # ≤ cap + best audio, falling back to a combined stream ≤ cap.
    fmt = f"bv[height<={max_source_height}]+ba/b[height<={max_source_height}]"
    if video_only and audio_only:
        # Muxing DASH video+audio REQUIRES ffmpeg as the merger — fail loud early if
        # it's missing rather than letting yt-dlp die with 'Conversion failed'.
        C.require_exe("ffmpeg")
        best_v = max(video_only, key=lambda f: (f["height"], f.get("tbr") or 0, _size_of(f)))
        best_a = max(audio_only, key=lambda f: (f.get("abr") or f.get("tbr") or 0, _size_of(f)))
        raw_bytes = _size_of(best_v) + _size_of(best_a)
        est = _human_size(raw_bytes)
        C.log(f"Drive file: chose format {best_v['format_id']}+{best_a['format_id']} "
              f"({best_v['height']}p video + audio, muxed, est. {est}) "
              f"instead of the {orig} original — downloading.")
        # Peak disk during mux ≈ the two source streams + the muxed output (~2×), plus
        # headroom. Check BEFORE muxing so we fail with the real number needed.
        _check_free_space(staging, int(raw_bytes * 2.2) if raw_bytes else 0,
                          "download and mux video+audio")
        try:
            files = _fetch_ytdlp(url, staging, cookies_from_browser=cookies_from_browser,
                                 format_id=fmt)
        except DownloadError as e:
            # Mux failed (e.g. ffmpeg 'Conversion failed'). Fall back to a pre-muxed
            # progressive stream that needs no post-processing.
            C.warn(f"video+audio mux failed, falling back to a pre-muxed progressive "
                   f"stream:\n{e}")
            files = _fetch_ytdlp(url, staging, cookies_from_browser=cookies_from_browser,
                                 format_id=progressive_fmt, merge_output_format=None)
    else:
        best_c = max(combined, key=lambda f: (f["height"], f.get("tbr") or 0, _size_of(f)))
        est = _human_size(_size_of(best_c))
        C.log(f"Drive file: chose format {best_c['format_id']} "
              f"({best_c['height']}p {best_c.get('ext', '?')}, combined A/V, est. {est}) "
              f"instead of the {orig} original — downloading.")
        _check_free_space(staging, int(_size_of(best_c) * 1.5) if _size_of(best_c) else 0,
                          "download the combined stream")
        files = _fetch_ytdlp(url, staging, cookies_from_browser=cookies_from_browser,
                             format_id=fmt, merge_output_format=None)

    _require_audio(files)   # verify audio landed before marking the download complete
    return files


def download_source(url, cookies_from_browser=None, max_source_height=720, original=False):
    """Download one source and route each resulting file by type.

    Returns a list of {"path": <rel-to-ROOT>, "kind": "footage"|"asset"|"doc"|"other"}.
    Raises DownloadError on a recoverable per-source failure.

    For Drive *file* links, prefer a transcoded stream <= max_source_height; original=True
    forces the raw file.
    """
    staging = Path(tempfile.mkdtemp(prefix="clipper_dl_"))
    try:
        if not is_url(url):
            raw = _fetch_local(url, staging)
        elif is_drive_folder(url):
            raw = _fetch_drive(url, staging)
        elif is_drive_file(url):
            raw = _fetch_drive_file(url, staging, max_source_height=max_source_height,
                                    original=original,
                                    cookies_from_browser=cookies_from_browser)
        else:
            raw = _fetch_ytdlp(url, staging, cookies_from_browser=cookies_from_browser)

        entries = []
        for p in raw:
            kind = classify_file(p)
            target = _dest_dir(kind) / p.name
            # If an identically-named file with the same size is already present, the
            # source is unchanged — reuse it instead of writing a "_1" duplicate.
            if target.exists() and target.stat().st_size == p.stat().st_size:
                rel = os.path.relpath(target, C.ROOT)
                entries.append({"path": rel, "kind": kind})
                C.log(f"  = {kind} (already present, unchanged): {rel}")
                continue
            dest = _unique_dest(target)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            rel = os.path.relpath(dest, C.ROOT)
            entries.append({"path": rel, "kind": kind})
            C.log(f"  -> {kind}: {rel}")
        if not entries:
            raise DownloadError("source produced no files")
        return entries
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ensure_downloaded(manifest, cookies_from_browser=None, max_source_height=720,
                      original=False):
    """Pipeline 'download' stage: re-fetch any manifest file that's gone missing.
    Non-fatal per source. Returns the number of sources re-fetched."""
    n = 0
    missing_sources = {}
    for item in manifest.get("downloads", []):
        p = C.ROOT / item["path"]
        if not p.exists():
            missing_sources.setdefault(item.get("source"), True)
    for src in missing_sources:
        if not src:
            continue
        C.warn(f"re-downloading missing source: {src}")
        try:
            download_source(src, cookies_from_browser=cookies_from_browser,
                            max_source_height=max_source_height, original=original)
            n += 1
        except DownloadError as e:
            C.warn(f"could not re-download {src}: {e}")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download one source into the campaign.")
    ap.add_argument("url", help="Drive folder / Drive file / VOD URL / direct file / local path")
    ap.add_argument("--cookies-from-browser", help="e.g. chrome, edge, firefox (for gated VODs)")
    ap.add_argument("--max-source-height", type=int, default=720,
                    help="max height for Drive transcoded preview streams (default 720)")
    ap.add_argument("--original", action="store_true",
                    help="for Drive files, force the raw original instead of a preview stream")
    args = ap.parse_args()
    C.ensure_dirs()
    try:
        for e in download_source(args.url, cookies_from_browser=args.cookies_from_browser,
                                 max_source_height=args.max_source_height,
                                 original=args.original):
            print(f"{e['kind']}: {e['path']}")
    except DownloadError as e:
        C.fail(str(e))
