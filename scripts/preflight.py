"""PREFLIGHT — read-only overnight-run readiness check.

Runs a battery of NON-DESTRUCTIVE checks across the whole pipeline and prints a clear
PASS / WARN / FAIL line for each, so you can eyeball the system before kicking off an
overnight `run.py --auto-advance`. It changes NOTHING (no downloads, no Groq calls, no
state writes) — it only reads config, files, env, and pings the local token server.

    python scripts/preflight.py

Exit code is NONZERO if any HARD failure is found (a thing that will actually break the
overnight run: missing ffmpeg/yt-dlp, no LLM key, wrong source-height config, unreadable
scout board, bgutil providers not registered). Warnings never fail the exit — they're
things worth a glance but not blockers.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# run.py holds DEFAULT_CONFIG (the source of truth for effective config) and the pickcampaign
# helpers mirror scout's ranking. Import them read-only; degrade loudly if they can't load.
try:
    import run as RUN
    _DEFAULT_CONFIG = RUN.DEFAULT_CONFIG
except Exception as e:                       # pragma: no cover - import guard
    _DEFAULT_CONFIG = None
    _RUN_IMPORT_ERR = e
try:
    import pickcampaign as PICK
except Exception as e:                        # pragma: no cover - import guard
    PICK = None
    _PICK_IMPORT_ERR = e

SCOUT_DIR = r"C:\whop\scout"
SCOUT_JSON = os.path.join(SCOUT_DIR, "campaigns.json")
DISK_WARN_GB = 15.0        # warn under this much free space on C:
DISK_FAIL_GB = 5.0         # hard-fail under this (a single 1080p VOD is easily 1-2 GB)
STALE_BOARD_HOURS = 20.0   # matches scout's once-daily guard / pickcampaign default


# --- report plumbing -----------------------------------------------------------
class Report:
    """Collects PASS/WARN/FAIL lines grouped by section and tallies the totals."""
    def __init__(self):
        self.passed = 0
        self.warnings = 0
        self.failures = 0

    def section(self, title):
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    def _emit(self, tag, msg, detail):
        line = f"  [{tag:^6}] {msg}"
        print(line)
        if detail:
            for d in (detail if isinstance(detail, (list, tuple)) else [detail]):
                print(f"           {d}")

    def ok(self, msg, detail=None):
        self.passed += 1
        self._emit("PASS", msg, detail)

    def warn(self, msg, detail=None):
        self.warnings += 1
        self._emit("WARN", msg, detail)

    def fail(self, msg, detail=None):
        self.failures += 1
        self._emit("FAIL", msg, detail)

    def info(self, msg, detail=None):
        # neutral line — not counted
        self._emit("INFO", msg, detail)


R = Report()


# --- helpers -------------------------------------------------------------------
def _effective_config():
    """DEFAULT_CONFIG overlaid with any state.json config — exactly what run.py would use."""
    base = dict(_DEFAULT_CONFIG or {})
    state = C.load_json(C.STATE_PATH) if C.STATE_PATH.exists() else None
    if state and isinstance(state.get("config"), dict):
        base.update(state["config"])
    return base, state


def _rel(p):
    try:
        return str(Path(p).relative_to(C.ROOT))
    except Exception:
        return str(p)


def _count_files(folder):
    if not Path(folder).exists():
        return 0
    return sum(1 for p in Path(folder).iterdir() if p.is_file())


# --- CONFIG --------------------------------------------------------------------
def check_config(cfg):
    R.section("CONFIG")
    if _DEFAULT_CONFIG is None:
        R.fail("could not import run.py DEFAULT_CONFIG",
               f"{type(_RUN_IMPORT_ERR).__name__}: {_RUN_IMPORT_ERR}")
        return

    msh = cfg.get("max_source_height")
    yfmt = str(cfg.get("youtube_format") or "")
    # youtube_format is a yt-dlp format string; the resolution contract is the height<=N clamp.
    yfmt_1080 = "height<=1080" in yfmt.replace(" ", "")
    yfmt_720 = "height<=720" in yfmt.replace(" ", "")
    if msh == 1080 and yfmt_1080 and not yfmt_720:
        R.ok("source resolution locked to 1080p (max_source_height=1080, youtube_format height<=1080)")
    else:
        R.fail("source-resolution config mismatch — expected 1080p on both",
               [f"max_source_height = {msh!r} (want 1080)",
                f"youtube_format    = {yfmt!r}",
                f"  height<=1080 present: {yfmt_1080}; height<=720 present: {yfmt_720}"])

    # Export quality knobs — reported for eyeball; sanity-warn on obviously-wrong values.
    crf = cfg.get("output_crf")
    preset = cfg.get("output_preset")
    abr = cfg.get("output_audio_bitrate")
    detail = [f"output_crf           = {crf!r}",
              f"output_preset        = {preset!r}",
              f"output_audio_bitrate = {abr!r}",
              f"content_crf (TRACK)  = {cfg.get('content_crf')!r}"]
    try:
        crf_bad = not (0 <= int(crf) <= 28)
    except (TypeError, ValueError):
        crf_bad = True
    if crf_bad:
        R.warn("output_crf outside the sane 0-28 range", detail)
    else:
        R.ok("export quality config", detail)

    # Full key-config dump for eyeball.
    key_keys = [
        "layout", "track_subject_scale", "track_max_upscale",
        "clip_min_seconds", "clip_max_seconds", "min_separation_seconds",
        "merge_gap_seconds", "merge_max_span_seconds",
        "select_min_quality", "select_hard_cap", "select_per_campaign_cap",
        "select_dead_floor", "select_max_candidates",
        "footage_cap_hours", "channel_max_videos",
        "youtube_daily_cap", "download_rate_limit",
        "walk_spacing_seconds", "walk_throttle_backoff_seconds", "target_batch_min",
        "gate_language", "gate_language_min_prob", "llm_providers",
        "subtitles_enabled", "subtitle_ass_fontsize", "hook_style",
        "output_fps", "ffmpeg_threads",
        "token_server_url", "token_server_dir",
    ]
    R.info("key config values (effective = DEFAULT_CONFIG + state.json):",
           [f"{k:28} = {cfg.get(k)!r}" for k in key_keys])


# --- STATE / CLEANLINESS -------------------------------------------------------
def check_state(state):
    R.section("STATE / CLEANLINESS")

    if state is None:
        R.ok("no state.json — fresh start (nothing to resume onto)")
    else:
        done = [n for n, m in (state.get("stages") or {}).items() if m.get("done")]
        camp = state.get("campaign")
        if done:
            R.warn("state.json present with completed stages — an overnight run will RESUME onto it",
                   [f"active campaign: {camp!r}",
                    f"stages done: {', '.join(done)}",
                    "use `run.py --force` (or delete state.json) for a truly fresh run"])
        else:
            R.ok("state.json present but no stages marked done", f"active campaign: {camp!r}")

    # footage
    fcount = _count_files(C.FOOTAGE)
    if fcount == 0:
        R.ok("campaign/footage/ empty (clean)")
    else:
        names = [p.name for p in C.FOOTAGE.iterdir() if p.is_file()][:8]
        R.warn(f"campaign/footage/ holds {fcount} file(s) from a prior campaign",
               names + (["…"] if fcount > len(names) else []))

    # per-stage json outputs
    for label, path in (("moments.json", C.MOMENTS_JSON),
                        ("selected.json", C.SELECTED_JSON),
                        ("captions.json", C.CAPTIONS_JSON)):
        if path.exists():
            R.warn(f"campaign/{label} present (stale from a prior campaign — will be overwritten)",
                   _rel(path))
        else:
            R.ok(f"campaign/{label} clean (absent)")

    # drafts staging
    for label, folder in (("drafts/", C.DRAFTS), ("drafts_batch/", C.DRAFTS_BATCH)):
        n = _count_files(folder)
        if n == 0 and (not folder.exists() or not any(folder.iterdir())):
            R.ok(f"{label} empty")
        else:
            total = sum(1 for _ in folder.iterdir()) if folder.exists() else 0
            R.warn(f"{label} not empty ({total} item(s)) — prior output present", _rel(folder))


# --- ENVIRONMENT ---------------------------------------------------------------
def _ping(url, timeout=1.5):
    p = urlparse(url)
    host, port = (p.hostname or "127.0.0.1"), (p.port or 4416)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _tool_version(exe, args):
    path = shutil.which(exe)
    if not path:
        return None, None
    try:
        out = subprocess.run([exe] + args, capture_output=True, text=True, timeout=15)
        line = (out.stdout or out.stderr or "").strip().splitlines()
        return path, (line[0] if line else "")
    except Exception as e:
        return path, f"<version check failed: {e}>"


def check_environment(cfg):
    R.section("ENVIRONMENT")

    # PO-token server
    url = cfg.get("token_server_url") or "http://127.0.0.1:4416"
    if _ping(url):
        R.ok(f"PO-token server reachable at {url}")
    else:
        R.warn(f"PO-token server DOWN at {url} — YouTube may 403 until it starts",
               [f"autostart is {'ON' if cfg.get('token_server_autostart', True) else 'OFF'} "
                f"(run.py/intake.py will try `node build/main.js`)",
                f"dir: {cfg.get('token_server_dir')}"])

    # ffmpeg / ffprobe
    for exe in ("ffmpeg", "ffprobe"):
        path, ver = _tool_version(exe, ["-version"])
        if path:
            R.ok(f"{exe} available", ver)
        else:
            R.fail(f"{exe} NOT found on PATH — the cut/index stages cannot run")

    # yt-dlp (import, since the pipeline uses it as a library)
    ypath = shutil.which("yt-dlp")
    try:
        import yt_dlp
        try:
            from yt_dlp.version import __version__ as ytver
        except Exception:
            ytver = getattr(yt_dlp, "__version__", "?")
        R.ok("yt-dlp importable", f"version {ytver}"
             + (f"  (CLI also on PATH: {ypath})" if ypath else "  (no CLI on PATH — library use only)"))
    except Exception as e:
        R.fail("yt-dlp NOT importable — downloads cannot run", f"{type(e).__name__}: {e}")

    # cookies.txt
    cpath = os.environ.get("COOKIES_FILE") or str(C.ROOT / "cookies.txt")
    if os.path.exists(cpath) and os.path.getsize(cpath) > 0:
        R.ok("cookies.txt present and non-empty", f"{_rel(cpath)} ({os.path.getsize(cpath)} bytes)")
    elif os.path.exists(cpath):
        R.warn("cookies.txt is EMPTY — gated YouTube/Kick VODs will 403", _rel(cpath))
    else:
        R.warn("cookies.txt MISSING — gated YouTube/Kick VODs may 403",
               f"expected at {_rel(cpath)} (or set COOKIES_FILE)")

    # LLM keys (count only — NEVER print the values)
    def _count(base):
        return sorted(k for k in os.environ if k == base or k.startswith(base + "_")
                      and os.environ.get(k, "").strip())
    groq = [k for k in ("GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4")
            if os.environ.get(k, "").strip()]
    groq_bare = bool(os.environ.get("GROQ_API_KEY", "").strip())
    gemini = bool([k for k in os.environ if k.startswith("GEMINI_API_KEY")
                   and os.environ.get(k, "").strip()])
    cerebras = bool([k for k in os.environ if k.startswith("CEREBRAS_API_KEY")
                     and os.environ.get(k, "").strip()])
    groq_total = len(groq) + (1 if groq_bare and not groq else 0)
    detail = [f"GROQ_API_KEY_1..4 set: {len(groq)} ({', '.join(groq) or 'none'})"
              + (f" + bare GROQ_API_KEY" if groq_bare else ""),
              f"GEMINI fallback: {'yes' if gemini else 'no'}",
              f"CEREBRAS fallback: {'yes' if cerebras else 'no'}"]
    if groq_total >= 1:
        R.ok(f"Groq keys available: {groq_total} (rotates before failing over)", detail)
    elif gemini or cerebras:
        R.warn("NO Groq key — chain will run on the fallback provider(s) only", detail)
    else:
        R.fail("NO LLM key at all (Groq/Gemini/Cerebras) — select/captions cannot run", detail)

    # disk free on C:
    try:
        free_gb = shutil.disk_usage("C:/").free / 1e9
        d = f"{free_gb:.1f} GB free on C:"
        if free_gb < DISK_FAIL_GB:
            R.fail(f"critically low disk space — {d}")
        elif free_gb < DISK_WARN_GB:
            R.warn(f"low disk space (< {DISK_WARN_GB:.0f} GB) — {d}")
        else:
            R.ok(f"disk space OK — {d}")
    except Exception as e:
        R.warn("could not read disk usage on C:", str(e))


# --- SCOUT HANDOFF -------------------------------------------------------------
def _git_status(repo_dir):
    """(clean, code_changes, other_changes) for a repo. CODE = *.py/*.sh/*.bat/*.toml/*.ini
    or anything under scripts/; everything else (campaign/, memory/, drafts, state.json,
    campaigns.json, *.md docs, caches, logs) is runtime/data and ignored for the code warning."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir,
                             capture_output=True, text=True, timeout=20)
    except Exception as e:
        return None, [f"<git failed: {e}>"], []
    if out.returncode != 0:
        return None, [f"<git error: {(out.stderr or '').strip()[:120]}>"], []
    code, other = [], []
    for ln in (out.stdout or "").splitlines():
        path = ln[3:].strip().strip('"')
        # handle rename "old -> new"
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        low = path.lower()
        is_code = (low.endswith((".py", ".sh", ".bat", ".toml", ".ini", ".cfg"))
                   or low.startswith("scripts/"))
        (code if is_code else other).append(path)
    return (not code and not other), code, other


def check_scout():
    R.section("SCOUT HANDOFF")

    if not os.path.exists(SCOUT_JSON):
        R.fail("scout campaigns.json NOT found — nothing to pick from", SCOUT_JSON)
    else:
        try:
            data = json.loads(open(SCOUT_JSON, encoding="utf-8").read())
        except Exception as e:
            data = None
            R.fail("scout campaigns.json is unreadable/corrupt", f"{type(e).__name__}: {e}")
        if data is not None:
            campaigns = data.get("campaigns") if isinstance(data, dict) else data
            campaigns = campaigns or []
            R.ok(f"scout campaigns.json present ({len(campaigns)} campaigns)", SCOUT_JSON)

            # age / staleness
            age = PICK.board_age_hours(SCOUT_JSON) if PICK else None
            if age is None:
                R.warn("scout board age UNKNOWN (no generated_at) — can't confirm freshness")
            elif age >= STALE_BOARD_HOURS:
                R.warn(f"STALE scout board — generated {age:.1f}h ago (>= {STALE_BOARD_HOURS:.0f}h)",
                       "scout likely skipped its daily scrape; run scout --force for fresh data")
            else:
                R.ok(f"scout board fresh — generated {age:.1f}h ago (< {STALE_BOARD_HOURS:.0f}h)")

            # category + clippable counts (pure, no network)
            if PICK:
                try:
                    podcast = sum(1 for c in campaigns if "podcast_talking" in PICK._cats(c))
                    ranked = PICK.rank_campaigns(campaigns)
                    done_ids = PICK._load_done_ids(SCOUT_DIR)
                    clip = sum(1 for c in ranked if PICK.clippable(c, done_ids)[0])
                    detail = [f"total campaigns : {len(campaigns)}",
                              f"podcast_talking : {podcast}",
                              f"rankable        : {len(ranked)}",
                              f"clippable now   : {clip}  (rankable + rules-readable + has footage link)"]
                    if clip == 0:
                        R.fail("ZERO clippable campaigns on the board — the walk has nothing to do",
                               detail)
                    else:
                        R.ok(f"{clip} clippable campaign(s) ready for the walk", detail)
                except Exception as e:
                    R.warn("could not compute clippable/category counts",
                           f"{type(e).__name__}: {e}")
            else:
                R.warn("pickcampaign.py not importable — skipped clippable/category counts",
                       f"{type(_PICK_IMPORT_ERR).__name__}: {_PICK_IMPORT_ERR}")

    # git status of both repos
    for label, repo in (("clipper", str(C.ROOT)), ("scout", SCOUT_DIR)):
        if not os.path.isdir(os.path.join(repo, ".git")):
            R.warn(f"{label} repo: not a git repo (or .git missing)", repo)
            continue
        clean, code, other = _git_status(repo)
        if clean:
            R.ok(f"{label} repo clean (no changes)")
        elif code:
            R.warn(f"{label} repo has UNCOMMITTED CODE changes ({len(code)} file(s))",
                   code[:12] + ([f"… (+{len(code) - 12} more)"] if len(code) > 12 else [])
                   + ([f"(+{len(other)} runtime/data file(s), ignored)"] if other else []))
        else:
            R.ok(f"{label} repo: only runtime/data changes ({len(other)} file(s), ignored)",
                 other[:8] + (["…"] if len(other) > 8 else []))


# --- PLUGINS -------------------------------------------------------------------
# Run the bgutil / PO-token provider load in a PRISTINE subprocess (no common.py idempotency
# guard) so a genuine duplicate-registration ("already registered" / AssertionError) surfaces
# instead of being silently patched. Returns a JSON blob we parse below.
_PLUGIN_PROBE = r"""
import io, json, contextlib, glob, os, sys
result = {"providers": [], "plugin_dirs": [], "bgutil_dirs": [], "markers": [], "error": None,
          "pip_version": None}
buf = io.StringIO()
try:
    import logging
    logging.basicConfig(stream=buf, level=logging.DEBUG)
except Exception:
    pass
try:
    from importlib.metadata import version
    try:
        result["pip_version"] = version("bgutil-ytdlp-pot-provider")
    except Exception:
        result["pip_version"] = None
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        from yt_dlp import YoutubeDL
        from yt_dlp.plugins import load_all_plugins, directories
        load_all_plugins()
        ydl = YoutubeDL({"quiet": True})
    from yt_dlp.extractor.youtube.pot.provider import _pot_providers
    reg = getattr(_pot_providers, "value", _pot_providers)
    result["providers"] = sorted(reg.keys()) if hasattr(reg, "keys") else list(reg)
    dirs = list(directories())
    result["plugin_dirs"] = [str(d) for d in dirs]
    # find every plugin dir that actually ships a bgutil getpot file (duplicate detection)
    seen = []
    for d in dirs:
        hits = glob.glob(os.path.join(str(d), "**", "getpot_bgutil*.py"), recursive=True)
        if hits:
            seen.append(str(d))
    result["bgutil_dirs"] = seen
    combined = err.getvalue() + buf.getvalue()
    for ln in combined.splitlines():
        low = ln.lower()
        if "already registered" in low or "assertionerror" in low or "traceback" in low:
            result["markers"].append(ln.strip()[:200])
except Exception as e:
    result["error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(result))
"""


def check_plugins():
    R.section("PLUGINS (bgutil PO-token provider)")
    try:
        out = subprocess.run([sys.executable, "-c", _PLUGIN_PROBE],
                             capture_output=True, text=True, timeout=90)
    except Exception as e:
        R.fail("bgutil plugin probe could not run", f"{type(e).__name__}: {e}")
        return
    line = (out.stdout or "").strip().splitlines()
    blob = None
    for ln in reversed(line):
        try:
            blob = json.loads(ln)
            break
        except Exception:
            continue
    if blob is None:
        R.fail("bgutil plugin probe returned no parseable result",
               [(out.stdout or "")[-300:], (out.stderr or "")[-300:]])
        return
    if blob.get("error"):
        R.fail("bgutil plugin load raised an error", blob["error"])
        return

    # one clean install: exactly one plugin dir provides getpot_bgutil files
    bgutil_dirs = blob.get("bgutil_dirs") or []
    if len(bgutil_dirs) == 1:
        R.ok("bgutil installed once (single plugin dir provides the getpot files)",
             [bgutil_dirs[0], f"pip package: bgutil-ytdlp-pot-provider {blob.get('pip_version')}"])
    elif len(bgutil_dirs) == 0:
        R.fail("bgutil getpot plugin files NOT found in any yt-dlp plugin dir",
               ["plugin dirs searched:"] + (blob.get("plugin_dirs") or []))
    else:
        R.fail(f"DUPLICATE bgutil plugin dirs ({len(bgutil_dirs)}) — will trigger "
               "'already registered' spam", bgutil_dirs)

    # a stray ACTIVE repo plugin dir (should be renamed to plugin_DISABLED)
    repo_plugin = Path(r"C:\Users\knigh\bgutil-ytdlp-pot-provider\plugin")
    if repo_plugin.is_dir():
        R.warn("an ACTIVE 'plugin' dir exists in the bgutil repo — rename it to plugin_DISABLED",
               str(repo_plugin))

    # providers registered
    provs = blob.get("providers") or []
    expected = {"BgUtilHTTP", "BgUtilScriptNode"}
    if expected.issubset(set(provs)):
        R.ok("PO-token providers registered", ", ".join(provs))
    elif provs:
        R.warn("PO-token providers registered but missing an expected one",
               f"got: {', '.join(provs)}")
    else:
        R.fail("NO PO-token providers registered — YouTube downloads will 403")

    # duplicate-registration markers
    markers = blob.get("markers") or []
    if not markers:
        R.ok("yt-dlp loaded the bgutil provider with NO 'already registered' errors")
    else:
        R.fail("duplicate-registration / traceback markers seen during plugin load",
               markers[:6])


# --- main ----------------------------------------------------------------------
def main():
    print("PREFLIGHT — read-only overnight-run readiness check")
    print(f"clipper root: {C.ROOT}")
    cfg, state = _effective_config()
    check_config(cfg)
    check_state(state)
    check_environment(cfg)
    check_scout()
    check_plugins()

    print(f"\n{'=' * 72}")
    print(f"PREFLIGHT: {R.passed} passed, {R.warnings} warnings, {R.failures} failures")
    print("=" * 72)
    if R.failures:
        print("\n✗ NOT ready — resolve the FAIL item(s) above before the overnight run.")
        sys.exit(1)
    if R.warnings:
        print("\n⚠ Ready with warnings — glance at the WARN item(s), then you can run.")
    else:
        print("\n✓ All clear — good to go.")
    sys.exit(0)


if __name__ == "__main__":
    main()
