"""Shared plumbing for the clipper pipeline: paths, state/checkpointing, fail-loud
logging, subprocess + ffmpeg helpers, and the Groq client.

Every stage imports this. Nothing here guesses: on any ambiguity or missing tool it
calls fail() and stops, per instructions.md ("FAIL LOUD").
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Captions/moment text and status glyphs are UTF-8; Windows consoles default to
# cp1252 and would crash on them. Force UTF-8 output everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- paths (ROOT = the clipper/ folder, i.e. the parent of scripts/) -----------
ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = ROOT / "campaign"
FOOTAGE = CAMPAIGN / "footage"
ASSETS = CAMPAIGN / "assets"
DOCS = CAMPAIGN / "docs"          # brand guides / rule docs / sheets / pdfs
OTHER = CAMPAIGN / "other"        # kept-but-unhandled files (never silently dropped)
TRANSCRIPTS = CAMPAIGN / "transcripts"
DRAFTS = ROOT / "drafts"
DRAFTS_ARCHIVE = ROOT / "drafts_archive"   # prior-campaign drafts moved here (never deleted)
MEMORY = ROOT / "memory"
STATE_PATH = ROOT / "state.json"

BRIEF_MD = CAMPAIGN / "brief.md"
RULES_JSON = CAMPAIGN / "rules.json"
KNOWLEDGE_MD = CAMPAIGN / "knowledge.md"   # per-campaign digest (never leaks to longterm)
CAMPAIGN_MANIFEST = CAMPAIGN / "manifest.json"
MOMENTS_JSON = CAMPAIGN / "moments.json"
SELECTED_JSON = CAMPAIGN / "selected.json"
CAPTIONS_JSON = CAMPAIGN / "captions.json"
DRAFTS_MANIFEST = DRAFTS / "manifest.json"
# Per-item Groq checkpoints (Unit 2b): select/captions write finished work here so a DAILY
# Groq-cap stop can --resume without redoing completed API calls. Deleted on stage completion.
SELECT_PARTIAL = CAMPAIGN / "select_partial.json"
CAPTIONS_PARTIAL = CAMPAIGN / "captions_partial.json"

# Default to the 70B model: the 8B ('llama-3.1-8b-instant') scored moments randomly
# (100 to a garbage clip in one run, 0 to everything the next), which wrecked both
# selection and caption quality. 70B scores consistently with sensible reasons.
# Override with GROQ_MODEL=llama-3.1-8b-instant if you hit free-tier daily token limits.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# The mandatory blocklist floor for the current campaign context (WTF Leagues).
# Intake merges these with anything it finds in the brief.
DEFAULT_BANNED_WORDS = ["bet", "gamble", "gambling", "casino", "odds", "wager", "stake"]

# Audience framing injected into EVERY Groq prompt (select + captions) so the model
# writes to the account's voice, not a generic one. This is the current Account A /
# WTF Leagues context; it augments (does not replace) per-campaign knowledge.md.
AUDIENCE_CONTEXT = (
    "WTF Leagues: hamster racing / novelty sports league. Audience: Gen Z meme "
    "culture, F1/sports-parody crossover humor, chaos enjoyers. Reference caption "
    "tone: 'omg mum they race hamsters', 'Hamdo Norris'. Unhinged-but-deadpan "
    "energy (final casing is Title Case, applied downstream — write for voice, not case)."
)


# --- logging / fail-loud -------------------------------------------------------
def log(msg):
    print(f"[clipper] {msg}", flush=True)


def warn(msg):
    print(f"[clipper] ⚠ {msg}", flush=True)


def fail(msg, code=1):
    """Print a clear error and stop. Never return."""
    print(f"\n✗ FAIL: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(code)


# Distinct exit code for a resumable STOP (Groq daily cap) so an orchestrator can tell it apart
# from a hard failure (code 1) — the run isn't broken, it's paused until the quota resets.
RESUMABLE_STOP_CODE = 7


def stop_resumable(msg, code=RESUMABLE_STOP_CODE):
    """A clean, RESUMABLE stop (not a crash): print a clear ⏸ notice and exit with a distinct
    code. Used when the Groq DAILY cap is hit mid-stage — progress is checkpointed, and a
    `--resume` after the cap resets continues exactly where it stopped."""
    print(f"\n⏸ STOPPED (resumable): {msg}\n", file=sys.stderr, flush=True)
    sys.exit(code)


class GroqDailyCapError(Exception):
    """Groq DAILY token/request cap (TPD/RPD) — distinct from a short per-minute limit. Raised
    by groq_chat so the caller can checkpoint finished work and stop_resumable() rather than
    spin retries into a quota that won't reset until tomorrow."""
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


class NothingUsable(Exception):
    """A stage found NOTHING clippable for the picked campaign — no footage, zero indexable
    sources, or no live moments. Raised (instead of fail()) so the orchestrator can either
    dead-end loud (default) or, with --auto-advance, move on to the next ranked campaign."""


def offline_mode():
    """Degraded/offline mode for testing without Groq or model downloads.

    Enabled by CLIPPER_OFFLINE=1. In this mode, select/captions use deterministic
    local heuristics and index skips whisper transcription. Normal runs require the
    real tools and fail loud without them.
    """
    return os.environ.get("CLIPPER_OFFLINE") == "1"


# --- filesystem / json ---------------------------------------------------------
def ensure_dirs():
    for d in (CAMPAIGN, FOOTAGE, ASSETS, DOCS, OTHER, TRANSCRIPTS, DRAFTS,
              MEMORY, MEMORY / "accounts"):
        d.mkdir(parents=True, exist_ok=True)


def load_knowledge():
    """Per-campaign digest (knowledge.md) as text, or '' if not built yet. Every
    downstream stage reads this so campaign context is applied, not re-derived."""
    p = KNOWLEDGE_MD
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"corrupt JSON at {path}: {e}")


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --- state / checkpointing -----------------------------------------------------
def load_state():
    return load_json(STATE_PATH, default={"campaign": None, "stages": {}, "config": {}})


def save_state(state):
    save_json(STATE_PATH, state)


def stage_done(state, name):
    return bool(state.get("stages", {}).get(name, {}).get("done"))


def mark_stage(state, name, **extra):
    state.setdefault("stages", {})[name] = {"done": True, "at": now_iso(), **extra}
    save_state(state)


def stage_meta(state, name):
    return state.get("stages", {}).get(name, {})


def _archive_drafts(label):
    """Move everything under drafts/ into drafts_archive/<label>-<timestamp>/ so a new
    campaign never mixes its clips with the previous one's. Never deletes — always moves."""
    if not DRAFTS.exists():
        return
    items = [p for p in DRAFTS.iterdir()]
    if not items:
        return
    slug = re.sub(r"[^a-z0-9]+", "-", (label or "prev").lower()).strip("-") or "prev"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = DRAFTS_ARCHIVE / f"{slug}-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in items:
        try:
            shutil.move(str(p), str(dest / p.name))
            moved += 1
        except Exception as e:
            warn(f"could not archive draft {p.name}: {e}")
    if moved:
        log(f"archived {moved} prior draft item(s) → {dest.relative_to(ROOT)}")


def activate_campaign(state, name):
    """Scope stage checkpoints PER-CAMPAIGN so a NEW campaign runs fresh without --force
    and never inherits the prior campaign's 'done' stages or its drafts.

    No-op when `name` is already active (so re-running intake or a stage on the SAME
    campaign keeps its checkpoints — still resumable). On a real switch: stash the
    outgoing campaign's stages under state['campaigns'][old], archive its drafts, then
    restore the incoming campaign's stages (empty for a brand-new one → clean run).
    Returns the (possibly mutated) state."""
    if not name:
        return state
    old = state.get("campaign")
    if old == name:
        return state
    camps = state.setdefault("campaigns", {})
    if old:
        camps[old] = {"stages": state.get("stages", {})}
        _archive_drafts(old)
    elif DRAFTS.exists() and any(DRAFTS.iterdir()):
        # legacy state with drafts but no recorded campaign — don't let them bleed through
        _archive_drafts("previous")
    state["stages"] = camps.get(name, {}).get("stages", {})
    state["campaign"] = name
    save_state(state)
    return state


# --- external tools ------------------------------------------------------------
def require_exe(name):
    if shutil.which(name) is None:
        fail(f"'{name}' not found on PATH. Install it (see README.md / setup.sh).")


def run_cmd(cmd, desc=None, check=True, capture=False):
    """Run a subprocess. On failure (when check), fail loud with the tail of stderr."""
    if desc:
        log(desc)
    proc = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        fail(f"command failed ({' '.join(str(c) for c in cmd[:3])} …):\n" + "\n".join(tail))
    return proc


def ffprobe_duration(path):
    """Duration in seconds via ffprobe, or fail loud."""
    require_exe("ffprobe")
    proc = run_cmd(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture=True,
    )
    try:
        return float((proc.stdout or "").strip())
    except (ValueError, AttributeError):
        fail(f"could not read duration for {path}")


def audio_stream_count(path):
    """Number of audio streams in the source (0 if none). VODs often split mic and
    game/commentary onto SEPARATE tracks; using only the first drops audio, so callers
    merge all tracks when this is >1."""
    require_exe("ffprobe")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return len([ln for ln in (proc.stdout or "").splitlines() if ln.strip()])


# --- groq ----------------------------------------------------------------------
def groq_client():
    """Return a Groq client, or None in offline mode. Fail loud if the key/package
    is missing in a normal (non-offline) run."""
    if offline_mode():
        return None
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        fail("GROQ_API_KEY is not set. Set it (export GROQ_API_KEY=...) or run in "
             "offline test mode with CLIPPER_OFFLINE=1.")
    try:
        from groq import Groq
    except ImportError:
        fail("the 'groq' package is not installed. Run: pip install -r requirements.txt")
    return Groq(api_key=key)


# A required wait longer than this is a DAILY cap, not a per-minute one (per-minute waits are
# seconds; daily waits are minutes/hours). Retrying into a daily cap only hangs the run.
_MAX_MINUTE_WAIT_S = 90.0


def _parse_wait_seconds(msg):
    """Seconds from a Groq 'try again in 5m30s' / '8.5s' / '2h34m' hint, or None."""
    m = re.search(r"try again in ([0-9hms.\s]+)", msg, re.I)
    if not m:
        return None
    total, found = 0.0, False
    for val, unit in re.findall(r"([\d.]+)\s*(h|m|s)", m.group(1), re.I):
        found = True
        total += float(val) * {"h": 3600, "m": 60, "s": 1}[unit.lower()]
    return total if found else None


def classify_rate_limit(msg):
    """('daily' | 'minute' | 'error', wait_seconds_or_None). Mirrors scout's classifier: 'daily'
    = TPD/RPD/'per day' or a wait longer than a per-minute window (retrying can't help today);
    'minute' = a short recoverable per-minute (TPM/RPM) limit; 'error' = a non-rate failure."""
    low = msg.lower()
    is_rate = ("429" in msg or "rate limit" in low or "rate_limit" in low
               or "tpm" in low or "tpd" in low or "rpm" in low or "rpd" in low)
    if not is_rate:
        return "error", None
    wait = _parse_wait_seconds(msg)
    is_daily = ("per day" in low or "tpd" in low or "rpd" in low or "daily" in low
                or (wait is not None and wait > _MAX_MINUTE_WAIT_S))
    return ("daily" if is_daily else "minute"), wait


def groq_chat(client, system, user, temperature=0.8, max_tokens=1024, retries=6):
    """One chat completion. A short PER-MINUTE rate limit (429 TPM/RPM) is transient: back off
    and retry, honoring Groq's 'try again in Xs' hint. A DAILY cap (TPD/RPD, or a wait longer
    than a per-minute window) raises GroqDailyCapError immediately — retrying a dead daily quota
    only hangs the run; the caller checkpoints + stops resumably. Any other error fails loud."""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except GroqDailyCapError:
            raise
        except Exception as e:
            msg = str(e)
            kind, wait = classify_rate_limit(msg)
            if kind == "daily":
                raise GroqDailyCapError(msg, retry_after=wait)
            if kind == "minute" and attempt < retries:
                w = (wait + 0.5) if wait else min(2.0 * (attempt + 1), 20.0)
                warn(f"Groq rate limit — waiting {w:.1f}s then retrying "
                     f"(attempt {attempt + 1}/{retries})…")
                time.sleep(w)
                continue
            fail(f"Groq API call failed ({GROQ_MODEL}): {e}")
