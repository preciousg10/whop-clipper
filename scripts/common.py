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
    "tone: 'omg mum they race hamsters', 'Hamdo Norris'. Lowercase, "
    "unhinged-but-deadpan energy."
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


def groq_chat(client, system, user, temperature=0.8, max_tokens=1024, retries=6):
    """One chat completion. Transient rate-limit (429 TPM/RPM on the free tier) is NOT
    fatal: back off and retry, honoring Groq's 'try again in Xs' hint. Any other API
    error, or exhausting retries, fails loud. Returns the message string."""
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
        except Exception as e:
            msg = str(e)
            is_rate = "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()
            if is_rate and attempt < retries:
                m = re.search(r"try again in ([\d.]+)\s*s", msg)
                wait = (float(m.group(1)) + 0.5) if m else min(2.0 * (attempt + 1), 20.0)
                warn(f"Groq rate limit — waiting {wait:.1f}s then retrying "
                     f"(attempt {attempt + 1}/{retries})…")
                time.sleep(wait)
                continue
            fail(f"Groq API call failed ({GROQ_MODEL}): {e}")
