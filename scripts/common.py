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
DRAFTS_BATCH = ROOT / "drafts_batch"       # auto-advance batch accumulation (staging across
                                           # campaigns; finalized back into drafts/ at walk end)
MEMORY = ROOT / "memory"
STATE_PATH = ROOT / "state.json"

BRIEF_MD = CAMPAIGN / "brief.md"
RULES_JSON = CAMPAIGN / "rules.json"
KNOWLEDGE_MD = CAMPAIGN / "knowledge.md"   # per-campaign digest (never leaks to longterm)
POSTING_CHECKLIST = CAMPAIGN / "POSTING_CHECKLIST.md"  # human do-this-when-posting checklist
CAMPAIGN_MANIFEST = CAMPAIGN / "manifest.json"
MOMENTS_JSON = CAMPAIGN / "moments.json"
SELECTED_JSON = CAMPAIGN / "selected.json"
CAPTIONS_JSON = CAMPAIGN / "captions.json"
DRAFTS_MANIFEST = DRAFTS / "manifest.json"
# Per-item Groq checkpoints (Unit 2b): select/captions write finished work here so a DAILY
# Groq-cap stop can --resume without redoing completed API calls. Deleted on stage completion.
SELECT_PARTIAL = CAMPAIGN / "select_partial.json"
CAPTIONS_PARTIAL = CAMPAIGN / "captions_partial.json"

# Default to gpt-oss-120b: llama-3.3-70b-versatile was DEPRECATED/retired by Groq mid-2026,
# and the 8B ('llama-3.1-8b-instant') scored moments randomly (100 to garbage one run, 0 the
# next), wrecking selection + caption quality. gpt-oss-120b scores consistently with sensible
# reasons. NOTE it is a REASONING model: hidden reasoning tokens count against max_tokens, so
# _GroqProvider forces reasoning_effort=low + a token reserve (see _is_reasoning_model) — without
# that a large prompt burns the whole budget on reasoning and returns EMPTY content.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# LLM FAILOVER CHAIN models (all free-tier). Order + enable via config `llm_providers`.
# NOTE: gemini-2.0-flash was retired by Google (404) mid-2026 — default is now the current
# free-tier gemini-2.5-flash. Groq/Cerebras 70B defaults are still live. Override any via env.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")
DEFAULT_LLM_PROVIDERS = ["groq", "gemini", "cerebras"]

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


# --- yt-dlp auth cookies (gated YouTube/Kick VODs 403 without a logged-in session) ------
_COOKIES_WARNED = False


def cookies_file():
    """Absolute path to the Netscape cookies.txt yt-dlp should use for gated VODs
    (YouTube/Kick 403 without a logged-in session), or None if it isn't present.

    Path comes from the COOKIES_FILE env var, else the default ROOT/cookies.txt. When the
    file is MISSING we warn ONCE (some sources still work unauthenticated) and return None —
    callers still attempt the download. The file's CONTENTS are never logged (it holds a live
    session); only its path is printed."""
    global _COOKIES_WARNED
    p = os.environ.get("COOKIES_FILE") or str(ROOT / "cookies.txt")
    if os.path.exists(p):
        return p
    if not _COOKIES_WARNED:
        warn(f"no cookies file at {p} — YouTube/Kick may 403 (set COOKIES_FILE or place "
             f"cookies.txt there). Attempting anyway; some sources work without.")
        _COOKIES_WARNED = True
    return None


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


# --- LLM failover chain: Groq(keys 1..N) -> Gemini -> Cerebras ---------------------
# All three are free-tier; stacking them + MULTIPLE KEYS per provider multiplies daily capacity.
# Keys come ONLY from env vars — never hardcoded, never written to disk, never logged. Each
# provider expands to one entry PER KEY: numbered keys `GROQ_API_KEY_1..N` (however many are set)
# then the bare `GROQ_API_KEY` as a fallback when no numbered keys exist (same for GEMINI_/
# CEREBRAS_). On a DAILY cap (or a very long retry-after) we advance to the NEXT KEY of the same
# provider, and only after ALL that provider's keys are capped do we fall to the next provider —
# staying put for the rest of the run; short per-minute limits back off + retry on the CURRENT
# key first. All keys of all providers capped -> GroqDailyCapError
# (the caller checkpoints + stops resumably). Every provider returns a PLAIN STRING so the
# existing select/caption JSON parsers are unchanged.
_MAX_MINUTE_WAIT_S = 90.0     # a required wait longer than this reads as a daily cap


def _parse_wait_seconds(msg):
    """Seconds from a provider's retry hint — Groq 'try again in 5m30s', Gemini
    'retry_delay { seconds: N }', or 'retry after Ns' — or None."""
    m = re.search(r"try again in ([0-9hms.\s]+)", msg, re.I)
    if m:
        total, found = 0.0, False
        for val, unit in re.findall(r"([\d.]+)\s*(h|m|s)", m.group(1), re.I):
            found = True
            total += float(val) * {"h": 3600, "m": 60, "s": 1}[unit.lower()]
        if found:
            return total
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"retry[\s_-]*after[\"'\s:]*(\d+)", msg, re.I)
    if m:
        return float(m.group(1))
    return None


def is_dead_model_error(msg):
    """True when the provider says the MODEL itself is gone/unusable — a 404 / model-not-found
    (e.g. Gemini retired gemini-2.0-flash). This is NOT transient: retrying the same provider
    just re-404s, so the chain must skip it IMMEDIATELY rather than burn all its retries."""
    low = msg.lower()
    if "model_not_found" in low or "modelnotfound" in low:
        return True
    # a bare 404 from any provider means the endpoint/model isn't there — never a rate limit
    if "404" in msg and "429" not in msg:
        return True
    model_words = ("model" in low or "models/" in low)
    gone = ("not found" in low or "does not exist" in low or "is not supported" in low
            or "not available" in low or "deprecated" in low or "has been removed" in low
            or "no longer available" in low)
    return model_words and gone


def classify_rate_limit(msg):
    """('daily' | 'minute' | 'error', wait_seconds_or_None) — works across Groq (TPD/RPD),
    Gemini (…PerDay/…PerMinute quota, ResourceExhausted) and Cerebras/OpenAI (429 rate limit).
    'daily' = per-day quota or a wait longer than a per-minute window (switch providers);
    'minute' = a short recoverable per-minute limit (back off, retry same); 'error' = non-rate."""
    low = msg.lower()
    is_rate = ("429" in msg or "rate limit" in low or "rate_limit" in low
               or "tpm" in low or "tpd" in low or "rpm" in low or "rpd" in low
               or "quota" in low or "resource_exhausted" in low or "resourceexhausted" in low
               or "insufficient_quota" in low)
    if not is_rate:
        return "error", None
    wait = _parse_wait_seconds(msg)
    daily = any(k in low for k in ("per day", "perday", "per_day", "tpd", "rpd", "daily",
                                   "requests per day", "tokens per day"))
    minute = any(k in low for k in ("per minute", "perminute", "per_minute", "tpm", "rpm"))
    if daily and not minute:
        return "daily", wait
    if minute and not daily:
        return "minute", wait
    # ambiguous (bare 'quota' / 'resource_exhausted'): decide by the wait hint.
    if wait is not None and wait > _MAX_MINUTE_WAIT_S:
        return "daily", wait
    return "minute", wait


# --- per-provider adapters (each returns a PLAIN STRING; keys read from env only) ----------
# Opt-in raw-response logging: CLIPPER_DEBUG_LLM=1 dumps the FULL provider response object
# (model_dump) for the first call of each provider, so a shape change (e.g. a new reasoning
# model returning empty content) is diagnosable against reality instead of guessed at.
_LLM_DEBUG = os.environ.get("CLIPPER_DEBUG_LLM") == "1"
_LLM_DEBUG_DUMPED = set()

# gpt-oss and other reasoning models emit HIDDEN reasoning tokens that count against
# max_tokens BEFORE any answer content. With the default ('medium'/'high') effort a large
# prompt burns the entire budget on reasoning and returns finish_reason='length' with EMPTY
# content — the exact select/caption failure after switching to openai/gpt-oss-120b. We force
# 'low' effort AND add a reasoning reserve to max_tokens so answer content is never starved.
_REASONING_MODEL_HINTS = ("gpt-oss", "o1", "o3", "o4", "deepseek-r1", "qwq", "reasoning")
_REASONING_TOKEN_RESERVE = 700


def _is_reasoning_model(model):
    m = (model or "").lower()
    return any(h in m for h in _REASONING_MODEL_HINTS)


def _openai_content(resp, provider_name):
    """Normalize an OpenAI-shaped chat response (Groq/Cerebras) to a PLAIN answer string.

    Reasoning models split output: the chain-of-thought lands in message.reasoning and the
    real answer in message.content. We return content; if it's empty (e.g. truncated by a
    length finish) we log a LOUD diagnostic (finish_reason + reasoning-token count + a
    reasoning preview) instead of silently handing '' to the parser. CLIPPER_DEBUG_LLM=1
    additionally dumps the whole response object once per provider."""
    if _LLM_DEBUG and provider_name not in _LLM_DEBUG_DUMPED:
        _LLM_DEBUG_DUMPED.add(provider_name)
        try:
            log(f"[LLM RAW {provider_name}] {json.dumps(resp.model_dump(), default=str)[:4000]}")
        except Exception as e:
            log(f"[LLM RAW {provider_name}] <could not dump: {e}>")
    choice = resp.choices[0]
    content = (getattr(choice.message, "content", None) or "").strip()
    if content:
        return content
    # Empty content — diagnose loudly (this is what silently produced flat scores/templates).
    finish = getattr(choice, "finish_reason", "?")
    reasoning = (getattr(choice.message, "reasoning", None) or "")
    rtoks = None
    try:
        rtoks = resp.usage.completion_tokens_details.reasoning_tokens
    except Exception:
        pass
    warn(f"{provider_name} returned EMPTY content (finish_reason={finish}, "
         f"reasoning_tokens={rtoks}). Likely reasoning ate the token budget — raising "
         f"max_tokens / lowering reasoning_effort. reasoning preview: {reasoning[:160]!r}")
    return ""


class _BaseProvider:
    """Shared per-instance key metadata + a human LABEL for logging. One provider CLASS can be
    instantiated once PER KEY (GROQ key 1/4, key 2/4, …); `label` distinguishes them in logs
    without EVER printing the key value."""
    name = "?"

    def _set_key_meta(self, key_index, key_total):
        self.key_index = key_index
        self.key_total = key_total

    @property
    def label(self):
        if getattr(self, "key_total", 1) > 1:
            return f"{self.name} key {self.key_index}/{self.key_total}"
        return self.name


class _GroqProvider(_BaseProvider):
    name = "GROQ"

    def __init__(self, api_key, key_index=1, key_total=1):
        self._set_key_meta(key_index, key_total)
        from groq import Groq
        self.model = GROQ_MODEL
        self._client = Groq(api_key=api_key)
        self._reasoning = _is_reasoning_model(self.model)
        if self._reasoning and key_index == 1:
            log(f"GROQ: '{self.model}' is a reasoning model — using reasoning_effort=low "
                f"+ token reserve so answer content isn't starved by reasoning tokens.")

    def complete(self, system, user, temperature, max_tokens):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens)
        if self._reasoning:
            kwargs["reasoning_effort"] = "low"                  # minimize hidden reasoning tokens
            kwargs["max_tokens"] = max_tokens + _REASONING_TOKEN_RESERVE   # headroom for the answer
        resp = self._client.chat.completions.create(**kwargs)
        return _openai_content(resp, self.name)


class _GeminiProvider(_BaseProvider):
    name = "GEMINI"

    def __init__(self, api_key, key_index=1, key_total=1):
        self._set_key_meta(key_index, key_total)
        import warnings
        with warnings.catch_warnings():          # hush the lib's own deprecation FutureWarning
            warnings.simplefilter("ignore")
            import google.generativeai as genai
        self.model = GEMINI_MODEL
        # Gemini configures the key globally per generate; we re-apply it before each call so
        # multiple Gemini keys don't clobber each other on the module-level config.
        self._api_key = api_key
        genai.configure(api_key=api_key)
        self._genai = genai

    def complete(self, system, user, temperature, max_tokens):
        self._genai.configure(api_key=self._api_key)   # ensure THIS key is active (multi-key safe)
        model = self._genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(
            user, generation_config={"temperature": temperature,
                                     "max_output_tokens": max_tokens})
        # Normalize to a plain string like the OpenAI-shaped providers. `.text` raises when a
        # response was blocked/empty — fall back to stitching candidate parts, else "".
        try:
            return resp.text or ""
        except Exception:
            out = []
            for cand in (getattr(resp, "candidates", None) or []):
                for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
                    if getattr(part, "text", None):
                        out.append(part.text)
            return "".join(out)


class _CerebrasProvider(_BaseProvider):
    name = "CEREBRAS"

    def __init__(self, api_key, key_index=1, key_total=1):
        self._set_key_meta(key_index, key_total)
        from openai import OpenAI          # Cerebras exposes an OpenAI-compatible endpoint
        self.model = CEREBRAS_MODEL
        self._client = OpenAI(api_key=api_key,
                              base_url="https://api.cerebras.ai/v1")
        self._reasoning = _is_reasoning_model(self.model)

    def complete(self, system, user, temperature, max_tokens):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens)
        if self._reasoning:
            kwargs["reasoning_effort"] = "low"
            kwargs["max_tokens"] = max_tokens + _REASONING_TOKEN_RESERVE
        resp = self._client.chat.completions.create(**kwargs)
        return _openai_content(resp, self.name)


_PROVIDER_SPECS = {
    "groq": ("GROQ_API_KEY", _GroqProvider),
    "gemini": ("GEMINI_API_KEY", _GeminiProvider),
    "cerebras": ("CEREBRAS_API_KEY", _CerebrasProvider),
}
_MISSING_WARNED = set()      # warn once per missing provider


def _collect_provider_keys(base):
    """Ordered list of API-key VALUES for a provider, read ONLY from env (never logged/returned
    to callers that could log them). Detects HOWEVER MANY numbered keys are set — `BASE_1`,
    `BASE_2`, … in numeric order (non-contiguous is fine: 1,2,4 → three keys). The bare `BASE`
    (e.g. GROQ_API_KEY) is used as a FALLBACK/alias only when NO numbered keys exist, so adding a
    5th numbered key later is picked up automatically with zero code change."""
    pat = re.compile(r"^" + re.escape(base) + r"_(\d+)$")
    numbered = []
    for env_name, val in os.environ.items():
        m = pat.match(env_name)
        if m and val and val.strip():
            numbered.append((int(m.group(1)), val.strip()))
    if numbered:
        numbered.sort(key=lambda t: t[0])
        return [v for _, v in numbered]
    bare = os.environ.get(base)
    return [bare.strip()] if bare and bare.strip() else []


def _make_providers(name):
    """Build ALL available provider instances for `name` — ONE per env key (GROQ key 1/4 … 4/4)
    — or [] if no key is set / the library won't import. Multiple keys of a provider are
    exhausted (in order) before the chain falls to the NEXT provider."""
    spec = _PROVIDER_SPECS.get(name)
    if not spec:
        warn(f"unknown LLM provider '{name}' in llm_providers — skipping.")
        return []
    env_base, cls = spec
    keys = _collect_provider_keys(env_base)
    if not keys:
        if name not in _MISSING_WARNED:
            warn(f"LLM provider {name.upper()} skipped — no key set "
                 f"({env_base} or {env_base}_1, {env_base}_2, …).")
            _MISSING_WARNED.add(name)
        return []
    total = len(keys)
    out = []
    for i, key in enumerate(keys, 1):
        try:
            out.append(cls(api_key=key, key_index=i, key_total=total))
        except Exception as e:
            # A library/import problem hits every key the same way — warn once and stop trying
            # this provider. (A per-key credential problem surfaces later as a normal API error.)
            if name not in _MISSING_WARNED:
                warn(f"LLM provider {name.upper()} unavailable "
                     f"({e.__class__.__name__}: {e}) — skipping.")
                _MISSING_WARNED.add(name)
            break
    if total > 1 and out:
        log(f"LLM provider {name.upper()}: {len(out)} keys detected → will rotate "
            f"key 1..{len(out)} before failing over.")
    return out


class LLMChain:
    """An ordered chain of available provider+key entries with a CURRENT pointer (Groq key1,
    Groq key2, …, Gemini, Cerebras). On a daily cap the pointer advances to the next KEY/provider
    (and never rewinds within the process); per-minute limits retry in place. A process restart
    (a fresh `run.py`) rebuilds the chain at the front (Groq key 1)."""

    def __init__(self, providers):
        self.providers = providers
        self.idx = 0
        self._answered = set()

    def status(self):
        chain = "→".join(p.label for p in self.providers)
        return f"{chain} (active: {self.providers[self.idx].label})"

    def _try(self, p, system, user, temperature, max_tokens, retries):
        """(text, 'ok', None) on success, or (None, 'failover', reason). Backs off + retries a
        short per-minute limit (or transient error) on THIS provider before giving up."""
        for attempt in range(retries + 1):
            try:
                return p.complete(system, user, temperature, max_tokens), "ok", None
            except Exception as e:
                if is_dead_model_error(str(e)):     # 404/model-gone: not transient, skip NOW
                    return None, "failover", (f"{p.label} model '{p.model}' unavailable "
                                              f"(404/model-not-found) — {str(e)[:80]}")
                kind, wait = classify_rate_limit(str(e))
                if kind == "daily":
                    return None, "failover", f"{p.label} DAILY cap"
                if attempt < retries:
                    w = (wait + 0.5) if (wait and kind == "minute") else min(2.0 * (attempt + 1), 20.0)
                    warn(f"{p.label} {'rate limit' if kind == 'minute' else 'error'} "
                         f"({str(e)[:80]}) — retry in {w:.1f}s ({attempt + 1}/{retries})…")
                    time.sleep(w)
                    continue
                return None, "failover", (f"{p.label} per-minute limit persisted"
                                          if kind == "minute" else f"{p.label} error: {str(e)[:80]}")

    def chat(self, system, user, temperature=0.8, max_tokens=1024, retries=6):
        tried = []
        while self.idx < len(self.providers):
            p = self.providers[self.idx]
            text, action, reason = self._try(p, system, user, temperature, max_tokens, retries)
            if action == "ok":
                if p.label not in self._answered:     # log which provider+key answered (once each)
                    log(f"LLM: answered by {p.label} ({p.model}).")
                    self._answered.add(p.label)
                return text
            tried.append(p.label)
            nxt = self.providers[self.idx + 1].label if self.idx + 1 < len(self.providers) else None
            if nxt:
                warn(f"{reason} → failing over to {nxt}.")
            else:
                warn(f"{reason} — no more providers/keys in the chain.")
            self.idx += 1
        raise GroqDailyCapError(
            f"all LLM providers capped/unavailable ({', '.join(tried)}) — checkpoint and "
            f"--resume after a quota resets.")


_LLM_CHAIN = None


def llm_client(cfg=None):
    """The process-wide LLM failover chain (Groq→Gemini→Cerebras by default), or None in offline
    mode. Built ONCE: order/enable comes from cfg['llm_providers']; a provider with a missing key
    or library is dropped (warned once). Fail loud only if NONE are available."""
    global _LLM_CHAIN
    if offline_mode():
        return None
    if _LLM_CHAIN is None:
        order = list((cfg or {}).get("llm_providers") or DEFAULT_LLM_PROVIDERS)
        providers = []
        for n in order:                       # each provider expands to ONE entry per env key
            providers.extend(_make_providers(str(n).lower().strip()))
        if not providers:
            fail("no LLM provider available — set at least one of GROQ_API_KEY / GEMINI_API_KEY "
                 "/ CEREBRAS_API_KEY (or run offline with CLIPPER_OFFLINE=1).")
        _LLM_CHAIN = LLMChain(providers)
        log(f"LLM chain ready: {_LLM_CHAIN.status()}")
    return _LLM_CHAIN


def llm_chat(chain, system, user, temperature=0.8, max_tokens=1024, retries=6):
    """One completion through the failover chain. Raises GroqDailyCapError only when EVERY
    provider is capped/unavailable (caller then checkpoints + stops resumably)."""
    return chain.chat(system, user, temperature=temperature, max_tokens=max_tokens, retries=retries)


# Back-compat aliases — existing call sites (intake/analyze) get failover with no changes.
def groq_client():
    return llm_client()


def groq_chat(client, system, user, temperature=0.8, max_tokens=1024, retries=6):
    return llm_chat(client, system, user, temperature=temperature, max_tokens=max_tokens,
                    retries=retries)
