"""Orchestrator — runs the pipeline stages in order, checkpointing after each.

Stages: download -> index -> select -> captions -> cut. Stage 0 (intake) is run
separately first (it needs the brief + links); run.py verifies its outputs exist.

Resumable: each stage marks itself done in state.json. A normal run skips stages
already done (so it naturally continues where a crash left off); --force re-runs
everything. Long stages (index) also checkpoint internally per VOD chunk.

    python run.py                        # run/continue the pipeline
    python run.py --resume               # same, explicit
    python run.py --force                # re-run all stages from scratch
    python run.py --clips-per-batch 6 --clip-min 12 --clip-max 30 --force
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import download as DL
import index as index_stage
import selectclips as select_stage
import captions as captions_stage
import cut as cut_stage

# ---- CONFIG (defaults; all tunable per run via flags) ----
DEFAULT_CONFIG = {
    "clips_per_batch": 25,          # legacy soft target; real limiters are the two below
    # Intake footage cap: download VODs one at a time and stop starting new ones once the
    # cumulative footage duration reaches this many hours (the crossing VOD is kept in full).
    # Bounds transcription/index time on 12h+ campaigns. Consumed by intake.py.
    "footage_cap_hours": 10,
    # Highlight selection (select stage). We take the genuinely-good peaks, not a fixed
    # count: select_min_quality is the 0-100 Groq bar a moment must clear to ship, and
    # select_hard_cap is the SAFETY CEILING on clips per run (each clip = one caption Groq
    # call downstream, so this stops a runaway from burning the free tier overnight).
    "select_hard_cap": 50,
    "select_min_quality": 60,
    # LOW dead-floor: if even the BEST moment scores below this, the campaign is genuinely dead
    # (the score is text-blind, so keep this forgiving) → stop + auto-advance. 40+ ships.
    "select_dead_floor": 40,
    # LLM failover chain order (Unit: never dead-end on one provider's daily cap). All free-tier;
    # keys come ONLY from env (GROQ_API_KEY / GEMINI_API_KEY / CEREBRAS_API_KEY). Drop a name to
    # disable it, or reorder. A provider with a missing key/library is skipped automatically.
    "llm_providers": ["groq", "gemini", "cerebras"],
    # Footage-language gate (index stage). AFTER transcription (local + free) but BEFORE
    # select/captions (LLM token spend), bail if the dominant FOOTAGE language isn't
    # gate_language — catches campaigns whose brief/name is English but whose footage is not
    # (slipped scout's derank), before a single token is spent. Fails OPEN below the confidence
    # share; bypass a run with --allow-any-language.
    "gate_language": "en",
    "gate_language_min_prob": 0.6,  # min dominant-language char-share to trust the gate
    "allow_any_language": False,    # set per-run by --allow-any-language (never sticky)
    # AUTO-ADVANCE WALK (Unit 2c). With --auto-advance, walk DOWN the ranked list — pick →
    # intake → run; if a campaign yields nothing usable (no footage / dead / dead-floor best<40 /
    # language gate / failed intake) advance to the NEXT ranked campaign — until clips are
    # produced, `auto_advance_max` campaigns are tried, or the list is exhausted. Anti-throttle:
    # each attempt hits YouTube, so SPACE the attempts out (walk_spacing_seconds), and if a
    # YouTube bot-check/throttle is detected back off harder (walk_throttle_backoff_seconds)
    # instead of hammering. Cookies (cookies.txt / --cookies-from-browser) stay applied throughout.
    "walk_spacing_seconds": 20,           # gentle delay between auto-advance attempts
    "walk_throttle_backoff_seconds": 300,  # longer back-off once a bot-check/throttle is seen
    "layout": "auto",              # vertical fill: "auto" (TRACK a single subject, else blur_fill),
                                   # "track" (force face/person-tracked 9:16 crop), "blur_fill"
                                   # (whole frame on a blurred bg), or "crop_fill" (COVER+center-crop)
    "blur_fg_zoom": 1.2,           # blur_fill foreground zoom: 1.0 = pure no-crop
                                   # letterbox; >1 trims only far L/R edges to fill more
                                   # vertical space (never crops top/bottom or subjects)
    # TRACK mode (OpenShorts-style face/person reframe; needs mediapipe+ultralytics+opencv).
    "track_subject_scale": 0.60,   # ZOOM knob: fraction of the OUTPUT HEIGHT the subject fills.
                                   # ~0.6 = head+shoulders+room (looser); LOWER = looser/more room,
                                   # HIGHER = tighter head crop. THIS controls zoom (not track_safe_zone).
    "track_safe_zone": 0.35,       # PAN trigger only (NOT zoom): center dead-zone as a fraction of
                                   # the crop width — the crop HOLDS while the subject stays inside
                                   # it and only pans when they leave it (anti-jitter)
    "track_smooth": 0.12,          # pan easing when the subject leaves the safe zone (0-1; lower=slower)
    "track_max_pan": 12.0,         # max crop pan speed in source px/frame (caps fast whip-pans)
    "track_detect_every": 3,       # run detection every Nth frame (reuse between) — ~10fps at 30fps
    "track_min_single_frac": 0.5,  # auto: min share of sampled frames with exactly ONE subject → TRACK
    "track_max_multi_frac": 0.3,   # auto: above this share of multi-subject frames → GENERAL (group)
    "track_max_none_frac": 0.5,    # auto: above this share of subject-less frames → GENERAL (landscape)
    "clip_min_seconds": 15,        # allow tight action clips (don't over-pad)
    "clip_max_seconds": 30,        # hard cap — a tight 20-25s clip beats a padded 45s one
    "min_separation_seconds": 60,  # min gap between two selected moments (same source)
    "merge_gap_seconds": 7,        # merge moments closer than this into one (was 15 —
                                   # chained non-stop commentary into 400-550s blobs)
    "merge_max_span_seconds": 60,  # hard cap on a merged moment's span (stop merging past)
    "story_pre_seconds": 4,        # start AT the action — minimal setup lead-in (max)
    "story_post_seconds": 4,       # end near the payoff — minimal resolution tail (max)
    # COLD-OPEN (conditional): only tease a clip whose audio window has ONE genuine sharp peak.
    "coldopen_peak_range_min": 3.0,  # min "triangle range" (σ of window peak above its median,
                                   # in the source's own RMS std units) to treat a peak as sharp;
                                   # below this the clip is flat-high and plays straight
    "coldopen_min_separation": 8.0,  # min OUTPUT seconds between the teaser and the payoff's
                                   # natural arrival in the body — else it reads as an instant repeat.
                                   # If a clip's peak is too early to leave an 8s gap, plan_cold_open
                                   # rejects it and the clip plays as a straight cut.
    "output_fps": 30,              # cut renders at this constant frame rate (CFR) — fixes the
                                   # VFR concat-seam stutter on cold-open→setup transitions
    "emoji_in_caption": True,      # flzsh DNA: keep emoji as caption punctuation
    "subtitles_enabled": True,     # burn word-level karaoke spoken-word subtitles (ASS/libass)
    # Karaoke subtitle look (lower-center, ASS/libass). The active (currently-spoken) word
    # pops in the accent colour + upscale, then reverts as the next word speaks. Accent is a
    # per-account/config value in ASS &HBBGGRR order (reversed hex) — default punchy yellow.
    # NOTE: per-clip VARIETY (accent colour, active-word emphasis mode, hook position) now comes
    # from the named STYLE-SET in cut.py (DEFAULT_STYLE_SET, resolve_clip_style), rotated
    # deterministically per clip. Override the whole set per account/category via config
    # "style_set": {...}. subtitle_accent_color below is only the FALLBACK when no style-set applies.
    "subtitle_accent_color": "&H00FFFF&",  # active-word colour (ASS &HBBGGRR); yellow
    "subtitle_active_scale": 110,  # % upscale applied to the active word (the "pop")
    "subtitle_ass_fontsize": 54,   # subtitle font size at 1080x1920 output res
    "subtitle_pause_gap": 0.1,    # sec of silence between words that starts a NEW karaoke line
                                   # (breaks on the speaker's natural pauses, even mid-cap)
    "subtitle_max_words": 3,       # cap within a continuous (pause-free) run of speech
    "subtitle_censor_profanity": True,  # karaoke swears get a light vowel-censor ("sh*t");
                                   # campaign banned words (bet/gamble) stay FULLY masked
    "subtitle_band_margin": 28,    # px below the footage rectangle to pin the karaoke line
                                   # (blur_fill) so it sits in the lower black band, off the video
    # HOOK (top caption) plate/outline PRESET (A|B|C|D — see cut.HOOK_STYLES). HOOK-ONLY; the
    # karaoke keeps its own per-clip colour/box variety. A=no plate + thick outline, B=no plate +
    # thin outline, C=thin semi-transparent plate, D=thick plate. LOCKED to A.
    "hook_style": "A",
    "hook_max_font_size": 64,      # cap on the hook font size so SHORT hooks don't balloon (px @
                                   # 1080-wide; ~clipA reference). Long hooks still shrink to fit.
    # Caption/subtitle readability (understated but always legible on any background). The dark
    # plate guarantees white-text contrast; keep the outline thin so it stays clean.
    "plate_opacity": 120,          # dark plate alpha behind caption+subtitles (0-255; 0 = off)
    "caption_outline_width": 2,    # text outline stroke in px for both (0 = no outline)
    "watermark_scale": 0.18,       # fraction of 1080px width
    "watermark_margin": 40,        # px from edges
    "watermark_file": None,        # exact/substring name in assets/; None = auto-pick
    "max_source_height": 720,      # cap for downloads AND the render: cut downscales the
                                   # source to this height before the blur-fill graph (4K
                                   # frames through split+scale+overlay OOM), and download.py
                                   # caps the yt-dlp format at this height (never pull 4K).
    "ffmpeg_threads": 2,           # fewer threads = lower peak RAM in the cut stage
}

_COOKIES = None                   # browser for cookies during the download stage
_ORIGINAL = False                 # force raw Drive files instead of preview streams


def require_intake():
    for p in (C.RULES_JSON, C.CAMPAIGN_MANIFEST, C.BRIEF_MD):
        if not p.exists():
            C.fail(f"{p.name} missing — run Stage 0 first:\n"
                   "  python scripts/intake.py --brief <file> --links <file>")


def _stop_daily_cap(stage):
    """Groq DAILY cap hit mid-stage (Unit 2b). The stage has already checkpointed its finished
    Groq work (select_partial / captions_partial), so we just report N-of-M and stop RESUMABLY —
    a `python run.py --resume` after the quota resets continues from the checkpoint, re-spending
    NO completed calls."""
    if stage == "captions":
        done = len((C.load_json(C.CAPTIONS_PARTIAL) or {}).get("clips", []))
        total = len((C.load_json(C.SELECTED_JSON) or {}).get("selected", []))
        C.stop_resumable(f"Groq daily cap hit during captions — {done} of {total} clip(s) done. "
                         f"Re-run with --resume after the cap resets.")
    elif stage == "select":
        done = len((C.load_json(C.SELECT_PARTIAL) or {}).get("scored", []))
        C.stop_resumable(f"Groq daily cap hit during select — {done} moment(s) scored so far. "
                         f"Re-run with --resume after the cap resets (already-scored batches skip).")
    C.stop_resumable(f"Groq daily cap hit during {stage}. Re-run with --resume after reset.")


def cut_only(state):
    """Run ONLY the cut stage on the already-selected/captioned clips — ZERO Groq calls,
    no download/index/select/captions. For iterating on the render (e.g. the ASS karaoke)
    without re-picking or burning the Groq daily cap.

    cut consumes campaign/captions.json (the hook captions already generated for the picks in
    selected.json). We touch ONLY the source files those clips reference — any other campaign's
    footage sitting in campaign/footage/ is ignored (cut never scans the folder; that's index)."""
    caps = C.load_json(C.CAPTIONS_JSON)
    if not caps or not caps.get("clips"):
        C.fail("--cut-only needs campaign/captions.json (the captions already generated for "
               "the selected clips) but it's missing/empty. It's produced by the captions "
               "stage; run the full pipeline once (or just the captions stage) first — "
               "--cut-only never calls Groq itself.")
    if not C.SELECTED_JSON.exists():
        C.warn("campaign/selected.json not found — cutting from captions.json alone (the picks "
               "it was built from are gone, but the captions carry everything cut needs).")

    # Only these source files will be opened — list them and confirm they exist, so a mixed
    # footage/ folder (multiple campaigns) can't pull in the wrong files.
    sources = sorted({c["source"] for c in caps["clips"]})
    C.log(f"cut-only: {len(caps['clips'])} clip(s) referencing {len(sources)} source file(s) "
          f"(every other file in campaign/footage/ is ignored):")
    missing = []
    for s in sources:
        p = C.ROOT / s
        C.log(f"    {'[OK]     ' if p.exists() else '[MISSING]'} {s}")
        if not p.exists():
            missing.append(s)
    if missing:
        C.fail(f"--cut-only: {len(missing)} referenced source file(s) not found: {missing}. "
               "Restore them (cut only touches the sources named in captions.json).")

    # Always re-run cut (it may be marked done from a prior run); leave every other stage alone.
    state.get("stages", {}).pop("cut", None)
    C.save_state(state)
    C.log("== stage: cut ONLY (download, index, select, captions all skipped — no Groq) ==")
    cut_stage.run(state)
    C.log("cut-only complete — drafts in drafts/ (best first), see drafts/manifest.json.")


def stage_download(state):
    manifest = C.load_json(C.CAMPAIGN_MANIFEST)
    if not manifest:
        C.fail("campaign/manifest.json missing — run intake.py first.")
    cfg = state.get("config", {})
    repaired = DL.ensure_downloaded(
        manifest, cookies_from_browser=_COOKIES,
        max_source_height=cfg.get("max_source_height", 720), original=_ORIGINAL)
    C.mark_stage(state, "download", repaired=repaired)
    C.log(f"download stage: {repaired} file(s) re-fetched, rest present.")


STAGES = [
    ("download", stage_download),
    ("index", index_stage.run),
    ("select", select_stage.run),
    ("captions", captions_stage.run),
    ("cut", cut_stage.run),
]


def offer_cleanup(args):
    footage = [p for p in C.FOOTAGE.glob("*") if p.is_file()]
    if not footage or args.no_cleanup:
        return
    do = args.cleanup
    if not do:
        try:
            interactive = sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            interactive = False
        if not interactive:      # background / piped run — never block on input()
            C.log("Source cleanup available: pass --cleanup to delete raw footage "
                  "(transcripts + moments.json are always kept for re-cuts).")
            return
        try:
            ans = input(f"\nDelete {len(footage)} raw footage file(s) from campaign/footage/? "
                        "Transcripts + moments.json are kept so re-cutting different moments "
                        "needs no re-download/re-transcribe. [y/N] ")
        except EOFError:
            return
        do = ans.strip().lower() in ("y", "yes")
    if do:
        for p in footage:
            try:
                p.unlink()
            except Exception as e:
                C.warn(f"could not delete {p.name}: {e}")
        C.log("Raw footage deleted; transcripts + moments.json retained.")


def _prepare_state(args):
    """Load state, scope it to the campaign on disk (rules.json), merge config + CLI overrides,
    and honor --force. Returns the ready-to-run state (config saved). Re-run each pass of the
    auto-advance loop so a newly-picked campaign is activated correctly."""
    state = C.load_state()
    # Defensive: if the campaign on disk (rules.json) differs from the state's active one, scope
    # stages to it so we never skip stages a PRIOR campaign left 'done'. This is ALSO how an
    # auto-advanced campaign gets its own fresh (empty) stage set.
    rules_campaign = (C.load_json(C.RULES_JSON) or {}).get("campaign")
    C.activate_campaign(state, rules_campaign)
    cfg = {**DEFAULT_CONFIG, **state.get("config", {})}
    if args.clips_per_batch:
        cfg["clips_per_batch"] = args.clips_per_batch
    if args.layout:
        cfg["layout"] = args.layout
    if args.clip_min:
        cfg["clip_min_seconds"] = args.clip_min
    if args.clip_max:
        cfg["clip_max_seconds"] = args.clip_max
    if args.watermark_file:
        cfg["watermark_file"] = args.watermark_file
    if args.max_source_height:
        cfg["max_source_height"] = args.max_source_height
    # Set from the flag EVERY run so it's a per-run override, never sticky in state.json (a
    # once-passed --allow-any-language must not silently disable the gate on later runs).
    cfg["allow_any_language"] = bool(args.allow_any_language)
    state["config"] = cfg
    if args.force:
        for name, _ in STAGES:
            state.get("stages", {}).pop(name, None)
    C.save_state(state)
    return state


def _run_stages(state, args):
    """Run the pipeline stages in order. A GroqDailyCapError stops resumably; a NothingUsable
    propagates to the caller (which may auto-advance)."""
    for name, fn in STAGES:
        if C.stage_done(state, name) and not args.force:
            C.log(f"skip {name} (already done)")
            continue
        C.log(f"== stage: {name} ==")
        try:
            fn(state)
        except C.GroqDailyCapError:
            _stop_daily_cap(name)     # checkpoint already saved by the stage — resumable stop


# --- AUTO-ADVANCE WALK ---------------------------------------------------------------
# A campaign that yields nothing usable (no footage / dead / dead-floor / language gate / failed
# intake) is not a dead-end under --auto-advance: we walk DOWN the ranked list until one produces
# clips, `auto_advance_max` are tried, or the list is exhausted. YouTube is hit on every attempt,
# so attempts are SPACED (anti-throttle) and a detected bot-check backs off harder.
_THROTTLE_MARKERS = (
    "sign in to confirm you're not a bot", "confirm you're not a bot", "not a bot",
    "http error 429", "429: too many requests", "too many requests", "429 too many",
    "temporarily blocked", "rate-limited", "rate limited", "verify you're human",
    "unusual traffic", "this content isn't available",
)


def _looks_like_throttle(text):
    """True when yt-dlp/YouTube output looks like a bot-check / rate-limit (so the walk backs off
    instead of hammering the next request)."""
    low = (text or "").lower()
    return any(m in low for m in _THROTTLE_MARKERS)


def _reason_tag(msg):
    """Short label for a NothingUsable reason, for the overnight walk log."""
    low = (msg or "").lower()
    if "language" in low:
        return "language-gate"
    if "dead-floor" in low or "dead floor" in low:
        return "dead-floor (best < 40)"
    if "zero usable" in low or "no usable" in low:
        return "no usable sources"
    if "no footage" in low:
        return "no footage"
    if "no candidate moments" in low or "no moments" in low:
        return "no moments"
    return "nothing usable"


def _walk_sleep(spacing, throttle_backoff, throttled):
    """Anti-throttle pause before the next attempt. Normal gap = `spacing`; a detected bot-check
    escalates to `throttle_backoff`. Cookies stay applied by the download/intake stages."""
    import time
    delay = max(0.0, float(throttle_backoff if throttled else spacing))
    if delay <= 0:
        return
    if throttled:
        C.warn(f"  anti-throttle: a YouTube bot-check/throttle was detected — backing off "
               f"{delay:.0f}s before the next attempt (cookies still applied).")
    else:
        C.log(f"  spacing {delay:.0f}s before the next campaign (gentle on YouTube)…")
    time.sleep(delay)


def _tee(cmd):
    """Run a subprocess, STREAM its output live (so the overnight log shows the whole walk) AND
    capture it, so the text can be scanned for a bot-check/throttle. Returns (returncode, text)."""
    import subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    chunks = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    proc.wait()
    return proc.returncode, "".join(chunks)


def _pick_and_intake_next(excluded, args):
    """Pick the NEXT ranked clippable campaign (excluding the ones already tried) and run intake
    on it. Returns (status, captured_text): 'ready' (a campaign is intaken + ready to run),
    'intake_failed' (picked but intake errored — caller advances again), or 'exhausted' (no
    further clippable campaign). The just-tried campaign's id is added to `excluded` so it's
    never re-picked."""
    scripts = os.path.dirname(os.path.abspath(__file__))
    pick = C.load_json(C.ROOT / "campaign_inputs" / "pick.json") or {}
    tried_id = pick.get("scout_id")
    if tried_id and tried_id not in excluded:
        excluded.append(tried_id)

    pc = [sys.executable, os.path.join(scripts, "pickcampaign.py")]
    if pick.get("scout_json"):
        pc += ["--scout-json", pick["scout_json"]]
    if pick.get("rank_mode") == "streamer_only":
        pc.append("--streamer-only")
    for eid in excluded:
        if eid:
            pc += ["--exclude-id", str(eid)]
    rc, txt = _tee(pc)
    if rc != 0:
        return "exhausted", txt

    rc2, txt2 = _tee([sys.executable, os.path.join(scripts, "intake.py"), "--from-pick"])
    text = txt + txt2
    if rc2 != 0:
        # Exclude the just-picked campaign too, so the next advance skips it.
        newpick = C.load_json(C.ROOT / "campaign_inputs" / "pick.json") or {}
        nid = newpick.get("scout_id")
        if nid and nid not in excluded:
            excluded.append(nid)
        return "intake_failed", text
    newpick = C.load_json(C.ROOT / "campaign_inputs" / "pick.json") or {}
    C.log(f"auto-advance: now on {newpick.get('campaign')!r} — running the pipeline for it.")
    return "ready", text


def walk(process_fn, advance_fn, walk_depth):
    """Core auto-advance walk (pure, so it's unit-testable without downloads).

    process_fn(attempt) runs the pipeline on the CURRENT campaign — returns on success, raises
    C.NothingUsable(reason) on a dead/no-footage/language failure. advance_fn(attempt) prepares
    the NEXT campaign (spacing → pick → intake) and returns one of 'ready' / 'intake_failed' /
    'exhausted'. Walks up to `walk_depth` ATTEMPTS (a pipeline run OR a failed intake each count
    as one), stopping on the first success, the depth cap, or an exhausted list.

    Returns (ok, attempts, reasons)."""
    reasons = []
    attempt = 0
    need_advance = False
    while attempt < walk_depth:
        if need_advance:
            status = advance_fn(attempt)
            if status == "exhausted":
                reasons.append("exhausted (no further ranked campaign)")
                return False, attempt, reasons
            if status == "intake_failed":
                attempt += 1
                reasons.append("intake failed")
                C.warn(f"  [walk {attempt}/{walk_depth}] intake failed — advancing.")
                continue                      # need_advance stays True → pick the next one
            need_advance = False              # 'ready' → fall through and run the pipeline
        attempt += 1
        try:
            process_fn(attempt)
            return True, attempt, reasons
        except C.NothingUsable as e:
            tag = _reason_tag(str(e))
            reasons.append(tag)
            C.warn(f"  [walk {attempt}/{walk_depth}] FAILED: {tag}.")
            need_advance = True
    return False, attempt, reasons


def main():
    ap = argparse.ArgumentParser(description="Clipper pipeline orchestrator.")
    ap.add_argument("--resume", action="store_true", help="continue from last completed stage (default behavior)")
    ap.add_argument("--force", action="store_true", help="re-run all stages from scratch")
    ap.add_argument("--cut-only", action="store_true", dest="cut_only",
                    help="run ONLY the cut stage on the existing selected/captioned clips "
                         "(skips download, index, select, captions — ZERO Groq calls)")
    ap.add_argument("--clips-per-batch", type=int)
    ap.add_argument("--layout", choices=["auto", "track", "blur_fill", "crop_fill"],
                    help="vertical layout (default auto: face/person-tracked crop for a single "
                         "subject, else blur_fill). 'track' forces the crop; 'blur_fill' the bg.")
    ap.add_argument("--clip-min", type=int, dest="clip_min")
    ap.add_argument("--clip-max", type=int, dest="clip_max")
    ap.add_argument("--watermark-file", help="watermark filename in assets/ (exact or substring)")
    ap.add_argument("--cookies-from-browser", help="browser for cookies when re-fetching gated VODs")
    ap.add_argument("--max-source-height", type=int, dest="max_source_height",
                    help="cap for Drive transcoded preview streams in px (default 720)")
    ap.add_argument("--original", action="store_true",
                    help="force raw original Drive files instead of preview streams")
    ap.add_argument("--allow-any-language", action="store_true", dest="allow_any_language",
                    help="bypass the footage-language gate — process non-English footage anyway "
                         "(default: STOP before select/captions if footage isn't English).")
    ap.add_argument("--cleanup", action="store_true", help="delete raw footage after a successful batch")
    ap.add_argument("--no-cleanup", action="store_true", help="never prompt for footage cleanup")
    ap.add_argument("--auto-advance", action="store_true", dest="auto_advance",
                    help="if the picked campaign produces NOTHING usable (no footage / zero "
                         "indexable sources / dead-floor / language gate), WALK DOWN the ranked "
                         "list — re-pick + intake + run the next campaign — until one produces "
                         "clips, --auto-advance-max are tried, or the list is exhausted. OFF by "
                         "default — it downloads more campaigns, so it never happens unless asked.")
    ap.add_argument("--auto-advance-max", type=int, default=10, dest="auto_advance_max",
                    help="max campaigns to walk through before giving up (default 10). Each "
                         "attempt hits YouTube, so attempts are SPACED (see --walk-spacing).")
    ap.add_argument("--walk-spacing", type=int, default=None, dest="walk_spacing",
                    help="seconds to wait between auto-advance attempts (anti-throttle; default "
                         "from config walk_spacing_seconds=20). A detected bot-check backs off "
                         "longer (walk_throttle_backoff_seconds).")
    args = ap.parse_args()

    global _COOKIES, _ORIGINAL
    _COOKIES = args.cookies_from_browser
    _ORIGINAL = args.original

    C.ensure_dirs()

    if args.cut_only:
        require_intake()
        state = _prepare_state(args)
        C.log(f"config: {state['config']}")
        cut_only(state)
        return

    # Default (no --auto-advance): run the pipeline ONCE and dead-end LOUD on NothingUsable —
    # exactly as before.
    if not args.auto_advance:
        require_intake()
        state = _prepare_state(args)
        C.log(f"config: {state['config']}")
        try:
            _run_stages(state, args)
        except C.NothingUsable as e:
            C.fail(str(e))
        C.log("pipeline complete — drafts in drafts/ (best first), see drafts/manifest.json.")
        offer_cleanup(args)
        return

    # AUTO-ADVANCE WALK (Unit 2c): walk DOWN the ranked list until a campaign produces clips,
    # --auto-advance-max are tried, or the list is exhausted. Spacing between attempts (+ a
    # longer back-off on a detected bot-check) keeps 10 attempts from throttling YouTube.
    walk_depth = max(1, args.auto_advance_max)
    excluded = []
    throttled = {"v": False}

    def process_fn(attempt):
        require_intake()                       # re-checked each pass (a new campaign after advance)
        state = _prepare_state(args)
        campaign = (C.load_json(C.RULES_JSON) or {}).get("campaign")
        C.log("=" * 70)
        C.log(f"WALK attempt {attempt}/{walk_depth} — campaign {campaign!r}")
        C.log("=" * 70)
        C.log(f"config: {state['config']}")
        _run_stages(state, args)               # raises C.NothingUsable on a dead campaign

    def advance_fn(attempt):
        cfg = {**DEFAULT_CONFIG}
        spacing = args.walk_spacing if args.walk_spacing is not None else cfg["walk_spacing_seconds"]
        _walk_sleep(spacing, cfg["walk_throttle_backoff_seconds"], throttled["v"])
        status, text = _pick_and_intake_next(excluded, args)
        throttled["v"] = _looks_like_throttle(text)   # back off harder next time if throttled
        return status

    ok, attempts, reasons = walk(process_fn, advance_fn, walk_depth)
    if not ok:
        C.fail(f"auto-advance walk stopped after {attempts} attempt(s) without a clippable "
               f"campaign. Reasons: {reasons}. Excluded ids: {excluded}.")

    C.log(f"pipeline complete after {attempts} walk attempt(s) — drafts in drafts/ "
          f"(best first), see drafts/manifest.json.")
    offer_cleanup(args)


if __name__ == "__main__":
    main()
