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
    # Highlight selection (select stage). We take the genuinely-good peaks, not a fixed
    # count: select_min_quality is the 0-100 Groq bar a moment must clear to ship, and
    # select_hard_cap is the SAFETY CEILING on clips per run (each clip = one caption Groq
    # call downstream, so this stops a runaway from burning the free tier overnight).
    "select_hard_cap": 50,
    "select_min_quality": 60,
    "layout": "blur_fill",         # vertical fill: "blur_fill" (whole frame, blurred bg)
                                   # or "crop_fill" (COVER + center-crop, crops edges)
    "blur_fg_zoom": 1.2,           # blur_fill foreground zoom: 1.0 = pure no-crop
                                   # letterbox; >1 trims only far L/R edges to fill more
                                   # vertical space (never crops top/bottom or subjects)
    "clip_min_seconds": 15,        # allow tight action clips (don't over-pad)
    "clip_max_seconds": 30,        # hard cap — a tight 20-25s clip beats a padded 45s one
    "min_separation_seconds": 60,  # min gap between two selected moments (same source)
    "merge_gap_seconds": 7,        # merge moments closer than this into one (was 15 —
                                   # chained non-stop commentary into 400-550s blobs)
    "merge_max_span_seconds": 60,  # hard cap on a merged moment's span (stop merging past)
    "story_pre_seconds": 4,        # start AT the action — minimal setup lead-in (max)
    "story_post_seconds": 4,       # end near the payoff — minimal resolution tail (max)
    "emoji_in_caption": True,      # flzsh DNA: keep emoji as caption punctuation
    "subtitles_enabled": True,     # burn synced karaoke-style spoken-word subtitles
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


def main():
    ap = argparse.ArgumentParser(description="Clipper pipeline orchestrator.")
    ap.add_argument("--resume", action="store_true", help="continue from last completed stage (default behavior)")
    ap.add_argument("--force", action="store_true", help="re-run all stages from scratch")
    ap.add_argument("--clips-per-batch", type=int)
    ap.add_argument("--layout", choices=["blur_fill", "crop_fill"],
                    help="vertical fill mode (default blur_fill: whole frame on a blurred bg)")
    ap.add_argument("--clip-min", type=int, dest="clip_min")
    ap.add_argument("--clip-max", type=int, dest="clip_max")
    ap.add_argument("--watermark-file", help="watermark filename in assets/ (exact or substring)")
    ap.add_argument("--cookies-from-browser", help="browser for cookies when re-fetching gated VODs")
    ap.add_argument("--max-source-height", type=int, dest="max_source_height",
                    help="cap for Drive transcoded preview streams in px (default 720)")
    ap.add_argument("--original", action="store_true",
                    help="force raw original Drive files instead of preview streams")
    ap.add_argument("--cleanup", action="store_true", help="delete raw footage after a successful batch")
    ap.add_argument("--no-cleanup", action="store_true", help="never prompt for footage cleanup")
    args = ap.parse_args()

    global _COOKIES, _ORIGINAL
    _COOKIES = args.cookies_from_browser
    _ORIGINAL = args.original

    C.ensure_dirs()
    require_intake()

    state = C.load_state()
    # Defensive: if the campaign on disk (rules.json) differs from the state's active one,
    # scope stages to it so we never skip stages left 'done' by a PRIOR campaign (that made
    # run.py think everything was already built and mixed old drafts in). No-op in the
    # normal flow where intake already switched.
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
    state["config"] = cfg
    C.save_state(state)

    if args.force:
        for name, _ in STAGES:
            state.get("stages", {}).pop(name, None)
        C.save_state(state)

    C.log(f"config: {cfg}")
    for name, fn in STAGES:
        if C.stage_done(state, name) and not args.force:
            C.log(f"skip {name} (already done)")
            continue
        C.log(f"== stage: {name} ==")
        fn(state)

    C.log("pipeline complete — drafts in drafts/ (best first), see drafts/manifest.json.")
    offer_cleanup(args)


if __name__ == "__main__":
    main()
