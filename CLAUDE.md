# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first
`instructions.md` is the product brain (agent role, memory rules, run types, style
DNA, fail-loud discipline) — **read it fully every session; do not modify without
asking.** `README.md` has the user-facing commands. This file is the code map.

## What this is
An on-demand, checkpointed clipping pipeline for **Account A ("flzsh")** production.
Scope is v1 Account A + intake only — **do not build Account B** (EDL plans, garnish,
VO). The user reviews/posts/submits manually; **never automate or suggest automating
posting.**

## Commands
```bash
# Scout→Clipper handoff (Stage -1): pick scout's #1 ranked campaign and write its
# brief + footage links into campaign_inputs/, then intake takes over automatically.
python scripts/pickcampaign.py                       # select scout's #1 (reads scout/campaigns.json)
python scripts/intake.py --from-pick                 # Stage 0 on the picked campaign
python scripts/run.py                                # download → index → select → captions → cut

# Or intake by hand (no scout):
python scripts/intake.py --brief <file> --links <file> --campaign "<name>"   # Stage 0
python scripts/run.py --resume   # continue after a crash (per-stage + per-chunk checkpoints)
python scripts/run.py --force    # re-run all stages
python scripts/selftest.py       # offline self-test (no ffmpeg → ffmpeg stages SKIP, rest run)
```
**Full chain the user runs:** `python scout.py` (in ../scout, ranks campaigns) →
`python scripts/pickcampaign.py` → `python scripts/intake.py --from-pick` →
`python scripts/run.py`.
Real runs need `GROQ_API_KEY` and ffmpeg/ffprobe. `CLIPPER_OFFLINE=1` forces
deterministic heuristics (no Groq) and skips whisper — **test only**.

## Architecture / data flow
Stages are plain modules with a `run(state)` entrypoint, orchestrated by `run.py`,
each writing its output to `campaign/` and checkpointing `state.json` before the next.
`scripts/common.py` is the shared core (paths, state, `fail()` fail-loud, `run_cmd`,
ffprobe, Groq client, UTF-8 stdout).

`pickcampaign.py` (Scout→Clipper handoff, Stage -1) → reads `../scout/campaigns.json`, ranks
it with scout's own composite (mirrors scout's report sort), selects the **#1 rankable**
campaign (skips disqualified + UNKNOWN-only), and writes `campaign_inputs/{brief.txt, links.txt,
pick.json}` from the campaign's scraped `rules_text` + `source_links`. **Fail-loud, strict #1:**
if the top pick has no footage links (nothing to clip) it STOPS and reports — it does NOT skip
to #2, and it does NOT scrape Whop (scout owns the browser). Live "search Whop by name" is
deferred: it prints the Whop search URL for the user to open instead of running unverified
automation. `--rank N` is a manual override. `intake.py --from-pick` (`apply_pick`) then reads
`pick.json` and runs Stage 0 on it — the wiring that lets intake take the scout pick instead of
hand-placed files (fail-loud if the pick or its files are missing).

`intake.py` (brief+links) → `campaign/{brief.md, rules.json, knowledge.md, manifest.json}`
+ `footage/ assets/ docs/ other/`
→ `index.py` → `campaign/moments.json` (whisper transcript + audio-spike moments)
→ `selectclips.py` → `campaign/selected.json`
→ `captions.py` → `campaign/captions.json`
→ `cut.py` → `drafts/NN_score_slug.mp4` + `drafts/manifest.json`.

**Intake is an analyst** (`intake.py` + `analyze.py`): it routes every file by type
(video/image/doc/other — nothing dropped), extracts text from every doc + link-shared
Google Docs, probes videos, harvests URLs and recurses ONE level, then LLM-extracts
structured rules via Groq (deterministic keyword rules are a FLOOR the LLM augments,
never removes). It writes `knowledge.md` (per-campaign digest) and ends with a coverage
report; unused/ambiguous items are flagged, never guessed.

`rules.json` (banned words + **banned topics** + required elements + spatial
constraints + …) is the contract the caption gauntlet and the cut-stage rules gate
consume. **Banned words/topics are killed before caption scoring** and re-checked in
`cut.py` (fail loud if one slips). Every stage also reads `knowledge.md`
(`common.load_knowledge()`) so campaign context is applied, not re-derived — the Groq
select/caption prompts include it. **`knowledge.md` is per-campaign and must never leak
into `memory/longterm.md`.**

## Conventions that matter
- **Fail loud, never guess.** Missing tool / ambiguous rule / no footage → `C.fail(...)`
  and stop. Intake flags ambiguities for the user rather than inventing rules.
- **Everything is resumable.** New work must checkpoint to `state.json` (and to disk)
  before the next step. `index.py` checkpoints per 5-min VOD chunk — preserve that.
- **Stage checkpoints are per-campaign.** `common.activate_campaign(state, name)` scopes
  `state["stages"]` to the active campaign (inactive ones stashed under
  `state["campaigns"][name]`); switching campaigns runs fresh WITHOUT `--force` and moves
  the prior campaign's `drafts/` into `drafts_archive/` (moved, never deleted). Called by
  `intake.py` (primary) and `run.py` (defensive, keyed off `rules.json`'s campaign) — a
  no-op on the same campaign, so re-runs stay resumable.
- **Offline mode** (`common.offline_mode()`) must keep working so the self-test runs
  without Groq/model downloads.
- **Campaign rules never leak into memory/longterm.md** — see instructions.md MEMORY.

## Non-obvious gotchas
- The select stage is `selectclips.py`, **not `select.py`** — a `select` module shadows
  Python's stdlib `select` (asyncio/httpx/subprocess on Linux) and breaks real runs.
- Captions are rendered as PNGs with Pillow and overlaid by ffmpeg (see
  `cut.render_caption_png`), deliberately avoiding ffmpeg `drawtext` font/escaping
  issues. `CLIPPER_FONT` overrides the bold TTF; `CLIPPER_EMOJI_FONT` the color-emoji TTF.
- **Selection hunts HIGHLIGHTS, not a fixed count (`selectclips.py`).** The Groq pass scores
  each moment 0-100 on highlight-worthiness — the funny / high-energy / chaotic / surprising
  peaks a person would actually clip, not only "physical events". Audio `intensity`/`peak` is
  the ENERGY signal (which moments reach Groq), but Groq must confirm the spike is genuinely
  good, not just loud. Two config knobs are the real limiters (not `clips_per_batch`, now a
  legacy soft target): `select_min_quality` (default 60) is the bar a moment must clear to
  ship, and `select_hard_cap` (default 50) is the **safety ceiling** so a runaway can't burn
  the Groq free tier overnight (each selected clip = one downstream caption Groq call). We take
  every moment that clears the bar up to the ceiling and **stop — never pad to a number**; if
  nothing clears it we ship the best available, capped conservatively.
- **Caption + subtitle case is Title Case (Like This), per user preference** —
  `captions.titlecase` is the single choke point, applied in `finalize_caption` (hook caption)
  AND `cut.render_subtitle_png` (burned subtitles). It capitalizes each word's first letter and
  leaves the rest untouched, so contractions and emoji survive ("don't"→"Don't", masked
  "b**"→"B**"). This OVERRODE the old lowercase default. `emoji_in_caption` (config, default true) keeps emoji;
  cut renders them with a color-emoji font and DROPS any glyph the font can't draw so a
  tofu box never ships. Tofu-detection uses fontTools' live cmap if installed, else a
  curated allowlist (`cut._ALLOWED_EMOJI_CP`) — because Segoe UI Emoji draws unknown code
  points as a visible box that a pixel probe can't distinguish from a real glyph.
- **Burned subtitles** (`subtitles_enabled`, config default true; auto-off in offline
  mode — needs whisper). Karaoke-style short phrase chunks, rendered as Pillow PNGs and
  overlaid via `enable='between(t,…)'` (same PNG approach as captions). Timing is
  edit-proof: cut RE-TRANSCRIBES the FINAL 30s clip (`cut.transcribe_clip_words`) rather
  than remapping source timestamps, because cold-open reorder + dead-air trims desync the
  source. **Campaign-banned words are masked before burn** (`cut._mask_banned_word`) and
  re-checked per chunk (fail loud if one slips) — same gauntlet discipline as captions.
  Subtitles are pushed DOWN into the lower letterbox/black band, OFF the footage frame
  (`SUBTITLE_CENTER_Y`=1440 ≈ 75% down, `_BOX_W`=640 centered): below the footage (which in
  blur_fill occupies ~y595-1325), clear of the top hook plate, the bottom-right watermark, and
  the bottom UI / right action-rail notch.
- **Merge discipline (select):** `merge_gap_seconds` (7) + `merge_max_span_seconds` (60)
  keep a merged moment one real beat; over-length moments are NOT clamped from the start —
  `cut.clip_bounds` centers the clip window on the moment's peak. Every speech moment now
  carries an RMS energy `peak` (`index._energy_peak`), so cold-open can fire on speech too.
- `common.py` forces UTF-8 on stdout/stderr — Windows consoles are cp1252 and crash on
  caption emoji / status glyphs otherwise.
- Grid/list card scraping lives in the sibling `scout` project. The two are now joined by the
  handoff: scout ranks + writes `scout/campaigns.json`; `pickcampaign.py` reads it. Scout still
  owns ALL Whop browser automation — the clipper never drives Whop (see pickcampaign's deferred
  name-search).
