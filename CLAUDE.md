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
python scripts/intake.py --brief <file> --links <file> --campaign "<name>"   # Stage 0, run first
python scripts/run.py            # download → index → select → captions → cut
python scripts/run.py --resume   # continue after a crash (per-stage + per-chunk checkpoints)
python scripts/run.py --force    # re-run all stages
python scripts/selftest.py       # offline self-test (no ffmpeg → ffmpeg stages SKIP, rest run)
```
Real runs need `GROQ_API_KEY` and ffmpeg/ffprobe. `CLIPPER_OFFLINE=1` forces
deterministic heuristics (no Groq) and skips whisper — **test only**.

## Architecture / data flow
Stages are plain modules with a `run(state)` entrypoint, orchestrated by `run.py`,
each writing its output to `campaign/` and checkpointing `state.json` before the next.
`scripts/common.py` is the shared core (paths, state, `fail()` fail-loud, `run_cmd`,
ffprobe, Groq client, UTF-8 stdout).

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
- **Offline mode** (`common.offline_mode()`) must keep working so the self-test runs
  without Groq/model downloads.
- **Campaign rules never leak into memory/longterm.md** — see instructions.md MEMORY.

## Non-obvious gotchas
- The select stage is `selectclips.py`, **not `select.py`** — a `select` module shadows
  Python's stdlib `select` (asyncio/httpx/subprocess on Linux) and breaks real runs.
- Captions are rendered as PNGs with Pillow and overlaid by ffmpeg (see
  `cut.render_caption_png`), deliberately avoiding ffmpeg `drawtext` font/escaping
  issues. `CLIPPER_FONT` overrides the bold TTF.
- `common.py` forces UTF-8 on stdout/stderr — Windows consoles are cp1252 and crash on
  caption emoji / status glyphs otherwise.
- Grid/list card work lives in a different project (`scout`); this repo is unrelated.
