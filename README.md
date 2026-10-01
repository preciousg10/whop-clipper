# Clipper

On-demand short-form clipping pipeline for **Account A ("flzsh")** production. You
give it a campaign brief + footage links; it downloads everything, finds the strong
moments, writes flzsh-style captions (rules-checked), and renders finished vertical
clips with the watermark burned in. **You review, post, and submit — never the tool.**

> Companion project: [**whop-scout**](https://github.com/preciousg10/whop-scout) researches and
> ranks which campaigns are worth clipping for; this repo (**whop-clipper**) produces the clips.

The agent's brain is `instructions.md` (read it). Groq does the volume work
(moment scanning, caption first-drafts); final taste calls happen when you run the
agent in Claude Code with `instructions.md` loaded. These scripts prepare and execute.

## Requirements

- Python 3.12 (3.11 works) and **ffmpeg + ffprobe on PATH**.
- `GROQ_API_KEY` in the environment (for real runs).

```bash
./setup.sh                      # venv + deps, checks ffmpeg
export GROQ_API_KEY=...          # required for real runs
```

Manual install: `pip install -r requirements.txt` and install ffmpeg
(`brew install ffmpeg` / `apt-get install ffmpeg` / `winget install Gyan.FFmpeg`).

## Commands

### 1. Intake (Stage 0 — run first, once per campaign)
Parses the brief, extracts rules, downloads all footage + assets.

```bash
python scripts/intake.py --campaign "WTF Leagues" \
    --brief brief.txt \
    --links links.txt          # one Drive folder / VOD URL / file / local path per line
# or inline:
python scripts/intake.py --campaign "WTF Leagues" \
    --brief-text "..." --link <drive-folder-url> --link <kick-vod-url> --link watermark.png

# gated VODs (Kick often 403s unauthenticated): pull cookies from your browser
python scripts/intake.py --campaign "WTF Leagues" --brief brief.txt --links links.txt \
    --cookies-from-browser chrome        # or edge / firefox
```
Intake is a **campaign analyst**, not just a downloader. It:
- **Routes every file by type** — video → `footage/`, image → `assets/`, docs
  (pdf/docx/xlsx/txt/md/Google Docs) → `docs/`, unknown → `other/` (kept, never
  dropped).
- **Reads everything** — extracts text from every doc (and link-shared Google
  Docs/Sheets), probes every video (duration/resolution/aspect), classifies reference
  images (safezones/placement), and **harvests URLs from all text, recursing one level**
  to pull linked Drive folders / VODs.
- **LLM-analyzes the whole corpus** (brief + every doc + filenames) via Groq into
  structured `rules.json`: required elements, banned words + **banned topics**, platform
  rules, format specs, hashtags/mentions, submission process, deadlines, payout terms,
  style guidance, good/bad examples. Deterministic keyword rules are a **floor** the LLM
  augments — never removes. (Offline / no `GROQ_API_KEY` → deterministic-only, and it
  says so.)
- Writes **`campaign/knowledge.md`** — a human digest every later stage reads so
  campaign context is applied, not re-derived. (Per-campaign; never leaks to
  `memory/longterm.md`.)
- Ends with a **coverage report**: every resource and exactly how it was used
  (`read + extracted`, `applied as safezone`, `watermark candidate`, `links harvested`,
  `UNUSED — no handler`). **Nothing is silently ignored**; unused/ambiguous items and
  spatial constraints needing confirmation are flagged. **It never guesses a rule.**

An **optional source that fails** (e.g. a Kick VOD 403, or a Drive link when `gdown`
isn't installed) is logged and skipped — intake still succeeds as long as some footage
downloaded.

Optional doc parsers (`pypdf`, `python-docx`, `openpyxl`) are in `requirements.txt`; if
one is missing, that doc is reported as unread rather than crashing intake.

### 2. Full run
```bash
python scripts/run.py                         # download → index → select → captions → cut
python scripts/run.py --clips-per-batch 6 --clip-min 12 --clip-max 30
```

### 3. Resume (after a crash / interruption)
```bash
python scripts/run.py --resume                # continues from the last completed stage
python scripts/run.py --force                 # re-run everything from scratch
```
Every stage checkpoints to `state.json`; the long INDEX stage also checkpoints per
VOD chunk, so a resume redoes zero work.

### Offline self-test
```bash
python scripts/selftest.py                    # synthetic video + brief; no Groq/key needed
```

## What each stage produces

| Stage | Script | Output |
|-------|--------|--------|
| 0 Intake | `intake.py` | `brief.md`, `rules.json`, **`knowledge.md`**, `manifest.json`, and `footage/` `assets/` `docs/` `other/` |
| Download | `download.py` | (re)fetches any missing manifest file |
| Index | `index.py` | `campaign/moments.json` (transcript + audio-spike moments) + `campaign/transcripts/` |
| Select | `selectclips.py` | `campaign/selected.json` (top moments, scored) |
| Captions | `captions.py` | `campaign/captions.json` (10 candidates → gauntlet → best + variant + per-platform text) |
| Cut | `cut.py` | **`drafts/NN_score_slug.mp4`** (best first) + **`drafts/manifest.json`** |

`drafts/manifest.json` per clip: filename, caption, source timestamps, score,
tiktok_caption, shorts_title, reels_hashtags, suggested_post_window — all
rules.json-compliant (banned words auto-killed before scoring).

## Notes

- **Long VODs:** Kick VODs can be 3+ hours. INDEX processes audio in 5-minute chunks,
  checkpointing each, and prints progress (`transcribed 47/180 min`). It uses
  faster-whisper **base** on CPU — **the first transcription of a long VOD is the
  slowest stage (~30–60+ min) and that's normal.** A crash mid-VOD resumes at the
  last chunk.
- **Safe-margin footage:** if intake pulls a pre-cropped "safe margin" TikTok/IG
  folder, INDEX prefers those versions automatically.
- **Watermark is mandatory** on every clip — CUT fails loud if no usable image is in
  `campaign/assets/`. It auto-picks the first real watermark PNG and **ignores
  reference images** (filenames with `example`/`placement`/`safezone`/etc. — those are
  placement/safe-zone guides, not overlays). Force a specific file with
  `--watermark-file "SPONGEBOB WATERMARK.png"` (exact or substring). Size/position are
  tunable (`watermark_scale`, `watermark_margin`).
- **Framing is crop-fill:** the source is scaled to COVER 1080×1920 and center-cropped
  — fills edge to edge, no black bars.
- **Captions** are bold white + black outline, top-center, below the top-8% TikTok UI
  safe zone, max 2 lines, auto font-size (rendered with Pillow; `CLIPPER_FONT` overrides
  the bold TTF). **Emoji** are rendered with a color-emoji font (Segoe UI Emoji, or
  `CLIPPER_EMOJI_FONT`); if none is available, emoji are **stripped** rather than shown
  as tofu boxes. Caption content is enforced: **≤8 words, lowercase, must create
  curiosity or state stakes+outcome** — merely-descriptive lines are killed before
  scoring.
- **Selection editorial:** filler segments (stream intros / "we're live" / countdowns)
  are killed; same-event moments within `merge_gap_seconds` (15s) are merged; picks are
  kept `min_separation_seconds` (60s) apart in the same source; each clip is expanded to
  a complete beat — back to its setup (`story_pre_seconds`, 20s) and forward to its
  resolution (`story_post_seconds`, 15s). Clip length is `clip_min_seconds`–
  `clip_max_seconds` (15–45s). All tunable in `run.py`'s `DEFAULT_CONFIG`.
- **After a successful batch**, `run.py` offers to delete raw footage from
  `campaign/footage/` (pass `--cleanup`, or `--no-cleanup` to never prompt). Transcripts
  + `moments.json` are always kept, so re-cutting different moments needs no
  re-download/re-transcribe.
- **Offline test mode** (`CLIPPER_OFFLINE=1`): select/captions use deterministic
  heuristics and index skips whisper — for the self-test only. Real runs require Groq
  and fail loud without `GROQ_API_KEY`.

## Deviations from the build spec (intentional)

- The select stage file is **`selectclips.py`**, not `select.py`: a module named
  `select` shadows Python's stdlib `select` (imported by asyncio/httpx/subprocess on
  Linux) and would break real runs. Stage name/behavior are unchanged.
- Captions are rendered with Pillow and overlaid, rather than ffmpeg `drawtext`, to
  avoid cross-platform font/escaping breakage. Same visual spec.

## Scope

v1 = Account A (flzsh) production + intake. Account B (edited style: EDL plans,
garnish, VO) is explicitly **not built** — see `instructions.md` → FUTURE.
