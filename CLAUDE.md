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
python scripts/run.py --cut-only # re-render ONLY the cut stage from the existing captions.json
                                 # (skips download/index/select/captions — ZERO Groq calls; for
                                 # iterating on the render without burning the Groq daily cap)
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
automation. `--rank N` is a manual override (and BYPASSES the pre-edited filter below).
`intake.py --from-pick` (`apply_pick`) then reads `pick.json` and runs Stage 0 on it — the
wiring that lets intake take the scout pick instead of hand-placed files (fail-loud if the
pick or its files are missing).
**Pre-edited footage filter** (`preedited_footage_skip`): during the walk, a campaign is
skipped if its footage is ENTIRELY pre-edited short clips — resolvable footage where EVERY
measured file is under `--preedited-min-seconds` (default 120s). Measurement is download-free:
gdown lists a Drive folder (`skip_download=True`), yt-dlp reads each file's `duration` from
metadata (`extract_info download=False`), probed concurrently with retries. YouTube
channels/playlists are raw VOD sources and PASS without probing; any file ≥ threshold PASSES.
FAILS OPEN on anything unmeasurable (unlistable folder, majority-unreadable) — never wrongly
excludes; `--rank` forces a skipped one anyway. pickcampaign has NO state.json config — these
are CLI args, so no state key to add. **Measurement is flake-proof:** `_measure_preedited`
returns a 3-way verdict — `skip` / `pass` / `unmeasurable` — and `preedited_footage_skip` logs
EXACTLY one outcome every call (never silent): `[MEASURED → SKIP/PASS]`, `[CACHED SKIP/PASS]`,
or `[UNMEASURABLE … → FAIL OPEN]`. Both the Drive folder listing (`_list_drive_folder`) and the
per-file duration probe (`_probe_duration`) RETRY on transient errors. A **sticky cache**
(`campaign_inputs/preedited_cache.json`, keyed by campaign id + footage-links fingerprint)
persists a successful verdict: once a campaign measured all-short it STAYS skipped on later runs
(reused without re-measuring while the links are unchanged), so a transient listing/probe
failure can never flip a known `skip` into a fail-open pass — the bug where LETSGO measured fine
one run and slipped through the next. `--preedited-refresh` ignores the cache and re-measures.
**Stale-board warning** (`warn_if_stale_board`, Unit 2d): scout's once-daily 20h guard skips
its scrape SILENTLY (exit 0), so the chained `whop.bat` run could pick from a day-old board
unnoticed. pickcampaign reads the top-level `generated_at` from `campaigns.json` and, if the
board is ≥ `--stale-board-hours` (default 20) old, prints a LOUD banner (scout likely skipped;
run scout `--force` for fresh data). Self-contained in the clipper — catches staleness from any
cause, not just the guard.

`intake.py` (brief+links) → `campaign/{brief.md, rules.json, knowledge.md, manifest.json}`
+ `footage/ assets/ docs/ other/`
→ `index.py` → `campaign/moments.json` (whisper transcript + per-word timings + audio-spike moments)
→ `selectclips.py` → `campaign/selected.json`
→ `captions.py` → `campaign/captions.json`
→ `cut.py` → `drafts/NN_score_slug.mp4` + `drafts/manifest.json`.

**Intake is an analyst** (`intake.py` + `analyze.py`): it routes every file by type
(video/image/doc/other — nothing dropped), extracts text from every doc + link-shared
Google Docs, probes videos, HUNTS for nested footage (below), then LLM-extracts
structured rules via Groq (deterministic keyword rules are a FLOOR the LLM augments,
never removes). It writes `knowledge.md` (per-campaign digest) and ends with a coverage
report; unused/ambiguous items are flagged, never guessed.

**TIERED FOOTAGE HUNT** (`intake.hunt_and_download_footage` + `hunt.py`): scout hands over
links; intake resolves WHERE the real footage actually is before giving up, simplest-method-first
via a bounded frontier loop (`hunt.MAX_HOPS`=3 hops: resource → doc → drive/link). **TIER 1 (no
browser):** direct footage (YouTube/Drive/Kick/direct video URL), Google Docs (fetch text + follow
the footage links inside), Drive folders (list/route — if only docs inside, a doc one level deeper
is followed); `_classify_hop` routes each URL — **a Drive *file* link is `download`, NOT a gdoc**
(`AN.classify_url` lumps all of drive.google.com under gdoc, which would misroute a Drive video
found in a doc). Footage is downloaded with the EXISTING capped machinery (footage cap / cookies /
per-file skip-not-fail all intact). **TIER 2 (Playwright, ONLY when needed):** a THIRD-PARTY website
whose links a simple fetch can't extract → `hunt.extract_footage_links_from_site` (plain fetch
FIRST, escalate to headless chromium only if that finds nothing) scrapes the Drive/YouTube links off
the rendered page. NEVER used for Drive/YouTube/Doc. Playwright is OPTIONAL — missing lib/binary
degrades Tier 2 to "unresolved" with a loud install hint, never a crash. **HARD STOP:** if reaching
footage needs login / signup / payment / any manual step (`hunt.detect_barrier`), intake NEVER
proceeds — it fails loud ("footage requires login/signup/payment — skipping campaign") so the
auto-advance walk moves on. We never enter credentials or pay. A gate is acted on ONLY when NO free
footage was found (a route we can go around is ignored: footage-found wins). Campaign fails only if
NO reachable footage exists. The whole hunt path is logged ("Google Doc → found link → downloading"
/ "site → Playwright loaded → 5 Drive links" / "site → signup wall — skipped") in the coverage report.

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
- **Survive one bad input; only die when EVERYTHING is unusable (Unit 2).** `index.py` probes
  each footage file for a readable audio stream FIRST and SKIPS a no-audio/corrupt file (warn,
  continue) — the stage completes as long as ≥1 file indexed, hard-failing (`C.NothingUsable`)
  only at ZERO usable sources. **LLM calls go through a 3-provider FAILOVER CHAIN**
  (`common.LLMChain`, `llm_client`/`llm_chat`): GROQ → GEMINI → CEREBRAS, all free-tier, order
  via config `llm_providers`. **MULTIPLE KEYS PER PROVIDER**: each provider expands to ONE chain
  entry per env key — numbered `GROQ_API_KEY_1..N` (however many are set, auto-detected — add a
  5th and it's picked up with zero code change) then the bare `GROQ_API_KEY` as a fallback when
  no numbered keys exist (same scheme for `GEMINI_API_KEY_*` / `CEREBRAS_API_KEY_*`). Effective
  order: GROQ key1 → key2 → … → GEMINI → CEREBRAS (`_collect_provider_keys` + `_make_providers`).
  Keys come ONLY from env — never written to disk, never logged (logs name the provider + key
  INDEX only, e.g. "GROQ key 2/4"); a provider with no key/library is skipped (warn once).
  `common.classify_rate_limit` tells a DAILY cap (TPD/RPD/PerDay, or a wait > 90s)
  from a short per-minute limit: a per-minute limit backs off + retries the CURRENT key; a
  DAILY cap advances to the NEXT KEY of the same provider, and only after ALL that provider's
  keys are capped does it fall to the next provider (STAYS there — no bounce back; a fresh
  process restarts at Groq key 1). Every provider returns a PLAIN STRING so the select/
  caption JSON parsers, prompts, and banned-word logic are unchanged. Only when EVERY key of
  EVERY provider is capped does it raise `C.GroqDailyCapError`; select/captions checkpoint each
  finished batch/clip
  to `select_partial.json` / `captions_partial.json`, so run.py reports "N of M done" and
  `stop_resumable`s (exit 7) — `--resume` continues without re-spending calls (partials deleted
  on stage completion). `groq_client`/`groq_chat` remain as back-compat aliases (intake/analyze
  get failover free). A campaign that yields
  NOTHING usable raises `C.NothingUsable`; run.py dead-ends loud by default, or with
  `--auto-advance` **WALKS DOWN the ranked list** (`run.walk`): pick → intake → run, and on a
  dead campaign (no footage / dead-floor best<40 / language gate / failed intake) re-picks
  (`pickcampaign --exclude-id` each tried one) + intake and runs the NEXT one — until clips are
  produced, `--auto-advance-max` campaigns are tried (default **10**), or the list is exhausted.
  Opt-in (it downloads more campaigns). **Anti-throttle:** every attempt hits YouTube, so attempts
  are SPACED (`walk_spacing_seconds`, default 20; `--walk-spacing`) and a detected bot-check /
  429 (`_looks_like_throttle`) triggers a longer back-off (`walk_throttle_backoff_seconds`,
  default 300) instead of hammering — cookies stay applied throughout. `run.walk` is a pure,
  unit-tested core (inject process/advance fns); each pipeline run OR failed intake = one attempt.
- **Everything is resumable.** New work must checkpoint to `state.json` (and to disk)
  before the next step. `index.py` checkpoints per 5-min VOD chunk — preserve that. It
  REUSES the on-disk transcript/rms/words partials ONLY when `chunks_done > 0` (a genuine
  resume); a fresh start or `--force` (which resets `chunks_done` to 0) ignores the stale
  partials and rebuilds — appending to them instead DOUBLES the transcript/words.
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
- **Two vertical layouts (`config layout`, default `auto`).** GENERAL = the original single-pass
  `blur_fill` (whole 16:9 frame on a blurred bg). TRACK (`scripts/reframe.py`, OpenShorts-style)
  crops 16:9→9:16 following the subject so the streamer FILLS the frame. `auto` picks per clip:
  a single clear subject → TRACK, a group shot / subject-less landscape / already-vertical →
  GENERAL (`reframe.decide_track`). Detection = **MediaPipe BlazeFace (Tasks API) → YOLOv8n
  person** fallback; models auto-download once into `models/` (gitignored).
  **ZOOM/tightness = `track_subject_scale`** (fraction of output height the subject fills; ~0.6 =
  head+shoulders+room, lower = looser, higher = tighter) — sized ONCE per clip from the subject
  bbox (`_measure_subject`/`_plan_crop`) so zoom never pulses; a too-close webcam subject is scaled
  onto a blurred letterbox to still reach a looser target. `track_safe_zone` is a DIFFERENT knob —
  the `SmoothedCameraman` PAN dead-zone (anti-jitter), NOT zoom; don't confuse them. Stabilizer
  (`SmoothedCameraman`, "heavy tripod"): HOLDS the crop while the subject stays in that safe zone,
  pans only when they leave it, eased + speed-capped (`track_max_pan`) so it never jitters.
  TRACK renders in **3 passes** — `build_content_cmd` (trim/cold-open/CFR 16:9 + audio) →
  `reframe.track_reframe` (OpenCV per-frame crop, detection every `track_detect_every` frames) →
  `build_overlay_cmd` (hook + karaoke + watermark burned on top) — so EVERY other feature is
  identical; only the pixels underneath change. `_content_fc` is the shared trim/CFR/audio core
  both blur_fill and the content pass use (keep it single-source so FIX 3's CFR can't drift).
  Heavy deps (mediapipe/ultralytics/opencv/torch) are **optional**: `reframe.deps_available()`
  warns loudly with the pip line and falls back to blur_fill — never a silent break. In TRACK the
  footage fills the frame (no letterbox band), so `subtitle_top_y` uses the full-height fallback.
- **Hook caption case is Title Case (Like This), per user preference** — `captions.titlecase`
  is applied in `finalize_caption` (hook caption). It capitalizes each word's first letter and
  leaves the rest untouched, so contractions survive ("don't"→"Don't", masked "b**"→"B**").
  **The karaoke SUBTITLES are styled DIFFERENTLY** (flzsh / Gen Z look, `cut._subtitle_word`):
  all lowercase, apostrophes dropped so contractions read as one word ("what's"→"whats"), and
  ALL other punctuation stripped ("Saki."→"saki", "difference?"→"difference"); digits survive.
  This runs BEFORE masking in `_group_lines`, and sentence-boundary line breaks read the RAW
  word's trailing ".!?" before it's stripped. Only the lower karaoke band is restyled — the hook
  plate keeps Title Case + punctuation.
- **Two DIFFERENT censors on karaoke words** (`_group_lines`, lower band only — never the hook):
  a CAMPAIGN-banned word (bet/gamble, rules compliance) is FULLY masked by `_mask_banned_word`
  ("b**"); anything else that is PROFANITY gets a LIGHTER stylistic vowel-censor by
  `_censor_profanity` — keep consonants + shape, replace vowels with "*" ("shit"→"sh*t",
  "fuck"→"f*ck", "ass"→"*ss", "fucking"→"f*ck*ng"). Banned takes precedence (full mask wins).
  The profanity set (`_PROFANITY_ROOTS`, swears + slurs) is matched case-insensitively with
  inflections (fucking/bitches/shitty via an anchored root + optional doubled consonant + short
  suffix) so it catches suffixed forms WITHOUT false-positiving innocent look-alikes (assault,
  assess, hello, cocktail stay untouched). Toggle: `subtitle_censor_profanity` (default true).
  `emoji_in_caption` (config, default true) keeps emoji IN THE HOOK caption; cut renders them
  with a color-emoji font and
  DROPS any glyph the font can't draw so a tofu box never ships. Tofu-detection uses fontTools'
  live cmap if installed, else a curated allowlist (`cut._ALLOWED_EMOJI_CP`) — because Segoe UI
  Emoji draws unknown code points as a visible box that a pixel probe can't distinguish from a
  real glyph. (Karaoke subtitles are ASCII-only — spoken words, no emoji.)
- **Word-level karaoke subtitles** (`subtitles_enabled`, config default true; auto-off in
  offline mode and when moments.json predates the feature — both lack word timings).
  Rendered via **ASS/libass** (`cut.build_ass`), NOT Pillow/drawtext: one ASS Dialogue per
  word shows the FULL line with only the currently-spoken word emphasised so it POPS, reverting
  as the next word speaks. **Per-clip KARAOKE VARIETY comes from the named STYLE-SET**
  (`DEFAULT_STYLE_SET` + `resolve_clip_style`, overridable via config `style_set` so each account/
  category can ship its own): each clip's accent COLOUR (curated high-contrast palette) and
  active-word EMPHASIS ("color" = accent fill `{\c…}` / "box" = accent highlight, a THIN accent border
  `{\3c…\bord…}` via `box_border`, default 5 — subtle, not a heavy box) are rotated DETERMINISTICALLY
  off a stable md5 hash of the moment id (`_style_hash`) — the pipeline is audio-only, so variety is
  rotation-based, NOT matched to visuals, and identical on every re-cut. `subtitle_accent_color` is
  only the FALLBACK when no style-set applies. **The HOOK is SEPARATE and intentionally CONSTANT** —
  plain white text at the fixed TOP position; only its plate/outline changes, via the `hook_style`
  preset (`cut.HOOK_STYLES` A|B|C|D, **LOCKED to A** = no-plate + thick black outline; B=no-plate+thin,
  C=thin semi-transparent plate, D=thick plate). Hook font auto-shrinks to fit width but is CAPPED at
  `HOOK_MAX_FONT` (config `hook_max_font_size`, ~64px) so a SHORT hook lands at/near the reference size
  instead of ballooning. Per-clip hook colour/position variance was reverted so the style is fixed.
  **Karaoke lowercasing keeps the pronoun "I"
  CAPITAL** (`_subtitle_word`: `i`→`I`, `I'm/I'll/I've/I'd`→`Im/Ill/Ive/Id`, gated on a real
  apostrophe so `ill`/`id` and `i`-inside-words are untouched). **Timing (the critical part):**
  index.py stores per-word `{word,start,end}` (whisper `word_timestamps`) in moments.json;
  cut.py's `map_words_to_output` maps those SOURCE-absolute times through the EXACT same
  `segments` list compose plays (cold-open reorder + dead-air trims + cmax tail cut), so a word
  shown in the cold-open teaser AND again in the setup lands correctly in both — **do NOT
  remap against source order or captions drift silently.** `map_words_to_output` renders each
  word ONCE PER TIME IT IS HEARD via two dedup passes: it drops exact-duplicate SOURCE words
  (whisper/index can emit the same `{word,start,end}` twice — e.g. a re-indexed VOD — which
  otherwise burned "That That"), and merges the contiguous output fragments a single word
  produces when it straddles a segment boundary; the two far-apart events from a real cold-open
  replay are KEPT (heard twice). The `ass` filter is chained LAST in `build_compose_cmd` (after
  the 1080x1920 blur-fill) so subs render at output res; its Windows path is escaped via
  `_ass_filter_arg` (`ass='C\:/…/x.ass'`). **Campaign-banned words are masked before burn**
  (`cut._mask_banned_word`) and each finished line is re-checked (fail loud if one slips) —
  same gauntlet discipline as captions. **Placement** must never sit on the footage: the line's
  TOP is pinned into the lower letterbox band via ASS Alignment 8 (top-center) + MarginV=`top_y`,
  where `subtitle_top_y` computes the footage-rectangle bottom from the SAME blur_fill geometry
  compose renders (`blur_fill_footage_bottom`, driven by the source aspect via `probe_dimensions`)
  and drops `subtitle_band_margin` px below it — so a 2nd line grows DOWN into the band, never up
  onto the footage. `SUBTITLE_BOX_W`=640 → MarginL/R (safe box); `SUBTITLE_CENTER_Y`=1440 is only
  the fallback (crop_fill / full-height source with no band). Clear of the top hook plate, the
  bottom-right watermark, and the bottom UI / right action-rail notch. **Line grouping follows
  speech RHYTHM** (`_group_lines`): the PRIMARY break is a PAUSE — when the silence between two
  words exceeds `subtitle_pause_gap` (default 0.35s) the line ends there, even at 1 word, so each
  spoken burst is its own line ("hello" [pause] "how are you" → two lines). Gaps are read from
  the OUTPUT-timeline events so the rhythm matches the FINAL edit (post reorder/dead-air trim).
  Within a continuous pause-free run it still caps at `subtitle_max_words` (~3, +1 to avoid an
  orphan); always ONE physical line (ASS `WrapStyle: 2` forbids auto-wrap).
- **Non-speech sound labels** (karaoke `*scream*` / `*laughing*` fills). Whisper only
  transcribes SPEECH, so a scream/laugh/crash is a silent GAP in the word karaoke. The captions
  stage (`captions._nonspeech_beats` + `_label_sound_beats`) finds loud `audio_spike` moments
  with NO overlapping transcript and asks Groq for a SHORT accurate descriptor — returning null
  (no label) when it can't tell, so it NEVER guesses randomly — storing `sound_fx`:
  `[{start,end,peak,label}]` (source-absolute) per clip in captions.json. Only fires when a clip
  has a real non-speech beat (≤3 per clip), so it adds Groq calls sparingly and NONE for
  all-speech clips. cut.py stays ZERO-Groq: it maps `sound_fx` through the same `segments` and
  burns each as a standalone `*label*` line (`cut._fx_line_text`, lowercase + asterisks, banned
  words still masked) dropped into the word gap.
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
