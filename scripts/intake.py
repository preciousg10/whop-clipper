"""STAGE 0 — INTAKE as a thorough campaign analyst.

Instead of just filing downloads and keyword-scanning the brief, intake now:
  1. READS EVERYTHING — extracts text from every doc (pdf/docx/sheets/txt/md/gdoc),
     probes every video (duration/resolution/aspect), classifies reference images,
     and HUNTS for nested footage (hunt_and_download_footage + hunt.py): a tiered
     frontier that follows docs/gdocs/drive up to 3 hops (Tier 1, no browser) and loads
     a third-party site with Playwright only when needed (Tier 2), skipping any campaign
     whose footage sits behind a login/signup/payment/manual gate.
  2. LLM-ANALYZES the whole corpus (brief + every doc + filenames) via Groq into
     structured rules.json. Deterministic keyword rules are a FLOOR the LLM augments,
     never removes.
  3. Writes campaign/knowledge.md — a human digest every later stage reads.
  4. Ends with a COVERAGE REPORT: every resource + how it was used. Nothing is
     silently ignored; unused/ambiguous items are flagged.
  5. Fails loud on ambiguity — never guesses a rule.

  python intake.py --campaign "WTF Leagues" --brief brief.txt --links links.txt \
      [--cookies-from-browser chrome]
"""
import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import download as DL
import analyze as AN
import hunt as HUNT

HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.]+")
SAFE_MARGIN_HINTS = ("safe margin", "safe-margin", "safemargin", "cropped",
                     "safe zone", "tiktok/ig", "9:16")
SECTION_CAP = 40


# --- scout handoff (Task C) ----------------------------------------------------
def apply_pick(args):
    """Load the scout->clipper handoff (campaign_inputs/pick.json from pickcampaign.py) and
    point intake's --campaign/--brief/--links at it. This is the wiring that lets intake run
    on the campaign scout ranked #1 instead of hand-placed files. Fail loud if the pick or
    its files are missing — never fall back to whatever stale inputs happen to be on disk."""
    pick_path = os.path.expanduser(args.pick_json) if args.pick_json \
        else str(C.ROOT / "campaign_inputs" / "pick.json")
    if not os.path.exists(pick_path):
        C.fail(f"--from-pick set but no pick file at {pick_path}.\n"
               "Run the picker first: python scripts/pickcampaign.py")
    meta = C.load_json(pick_path)
    if not meta:
        C.fail(f"pick file is empty or corrupt: {pick_path}")

    def _abs(rel):
        if not rel:
            return None
        return rel if os.path.isabs(rel) else str(C.ROOT / rel)

    args.campaign = meta.get("campaign") or args.campaign
    args.brief = _abs(meta.get("brief"))
    args.links = _abs(meta.get("links"))
    if not args.brief or not os.path.exists(args.brief):
        C.fail(f"pick references a brief that isn't on disk: {args.brief!r} "
               f"(from {pick_path}). Re-run pickcampaign.py.")
    if not args.links or not os.path.exists(args.links):
        C.fail(f"pick references a links file that isn't on disk: {args.links!r} "
               f"(from {pick_path}). Re-run pickcampaign.py.")
    C.log(f"intake from scout pick: campaign={args.campaign!r} "
          f"(rank {meta.get('rank')}, composite {meta.get('composite_score')}); "
          f"brief={args.brief}, links={args.links}")


# --- clean slate on a NEW campaign ---------------------------------------------
def _wipe_dir_contents(d):
    """Delete every file/subdir INSIDE d (keep d itself — OneDrive can lock the dir).
    Returns the count removed. Only ever called on the clipper's own campaign working dirs."""
    if not d.exists():
        return 0
    removed = 0
    for p in sorted(d.iterdir()):
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
        except Exception as e:
            C.warn(f"could not remove {p}: {e}")
    return removed


def clear_prior_campaign(new_campaign, prior_manifest):
    """When intake starts a DIFFERENT campaign than the one currently on disk, clear the prior
    campaign's footage + transcripts + stale derived artifacts (moments/selected/captions) so
    the new campaign starts clean — otherwise old junk (e.g. LETSGO's NBA/Pokemon footage) piles
    up in campaign/footage/ and gets re-indexed, wasting time and polluting results.

    SAME campaign (resume / re-run) → NO-OP: clearing would force a needless re-download and
    defeat the unchanged-file reuse. SAFETY: only ever touches C.FOOTAGE / C.TRANSCRIPTS and the
    named artifact files — all fixed paths under campaign/, never anything outside them."""
    prior_campaign = (prior_manifest or {}).get("campaign")
    if not prior_campaign or prior_campaign == new_campaign:
        return  # fresh (nothing recorded) or same campaign → keep everything for reuse/resume
    C.warn(f"campaign changed ({prior_campaign!r} → {new_campaign!r}) — clearing the prior "
           f"campaign's footage/transcripts/derived files so the new one starts clean "
           f"(stale footage would otherwise be re-indexed and pollute results).")
    cleared = []
    for d in (C.FOOTAGE, C.TRANSCRIPTS):
        n = _wipe_dir_contents(d)
        if n:
            cleared.append(f"{d.relative_to(C.ROOT).as_posix()}/ ({n} item(s))")
    for f in (C.MOMENTS_JSON, C.SELECTED_JSON, C.CAPTIONS_JSON,
              C.SELECT_PARTIAL, C.CAPTIONS_PARTIAL):
        if f.exists():
            try:
                f.unlink()
                cleared.append(f.relative_to(C.ROOT).as_posix())
            except Exception as e:
                C.warn(f"could not remove {f}: {e}")
    C.log("cleared: " + (", ".join(cleared) if cleared else "(nothing to clear)"))


# --- inputs --------------------------------------------------------------------
def read_brief(args):
    if args.brief_text:
        return args.brief_text
    if args.brief:
        p = os.path.expanduser(args.brief)
        if not os.path.exists(p):
            C.fail(f"brief file not found: {p}")
        return open(p, encoding="utf-8", errors="replace").read()
    C.fail("no brief provided. Use --brief <file> or --brief-text \"...\".")


def collect_links(args):
    links = [u.strip() for u in (args.link or [])]
    if args.links:
        p = os.path.expanduser(args.links)
        if not os.path.exists(p):
            C.fail(f"links file not found: {p}")
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                links.append(ln)
    if not links:
        C.fail("no links provided. Use --links <file> and/or --link <url> (repeatable).")
    return links


def looks_safe_margin(url):
    return any(h in url.lower() for h in SAFE_MARGIN_HINTS)


# --- deterministic floor (applied to the WHOLE corpus, not just the brief) ------
def _lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _section(text, keywords):
    out = [ln for ln in _lines(text) if any(k in ln.lower() for k in keywords)]
    return out[:SECTION_CAP]


def extract_banned_words(text):
    banned = set(w.lower() for w in C.DEFAULT_BANNED_WORDS)
    markers = ("banned", "prohibited", "do not use", "don't use", "no mention of",
               "avoid", "not allowed", "forbidden", "blocklist", "blacklist")
    for ln in _lines(text.lower()):
        if any(m in ln for m in markers):
            frag = re.split(r"[:\-–]", ln, maxsplit=1)
            frag = frag[1] if len(frag) > 1 else ln
            for tok in re.split(r"[,/;]| and | or ", frag):
                w = re.sub(r"[^a-z0-9'+\- ]", "", tok).strip()
                if w and 1 <= len(w.split()) <= 3 and len(w) >= 2 and not w.isdigit():
                    banned.add(w)
    return sorted(banned)


def extract_required_elements(text, asset_names):
    req, low = [], text.lower()
    if "watermark" in low or any("watermark" in a.lower() or "logo" in a.lower() for a in asset_names):
        req.append({"type": "watermark", "detail": "Campaign watermark PNG on EVERY clip."})
    for tag in sorted(set(HASHTAG_RE.findall(text))):
        req.append({"type": "hashtag", "detail": f"Include {tag}"})
    for m in sorted(set(MENTION_RE.findall(text))):
        req.append({"type": "mention", "detail": f"Mention {m}"})
    if any(k in low for k in ("disclosure", "#ad", "ftc", "sponsored", "paid partnership")):
        req.append({"type": "disclosure", "detail": "FTC disclosure required (e.g. #ad)."})
    return req


# --- automatic watermark decision (no human prompt) ----------------------------
# The watermark policy is decided from the campaign rules/brief/knowledge, NOT by stopping to
# ask a human. Required -> burn it (use a provided image if named); no requirement -> skip;
# genuinely ambiguous (contradictory or an unclear bare mention) -> flag for review but STILL
# proceed with watermark OFF (a missing watermark is fixable in review; a wrongly-added overlay
# is worse). cut.py keys off rules.json['watermark_required'] (True/False), which we always set.
WATERMARK_REQUIRED_RE = re.compile(
    # "watermark ... required/mandatory/on every clip" (the requirement follows the noun)
    r"\b(watermark|logo|branding|brand overlay)s?\b[^.\n]{0,40}?\b(?<!not )(?<!not  )"
    r"(required|mandatory|must|always|needed|on (all|every|each)|in all (videos|clips|posts))"
    # "must/always/need to ... use/include/add ... watermark" (the requirement precedes it)
    r"|\b(must|always|required to|need to|have to|has to)\b[^.\n]{0,40}?\b(use|include|add|apply|have)\b"
    r"[^.\n]{0,20}?\b(watermark|logo|branding)"
    # "use/add watermark on every clip / in all videos"
    r"|\b(use|include|add|apply)\b[^.\n]{0,20}?\b(watermark|logo)\b[^.\n]{0,30}?"
    r"\b(on (all|every|each)|in all|every (clip|video|post))",
    re.I)
WATERMARK_NONE_RE = re.compile(
    r"\bno watermark\b|\bwithout (a |the )?watermark\b"
    r"|\bwatermark\b[^.\n]{0,20}?\b(not (required|needed|necessary|mandatory)|optional)"
    r"|\b(don'?t|do not|no need to) (use|need|add|include|apply) (a |the )?(watermark|logo)"
    r"|\bno logo\b|\blogo\b[^.\n]{0,15}?not (required|needed)",
    re.I)
WATERMARK_MENTION_RE = re.compile(r"\bwatermark\b|\blogo overlay\b", re.I)


def _wm_phrase(m):
    return re.sub(r"\s+", " ", m.group(0)).strip()[:60]


def decide_watermark(rules, corpus, asset_names):
    """Automatically decide the watermark policy from the campaign text + provided assets.

    Returns (required: bool, reason: str, ambiguity: str|None, asset_hint: str|None):
      - required  -> the value written to rules.json['watermark_required'] (cut.py obeys it).
      - reason    -> a one-line WHY for the log.
      - ambiguity -> a review flag ONLY when contradictory/unclear (still proceeds OFF), else None.
      - asset_hint-> a watermark image name found in this campaign's assets, when required.
    """
    text = corpus or ""
    req = WATERMARK_REQUIRED_RE.search(text)
    neg = WATERMARK_NONE_RE.search(text)
    mention = WATERMARK_MENTION_RE.search(text)
    wm_asset = next((a for a in asset_names
                     if "watermark" in a.lower() or "logo" in a.lower()), None)

    if req and neg:
        # A required-looking match may just be the negation phrase self-matching ("watermark not
        # required" contains both 'watermark' and 'required'). Re-test with negations removed: if a
        # requirement still stands, it's a GENUINE contradiction; otherwise it's simply not required.
        if WATERMARK_REQUIRED_RE.search(WATERMARK_NONE_RE.sub(" ", text)):
            return (False,
                    f"CONTRADICTORY rules — required ('{_wm_phrase(req)}') AND not-required "
                    f"('{_wm_phrase(neg)}') both found; defaulting OFF (safe).",
                    "Watermark rules contradict each other (both 'required' and 'not required' "
                    "found) — defaulted to OFF; confirm the correct policy.",
                    None)
        return (False, f"not required per rule '{_wm_phrase(neg)}' — skipping.", None, None)
    if req:
        hint = (f" → using assets/{wm_asset}" if wm_asset else
                " (no watermark image in assets/ yet — add one or CUT will fail loud)")
        return (True, f"required per rule '{_wm_phrase(req)}'{hint}.", None, wm_asset)
    if neg:
        return (False, f"not required per rule '{_wm_phrase(neg)}' — skipping.", None, None)
    if wm_asset:
        return (True, f"no explicit rule, but a watermark image was provided "
                      f"(assets/{wm_asset}) → treating as required and using it.", None, wm_asset)
    if mention:
        return (False,
                f"'watermark' mentioned ('{_wm_phrase(mention)}') but no clear required/"
                f"not-required rule — defaulting OFF (safe).",
                "Watermark is mentioned but the requirement is unclear — defaulted to OFF; "
                "confirm whether one is mandatory.",
                None)
    return (False, "no watermark requirement found in rules/brief/knowledge — skipping.",
            None, None)


def _sync_watermark_element(rules, required):
    """Keep required_elements coherent with the decision: a watermark entry present iff required.
    (The word-presence heuristic in extract_required_elements can list one even for a negation.)"""
    els = list(rules.get("required_elements", []))
    has = any(e.get("type") == "watermark" for e in els)
    if required and not has:
        els.append({"type": "watermark", "detail": "Campaign watermark PNG on EVERY clip."})
    elif not required and has:
        els = [e for e in els if e.get("type") != "watermark"]
    rules["required_elements"] = els


def build_floor(text, asset_names):
    return {
        "banned_words": extract_banned_words(text),
        "banned_topics": [],
        "required_elements": extract_required_elements(text, asset_names),
        "platform_rules": _section(text, ("tiktok", "reels", "shorts", "instagram",
                                          "youtube", "aspect", "vertical", "9:16",
                                          "length", "seconds", "duration")),
        "format_specs": {},
        "hashtags": sorted(set(HASHTAG_RE.findall(text))),
        "mentions": sorted(set(MENTION_RE.findall(text))),
        "submission_process": _section(text, ("submit", "submission", "whop", "post link",
                                              "link in", "how to enter")),
        "deadlines": _section(text, ("deadline", "due date", "ends on", "closes", "expires")),
        "payout_terms": _section(text, ("$", "per 1", "cpm", "budget", "payout", "rate", "rpm")),
        "style_guidance": "",
        "examples_good": [],
        "examples_bad": [],
    }


# --- resource processing -------------------------------------------------------
def process_resource(res):
    """Enrich one downloaded resource in place; return any extracted text (for corpus)."""
    p = C.ROOT / res["path"]
    res.setdefault("usage", [])
    res.setdefault("notes", [])
    res.setdefault("urls_found", [])
    kind = res["kind"]

    if kind == "footage":
        v = AN.probe_video(p)
        res["video"] = v
        res["usage"].append(
            f"probed video ({v.get('width')}x{v.get('height')} {v.get('aspect')}, "
            f"{v.get('duration_sec')}s)")
        return ""

    if kind == "asset":
        w, h = AN.image_dims(p)
        res["dims"] = [w, h]
        purpose = AN.reference_purpose(p.name)
        if purpose:
            res["purpose"] = purpose
            res["usage"].append(f"applied as {purpose} reference ({w}x{h})")
        else:
            res["usage"].append(f"watermark candidate ({w}x{h})")
        return ""

    if kind == "doc":
        text, method = AN.extract_text(p)
        if text and text.strip():
            res["text_len"] = len(text)
            res["usage"].append(f"read + extracted ({len(text)} chars via {method})")
            urls = AN.harvest_urls(text)
            res["urls_found"] = urls
            if urls:
                res["usage"].append(f"links harvested ({len(urls)})")
            return text
        res["usage"].append(f"UNREAD — {method}")
        res["notes"].append(f"could not extract text: {method}")
        return ""

    res["usage"].append("UNUSED — no handler")
    res["notes"].append("no handler for this file type")
    return ""


def _reuse_if_unchanged(url, prior_by_source):
    """If every file a source produced last time is still on disk with the same
    name+size, return reusable resource dicts (skip the re-download). Any missing or
    resized file -> None (re-fetch the whole source). Missing size (old manifest) also
    forces a re-fetch; the download step's name+size guard then prevents duplicates."""
    prior = prior_by_source.get(url)
    if not prior:
        return None
    reused = []
    for d in prior:
        p = C.ROOT / d["path"]
        size = d.get("size")
        if size is None or not p.exists() or p.stat().st_size != size:
            return None
        reused.append({"path": d["path"], "kind": d["kind"], "source": url,
                       "safe_margin": d.get("safe_margin", False)})
    return reused


def _is_footage_source(url):
    """True if this link will produce VIDEO footage (and is therefore subject to the footage
    cap). Google Docs/Sheets and other non-video links are never capped."""
    if not DL.is_url(url):
        return os.path.splitext(url)[1].lower() in DL.VIDEO_EXTS
    if AN.classify_url(url) in ("vod", "drive_folder") or DL.is_drive_file(url):
        return True
    # bare http(s) link straight to a video file
    return os.path.splitext(url.split("?", 1)[0])[1].lower() in DL.VIDEO_EXTS


def _footage_seconds(entries):
    """Total playable footage seconds across `entries` (ffprobe each footage file)."""
    total = 0.0
    for e in entries:
        if e.get("kind") == "footage":
            try:
                total += float(C.ffprobe_duration(C.ROOT / e["path"]) or 0.0)
            except Exception:
                pass
    return total


# A YouTube CHANNEL link expands to this many recent VODs at most (newest-first); the footage
# cap is the real limiter — we stop as soon as the running total fills, usually well before this.
CHANNEL_MAX_VIDEOS = 40


def download_links(links, cookies, downloaded, max_source_height=720, original=False,
                   prior_by_source=None, budget=None):
    """Download each source, routing files by type. When `budget` is given
    ({"seconds": <float>, "cap": <seconds or None>}) footage is capped VOD-by-VOD: sources are
    downloaded one at a time while the running total is tracked, and once the total reaches the
    cap NO further footage source is STARTED (the source that crosses the line is fully kept —
    we only stop before starting a new one). Non-footage links are never capped.

    A YouTube CHANNEL link is EXPANDED into its recent VODs (newest-first) and each is downloaded
    as its own capped, individually-skippable source — so a channel handle (@name) works, one bad
    video skips instead of failing the campaign, and we never try to pull the whole channel."""
    prior_by_source = prior_by_source or {}
    resources, failures = [], []
    cap = (budget or {}).get("cap")

    def _capped():
        return cap is not None and budget["seconds"] >= cap

    def _download_one(url):
        """Reuse-or-download a SINGLE source (video/folder/file), account footage, collect
        results. Per-source failure is recoverable (skip + record), never fatal."""
        safe = looks_safe_margin(url)
        reused = _reuse_if_unchanged(url, prior_by_source)
        if reused is not None:
            C.log(f"unchanged — skipping re-download: {url} ({len(reused)} file(s) present)")
            resources.extend(reused)
            _account_footage(budget, reused, url, cap)
            return
        try:
            entries = DL.download_source(url, cookies_from_browser=cookies,
                                         max_source_height=max_source_height,
                                         original=original)
        except DL.DownloadError as e:
            C.warn(f"optional source failed — skipping and continuing: {url} — {e}")
            failures.append({"source": url, "error": str(e)})
            return
        added = [{"path": e["path"], "kind": e["kind"], "source": url,
                  "safe_margin": bool(safe) if e["kind"] == "footage" else False}
                 for e in entries]
        resources.extend(added)
        _account_footage(budget, added, url, cap)

    for url in links:
        if url in downloaded:
            continue
        downloaded.add(url)
        # YOUTUBE CHANNEL/PLAYLIST → expand to recent VODs and download each individually.
        if DL.is_youtube_channel(url):
            if _capped():
                C.log(f"footage cap reached — not expanding channel: {url}")
                continue
            try:
                vids = DL.list_channel_videos(url, limit=CHANNEL_MAX_VIDEOS,
                                              cookies_from_browser=cookies)
            except DL.DownloadError as e:
                C.warn(f"channel could not be listed — skipping and continuing: {url} — {e}")
                failures.append({"source": url, "error": str(e)})
                continue
            if not vids:
                C.warn(f"channel resolved but no videos found — skipping: {url}")
                failures.append({"source": url, "error": "no videos found in channel/playlist"})
                continue
            C.log(f"channel {url}: {len(vids)} recent video(s) found — downloading newest-first "
                  f"up to the footage cap.")
            for v in vids:
                if v in downloaded:
                    continue
                downloaded.add(v)
                if _capped():
                    C.log(f"footage cap: running total {budget['seconds'] / 3600:.2f}h ≥ cap "
                          f"{cap / 3600:.1f}h — stopping channel {url} (budget filled).")
                    break
                _download_one(v)
            continue
        # FOOTAGE CAP: don't START a new footage source once we're already at/over the cap.
        if _capped() and _is_footage_source(url):
            C.log(f"footage cap: running total {budget['seconds'] / 3600:.2f}h ≥ cap "
                  f"{cap / 3600:.1f}h — skipping remaining footage source: {url}")
            continue
        _download_one(url)
    return resources, failures


def _account_footage(budget, entries, url, cap):
    """Add this source's footage duration to the running budget and log the running total."""
    if budget is None:
        return
    added = _footage_seconds(entries)
    if added <= 0:
        return
    budget["seconds"] += added
    tail = f" / cap {cap / 3600:.1f}h" if cap else ""
    C.log(f"footage: +{added / 3600:.2f}h from {url} → running total "
          f"{budget['seconds'] / 3600:.2f}h{tail}")


# --- TIERED FOOTAGE HUNT (Tiers 1 & 2) -----------------------------------------
# Scout hands over links; intake HUNTS for where the real footage actually is before giving up.
# A bounded frontier loop (hunt.MAX_HOPS hops) expands the seed links: footage is downloaded &
# routed, Google Docs are read and their inner footage links followed, and a THIRD-PARTY website
# is loaded with Playwright (only when a simple fetch can't extract its links). A login/signup/
# payment/manual gate is recorded; intake acts on it only if NO free footage was found. See hunt.py.
def _classify_hop(url):
    """Route a discovered URL for the hunt:
      'download' — hand to download_source, which routes by type (video→footage, image→asset,
                   doc→doc): a Drive folder/file, YouTube/Kick/Twitch VOD or channel, a direct
                   downloadable-file URL, or ANY local path (a provided watermark/doc/clip);
      'gdoc'     — a Google Doc/Sheet: fetch its text (Tier 1) and follow the links inside;
      'site'     — a third-party webpage: resolve via Tier 2 (simple fetch → Playwright);
      'other'    — anything else: listed, not fetched.
    NOTE: a Drive *file* link is 'download' here (DL.is_drive_file), NOT a gdoc — AN.classify_url
    lumps all of drive.google.com under 'gdoc', which would misroute a Drive video found in a doc."""
    if not DL.is_url(url):
        return "download"                       # local path — download_source._fetch_local routes it
    if DL.is_drive_folder(url) or DL.is_drive_file(url):
        return "download"
    cu = AN.classify_url(url)
    if cu == "gdoc":
        return "gdoc"
    if cu == "vod":
        return "download"
    # A direct URL straight to a downloadable file (video/image/doc) → fetch & route it.
    ext = os.path.splitext(url.split("?", 1)[0])[1].lower()
    if ext in DL.VIDEO_EXTS or ext in DL.IMAGE_EXTS or ext in DL.DOC_EXTS:
        return "download"
    return "site"


def hunt_and_download_footage(seed_urls, *, cookies, downloaded, max_source_height, original,
                              prior_by_source, budget, resources, corpus_parts, failures):
    """Expand `seed_urls` into ALL freely-reachable footage, following docs/gdocs/drive/sites up
    to hunt.MAX_HOPS hops (Tier 1 first; Playwright only for third-party sites). Downloads
    discovered footage with the CAPPED machinery (footage cap, cookies, per-file skip-not-fail
    all intact). Mutates `resources`/`corpus_parts`/`failures` in place.

    Returns (harvested_urls, other_urls, barriers, path_log): barriers is a list of
    (category, url) gates encountered; path_log is the human hunt trace for the coverage report."""
    harvested, other_urls, barriers, path_log = [], [], [], []
    seen_docs, seen_sites, seen_all = set(), set(), set()

    def _dl(urls, label):
        """Download footage-class urls with the capped machinery; process any docs among the
        results and return the URLs harvested from those docs (to feed the next hop)."""
        if not urls:
            return []
        C.log(f"  [{label}] resolving {len(urls)} footage/source link(s)…")
        r2, f2 = download_links(urls, cookies, downloaded, max_source_height=max_source_height,
                                original=original, prior_by_source=prior_by_source, budget=budget)
        failures.extend(f2)
        new_urls = []
        for r in r2:
            t = process_resource(r)
            if t:
                corpus_parts.append(f"\n\n### {r['path']}\n{t}")
            new_urls += r.get("urls_found", [])
        resources.extend(r2)
        n_foot = sum(1 for r in r2 if r["kind"] == "footage")
        if r2 or f2:
            path_log.append(f"[{label}] downloaded {len(r2)} file(s) ({n_foot} footage"
                            f"{f', {len(f2)} failed' if f2 else ''}); harvested {len(new_urls)} "
                            f"link(s) from docs")
        return new_urls

    frontier = list(dict.fromkeys(seed_urls))
    hop = 0
    while frontier and hop <= HUNT.MAX_HOPS:
        label = "seed" if hop == 0 else f"hop{hop}"
        media, gdocs, sites = [], [], []
        for u in frontier:
            seen_all.add(u)
            harvested.append(u)
            k = _classify_hop(u)
            if k == "download":
                media.append(u)
            elif k == "gdoc":
                gdocs.append(u)
            elif k == "site":
                sites.append(u)
            else:
                other_urls.append(u)
        next_frontier = []

        # 1) DOWNLOAD & ROUTE (Tier 1) — footage/asset/doc. Docs pulled from a Drive folder or a
        #    direct doc URL harvest more links to follow.
        next_frontier += _dl(media, label)

        # 2) GOOGLE DOCS (Tier 1) — fetch text, follow the footage links inside.
        for u in gdocs:
            if u in seen_docs:
                continue
            seen_docs.add(u)
            downloaded.add(u)
            text, method = AN.fetch_gdoc_text(u)
            if text and text.strip():
                dest = C.DOCS / ("gdoc_" + re.sub(r"\W+", "_", u)[-40:] + ".txt")
                dest.write_text(text, encoding="utf-8")
                inner = AN.harvest_urls(text)
                resources.append({
                    "path": os.path.relpath(dest, C.ROOT), "kind": "doc", "source": u,
                    "usage": [f"fetched linked Google file text ({len(text)} chars via {method})"],
                    "notes": [], "urls_found": inner})
                corpus_parts.append(f"\n\n### {u}\n{text}")
                next_frontier += inner
                C.log(f"  [{label}] Google Doc → read ({len(text)} chars) → {len(inner)} link(s): {u[:60]}")
                path_log.append(f"[{label}] Google Doc {u[:55]} → read → found {len(inner)} link(s)")
            else:
                b = HUNT.detect_barrier(method)
                if b or "login" in method.lower():
                    b = b or "login"
                    barriers.append((b, u))
                    C.warn(f"  [{label}] Google Doc requires {b} ({method}) — not following: {u}")
                    path_log.append(f"[{label}] Google Doc {u[:55]} → {b} wall ({method})")
                else:
                    other_urls.append(u)
                    path_log.append(f"[{label}] Google Doc {u[:55]} → unreadable ({method})")

        # 3) THIRD-PARTY SITES (Tier 2) — simple fetch, then Playwright only if needed.
        for u in sites:
            if u in seen_sites:
                continue
            seen_sites.add(u)
            other_urls.append(u)
            res = HUNT.extract_footage_links_from_site(u, cookies=cookies)
            links = res.get("links") or []
            if links:
                how = "simple fetch" if res.get("tier") == "fetch" else "Playwright loaded"
                C.log(f"  [{label}] third-party site → {how} → found {len(links)} footage/doc "
                      f"link(s): {u[:60]}")
                path_log.append(f"[{label}] site {u[:55]} → {how} → {len(links)} link(s) found")
                next_frontier += links
            elif res.get("barrier"):
                barriers.append((res["barrier"], u))
                C.warn(f"  [{label}] site requires {res['barrier']} — skipping this route: {u}")
                path_log.append(f"[{label}] site {u[:55]} → {res['barrier']} wall — skipped")
            else:
                note = res.get("note") or "no footage links found"
                C.warn(f"  [{label}] site yielded no footage ({note}): {u}")
                path_log.append(f"[{label}] site {u[:55]} → {note}")

        # Advance: only follow links we haven't already handled/downloaded.
        frontier = [u for u in dict.fromkeys(next_frontier)
                    if u not in downloaded and u not in seen_docs and u not in seen_sites
                    and u not in seen_all]
        hop += 1

    return (list(dict.fromkeys(harvested)), list(dict.fromkeys(other_urls)), barriers, path_log)


# --- outputs -------------------------------------------------------------------
def write_brief_md(campaign, rules, raw):
    def block(items):
        return "\n".join(f"- {x}" for x in items) if items else "_(none found — verify)_"
    reqs = [f"{r['type']}: {r['detail']}" for r in rules.get("required_elements", [])]
    C.BRIEF_MD.write_text(f"""# Campaign brief — {campaign}

> Parsed by intake. Machine-readable rules live at campaign/rules.json; the full
> digest (every resource, every rule) is at campaign/knowledge.md.

## Payout terms
{block(rules.get('payout_terms'))}

## Platform rules
{block(rules.get('platform_rules'))}

## Required elements
{block(reqs)}

## Banned words
{block(rules.get('banned_words'))}

## Banned topics
{block(rules.get('banned_topics'))}

## Submission process
{block(rules.get('submission_process'))}

## ⚠ Ambiguities to resolve (never guessed)
{block(rules.get('ambiguities'))}

---
## Raw brief (verbatim)
```
{raw.strip()}
```
""", encoding="utf-8")


# --- POSTING / SUBMISSION checklist (what the HUMAN must do when posting) -------
# Posting is ALWAYS manual, and campaigns reject submissions for a missing tag/mention/hashtag/
# time-window the clip itself can't show. This builds a plain-English do-this-when-posting
# checklist STRICTLY from the already-extracted rules.json fields (+ a light scan of the raw
# brief for time-windows / link-in-bio that structured fields commonly miss). It never invents a
# requirement — anything unclear is surfaced as "VERIFY:", not asserted.
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_.]+")
_TAG_RE = re.compile(r"#[A-Za-z0-9_]+")
# a posting/submission sentence that also names a time window ("submit within 30 min of posting")
_POST_WORD_RE = re.compile(r"\b(submit|submission|post|posting|upload|publish)\b", re.I)
_TIME_WIN_RE = re.compile(r"\b\d+\s*(?:min(?:ute)?s?|hours?|hrs?|days?)\b", re.I)
_LINK_BIO_RE = re.compile(r"link[\s-]*in[\s-]*bio|in\s+bio", re.I)
# min views / watch-time for payout ("min 1000 views", "at least 10 seconds")
_MIN_VIEWS_RE = re.compile(r"(?:min(?:imum)?|at least|>=|over)\s*[^.\n]*?\b[\d,]+\s*(?:k|m)?\s*views?"
                           r"|\b[\d,]+\s*(?:k|m)?\s*views?\s*(?:min(?:imum)?|required|to qualify)", re.I)
_MIN_DUR_RE = re.compile(r"(?:min(?:imum)?|at least)\s*[^.\n]*?\b\d+\s*(?:seconds?|secs?|s)\b", re.I)


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+|\n+", text or "") if s.strip()]


def build_posting_checklist(campaign, rules, raw_brief=""):
    """Return the POSTING_CHECKLIST.md text for `campaign`, built ONLY from `rules` (rules.json)
    + `raw_brief`. Deterministic — every item traces to an extracted rule or a quoted line; no
    requirement is invented. Ambiguities become VERIFY items."""
    req = rules.get("required_elements", []) or []
    by_type = {}
    for r in req:
        by_type.setdefault(str(r.get("type") or "other").lower(), []).append(str(r.get("detail") or "").strip())

    lines, seen_items = [], set()

    def emit(kind, text):
        text = " ".join((text or "").split())
        if not text:
            return
        key = (kind, text.lower())
        if key in seen_items:
            return
        seen_items.add(key)
        if kind == "box":
            lines.append(f"- [ ] {text}")
        elif kind == "verify":
            lines.append(f"- [ ] **VERIFY:** {text}")
        else:
            lines.append(f"- {text}")

    def section(title):
        lines.append("")
        lines.append(f"## {title}")

    # combined raw text we can quote from (real strings only)
    scan_parts = [raw_brief] + list(rules.get("submission_process") or []) \
        + list(rules.get("platform_rules") or []) + list(rules.get("deadlines") or []) \
        + [d for ds in by_type.values() for d in ds] + list(rules.get("payout_terms") or [])
    scan_text = "\n".join(p for p in scan_parts if p)

    # 1) CAPTION — account mentions / tags -------------------------------------
    mentions = []
    for src in list(rules.get("mentions") or []) + by_type.get("mention", []) + by_type.get("tag", []):
        mentions += _HANDLE_RE.findall(src)
    mentions = list(dict.fromkeys(mentions))
    section("Caption — mentions & tags")
    if mentions:
        for h in mentions:
            emit("box", f"mention {h} in the caption")
    for d in by_type.get("caption", []):
        emit("box", d)
    if not mentions and not by_type.get("caption"):
        lines.append("- _(no required caption mention/tag found in the rules)_")

    # 2) HASHTAGS --------------------------------------------------------------
    tags = []
    for src in list(rules.get("hashtags") or []) + by_type.get("hashtag", []):
        found = _TAG_RE.findall(src)
        tags += found or ([src.strip()] if src.strip().startswith("#") else [])
    tags = list(dict.fromkeys(t if t.startswith("#") else f"#{t}" for t in tags if t))
    section("Hashtags")
    if tags:
        for t in tags:
            emit("box", f"include {t}")
    else:
        lines.append("- _(no required hashtag found in the rules)_")

    # 3) LINK IN BIO -----------------------------------------------------------
    bio_hits = [s for s in _split_sentences(scan_text) if _LINK_BIO_RE.search(s)]
    if bio_hits:
        section("Link in bio")
        for s in bio_hits[:4]:
            emit("box", s)

    # 4) ON-SCREEN / WATERMARK -------------------------------------------------
    section("On-screen / watermark")
    wm = rules.get("watermark_required")
    if wm is True:
        emit("note", "Watermark: REQUIRED — the pipeline auto-burns the campaign watermark; "
                     "confirm it's visible on the exported clip before posting.")
    elif wm is False:
        emit("note", "Watermark: not required for this campaign.")
    else:
        emit("verify", "Watermark requirement unclear — check the campaign page.")
    for d in by_type.get("logo", []) + by_type.get("text_overlay", []) + by_type.get("overlay", []):
        emit("box", d)
    # a restriction that forbids watermarks/end-screens can conflict with the above — surface it
    for d in by_type.get("restriction", []):
        if re.search(r"watermark|end\s*screen|logo", d, re.I) and wm is True:
            emit("verify", f"possible conflict — rules also say: \"{d}\" (confirm which watermark/"
                           f"overlay is allowed).")
        else:
            emit("note", f"restriction: {d}")

    # 5) QUALITY / FORMAT ------------------------------------------------------
    fmt = rules.get("format_specs") or {}
    quality_items = []
    if fmt.get("resolution"):
        quality_items.append(f"resolution: {fmt['resolution']}")
    if fmt.get("aspect_ratio"):
        quality_items.append(f"aspect ratio: {fmt['aspect_ratio']}")
    if fmt.get("length"):
        quality_items.append(f"length: {fmt['length']}")
    for d in by_type.get("quality", []) + by_type.get("resolution", []):
        quality_items.append(d)
    if quality_items:
        section("Quality / format")
        for q in dict.fromkeys(quality_items):
            emit("box", q)

    # 6) SUBMISSION (how + time window) ---------------------------------------
    section("Submission")
    subs = [s for s in (rules.get("submission_process") or []) if not s.strip().startswith("#")]
    for s in subs:
        emit("box", s)
    # time window: a sentence that mentions posting/submitting AND a duration
    windows = []
    for s in _split_sentences(scan_text):
        if _POST_WORD_RE.search(s) and _TIME_WIN_RE.search(s):
            windows.append(s)
    for w in list(dict.fromkeys(windows))[:4]:
        emit("box", f"time window — {w}")
    for d in (rules.get("deadlines") or []):
        emit("box", f"deadline: {d}")
    if not subs and not windows and not rules.get("deadlines"):
        lines.append("- _(no explicit submission step/time-window found — submit via the "
                     "campaign page; VERIFY any post→submit window)_")

    # 7) PLATFORM-SPECIFIC -----------------------------------------------------
    plats = [p for p in (rules.get("platform_rules") or []) if not p.strip().startswith("#")]
    if plats:
        section("Platform notes (TikTok / Reels / Shorts / etc.)")
        for p in plats:
            emit("note", p)

    # 8) PAYOUT — minimum views / watch-time ----------------------------------
    payout = [p for p in (rules.get("payout_terms") or []) if not p.strip().startswith("#")]
    view_hits = [s for s in _split_sentences(scan_text) if _MIN_VIEWS_RE.search(s)]
    dur_hits = [s for s in _split_sentences(scan_text) if _MIN_DUR_RE.search(s)]
    if payout or view_hits or dur_hits:
        section("Payout requirements")
        for s in list(dict.fromkeys(view_hits + dur_hits))[:6]:
            emit("box", s)
        for p in dict.fromkeys(payout):
            emit("note", f"payout: {p}")

    # 8b) OTHER required elements — catch-all so NO extracted requirement is dropped just
    # because its type isn't one of the sections above (e.g. content_source, account_setup,
    # multi_platform_posting). Better to surface a real rule verbatim than silently miss it.
    consumed = {"mention", "tag", "caption", "hashtag", "logo", "text_overlay", "overlay",
                "quality", "resolution", "restriction"}
    other = []
    for r in req:
        t = str(r.get("type") or "other").lower()
        d = str(r.get("detail") or "").strip()
        if d and t not in consumed:
            other.append(d)
    if other:
        section("Other requirements")
        for d in dict.fromkeys(other):
            emit("box", d)

    # 9) VERIFY (ambiguities the rules left unclear) --------------------------
    ambig = rules.get("ambiguities") or []
    if ambig:
        section("⚠ VERIFY before relying on these (unclear in the rules)")
        for a in ambig:
            emit("verify", a)

    body = "\n".join(lines)
    return (f"# Posting checklist — {campaign}\n\n"
            "> Auto-generated by intake from campaign/rules.json. **Posting is manual** — run "
            "through this when you post EACH clip so a submission isn't rejected for a missing "
            "mention/tag/hashtag or a blown time-window. Every item comes from the campaign's "
            "own rules; items marked **VERIFY** were unclear and should be confirmed on the "
            "campaign page (never guessed).\n\n"
            "## This campaign requires when posting:\n"
            f"{body}\n")


def write_posting_checklist(campaign, rules, raw_brief=""):
    C.POSTING_CHECKLIST.write_text(build_posting_checklist(campaign, rules, raw_brief),
                                   encoding="utf-8")


def write_knowledge_md(campaign, rules, resources, harvested, other_urls, corpus_chars, llm_used):
    L = [f"# Campaign knowledge — {campaign}", "",
         "_Per-campaign memory built by intake. Every later stage (select, captions, cut) "
         "reads this + rules.json so campaign context is applied, not re-derived. "
         "This NEVER leaks into memory/longterm.md._", "",
         f"Corpus analyzed: {corpus_chars} chars across {len(resources)} resource(s). "
         f"LLM extraction: {'yes' if llm_used else 'skipped (deterministic only)'}.", ""]

    L += ["## Resources & how each was used", ""]
    for r in resources:
        L.append(f"- **{r['path']}** ({r['kind']}) — {'; '.join(r.get('usage', [])) or 'n/a'}")
        for n in r.get("notes", []):
            L.append(f"    - note: {n}")
        if r.get("urls_found"):
            L.append(f"    - urls: {', '.join(r['urls_found'][:8])}")
    L.append("")

    def sec(title, items):
        L.append(f"## {title}")
        if not items:
            L.append("_(none)_")
        elif isinstance(items, dict):
            for k, v in items.items():
                L.append(f"- **{k}**: {v}")
        else:
            for it in items:
                if isinstance(it, dict):
                    L.append(f"- {it.get('type', '')}: {it.get('detail', '')}")
                else:
                    L.append(f"- {it}")
        L.append("")

    sec("Required elements", rules.get("required_elements"))
    sec("Banned words", rules.get("banned_words"))
    sec("Banned topics", rules.get("banned_topics"))
    sec("Platform rules", rules.get("platform_rules"))
    sec("Format specs", rules.get("format_specs"))
    sec("Hashtags / mentions", (rules.get("hashtags") or []) + (rules.get("mentions") or []))
    sec("Submission process", rules.get("submission_process"))
    sec("Deadlines", rules.get("deadlines"))
    sec("Payout terms", rules.get("payout_terms"))
    sec("Style guidance", [rules.get("style_guidance")] if rules.get("style_guidance") else [])
    sec("Examples — good", rules.get("examples_good"))
    sec("Examples — bad", rules.get("examples_bad"))
    sec("Spatial constraints (need confirmation)", [
        f"{s['type']} — {s['file']} {s.get('dims')} — {s['note']}"
        for s in rules.get("spatial_constraints", [])])
    sec("Harvested links (other, not downloaded)", other_urls)
    sec("⚠ Open questions / ambiguities", rules.get("ambiguities"))

    C.KNOWLEDGE_MD.write_text("\n".join(L), encoding="utf-8")


def build_manifest(campaign, resources, failures, harvested, other_urls):
    total = sum((r.get("video") or {}).get("duration_sec") or 0
                for r in resources if r["kind"] == "footage")
    def _size(rel):
        p = C.ROOT / rel
        return p.stat().st_size if p.exists() else None
    downloads = [{
        "path": r["path"], "kind": r["kind"], "source": r["source"],
        "size": _size(r["path"]),      # for name+size unchanged-detection on re-run
        "safe_margin": r.get("safe_margin", False),
        "duration_sec": (r.get("video") or {}).get("duration_sec"),
        "video": r.get("video"), "usage": r.get("usage", []),
    } for r in resources]
    return {"campaign": campaign, "created_at": C.now_iso(), "downloads": downloads,
            "footage_total_hours": round(total / 3600.0, 2), "failures": failures,
            "harvested_urls": harvested, "other_urls": other_urls}


def build_ambiguities(rules, resources, llm_used):
    amb = []
    if not rules.get("payout_terms"):
        amb.append("No payout terms found — confirm rate/budget before producing.")
    # Watermark is decided AUTOMATICALLY (decide_watermark) — no unconditional prompt here. Only a
    # genuinely ambiguous decision adds a flag, and main() inserts that separately.
    if not rules.get("submission_process"):
        amb.append("No submission process / deadline found — confirm how to submit.")
    for r in resources:
        if any(u.startswith(("UNUSED", "UNREAD")) for u in r.get("usage", [])):
            amb.append(f"Resource not fully used: {r['path']} — {'; '.join(r['usage'])}")
    for s in rules.get("spatial_constraints", []):
        amb.append(f"Spatial constraint from {s['file']} needs confirmation (exact margins/positions).")
    if not llm_used:
        amb.append("LLM extraction skipped (offline / no GROQ_API_KEY) — rules are "
                   "deterministic-only; re-run with Groq for full extraction.")
    return amb


def print_coverage(campaign, rules, resources, manifest, failures, other_urls, hunt_path=None):
    foot = [r for r in resources if r["kind"] == "footage"]
    print("\n" + "=" * 66)
    print(f"INTAKE COVERAGE REPORT — {campaign}")
    print("=" * 66)
    print(f"Footage: {len(foot)} file(s), ~{manifest['footage_total_hours']} h")
    if hunt_path:
        print("\nFootage hunt path (how footage was resolved):")
        for step in hunt_path:
            print(f"  → {step}")
    print("\nEvery resource and how it was used:")
    for r in resources:
        print(f"  • {r['path']}  [{r['kind']}]")
        print(f"      → {'; '.join(r.get('usage', [])) or 'n/a'}")
    if other_urls:
        print(f"\nHarvested links (listed, not downloaded): {len(other_urls)}")
        for u in other_urls[:15]:
            print(f"  • {u}")
    if failures:
        print("\n✗ Download failures (skipped):")
        for f in failures:
            print(f"  • {f['source']} — {f['error']}")
    print(f"\nRules: {len(rules.get('banned_words', []))} banned words, "
          f"{len(rules.get('required_elements', []))} required elements, "
          f"{len(rules.get('spatial_constraints', []))} spatial constraint(s).")
    if rules.get("ambiguities"):
        print("\n⚠ CLARIFY BEFORE PRODUCING (never guessed):")
        for a in rules["ambiguities"]:
            print(f"  • {a}")
    print("=" * 66 + "\n")


# --- main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 0 — campaign intake / analysis.")
    ap.add_argument("--campaign", default="Untitled campaign")
    ap.add_argument("--brief", help="path to a brief text/markdown file")
    ap.add_argument("--brief-text", help="brief text pasted inline")
    ap.add_argument("--links", help="file with one link/path per line")
    ap.add_argument("--link", action="append", help="a single link/path (repeatable)")
    ap.add_argument("--from-pick", action="store_true",
                    help="take the campaign from the scout handoff (campaign_inputs/pick.json, "
                         "written by pickcampaign.py) instead of hand-placed --brief/--links.")
    ap.add_argument("--pick-json",
                    help="path to a specific pick.json (implies --from-pick).")
    ap.add_argument("--cookies-from-browser",
                    help="browser for cookies on gated VODs (chrome/edge/firefox) — Kick needs this")
    ap.add_argument("--max-source-height", type=int, default=720,
                    help="cap for Drive transcoded preview streams in px (default 720)")
    ap.add_argument("--original", action="store_true",
                    help="force raw original Drive files instead of preview streams")
    ap.add_argument("--footage-cap-hours", type=float, default=None,
                    help="stop downloading more footage once cumulative duration reaches this "
                         "many hours, VOD-by-VOD (default: config footage_cap_hours or 10)")
    args = ap.parse_args()
    if args.from_pick or args.pick_json:
        apply_pick(args)

    C.ensure_dirs()
    brief = read_brief(args)
    links = collect_links(args)
    downloaded = set()

    # Prior manifest lets us skip re-downloading sources whose files are unchanged.
    prior = C.load_json(C.CAMPAIGN_MANIFEST) or {}
    # If this is a DIFFERENT campaign than what's on disk, wipe the prior footage/transcripts/
    # derived files first so old campaign junk isn't re-indexed (no-op on a same-campaign re-run).
    clear_prior_campaign(args.campaign, prior)
    prior_by_source = {}
    for d in prior.get("downloads", []):
        prior_by_source.setdefault(d.get("source"), []).append(d)

    # FOOTAGE CAP (VOD-by-VOD): resolve hours from CLI > state config > default 10, and thread a
    # shared budget through the whole footage hunt so the running total spans every hop.
    cfg_cap = (C.load_json(C.STATE_PATH, default={}) or {}).get("config", {}).get("footage_cap_hours")
    cap_hours = args.footage_cap_hours if args.footage_cap_hours is not None else float(cfg_cap or 10)
    budget = {"seconds": 0.0, "cap": (cap_hours * 3600.0 if cap_hours and cap_hours > 0 else None)}
    if budget["cap"]:
        C.log(f"footage cap: downloading VODs one at a time up to ~{cap_hours:g}h cumulative.")

    # TIERED FOOTAGE HUNT: expand the seed links (+ URLs in the brief) into all freely-reachable
    # footage — following docs/gdocs/drive/sites up to hunt.MAX_HOPS hops (Playwright only for
    # third-party sites). Footage is downloaded with the CAPPED machinery (cap/cookies/skip intact).
    resources, failures = [], []
    corpus_parts = [brief]
    seed_frontier = list(dict.fromkeys(list(links) + AN.harvest_urls(brief)))
    C.log(f"footage hunt: resolving from {len(seed_frontier)} seed link(s) "
          f"(up to {HUNT.MAX_HOPS} hop(s); Playwright only for third-party sites)…")
    harvested, other_urls, barriers, hunt_path = hunt_and_download_footage(
        seed_frontier, cookies=args.cookies_from_browser, downloaded=downloaded,
        max_source_height=args.max_source_height, original=args.original,
        prior_by_source=prior_by_source, budget=budget,
        resources=resources, corpus_parts=corpus_parts, failures=failures)

    if hunt_path:
        C.log("footage hunt path:")
        for step in hunt_path:
            C.log(f"    {step}")

    # HARD STOP — decide reachability BEFORE spending LLM tokens. If the hunt found no video and a
    # login / signup / payment / manual gate stood between us and the footage, name it and skip
    # (fail loud → the auto-advance walk moves on). We NEVER enter credentials or pay, ever.
    footage_now = [r for r in resources if r["kind"] == "footage"]
    if not footage_now:
        if barriers:
            cats = ", ".join(sorted({b for b, _ in barriers}))
            urls = ", ".join(u for _, u in barriers[:4])
            C.fail(f"footage requires {cats} — skipping campaign. No freely-reachable footage was "
                   f"found; every route to it hit a login/signup/payment/manual wall ({urls}).")
        C.fail("intake produced no reachable footage — nothing to clip. Followed every free "
               "link / doc / site to the hop limit and found no video. Check the links/log above.")

    tree = "\n".join(r["path"] for r in resources)
    corpus = "\n".join(corpus_parts) + "\n\n### FILES\n" + tree

    # Deterministic floor + LLM extraction, merged (LLM augments, never removes).
    asset_names = [os.path.basename(r["path"]) for r in resources if r["kind"] == "asset"]
    floor = build_floor(corpus, asset_names)
    client = C.groq_client()
    llm = AN.groq_extract(client, args.campaign, corpus, floor) if client else {}
    llm_used = client is not None and bool(llm)
    if client is None:
        C.warn("offline / no GROQ_API_KEY — skipping LLM extraction; deterministic rules only.")
    rules = AN.merge_rules(floor, llm)
    rules["campaign"] = args.campaign

    # WATERMARK: decide automatically (never stop to ask). A prior EXPLICIT bool for the SAME
    # campaign is kept (respects a manual correction and avoids flip-flopping on re-intake);
    # otherwise decide_watermark reads the rules/brief/knowledge and sets True/False. Ambiguous
    # cases still proceed OFF but add a review flag.
    prior_rules = C.load_json(C.RULES_JSON) or {}
    prior_wm = (prior_rules.get("watermark_required")
                if prior_rules.get("campaign") == args.campaign else None)
    wm_ambiguity = None
    if isinstance(prior_wm, bool):
        rules["watermark_required"] = prior_wm
        _sync_watermark_element(rules, prior_wm)
        C.log(f"watermark: keeping prior decision for this campaign (watermark_required={prior_wm}).")
    else:
        wm_required, wm_reason, wm_ambiguity, _wm_asset = decide_watermark(
            rules, corpus, asset_names)
        rules["watermark_required"] = wm_required
        _sync_watermark_element(rules, wm_required)
        C.log(f"watermark: {wm_reason}")

    # Spatial constraints from reference images (flagged — never auto-guess margins).
    rules["spatial_constraints"] = [{
        "type": r["purpose"], "file": r["path"], "dims": r.get("dims"),
        "needs_confirmation": True,
        "note": "exact margins/positions not auto-derived — confirm from the image",
    } for r in resources if r.get("purpose")]

    rules["ambiguities"] = build_ambiguities(rules, resources, llm_used)
    if wm_ambiguity:                       # only when the auto-decision was genuinely unclear
        rules["ambiguities"].insert(0, wm_ambiguity)

    C.save_json(C.RULES_JSON, rules)
    write_brief_md(args.campaign, rules, brief)
    manifest = build_manifest(args.campaign, resources, failures, harvested, other_urls)
    C.save_json(C.CAMPAIGN_MANIFEST, manifest)
    write_knowledge_md(args.campaign, rules, resources, harvested, other_urls, len(corpus), llm_used)
    # Human-facing do-this-when-posting checklist (posting is always manual) — real requirements
    # pulled from rules.json, ambiguities flagged as VERIFY, never invented.
    write_posting_checklist(args.campaign, rules, brief)
    C.log(f"wrote {C.POSTING_CHECKLIST.name} — posting/submission checklist for this campaign.")

    state = C.load_state()
    # Scope checkpoints per-campaign: if this is a NEW campaign, stash the prior campaign's
    # stages + archive its drafts so downstream stages run fresh (no --force) and old clips
    # don't mix in. No-op when re-intaking the SAME campaign (keeps its checkpoints).
    C.activate_campaign(state, args.campaign)
    C.mark_stage(state, "intake", footage_hours=manifest["footage_total_hours"],
                 banned_words=len(rules.get("banned_words", [])), resources=len(resources))

    print_coverage(args.campaign, rules, resources, manifest, failures, other_urls, hunt_path)

    # Footage reachability was already enforced (HARD STOP) right after the hunt — by here we have
    # at least one footage file. A gate we routed AROUND (free footage still found) is just noted.
    footage = [r for r in resources if r["kind"] == "footage"]
    if barriers:
        cats = ", ".join(sorted({b for b, _ in barriers}))
        C.warn(f"note: {len(barriers)} gated route(s) ({cats}) were skipped, but "
               f"{len(footage)} footage file(s) were freely reachable — proceeding.")
    if failures:
        C.warn(f"{len(failures)} optional source(s) failed (see report) — continuing with "
               f"{len(footage)} footage file(s).")


if __name__ == "__main__":
    main()
