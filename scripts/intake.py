"""STAGE 0 — INTAKE as a thorough campaign analyst.

Instead of just filing downloads and keyword-scanning the brief, intake now:
  1. READS EVERYTHING — extracts text from every doc (pdf/docx/sheets/txt/md/gdoc),
     probes every video (duration/resolution/aspect), classifies reference images,
     and harvests URLs from all text (recursing ONE level to pull linked Drive/VODs).
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


def download_links(links, cookies, downloaded, max_source_height=720, original=False,
                   prior_by_source=None):
    prior_by_source = prior_by_source or {}
    resources, failures = [], []
    for url in links:
        if url in downloaded:
            continue
        downloaded.add(url)
        safe = looks_safe_margin(url)
        reused = _reuse_if_unchanged(url, prior_by_source)
        if reused is not None:
            C.log(f"unchanged — skipping re-download: {url} ({len(reused)} file(s) present)")
            resources.extend(reused)
            continue
        try:
            entries = DL.download_source(url, cookies_from_browser=cookies,
                                         max_source_height=max_source_height,
                                         original=original)
        except DL.DownloadError as e:
            C.warn(f"optional source failed — skipping and continuing: {url} — {e}")
            failures.append({"source": url, "error": str(e)})
            continue
        for e in entries:
            resources.append({"path": e["path"], "kind": e["kind"], "source": url,
                              "safe_margin": bool(safe) if e["kind"] == "footage" else False})
    return resources, failures


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


def print_coverage(campaign, rules, resources, manifest, failures, other_urls):
    foot = [r for r in resources if r["kind"] == "footage"]
    print("\n" + "=" * 66)
    print(f"INTAKE COVERAGE REPORT — {campaign}")
    print("=" * 66)
    print(f"Footage: {len(foot)} file(s), ~{manifest['footage_total_hours']} h")
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

    # Pass 1: download the given links, then read/probe each.
    C.log(f"downloading {len(links)} source(s)…")
    resources, failures = download_links(links, args.cookies_from_browser, downloaded,
                                         max_source_height=args.max_source_height,
                                         original=args.original,
                                         prior_by_source=prior_by_source)
    corpus_parts = [brief]
    for r in resources:
        t = process_resource(r)
        if t:
            corpus_parts.append(f"\n\n### {r['path']}\n{t}")

    # Harvest URLs from brief + all docs; recurse ONE level.
    harvested = AN.harvest_urls("\n".join(corpus_parts))
    for r in resources:
        harvested += r.get("urls_found", [])
    harvested = list(dict.fromkeys(harvested))
    media = [u for u in harvested if AN.classify_url(u) in ("drive_folder", "vod") and u not in downloaded]
    gdocs = [u for u in harvested if AN.classify_url(u) == "gdoc" and u not in downloaded]
    other_urls = [u for u in harvested if AN.classify_url(u) == "other"]

    if media:
        C.log(f"recursing one level: downloading {len(media)} linked media source(s)…")
        r2, f2 = download_links(media, args.cookies_from_browser, downloaded,
                                max_source_height=args.max_source_height,
                                original=args.original,
                                prior_by_source=prior_by_source)
        failures += f2
        for r in r2:
            t = process_resource(r)
            if t:
                corpus_parts.append(f"\n\n### {r['path']}\n{t}")
        resources += r2

    for u in gdocs:                       # fetch linked Google Docs/Sheets as text
        downloaded.add(u)
        text, method = AN.fetch_gdoc_text(u)
        dest = C.DOCS / ("gdoc_" + re.sub(r"\W+", "_", u)[-40:] + ".txt")
        res = {"path": os.path.relpath(dest, C.ROOT), "kind": "doc", "source": u,
               "usage": [], "notes": [], "urls_found": []}
        if text and text.strip():
            dest.write_text(text, encoding="utf-8")
            res["text_len"] = len(text)
            res["usage"].append(f"fetched linked Google file text ({len(text)} chars via {method})")
            corpus_parts.append(f"\n\n### {u}\n{text}")
        else:
            res["usage"].append(f"UNREAD linked Google file — {method}")
            res["notes"].append(method)
        resources.append(res)

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

    state = C.load_state()
    # Scope checkpoints per-campaign: if this is a NEW campaign, stash the prior campaign's
    # stages + archive its drafts so downstream stages run fresh (no --force) and old clips
    # don't mix in. No-op when re-intaking the SAME campaign (keeps its checkpoints).
    C.activate_campaign(state, args.campaign)
    C.mark_stage(state, "intake", footage_hours=manifest["footage_total_hours"],
                 banned_words=len(rules.get("banned_words", [])), resources=len(resources))

    print_coverage(args.campaign, rules, resources, manifest, failures, other_urls)

    footage = [r for r in resources if r["kind"] == "footage"]
    if not footage:
        C.fail("intake produced no footage — cannot produce clips. Check the links/log above.")
    if failures:
        C.warn(f"{len(failures)} optional source(s) failed (see report) — continuing with "
               f"{len(footage)} footage file(s).")


if __name__ == "__main__":
    main()
