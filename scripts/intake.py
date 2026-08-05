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
    if not any(r.get("type") == "watermark" for r in rules.get("required_elements", [])):
        amb.append("No watermark requirement detected — confirm whether one is mandatory.")
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
    ap.add_argument("--cookies-from-browser",
                    help="browser for cookies on gated VODs (chrome/edge/firefox) — Kick needs this")
    ap.add_argument("--max-source-height", type=int, default=720,
                    help="cap for Drive transcoded preview streams in px (default 720)")
    ap.add_argument("--original", action="store_true",
                    help="force raw original Drive files instead of preview streams")
    args = ap.parse_args()

    C.ensure_dirs()
    brief = read_brief(args)
    links = collect_links(args)
    downloaded = set()

    # Prior manifest lets us skip re-downloading sources whose files are unchanged.
    prior = C.load_json(C.CAMPAIGN_MANIFEST) or {}
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

    # Spatial constraints from reference images (flagged — never auto-guess margins).
    rules["spatial_constraints"] = [{
        "type": r["purpose"], "file": r["path"], "dims": r.get("dims"),
        "needs_confirmation": True,
        "note": "exact margins/positions not auto-derived — confirm from the image",
    } for r in resources if r.get("purpose")]

    rules["ambiguities"] = build_ambiguities(rules, resources, llm_used)

    C.save_json(C.RULES_JSON, rules)
    write_brief_md(args.campaign, rules, brief)
    manifest = build_manifest(args.campaign, resources, failures, harvested, other_urls)
    C.save_json(C.CAMPAIGN_MANIFEST, manifest)
    write_knowledge_md(args.campaign, rules, resources, harvested, other_urls, len(corpus), llm_used)

    state = C.load_state()
    state["campaign"] = args.campaign
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
