"""STAGE -1 — PICK CAMPAIGN. The Scout -> Clipper handoff.

Scout ranks campaigns and writes them to scout/campaigns.json. This module reads that
ranking, WALKS it top-down (highest composite first), and picks the FIRST campaign that
passes the clippability preconditions — writing its brief + footage links into
campaign_inputs/ so `intake.py --from-pick` can take over. It is the front of the chain:

    scout (rank) -> pickcampaign (walk -> first clippable + write inputs) -> intake -> run

Preconditions (dry, no downloads — we check for the PRESENCE of a footage link, we never
fetch it here; the real download happens in intake): a campaign must (1) not be on scout's
done list, (2) have readable rules (banned-word compliance depends on it — rules_unreadable
campaigns are skipped), and (3) have at least one footage link (Drive folder / VOD / video).

Fail-loud discipline (instructions.md): we never silently clip a campaign we can't verify.
Every skipped campaign is printed with its specific reason. A SAFETY CAP (--max-walk, default
10) bounds the walk: if none of the top N are clippable we STOP and fail loud with the full
skip list — better than churning through 400. Live re-scraping of Whop belongs to scout (it
owns the browser); this module only READS scout's campaigns.json, never writes to it.

    python scripts/pickcampaign.py                     # walk from #1, pick first clippable
    python scripts/pickcampaign.py --max-walk 20        # look deeper before giving up
    python scripts/pickcampaign.py --rank 3            # manual override: force a specific rank
    python scripts/pickcampaign.py --scout-dir D:/whop/scout
"""
import argparse
import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# Scout lives in a sibling project. Default to the known layout; override with --scout-dir.
DEFAULT_SCOUT_DIR = r"C:\whop\scout"
INPUTS_DIR = C.ROOT / "campaign_inputs"
BRIEF_TXT = INPUTS_DIR / "brief.txt"
LINKS_TXT = INPUTS_DIR / "links.txt"
PICK_JSON = INPUTS_DIR / "pick.json"

RANKABLE_STATUSES = ("scraped", "refreshed")
DEFAULT_MAX_WALK = 10        # safety cap: only walk the top N ranked campaigns looking for a
                             # clippable one; if none pass, fail loud rather than churn 400.


# --- ranking (mirrors scout/report.py sort so the clipper and scout agree on "#1") -----
def _composite(c):
    return c.get("composite_score") or 0


def _core_known(c):
    """How many of scout's 5 CORE signals were actually measured (not neutral defaults).
    A campaign with 0 here ranked purely on UNKNOWN neutrals — a gamble scout itself
    discounts — so we treat it as UNKNOWN-only and skip it."""
    return (c.get("composite_breakdown") or {}).get("core_signals_known") or 0


def _pre_score(c):
    return c.get("pre_score") or 0


def rank_campaigns(campaigns):
    """Rankable = a scraped/refreshed, non-disqualified campaign that rests on at least one
    real signal (not UNKNOWN-only). Sorted by scout's composite, tie-broken toward the
    better-understood campaign then pre_score — identical to scout's own report ordering."""
    rankable = [
        c for c in campaigns
        if c.get("status") in RANKABLE_STATUSES
        and not c.get("disqualified")
        and not c.get("rules_unreadable")   # scout excluded it: rules only in an unreadable source
        and _composite(c) > 0
        and _core_known(c) > 0              # skip UNKNOWN-only (ranked on neutrals alone)
    ]
    rankable.sort(key=lambda c: (_composite(c), _core_known(c), _pre_score(c)), reverse=True)
    return rankable


# --- locator resolution --------------------------------------------------------
def whop_search_url(name):
    """A human-openable Whop search URL for a campaign name. This is the deferred
    'search Whop by name' fallback: rather than driving a browser we can't verify
    offline, we hand the user the exact search page to open (or to feed back to scout)."""
    q = urllib.parse.quote((name or "").strip())
    return f"https://whop.com/discover/search/?query={q}" if q else "https://whop.com/discover/"


def resolve_locator(c):
    """(locator_url, how) for a campaign — provenance only. Scout confirmed the campaign_id
    (a UUID) is an IDENTIFIER, not a resolvable URL (the whop.com/<app>/<UUID> form lands on
    Discover), so we do NOT build a link from it. Use scout's stored url if it ever captured a
    real one, else the human-openable Whop search URL. The campaign is actually located via its
    resource_links + on-modal rules, not this url. Never raises."""
    url = c.get("url")
    if url:
        return url, "stored url"
    return whop_search_url(c.get("name")), "name-search (deferred — open manually)"


def _resource_links(c):
    """Scout's captured modal resource anchors [{url, label}], deduped by url. Rules docs
    (Brief/Requirements/Guidelines) AND footage folders (Content) both live here."""
    seen, out = set(), []
    for r in (c.get("resource_links") or []):
        u = (r.get("url") or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append({"url": u, "label": (r.get("label") or "link").strip()})
    return out


def _is_footage_link(url):
    """A link intake should download as footage: a Drive FOLDER or a VOD host. A Google
    *document* (rules doc) is NOT footage — it goes in the brief text to be read, not
    downloaded."""
    u = (url or "").lower()
    if "drive.google.com" in u and ("/folders/" in u or "folderview" in u):
        return True
    return any(h in u for h in ("youtube.com", "youtu.be", "kick.com", "twitch.tv", "vimeo.com"))


# --- brief assembly ------------------------------------------------------------
def _fmt_stats(stats):
    if not stats:
        return None
    parts = []
    if stats.get("pay_rate_text") or stats.get("pay_per_1k") is not None:
        parts.append(f"pay {stats.get('pay_rate_text') or ('$%.2f/1K' % stats['pay_per_1k'])}")
    if stats.get("budget_total") is not None:
        rem = stats.get("budget_remaining_fraction")
        rem_txt = f", {rem * 100:.0f}% remaining" if rem is not None else ""
        parts.append(f"budget ${stats.get('budget_paid') or 0:,.0f}/${stats['budget_total']:,.0f}{rem_txt}")
    if stats.get("min_payout") is not None:
        parts.append(f"min payout ${stats['min_payout']:,.2f}")
    if stats.get("max_per_video") is not None:
        parts.append(f"max/video ${stats['max_per_video']:,.0f}")
    return "; ".join(parts) or None


def build_brief(c, locator, how, resources):
    """The brief SOURCE intake will parse. Rules live in DIFFERENT places per campaign, so we
    hand intake ALL of them: (1) scout's on-modal requirements text + scraped rules_text (the
    on-page rules), and (2) every resource link with its label — intake harvests the doc/Notion
    URLs from this text and reads them, so it gets rules from BOTH places. Plus the on-modal
    stats. Empty rules are NOT fatal here (intake applies its banned-word floor + flags the
    ambiguity); truly-nothing-to-clip IS fatal (handled in main)."""
    name = c.get("name") or "(unnamed campaign)"
    modal = (c.get("modal_requirements_text") or "").strip()
    rules = (c.get("rules_text") or "").strip()
    stats_txt = _fmt_stats(c.get("modal_stats"))
    lines = [
        f"# Campaign: {name}",
        f"# Source: Scout rank handoff (composite {_composite(c):.4f}, "
        f"{_core_known(c)}/5 core signals known)",
        f"# Locator: {locator}  ({how})",
        f"# Campaign id (identifier, not a URL): {c.get('campaign_id') or 'unknown'}",
        f"# On-modal stats: {stats_txt or 'none captured'}",
        "",
        "## On-modal requirements / guidelines (scraped from the campaign page)",
        modal if modal else "(none on the modal — see linked resources / rules_text below)",
        "",
        "## Rules text (scout detail extraction)",
        rules if rules else "(none)",
        "",
        "## Resource links (rules docs + footage — intake reads/downloads these)",
    ]
    if resources:
        for r in resources:
            kind = "footage" if _is_footage_link(r["url"]) else "rules doc"
            lines.append(f"- {r['label']} ({kind}): {r['url']}")
    else:
        lines.append("(none captured)")
    if not modal and not rules and not resources:
        lines += ["", "> ⚠ No rules found on the modal, in rules_text, or in any linked doc — "
                  "verify the brief manually before producing; banned-word compliance depends on it."]
    return "\n".join(lines) + "\n"


def footage_links(c):
    """The footage links intake downloads: scout's source_links UNION the resource_links that
    are footage (Drive folders / VODs) — so a campaign whose footage is only in a labelled
    'Content' folder (e.g. DoorDash) is still clippable. De-duped, order preserved. Rules-doc
    links are deliberately excluded here (they go in the brief text to be read, not downloaded)."""
    seen, out = set(), []
    for u in (c.get("source_links") or []):
        u = (u or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    for r in _resource_links(c):
        u = r["url"]
        if u not in seen and _is_footage_link(u):
            seen.add(u)
            out.append(u)
    return out


# --- clippability preconditions (dry — presence checks only, no downloads) -----
def _load_done_ids(scout_dir):
    """Scout's campaign-level DONE list (completed_campaigns.json in the scout dir). Scout
    already marks these status='completed' so they usually never reach the ranked set, but we
    also honor the list directly in case campaigns.json wasn't regenerated after a --mark-done.
    Accepts {"completed": [...]} or a bare list; ids may be strings or {"id": ...}. Never raises."""
    p = os.path.join(scout_dir, "completed_campaigns.json")
    if not os.path.exists(p):
        return set()
    try:
        data = json.loads(open(p, encoding="utf-8").read())
        items = data.get("completed", []) if isinstance(data, dict) else data
        return {(x.get("id") if isinstance(x, dict) else x) for x in items if x}
    except Exception:
        return set()


def _rules_readable(c):
    """(readable, reason). Scout resolves rules_source per campaign; we must not clip one whose
    banned-word list is unknown. rules_unreadable is a hard skip. Old records without the field
    fall back to whether ANY rules text/doc was captured."""
    if c.get("rules_unreadable"):
        return False, "rules unreadable (rules only in a Notion page scout couldn't read)"
    src = c.get("rules_source")
    if src in ("modal", "gdoc", "notion"):
        return True, None
    if src in ("unreadable",):
        return False, "rules unreadable"
    if src == "none":
        return False, "no readable rules found on the campaign"
    # Older scout data without rules_source: allow only if some rules content was captured.
    if c.get("modal_requirements_text") or (c.get("rules_text") or "").strip() or c.get("resource_links"):
        return True, None
    return False, "no rules captured (can't verify banned words)"


def clippable(c, done_ids):
    """(ok, reason) — dry preconditions a campaign must pass to be worth committing to, checked
    WITHOUT downloading anything (presence of a footage link, not a fetch of it):
      1. not on the done/posted list,
      2. rules are readable (banned-word compliance depends on it),
      3. has at least one footage link (Drive folder / VOD / video source).
    Returns the FIRST failing reason so the skip line is specific."""
    if c.get("id") in done_ids:
        return False, "already done (on scout's completed list)"
    ok, why = _rules_readable(c)
    if not ok:
        return False, why
    if not footage_links(c):
        return False, "no footage link (no Drive folder / VOD / video source)"
    return True, None


# --- main ----------------------------------------------------------------------
def load_scout_campaigns(scout_json):
    if not os.path.exists(scout_json):
        C.fail(f"scout campaigns file not found: {scout_json}\n"
               "Run scout first (python scout.py) or pass --scout-dir / --scout-json.")
    try:
        data = json.loads(open(scout_json, encoding="utf-8").read())
    except Exception as e:
        C.fail(f"could not read scout campaigns json ({scout_json}): {e}")
    campaigns = data.get("campaigns") if isinstance(data, dict) else data
    if not campaigns:
        C.fail(f"scout campaigns file has no campaigns: {scout_json}")
    return campaigns


def _commit_pick(pick, rank, scout_json):
    """Write the intake inputs (brief.txt, links.txt, pick.json) for the chosen campaign and
    print the summary + next steps. Only called AFTER `clippable` passed, so links is non-empty."""
    name = pick.get("name") or "(unnamed campaign)"
    locator, how = resolve_locator(pick)
    resources = _resource_links(pick)
    links = footage_links(pick)

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    BRIEF_TXT.write_text(build_brief(pick, locator, how, resources), encoding="utf-8")
    LINKS_TXT.write_text("\n".join(links) + "\n", encoding="utf-8")
    meta = {
        "campaign": name,
        "scout_id": pick.get("id"),
        "campaign_id": pick.get("campaign_id"),
        "url": pick.get("url"),
        "locator": locator,
        "locator_how": how,
        "locator_missing": bool(pick.get("locator_missing")),
        "rank": rank,
        "composite_score": _composite(pick),
        "core_signals_known": _core_known(pick),
        "rules_source": pick.get("rules_source"),
        "footage_links": links,
        "resource_links": resources,
        "modal_stats": pick.get("modal_stats"),
        "has_modal_requirements": bool(pick.get("modal_requirements_text")),
        "brief": os.path.relpath(BRIEF_TXT, C.ROOT),
        "links": os.path.relpath(LINKS_TXT, C.ROOT),
        "scout_json": scout_json,
        "picked_at": C.now_iso(),
    }
    C.save_json(PICK_JSON, meta)

    print("\n" + "=" * 66)
    print(f"PICKED #{rank}: {name}")
    print("=" * 66)
    print(f"  composite     : {_composite(pick):.4f}  ({_core_known(pick)}/5 core signals known)")
    print(f"  locator       : {locator}  ({how})")
    print(f"  rules source  : {pick.get('rules_source') or 'unknown'}")
    print(f"  on-modal stats: {_fmt_stats(pick.get('modal_stats')) or 'none'}")
    print(f"  on-modal rules: {'yes' if pick.get('modal_requirements_text') else 'no'}"
          f"  ({len(pick.get('modal_requirements_text') or '')} chars)")
    print(f"  resource links: {len(resources)}")
    for r in resources:
        kind = "footage" if _is_footage_link(r["url"]) else "rules doc"
        print(f"      - [{r['label']}] ({kind}) {r['url']}")
    print(f"  footage links : {len(links)}")
    for u in links:
        print(f"      - {u}")
    print(f"\n  wrote {os.path.relpath(BRIEF_TXT, C.ROOT)}, "
          f"{os.path.relpath(LINKS_TXT, C.ROOT)}, {os.path.relpath(PICK_JSON, C.ROOT)}")
    print("\n  NEXT:")
    print("    python scripts/intake.py --from-pick")
    print("    python scripts/run.py")
    print("=" * 66 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Stage -1 — walk scout's ranking and pick the first clippable campaign.")
    ap.add_argument("--scout-dir", default=DEFAULT_SCOUT_DIR,
                    help=f"scout project dir (default {DEFAULT_SCOUT_DIR})")
    ap.add_argument("--scout-json", help="explicit path to scout's campaigns.json "
                    "(overrides --scout-dir)")
    ap.add_argument("--max-walk", type=int, default=DEFAULT_MAX_WALK,
                    help=f"safety cap: only consider the top N ranked campaigns (default "
                         f"{DEFAULT_MAX_WALK}). If none of the top N are clippable, fail loud "
                         f"rather than descend the whole list.")
    ap.add_argument("--rank", type=int, default=None,
                    help="manual override: force this exact 1-based rank (must pass the "
                         "clippability preconditions, else fail loud). Default: walk from #1.")
    args = ap.parse_args()

    scout_json = args.scout_json or os.path.join(args.scout_dir, "campaigns.json")
    scout_dir = os.path.dirname(scout_json) or "."
    campaigns = load_scout_campaigns(scout_json)
    ranked = rank_campaigns(campaigns)
    if not ranked:
        C.fail("no rankable campaigns in scout's output (all disqualified, rules-unreadable, "
               "UNKNOWN-only, or zero composite). Nothing to clip — re-run scout.")
    done_ids = _load_done_ids(scout_dir)

    # Show the shortlist so the pick is transparent.
    C.log(f"scout ranked {len(ranked)} candidate(s) (from {scout_json}); "
          f"walking the top {min(args.max_walk, len(ranked))} for the first clippable one:")
    for i, c in enumerate(ranked[:max(args.max_walk, 8)], 1):
        C.log(f"    #{i}  comp {_composite(c):.4f}  {_core_known(c)}/5 known  "
              f"{len(footage_links(c))} footage-link(s)  {c.get('name')!r}")

    # Manual override: force a specific rank (old strict behavior for that one campaign).
    if args.rank is not None:
        if args.rank < 1 or args.rank > len(ranked):
            C.fail(f"--rank {args.rank} is out of range (1..{len(ranked)}).")
        pick = ranked[args.rank - 1]
        ok, why = clippable(pick, done_ids)
        if not ok:
            C.fail(f"--rank {args.rank} '{pick.get('name')}' is not clippable: {why}. "
                   "Drop --rank to walk to the first clippable campaign instead.")
        _commit_pick(pick, args.rank, scout_json)
        return

    # Walk the top N; take the FIRST that passes all preconditions, logging every skip + reason.
    cap = max(1, args.max_walk)
    walked = ranked[:cap]
    skips = []
    for i, c in enumerate(walked, 1):
        ok, why = clippable(c, done_ids)
        if ok:
            for sr, sc, sw in skips:
                C.log(f"  skip #{sr}  {sc.get('name')!r} — {sw}")
            C.log(f"  -> clippable at #{i}: {c.get('name')!r}")
            _commit_pick(c, i, scout_json)
            return
        skips.append((i, c, why))

    # None of the top N were clippable — STOP; do NOT descend the rest of the list.
    lines = [f"    #{r}  {sc.get('name')!r} — {sw}" for r, sc, sw in skips]
    remaining = len(ranked) - len(walked)
    C.fail(
        f"none of the top {len(walked)} ranked campaign(s) are clippable — stopping (not "
        f"descending the remaining {remaining}). Why each was skipped:\n"
        + "\n".join(lines) + "\n\n"
        "Fix a footage link (add the Drive/VOD source), confirm rules are readable, or raise "
        "--max-walk to look deeper. Re-run scout if its captures are stale.")


if __name__ == "__main__":
    main()
