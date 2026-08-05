"""STAGE -1 — PICK CAMPAIGN. The Scout -> Clipper handoff.

Scout ranks campaigns and writes them to scout/campaigns.json. This module reads that
ranking, selects the #1 rankable campaign (by Scout's existing composite score), resolves
it to something intake can actually run on, and writes the brief + footage links into
campaign_inputs/ so `intake.py --from-pick` can take over. It is the front of the chain:

    scout (rank) -> pickcampaign (select #1 + write inputs) -> intake (--from-pick) -> run

Fail-loud discipline (instructions.md): we NEVER silently proceed on a campaign we can't
clip. A #1 pick with no footage links and no locator is reported and STOPS the chain — we
do not skip to #2 (that would hide a ranking/scout problem), and we do not fabricate a
source. Live re-scraping of Whop belongs to scout (it owns the logged-in browser); this
module only reads what scout already captured and, when it can't, tells the user exactly
what to open.

    python scripts/pickcampaign.py                     # pick #1 from scout/campaigns.json
    python scripts/pickcampaign.py --rank 2            # manual override: take the Nth pick
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
        and _composite(c) > 0
        and _core_known(c) > 0            # skip UNKNOWN-only (ranked on neutrals alone)
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
    """(locator_url, how) for a campaign. Priority: the URL scout captured, else a URL
    built from the captured campaign_id, else the name-search URL. `how` records which tier
    fired so the report and pick.json are honest about provenance. Never raises."""
    url = c.get("url")
    if url:
        return url, "stored url"
    cid = c.get("campaign_id")
    if cid:
        # apps.whop.com campaign ids look like app_XXXX; reconstruct the discover app URL.
        return f"https://whop.com/discover/app/{cid}/", "built from campaign_id"
    return whop_search_url(c.get("name")), "name-search (deferred — open manually)"


# --- brief assembly ------------------------------------------------------------
def build_brief(c, locator, how):
    """The brief SOURCE intake will parse: scout's scraped rules text, headed by the
    campaign facts scout already knows (name, pay, budget, locator). Empty rules are NOT
    fatal here — intake applies its deterministic banned-word floor and flags the thin
    brief as an ambiguity — but no-footage IS fatal (handled in main)."""
    name = c.get("name") or "(unnamed campaign)"
    pay = c.get("pay_per_1k")
    pay_txt = f"${pay:.2f} per 1K views" if pay is not None else "unknown"
    rules = (c.get("rules_text") or "").strip()
    lines = [
        f"# Campaign: {name}",
        f"# Source: Scout rank handoff (composite {_composite(c):.4f}, "
        f"{_core_known(c)}/5 core signals known)",
        f"# Locator: {locator}  ({how})",
        f"# Campaign id: {c.get('campaign_id') or 'unknown'}",
        f"# Pay: {pay_txt}",
        "",
        "## Rules / brief (as scraped by scout)",
        rules if rules else "(scout captured no rules text for this campaign — verify "
                            "the brief manually before producing; banned-word compliance "
                            "depends on it.)",
    ]
    return "\n".join(lines) + "\n"


def footage_links(c):
    """The footage/asset links intake downloads. Scout stores them per campaign as
    source_links (Drive folders, YouTube/Kick VODs). De-duped, order preserved."""
    seen, out = set(), []
    for u in (c.get("source_links") or []):
        u = (u or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


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


def main():
    ap = argparse.ArgumentParser(description="Stage -1 — pick scout's #1 campaign for the clipper.")
    ap.add_argument("--scout-dir", default=DEFAULT_SCOUT_DIR,
                    help=f"scout project dir (default {DEFAULT_SCOUT_DIR})")
    ap.add_argument("--scout-json", help="explicit path to scout's campaigns.json "
                    "(overrides --scout-dir)")
    ap.add_argument("--rank", type=int, default=1,
                    help="1-based rank to pick (default 1 = scout's #1). A manual override; "
                         "the default takes the top-ranked campaign.")
    args = ap.parse_args()

    scout_json = args.scout_json or os.path.join(args.scout_dir, "campaigns.json")
    campaigns = load_scout_campaigns(scout_json)
    ranked = rank_campaigns(campaigns)
    if not ranked:
        C.fail("no rankable campaigns in scout's output (all disqualified, UNKNOWN-only, or "
               "zero composite). Nothing to clip — re-run scout.")

    # Show the shortlist so the pick is transparent and a manual --rank is easy.
    C.log(f"scout ranked {len(ranked)} clippable candidate(s) (from {scout_json}):")
    for i, c in enumerate(ranked[:8], 1):
        marker = "->" if i == args.rank else "  "
        C.log(f"  {marker} #{i}  comp {_composite(c):.4f}  {_core_known(c)}/5 known  "
              f"{len(footage_links(c))} link(s)  {c.get('name')!r}")

    if args.rank < 1 or args.rank > len(ranked):
        C.fail(f"--rank {args.rank} is out of range (1..{len(ranked)}).")
    pick = ranked[args.rank - 1]

    name = pick.get("name") or "(unnamed campaign)"
    locator, how = resolve_locator(pick)
    links = footage_links(pick)

    # FAIL LOUD: strict #1 (or chosen rank) with nothing to clip. We do NOT skip to the next
    # campaign (that hides a scout/ranking gap) and we do NOT scrape Whop from here (scout
    # owns the browser). Tell the user exactly which campaign and where to look.
    if not links:
        C.fail(
            f"picked #{args.rank} '{name}' but scout stored NO footage links for it "
            f"(source_links is empty), so there is nothing to download and clip.\n\n"
            f"  Locator ({how}): {locator}\n"
            f"  Campaign id     : {pick.get('campaign_id') or 'not captured'}\n"
            f"  locator_missing : {pick.get('locator_missing')}\n\n"
            "This campaign cannot be clipped without its footage links. Options:\n"
            "  1. Open the locator above and confirm it really has no Drive/VOD footage.\n"
            "  2. Re-run scout so it re-opens the campaign and captures source_links + url.\n"
            "  3. Mark it done in scout (python scout.py --mark-done <id>) to drop it from the\n"
            "     board, then re-run this picker for the next campaign.\n"
            f"  (scout id: {pick.get('id')})")

    # Resolved + clippable: write the intake inputs.
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    BRIEF_TXT.write_text(build_brief(pick, locator, how), encoding="utf-8")
    LINKS_TXT.write_text("\n".join(links) + "\n", encoding="utf-8")
    meta = {
        "campaign": name,
        "scout_id": pick.get("id"),
        "campaign_id": pick.get("campaign_id"),
        "url": pick.get("url"),
        "locator": locator,
        "locator_how": how,
        "locator_missing": bool(pick.get("locator_missing")),
        "rank": args.rank,
        "composite_score": _composite(pick),
        "core_signals_known": _core_known(pick),
        "footage_links": links,
        "brief": os.path.relpath(BRIEF_TXT, C.ROOT),
        "links": os.path.relpath(LINKS_TXT, C.ROOT),
        "scout_json": scout_json,
        "picked_at": C.now_iso(),
    }
    C.save_json(PICK_JSON, meta)

    print("\n" + "=" * 66)
    print(f"PICKED #{args.rank}: {name}")
    print("=" * 66)
    print(f"  composite     : {_composite(pick):.4f}  ({_core_known(pick)}/5 core signals known)")
    print(f"  locator       : {locator}  ({how})")
    print(f"  footage links : {len(links)}")
    for u in links:
        print(f"      - {u}")
    print(f"\n  wrote {os.path.relpath(BRIEF_TXT, C.ROOT)}, "
          f"{os.path.relpath(LINKS_TXT, C.ROOT)}, {os.path.relpath(PICK_JSON, C.ROOT)}")
    print("\n  NEXT:")
    print("    python scripts/intake.py --from-pick")
    print("    python scripts/run.py")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
