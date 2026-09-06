"""PICK + INTAKE with auto-advance — the hands-off front half of the chain.

whop.bat's old `pickcampaign && intake && run` chain DIED on the first intake failure
(dead links / offline Kick VOD / no footage — like Zlamdunk's offline stream): a bare
`if errorlevel 1 goto :end` stopped the whole unattended run. This orchestrator instead
picks scout's #1, runs intake, and if intake FAILS it excludes that campaign and advances
to the NEXT ranked one — up to --max-advance times — before giving up. Only when a campaign
intakes successfully do we hand off to run.py.

Mirrors run.py's own NothingUsable auto-advance (_advance_to_next_campaign), but for the
BEFORE-run.py stage (pick + intake), which run.py can't reach because it needs intake done
first. Accumulates excluded scout ids across attempts so a second failure can't re-pick a
campaign that already failed.

Exit 0 = a campaign is picked + intaken and ready for run.py.
Exit 1 = no campaign could be prepared (board exhausted or advance limit hit).

    python scripts/prepare_campaign.py --category podcast --max-advance 2
    python scripts/prepare_campaign.py --streamer-only --max-advance 2   # alias for --category streamer
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PICK_JSON = C.ROOT / "campaign_inputs" / "pick.json"


def _run(cmd):
    """Run a subprocess with the SAME interpreter, inheriting stdio. Returns exit code."""
    return subprocess.run([sys.executable] + cmd).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", default=None, metavar="LANE",
                    help="PREFERRED narrow — pass through to pickcampaign: rank the walk to "
                         "campaigns scout tagged with this AUDIENCE-LANE (a topic/audience people "
                         "follow, e.g. ENTERTAINMENT_STREAMER, GAMING, SPORTS, MONEY, HEALTH — or a "
                         "short form). A campaign can be in several lanes. Composes with --category.")
    ap.add_argument("--category", default=None, metavar="CAT",
                    help="pass through to pickcampaign: narrow the walk to ONE scout category "
                         "(e.g. podcast, streamer, gaming, sports, music, brand, meme, news, "
                         "movie — or an exact scout tag). --lane is the newer preferred narrow.")
    ap.add_argument("--streamer-only", action="store_true",
                    help="pass through to pickcampaign (alias for --category streamer)")
    ap.add_argument("--scout-json", help="explicit scout campaigns.json (pass-through)")
    ap.add_argument("--scout-dir", help="scout directory (pass-through)")
    ap.add_argument("--max-advance", type=int, default=2,
                    help="max campaigns to advance THROUGH on intake failure (default 2 → up to "
                         "3 intake attempts)")
    args = ap.parse_args()

    scripts = os.path.dirname(os.path.abspath(__file__))
    pickcampaign = os.path.join(scripts, "pickcampaign.py")
    intake = os.path.join(scripts, "intake.py")

    def pick_cmd(excludes):
        cmd = [pickcampaign]
        if args.lane:
            cmd += ["--lane", args.lane]
        if args.category:
            cmd += ["--category", args.category]
        if args.streamer_only:
            cmd.append("--streamer-only")
        if args.scout_json:
            cmd += ["--scout-json", args.scout_json]
        if args.scout_dir:
            cmd += ["--scout-dir", args.scout_dir]
        for eid in excludes:
            if eid:
                cmd += ["--exclude-id", str(eid)]
        return cmd

    excludes = []
    # attempt 0 = the #1 pick; each later attempt advances past a failed intake.
    for attempt in range(args.max_advance + 1):
        if _run(pick_cmd(excludes)) != 0:
            C.warn("prepare: pickcampaign found no further clippable campaign "
                   f"(excluded {len(excludes)}). Nothing to prepare.")
            return 1
        pick = C.load_json(PICK_JSON) or {}
        cur_id, cur_name = pick.get("scout_id"), pick.get("campaign")
        C.log(f"prepare: attempt {attempt + 1}/{args.max_advance + 1} — intake for "
              f"{cur_name!r} (scout id {cur_id}).")

        if _run([intake, "--from-pick"]) == 0:
            C.log(f"prepare: intake succeeded for {cur_name!r} — ready for run.py.")
            return 0

        # Intake failed (dead links / no footage). Exclude this campaign and advance.
        bar = "=" * 70
        C.warn(bar)
        C.warn(f"prepare: INTAKE FAILED for {cur_name!r} (scout id {cur_id}).")
        if attempt >= args.max_advance:
            C.warn(f"  advance limit reached ({args.max_advance}) — giving up.")
            C.warn(bar)
            return 1
        if cur_id and cur_id not in excludes:
            excludes.append(cur_id)
        C.warn(f"  advancing to the next ranked campaign (excluding {len(excludes)} failed).")
        C.warn(bar)

    return 1


if __name__ == "__main__":
    sys.exit(main())
