"""Scout board freshness gate — decide whether whop.bat should re-scrape.

Reads scout's campaigns.json top-level `generated_at` timestamp and compares its age to a
threshold (default 72h / 3 days). Purely stdlib + read-only — it never scrapes or writes.

Exit codes (so a .bat can branch on it):
    0  -> board is FRESH  (age < threshold)            -> caller SKIPS the scout scrape
    1  -> board is STALE  (age >= threshold)           -> caller RUNS scout --force
    2  -> board MISSING / unreadable / no timestamp    -> caller RUNS scout --force (treat as stale)

    python scripts/board_age.py                        # default path + 72h
    python scripts/board_age.py --hours 48
    python scripts/board_age.py --path D:/whop/scout/campaigns.json --hours 72
"""
import argparse
import datetime
import json
import sys

DEFAULT_PATH = r"C:\whop\scout\campaigns.json"
DEFAULT_HOURS = 72.0


def board_age_hours(path):
    """Hours since the board's `generated_at`, or None if the file is missing/unreadable or
    carries no usable timestamp. Never raises."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    ts = data.get("generated_at") if isinstance(data, dict) else None
    if not ts:
        return None
    try:
        gen = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if gen.tzinfo is None:
            gen = gen.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - gen).total_seconds() / 3600.0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Scout board freshness gate for whop.bat.")
    ap.add_argument("--path", default=DEFAULT_PATH,
                    help=f"path to scout campaigns.json (default {DEFAULT_PATH})")
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                    help=f"staleness threshold in hours (default {DEFAULT_HOURS:.0f} = 3 days)")
    args = ap.parse_args()

    age = board_age_hours(args.path)
    if age is None:
        print(f"[board_age] board missing/unreadable/no timestamp at {args.path} -> STALE")
        sys.exit(2)
    if age < args.hours:
        print(f"[board_age] board age {age:.1f}h < {args.hours:.0f}h -> FRESH")
        sys.exit(0)
    print(f"[board_age] board age {age:.1f}h >= {args.hours:.0f}h -> STALE")
    sys.exit(1)


if __name__ == "__main__":
    main()
