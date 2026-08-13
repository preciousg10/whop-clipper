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
    python scripts/pickcampaign.py --streamer-only      # rank + pick STREAMER/IRL campaigns only
    python scripts/pickcampaign.py --max-walk 20        # look deeper before giving up
    python scripts/pickcampaign.py --rank 3            # manual override: force a specific rank
    python scripts/pickcampaign.py --scout-dir D:/whop/scout
"""
import argparse
import json
import os
import re
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
PREEDITED_CACHE = INPUTS_DIR / "preedited_cache.json"   # sticky footage-length verdicts (Unit 1)

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


def rank_campaigns(campaigns, *, streamer_only=False):
    """Rankable = a scraped/refreshed, non-disqualified campaign that rests on at least one
    real signal (not UNKNOWN-only). Sorted by scout's composite, tie-broken toward the
    better-understood campaign then pre_score — identical to scout's own report ordering.

    streamer_only: narrow to the STREAMER/IRL handoff set (see `streamer_mode_class`) and
    order by the penalty-adjusted composite. Scout's full board (campaigns.json) is only READ
    here — never modified — so this changes ONLY what gets ranked + handed to the clipper."""
    rankable = [
        c for c in campaigns
        if c.get("status") in RANKABLE_STATUSES
        and not c.get("disqualified")
        and not c.get("rules_unreadable")   # scout excluded it: rules only in an unreadable source
        and not c.get("excluded_prohibited")  # scout excluded it: prohibited/vice category
        and _composite(c) > 0
        and _core_known(c) > 0              # skip UNKNOWN-only (ranked on neutrals alone)
    ]
    if streamer_only:
        rankable = [c for c in rankable if streamer_mode_class(c)[0]]
        # Order by the penalty-adjusted composite (sports-with-signal is demoted x0.6), then
        # the same tie-breaks. streamer_effective never mutates the stored composite.
        rankable.sort(key=lambda c: (streamer_effective(c), _core_known(c), _pre_score(c)),
                      reverse=True)
        return rankable
    rankable.sort(key=lambda c: (_composite(c), _core_known(c), _pre_score(c)), reverse=True)
    return rankable


# --- STREAMER/IRL-only handoff mode (--streamer-only) --------------------------
# Narrows the ranked/handed-off set to STREAMER/IRL content only, WITHOUT touching scout's
# full board (campaigns.json is read-only here). Membership reuses the categories scout
# already assigned (its Groq categorizer) PLUS a plain keyword recheck for `sports` —
# deliberately NO LLM call and NO re-categorization (that would risk scout's free-tier quota).
#   - tagged streamer_irl             -> included at full priority.
#   - tagged sports + streamer signal -> included at LOWER priority (composite x0.6): a
#                                        streamer who also plays sports ranks below pure IRL.
#   - tagged sports, no signal         -> excluded (pure sports).
#   - anything else                   -> excluded.
STREAMER_SPORTS_PENALTY = 0.6   # sports-with-signal composite multiplier (lower priority)

# Generic streamer/IRL vocabulary (word-boundary matched, case-insensitive).
_STREAMER_KEYWORDS_RE = re.compile(
    r"\b(kick|twitch|stream|streamer|streaming|streamed|irl|vlog|vlogger|"
    r"subathon|just\s+chatting|live\s*stream)\b", re.I)
# Well-known streamer names/handles — a name match alone is a strong streamer signal. Edit
# this list to tune which streamers rescue a `sports`-tagged campaign into the mode.
_STREAMER_NAMES_RE = re.compile(
    r"\b(kai\s+cenat|ishowspeed|adin\s+ross|xqc|jynxzi|caseoh|duke\s+dennis|agent00|"
    r"stable\s*ronaldo|ludwig|hasanabi|amouranth|sketch|nickmercs|n3on|yourrage|fanum)\b", re.I)


def _streamer_text(c):
    """Name + rules text used for the keyword recheck (rules_text + on-modal requirements —
    the same rules the brief is assembled from)."""
    return " ".join(str(x) for x in (
        c.get("name"), c.get("rules_text"), c.get("modal_requirements_text")) if x)


def streamer_signal(c):
    """True if the campaign's name/rules carry a streamer/IRL signal (plain keyword match —
    NO LLM). Used to rescue `sports`-tagged campaigns that are really streamer content."""
    t = _streamer_text(c)
    return bool(_STREAMER_KEYWORDS_RE.search(t) or _STREAMER_NAMES_RE.search(t))


def _cats(c):
    """Every category tag scout assigned (primary first, then secondaries), de-duped."""
    out = []
    for t in [c.get("category")] + list(c.get("categories") or []):
        if t and t not in out:
            out.append(t)
    return out


def streamer_mode_class(c):
    """(include, tier, factor, reason) for the streamer-only mode. `factor` multiplies the
    composite for RANKING only — the stored composite (full board) is never touched."""
    cats = _cats(c)
    if "streamer_irl" in cats:
        return True, "streamer_irl", 1.0, "tagged streamer_irl"
    if "sports" in cats:
        if streamer_signal(c):
            return (True, "sports+streamer", STREAMER_SPORTS_PENALTY,
                    "sports with streamer/IRL signal (lower priority)")
        return False, "sports", 0.0, "pure sports — no streamer/IRL signal"
    return False, (c.get("category") or "other"), 0.0, "not streamer_irl / sports"


def streamer_effective(c):
    """Composite used to ORDER campaigns within streamer-only mode (stored composite x the
    tier factor). Pure — never mutates the record."""
    return _composite(c) * streamer_mode_class(c)[2]


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


# --- pre-edited footage filter (Unit 1) ----------------------------------------
# Some campaigns' "footage" is a Drive folder of already-edited short vertical clips
# (15-45s each), not raw VODs — you can't clip moments out of an edited clip. We SKIP a
# campaign whose footage is ENTIRELY such shorts. Measurement is cheap + download-free:
# gdown lists the Drive folder (skip_download) and yt-dlp reads each file's duration from
# metadata (extract_info download=False). YouTube channels/playlists are raw VOD sources
# and PASS without probing. FAIL OPEN on anything we can't measure — better to let a
# campaign through than wrongly exclude it (--rank can always force one anyway).
DEFAULT_PREEDITED_MIN_SECONDS = 120      # a footage file >= this is a "real VOD" (config)
DEFAULT_PREEDITED_MAX_PROBE = 40         # cap files probed per Drive folder (perf bound)

_VIDEO_EXT_RE = re.compile(r"\.(mp4|mov|mkv|webm|m4v|avi|ts|flv|m2ts)$", re.I)
_NONVIDEO_EXT_RE = re.compile(
    r"\.(png|jpe?g|webp|gif|pdf|docx?|txt|md|markdown|csv|tsv|xlsx?|json|rtf|"
    r"gdoc|gsheet|gslides)$", re.I)


def _footage_kind(url):
    """Coarse type of a footage link for the pre-edited check."""
    low = (url or "").lower()
    if "youtube.com" in low or "youtu.be" in low:
        # A channel or playlist resolves to full VODs (raw source) — never pre-edited.
        if any(t in low for t in ("/@", "/channel/", "/c/", "/user/", "playlist", "list=")):
            return "raw_channel"
        return "vod"                                  # a single YouTube video
    if "drive.google.com" in low and ("/folders/" in low or "folderview" in low):
        return "drive_folder"
    if "drive.google.com" in low:
        return "drive_file"
    return "vod"                                      # kick / twitch / vimeo / direct


def _looks_video(name):
    """A Drive folder entry likely to be a video: a video extension, OR no clearly
    non-video extension (Drive frequently lists files without an extension)."""
    n = name or ""
    if _VIDEO_EXT_RE.search(n):
        return True
    if _NONVIDEO_EXT_RE.search(n):
        return False
    return True


class _QuietLogger:
    """Swallow yt-dlp's per-file error/warning chatter (a transient 503 on one probe is not
    worth printing — it's retried, and the aggregate decision reports what it saw)."""
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


def _probe_duration(url, timeout=20, attempts=3):
    """Duration in seconds from yt-dlp metadata (NO download). Retries on TRANSIENT errors
    (Drive 503s / timeouts) so a flaky request doesn't read as 'unknowable'. Returns None
    only when the metadata genuinely carries no duration, or every attempt failed."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return None
    import time as _t
    for a in range(attempts):
        try:
            with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                            "socket_timeout": timeout, "noplaylist": True,
                            "logger": _QuietLogger()}) as ydl:
                info = ydl.extract_info(url, download=False)
            d = info.get("duration")
            return float(d) if d is not None else None   # success (duration may be absent)
        except Exception:
            if a < attempts - 1:
                _t.sleep(1.0 * (a + 1))                   # brief backoff, then retry
    return None


def _list_drive_folder(url, attempts=3):
    """[(id, name), ...] for a Drive folder WITHOUT downloading (gdown skip_download), or
    None if it genuinely can't be listed. RETRIES on transient failure (rate-limited / flaky
    listing) with short backoff — a single hiccup must not read as 'unlistable' (that was the
    fail-open bug that let LETSGO through)."""
    try:
        import gdown
    except ImportError:
        return None
    import time as _t
    for a in range(attempts):
        try:
            files = gdown.download_folder(url=url, skip_download=True, quiet=True,
                                          use_cookies=False)
            out = [(getattr(f, "id", None), getattr(f, "path", None) or "") for f in (files or [])]
            out = [(fid, name) for fid, name in out if fid]
            if out:
                return out
            # Empty listing is suspicious (a shared folder has files) — treat as transient.
        except Exception:
            pass
        if a < attempts - 1:
            _t.sleep(1.5 * (a + 1))
    return None


def _short_name(s, n=34):
    s = str(s or "").strip()
    return (s[:n - 1] + "…") if len(s) > n else s


# --- measurement cache (fix: a transient measurement failure must NOT flip a known verdict) --
def _links_fingerprint(links):
    """Stable fingerprint of a campaign's footage links — the cache key alongside campaign id.
    If the links change, the fingerprint changes and we re-measure; if they don't, the cached
    verdict is reused (a campaign skipped as pre-edited STAYS skipped)."""
    return "|".join(sorted(str(u).strip() for u in (links or []) if str(u).strip()))


def _load_preedited_cache():
    try:
        return C.load_json(PREEDITED_CACHE) or {}
    except Exception:
        return {}


def _cache_put(cache, cid, fp, verdict, reason, name):
    cache[cid] = {"fingerprint": fp, "verdict": verdict, "reason": reason,
                  "name": name, "measured_at": C.now_iso()}
    try:
        PREEDITED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        C.save_json(PREEDITED_CACHE, cache)
    except Exception as e:
        C.warn(f"    pre-edited: could not persist measurement cache ({e}).")


def _measure_preedited(c, links, min_seconds, max_probe):
    """Actually measure the footage lengths (download-free). Returns (verdict, reason) where
    verdict is one of:
      'pass'         — a raw channel/playlist, OR a file >= min_seconds (a real VOD → clippable);
      'skip'         — measured a clear majority of files and EVERY one is short (pre-edited);
      'unmeasurable' — couldn't measure enough to decide (transient/unknown) → caller uses cache
                       or fails open. Distinct from 'skip'/'pass' so the caller can tell them apart."""
    to_probe, source_unknown, sampled_note = [], False, ""
    for url in links:
        kind = _footage_kind(url)
        if kind == "raw_channel":
            return "pass", "raw YouTube channel/playlist (full VODs — not pre-edited)"
        if kind == "drive_folder":
            entries = _list_drive_folder(url)
            if entries is None:
                source_unknown = True                 # couldn't list even after retries
                continue
            vids = [(f"https://drive.google.com/file/d/{fid}/view", name)
                    for fid, name in entries if _looks_video(name)]
            if not vids:
                source_unknown = True
                continue
            if len(vids) > max_probe:
                sampled_note = f" (sampled {max_probe} of {len(vids)})"
                vids = vids[:max_probe]
            to_probe.extend(vids)
        else:                                         # single vod / drive file
            to_probe.append((url, url))
    if not to_probe:
        return "unmeasurable", ("could not list any Drive footage folder (rate-limited/flaky)"
                                if source_unknown else "no measurable footage source")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda t: (t[1], _probe_duration(t[0])), to_probe))

    measured, unreadable = [], 0
    for name, d in results:
        if d is None:
            unreadable += 1
        elif d >= min_seconds:
            return "pass", f"a footage file is >= {min_seconds}s ({_short_name(name)}={d:.0f}s)"
        else:
            measured.append((name, d))
    # SKIP requires a clear majority measured, every one short, and NO whole source we failed to
    # list (that source could be a VOD). Otherwise it's genuinely unmeasurable → the caller
    # decides via cache / fail-open (NOT a silent skip).
    if measured and len(measured) >= max(1, unreadable) and not source_unknown:
        lens = ", ".join(f"{_short_name(n)}={d:.0f}s" for n, d in measured[:6])
        more = f", +{len(measured) - 6} more" if len(measured) > 6 else ""
        extra = f", {unreadable} unreadable" if unreadable else ""
        return "skip", (f"pre-edited footage — {len(measured)} measured file(s) < "
                        f"{min_seconds}s{sampled_note}{extra} [{lens}{more}]")
    bits = f"{len(measured)} measured short, {unreadable} unreadable"
    if source_unknown:
        bits += ", a folder was unlistable"
    return "unmeasurable", f"too little measured to decide ({bits})"


def preedited_footage_skip(c, min_seconds=DEFAULT_PREEDITED_MIN_SECONDS,
                           max_probe=DEFAULT_PREEDITED_MAX_PROBE, use_cache=True):
    """(skip, reason) — True only when the footage is ENTIRELY pre-edited short clips. Robust to
    flaky no-download measurements via a STICKY per-campaign cache (keyed by id + footage-links
    fingerprint): once measured, the verdict is reused as long as the links don't change, so a
    later transient failure can never flip a known 'skip' to a fail-open pass. Every call logs
    exactly ONE of: CACHED / MEASURED→SKIP / MEASURED→PASS / UNMEASURABLE(→cache|→fail-open) —
    it is never silent. Metadata only — never downloads."""
    links = footage_links(c)
    if not links:
        return False, None                            # 'no footage' is clippable()'s job
    fp = _links_fingerprint(links)
    cid = str(c.get("id") or c.get("name") or fp)
    cache = _load_preedited_cache() if use_cache else {}
    cached = cache.get(cid) if use_cache else None
    cached_ok = bool(cached and cached.get("fingerprint") == fp
                     and cached.get("verdict") in ("skip", "pass"))

    # STICKY REUSE (fix #4): same campaign + same links → reuse the prior verdict WITHOUT
    # re-measuring, so a transient failure can't sneak a pre-edited campaign through.
    if cached_ok:
        v = cached["verdict"]
        C.log(f"    pre-edited: [CACHED {v.upper()}] measured {str(cached.get('measured_at'))[:19]}, "
              f"links unchanged — {cached.get('reason')}")
        return (v == "skip"), cached.get("reason")

    # No usable cache (new campaign, or links changed) → measure now (with retries inside).
    verdict, reason = _measure_preedited(c, links, min_seconds, max_probe)
    if verdict in ("skip", "pass"):
        C.log(f"    pre-edited: [MEASURED → {verdict.upper()}] {reason}")
        if use_cache:
            _cache_put(cache, cid, fp, verdict, reason, c.get("name"))
        return (verdict == "skip"), (reason if verdict == "skip" else None)

    # UNMEASURABLE: reuse a stale-but-same-links cache if we somehow have one; else fail open.
    if cached and cached.get("fingerprint") == fp and cached.get("verdict") in ("skip", "pass"):
        v = cached["verdict"]
        C.warn(f"    pre-edited: [UNMEASURABLE now → reusing CACHED {v.upper()}] {reason}")
        return (v == "skip"), cached.get("reason")
    C.warn(f"    pre-edited: [UNMEASURABLE, no cache → FAIL OPEN, letting it through] {reason}")
    return False, None


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


def board_age_hours(scout_json):
    """Hours since scout GENERATED this board (top-level `generated_at`), or None if unknown.
    Lets the clipper detect a STALE board even though scout's once-daily 20h guard skips
    SILENTLY (exit 0): if scout didn't scrape today, generated_at is yesterday's."""
    try:
        data = json.loads(open(scout_json, encoding="utf-8").read())
    except Exception:
        return None
    ts = data.get("generated_at") if isinstance(data, dict) else None
    if not ts:
        return None
    try:
        import datetime
        gen = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return (datetime.datetime.now(datetime.timezone.utc) - gen).total_seconds() / 3600.0
    except Exception:
        return None


def warn_if_stale_board(scout_json, max_hours):
    """Loudly flag a stale scout board (Unit 2d) so the chained `whop.bat` run can NEVER pick
    from an old board without the user seeing it. Warns; does not stop (the pick may still be
    fine on a day-old board) — but it is unmistakable."""
    age = board_age_hours(scout_json)
    if age is None:
        C.warn("scout board age UNKNOWN (no generated_at timestamp) — can't confirm it's fresh.")
        return
    if age >= max_hours:
        bar = "!" * 70
        C.warn(bar)
        C.warn(f"STALE SCOUT BOARD — campaigns.json was generated {age:.1f}h ago (>= {max_hours}h).")
        C.warn("Scout most likely SKIPPED its once-daily scrape (20h guard) and you are picking")
        C.warn("from an OLD board. For fresh data run scout with --force, then re-run the pick.")
        C.warn(bar)
    else:
        C.log(f"scout board age: {age:.1f}h (fresh, < {max_hours}h).")


def _commit_pick(pick, rank, scout_json, streamer_only=False):
    """Write the intake inputs (brief.txt, links.txt, pick.json) for the chosen campaign and
    print the summary + next steps. Only called AFTER `clippable` passed, so links is non-empty."""
    name = pick.get("name") or "(unnamed campaign)"
    locator, how = resolve_locator(pick)
    resources = _resource_links(pick)
    links = footage_links(pick)
    sm_tier = streamer_mode_class(pick)[1] if streamer_only else None

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
        "rank_mode": "streamer_only" if streamer_only else "full_board",
        "streamer_tier": sm_tier,
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
    if streamer_only:
        print(f"  mode          : STREAMER/IRL only  (tier: {sm_tier})")
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
                         "clippability preconditions, else fail loud). BYPASSES the pre-edited "
                         "footage filter, so you can force a campaign the walk would skip. "
                         "Default: walk from #1.")
    ap.add_argument("--streamer-only", action="store_true",
                    help="STREAMER/IRL-only handoff: rank + pick only streamer_irl campaigns "
                         "(plus sports campaigns that carry a streamer/IRL keyword signal, at "
                         "x0.6 lower priority). Reads scout's existing category tags — no "
                         "re-categorization. The full board in campaigns.json is untouched; "
                         "only what's ranked + handed to the clipper changes.")
    ap.add_argument("--preedited-min-seconds", type=int, default=DEFAULT_PREEDITED_MIN_SECONDS,
                    help=f"pre-edited filter: a footage file this long or longer counts as a "
                         f"real VOD (default {DEFAULT_PREEDITED_MIN_SECONDS}s). A campaign whose "
                         f"footage is ENTIRELY shorter clips is skipped during the walk.")
    ap.add_argument("--preedited-max-probe", type=int, default=DEFAULT_PREEDITED_MAX_PROBE,
                    help=f"pre-edited filter: max files probed per Drive folder (default "
                         f"{DEFAULT_PREEDITED_MAX_PROBE}).")
    ap.add_argument("--no-preedited-filter", action="store_true",
                    help="disable the pre-edited-footage skip (walk exactly as before).")
    ap.add_argument("--preedited-refresh", action="store_true",
                    help="ignore the cached footage-length verdicts and re-measure from scratch "
                         "(otherwise a measured campaign reuses its sticky verdict).")
    ap.add_argument("--exclude-id", action="append", default=[], metavar="SCOUT_ID",
                    help="scout campaign id to SKIP during the walk (repeatable). Used by "
                         "run.py --auto-advance to move past a campaign that produced nothing.")
    ap.add_argument("--stale-board-hours", type=float, default=20.0,
                    help="warn LOUDLY if scout's campaigns.json is at least this many hours old "
                         "(default 20, matching scout's once-daily guard) — so a silently-skipped "
                         "scout scrape never picks from a stale board unnoticed.")
    args = ap.parse_args()

    scout_json = args.scout_json or os.path.join(args.scout_dir, "campaigns.json")
    scout_dir = os.path.dirname(scout_json) or "."
    campaigns = load_scout_campaigns(scout_json)
    warn_if_stale_board(scout_json, args.stale_board_hours)   # Unit 2d: never silent on a stale board
    ranked = rank_campaigns(campaigns, streamer_only=args.streamer_only)
    if not ranked:
        if args.streamer_only:
            C.fail("no STREAMER/IRL campaigns in scout's ranked output (no streamer_irl tags, "
                   "and no sports campaign carried a streamer/IRL keyword signal). Re-run scout, "
                   "or drop --streamer-only to walk the full board.")
        C.fail("no rankable campaigns in scout's output (all disqualified, rules-unreadable, "
               "UNKNOWN-only, or zero composite). Nothing to clip — re-run scout.")
    done_ids = _load_done_ids(scout_dir)

    # Show the shortlist so the pick is transparent.
    mode_txt = " [STREAMER/IRL only]" if args.streamer_only else ""
    C.log(f"scout ranked {len(ranked)} candidate(s){mode_txt} (from {scout_json}); "
          f"walking the top {min(args.max_walk, len(ranked))} for the first clippable one:")
    for i, c in enumerate(ranked[:max(args.max_walk, 8)], 1):
        extra = ""
        if args.streamer_only:
            _, tier, factor, _reason = streamer_mode_class(c)
            extra = f"  [{tier}{'' if factor == 1.0 else ' x%.2f' % factor}]"
        C.log(f"    #{i}  comp {_composite(c):.4f}{extra}  {_core_known(c)}/5 known  "
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
        _commit_pick(pick, args.rank, scout_json, streamer_only=args.streamer_only)
        return

    # Walk the top N; take the FIRST that passes all preconditions, logging every skip + reason.
    cap = max(1, args.max_walk)
    walked = ranked[:cap]
    skips = []
    exclude = set(args.exclude_id or [])
    for i, c in enumerate(walked, 1):
        if c.get("id") in exclude:
            skips.append((i, c, "excluded (--exclude-id; auto-advance skip)"))
            continue
        ok, why = clippable(c, done_ids)
        # Pre-edited footage filter (Unit 1): only AFTER the cheap checks pass (it probes the
        # network for durations, so we never run it on a campaign that already failed).
        if ok and not args.no_preedited_filter:
            C.log(f"    #{i} {c.get('name')!r}: checking footage length (no download)…")
            skip_pe, why_pe = preedited_footage_skip(
                c, min_seconds=args.preedited_min_seconds, max_probe=args.preedited_max_probe,
                use_cache=not args.preedited_refresh)
            if skip_pe:
                ok, why = False, why_pe
        if ok:
            for sr, sc, sw in skips:
                C.log(f"  skip #{sr}  {sc.get('name')!r} — {sw}")
            C.log(f"  -> clippable at #{i}: {c.get('name')!r}")
            _commit_pick(c, i, scout_json, streamer_only=args.streamer_only)
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
