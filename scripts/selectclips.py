"""SELECT stage — Groq first-pass moment scoring, then pick the top clips.

(Named selectclips.py, not select.py: a module named `select` would shadow Python's
stdlib `select`, which asyncio/httpx/subprocess import on Linux — that would break
real runs. Everything else matches the spec's stage name "select".)

Groq does the volume work (scan the moment index, score each). Final taste is the
user's when they run the agent in Claude Code. Dedup against already-posted moments
(memory/posted_moments.json) so we never re-clip something we've shipped.

Offline mode scores by spike intensity + transcript richness (deterministic), so the
pipeline is testable without Groq.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

POSTED = C.MEMORY / "posted_moments.json"
MAX_CANDIDATES = 120     # cap what we hand Groq, to fit context
MIN_LIVE_SCORE = 1       # drop only the model's flat-0 "dead" picks (countdown/hype);
                         # everything else ships in relative-score order (original top-N)
# Groq free tier is 6000 tokens/min AND 30 req/min. 97 moments in one call was
# ~8.7k tokens -> 413 "Request too large". Batch the index into small chunks
# (~15-20 moments ≈ under 5k tokens each), score each, then combine + rank.
BATCH_MOMENTS = 18       # moments per Groq request (stays well under the token cap)
BATCH_DELAY_SECONDS = 4  # pause between requests (free tier 6000 TPM / 30 RPM)

# Dead-air categories killed OUTRIGHT — "if nothing has happened yet, it's not a
# moment." Covers stream logistics, intros/outros, promo/ticket talk, and pre-event
# ANTICIPATION (buildup before any actual action). A valid moment is a physical event
# or payoff, never talking about one that hasn't happened.
FILLER_RE = re.compile(
    r"\b("
    # intros / stream logistics
    r"we'?re live|we are live|going live|stream(ing)?( is)?( starting| soon)|"
    r"starting soon|start(ing)? in|be right back|brb|countdown|count down|"
    r"waiting (for|on)|hold on|one sec|give it a (sec|second|minute|moment)|"
    r"welcome (to|back)|mic check|sound check|can you (hear|guys hear)|test test|"
    r"chat check|is the (stream|mic|audio|sound)|almost ready|setting up|"
    r"getting started|two seconds|gimme a|"
    # pre-event buildup / anticipation (NOT the event itself)
    r"about to (start|begin|go|kick off|fight|race|drop)|any (second|minute) now|"
    r"get ready|gonna (start|begin|kick off)|before we (start|begin|get)|"
    r"in (three|five|ten) (seconds|minutes)|coming up (next|soon)|stay tuned|"
    # outros / sign-offs
    r"thanks? (for|so much for) (tuning|watching|joining|coming|being)|thank you (all )?for|"
    r"thanks for tuning|see (you|ya) (next|later|guys|tomorrow)|catch (you|ya) (next|later)|"
    r"that'?s (all|it) for|that wraps|wrap(ping)? (it |this )?up|good ?night everyone|"
    r"like and subscribe|smash (that|the) like|hit the (like|follow|sub|bell)|drop a follow|"
    # promo / tickets / logistics
    r"link in (bio|the description|desc)|check the link|sign ?up|how (to|do i|do you) (enter|join|play)|"
    r"tickets?|promo code|use code|discount|giveaway|sponsor(ed|ship)?|"
    r"go to our|on our (website|site)|redeem|dot com"      # <- NO trailing '|' (empty
    r")\b", re.I)                                          #    alt would match everything)


def _overlaps(a, b):
    return a["source"] == b["source"] and not (a["end"] <= b["start"] or a["start"] >= b["end"])


def _is_filler(m):
    return bool(m.get("text") and FILLER_RE.search(m["text"]))


def _is_junk(m):
    """Drop moments whose (enriched) transcript is garbage whisper output — dot runs
    ('. . . .'), stutter ('e e e e'), or heavy single-word repetition ('here we go' x20,
    'dumb dumb dumb'). These carry no real event and otherwise pollute the pool (a weak
    batch's 'best' junk can still get a high relative score from the model)."""
    from collections import Counter
    words = re.findall(r"[a-zA-Z']+", m.get("text") or "")
    if len(words) < 4:
        return True
    top = Counter(w.lower() for w in words).most_common(1)[0][1]
    return top >= max(5, len(words) * 0.5)


# The actual ACTION signal lives in the commentary, not the audio level (music/hype is
# the loudest thing in the stream). Rank candidates by how much their transcript reads
# like a physical event / payoff being called, so real moments reach the model instead
# of countdown noise.
ACTION_RE = re.compile(
    r"\b(crash\w*|wreck\w*|flip\w*|fly\w*|overtak\w*|pass(?:es|ed|ing)?|"
    r"wins?|won|winner|victory|finish\w*|photo ?finish|last lap|final lap|"
    r"disqualif\w*|dq|penalt\w*|knock\w*|wipeout|spun|spins? out|"
    r"lead|leads|takes the lead|neck and neck|comeback|from (?:last|behind)|"
    r"dive[sd]?|jump\w*|collision|collide\w*|slam\w*|smash\w*|"
    r"insane|unbelievable|no way|no shot|can'?t believe|what a|incredible|"
    r"robbed|goes down|down goes|let'?s go+|oh my|holy|nail ?biter|photo)\b", re.I)


def _nearby_text(tr_segs, start, end, pre=8.0, post=4.0, limit=300):
    lo, hi = start - pre, end + post
    return " ".join(s["text"] for s in tr_segs
                    if s.get("text") and s.get("end", 0) >= lo and s.get("start", 0) <= hi)[:limit]


def _candidate_rank(m):
    """Higher = more likely a real moment. Blend of (a) action words in the enriched
    transcript — the caster CALLING a physical event — and (b) audio intensity — loud
    chaos/crashes. Both signals surface real moments; the Groq pass + dead-score drop do
    the final judging."""
    text = m.get("text") or ""
    action = len(ACTION_RE.findall(text))
    base = float(m.get("intensity", 0) or 0)
    return action * 4.0 + base * 2.0


def merge_close(moments, gap):
    """Merge moments in the same source whose gap < `gap` seconds into one moment
    (union bounds, max intensity, joined text). Kills same-event duplication."""
    from collections import defaultdict
    by_src = defaultdict(list)
    for m in moments:
        by_src[m["source"]].append(m)
    out = []
    for ms in by_src.values():
        ms.sort(key=lambda x: x["start"])
        cur = None
        for m in ms:
            if cur and m["start"] - cur["end"] <= gap:
                cur["end"] = max(cur["end"], m["end"])
                if m.get("intensity", 0) > cur.get("intensity", 0):
                    cur["intensity"] = m["intensity"]
                    cur["type"] = m["type"]
                    cur["peak"] = m.get("peak")   # follow the peak to the loudest sub-moment
                if m.get("text"):
                    cur["text"] = ((cur.get("text", "") + " " + m["text"]).strip())[:400]
            else:
                if cur:
                    out.append(cur)
                cur = dict(m)
        if cur:
            out.append(cur)
    return out


def dedup(moments):
    posted = C.load_json(POSTED, default=[]) or []
    if not posted:
        return moments
    kept = [m for m in moments if not any(_overlaps(m, p) for p in posted)]
    C.log(f"dedup: dropped {len(moments) - len(kept)} already-posted moment(s).")
    return kept


def _spread_ok(m, used, min_sep):
    """True unless a pick from the same source is within min_sep seconds."""
    return all(m["source"] != p["source"] or abs(m["start"] - p["start"]) >= min_sep
               for p in used)


def _heuristic_scores(moments, n, min_sep):
    """Offline: intensity-led, favor moments that have transcript text, spread out."""
    scored = []
    for m in moments:
        s = m.get("intensity", 0) * (10 if m["type"] == "audio_spike" else 6)
        if m.get("text"):
            s += min(len(m["text"]), 120) / 12.0
        scored.append((s, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked, used = [], []
    for s, m in scored:
        if not _spread_ok(m, used, min_sep):
            continue
        picked.append({**m, "score": round(float(s), 2),
                       "reason": f"{m['type']} intensity {m.get('intensity')}"})
        used.append(m)
        if len(picked) >= n:
            break
    return picked


def _parse_json_array(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _score_batch(client, campaign, knowledge, batch, n):
    """Score one small batch of moments with Groq. Returns a list of
    {id, score, reason} dicts (empty on an unparseable response)."""
    lines = [{"id": m["id"], "type": m["type"], "intensity": m.get("intensity"),
              "t": round(m["start"], 1), "peak": m.get("peak"),
              "text": (m.get("text") or "")[:160]} for m in batch]
    system = ("You are an elite short-form clipper doing FIRST-PASS moment scoring for a "
              "vertical clip campaign. A valid moment MUST contain an ACTUAL PHYSICAL EVENT "
              "or PAYOFF — a hit, crash, overtake, knockout, win, wipeout, big reaction to "
              "something that just happened. NOT anticipation of one. "
              "SCORE 0-15 (dead — reject) for: pre-stream/countdown, intros, 'we're live' / "
              "'starting soon', outros / 'thanks for tuning in', pre-fight or pre-race "
              "BUILDUP before anything happens, and logistics/ticket/promo/sponsor talk. "
              "These are visually dead even when loud or wordy — do not be fooled by an "
              "audio spike over a countdown or crowd hype. "
              "PRIMARY criterion for real moments: WOULD THE FIRST 2 SECONDS STOP A SCROLL? "
              "A moment can be opened on its peak (field 'peak' = the loudest/chaos second); "
              "score how hard that opening beat hits. Secondary: self-contained payoff, "
              "chaos/emotion, fit with the campaign audience + rules below. "
              "Return ONLY a JSON array of objects "
              '{"id","score","reason"} with score 0-100. No prose.')
    user = (f"Campaign: {campaign}\n"
            f"Audience: {C.AUDIENCE_CONTEXT}\n\n"
            + (f"Campaign knowledge (apply this):\n{knowledge}\n\n" if knowledge else "")
            + f"Score EVERY moment in this batch on the first-2-seconds scroll-stop "
            f"first, moment quality second "
            f"(id, type, intensity, start seconds t, peak second, transcript text):\n"
            f"{json.dumps(lines, ensure_ascii=False)}")
    raw = C.groq_chat(client, system, user, temperature=0.4, max_tokens=900)
    arr = _parse_json_array(raw)
    if not arr:
        C.warn("Groq returned unparseable scores for a batch — skipping it.")
        return []
    return arr


def _groq_scores(client, campaign, moments, n, min_sep):
    # Candidate pool = top by audio INTENSITY (the original approach that surfaced the
    # batch-#1 keepers — loud crashes/fights/finishes). The fixed filler-kill + Groq's
    # dead-score drop remove the countdown/hype that used to slip through; action words
    # in the enriched transcript break intensity ties toward real calls.
    cand = sorted(moments, key=lambda m: (m.get("intensity", 0), _candidate_rank(m)),
                  reverse=True)[:MAX_CANDIDATES]
    knowledge = C.load_knowledge()[:800]
    by_id = {m["id"]: m for m in moments}

    # Batch to stay under the free-tier token/request caps, then combine + rank all
    # batch results together to pick the final top clips.
    batches = [cand[i:i + BATCH_MOMENTS] for i in range(0, len(cand), BATCH_MOMENTS)]
    scored, seen = [], set()
    for bi, batch in enumerate(batches):
        if bi:
            time.sleep(BATCH_DELAY_SECONDS)   # respect 30 req/min free-tier limit
        C.log(f"select: scoring batch {bi + 1}/{len(batches)} ({len(batch)} moments).")
        for item in _score_batch(client, campaign, knowledge, batch, n):
            m = by_id.get(item.get("id"))
            if not m or m["id"] in seen:
                continue
            seen.add(m["id"])
            scored.append({**m, "score": float(item.get("score", 0)),
                           "reason": str(item.get("reason", ""))[:200]})

    if not scored:
        C.warn("Groq scored no moments across all batches — falling back to heuristic.")
        return _heuristic_scores(moments, n, min_sep)

    # NEVER pad the batch with dead moments just to hit N. A moment the model judged
    # dead (buildup/countdown/hype/logistics = low score) must not ship — a handful of
    # real clips beats 25 countdown picks. If everything scores low, that signals the
    # candidate POOL has no action (e.g. spikes computed on the wrong audio track).
    # Drop the model-judged-dead AND junk-text (dots/stutter/repetition) moments — the
    # rest ship in relative-score order (original behavior), so we still deliver a batch.
    alive = [m for m in scored if m["score"] >= MIN_LIVE_SCORE and not _is_junk(m)]
    dead = len(scored) - len(alive)
    if dead:
        C.log(f"select: dropped {dead} dead/junk moment(s) "
              f"(score < {MIN_LIVE_SCORE} or garbage transcript).")
    if not alive:
        C.warn(f"select: EVERY candidate scored < {MIN_LIVE_SCORE} — the pool has no real "
               "action. This usually means moments.json spikes were built on the wrong "
               "audio track; recompute spike detection on the merged audio.")

    alive.sort(key=lambda x: x["score"], reverse=True)
    picked, used = [], []
    for m in alive:
        if not _spread_ok(m, used, min_sep):
            continue
        picked.append(m)
        used.append(m)
        if len(picked) >= n:
            break
    return picked


def run(state):
    data = C.load_json(C.MOMENTS_JSON)
    if not data:
        C.fail("campaign/moments.json missing — run the index stage first.")
    cfg = state.get("config", {})
    n = int(cfg.get("clips_per_batch", 10))
    min_sep = float(cfg.get("min_separation_seconds", 60))
    merge_gap = float(cfg.get("merge_gap_seconds", 15))

    moments = data.get("moments", [])
    raw_count = len(moments)
    # Enrich every moment with the surrounding commentary FIRST — so a text-less audio
    # spike gets context, the action ranker can see what's happening, and hype spikes
    # (e.g. over a 'sneak peek' / 'welcome to' intro) get caught by the filler kill.
    tr_by_source = {s["source"]: s.get("transcript", []) for s in data.get("sources", [])}
    for m in moments:
        if not (m.get("text") or "").strip():
            m["text"] = _nearby_text(tr_by_source.get(m["source"], []),
                                     float(m["start"]), float(m["end"]))
    moments = [m for m in moments if not _is_filler(m)]                 # kill filler
    moments = merge_close(moments, merge_gap)                           # merge same-event
    moments = dedup(moments)                                            # drop already-posted
    C.log(f"moments: {raw_count} raw -> {len(moments)} after filler-kill + merge + dedup.")
    if not moments:
        C.fail("no candidate moments left after filtering — nothing to select.")

    campaign = (C.load_json(C.RULES_JSON) or {}).get("campaign", state.get("campaign") or "campaign")

    client = C.groq_client()
    if client is None:
        C.warn("offline mode — selecting via heuristic (no Groq).")
        selected = _heuristic_scores(moments, n, min_sep)
    else:
        selected = _groq_scores(client, campaign, moments, n, min_sep)

    if not selected:
        C.fail("select found NO live moments (every candidate scored below "
               f"{MIN_LIVE_SCORE} = dead buildup/hype/countdown). Refusing to ship a batch "
               "of dead clips. Root cause is almost always that moments.json spikes were "
               "built on the wrong audio track — recompute spike detection on the merged "
               "audio (cheap; reuses the cached transcript, no whisper re-run).")

    C.save_json(C.SELECTED_JSON, {"campaign": campaign, "selected": selected})
    C.mark_stage(state, "select", selected=len(selected))
    C.log(f"select done: {len(selected)} moment(s) chosen.")


if __name__ == "__main__":
    run(C.load_state())
