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
MIN_LIVE_SCORE = 1       # drop the model's flat-0 "dead" picks (countdown/hype/logistics)

# SAFETY CEILING + QUALITY BAR (Task D). We hunt HIGHLIGHT-worthiness, not a fixed count.
#   - HARD_CAP is a hard ceiling on how many clips one run can select, so a runaway can't
#     burn the whole Groq free tier overnight (each selected clip costs a caption Groq call
#     downstream). It is a CEILING, never a target — we do not pad up to it.
#   - GOOD_SCORE is the "genuinely good" bar on Groq's 0-100 highlight score. Only moments a
#     human would actually clip (funny / high-energy / chaotic / surprising peaks) clear it;
#     within the ceiling we take exactly those and stop. If a stream has only 6 real
#     highlights we ship 6, not 25. Both are overridable via config (select_hard_cap /
#     select_min_quality) — see run.py DEFAULT_CONFIG.
DEFAULT_HARD_CAP = 50
DEFAULT_GOOD_SCORE = 60
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


# A moment worth clipping isn't only a crash — it's anything a person would clip watching
# the whole stream: a physical event/payoff, a FUNNY beat, a wild reaction, chaos, or a
# surprise. Audio level alone is a trap (music/hype is the loudest thing), so we rank
# candidates by how much the surrounding commentary reads like one of those highlights,
# breaking intensity ties toward real moments and away from countdown noise.
ACTION_RE = re.compile(
    r"\b("
    # physical events / payoffs
    r"crash\w*|wreck\w*|flip\w*|fly\w*|overtak\w*|pass(?:es|ed|ing)?|"
    r"wins?|won|winner|victory|finish\w*|photo ?finish|last lap|final lap|"
    r"disqualif\w*|dq|penalt\w*|knock\w*|wipeout|spun|spins? out|"
    r"lead|leads|takes the lead|neck and neck|comeback|from (?:last|behind)|"
    r"dive[sd]?|jump\w*|collision|collide\w*|slam\w*|smash\w*|goes down|down goes|"
    # shock / disbelief / hype reactions
    r"insane|unbelievable|no way|no shot|can'?t believe|what a|incredible|"
    r"robbed|let'?s go+|oh my|holy|nail ?biter|"
    # funny / chaotic / surprising (highlight beats that aren't 'action')
    r"hilarious|funny|lmao+|lmfao|rofl|crying|dying|bruh|bro|"
    r"what the|wtf|diabolical|unserious|chaos|chaotic|"
    r"caught|exposed|clip (?:it|that)|clip this|you have to see|wait (?:for|till|until))\b",
    re.I)


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


def merge_close(moments, gap, max_span=60.0):
    """Merge moments in the same source whose gap < `gap` seconds into one moment
    (union bounds, max intensity, joined text). Kills same-event duplication.

    HARD CAP: never let a merged moment grow past `max_span` seconds. Non-stop
    commentary used to chain into 400-550s blobs (m2440 spanned 552s) that the cut
    stage then clamped to an arbitrary window — a tight gap plus this cap keeps a merged
    moment a single real beat, and cut.clip_bounds centers the clip on its peak."""
    from collections import defaultdict
    by_src = defaultdict(list)
    for m in moments:
        by_src[m["source"]].append(m)
    out = []
    for ms in by_src.values():
        ms.sort(key=lambda x: x["start"])
        cur = None
        for m in ms:
            if (cur and m["start"] - cur["end"] <= gap
                    and max(cur["end"], m["end"]) - cur["start"] <= max_span):
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
    system = ("You are an elite short-form clipper hunting HIGHLIGHTS in a livestream. Your "
              "job: find the moments a person would actually clip if they watched the whole "
              "stream — the FUNNY, HIGH-ENERGY, CHAOTIC, SURPRISING peaks. That includes a "
              "hit/crash/overtake/knockout/win/wipeout, but EQUALLY a hilarious bit, a wild "
              "or unhinged reaction, a clutch play, a brutal fail, a shocking take, a chaotic "
              "meltdown — anything scroll-stopping and self-contained. "
              "ENERGY IS A SIGNAL, NOT THE ANSWER: each moment carries an audio 'intensity' "
              "and a 'peak' second (the loudest/most chaotic instant). A spike means something "
              "MIGHT be happening — your job is to confirm it's actually GOOD, not just loud. "
              "Do NOT reward a spike that's only music, a hype sting, crowd noise, a countdown, "
              "or an intro. "
              "SCORE 0-15 (dead — reject): pre-stream/countdown, intros, 'we're live' / "
              "'starting soon', outros / 'thanks for tuning in', pre-event BUILDUP before "
              "anything happens, and logistics/ticket/promo/sponsor talk — dead even when loud. "
              "SCORE 70-100 (elite): only genuinely clip-worthy peaks a human would definitely "
              "clip. Be a HARSH grader — most moments are mid; reserve high scores. "
              "PRIMARY criterion: WOULD THE FIRST 2 SECONDS (opened on 'peak') STOP A SCROLL? "
              "Secondary: is it funny/chaotic/surprising, self-contained, and a fit for the "
              "campaign audience + rules below? "
              "Return ONLY a JSON array of objects "
              '{"id","score","reason"} with score 0-100. No prose.')
    user = (f"Campaign: {campaign}\n"
            f"Audience: {C.AUDIENCE_CONTEXT}\n\n"
            + (f"Campaign knowledge (apply this):\n{knowledge}\n\n" if knowledge else "")
            + f"Score EVERY moment on highlight-worthiness — would someone CLIP this? Judge "
            f"the first-2-seconds scroll-stop (opened on 'peak') first, overall moment "
            f"quality second. Confirm the audio spike is a real good moment, not just loud "
            f"(id, type, intensity, start seconds t, peak second, transcript text):\n"
            f"{json.dumps(lines, ensure_ascii=False)}")
    raw = C.groq_chat(client, system, user, temperature=0.4, max_tokens=900)
    arr = _parse_json_array(raw)
    if not arr:
        C.warn("Groq returned unparseable scores for a batch — skipping it.")
        return []
    return arr


def _groq_scores(client, campaign, moments, n, min_sep, min_quality, hard_cap):
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
    # RESUME (Unit 2b): reload batch scores checkpointed on a prior run so a DAILY-cap stop
    # doesn't re-score completed batches. Batches are deterministic (moments.json + posted are
    # stable), so a fully-scored batch is skipped by id. GroqDailyCapError propagates to run.py.
    partial = C.load_json(C.SELECT_PARTIAL) or {}
    scored = list(partial.get("scored", []))
    seen = {s["id"] for s in scored}
    if scored:
        C.log(f"select: resuming — {len(scored)} moment(s) already scored (checkpoint).")
    made_call = False
    for bi, batch in enumerate(batches):
        if all(m["id"] in seen for m in batch):
            continue                          # batch already scored on a prior run
        if made_call:
            time.sleep(BATCH_DELAY_SECONDS)   # respect 30 req/min free-tier limit (between calls)
        made_call = True
        C.log(f"select: scoring batch {bi + 1}/{len(batches)} ({len(batch)} moments).")
        for item in _score_batch(client, campaign, knowledge, batch, n):
            m = by_id.get(item.get("id"))
            if not m or m["id"] in seen:
                continue
            seen.add(m["id"])
            scored.append({**m, "score": float(item.get("score", 0)),
                           "reason": str(item.get("reason", ""))[:200]})
        C.save_json(C.SELECT_PARTIAL, {"scored": scored})   # checkpoint after each batch

    if not scored:
        C.warn("Groq scored no moments across all batches — falling back to heuristic.")
        return _heuristic_scores(moments, min(n, hard_cap), min_sep)

    # Drop model-judged-dead (buildup/countdown/hype/logistics) and junk-text
    # (dots/stutter/repetition) moments — never ship those.
    alive = [m for m in scored if m["score"] >= MIN_LIVE_SCORE and not _is_junk(m)]
    dead = len(scored) - len(alive)
    if dead:
        C.log(f"select: dropped {dead} dead/junk moment(s) "
              f"(score < {MIN_LIVE_SCORE} or garbage transcript).")
    if not alive:
        C.warn(f"select: EVERY candidate scored < {MIN_LIVE_SCORE} — the pool has no real "
               "action. This usually means moments.json spikes were built on the wrong "
               "audio track; recompute spike detection on the merged audio.")
        return []

    alive.sort(key=lambda x: x["score"], reverse=True)

    # HIGHLIGHT BAR + SAFETY CEILING. Prefer only the genuinely-good peaks (score >=
    # min_quality) and take AS MANY as clear it, up to hard_cap — we do NOT pad to a target
    # count. If a stream has 6 real highlights we ship 6; if it has 60 we still stop at the
    # ceiling so downstream caption Groq calls can't run away. When NOTHING clears the bar
    # (a flat stream) we don't fail — we ship the best available, but capped conservatively.
    good = [m for m in alive if m["score"] >= min_quality]
    if good:
        pool, cap = good, hard_cap
        C.log(f"select: {len(good)} moment(s) cleared the highlight bar "
              f"(score >= {min_quality:g}); taking up to the {hard_cap} ceiling, no padding.")
    else:
        pool = alive
        cap = max(1, min(hard_cap, n))
        C.warn(f"select: no moment cleared the highlight bar (score >= {min_quality:g}) — this "
               f"stream has no standout peaks. Shipping the {cap} best available instead.")

    picked, used = [], []
    for m in pool:
        if not _spread_ok(m, used, min_sep):
            continue
        picked.append(m)
        used.append(m)
        if len(picked) >= cap:
            break
    return picked


def run(state):
    data = C.load_json(C.MOMENTS_JSON)
    if not data:
        C.fail("campaign/moments.json missing — run the index stage first.")
    cfg = state.get("config", {})
    n = int(cfg.get("clips_per_batch", 10))
    min_sep = float(cfg.get("min_separation_seconds", 60))
    # Highlight bar + safety ceiling (Task D): the real limiters on how many clips ship.
    hard_cap = int(cfg.get("select_hard_cap", DEFAULT_HARD_CAP))
    min_quality = float(cfg.get("select_min_quality", DEFAULT_GOOD_SCORE))
    # merge_gap was 15s, which chained non-stop commentary into 400-550s blobs. Tight
    # gap (~7s) + a hard span cap keep a merged moment one real beat.
    merge_gap = float(cfg.get("merge_gap_seconds", 7))
    merge_max_span = float(cfg.get("merge_max_span_seconds", 60))

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
    moments = merge_close(moments, merge_gap, merge_max_span)           # merge same-event
    moments = dedup(moments)                                            # drop already-posted
    C.log(f"moments: {raw_count} raw -> {len(moments)} after filler-kill + merge + dedup.")
    if not moments:
        raise C.NothingUsable("no candidate moments left after filtering — nothing to select.")

    campaign = (C.load_json(C.RULES_JSON) or {}).get("campaign", state.get("campaign") or "campaign")

    client = C.groq_client()
    if client is None:
        C.warn("offline mode — selecting via heuristic (no Groq).")
        selected = _heuristic_scores(moments, min(n, hard_cap), min_sep)
    else:
        selected = _groq_scores(client, campaign, moments, n, min_sep, min_quality, hard_cap)

    if not selected:
        raise C.NothingUsable(
            "select found NO live moments (every candidate scored below "
            f"{MIN_LIVE_SCORE} = dead buildup/hype/countdown). Refusing to ship a batch "
            "of dead clips. Root cause is almost always that moments.json spikes were "
            "built on the wrong audio track — recompute spike detection on the merged "
            "audio (cheap; reuses the cached transcript, no whisper re-run).")

    C.save_json(C.SELECTED_JSON, {"campaign": campaign, "selected": selected})
    C.SELECT_PARTIAL.unlink(missing_ok=True)         # stage complete — drop the checkpoint
    C.mark_stage(state, "select", selected=len(selected))
    C.log(f"select done: {len(selected)} moment(s) chosen.")


if __name__ == "__main__":
    try:
        run(C.load_state())
    except C.NothingUsable as e:
        C.fail(str(e))            # standalone: still fail loud (only run.py --auto-advance moves on)
