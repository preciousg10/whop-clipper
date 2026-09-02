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
# CANDIDATE POOL handed to the LLM scorer. This is deliberately GENEROUS, not a loudness gate:
# the LLM judges CONTENT quality, so it must SEE a rich, content-diverse set (including quiet-but-
# interesting moments), then it — not a dumb audio threshold — decides what's good. Candidates are
# ranked by a CONTENT blend (_content_rank), NOT intensity, and we feed the top select_max_candidates
# (default 400) of them. 4 rotating Groq keys make the extra batches affordable. (Overridable via
# config select_max_candidates.) Old behavior took the top-120 by LOUDNESS — that starved the
# scorer, discarding ~93 text-rich dialogue moments before it ever saw them.
DEFAULT_MAX_CANDIDATES = 400
DEFAULT_MIN_CAND_SECONDS = 3.0   # drop true sub-clip fragments (silence/one-word) before ranking
MIN_LIVE_SCORE = 1       # drop the model's flat-0 "dead" picks (countdown/hype/logistics)

# SAFETY CEILING + QUALITY BAR. Selection ships DRAFTS for human review, so it feeds the funnel
# ALL moments >= dead-floor best-first (not only the ones that clear the bar), bounded by two caps.
#   - HARD_CAP is the absolute safety ceiling on how many clips one run can select, so a runaway
#     can't burn the whole Groq free tier overnight (each selected clip costs a caption Groq call
#     downstream). It is a CEILING, never a target — we do not pad up to it.
#   - GOOD_SCORE is the "genuinely good" bar on Groq's 0-100 score. Moments carrying a striking
#     STANDALONE STATEMENT (a quotable, screenshot-worthy line) clear it and rank at the top;
#     moments between the dead-floor and this bar are lower-confidence FILLER that still ships for
#     review (rejected there, not pre-filtered to one). We LOG how many cleared the bar vs filler.
#     Both are overridable via config (select_hard_cap / select_min_quality) — see run.py DEFAULT_CONFIG.
DEFAULT_HARD_CAP = 50
DEFAULT_GOOD_SCORE = 60
# Per-campaign REVIEW cap: how many draft candidates one campaign feeds the review funnel. We ship
# ALL moments >= dead-floor (best-first) up to this — these are DRAFTS a human approves before
# posting, so a fuller funnel is good; weak ones get rejected at review, not pre-filtered to one.
# Tighter than DEFAULT_HARD_CAP (the absolute safety ceiling) so one rich VOD can't dump 50 clips.
DEFAULT_PER_CAMPAIGN_CAP = 10
# LOW dead-floor (0-100). The score is LLM-judged from the TRANSCRIPT ONLY — it can't see the
# video, so it undersells visually-funny content; the floor is therefore forgiving. If even the
# BEST moment scores below this, the campaign is genuinely dead (flat/dead even in text) and we
# STOP + signal auto-advance. A best >= floor SHIPS for human review (we do NOT require 60).
DEFAULT_DEAD_FLOOR = 40
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


# CONTENT-INTEREST markers beyond ACTION_RE: story/reaction/controversy/opinion cues that make a
# spoken moment worth clipping even at NORMAL volume. Used only to RANK candidates so the LLM sees
# them — never to judge quality (that's the LLM's job). Kept broad on purpose.
MARKER_RE = re.compile(
    r"\b("
    # story / narrative setup
    r"so (i|we|he|she|they)|one time|the other day|turns out|i (told|said|asked|realized)|"
    r"you know what|let me tell you|here'?s the thing|the story|remember when|back when|"
    # opinion / controversy / hot-take / beef
    r"honestly|the truth is|hot take|unpopular|controversial|the problem (is|with)|"
    r"disagree|i'?m telling you|the reality|nobody (talks|says)|everybody|the fact that|"
    r"overrated|underrated|the worst|the best|literally the|"
    # reaction / emotion / emphasis
    r"i can'?t|oh my|are you (kidding|serious)|no way|that'?s (crazy|insane|wild|nuts)|"
    r"i swear|dead ass|deadass|lowkey|highkey|actually|"
    # money / stakes / numbers-driven interest (this campaign is finance/health heavy)
    r"million|billion|thousand|dollars|\$\d|per month|a month|net worth"
    r")\b", re.I)


def _content_rank(m):
    """CONTENT-driven candidate score — the replacement for the old loudness gate. Higher = more
    likely to be an interesting moment a person would clip, judged from the TRANSCRIPT, so a
    normal-volume good story survives to reach the LLM. Blend of:
      - richness:  how much is actually SAID (word count) — substance, not a fragment;
      - density:   words per second — real dialogue vs a long quiet stretch;
      - markers:   action + story/opinion/controversy/reaction/stakes cues in the text;
      - intensity: audio loudness — kept as ONE WEAK signal (max ~1 of ~14), never the gate.
    The LLM still does the actual quality judging; this only decides who gets SEEN by it."""
    text = m.get("text") or ""
    words = re.findall(r"[A-Za-z']+", text)
    nwords = len(words)
    dur = max(1.0, float(m.get("end", 0) or 0) - float(m.get("start", 0) or 0))
    richness = min(nwords, 80) / 80.0                       # 0..1 substance / length
    density = min(nwords / dur, 4.0) / 4.0                  # 0..1 dialogue density
    markers = len(ACTION_RE.findall(text)) + len(MARKER_RE.findall(text)) + text.count("?")
    intensity = float(m.get("intensity", 0) or 0)
    return (richness * 3.0 + density * 2.0 + min(markers, 8) * 1.0
            + min(intensity, 12.0) / 12.0 * 1.0)           # loudness: weak tiebreak ONLY


def _prefilter_candidates(moments, min_seconds):
    """Remove TRUE junk before ranking — but NEVER 'quiet': a normal-volume interesting moment
    must survive. Drops: sub-`min_seconds` fragments (too short to be a clip), garbage transcripts
    (dot-runs / stutter / heavy single-word repetition via _is_junk), and exact-duplicate text.
    A wordless but LOUD audio_spike is KEPT (a real non-speech beat — scream/crash/laughter — that
    the captions stage later labels), so junk-removal doesn't quietly delete non-speech action."""
    out, seen = [], set()
    for m in moments:
        dur = float(m.get("end", 0) or 0) - float(m.get("start", 0) or 0)
        if dur < min_seconds:
            continue
        loud_nonspeech = (m.get("type") == "audio_spike"
                          and float(m.get("intensity", 0) or 0) >= 6.0)
        if _is_junk(m) and not loud_nonspeech:
            continue
        txt = re.sub(r"\s+", " ", (m.get("text") or "").strip().lower())[:200]
        key = (m.get("source"), txt)
        if txt and key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


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
    system = ("You are an elite short-form clipper. Your ONE job: find the moment that contains "
              "the most STRIKING STANDALONE STATEMENT — a sentence or two that is wild, "
              "provocative, surprising, contrarian, or hooky ON ITS OWN, such that a stranger "
              "reading or hearing JUST THAT LINE (with zero prior context) immediately wants "
              "more. "
              "SCORE HIGHEST (70-100) the moments containing a quotable, SCREENSHOT-WORTHY line: "
              "a bold claim, a hot take, a shocking number/stat, a contrarian opinion, a 'wait, "
              "WHAT?' statement, a confession, a brutal callout — a line that LANDS whether or "
              "not the viewer has any context. The test: put this sentence on a screenshot with "
              "NO setup — does it still hit and make a stranger stop? Reward SELF-CONTAINED PUNCH. "
              "PENALIZE hard (push DOWN) moments that only make sense with prior episode/stream "
              "context: inside references, 'as I said earlier', mid-argument callbacks, pronouns "
              "with no antecedent, running bits, 'he/they' with no named subject — anything a "
              "cold viewer can't follow. If a stranger would think 'I don't get it / who or what "
              "are they talking about?', it is NOT a striking standalone statement, no matter how "
              "animated the delivery. "
              "LOUDNESS IS NOT REQUIRED: a CALM but INSANE claim beats a LOUD but boring one. Do "
              "NOT reward a moment for energy, screaming, or an audio spike on its own — a quiet, "
              "deadpan, jaw-dropping sentence should OUTSCORE hype noise. The 'intensity' and "
              "'peak' audio signals are ONLY a faint tiebreak between two otherwise equally "
              "striking lines — never a reason to raise a score by themselves. "
              "SCORE 0-15 (dead — reject): pre-stream/countdown, intros, 'we're live' / "
              "'starting soon', outros / 'thanks for tuning in', logistics/ticket/promo/sponsor "
              "talk — dead even when loud. "
              "SCORE 16-45 (flat/mundane — demote): ordinary conversation, neutral play-by-play, "
              "filler, 'nothing really happens' stretches, and ESPECIALLY context-dependent lines "
              "a cold viewer can't follow — even if loud or lively. A normal sentence is not a "
              "striking statement. "
              "SCORE 70-100 (elite): ONLY moments with a genuinely quotable, self-contained, "
              "scroll-stopping LINE. Be a HARSH grader — most moments are mid; reserve high scores. "
              "For EVERY moment, also return `line`: the EXACT striking sentence, quoted verbatim "
              "from that moment's transcript (<= ~200 chars), that earns the score — the one a "
              "viewer would screenshot. If no such line exists, return an empty string for `line` "
              "and score the moment low. "
              "Return ONLY a JSON array of objects "
              '{"id","score","line","reason"} with score 0-100. No prose.')
    user = (f"Campaign: {campaign}\n"
            f"Audience: {C.AUDIENCE_CONTEXT}\n\n"
            + (f"Campaign knowledge (apply this):\n{knowledge}\n\n" if knowledge else "")
            + f"For EVERY moment, find its most STRIKING STANDALONE STATEMENT and score how "
            f"much a stranger with ZERO prior context would want more after JUST that line. "
            f"Quotable/screenshot-worthy bold claim, hot take, shocking number, contrarian "
            f"opinion, or 'wait what' statement = high. Needs prior context to make sense = "
            f"low. Loudness does NOT matter (intensity/peak are only a faint tiebreak). Return "
            f"`line` = the exact striking sentence you scored on "
            f"(id, type, intensity, start seconds t, peak second, transcript text):\n"
            f"{json.dumps(lines, ensure_ascii=False)}")
    raw = C.llm_chat(client, system, user, temperature=0.4, max_tokens=900)
    arr = _parse_json_array(raw)
    if not arr:
        C.warn("Groq returned unparseable scores for a batch — skipping it.")
        return []
    return arr


def _groq_scores(client, campaign, moments, n, min_sep, min_quality, hard_cap, dead_floor,
                 per_campaign_cap=DEFAULT_PER_CAMPAIGN_CAP,
                 max_candidates=DEFAULT_MAX_CANDIDATES, min_cand_seconds=DEFAULT_MIN_CAND_SECONDS):
    # CANDIDATE POOL — content-ranked, NOT loudness-gated. Remove true junk (fragments/garbage/
    # exact-dupes), then rank by CONTENT (_content_rank: transcript richness, dialogue density,
    # story/reaction/controversy markers; intensity only a weak tiebreak) and hand the LLM a
    # generous top-N. The LLM — which actually judges content quality — is the filter now; a
    # quiet-but-interesting moment (normal volume) reaches it instead of being cut for being soft.
    pool = _prefilter_candidates(moments, min_cand_seconds)
    cand = sorted(pool, key=_content_rank, reverse=True)[:max_candidates]
    C.log(f"select: feeding {len(cand)} candidate(s) to the LLM — CONTENT-ranked, not loudness-"
          f"gated (from {len(moments)} merged → {len(pool)} after junk-prefilter → cap "
          f"{max_candidates}). Intensity is only a weak tiebreak.")
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
                           "line": str(item.get("line", ""))[:200],
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

    # DEAD-FLOOR (Change 2): the score is text-blind, so the floor is LOW/forgiving. Only when
    # even the BEST moment can't clear it is the campaign genuinely dead — STOP cleanly and signal
    # auto-advance (NothingUsable) so the pipeline rolls to the next ranked campaign. A best at or
    # above the floor SHIPS for human review (we do NOT require the 60 highlight bar — 40+ ships).
    best = alive[0]["score"]
    if best < dead_floor:
        raise C.NothingUsable(
            f"select: best moment {best:.0f} < dead-floor {dead_floor:.0f} — campaign has no "
            f"usable moments, advancing to the next ranked campaign.")
    C.log(f"select: best {best:.0f} >= dead-floor {dead_floor:.0f} — proceeding.")

    # REVIEW FUNNEL. These are DRAFTS a human approves before posting, so we ship ALL moments at or
    # above the dead-floor (best-first — `alive` is already sorted by score desc), NOT only the ones
    # that clear the highlight bar. Moments >= min_quality naturally rank at the top; 40-59 moments
    # are lower-confidence "filler" candidates that round out the batch and get accepted/rejected at
    # review, not pre-filtered to one. Two ceilings bound the count: per_campaign_cap (default 10)
    # keeps one rich VOD from dumping 50 clips into the funnel, and hard_cap is the absolute safety
    # ceiling on caption Groq spend — the effective cap is the tighter of the two.
    pool = [m for m in alive if m["score"] >= dead_floor]
    cap = max(1, min(per_campaign_cap, hard_cap))
    cleared = sum(1 for m in pool if m["score"] >= min_quality)
    filler = len(pool) - cleared
    C.log(f"select: {len(pool)} draft candidate(s) >= dead-floor {dead_floor:g} — "
          f"{cleared} cleared the highlight bar (>= {min_quality:g}), "
          f"{filler} are {dead_floor:g}-{min_quality - 1:g} filler. "
          f"Shipping up to {cap} best-first (per-campaign cap {per_campaign_cap}, "
          f"safety ceiling {hard_cap}).")

    picked, used = [], []
    for m in pool:
        if not _spread_ok(m, used, min_sep):
            continue
        picked.append(m)
        used.append(m)
        # Log the WINNING LINE (the striking sentence the model scored on) so it's obvious at a
        # glance WHY each moment was picked. Falls back to the model's reason if no line came back.
        line = (m.get("line") or "").strip()
        why = f"“{line}”" if line else (m.get("reason") or "no line").strip()
        C.log(f"select: PICK {m['id']} (score {m['score']:.0f}) — {why}")
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
    dead_floor = float(cfg.get("select_dead_floor", DEFAULT_DEAD_FLOOR))
    # Per-campaign review cap — how many draft candidates this campaign contributes best-first.
    per_campaign_cap = int(cfg.get("select_per_campaign_cap", DEFAULT_PER_CAMPAIGN_CAP))
    # Candidate pool fed to the LLM scorer — content-ranked, generous (not the loudest few).
    max_candidates = int(cfg.get("select_max_candidates", DEFAULT_MAX_CANDIDATES))
    min_cand_seconds = float(cfg.get("select_min_candidate_seconds", DEFAULT_MIN_CAND_SECONDS))
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

    client = C.llm_client(cfg)
    if client is None:
        C.warn("offline mode — selecting via heuristic (no LLM).")
        selected = _heuristic_scores(moments, min(n, hard_cap), min_sep)
    else:
        C.log(f"select: LLM provider = {client.status()}")
        selected = _groq_scores(client, campaign, moments, n, min_sep, min_quality, hard_cap,
                                dead_floor, per_campaign_cap, max_candidates, min_cand_seconds)

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
