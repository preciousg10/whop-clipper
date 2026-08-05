"""CAPTION GAUNTLET — 10 caption lines per clip, then attack them.

flzsh style: ONE top line, hook+context+joke in one, stakes+outcome+emotion, casual
lowercase energy, emoji as punctuation. Groq drafts; the gauntlet kills any candidate
containing a rules.json banned word BEFORE scoring, scores survivors against the
account style, and keeps the best (runner-up saved as a variant).

Also produces the rules-compliant per-platform text used in drafts/manifest.json.
Offline mode uses deterministic templates so the pipeline is testable without Groq.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# One small Groq call per clip, but the free tier is 6000 tokens/min. At ~700 tokens
# per caption call that's ~8 calls/min, so pace ~8s apart (groq_chat also backs off on
# any 429 as a safety net).
CAPTION_DELAY_SECONDS = 8

EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⁩⁦]")

MAX_WORDS = 8
N_CANDIDATES = 12
# The caption's ONLY job is to make the payoff feel mandatory. Every kept caption must
# hit one of these five proven hook patterns; anything that just describes or gives the
# payoff away is killed.
HOOK_PATTERNS = {
    # open question
    "question": re.compile(
        r"(\?|^\s*(how|why|what|who|when|did|does|do|is|are|which)\b)", re.I),
    # stakes (money / high-stakes framing that still withholds the outcome)
    "stakes": re.compile(
        r"(\$\s*\d|\b(on the line|for the (win|title|lead)|to win|everything|last (place|second|lap)|"
        r"final (lap|round|race)|match point|sudden death|winner takes)\b)", re.I),
    # disbelief
    "disbelief": re.compile(
        r"\b(no way|no shot|nah|cant believe|can'?t believe|cant be real|can'?t be real|"
        r"not real|how is this real|unreal|i'?m done|im done)\b", re.I),
    # controversy bait
    "controversy": re.compile(
        r"\b(shouldn'?t|should not|didn'?t count|doesn'?t count|robbed|rigged|cheated|"
        r"illegal|not fair|how is this allowed|this ain'?t it)\b", re.I),
    # direct address / open loop
    "direct": re.compile(
        r"\b(wait for|wait till|watch(?: till| until)?|keep watching|you (have|need) to see|"
        r"tell me why|pov|the way|watch this|look what)\b", re.I),
}
HOOK_RE = re.compile("|".join(p.pattern for p in HOOK_PATTERNS.values()), re.I)

# Kill outright: descriptions and past-tense summaries that give the payoff away.
DESCRIPTIVE_RE = re.compile(
    r"\b(we (just|made)|here (is|are)|this is (a|the)|reacts?|reaction|highlights?|clip of|"
    r"footage|compilation|moment where|when (he|she|they) said)\b", re.I)
# Past-tense reveal: subject + past-tense outcome verb = the ending is spoiled. Only a
# giveaway when the caption isn't also withholding via a hook (checked in quality_kill).
GIVEAWAY_RE = re.compile(
    r"\b(he|she|they|it|the \w+)\s+(won|lost|crashed|died|scored|beat|smashed|flipped|"
    r"finished|ended|fell|missed|nailed|dropped|broke)\b", re.I)

OFFLINE_TEMPLATES = [
    "how did this even happen 😭",
    "no way this actually happened 💀",
    "wait for the last second 🔥",
    "this shouldn't have counted fr 😭",
    "why is this so unserious 💀",
    "who told him to do this 😭",
    "watch till the very end 👀",
    "how is this even real 😭",
    "tell me why he did this 😭",
    "nah this cant be real 💀",
    "the way this ends is diabolical 💀",
    "pov you witness peak chaos 🐹",
]


# --- banned-word gauntlet (the load-bearing rules gate) ------------------------
def banned_hit(text, banned):
    """Return the first banned word/phrase present in text, or None."""
    low = (text or "").lower()
    for term in banned:
        t = term.lower().strip()
        if not t:
            continue
        if " " in t:            # multi-word phrase: substring match
            if t in low:
                return term
        elif re.search(rf"\b{re.escape(t)}\b", low):
            return term
    return None


def gauntlet(candidates, banned):
    """Kill any candidate containing a banned word; return (survivors, killed)."""
    survivors, killed = [], []
    for c in candidates:
        hit = banned_hit(c, banned)
        if hit:
            killed.append({"caption": c, "reason": f"banned word: {hit}"})
        else:
            survivors.append(c)
    return survivors, killed


# --- quality gate + style scoring (flzsh) --------------------------------------
def _word_count(text):
    """Words for the length cap — emoji/punctuation tokens don't count (they're
    punctuation, not words), so an 8-word line + emoji isn't wrongly killed."""
    return sum(1 for w in text.split() if any(ch.isalnum() for ch in w))


def quality_kill(cap):
    """Hard-kill reasons (before scoring). Returns a reason or None."""
    if _word_count(cap) > MAX_WORDS:
        return f"too long (>{MAX_WORDS} words)"
    if "\n" in cap:
        return "multi-line"
    if DESCRIPTIVE_RE.search(cap):
        return "merely describes (no hook)"
    if not HOOK_RE.search(cap):
        return "no hook pattern (question/stakes/disbelief/controversy/direct address)"
    if GIVEAWAY_RE.search(cap) and not HOOK_PATTERNS["question"].search(cap):
        return "past-tense summary gives the payoff away"
    return None


def score_caption(text):
    """Higher = more scroll-stopping. Rewards the five hook patterns, punishes
    description and payoff-spoiling past-tense summaries."""
    s = 0.0
    if EMOJI_RE.search(text):
        s += 2
    hooks = [name for name, rx in HOOK_PATTERNS.items() if rx.search(text)]
    s += 4 * len(hooks)                           # each hook pattern present stops a scroll
    if "question" in hooks or "disbelief" in hooks:
        s += 2                                    # the strongest cold openers
    if DESCRIPTIVE_RE.search(text):
        s -= 6
    if GIVEAWAY_RE.search(text) and "question" not in hooks:
        s -= 5                                    # spoils the payoff
    wc = _word_count(text)
    if wc <= 6:
        s += 2
    if wc > MAX_WORDS:
        s -= 8
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum(ch.islower() for ch in letters) / len(letters) > 0.85:
        s += 1                                    # lowercase energy
    if text.strip().endswith("."):
        s -= 1
    return s


# --- candidate generation ------------------------------------------------------
def _offline_candidates(moment):
    # Rotate templates by moment id so different clips get different captions (offline
    # is for the self-test; real captions come from Groq).
    import hashlib
    rot = int(hashlib.md5((moment.get("id", "")).encode()).hexdigest(), 16) % len(OFFLINE_TEMPLATES)
    return OFFLINE_TEMPLATES[rot:] + OFFLINE_TEMPLATES[:rot]


_LOGGED_BAD_GROQ = {"done": False}


def _parse_caption_lines(raw):
    """Tolerantly pull caption strings out of Groq output. Handles: markdown code
    fences, a JSON array anywhere in the text (even with trailing prose), or plain
    one-per-line output with bullets/numbering/quotes. Returns [] if nothing usable."""
    s = (raw or "").strip()
    s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)          # opening fence
    s = re.sub(r"\s*```\s*$", "", s).strip()            # closing fence
    # 1) JSON array anywhere in the blob
    mm = re.search(r"\[.*\]", s, re.S)
    if mm:
        try:
            arr = [str(x).strip() for x in json.loads(mm.group(0)) if str(x).strip()]
            if arr:
                return arr
        except Exception:
            pass
    # 2) line-by-line
    out = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', "", ln)      # strip bullet / numbering
        ln = ln.strip().strip('"').strip("'").rstrip(",").strip()
        low = ln.lower()
        if len(ln) < 2 or low.startswith(("here are", "here's", "sure", "captions:", "caption:")):
            continue
        out.append(ln)
    return out


def _nearby_transcript(tr_segs, start, end, pad_pre=10.0, pad_post=4.0, limit=400):
    """Commentary spoken AROUND a moment (segments overlapping [start-pad, end+pad]).
    Audio-spike moments carry no transcript of their own, but the caster is almost always
    reacting to what just happened — that nearby speech is what makes a caption specific
    instead of generic."""
    lo, hi = start - pad_pre, end + pad_post
    parts = [s["text"] for s in tr_segs
             if s.get("text") and s.get("end", 0) >= lo and s.get("start", 0) <= hi]
    return " ".join(parts).strip()[:limit]


def _groq_candidates(client, campaign, moment, style_notes, event=""):
    event = (event or moment.get("text") or "").strip()
    system = (
        "You write TOP captions for viral vertical sports/gaming/racing clips. The "
        "caption's ONLY job is to make the payoff feel MANDATORY to watch, and it MUST "
        "reference the ACTUAL event in THIS clip (use the transcript / what happens), "
        "NOT a generic phrase. RULES: ONE line, MAX 8 words, casual grammar, NO emoji, "
        "NO hashtags. Every caption MUST use one of these five proven hook patterns:\n"
        "  1) open question — 'how did the yellow hamster win THIS'\n"
        "  2) stakes — '$10k on the line and he does THIS'\n"
        "  3) disbelief — 'no way that overtake just happened'\n"
        "  4) controversy — 'that lap should NOT have counted'\n"
        "  5) direct address — 'wait for the jump at the last second'\n"
        "BANNED: vague filler ('what just happened', bare 'wait for it', 'why does he "
        "have the lead'), descriptions, past-tense summaries, or GIVING AWAY who won. "
        "Be SPECIFIC to this clip. Respect the campaign banned words/topics in the "
        "knowledge below. "
        f"Return {N_CANDIDATES} captions, ONE PER LINE — no numbering, no quotes, no JSON.")
    knowledge = C.load_knowledge()[:600]
    ev_line = (f"What happens / is said in THIS clip (reference it specifically): {event!r}\n"
               if event else
               "This clip has no transcript — it is a loud, chaotic physical moment; keep the "
               "caption specific to visible sports/racing action, not generic.\n")
    user = (f"Audience: {C.AUDIENCE_CONTEXT}\n\n"
            + (f"Campaign knowledge (apply this):\n{knowledge}\n\n" if knowledge else "")
            + f"Campaign: {campaign}\nStyle notes: {style_notes}\n"
            + ev_line
            + f"Give {N_CANDIDATES} caption lines, each using one hook pattern and "
              f"referencing the actual moment.")
    last = ""
    for _attempt in range(3):                          # initial + up to 2 retries
        last = C.groq_chat(client, system, user, temperature=0.9, max_tokens=320)
        cands = [c for c in _parse_caption_lines(last) if c][:N_CANDIDATES]
        if len(cands) >= 3:
            return cands
    if not _LOGGED_BAD_GROQ["done"]:                   # diagnose the failure ONCE
        C.warn("Groq caption output unparseable after 3 attempts — raw sample:\n"
               + (last[:500] if last else "<empty>"))
        _LOGGED_BAD_GROQ["done"] = True
    C.warn(f"caption fallback (templates) for moment {moment.get('id')}.")
    return _offline_candidates(moment)


# --- per-platform text ---------------------------------------------------------
def _required_hashtags(rules):
    tags = []
    for r in rules.get("required_elements", []):
        if r["type"] == "hashtag":
            tag = r["detail"].replace("Include", "").strip()
            if tag:
                tags.append(tag)
    return tags


def platform_text(caption, rules, banned):
    req = _required_hashtags(rules)
    base_tags = req + ["#fyp", "#clips", "#viral"]
    # filter tags through banned words too
    tags = [t for t in dict.fromkeys(base_tags) if not banned_hit(t, banned)]
    shorts = caption.rstrip("😭✌️🔥💀🙏✋ ").strip() or caption
    out = {
        "tiktok_caption": (caption + "  " + " ".join(req)).strip(),
        "shorts_title": shorts[:90],
        "reels_hashtags": tags,
        "suggested_post_window": "evening 6–9pm local (peak engagement)",
    }
    # final safety: none of the emitted text may carry a banned word
    for k in ("tiktok_caption", "shorts_title"):
        if banned_hit(out[k], banned):
            out[k] = caption
    return out


def _strip_symbols(text):
    """Drop every non-ASCII char (all emoji/symbols) and collapse the gaps — the
    guaranteed no-emoji gate on the caption text itself (cut.py also strips at render)."""
    ascii_only = (text or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s{2,}", " ", ascii_only).strip()


def title_case(text):
    """Capitalize the first letter of every word (e.g. 'wait for the jump' -> 'Wait For
    The Jump'). Only a leading ALPHA char is upper-cased, so digit-led words stay natural
    ('2nd' -> '2nd', '38kg' -> '38kg'); the rest of each word is untouched ('THIS' stays
    'THIS')."""
    def cap(w):
        return (w[0].upper() + w[1:]) if w[:1].isalpha() else w
    return " ".join(cap(w) for w in text.split())


def finalize_caption(text):
    """Every caption that reaches a clip goes through here: emoji/symbols stripped,
    then forced to Title Case. Applied to ALL clips, template fallbacks included."""
    return title_case(_strip_symbols(text)) or "Clip"


def run(state):
    selected = C.load_json(C.SELECTED_JSON)
    if not selected:
        C.fail("campaign/selected.json missing — run the select stage first.")
    rules = C.load_json(C.RULES_JSON) or {}
    # Gauntlet screens both banned words AND banned topics from intake's analysis.
    banned = list(rules.get("banned_words", C.DEFAULT_BANNED_WORDS)) + list(rules.get("banned_topics", []))
    campaign = rules.get("campaign", state.get("campaign") or "campaign")
    # Per-source transcript, so a text-less audio_spike can borrow the caster's nearby
    # reaction and get a SPECIFIC caption instead of a generic template.
    moments_doc = C.load_json(C.MOMENTS_JSON) or {}
    tr_by_source = {s["source"]: s.get("transcript", []) for s in moments_doc.get("sources", [])}
    client = C.groq_client()
    if client is None:
        C.warn("offline mode — generating captions from templates (no Groq).")

    clips = []
    for idx, m in enumerate(selected["selected"]):
        if client is not None:
            if idx:
                time.sleep(CAPTION_DELAY_SECONDS)   # respect free-tier rate limits
            C.log(f"captions: clip {idx + 1}/{len(selected['selected'])} (moment {m['id']}).")
        event = (m.get("text") or "").strip() or _nearby_transcript(
            tr_by_source.get(m["source"], []), float(m["start"]), float(m["end"]))
        cands = _offline_candidates(m) if client is None else _groq_candidates(
            client, campaign, m, "stakes+outcome+emotion, specific to the clip", event)
        # normalize: one line, lowercase energy (flzsh)
        cands = [" ".join(c.split()).lower() for c in cands if c and c.strip()]
        # gauntlet 1: banned words (HARD — these can never ship)
        banned_clean, killed = gauntlet(cands, banned)
        # gauntlet 2: quality (max 8 words, no describing, single line, has a hook)
        kept = []
        for c in banned_clean:
            reason = quality_kill(c)
            if reason:
                killed.append({"caption": c, "reason": reason})
            else:
                kept.append(c)
        # Prefer hook-passing captions; but if the quality gate empties the pool, keep the
        # best banned-clean Groq line (still SPECIFIC to this clip) rather than dropping to
        # a generic template. Generic template is the last resort only when Groq gave us
        # nothing usable at all.
        if kept:
            pool = kept
        elif banned_clean:
            C.warn(f"moment {m['id']}: no candidate hit a hook pattern — keeping best raw "
                   f"Groq caption (specific to the clip) over a generic template.")
            pool = banned_clean
        else:
            C.warn(f"moment {m['id']}: no usable Groq caption — generic curiosity fallback.")
            pool = ["you have to see this"]
        ranked = sorted(pool, key=score_caption, reverse=True)
        # ALL captions on ALL clips: emoji-stripped + Title Case, no exceptions.
        best = finalize_caption(ranked[0])
        variant = finalize_caption(ranked[1]) if len(ranked) > 1 else None
        clips.append({
            "moment_id": m["id"], "source": m["source"],
            "start": m["start"], "end": m["end"], "type": m["type"],
            "peak": m.get("peak"),                # cold-open anchor for the cut stage
            "score": m.get("score"), "reason": m.get("reason"),
            "candidates": cands, "killed": killed,
            "caption": best, "variant": variant,
            **platform_text(best, rules, banned),
        })

    C.save_json(C.CAPTIONS_JSON, {"campaign": campaign, "clips": clips})
    C.mark_stage(state, "captions", clips=len(clips), killed=sum(len(c["killed"]) for c in clips))
    C.log(f"captions done: {len(clips)} clip(s); "
          f"{sum(len(c['killed']) for c in clips)} candidate(s) killed by rules.")


if __name__ == "__main__":
    run(C.load_state())
