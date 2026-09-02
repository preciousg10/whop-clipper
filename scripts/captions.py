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
# LAZY CLICKBAIT (FIX 3): pure-bait phrasings that promise a reveal WITHOUT saying anything
# concrete ("X just said WHAT", "you won't believe", "you'll never guess", "wait till you see
# what", "the reason why"). These read as bait and carry zero clip specificity — killed so the
# caption is forced to use a real detail from the transcript instead. Note the terminal "…said
# WHAT" form is anchored to the END so a normal "said what he meant" is NOT caught.
BAIT_RE = re.compile(
    r"((^|\b)(just\s+)?(said|says|say|did|does|do|goes)\s+what\b[\s\W]*$)|"
    r"\b(you'?ll never guess|you wo?n'?t believe|wo?n'?t believe (this|that|what|it|how|why)|"
    r"will (shock|surprise|blow) (you|your mind|the world)|what happens? next|"
    r"the reason (why|is why)|wait ?(til+|till) you (see|hear) what|"
    r"number \w+ will (shock|surprise))\b", re.I)

# --- hook GROUNDING (Task 2: no invented / mis-transcribed nouns) --------------
# A hook must be about what THIS clip actually contains. A hook may freely use structural
# hook words + generic reactions (below); but any OTHER content word must appear in the
# moment's transcript. A specific word that is neither — a name/brand/object the model
# invented or a mis-transcription ("hibush"), or an example-bleed noun ("hamster" on a
# jewelry clip) — marks the candidate as UNGROUNDED, and we drop it rather than ship
# catchy nonsense. (Only enforced when we actually have transcript to ground on.)
GENERIC_HOOK_VOCAB = frozenset("""
a an the this that these those it its he she her him his they them their you your yours we
us our my mine i me is are was were be been being am do does did doing done has have had
having will would can could should shall may might must cant cannot dont doesnt didnt isnt
arent wasnt werent wont couldnt shouldnt wouldnt aint not no nope yes yep and or but so
then than as at by for from in into of off on onto out to up down with without within over
under above below near about after before again just even still only really actually
literally lowkey highkey fr ong deadass bro bruh nah yo omg lol lmao lmfao istg pov wait
waiting watch watching keep keeps look looks looking see seen how why what who when where
which whose whats hows whys whos way ways shot real unreal insane crazy craziest wild
wildest nuts mad diabolical unserious chaos chaotic peak wilding goes go going gone went
here there comes coming came told tell telling gonna tryna wanna finna yall till until
moment moments thing things stuff someone something anything nothing everything everyone
everybody nobody anyone first last next best worst most least more less super so very too
much many few big small huge tiny new old good bad better worse wow whoa damn hell heck
bruv fam man dude guy guys girl girls people ever never always almost about gotta got get
gets getting make makes made makin making happen happens happening happened turn turns
turned drop drops dropped pull pulls pulled hold holds held put puts break breaks broke
run runs ran hit hits win wins won lose loses lost end ends ended way thats theres
heres lets let bruhh nahh deadset ongod
""".split())

_GROUND_WORD_RE = re.compile(r"[a-z']+")


def _transcript_vocab(event):
    """Lowercase word set of the moment's transcript/event text — the words a hook is
    allowed to be specific about."""
    return {w.strip("'") for w in _GROUND_WORD_RE.findall((event or "").lower())}


def _word_grounded(w, tvocab):
    """True if caption word `w` is supported by the transcript. Exact match, or a >=4-char
    shared prefix so simple morphology (diamond/diamonds, rolex/rolexes) still counts."""
    if w in tvocab:
        return True
    if len(w) >= 4:
        p = w[:4]
        for tv in tvocab:
            if tv.startswith(p) or (len(tv) >= 4 and w.startswith(tv[:4])):
                return True
    return False


def _ungrounded_terms(caption, tvocab):
    """Content words in `caption` that are neither generic hook vocabulary nor present in
    the transcript — i.e. specifics the model likely invented/mis-transcribed. Tokens <=2
    chars and pure numbers are ignored (never 'names')."""
    out = []
    for w in _GROUND_WORD_RE.findall((caption or "").lower()):
        w = w.strip("'")
        if len(w) <= 2 or w in GENERIC_HOOK_VOCAB:
            continue
        if not _word_grounded(w, tvocab):
            out.append(w)
    return out


def _has_specific_term(caption, tvocab):
    """True if the caption carries at least ONE clip-specific content word — a transcript word that
    isn't generic hook vocabulary. FIX A: a caption with NONE is a generic anchor template ('nah
    this is actually crazy', 'no way that just happened') and must never win over a real hook when
    we have a transcript to be specific about."""
    for w in _GROUND_WORD_RE.findall((caption or "").lower()):
        w = w.strip("'")
        if len(w) <= 2 or w in GENERIC_HOOK_VOCAB:
            continue
        if _word_grounded(w, tvocab):
            return True
    return False


# Neutral hooks that assert NOTHING specific — every word is generic hook vocabulary, so
# they are always grounded. Last-resort ONLY: used when the clip has no transcript to build a
# specific hook from (a text-less audio spike) — a specific hook (below) is always preferred.
GROUNDED_FALLBACKS = [
    "wait for the end 👀",
    "you have to see this 😳",
    "nah this is actually crazy 😭",
    "why did this even happen 😭",
    "the way this ends is diabolical 💀",
    "tell me why this happened 😭",
    "no shot this just happened 😳",
    "how is this even real 😭",
]


def _specific_fallbacks(event, banned=()):
    """CLIP-SPECIFIC last-resort hooks built from THIS clip's transcript (FIX A) — used instead of
    the generic GROUNDED_FALLBACKS whenever no Groq caption survives but we DO have transcript, so
    a campaign that doesn't require a template never ships a generic anchor. Pulls the clip's first
    few salient content words and wraps each in a short hook that still hits a hook pattern.
    Screened against banned/restriction terms (FIX B). Returns [] when there's no transcript."""
    seen, terms = set(), []
    for w in _GROUND_WORD_RE.findall((event or "").lower()):
        w = w.strip("'")
        if len(w) < 4 or w in GENERIC_HOOK_VOCAB or w in seen:
            continue
        seen.add(w)
        terms.append(w)
        if len(terms) >= 4:
            break
    if not terms:
        return []
    a = terms[0]
    b = terms[1] if len(terms) > 1 else a
    cands = [
        f"wait till you hear about {a} 👀",
        f"why {a} changes everything 😳",
        f"no way this is about {a} 😭",
        f"how {a} actually works 👀",
        f"tell me why {b} matters 😭",
    ]
    return [c for c in cands if not banned_hit(c, banned)]

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
    if BAIT_RE.search(cap):
        return "lazy clickbait (no clip specifics — use a real detail from the transcript)"
    if not HOOK_RE.search(cap):
        return "no hook pattern (question/stakes/disbelief/controversy/direct address)"
    if GIVEAWAY_RE.search(cap) and not HOOK_PATTERNS["question"].search(cap):
        return "past-tense summary gives the payoff away"
    return None


def _hook_prefix(cap, n=2):
    """The first `n` alphabetic words of a hook — its structural signature ('how did', 'no way',
    'wait for', 'this should'). FIX 6 rotates on this so one phrasing (e.g. 'how did this even
    happen', which dominated ~half a batch) can't repeat across many clips."""
    return " ".join(re.findall(r"[a-z']+", (cap or "").lower())[:n])


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


def _clip_window_transcript(tr_segs, start, end, limit=600):
    """The transcript spoken strictly WITHIN this clip's own window [start, end] — nothing from
    the rest of the episode. FIX 2: the caption must hook on what actually happens IN this span, so
    the LLM is fed ONLY the words inside the clip, never padded surrounding context or the whole
    show. Empty for a text-less audio-spike moment (caller then falls back to nearby speech)."""
    parts = [s["text"] for s in tr_segs
             if s.get("text") and s.get("end", 0) > start and s.get("start", 0) < end]
    return " ".join(parts).strip()[:limit]


# --- non-speech sound labels (karaoke *scream* / *laughing* fills) --------------
# Whisper only transcribes SPEECH, so a scream / laugh / crash leaves a silent gap in the
# word-level karaoke. We fill that gap with a SHORT accurate descriptor ("*scream*"), but
# only for a genuine non-speech beat and only when Groq can tell what it is from context —
# never a random guess. Detection here (captions stage, where Groq already runs); cut.py
# maps the stored beats through the clip reorder and burns them (staying zero-Groq).
SOUND_FX_WINDOW = 22.0        # seconds around the moment to scan for non-speech spikes
SOUND_FX_MAX = 3              # cap labels per clip (keeps it sparse + few tokens)


def _speech_covers(tr_segs, a, b, pad=0.4):
    """True if any transcribed speech overlaps [a-pad, b+pad] — i.e. the spike is NOT a
    non-speech beat (someone is talking there)."""
    return any((s.get("text") or "").strip() and s.get("end", 0) >= a - pad and s.get("start", 0) <= b + pad
               for s in tr_segs)


def _nonspeech_beats(m, tr_segs, spikes):
    """Loud audio_spike moments near this clip that carry NO speech — candidate non-speech
    beats. Returns the strongest few, source-absolute, sorted by time."""
    lo, hi = float(m["start"]) - SOUND_FX_WINDOW, float(m["end"]) + SOUND_FX_WINDOW
    beats = []
    for s in spikes:
        peak = s.get("peak", s["start"])
        if peak is None or not (lo <= float(peak) <= hi):
            continue
        if _speech_covers(tr_segs, float(s["start"]), float(s["end"])):
            continue
        beats.append(s)
    beats.sort(key=lambda s: s.get("intensity", 0), reverse=True)
    beats = beats[:SOUND_FX_MAX]
    beats.sort(key=lambda s: float(s["start"]))
    return beats


def _label_sound_beats(client, context, beats):
    """Ask Groq for a SHORT accurate descriptor per non-speech beat, or null when it can't
    tell (no random guessing). Returns [{start,end,peak,label}] for the labeled ones only."""
    if not client or not beats:
        return []
    payload = [{"i": i, "intensity": b.get("intensity")} for i, b in enumerate(beats)]
    system = (
        "You label NON-SPEECH audio moments for karaoke captions on a short clip. You're given "
        "the surrounding transcript (what was said around the sound) and a list of loud moments "
        "where NO words were spoken. For EACH moment, output a SHORT lowercase descriptor of the "
        "sound (1-2 words, e.g. scream, screaming, laughing, gasp, cheering, crowd goes wild, "
        "crash, groan, scared) — but ONLY if the surrounding context makes the sound reasonably "
        "clear. If you CANNOT tell what the sound is, return null for that moment. NEVER guess "
        "randomly — an accurate null beats a wrong label. Return ONLY a JSON array, one object "
        'per moment: {"i": <index>, "label": <string or null>}. No prose.')
    user = (f"Surrounding transcript: {context!r}\n"
            f"Audience: {C.AUDIENCE_CONTEXT}\n"
            f"Non-speech moments (index, loudness intensity): {json.dumps(payload)}")
    raw = C.llm_chat(client, system, user, temperature=0.3, max_tokens=200)
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    by_i = {int(x["i"]): x.get("label") for x in arr if isinstance(x, dict) and "i" in x}
    out = []
    for i, b in enumerate(beats):
        lbl = by_i.get(i)
        if not lbl or not str(lbl).strip() or str(lbl).strip().lower() in ("null", "none", "unknown"):
            continue
        out.append({"start": float(b["start"]), "end": float(b["end"]),
                    "peak": b.get("peak"), "label": str(lbl).strip()})
    return out


# --- LLM-chosen cold-open moment (tease by CONTENT, not loudness) --------------
# cut.py normally anchors the cold-open teaser on the loudest audio second. But the best tease is
# often the most SUSPENSEFUL / craziest thing SAID, which can be spoken calmly (low audio energy)
# and get missed — e.g. "we're gonna open up the box" builds curiosity but isn't loud. Here, where
# the LLM already reads the transcript, we also ask it for the single most hook-worthy LINE in the
# clip and store its timestamp; cut.py teases THAT instant (falling back to the audio peak when we
# can't get a usable one). This adds one small Groq call per clip that actually has dialogue.
HOOK_MOMENT_MAX_LINES = 40    # cap transcript lines sent (token budget)


def _clip_transcript_lines(tr_segs, start, end):
    """Timestamped transcript lines overlapping the clip window [start, end] (source-absolute),
    for the cold-open hook-moment pick. Deduped/ordered by time, capped for the token budget."""
    out = []
    for s in tr_segs:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        st, en = float(s.get("start", 0)), float(s.get("end", 0))
        if en >= start and st <= end:
            out.append({"t": round(st, 2), "text": txt})
    out.sort(key=lambda x: x["t"])
    return out[:HOOK_MOMENT_MAX_LINES]


def _pick_hook_moment(client, lines, campaign):
    """Ask the LLM which transcribed line best works as a COLD-OPEN tease — the single most
    suspenseful / curiosity-baiting / shocking instant to flash BEFORE the clip plays in order.
    Returns its source-absolute timestamp (float) or None. Returns None on none/parse failure so
    cut.py falls back to the audio peak — never a guess."""
    if not client or len(lines) < 2:
        return None
    payload = [{"i": i, "t": ln["t"], "text": ln["text"]} for i, ln in enumerate(lines)]
    system = (
        "You pick the single most HOOK-WORTHY instant in a short clip to use as a COLD-OPEN "
        "tease — the line that, flashed BEFORE the clip plays in order, creates the most "
        "suspense, curiosity, or shock and makes a viewer NEED to keep watching. It is NOT "
        "necessarily the loudest moment: a CALM line that builds suspense ('wait, what is "
        "that', 'we're about to open the box', 'no way he just said that') is often the best "
        "tease. Pick the line whose first couple seconds would STOP a scroll. Do NOT pick the "
        "actual payoff/punchline (that spoils it) — pick the SETUP that makes the payoff feel "
        'mandatory. Return ONLY JSON: {"i": <index of the chosen line>}, or {"i": null} if no '
        "line clearly stands out. No prose.")
    user = (f"Campaign: {campaign}\nAudience: {C.AUDIENCE_CONTEXT}\n"
            f"Transcript lines (index, time in seconds, text):\n{json.dumps(payload)}")
    raw = C.llm_chat(client, system, user, temperature=0.3, max_tokens=60)
    mm = re.search(r"\{.*\}", raw, re.S)
    if not mm:
        return None
    try:
        i = json.loads(mm.group(0)).get("i")
        i = int(i)
    except (Exception, TypeError, ValueError):
        return None
    if 0 <= i < len(lines):
        return float(lines[i]["t"])
    return None


def _groq_candidates(client, campaign, moment, style_notes, event="", emoji_in_caption=True,
                     onscreen_pattern=None):
    event = (event or moment.get("text") or "").strip()
    knowledge = C.load_knowledge()[:600]
    ev_line = (f"What happens / is said in THIS clip (reference it specifically): {event!r}\n"
               if event else
               "This clip has no transcript — keep the text specific to the visible moment, "
               "not generic.\n")
    # REQUIRED ONSCREEN-TEXT FORMAT (FIX 2): when the campaign mandates a hook wording pattern
    # (e.g. "Santa Cruz explains ___"), the hook MUST follow it — the five generic hook patterns
    # do NOT apply. Generate pattern-conforming lines completed from THIS clip's content.
    if onscreen_pattern:
        tmpls = onscreen_pattern.get("templates") or []
        system = (
            "You write the ONSCREEN TEXT (the top hook) burned onto a vertical short-form clip. "
            "This campaign REQUIRES the onscreen text to follow a FIXED format — you MUST use it "
            "verbatim as the opener or the post is REJECTED. "
            + (f"Required format: {onscreen_pattern.get('description')} " if onscreen_pattern.get("description") else "")
            + "Use ONE of these exact openers, then COMPLETE it with what THIS clip is about:\n"
            + "\n".join(f"  - {t}" for t in tmpls) + "\n"
            "RULES: keep the opener wording EXACTLY, then finish it in ~2-6 words on the clip's "
            "real topic. Short, punchy, ONE line, max 10 words, NO hashtags, NO emoji. "
            "GROUNDING (CRITICAL): complete the opener using ONLY the topic/person/thing actually "
            "in the transcript below — never invent a subject that isn't in the clip. "
            f"Return {N_CANDIDATES} options, ONE PER LINE — no numbering, no quotes.")
        user = (f"Audience: {C.AUDIENCE_CONTEXT}\n\n"
                + (f"Campaign knowledge (apply this):\n{knowledge}\n\n" if knowledge else "")
                + f"Campaign: {campaign}\n" + ev_line
                + f"Write {N_CANDIDATES} onscreen-text options, each STARTING with one of the "
                  f"required openers and completed from THIS clip's actual content.")
        last = ""
        for _attempt in range(3):
            last = C.llm_chat(client, system, user, temperature=0.8, max_tokens=320)
            cands = [c for c in _parse_caption_lines(last) if c][:N_CANDIDATES]
            if len(cands) >= 2:
                return cands
        C.warn(f"caption (required-format) unparseable for moment {moment.get('id')} — synthesizing.")
        return [_synth_required(tmpls, event)]
    emoji_rule = (
        "all lowercase, end with 1-2 emoji as punctuation (from: 😭 💀 🔥 👀 ✌️ 🥀 😳 🤣), "
        if emoji_in_caption else
        "all lowercase, NO emoji, ")
    system = (
        "You write TOP captions for viral vertical short-form clips. The "
        "caption's ONLY job is to make the payoff feel MANDATORY to watch, and it MUST "
        "reference the ACTUAL event in THIS clip (use the transcript / what happens), "
        "NOT a generic phrase. "
        "THIS CLIP ONLY (CRITICAL): the transcript below is the ENTIRE clip — caption what "
        "happens IN it, and NOTHING about the wider episode, show, or format. NEVER use a generic "
        "episode/segment/format label ('rapidfire round', 'Q&A', 'the podcast', 'this interview', "
        "'story time'). If the clip is about houses, the caption is about houses; if it's a story "
        "about money, hook on that specific thing. The campaign knowledge is for RULES/banned-word "
        "compliance only — do NOT let it turn the caption into a description of the show. "
        "RULES: ONE line, MAX 8 words, casual grammar, " + emoji_rule +
        "NO hashtags. Use ONE of these hook TECHNIQUES, but write it ENTIRELY in THIS clip's own "
        "words/subject — do NOT paste a template prefix:\n"
        "  1) an open QUESTION about the specific claim/thing in the clip\n"
        "  2) the STAKES, or a bold curiosity-gap statement of the actual claim\n"
        "  3) DISBELIEF aimed at the specific thing said\n"
        "  4) CONTROVERSY over the specific claim\n"
        "  5) DIRECT ADDRESS to watch the specific payoff\n"
        "BANNED BOILERPLATE (FIX A): NEVER open with an empty generic prefix like 'how did this "
        "even happen', 'no way that just happened', 'wait for the very last second', 'this should "
        "not have counted', 'nah this is actually crazy', or '$10k on the line'. Those waste the "
        "word budget and say nothing about the clip. Every hook MUST name the clip's real subject "
        "in <= 8 words (e.g. if the clip is about melatonin, the hook says melatonin).\n"
        "GROUNDING (CRITICAL): use ONLY the names, people, brands, places, and objects that "
        "appear in the transcript below. NEVER invent, guess, or borrow a name from these "
        "instructions — if the transcript doesn't name it, don't name it (say 'this', "
        "'that', 'him', 'them' instead). A specific noun that isn't in the clip is an "
        "automatic reject. When in doubt, be PLAIN and ACCURATE, not catchy and wrong.\n"
        "BANNED: vague filler ('what just happened', bare 'wait for it'), descriptions, "
        "past-tense summaries, or GIVING AWAY the payoff. "
        "NO LAZY CLICKBAIT (FIX 3): never use empty bait that promises a reveal without saying "
        "anything concrete — 'X just said WHAT', 'you won't believe', 'you'll never guess', 'wait "
        "till you see what', 'the reason why', 'this will shock you'. Those are an automatic "
        "reject. Instead pull a CONCRETE specific from the clip (a number, an object, a name that "
        "IS in the transcript, the actual claim) and hook on THAT — curiosity-driven but grounded "
        "in what is really said. "
        "Be SPECIFIC to this clip. Respect the campaign banned words/topics in the "
        "knowledge below. "
        f"Return {N_CANDIDATES} captions, ONE PER LINE — no numbering, no quotes, no JSON.")
    user = (f"Audience: {C.AUDIENCE_CONTEXT}\n\n"
            + (f"Campaign knowledge (apply this):\n{knowledge}\n\n" if knowledge else "")
            + f"Campaign: {campaign}\nStyle notes: {style_notes}\n"
            + ev_line
            + f"Give {N_CANDIDATES} caption lines, each using one hook pattern and "
              f"referencing the actual moment.")
    last = ""
    for _attempt in range(3):                          # initial + up to 2 retries
        last = C.llm_chat(client, system, user, temperature=0.9, max_tokens=320)
        cands = [c for c in _parse_caption_lines(last) if c][:N_CANDIDATES]
        if len(cands) >= 3:
            return cands
    if not _LOGGED_BAD_GROQ["done"]:                   # diagnose the failure ONCE
        C.warn("Groq caption output unparseable after 3 attempts — raw sample:\n"
               + (last[:500] if last else "<empty>"))
        _LOGGED_BAD_GROQ["done"] = True
    C.warn(f"caption fallback (templates) for moment {moment.get('id')}.")
    return _offline_candidates(moment)


# --- required onscreen-text FORMAT (FIX 2) -------------------------------------
# When intake extracts a mandatory hook/onscreen-text pattern into rules.json
# (required_onscreen_text_pattern), the caption stage must MAKE the hook follow it, filled from the
# clip's real content — the generic five-hook style is bypassed for that campaign.
def _required_onscreen(rules):
    """The campaign's required onscreen-text pattern as {description, templates}, or None.

    FIX A: templates are used ONLY when required_onscreen_text_pattern.required is literally True.
    When required is false/absent (most campaigns) this returns None and captions are generated as
    PURE clip-specific hooks (no generic anchor template). Derived fresh from the passed-in rules
    every call — nothing is cached, so a prior campaign's format can never leak into this one."""
    p = (rules or {}).get("required_onscreen_text_pattern") or {}
    if isinstance(p, dict) and p.get("required") is True:
        tmpls = list(p.get("templates") or p.get("examples") or [])
        if tmpls:
            return {"description": p.get("description", ""), "templates": tmpls}
    return None


def _opener_words(template):
    """The FIXED opener words of a template — the text before the fill placeholder
    ('Santa Cruz explains ___' -> ['santa','cruz','explains'])."""
    head = re.split(r"_{2,}|\[|\{|<|\.\.\.|…", template or "")[0]
    return re.sub(r"[^a-z0-9 ]", " ", head.lower()).split()


def _required_anchor(templates):
    """Longest common leading word-run shared by every template — the brand anchor the hook must
    start with ('Santa Cruz [verb]…' family -> ['santa','cruz']). Empty if templates disagree."""
    seqs = [_opener_words(t) for t in templates if _opener_words(t)]
    if not seqs:
        return []
    pref = seqs[0]
    for s in seqs[1:]:
        i = 0
        while i < len(pref) and i < len(s) and pref[i] == s[i]:
            i += 1
        pref = pref[:i]
    return pref


def _matches_required(text, templates):
    """True if `text` begins with the required brand anchor (so it conforms to the mandated
    'Santa Cruz …' onscreen format). Anchor-based, so any allowed verb ('explains'/'reveals'/…)
    passes while an off-format line ('nah this is crazy') is rejected."""
    anchor = _required_anchor(templates)
    if not anchor:
        return True                                    # no shared anchor to enforce
    words = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()
    return words[:len(anchor)] == anchor


_SYNTH_STOP = {"that", "this", "them", "they", "just", "like", "really", "actually", "gonna",
               "yeah", "know", "what", "when", "your", "youre", "with", "have", "here", "there",
               "about", "because", "would", "could", "should", "their", "then", "than", "into"}


def _completion_terms(caption, anchor):
    """Content words in the caption BEYOND the required opener/anchor — the part that must carry
    THIS clip's specifics (not just the fixed template). Drops the anchor words, short tokens, and
    generic stopwords, so an anchor-only line ('Santa Cruz explains') yields []."""
    aw = {a.lower() for a in (anchor or [])}
    return [w for w in re.findall(r"[a-z0-9']+", (caption or "").lower())
            if w not in aw and len(w) > 2 and w not in _SYNTH_STOP]


def _synth_required(templates, event):
    """Guaranteed-valid onscreen text when the LLM output can't be used: the first template's
    opener + a couple of grounded content words from the clip ('Santa Cruz explains lab results').
    Warns (FIX 5) when the clip transcript yields NO specific word — the format is still emitted
    for compliance, but with only a minimal filler that should be checked manually."""
    opener = re.split(r"_{2,}|\[|\{|<|\.\.\.|…", templates[0])[0].strip().rstrip(":—-").strip()
    words = [w for w in re.findall(r"[A-Za-z]+", event or "") if len(w) > 3
             and w.lower() not in _SYNTH_STOP]
    if not words:
        C.warn("required-format hook: no clip-specific word in the transcript to complete the "
               "template — emitting the opener with minimal filler (verify manually).")
    tail = " ".join(words[:2]) if words else "this"
    return f"{opener} {tail}".strip()


def _required_format_clip(client, campaign, m, event, onscreen, banned):
    """Build the hook for a campaign that MANDATES an onscreen-text format. Returns
    (best, variant, cands, killed). Enforces: the required opener, banned words, and grounding on
    the completion; skips the generic five-hook gauntlet (the format IS the structure)."""
    tmpls = onscreen["templates"]
    cands = _groq_candidates(client, campaign, m, "", event, emoji_in_caption=False,
                             onscreen_pattern=onscreen)
    cands = [" ".join(c.split()) for c in cands if c and c.strip()]
    banned_clean, killed = gauntlet(cands, banned)
    # keep only lines that actually follow the required format
    conforming = [c for c in banned_clean if _matches_required(c, tmpls)]
    for c in banned_clean:
        if not _matches_required(c, tmpls):
            killed.append({"caption": c, "reason": "does not follow required onscreen-text format"})
    # grounding on the FILLED part (don't let the completion invent a subject)
    tvocab = _transcript_vocab(event)
    anchor_words = _required_anchor(tmpls)
    anchor = set(anchor_words)
    if len(tvocab) >= 3:
        grounded = [c for c in conforming
                    if not _ungrounded_terms(" ".join(w for w in c.split()
                                                       if w.lower() not in anchor), tvocab)]
        conforming = grounded or conforming
    # FIX 5 — COMBINE FORMAT + CLIP CONTENT, never ship the bare template. Require the completion
    # to carry a real clip-specific term (a grounded one when we have transcript to ground on).
    def _is_specific(c):
        terms = _completion_terms(c, anchor_words)
        if not terms:
            return False                               # anchor-only → just the generic template
        if len(tvocab) >= 3:
            return any(_word_grounded(t, tvocab) for t in terms)
        return True
    specific = [c for c in conforming if _is_specific(c)]
    if specific:
        conforming = specific
    else:
        if conforming:
            C.warn(f"moment {m.get('id')}: required-format hooks had no clip-specific completion "
                   f"(template-only) — synthesizing one grounded in the transcript.")
        conforming = []                                # force the grounded synth fill below
    pool = conforming or [_synth_required(tmpls, event)]
    # prefer the punchiest conforming line (short, not a bare opener)
    pool.sort(key=lambda c: (len(c.split()) >= 4, -len(c.split())), reverse=True)
    best = finalize_caption(pool[0], emoji_in_caption=False)
    variant = finalize_caption(pool[1], emoji_in_caption=False) if len(pool) > 1 else None
    return best, variant, cands, killed


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


def titlecase(text):
    """Title Case (Like This): capitalize the first letter of each word, leaving the rest of
    the word untouched so contractions and emoji survive ("don't" -> "Don't", "😭fire" ->
    "😭Fire", a masked "b**" -> "B**"). A leading number leaves the word alone ("10v1").
    This is the user-preferred caption/subtitle casing — it OVERRIDES the old lowercase
    default (kept as a single choke point so both the hook caption and burned subtitles
    agree)."""
    def cap(word):
        for i, ch in enumerate(word):
            if ch.isalnum():
                return (word[:i] + ch.upper() + word[i + 1:]) if ch.isalpha() else word
        return word
    return " ".join(cap(w) for w in (text or "").split())


def finalize_caption(text, emoji_in_caption=True):
    """Every caption that reaches a clip is normalized here:

      - TITLE CASE (Like This), per user preference — overrides the old flzsh lowercase
        default. Applied via titlecase() so both captions and burned subtitles share it.
      - Emoji are punctuation: kept when `emoji_in_caption` (default), stripped when off.

    The banned-word gauntlet and hook-pattern gate have already run upstream; this only
    fixes case/emoji. Rendering (cut.render_caption_png) drops any emoji the font can't
    draw, so a preserved emoji never becomes a tofu box."""
    text = " ".join((text or "").split())
    if not emoji_in_caption:
        text = _strip_symbols(text)
    text = titlecase(text).strip()
    return text or "clip"


def _generic_caption(cands, banned, event, m, hook_prefix_use, hook_prefix_cap, emoji_in_caption):
    """The default flzsh five-hook caption path: banned + quality + grounding gauntlets, then
    pick the best specific hook under the batch-variety cap. Returns (best, variant, killed).
    Used when the campaign has NO required onscreen-text format."""
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
    # GROUNDING (Task 2): a hook may only be specific about words that are in THIS moment's
    # transcript. Drop candidates that name something invented/mis-transcribed. Only enforced
    # when we actually have transcript to ground on.
    tvocab = _transcript_vocab(event)
    enforce_ground = len(tvocab) >= 3

    def _grounded_only(cs):
        if not enforce_ground:
            return list(cs)
        return [c for c in cs if not _ungrounded_terms(c, tvocab)]

    def _specific_only(cs):
        # FIX A: require POSITIVE clip specificity (a transcript content word), so a pure generic
        # template is never chosen over a real hook when we have a transcript to be specific about.
        if not enforce_ground:
            return list(cs)
        return [c for c in cs if _has_specific_term(c, tvocab)]

    kept_g, clean_g = _grounded_only(kept), _grounded_only(banned_clean)
    kept_gs, clean_gs = _specific_only(kept_g), _specific_only(clean_g)
    # FIX A: when the campaign requires NO onscreen template, the last resort is a hook built from
    # THIS clip's transcript — never a generic anchor. Generic templates only when there is no
    # transcript at all (a text-less audio spike). All fallbacks are banned-screened (FIX B).
    specific_fb = _specific_fallbacks(event, banned)
    fallbacks = specific_fb or [f for f in GROUNDED_FALLBACKS if not banned_hit(f, banned)]
    if kept_gs:
        pool = kept_gs                                   # grounded + hook-passing + specific (best)
    elif clean_gs:
        C.warn(f"moment {m['id']}: no hook-passing caption — using best grounded, clip-specific "
               f"Groq line (specific + accurate over a template).")
        pool = clean_gs
    elif kept_g:
        pool = kept_g                                    # grounded + hook-passing (no explicit term)
    elif enforce_ground:
        C.warn(f"moment {m['id']}: no clip-specific caption survived — synthesizing a hook from "
               f"the transcript ({'clip-specific' if specific_fb else 'no transcript terms'}).")
        pool = fallbacks
    elif kept:
        pool = kept
    elif banned_clean:
        C.warn(f"moment {m['id']}: no candidate hit a hook pattern — keeping best raw "
               f"Groq caption (specific to the clip) over a generic template.")
        pool = banned_clean
    else:
        C.warn(f"moment {m['id']}: no usable Groq caption — curiosity fallback.")
        pool = fallbacks
    ranked = sorted(pool, key=score_caption, reverse=True)
    # FIX 6 — HOOK VARIETY: prefer the best specific hook whose structure is still under the batch
    # cap; once every specific structure is capped, fall to the least-used fallback hook.
    best_raw = next((cap for cap in ranked
                     if hook_prefix_use[_hook_prefix(cap)] < hook_prefix_cap), None)
    if best_raw is None:
        best_raw = min(fallbacks or ranked,
                       key=lambda cap: (hook_prefix_use[_hook_prefix(cap)], -score_caption(cap)))
    hook_prefix_use[_hook_prefix(best_raw)] += 1
    alt = ([cap for cap in ranked if _hook_prefix(cap) != _hook_prefix(best_raw)]
           or [cap for cap in fallbacks if _hook_prefix(cap) != _hook_prefix(best_raw)])
    best = finalize_caption(best_raw, emoji_in_caption)
    variant = finalize_caption(alt[0], emoji_in_caption) if alt else None
    return best, variant, killed


def run(state):
    selected = C.load_json(C.SELECTED_JSON)
    if not selected:
        C.fail("campaign/selected.json missing — run the select stage first.")
    rules = C.load_json(C.RULES_JSON) or {}
    # Gauntlet screens both banned words AND banned topics from intake's analysis.
    banned = list(rules.get("banned_words", C.DEFAULT_BANNED_WORDS)) + list(rules.get("banned_topics", []))
    campaign = rules.get("campaign", state.get("campaign") or "campaign")
    cfg = state.get("config", {})
    emoji_in_caption = bool(cfg.get("emoji_in_caption", True))   # flzsh: emoji as punctuation
    # REQUIRED ONSCREEN-TEXT FORMAT (FIX 2): if the campaign mandates a hook wording pattern, the
    # hook must FOLLOW it (filled from each clip), not the generic flzsh style.
    onscreen = _required_onscreen(rules)
    if onscreen:
        C.log(f"captions: campaign REQUIRES onscreen-text format — hooks will follow it "
              f"({len(onscreen['templates'])} template(s): {onscreen['templates'][:3]}…).")
    else:
        C.log("captions: no required onscreen-text format — generating PURE clip-specific hooks "
              "from each clip's transcript (no generic anchor template).")
    # Per-source transcript, so a text-less audio_spike can borrow the caster's nearby
    # reaction and get a SPECIFIC caption instead of a generic template.
    moments_doc = C.load_json(C.MOMENTS_JSON) or {}
    tr_by_source = {s["source"]: s.get("transcript", []) for s in moments_doc.get("sources", [])}
    # Non-speech audio spikes per source — candidates for karaoke *scream* / *laughing* fills.
    spikes_by_source = {}
    for mm in moments_doc.get("moments", []):
        if mm.get("type") == "audio_spike":
            spikes_by_source.setdefault(mm["source"], []).append(mm)
    client = C.llm_client(cfg)
    if client is None:
        C.warn("offline mode — generating captions from templates (no LLM).")
    else:
        C.log(f"captions: LLM provider = {client.status()}")

    # RESUME (Unit 2b): reload any clips already captioned (checkpointed after each one) so a
    # prior DAILY-cap stop doesn't re-spend Groq on completed clips. We skip done moment ids,
    # checkpoint after every clip, and let a GroqDailyCapError propagate to run.py, which stops
    # resumably — the checkpoint below is already current when it fires.
    # DEFENSIVE (stale-captions guard): the resume checkpoint must match the CURRENT picks. If any
    # checkpointed clip references a moment id that is NOT in selected.json, select re-ran with new
    # picks and this partial is STALE — reusing it would caption the wrong moment (the m0290-vs-m0860
    # bug). Fail loud and regenerate from scratch rather than silently shipping stale captions.
    sel_ids = {m["id"] for m in selected["selected"]}
    partial = C.load_json(C.CAPTIONS_PARTIAL) or {}
    clips = list(partial.get("clips", []))
    stale = [c.get("moment_id") for c in clips if c.get("moment_id") not in sel_ids]
    if stale:
        C.warn(f"captions: checkpoint is STALE — {len(stale)} clip(s) reference moment id(s) "
               f"{stale} that are NOT in the current selected.json ({sorted(sel_ids)}). Select "
               f"re-ran with new picks; discarding the partial and regenerating from scratch.")
        clips = []
        C.CAPTIONS_PARTIAL.unlink(missing_ok=True)
    done_ids = {c.get("moment_id") for c in clips}
    total = len(selected["selected"])
    # FIX 6 — batch-wide hook-structure ledger (seeded from any resumed clips so diversity
    # survives a --resume) + a per-structure CAP. A specific hook keeps its structure until that
    # structure hits the cap; after that the clip takes a structurally-DIFFERENT neutral grounded
    # hook, so no single phrasing (e.g. 'how did') can dominate the batch.
    from collections import Counter
    hook_prefix_use = Counter(_hook_prefix(c.get("caption", "")) for c in clips)
    hook_prefix_cap = max(3, (total + 4) // 5)   # ceil(total/5): ~20% ceiling per structure
    if clips:
        C.log(f"captions: resuming — {len(clips)}/{total} clip(s) already done (checkpoint).")
    made_call = False
    for idx, m in enumerate(selected["selected"]):
        if m["id"] in done_ids:
            continue                                # already captioned on a prior run
        if client is not None:
            if made_call:
                time.sleep(CAPTION_DELAY_SECONDS)   # respect free-tier rate limits (between calls)
            C.log(f"captions: clip {idx + 1}/{total} (moment {m['id']}).")
            made_call = True
        # FIX 2: hook on THIS CLIP only. Prefer the transcript strictly inside the clip window
        # (nothing from the rest of the episode); fall back to the moment text, then — for a
        # text-less audio spike — the nearby reaction.
        tr_segs = tr_by_source.get(m["source"], [])
        event = (_clip_window_transcript(tr_segs, float(m["start"]), float(m["end"]))
                 or (m.get("text") or "").strip()
                 or _nearby_transcript(tr_segs, float(m["start"]), float(m["end"])))
        # REQUIRED-FORMAT PATH (FIX 2): campaign mandates a hook wording pattern → build a
        # pattern-conforming hook (own generator + acceptance) and skip the generic five-hook style.
        # Everything downstream (sound_fx, cold-open, platform text) is shared with the generic path.
        if onscreen and client is not None:
            best, variant, cands, killed = _required_format_clip(
                client, campaign, m, event, onscreen, banned)
        else:
            cands = _offline_candidates(m) if client is None else _groq_candidates(
                client, campaign, m, "stakes+outcome+emotion, specific to the clip", event,
                emoji_in_caption)
            # normalize: one line, lowercase energy (flzsh)
            cands = [" ".join(c.split()).lower() for c in cands if c and c.strip()]
            best, variant, killed = _generic_caption(
                cands, banned, event, m, hook_prefix_use, hook_prefix_cap, emoji_in_caption)
        # Non-speech sound labels for the karaoke (accurate, or nothing). Only fires when the
        # clip actually has a loud non-speech beat, and only adds ONE extra Groq call then.
        sound_fx = []
        beats = _nonspeech_beats(m, tr_by_source.get(m["source"], []),
                                 spikes_by_source.get(m["source"], []))
        if beats and client is not None:
            sound_fx = _label_sound_beats(client, event, beats)
            if sound_fx:
                C.log(f"captions: labeled {len(sound_fx)} non-speech beat(s) for {m['id']}: "
                      f"{[fx['label'] for fx in sound_fx]}")
        # COLD-OPEN HOOK MOMENT (tease by CONTENT): let the LLM pick the most suspenseful/
        # crazy LINE in the clip so cut.py teases THAT instant instead of the loudest second.
        # Only for clips with real dialogue; cut.py falls back to the audio peak otherwise.
        hook_moment = None
        if client is not None:
            # Restrict the LLM's choices to lines that will actually SHIP in the clip: mirror
            # cut.clip_bounds (expand by story pre/post, and when the beat exceeds clip_max center
            # the window on the peak). A line outside the shipped window can't be teased — cut.py
            # revalidates and falls back to the audio peak, but matching the window here keeps the
            # LLM pick usable instead of wasting it on a line that gets trimmed away.
            ws = float(m["start"]) - float(cfg.get("story_pre_seconds", 4))
            we = float(m["end"]) + float(cfg.get("story_post_seconds", 4))
            cmax = float(cfg.get("clip_max_seconds", 45))
            if we - ws > cmax:
                pk = m.get("peak")
                try:
                    center = float(pk) if pk is not None else (float(m["start"]) + float(m["end"])) / 2.0
                except (TypeError, ValueError):
                    center = (float(m["start"]) + float(m["end"])) / 2.0
                ws, we = center - cmax / 2.0, center + cmax / 2.0
            hook_lines = _clip_transcript_lines(tr_by_source.get(m["source"], []), ws, we)
            hook_moment = _pick_hook_moment(client, hook_lines, campaign)
            if hook_moment is not None:
                C.log(f"captions: cold-open hook moment for {m['id']} @ {hook_moment:.1f}s "
                      f"(LLM-chosen from {len(hook_lines)} line(s)).")
        clips.append({
            "moment_id": m["id"], "source": m["source"],
            "start": m["start"], "end": m["end"], "type": m["type"],
            "peak": m.get("peak"),                # audio-peak cold-open anchor (fallback)
            "hook_moment": hook_moment,           # LLM-chosen cold-open tease instant (preferred)
            "score": m.get("score"), "reason": m.get("reason"),
            "candidates": cands, "killed": killed,
            "caption": best, "variant": variant, "sound_fx": sound_fx,
            **platform_text(best, rules, banned),
        })
        # Checkpoint after EVERY clip so a daily-cap stop (or crash) resumes here, not from #1.
        C.save_json(C.CAPTIONS_PARTIAL, {"campaign": campaign, "clips": clips})

    C.save_json(C.CAPTIONS_JSON, {"campaign": campaign, "clips": clips})
    C.CAPTIONS_PARTIAL.unlink(missing_ok=True)       # stage complete — drop the checkpoint
    C.mark_stage(state, "captions", clips=len(clips), killed=sum(len(c["killed"]) for c in clips))
    C.log(f"captions done: {len(clips)} clip(s); "
          f"{sum(len(c['killed']) for c in clips)} candidate(s) killed by rules.")


if __name__ == "__main__":
    run(C.load_state())
