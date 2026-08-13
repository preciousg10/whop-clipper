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


# Neutral hooks that assert NOTHING specific — every word is generic hook vocabulary, so
# they are always grounded. Used only when no grounded Groq caption survives (better a
# plain accurate hook than a catchy invented one).
GROUNDED_FALLBACKS = [
    "wait for the end 👀",
    "you have to see this 😳",
    "watch this till the end 👀",
    "nah this is actually crazy 😭",
]

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


def _groq_candidates(client, campaign, moment, style_notes, event="", emoji_in_caption=True):
    event = (event or moment.get("text") or "").strip()
    emoji_rule = (
        "all lowercase, end with 1-2 emoji as punctuation (from: 😭 💀 🔥 👀 ✌️ 🥀 😳 🤣), "
        if emoji_in_caption else
        "all lowercase, NO emoji, ")
    system = (
        "You write TOP captions for viral vertical short-form clips. The "
        "caption's ONLY job is to make the payoff feel MANDATORY to watch, and it MUST "
        "reference the ACTUAL event in THIS clip (use the transcript / what happens), "
        "NOT a generic phrase. RULES: ONE line, MAX 8 words, casual grammar, " + emoji_rule +
        "NO hashtags. Every caption MUST use one of these five proven hook patterns:\n"
        "  1) open question — 'how did this even happen'\n"
        "  2) stakes — '$10k on the line and then THIS'\n"
        "  3) disbelief — 'no way that just happened'\n"
        "  4) controversy — 'this should NOT have counted'\n"
        "  5) direct address — 'wait for the very last second'\n"
        "GROUNDING (CRITICAL): use ONLY the names, people, brands, places, and objects that "
        "appear in the transcript below. NEVER invent, guess, or borrow a name from these "
        "instructions — if the transcript doesn't name it, don't name it (say 'this', "
        "'that', 'him', 'them' instead). A specific noun that isn't in the clip is an "
        "automatic reject. When in doubt, be PLAIN and ACCURATE, not catchy and wrong.\n"
        "BANNED: vague filler ('what just happened', bare 'wait for it'), descriptions, "
        "past-tense summaries, or GIVING AWAY the payoff. "
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
    partial = C.load_json(C.CAPTIONS_PARTIAL) or {}
    clips = list(partial.get("clips", []))
    done_ids = {c.get("moment_id") for c in clips}
    total = len(selected["selected"])
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
        event = (m.get("text") or "").strip() or _nearby_transcript(
            tr_by_source.get(m["source"], []), float(m["start"]), float(m["end"]))
        cands = _offline_candidates(m) if client is None else _groq_candidates(
            client, campaign, m, "stakes+outcome+emotion, specific to the clip", event,
            emoji_in_caption)
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
        # GROUNDING (Task 2): a hook may only be specific about words that are in THIS
        # moment's transcript. Drop candidates that name something invented/mis-transcribed
        # (e.g. "hibush") or bled from the prompt examples ("hamster" on a jewelry clip).
        # Only enforced when we actually have transcript to ground on.
        tvocab = _transcript_vocab(event)
        enforce_ground = len(tvocab) >= 3

        def _grounded_only(cands):
            if not enforce_ground:
                return list(cands)
            return [c for c in cands if not _ungrounded_terms(c, tvocab)]

        kept_g, clean_g = _grounded_only(kept), _grounded_only(banned_clean)
        # Prefer grounded hook-passing captions; then any grounded banned-clean line (plain
        # but accurate). If grounding is enforced and nothing grounded survives, use a
        # NEUTRAL fallback that asserts nothing specific — never a catchy invented hook.
        if kept_g:
            pool = kept_g
        elif clean_g:
            C.warn(f"moment {m['id']}: no grounded hook-passing caption — using best grounded "
                   f"Groq line (plain + accurate over catchy nonsense).")
            pool = clean_g
        elif enforce_ground:
            C.warn(f"moment {m['id']}: every candidate named something not in the transcript "
                   f"(invented/mis-transcribed) — falling back to a neutral grounded hook.")
            pool = GROUNDED_FALLBACKS
        elif kept:
            pool = kept
        elif banned_clean:
            C.warn(f"moment {m['id']}: no candidate hit a hook pattern — keeping best raw "
                   f"Groq caption (specific to the clip) over a generic template.")
            pool = banned_clean
        else:
            C.warn(f"moment {m['id']}: no usable Groq caption — generic curiosity fallback.")
            pool = GROUNDED_FALLBACKS
        ranked = sorted(pool, key=score_caption, reverse=True)
        # ALL captions on ALL clips: lowercase energy; emoji kept as punctuation unless
        # the campaign config turns them off.
        best = finalize_caption(ranked[0], emoji_in_caption)
        variant = finalize_caption(ranked[1], emoji_in_caption) if len(ranked) > 1 else None
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
        clips.append({
            "moment_id": m["id"], "source": m["source"],
            "start": m["start"], "end": m["end"], "type": m["type"],
            "peak": m.get("peak"),                # cold-open anchor for the cut stage
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
