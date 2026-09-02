"""CUT + DELIVER stage.

Per selected clip: extract the moment, tighten dead air, render to vertical
1080x1920, burn the flzsh caption at top (safe-zone aware), overlay the mandatory
watermark, and write drafts/<Campaign>_NN_score_slug.mp4 (per-campaign, best first)
+ drafts/manifest.json.

The top HOOK caption is rendered to a transparent PNG with Pillow (bold white, black
outline, top-center, <=2 lines, auto font-size) and overlaid by ffmpeg — this dodges
ffmpeg drawtext font/escaping issues and is identical across OSes. A SECOND, distinct
layer — word-level karaoke SUBTITLES lower-center — is rendered via ASS/libass (see
build_ass): the full spoken line shows with the currently-spoken word popped in an accent
colour + upscale, timings mapped through the cold-open reorder onto the final timeline.
Watermark is the campaign PNG from campaign/assets/. Fail loud if anything essential is
missing.
"""
import os
import re
import statistics
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

W, H = 1080, 1920
CAPTION_BOX_W = 1000
CAPTION_TOP_Y = 175           # below the top 8% (~154px) TikTok UI safe zone
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# --- burned subtitle safe zone (from campaign/assets/Safezones *.png) ----------
# Subtitles are pushed DOWN into the lower LETTERBOX / black band, OFF the footage frame
# (user preference). In the default blur_fill layout a 16:9 source fit to the 1080 width
# (with the default 1.2 fg-zoom) occupies roughly y 595-1325, so the band below ~1325 is
# blurred/black dead space. We anchor the karaoke subtitle line near ~1440: below the footage
# frame, clear of the top hook caption, above the bottom ~18-20% TikTok caption/UI zone
# (~1536+), and — at 640px wide, centered (x 220-860) — inside the right action-rail notch
# (~x<870) and above the bottom-right watermark. So subtitles never sit on the footage, the
# watermark, or the platform UI. These feed the ASS style (see _ass_header): SUBTITLE_BOX_W
# → MarginL/MarginR (the safe-box width), SUBTITLE_CENTER_Y → MarginV (distance up from the
# frame bottom for the bottom-center alignment). Both are config-overridable.
SUBTITLE_BOX_W = 640
SUBTITLE_CENTER_Y = 1440      # ~75% down: in the lower black band, off the video frame

# --- per-clip STYLE-SET (caption + hook variety) --------------------------------
# The pipeline is AUDIO-ONLY and cannot see frames, so style variety is ROTATION-based and
# DETERMINISTIC — seeded by the clip id so it's stable on re-cut (not random each render).
# This is a NAMED style-set block so LATER each account/category can ship its own; for now
# there is one curated default. Override wholesale via config `style_set` (a dict merged over
# this). Everything here is chosen to stay high-contrast, readable, and meme-fluent — variety
# within good bounds, not chaos.
DEFAULT_STYLE_SET = {
    "name": "flzsh_default",
    # Curated accent palette (RGB hex; converted to ASS &HBBGGRR at use). Punchy, high-contrast,
    # video-readable against the black outline/plate — no low-contrast or muddy colors.
    "accent_palette": ["FFFF00", "00BFFF", "39FF14", "FF2D95", "FF8C00"],
    #                   yellow    sky-blue  lime      hot-pink  orange
    # (was bright cyan 00E5FF — too washy on light footage; deep-sky-blue 00BFFF is punchier
    #  and holds contrast against a light background even with the outline behind it.)
    # Active-word emphasis: "color" = accent FILL (current look); "box" = accent HIGHLIGHT behind
    # the word (a THIN accent border/halo reads as a clean highlight around it — not a heavy box).
    "emphasis_modes": ["color", "box"],
    "box_border": 5,              # accent border thickness (px @ output res) for "box" mode
}
# NOTE: this style-set is KARAOKE-ONLY. The HOOK (top plate) is deliberately kept CONSTANT — plain
# white text, fixed TOP position — and its plate/outline look is chosen by the hook_style preset
# (HOOK_STYLES, config `hook_style`). Per-clip hook colour/position variance was intentionally
# reverted so a hook style can be judged and locked on its own.


def _style_hash(seed):
    """A STABLE (process-independent) integer hash of a clip id — Python's hash() is salted per
    run, so we use md5 to keep the per-clip style identical on every re-cut."""
    import hashlib
    return int(hashlib.md5(str(seed).encode("utf-8")).hexdigest()[:8], 16)


def _rgb_to_ass(rgb):
    """'FFFF00' (RRGGBB) -> ASS '&H00FFFF&' (&HBBGGRR, reversed byte order)."""
    s = str(rgb).lstrip("#")
    if len(s) != 6:
        return "&H00FFFF&"
    r, g, b = s[0:2], s[2:4], s[4:6]
    return f"&H{b}{g}{r}&".upper().replace("&HX", "&H")


def _rgb_to_pil(rgb):
    """'FFFF00' -> (255,255,0) for Pillow text fill."""
    s = str(rgb).lstrip("#")
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except (ValueError, IndexError):
        return (255, 255, 255)


def resolve_clip_style(cfg, clip_id):
    """Pick THIS clip's KARAOKE style deterministically from the (config-mergeable) style-set,
    seeded by the clip id — accent color + active-word emphasis mode. Different salts per attribute
    so a batch spreads across the palette / modes instead of moving in lockstep. (The HOOK is NOT
    styled here — it's constant white/top with a hook_style plate preset; see resolve_hook_style.)"""
    ss = {**DEFAULT_STYLE_SET, **(cfg.get("style_set") or {})}
    palette = list(ss.get("accent_palette") or ["FFFF00"])
    modes = list(ss.get("emphasis_modes") or ["color"])
    h = _style_hash(clip_id)
    accent_rgb = palette[h % len(palette)]
    return {
        "accent_rgb": accent_rgb,
        "accent_ass": _rgb_to_ass(accent_rgb),
        "emphasis": modes[(h // 7) % len(modes)],
        "box_border": int(ss.get("box_border", 12)),
    }


# --- cold-open restructure (the biggest hook lever) ----------------------------
COLD_OPEN_DUR = 2.0           # target length of the peak teaser (spec: 1.5–2.5s)
COLD_OPEN_MIN = 1.5           # never shorter than this or it reads as a glitch
COLD_OPEN_MIN_SETUP = 3.0     # peak must sit >= this many s past the natural start,
COLD_OPEN_MIN_PAYOFF = 2.0    # and leave >= this much clip after it, else play in order
MOTION_RADIUS = 1.0           # scan ±1s around the peak for the highest-motion frame
FADE_FRAMES = 2               # 2-frame fade on the cold-open->setup cut (reads intentional)
ASSUMED_FPS = 30.0            # fade duration basis when the true fps is unknown
OUTPUT_FPS = 30               # cut output is normalized to this CFR (FIX 3: kills VFR seam stutter)
# Cold-open is CONDITIONAL (FIX 1): only tease a moment that has ONE genuine sharp peak. We
# measure the clip window's audio SHAPE from index.py's per-second RMS — the "triangle range"
# = (window peak − window baseline/median) expressed in the source's own std units (so it's
# comparable across quiet vs loud VODs). A large range = a standout spike worth teasing; a small
# range = flat-high sustained energy, which we play straight. Tunable via config.
COLDOPEN_PEAK_RANGE_MIN = 3.0   # min z-range (σ above baseline) to treat a peak as "sharp"
COLDOPEN_MIN_SEPARATION = 8.0   # min OUTPUT seconds between the teaser and the payoff's natural
                                # arrival in the body (FIX 2) — else it reads as an instant repeat
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"   # per-clip audio normalization (social target)

# Per-second RMS arrays (index.py's <name>.rms.json partials) + their global std, cached per
# source. This is the audio-energy signal the cold-open shape decision reads.
_RMS_CACHE = {}
_RMS_STD_CACHE = {}


def _rms_stats(source):
    """(per-second RMS list, global population std) for a source, or (None, None) if the
    .rms.json partial isn't present (offline index / pre-feature moments.json)."""
    if source not in _RMS_CACHE:
        base = os.path.splitext(os.path.basename(source))[0]
        vals = C.load_json(C.TRANSCRIPTS / f"{base}.rms.json", default=None)
        arr = [float(v) for v in vals] if vals else None
        _RMS_CACHE[source] = arr
        _RMS_STD_CACHE[source] = (statistics.pstdev(arr) if arr and len(arr) > 1 else 0.0)
    return _RMS_CACHE[source], _RMS_STD_CACHE[source]


def coldopen_shape(rms, gstd, start, end):
    """Analyze the clip window [start,end] against the source RMS. Returns
    (true_peak_sec, z_range) or None when there's no usable RMS:
      - true_peak_sec: ABSOLUTE second of the loudest RMS in the window — the clip's REAL
        energy peak (FIX 4), used to anchor the teaser instead of an arbitrary/early frame.
      - z_range: (window_peak − window_median) / global_std — the triangle-vs-flat SHAPE
        metric (FIX 1). High = one sharp standout spike; low = flat-high sustained energy."""
    if not rms or gstd <= 0:
        return None
    lo = max(0, int(start))
    hi = min(len(rms), int(round(end)))
    if hi - lo < 3:
        return None
    win = rms[lo:hi]
    peak_val = max(win)
    true_peak = lo + win.index(peak_val)
    baseline = statistics.median(win)
    z_range = (peak_val - baseline) / gstd
    return float(true_peak), round(z_range, 2)


def coldopen_body_gap(true_peak, start, segments):
    """OUTPUT-timeline seconds between the END of the cold-open teaser (segments[0]) and the
    moment the teased peak arrives NATURALLY in the body (segments[1:]). Returns None if the
    peak was trimmed out of the body. Rejects cold-opens whose payoff replays too soon (FIX 2)
    — the separation is measured on the FINAL edit (post dead-air trim + cap), not source time,
    so trimmed-away setup can't collapse the gap unnoticed."""
    if len(segments) < 2:
        return None
    teaser_dur = segments[0][1] - segments[0][0]
    rel = true_peak - start
    out = teaser_dur
    for (a, b) in segments[1:]:
        if a <= rel < b:
            return (out + (rel - a)) - teaser_dur
        out += (b - a)
    return None


def plan_cold_open(src_path, source, start, end, keeps, cmax, cfg, hook_moment=None):
    """Decide whether THIS clip gets a cold-open, and where the teaser is (FIX 1/2/4).
    Returns (cold_rel | None, reason_string). cold_rel is the clip-relative (a,b) teaser span.

    WHETHER to cold-open still comes from the audio SHAPE: (1) needs a genuine SINGLE SHARP peak
    — window z-range >= threshold, else it's flat-high and plays straight. WHICH instant to tease
    is chosen by CONTENT: `hook_moment` (the LLM-picked most suspenseful/craziest LINE, from the
    caption stage) is preferred as the teaser anchor, falling back to the audio peak when it's
    absent or sits at a clip edge — the loudest second is often NOT the best tease. Then (2) the
    anchor must sit inside the clip, not at an edge; (3) a clean >=1.5s teaser window must carve
    out; (4) on the FINAL edit the payoff must arrive with real separation after the teaser."""
    range_min = float(cfg.get("coldopen_peak_range_min", COLDOPEN_PEAK_RANGE_MIN))
    cold_min_sep = float(cfg.get("coldopen_min_separation", COLDOPEN_MIN_SEPARATION))
    rms, gstd = _rms_stats(source)
    shape = coldopen_shape(rms, gstd, start, end)
    if shape is None:
        return None, "no RMS energy data → straight cut"
    true_peak, z_range = shape
    if z_range < range_min:
        return None, f"flat-high (range {z_range:.1f}σ < {range_min:g}σ) → straight cut"
    # WHICH instant to tease: prefer the LLM's hook line when it lands inside the clip window and
    # leaves setup+payoff room; else the audio peak. (The whether-gate above is unchanged.)
    def _edge_ok(t):
        return (t - start) >= COLD_OPEN_MIN_SETUP and (end - t) >= COLD_OPEN_MIN_PAYOFF
    anchor, anchor_src = true_peak, "audio peak"
    if hook_moment is not None:
        try:
            hm = float(hook_moment)
        except (TypeError, ValueError):
            hm = None
        if hm is not None and start <= hm <= end and _edge_ok(hm):
            anchor, anchor_src = hm, "LLM hook line"
    if not _edge_ok(anchor):
        return None, f"sharp peak (range {z_range:.1f}σ) but sits at the clip edge → straight cut"
    cand = cold_open_window(src_path, anchor, start, end)
    if cand is None:
        return None, f"sharp peak (range {z_range:.1f}σ) but no clean teaser window → straight cut"
    trial = cap_segments([cand] + keeps, cmax)          # measure separation on the FINAL edit
    gap = coldopen_body_gap(anchor, start, trial)
    if gap is None:
        return None, f"sharp peak (range {z_range:.1f}σ) but teased moment trimmed from body → straight cut"
    if gap < cold_min_sep:
        return None, (f"sharp peak (range {z_range:.1f}σ) but payoff replays too soon "
                      f"(+{gap:.1f}s < {cold_min_sep:g}s) → straight cut")
    return cand, (f"sharp peak (range {z_range:.1f}σ) → teasing {anchor_src} "
                  f"(payoff +{gap:.1f}s later)")


# --- fonts / caption rendering -------------------------------------------------
def find_bold_font():
    env = os.environ.get("CLIPPER_FONT")
    cands = [env] if env else []
    cands += [
        r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    C.fail("no bold TTF font found for captions. Set CLIPPER_FONT=/path/to/bold.ttf "
           "(Windows has arialbd.ttf; Linux: install fonts-dejavu).")


# Emoji code-point ranges (incl. variation selectors / ZWJ / skin tones).
EMOJI_RE = re.compile(
    "([\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U00002190-\U000021FF️‍\U0001F3FB-\U0001F3FF]+)")


def find_emoji_font():
    env = os.environ.get("CLIPPER_EMOJI_FONT")
    cands = [env] if env else []
    cands += [
        r"C:\Windows\Fonts\seguiemj.ttf",                       # Segoe UI Emoji (COLR, scalable)
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
    ]
    for c in cands:
        if not c or not os.path.exists(c):
            continue
        try:                       # bitmap emoji fonts often can't load at our sizes
            ImageFont.truetype(c, 64)
            return c
        except Exception:
            continue
    return None


def strip_emoji(text):
    return re.sub(r"\s{2,}", " ", EMOJI_RE.sub("", text)).strip()


# --- emoji tofu handling -------------------------------------------------------
# We can't read the emoji font's cmap without fontTools, and Segoe UI Emoji draws
# UNSUPPORTED code points as a visible .notdef box that a pixel/bbox probe can't tell
# apart from a real glyph. So to guarantee "drop a tofu, never render one" we render
# only code points we KNOW the color-emoji fonts (Segoe UI Emoji / Noto) carry — the
# flzsh punctuation set — and drop anything else. If fontTools ever gets installed we
# honor the live cmap instead (more permissive). Combiners (VS16 / ZWJ / skin tones)
# always pass through with their base glyph.
_ALLOWED_EMOJI_CP = {
    0x1F62D,  # 😭 loudly crying     0x1F480,  # 💀 skull
    0x1F525,  # 🔥 fire              0x1F440,  # 👀 eyes
    0x270C,   # ✌ victory hand      0x1F64F,  # 🙏 folded hands
    0x1F940,  # 🥀 wilted flower     0x1F633,  # 😳 flushed
    0x1F923,  # 🤣 rofl              0x1F602,  # 😂 joy
    0x1F605,  # 😅 sweat smile       0x1F624,  # 😤 huffing
    0x1F631,  # 😱 screaming         0x1F4AF,  # 💯 hundred
    0x26A1,   # ⚡ high voltage       0x1F439,  # 🐹 hamster
    0x1F3C1,  # 🏁 chequered flag    0x1F3CE,  # 🏎 racing car
    0x1F971,  # 🥱 yawn              0x1F97A,  # 🥺 pleading
    0x1F44F,  # 👏 clap              0x1F621,  # 😡 pouting
}
_EMOJI_COMBINER_CP = {0xFE0F, 0x200D}                 # VS16, zero-width joiner


def _emoji_cmap(font_path):
    """Live cmap of the emoji font via fontTools if it's installed, else None (we then
    fall back to the curated allowlist). Cached per font path."""
    cache = _emoji_cmap.__dict__.setdefault("_c", {})
    if font_path not in cache:
        try:
            from fontTools.ttLib import TTFont
            tt = TTFont(font_path, fontNumber=0, lazy=True)
            cache[font_path] = tt.getBestCmap()
            tt.close()
        except Exception:
            cache[font_path] = None
    return cache[font_path]


def _emoji_renderable(cp, font_path):
    cmap = _emoji_cmap(font_path)
    return (cp in cmap) if cmap is not None else (cp in _ALLOWED_EMOJI_CP)


def _drop_unrenderable_emoji(text, font_path):
    """Remove emoji code points the font can't draw (would be a tofu box); keep base
    text, combiners, and skin-tone modifiers. Never raises."""
    out = []
    for ch in text:
        cp = ord(ch)
        is_emoji = bool(EMOJI_RE.match(ch)) and cp not in _EMOJI_COMBINER_CP \
            and not (0x1F3FB <= cp <= 0x1F3FF)
        if is_emoji and not _emoji_renderable(cp, font_path):
            continue                                   # tofu → drop this emoji only
        out.append(ch)
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def _keep_ascii_and_emoji(text):
    """Collapse whitespace, keep ASCII + emoji/combiners, drop stray other non-ASCII."""
    out = [ch for ch in (text or "") if ord(ch) < 128 or EMOJI_RE.match(ch)]
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def strip_to_ascii(text):
    """Drop EVERY non-ASCII character (all emoji + symbol glyphs, not just the ranges
    EMOJI_RE knows) and collapse the whitespace they leave behind. This is the hard,
    guaranteed gate applied at the final render step so no emoji/symbol can ever be
    burned into the video — regardless of upstream captions or installed emoji fonts."""
    ascii_only = (text or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s{2,}", " ", ascii_only).strip()


def _segment(text):
    """[(substr, is_emoji), ...] preserving order."""
    parts = EMOJI_RE.split(text)
    return [(p, i % 2 == 1) for i, p in enumerate(parts) if p]


def _cells(text, tf, ef, scratch):
    """Layout cells: words/spaces (text font) and emoji runs (emoji font)."""
    cells = []
    for seg, is_emoji in _segment(text):
        if is_emoji:
            f = ef or tf
            w = scratch.textlength(seg, font=f, embedded_color=bool(ef))
            cells.append({"s": seg, "font": f, "w": w, "emoji": True, "space": False})
        else:
            for tok in re.split(r"(\s+)", seg):
                if not tok:
                    continue
                space = tok.isspace()
                s = " " if space else tok
                cells.append({"s": s, "font": tf, "w": scratch.textlength(s, font=tf),
                              "emoji": False, "space": space})
    return cells


def _wrap_cells(cells, inner, max_lines):
    lines, cur, curw = [], [], 0.0
    for cell in cells:
        if cell["space"] and not cur:
            continue                          # no leading spaces
        if curw + cell["w"] > inner and cur:
            while cur and cur[-1]["space"]:
                curw -= cur[-1]["w"]; cur.pop()
            lines.append(cur)
            cur, curw = [], 0.0
            if cell["space"]:
                continue
        cur.append(cell); curw += cell["w"]
    if cur:
        while cur and cur[-1]["space"]:
            cur.pop()
        lines.append(cur)
    return lines[:max_lines] if lines else [[]]


# The HOOK plate hugs the text — kept tight so the caption doesn't read as a bulky block.
# It is the scroll-stopper (bigger text, denser plate), independently tunable via config
# (hook_font_scale / hook_plate_opacity). The spoken-word SUBTITLES are a separate layer,
# rendered as word-level karaoke via ASS/libass (see build_ass), not a Pillow plate.
CAPTION_PLATE_PAD_X = 20
CAPTION_PLATE_PAD_Y = 9
CAPTION_PLATE_RADIUS = 18

# --- HOOK STYLE PRESETS (top caption plate/outline ONLY) ------------------------
# Selectable via config `hook_style` (A|B|C|D). HOOK-ONLY: this does NOT touch the karaoke
# subtitle styling (its per-clip colour/box/emphasis variety is separate — see resolve_clip_style).
# Each preset is a set of render_caption_png kwargs (plate opacity + outline stroke + plate padding).
HOOK_STYLES = {
    # A: NO plate, bold white text, THICK black outline.
    "A": {"plate_opacity": 0,   "stroke": 6, "pad_x": 20, "pad_y": 9, "radius": 18},
    # B: NO plate, bold white text, THIN black outline.
    "B": {"plate_opacity": 0,   "stroke": 2, "pad_x": 20, "pad_y": 9, "radius": 18},
    # C: THIN semi-transparent plate, small padding, thin outline.
    "C": {"plate_opacity": 90,  "stroke": 2, "pad_x": 12, "pad_y": 5, "radius": 12},
    # D: thick plate, thin outline.
    "D": {"plate_opacity": 105, "stroke": 2, "pad_x": 20, "pad_y": 9, "radius": 18},
}
DEFAULT_HOOK_STYLE = "A"          # LOCKED: no plate, bold white text, thick black outline

# The hook auto-shrinks to fit width, but SHORT hooks used to balloon (they fit at the big start
# size). Cap the start so a short hook lands at/near the reference (clipA rendered ~61px) instead
# of ~89px — a little natural size variation is fine, but nothing should exceed this noticeably.
HOOK_MAX_FONT = 64


def resolve_hook_style(cfg):
    """Return the render_caption_png kwargs for the configured hook_style (A|B|C|D), defaulting to
    the current thick-plate look (D). The HOOK is always plain WHITE text at the fixed TOP position;
    only the plate/outline changes between presets."""
    key = str(cfg.get("hook_style", DEFAULT_HOOK_STYLE)).strip().upper()
    return dict(HOOK_STYLES.get(key, HOOK_STYLES[DEFAULT_HOOK_STYLE]))


def render_caption_png(text, out_path, box_w=CAPTION_BOX_W, max_lines=2, stroke=2,
                       emoji=True, plate_opacity=105, font_scale=1.0, text_color="white",
                       pad_x=CAPTION_PLATE_PAD_X, pad_y=CAPTION_PLATE_PAD_Y,
                       radius=CAPTION_PLATE_RADIUS, max_font=HOOK_MAX_FONT):
    """The HOOK caption — the scroll-stopper, so it is the PROMINENT of the two text layers
    (larger/bolder than the burned subtitles). Clean, understated (Title Case, emoji as
    punctuation): white text on a SEMI-TRANSPARENT DARK PLATE so it stays legible on ANY
    background — bright, dark, or busy — with only a THIN (or no) outline; the plate does the
    contrast work, not a heavy stroke. `plate_opacity` (0-255, 0 = no plate), `font_scale`
    (>1 = larger/bolder-looking), `stroke` (0 = no outline), and the plate `pad_x`/`pad_y`/`radius`
    are the tunable knobs — driven by the hook_style preset (see HOOK_STYLES).

    When `emoji` is on we render emoji with a color-emoji font (Segoe UI Emoji / Noto) and DROP
    any glyph the font can't draw (never a tofu box); each emoji is vertically centered on the
    text's optical midline so it sits inline, not offset. When off — or when no emoji font is
    installed — every non-ASCII char is hard-stripped, the old guaranteed no-emoji gate."""
    text_font_path = find_bold_font()
    emoji_font_path = find_emoji_font() if emoji else None
    if emoji_font_path:
        text = _drop_unrenderable_emoji(_keep_ascii_and_emoji(text), emoji_font_path)
    else:
        text = strip_to_ascii(text or "clip")
    text = (text or "clip").strip() or "clip"

    scratch = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    # Leave room inside the box for the plate padding so a full-width line still fits the plate.
    inner = box_w - 2 * pad_x - 16
    # Start at the (capped) max and shrink to fit width. The cap is what stops a SHORT hook from
    # ballooning: it can only start as big as max_font, so it lands at/near the reference size, not
    # ~89px. Long hooks still shrink further to fit ≤max_lines.
    smin = max(int(round(30 * font_scale)), 20)
    smax = max(min(int(round(84 * font_scale)), int(max_font)), smin)
    size, lines, tf, ef = smax, [[]], None, None
    while size >= smin:
        tf = ImageFont.truetype(text_font_path, size)
        ef = ImageFont.truetype(emoji_font_path, size) if emoji_font_path else None
        lines = _wrap_cells(_cells(text, tf, ef, scratch), inner, max_lines)
        # accept if it fit within max_lines without truncation
        if len(_wrap_cells(_cells(text, tf, ef, scratch), inner, max_lines + 1)) <= max_lines:
            break
        size -= 4

    ascent, descent = tf.getmetrics()
    line_h = ascent + descent + 8
    img_h = line_h * len(lines) + 2 * pad_y
    img = Image.new("RGBA", (box_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Semi-transparent dark plate hugging the widest line — guarantees white-on-anything
    # contrast without a loud outline. plate_opacity<=0 disables it (no plate).
    maxw = max((sum(c["w"] for c in ln) for ln in lines), default=0.0)
    if plate_opacity > 0 and maxw > 0:
        pw = min(float(box_w), maxw + 2 * pad_x)
        px0 = (box_w - pw) / 2
        d.rounded_rectangle([px0, 0, px0 + pw, img_h], radius=radius,
                            fill=(0, 0, 0, int(plate_opacity)))

    # Text optical midline (from a sample ascender+descender glyph) — emoji are centered on it.
    tb = tf.getbbox("Ayg")
    y = pad_y
    for line in lines:
        total = sum(c["w"] for c in line)
        x = (box_w - total) / 2
        line_center = y + (tb[1] + tb[3]) / 2.0
        for c in line:
            if c["emoji"] and ef is not None:
                try:
                    eb = d.textbbox((0, 0), c["s"], font=c["font"], embedded_color=True)
                    ey = line_center - (eb[1] + eb[3]) / 2.0   # center emoji box on text midline
                except Exception:
                    ey = y
                try:
                    d.text((x, ey), c["s"], font=c["font"], embedded_color=True)
                except Exception:
                    pass
            elif stroke > 0:
                d.text((x, y), c["s"], font=c["font"], fill=text_color,
                       stroke_width=stroke, stroke_fill="black")
            else:
                d.text((x, y), c["s"], font=c["font"], fill=text_color)
            x += c["w"]
        y += line_h
    img.save(out_path)
    return out_path


# --- burned karaoke-style subtitles (word-level, rendered via ASS/libass) -------
# TIMING APPROACH: we DO NOT re-transcribe the clip. index.py persists per-word
# {word,start,end} (whisper word_timestamps) into moments.json, and here we MAP those
# SOURCE-absolute word timings through the EXACT same segment list the compose graph
# plays — cold-open teaser (peak shown first), dead-air trims, and the cmax tail cut.
# So a word can legitimately appear twice (once in the cold-open, once in the setup) and
# each occurrence lands on the FINAL output timeline. Get the mapping wrong and captions
# drift silently, so map_words_to_output() is driven by the very `segments` compose uses.
#
# RENDERING: one ASS Dialogue event per word, each covering that word's slice of the line;
# the event shows the FULL line with only the currently-spoken word wrapped in an accent
# colour + upscale override ({\c&Hbbggrr&\fscx..\fscy..}) so it POPS, then reverts as the
# next word speaks. libass (ffmpeg `ass` filter) burns it AFTER the 1080x1920 blur-fill so
# subs render at output resolution. Gated off in offline mode (no word timings) and by
# `subtitles_enabled=false`.


def _mask_banned_word(word, banned):
    """Mask any campaign-banned word before it can be burned into a subtitle
    ('bet' -> 'b**', 'gambling' -> 'g*******'). Surrounding punctuation is preserved;
    clean words pass through untouched. A banned word in a subtitle = campaign violation,
    so this is the FIRST line of defense (cut.py re-checks the whole line after)."""
    from captions import banned_hit
    m = re.match(r"^(\W*)(.*?)(\W*)$", word, re.S)
    pre, core, post = m.group(1), m.group(2), m.group(3)
    if core and banned_hit(core, banned):
        core = (core[0] + "*" * (len(core) - 1)) if len(core) > 1 else "*"
    return pre + core + post


def _subtitle_word(word):
    """flzsh / Gen Z karaoke styling for ONE spoken-word subtitle token — the LOWER band only,
    NOT the top hook plate. All lowercase, apostrophes dropped so contractions read as one word
    ('what's' -> 'whats', 'that's' -> 'thats'), and EVERY other punctuation/symbol stripped
    ('Saki.' -> 'saki', 'difference?' -> 'difference'). Keeps letters + digits ('10v1' survives).
    Returns '' for a punctuation-only token (the caller then skips it). Banned-word masking runs
    AFTER this (in _group_lines) so the mask's '*' are never stripped away.

    FIX 1: the English standalone pronoun 'I' must stay CAPITAL even in the lowercase style —
    'i' -> 'I', and the I-contractions I'm/I'll/I've/I'd (joined to im/ill/ive/id) -> Im/Ill/Ive/Id.
    Only the standalone pronoun: the contraction forms are gated on the raw word actually having
    an apostrophe, so a plain word like 'ill' (sick) or 'id' is NOT wrongly capitalized, and 'i'
    inside another word is untouched (we only rewrite the whole-token cases)."""
    raw = word or ""
    t = raw.lower().replace("'", "").replace("’", "")             # join contractions
    t = re.sub(r"[^a-z0-9]+", "", t)                              # drop all other punctuation
    if not t:
        return ""
    had_apostrophe = "'" in raw or "’" in raw
    if t == "i" or (had_apostrophe and t in ("im", "ill", "ive", "id")):
        t = "I" + t[1:]                                          # capital I only for the pronoun
    return t


# --- profanity censor (karaoke lower band only, SEPARATE from campaign banned masking) ------
# Campaign banned words (bet/gamble/…) are FULLY masked ("b**") by _mask_banned_word for rules
# compliance. Profanity gets a LIGHTER, stylistic censor instead: keep consonants + first/last,
# replace the VOWELS with '*' ("shit"->"sh*t", "fuck"->"f*ck", "ass"->"*ss", "fucking"->
# "f*ck*ng"). Applied ONLY to karaoke words (not the hook plate, not the campaign gauntlet).
_VOWELS = frozenset("aeiou")
_PROFANITY_ROOTS = frozenset("""
fuck shit bitch ass asshole damn hell dick cunt pussy bastard prick slut whore twat cum cock
dickhead motherfucker bullshit douchebag wanker jackass dumbass piss crap bollocks
nigger nigga faggot fag retard spic chink kike coon gook tranny dyke
""".split())
# Inflections we allow after a root (fucking, bitches, shitty→shit+t+y). Kept short + anchored
# so it catches swears but NOT innocent words that merely start the same (assault, assess,
# hello, cocktail — their remainder isn't an inflection, so they're left alone).
_PROF_SUFFIX = frozenset(["s", "es", "ed", "ing", "ings", "er", "ers", "in", "y", "ies", "a"])


def _is_profane(word):
    """True if `word` (lowercase, letters only) is a swear/slur — exact, or a root plus a short
    inflection (fucking, bitches, shitty via an optional doubled final consonant). Anchored so
    'assault'/'assess'/'hello'/'cocktail' are NOT matched."""
    for root in _PROFANITY_ROOTS:
        if word == root:
            return True
        if not word.startswith(root):
            continue
        rest = word[len(root):]
        if rest and root[-1] not in _VOWELS and rest[:1] == root[-1]:   # gemination: shit->shitt-
            rest = rest[1:]
        if rest == "" or rest in _PROF_SUFFIX:
            return True
    return False


def _censor_profanity(word):
    """Vowel-censor a profane word ('shit'->'sh*t'); return it unchanged if not profane."""
    if not word or not _is_profane(word):
        return word
    return "".join("*" if ch in _VOWELS else ch for ch in word)


def _fx_line_text(label, banned):
    """A non-speech sound label styled for the karaoke: lowercase, wrapped in asterisks
    ('Scream!' -> '*scream*', 'weird noises' -> '*weird noises*'). Spaces are kept (a two-word
    label stays one unit); banned words in the label are still masked. '' if empty."""
    inner = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (label or "").lower())).strip()
    if not inner:
        return ""
    inner = " ".join(_mask_banned_word(w, banned) for w in inner.split())
    return f"*{inner}*"


def map_words_to_output(words, start, segments):
    """Map SOURCE-absolute word timings onto the FINAL output (clip) timeline.

    `words`    : [{word,start,end}] in source-absolute seconds (already limited to the clip).
    `start`    : the clip's source start — `segments` are relative to it (see compose: the
                 input is seeked with -ss start, then trim=a:b runs on that seeked stream).
    `segments` : the ordered clip-relative (a,b) spans compose PLAYS, in order — cold-open
                 teaser first (when present), then the dead-air-trimmed body, already passed
                 through cap_segments. This is the SAME list build_compose_cmd consumes.

    Returns [{word,start,end}] in output seconds, sorted by start — each word rendered ONCE
    per time it is actually HEARD in the output. Two dedup passes make that true:
      (1) exact-duplicate SOURCE words are dropped first (whisper/index can emit the same
          {word,start,end} twice — e.g. a re-indexed VOD — which otherwise renders "That
          That"); and
      (2) a single source word that straddles a played-segment BOUNDARY produces contiguous
          output fragments, which we MERGE back into one event (tracked by source-word index).
    A source word that plays in TWO non-adjacent segments — the cold-open teaser AND again in
    the setup — is genuinely heard twice, so those two events are kept (they're far apart, not
    contiguous). Fades/scale/loudnorm don't shift time, so the output timeline is exactly the
    concatenation of the played segment durations."""
    # (1) drop exact-duplicate source words (same text at the same source instant).
    seen, uw = set(), []
    for w in words:
        key = (w["word"], round(float(w["start"]), 3), round(float(w["end"]), 3))
        if key in seen:
            continue
        seen.add(key)
        uw.append(w)

    raw, out_off = [], 0.0
    for (a, b) in segments:
        seg_dur = b - a
        src_lo, src_hi = start + a, start + b            # source window this segment covers
        for wi, w in enumerate(uw):
            lo = max(float(w["start"]), src_lo)
            hi = min(float(w["end"]), src_hi)
            if hi <= lo:
                continue
            raw.append({"wi": wi, "word": w["word"],
                        "start": out_off + (lo - src_lo),
                        "end": out_off + (hi - src_lo)})
        out_off += seg_dur
    raw.sort(key=lambda e: (e["start"], e["end"]))

    # (2) merge output-contiguous fragments of the SAME source word (boundary split). A gap
    # (cold-open replay) leaves the two events apart, so they stay separate = heard twice.
    events = []
    for e in raw:
        if events and events[-1]["wi"] == e["wi"] and e["start"] <= events[-1]["end"] + 0.06:
            events[-1]["end"] = max(events[-1]["end"], e["end"])
        else:
            events.append(dict(e))
    for e in events:
        e.pop("wi", None)
    return events


def _group_lines(events, banned, max_words=3, pause_gap=0.35, censor_profanity=True):
    """Group per-word events into karaoke LINES that follow the natural RHYTHM of speech.

    A line breaks PRIMARILY on a PAUSE: whenever the silence between one word's end and the
    next word's start exceeds `pause_gap` (default ~0.35s), the current line ends there — even
    if it's only ONE word — so each spoken burst is its own line ("hello" [pause] "how are you"
    -> two lines, not "hello how are you"). Sentence-ending punctuation (. ! ?) also breaks.
    WITHIN a continuous run of speech (no big gaps) we still cap at `max_words` (~3) so a fast
    unbroken sentence doesn't run long; a 4th word is pulled in only to avoid orphaning the next
    word as a lone trailing line. Everything is ONE physical line (WrapStyle 2 in _ass_header).

    The word gaps are read from the OUTPUT-timeline events (post reorder/dead-air-trim), so the
    rhythm matches the FINAL edit, not the raw source. Each word gets the flzsh karaoke styling
    (`_subtitle_word`: lowercase, no punctuation, contractions joined); a CAMPAIGN-banned word is
    then FULLY masked ('b**'), else (when `censor_profanity`) a swear gets the LIGHT vowel-censor
    ('sh*t'). Sentence breaks read the RAW word's trailing '.!?' BEFORE stripping. Lower band only
    — the top hook plate keeps its own casing."""
    styled = []
    for e in events:
        raw = (e["word"] or "").strip()
        clean = _subtitle_word(strip_to_ascii(raw))
        tok = _mask_banned_word(clean, banned)             # campaign rule: FULL mask if banned
        if tok == clean and censor_profanity:              # not banned → light profanity censor
            tok = _censor_profanity(clean)
        if not tok:
            continue
        styled.append({"tok": tok, "start": float(e["start"]), "end": float(e["end"]),
                       "eos": bool(re.search(r"[.!?]$", raw))})

    def gap_after(i):                     # True if a pause (or the clip end) follows word i
        return i + 1 >= len(styled) or styled[i + 1]["start"] - styled[i]["end"] > pause_gap

    lines, cur = [], []
    for i, w in enumerate(styled):
        cur.append(w)
        if w["eos"] or gap_after(i):                      # PRIMARY break: a pause / sentence end
            lines.append(cur); cur = []
            continue
        if len(cur) >= max_words:
            # No pause yet but the line is long — break at the word cap. Allow ONE extra word
            # only if the next word is itself the last of this burst (a pause/sentence follows
            # it), else it'd be stranded alone. Otherwise break now (keeps ~2-3 words).
            if len(cur) == max_words and i + 1 < len(styled) and (styled[i + 1]["eos"] or gap_after(i + 1)):
                continue
            lines.append(cur); cur = []
    if cur:
        lines.append(cur)
    for ln in lines:                      # drop the internal helper key before returning
        for w in ln:
            w.pop("eos", None)
    return lines


def _ass_time(t):
    """Seconds -> ASS timestamp H:MM:SS.cc (centiseconds)."""
    t = max(0.0, float(t))
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text):
    """Neutralize ASS override syntax in literal subtitle text: braces open an override
    block and backslash starts an escape, so remap them to safe look-alikes."""
    return (text or "").replace("\\", "/").replace("{", "(").replace("}", ")")


def blur_fill_footage_bottom(src_w, src_h, zoom):
    """Y (on the 1080x1920 canvas) of the BOTTOM edge of the footage rectangle under the
    blur_fill layout — computed from the SAME geometry build_compose_cmd uses (fit the whole
    frame into W*zoom x H preserving aspect, then center it vertically). Returns None if the
    source dims are unknown, or if the footage fills the full height (a portrait/near-square
    source leaves no letterbox band). Below this Y is the lower blurred/black band."""
    try:
        sw, sh = float(src_w), float(src_h)
    except (TypeError, ValueError):
        return None
    if sw <= 0 or sh <= 0:
        return None
    zoom = max(1.0, float(zoom))
    scale = min((W * zoom) / sw, H / sh)             # force_original_aspect_ratio=decrease
    vis_h = min(sh * scale, H)                        # crop clamps height to the canvas
    if vis_h >= H - 2:
        return None                                  # no band — footage fills the frame
    return (H + vis_h) / 2.0


def subtitle_top_y(cfg, src_w=None, src_h=None, layout=None):
    """Y (from the top of the 1920 canvas) at which the karaoke line's TOP should sit, so it
    lands in the lower letterbox band BELOW the footage and never overlaps it. In blur_fill we
    pin it `subtitle_band_margin` px under the computed footage bottom; without a band (crop_fill,
    TRACK, or a full-height source) we fall back to `subtitle_center_y`. `layout` overrides the
    config layout (the cut stage resolves it PER CLIP). Config `subtitle_top_y` forces a value."""
    forced = cfg.get("subtitle_top_y")
    if forced is not None:
        return int(forced)
    margin = int(cfg.get("subtitle_band_margin", 28))
    layout = str(layout or cfg.get("layout", "blur_fill")).lower()
    if layout not in ("crop_fill", "track"):
        fb = blur_fill_footage_bottom(src_w, src_h, cfg.get("blur_fg_zoom", 1.2))
        if fb is not None:
            # keep it inside the band (leave room below for a 2nd line before the very bottom)
            return int(min(fb + margin, H - 220))
    return int(cfg.get("subtitle_center_y", SUBTITLE_CENTER_Y))


def _ass_header(cfg, top_y):
    box_w = int(cfg.get("subtitle_box_w", SUBTITLE_BOX_W))
    font = str(cfg.get("subtitle_font_name", "Arial"))
    fontsize = int(cfg.get("subtitle_ass_fontsize", 54))
    # Readability: karaoke sits over unpredictable footage (bright/light backgrounds wash out
    # a colored word). A THICK black outline + a defined dark drop shadow keep every word legible
    # on ANY background, whatever the per-clip accent colour is. Both are tunable via config.
    outline = int(cfg.get("subtitle_outline", cfg.get("caption_outline_width", 4)) or 0)
    shadow = int(cfg.get("subtitle_shadow", 2))
    side = max(0, (W - box_w) // 2)                  # keep text inside the safe box (x)
    # Alignment 8 = TOP-center: MarginV is the gap from the FRAME TOP down to the text top, so
    # the line starts at `top_y` (just under the footage) and any 2nd line grows DOWN into the
    # band — it can never ride UP onto the footage. PrimaryColour white, OutlineColour black,
    # near-opaque black shadow (BackColour &H20…, AA=20 ≈ 87% opaque) so the text reads on light
    # footage regardless of accent colour. &HAABBGGRR (AA=00 opaque).
    margin_v = max(0, int(top_y))
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 2\n"                     # no auto-wrap: karaoke lines stay ONE line
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {W}\n"
        f"PlayResY: {H}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Sub,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H20000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},{shadow},8,{side},{side},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def build_ass(events, cfg, banned, out_path, top_y, fx_events=None, style=None):
    """Write a word-level karaoke ASS file for one clip and return its path, or None if
    there's nothing to burn. Emits one Dialogue per word: the full line is shown, and the
    active word is emphasised so it POPS, reverting as the next word speaks. `top_y` pins the
    line's top into the lower letterbox band (see subtitle_top_y). `fx_events` are non-speech
    sound labels ([{label,start,end}], already output-timed) inserted as standalone '*scream*'
    lines in the word gaps. Re-checks each finished line for banned words (fail-loud).

    `style` (per-clip, from resolve_clip_style) sets the accent COLOUR and the emphasis MODE:
    'color' tints the active word (accent FILL); 'box' puts an accent HIGHLIGHT (thick accent
    border) behind it. None → fall back to the config default accent, color mode."""
    lines = _group_lines(events, banned,
                         max_words=int(cfg.get("subtitle_max_words", 3)),
                         pause_gap=float(cfg.get("subtitle_pause_gap", 0.35)),
                         censor_profanity=bool(cfg.get("subtitle_censor_profanity", True)))
    # Non-speech sound labels become standalone single-token lines, merged into the timeline.
    for fx in (fx_events or []):
        tok = _fx_line_text(fx.get("label", ""), banned)
        if not tok:
            continue
        lines.append([{"tok": tok, "start": float(fx["start"]), "end": float(fx["end"])}])
    lines.sort(key=lambda ln: ln[0]["start"])
    if not lines:
        return None
    from captions import banned_hit
    style = style or {}
    accent = str(style.get("accent_ass") or cfg.get("subtitle_accent_color", "&H00FFFF&"))
    emphasis = str(style.get("emphasis", "color"))               # "color" (fill) | "box" (highlight)
    box_border = int(style.get("box_border", 12))
    scale = int(cfg.get("subtitle_active_scale", 110))            # % upscale of the active word
    hold = float(cfg.get("subtitle_hold", 0.25))                  # linger after the last word
    # The active-word override: color mode tints the FILL; box mode keeps the white fill but gives
    # the word a thick accent BORDER so it reads as a highlighted/boxed word. Both upscale it.
    if emphasis == "box":
        active_open = f"{{\\3c{accent}\\bord{box_border}\\fscx{scale}\\fscy{scale}}}"
    else:
        active_open = f"{{\\c{accent}\\fscx{scale}\\fscy{scale}}}"
    dialogues = []
    for li, line in enumerate(lines):
        toks = [w["tok"] for w in line]
        # defense-in-depth: masking ran per-word already; abort if a banned word survived.
        if banned_hit(" ".join(toks), banned):
            C.fail(f"banned word slipped into a subtitle line ({' '.join(toks)!r}) — aborting.")
        next_start = lines[li + 1][0]["start"] if li + 1 < len(lines) else None
        line_end = line[-1]["end"] + hold
        if next_start is not None:
            line_end = min(line_end, next_start - 0.03)
        for k, w in enumerate(line):
            ev_start = w["start"]
            ev_end = line[k + 1]["start"] if k + 1 < len(line) else max(w["end"], line_end)
            if ev_end <= ev_start:
                ev_end = ev_start + 0.05
            parts = []
            for j, tok in enumerate(toks):
                safe = _ass_escape(tok)
                if j == k:
                    parts.append(f"{active_open}{safe}{{\\r}}")
                else:
                    parts.append(safe)
            text = " ".join(parts)
            dialogues.append(
                f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Sub,,0,0,0,,{text}")
    out_path.write_text(_ass_header(cfg, top_y) + "\n".join(dialogues) + "\n", encoding="utf-8")
    return out_path


def _ass_filter_arg(path):
    """Escape an absolute path for use as the ffmpeg `ass` filter value: forward slashes,
    escaped drive colon, single-quoted. Verified on Windows (C\\:/…/x.ass)."""
    p = str(path).replace("\\", "/").replace(":", "\\:")
    return f"ass='{p}'"


# --- assets --------------------------------------------------------------------
# Reference images that are NOT overlays: placement examples + safe-zone guides.
REFERENCE_HINTS = ("example", "placement", "safezone", "safe zone", "safe-zone",
                   "safezones", "reference", "guide")


def find_watermark(cfg=None):
    """Pick the watermark PNG. Honors cfg['watermark_file'] (exact or substring match);
    otherwise ignores reference/safezone guides and defaults to the first real
    watermark image (watermark-named PNGs first)."""
    cfg = cfg or {}
    imgs = [p for p in C.ASSETS.glob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not imgs:
        C.fail("no watermark image in campaign/assets/ — a watermark is mandatory on "
               "every clip for this campaign.")

    want = cfg.get("watermark_file")
    if want:
        for p in imgs:
            if p.name.lower() == want.lower() or want.lower() in p.name.lower():
                return p
        C.fail(f"configured watermark_file '{want}' not found in campaign/assets/. "
               f"Available: {[p.name for p in imgs]}")

    cands = [p for p in imgs if not any(h in p.name.lower() for h in REFERENCE_HINTS)]
    if not cands:
        C.fail("campaign/assets/ has only reference images (examples/safezones) — no "
               "actual watermark PNG. Add one or set config watermark_file.")
    # watermark-named PNGs first, then other candidates, alphabetical within each.
    cands.sort(key=lambda p: (
        "watermark" not in p.name.lower(), p.suffix.lower() != ".png", p.name.lower()))
    return cands[0]


# --- bounds + dead air ---------------------------------------------------------
def clip_bounds(m, duration, cmin, cmax, pre=20.0, post=15.0):
    """Expand the moment to a complete beat: back to its setup (up to `pre`) and
    forward to its resolution (up to `post`), then clamp to [cmin, cmax] and the
    source duration.

    When the beat is longer than cmax we do NOT clamp from the start (that shipped an
    arbitrary opening window off a long merged moment). Instead we CENTER the cmax window
    on the moment's peak-intensity second (audio spike or, since Task 3, a speech moment's
    loudest second), so the payoff stays in frame and the cold-open has its anchor."""
    s = float(m["start"]) - pre
    e = float(m["end"]) + post
    if e - s > cmax:                 # too long: center the window on the peak
        peak = m.get("peak")
        try:
            center = float(peak) if peak is not None else None
        except (TypeError, ValueError):
            center = None
        if center is None:
            center = (float(m["start"]) + float(m["end"])) / 2.0
        s = center - cmax / 2.0
        e = center + cmax / 2.0
    if e - s < cmin:                 # too short: pad symmetrically
        pad = (cmin - (e - s)) / 2
        s, e = s - pad, e + pad
    s = max(0.0, s)
    e = min(duration, e)
    # If clamping to a source EDGE shrank the window below cmin, recover from the OTHER side so a
    # moment near the start/end of a long VOD still gets a full-length window. Genuinely short
    # sources can't be recovered here and fall through to the content-length gate in run().
    if e - s < cmin:
        if s <= 0.0:
            e = min(duration, s + cmin)
        elif e >= duration:
            s = max(0.0, e - cmin)
    if e - s < 1.0:
        e = min(duration, s + max(cmin, 3))
    return round(s, 2), round(e, 2)


def cap_segments(segments, cap):
    """Trim the played segments so their COMBINED duration <= cap seconds — the
    cold-open teaser is prepended on top of the body, so without this a 30s body + 2s
    teaser ships a 32s clip. Trimming the TAIL also ends the clip nearer the payoff
    instead of padding to a target length."""
    out, total = [], 0.0
    for a, b in segments:
        if total >= cap:
            break
        seg = b - a
        if total + seg > cap:
            out.append((round(a, 2), round(a + (cap - total), 2)))
            total = cap
            break
        out.append((a, b))
        total += seg
    return out or segments[:1]


def peak_motion_time(source, peak, start, end, radius=MOTION_RADIUS):
    """Scan ±radius seconds around the audio peak and return the absolute source time
    of the highest-motion frame (max ffmpeg scene score) — so the cold-open opens on
    visible chaos, never a static wide shot. Best-effort: falls back to `peak`."""
    lo = max(start, peak - radius)
    hi = min(end, peak + radius)
    if hi - lo < 0.2:
        return peak
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{lo}", "-t", f"{round(hi - lo, 3)}",
           "-i", str(source), "-an", "-vf", "select='gt(scene,0)',metadata=print",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except Exception:
        return peak
    best_t, best_score, cur_t = None, -1.0, None
    for line in (proc.stderr or "").splitlines():
        mt = re.search(r"pts_time:([\d.]+)", line)
        if mt:
            cur_t = float(mt.group(1))
            continue
        ms = re.search(r"scene_score=([\d.]+)", line)
        if ms and cur_t is not None and float(ms.group(1)) > best_score:
            best_score, best_t = float(ms.group(1)), cur_t
    if best_t is None:
        return peak                       # no scene change detected in the window
    return round(lo + best_t, 2)


def cold_open_window(source, peak, start, end):
    """Clip-relative (a, b) for the cold-open teaser opened on the peak's highest-motion
    frame, or None if a clean >=COLD_OPEN_MIN window can't be carved out."""
    open_a = peak_motion_time(source, peak, start, end)
    open_a = max(start, min(open_a, end - COLD_OPEN_MIN))
    open_b = min(end, open_a + COLD_OPEN_DUR)
    if open_b - open_a < COLD_OPEN_MIN:
        open_a = max(start, open_b - COLD_OPEN_DUR)
    if open_b - open_a < COLD_OPEN_MIN:
        return None
    return (round(open_a - start, 2), round(open_b - start, 2))


def dead_air_keeps(source, start, end, n_audio=1, min_sil=1.5, noise="-30dB"):
    """Keep-intervals (relative to clip start) after removing internal silences
    longer than min_sil. Returns [(a,b), ...]; a single interval means no trim.
    Detects silence on the MERGED audio (all tracks) — otherwise a segment that is
    silent on track 0 but carries commentary on track 1 gets wrongly trimmed."""
    length = end - start
    base = ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start}", "-t", f"{round(length, 3)}",
            "-i", str(source)]
    sd = f"silencedetect=noise={noise}:d={min_sil}"
    if n_audio >= 2:
        labels = "".join(f"[0:a:{k}]" for k in range(n_audio))
        cmd = base + ["-filter_complex", f"{labels}amix=inputs={n_audio}:normalize=0,{sd}[s]",
                      "-map", "[s]", "-f", "null", "-"]
    else:
        cmd = base + ["-af", sd, "-f", "null", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    sils, cur = [], None
    for line in (proc.stderr or "").splitlines():
        a = re.search(r"silence_start:\s*([\d.]+)", line)
        b = re.search(r"silence_end:\s*([\d.]+)", line)
        if a:
            cur = float(a.group(1))
        elif b and cur is not None:
            sils.append((max(0.0, cur), min(length, float(b.group(1)))))
            cur = None
    keeps, pos = [], 0.0
    for a, b in sils:
        if a - pos > 0.1:
            keeps.append((round(pos, 2), round(a, 2)))
        pos = max(pos, b)
    if length - pos > 0.1:
        keeps.append((round(pos, 2), round(length, 2)))
    return keeps or [(0.0, round(length, 2))]


# --- compose -------------------------------------------------------------------
def has_audio_stream(path):
    """True if the source has at least one audio stream."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return bool((proc.stdout or "").strip())


def probe_dimensions(path):
    """(width, height) of the source's first video stream, or (None, None). Used to compute
    the blur_fill letterbox band so subtitles land below the footage (see subtitle_top_y)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (proc.stdout or "").strip()
    try:
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except (ValueError, IndexError):
        return None, None


def _content_fc(segments, cold_open, has_audio, n_audio, cfg):
    """Shared filtergraph CORE used by BOTH the blur_fill single-pass compose and the TRACK
    content pass: trim → CFR (fps) → cold-open fades → concat → loudnorm, then downscale the
    video to ≤ max_source_height (still source 16:9 aspect). Returns
    (fc_list, video_label='[dsrc]', audio_label='[aout]'|None, fps).

    Keeping this in one place means CFR/fade/audio correctness (incl. FIX 3's seam fix) can't
    drift between the two render paths."""
    trimming = cold_open or len(segments) > 1
    # FIX 3 — force CONSTANT frame rate. YouTube livestream VODs are often variable-frame-rate;
    # trimming VFR pieces and concatenating them lands frames on mismatched time grids, which the
    # concat filter renders as a visible stutter/lag spike at the cold-open→setup seam. We resample
    # EACH segment to CFR (fps filter) BEFORE concat so every frame is on one uniform grid.
    fps = int(cfg.get("output_fps", OUTPUT_FPS) or OUTPUT_FPS)
    fd = FADE_FRAMES / fps
    fc = []
    audio_label = None
    # A filter OUTPUT label is single-use, so when we merge tracks we asplit the mix into one copy
    # per consumer (segment). A single track reuses the [0:a] input pad (which IS multi-use).
    n_consumers = len(segments) if trimming else 1
    if has_audio and n_audio >= 2:
        mix = "".join(f"[0:a:{k}]" for k in range(n_audio))
        outs = "".join(f"[am{i}]" for i in range(n_consumers))
        fc.append(f"{mix}amix=inputs={n_audio}:normalize=0,asplit={n_consumers}{outs};")
        def _apad(i):
            return f"[am{i}]"
    else:
        def _apad(i):
            return "[0:a]"

    if trimming:                              # cold-open and/or dead-air via trim+concat
        for i, (a, b) in enumerate(segments):
            # fps BEFORE setpts+concat = every segment on one CFR grid → seamless join (FIX 3).
            vf = f"[0:v]trim={a}:{b},fps={fps},setpts=PTS-STARTPTS"
            if cold_open and i == 0:          # fade OUT the end of the teaser
                vf += f",fade=t=out:st={max(0.0, (b - a) - fd):.3f}:d={fd:.3f}"
            elif cold_open and i == 1:        # fade IN the start of the setup
                vf += f",fade=t=in:st=0:d={fd:.3f}"
            fc.append(vf + f"[v{i}];")
            if has_audio:
                fc.append(f"{_apad(i)}atrim={a}:{b},asetpts=PTS-STARTPTS[a{i}];")
        n = len(segments)
        if has_audio:
            fc.append("".join(f"[v{i}][a{i}]" for i in range(n))
                      + f"concat=n={n}:v=1:a=1[tv][tacc];")
            audio_label = "[tacc]"
        else:
            fc.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[tv];")
        vsrc = "[tv]"
    else:
        vsrc = "[0:v]"
        if has_audio:
            audio_label = _apad(0)

    aout = None
    if has_audio:
        fc.append(f"{audio_label}{LOUDNORM}[aout];")   # loudness normalize to a social target
        aout = "[aout]"
    # Downscale to max_source_height first so the WHOLE graph runs on ≤720p frames (4K through
    # split+scale+overlay OOMs). ,fps also normalizes the single-segment path to CFR.
    max_src_h = int(cfg.get("max_source_height", 720) or 720)
    fc.append(f"{vsrc}scale=-2:'min(ih,{max_src_h})',fps={fps}[dsrc];")
    return fc, "[dsrc]", aout, fps


def build_content_cmd(source, start, end, segments, cold_open, out_path, cfg, has_audio, n_audio=1):
    """TRACK pass 1: render the trimmed / cold-open-reordered clip as a plain ≤max_h 16:9 CFR
    video + loudnormed audio — the 'content' on the FINAL output timeline. reframe.py crops it to
    9:16 next; build_overlay_cmd burns the hook/subtitles/watermark afterwards. No vertical fill
    or overlays here, so the reframe operates on clean footage."""
    fc, vsrc, aout, fps = _content_fc(segments, cold_open, has_audio, n_audio, cfg)
    fc.append(f"{vsrc}copy[vout]")                      # [dsrc] → [vout] (already ≤max_h 16:9 CFR)
    threads = str(max(1, int(cfg.get("ffmpeg_threads", 2) or 2)))
    dur = round(end - start, 3)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-filter_complex_threads", threads, "-threads", threads,
           "-ss", f"{start}", "-t", f"{dur}", "-i", str(source),
           "-filter_complex", "".join(fc), "-map", "[vout]"]
    if has_audio and aout:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(out_path)]
    return cmd


def build_overlay_cmd(video_1080, audio_src, caption_png, watermark_png, ass_path, out_path,
                      cfg, has_audio, caption_y=CAPTION_TOP_Y):
    """TRACK pass 3: composite the hook caption PNG + watermark + karaoke ASS onto the already
    reframed 1080x1920 video, carrying the loudnormed audio from the content pass. This mirrors
    the overlay TAIL of build_compose_cmd so hooks/subtitles/watermark render IDENTICALLY on a
    TRACK clip — the reframe only changed the pixels underneath. `caption_y` = the hook's top Y
    (varies per clip; see resolve_clip_style)."""
    wm_scale = cfg.get("watermark_scale", 0.18)
    wm_margin = cfg.get("watermark_margin", 40)
    wm_w = int(W * wm_scale)
    fps = int(cfg.get("output_fps", OUTPUT_FPS) or OUTPUT_FPS)
    vlast = "[vpre]" if ass_path else "[vout]"
    fc = []
    if watermark_png:                                  # inputs: 0=video 1=caption 2=wm 3=audio
        audio_in = 3
        fc.append(f"[0:v][1:v]overlay=(W-w)/2:{caption_y}:eof_action=repeat[cap];")
        fc.append(f"[2:v]scale={wm_w}:-2[wm];")
        fc.append(f"[cap][wm]overlay=W-w-{wm_margin}:H-h-{wm_margin + 20}:eof_action=repeat{vlast}")
    else:                                              # inputs: 0=video 1=caption 2=audio
        audio_in = 2
        fc.append(f"[0:v][1:v]overlay=(W-w)/2:{caption_y}:eof_action=repeat{vlast}")
    if ass_path:
        fc.append(f";[vpre]{_ass_filter_arg(ass_path)}[vout]")
    threads = str(max(1, int(cfg.get("ffmpeg_threads", 2) or 2)))
    cmd = ["ffmpeg", "-y", "-v", "error", "-filter_complex_threads", threads, "-threads", threads,
           "-i", str(video_1080), "-i", str(caption_png)]
    if watermark_png:
        cmd += ["-i", str(watermark_png)]
    cmd += ["-i", str(audio_src), "-filter_complex", "".join(fc), "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", f"{audio_in}:a", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-movflags", "+faststart", str(out_path)]
    return cmd


def build_compose_cmd(source, start, end, segments, cold_open, caption_png, watermark_png,
                      out_path, cfg, has_audio, n_audio=1, ass_path=None, caption_y=CAPTION_TOP_Y):
    """`segments` are clip-relative (a, b) spans played in order. When cold_open is set,
    segment 0 is the peak teaser (opened on the payoff) and segment 1 is the setup —
    a 2-frame fade straddles that cut so it reads as intentional. Remaining segments are
    the dead-air-trimmed body.

    Audio: ALL source tracks are merged (amix) so no audio is ever lost — this VOD keeps
    the commentary/action on a 2nd track, and first-track-only left clips silent.

    `ass_path` (optional): a word-level karaoke ASS file whose timings were mapped through
    THESE same `segments` — burned last via libass, at 1080x1920 output resolution."""
    wm_scale = cfg.get("watermark_scale", 0.18)
    wm_margin = cfg.get("watermark_margin", 40)
    wm_w = int(W * wm_scale)
    # Shared trim→CFR→(cold-open fades)→concat→loudnorm→downscale core (also used by the TRACK
    # content pass) so CFR/fade/audio behavior lives in ONE place. `vsrc` = the ≤max_h 16:9 video
    # label ([dsrc]); audio, when present, is already loudnormed to [aout].
    fc, vsrc, _aout, fps = _content_fc(segments, cold_open, has_audio, n_audio, cfg)

    # Vertical fill. Default BLUR-FILL keeps the ENTIRE source frame visible (no
    # cropping): a COVER-scaled + heavily-blurred copy fills the 1080x1920 canvas as
    # a background, and the full frame (fit to 1080 width) is letterboxed onto it,
    # centered vertically. CROP-FILL (opt-in) instead COVER-scales and center-crops
    # the overflow — edge-to-edge but it crops content out of frame.
    layout = str(cfg.get("layout", "blur_fill")).lower()
    if layout == "crop_fill":
        fc.append(f"{vsrc}scale={W}:{H}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{H},setsar=1[base];")
    else:  # blur_fill (default)
        # A 16:9 frame fit to the full 1080 width already touches both side edges, so
        # the only way to grow it vertically (shrink the blurred dead area) is to zoom
        # in and trim the far LEFT/RIGHT edges — never the top/bottom, and never the
        # subjects. blur_fg_zoom controls that: 1.0 = pure no-crop letterbox; the
        # default trims only the outer edges (stream logos), keeping all content.
        fg_zoom = max(1.0, float(cfg.get("blur_fg_zoom", 1.2)))
        fg_box_w = int(W * fg_zoom)
        fc.append(f"{vsrc}split=2[bgsrc][fgsrc];")
        # background: COVER the full canvas, then heavy gaussian blur. Blur at LOW res
        # then upscale — visually identical (it's a blurred bg) but ~8x cheaper than
        # gblur at 1080x1920, which was the cut stage's bottleneck.
        fc.append(f"[bgsrc]scale={W}:{H}:force_original_aspect_ratio=increase,"
                  f"crop={W}:{H},scale=360:640,gblur=sigma=12,scale={W}:{H},setsar=1[bg];")
        # foreground: fit the WHOLE frame (decrease, no letterbox-crop), enlarged by
        # fg_zoom, then center-crop ONLY the horizontal overflow back to 1080 (height
        # is clamped to the canvas so the top/bottom of the frame is never cut).
        fc.append(f"[fgsrc]scale={fg_box_w}:{H}:force_original_aspect_ratio=decrease,"
                  f"crop='min(iw,{W})':'min(ih,{H})',setsar=1[fg];")
        fc.append(f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];")
    # eof_action=repeat keeps the single-frame caption/watermark PNGs on-screen for
    # the whole clip (otherwise they'd show for one frame). The watermark is OPTIONAL: when
    # the campaign confirms none is required (watermark_png is None), we overlay only the
    # caption and add no watermark input — so no stale/wrong watermark is ever burned in.
    # `vlast` is the composited-video label. When we have karaoke subtitles we burn them
    # LAST (after the blur-fill + caption + watermark, so libass renders at 1080x1920 output
    # res) by chaining the `ass` filter onto this label → [vout].
    vlast = "[vpre]" if ass_path else "[vout]"
    if watermark_png:
        fc.append(f"[base][1:v]overlay=(W-w)/2:{caption_y}:eof_action=repeat[cap];")
        fc.append(f"[2:v]scale={wm_w}:-2[wm];")
        fc.append(f"[cap][wm]overlay=W-w-{wm_margin}:H-h-{wm_margin + 20}:eof_action=repeat{vlast}")
    else:
        fc.append(f"[base][1:v]overlay=(W-w)/2:{caption_y}:eof_action=repeat{vlast}")
    if ass_path:
        fc.append(f";[vpre]{_ass_filter_arg(ass_path)}[vout]")

    # Cap ffmpeg threads: each decode/filter/encode thread buffers frames, so fewer threads =
    # much lower peak RAM (helps avoid the 4K OOM alongside the downscale above). Default 2.
    threads = str(max(1, int(cfg.get("ffmpeg_threads", 2) or 2)))
    dur = round(end - start, 3)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-filter_complex_threads", threads, "-threads", threads,
           "-ss", f"{start}", "-t", f"{dur}", "-i", str(source),
           "-i", str(caption_png)]
    if watermark_png:
        cmd += ["-i", str(watermark_png)]
    cmd += ["-filter_complex", "".join(fc), "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(fps),   # stamp CFR at the encoder (FIX 3)
            "-movflags", "+faststart", str(out_path)]
    return cmd


def compose(source, start, end, segments, cold_open, caption_png, watermark_png, out_path,
            cfg, has_audio, n_audio=1, ass_path=None, caption_y=CAPTION_TOP_Y):
    cmd = build_compose_cmd(source, start, end, segments, cold_open, caption_png,
                            watermark_png, out_path, cfg, has_audio, n_audio, ass_path, caption_y)
    C.run_cmd(cmd, desc=f"cutting {out_path.name}")


def compose_track(source, start, end, segments, cold_open, caption_png, watermark_png, out_path,
                  cfg, has_audio, n_audio, ass_path, detectors, rank, caption_y=CAPTION_TOP_Y):
    """3-pass TRACK render: (1) build the trimmed 16:9 content clip on the output timeline,
    (2) reframe.py crops it 9:16 following the subject, (3) burn hook/subtitles/watermark on top.
    Returns True on success, or False to signal the caller to fall back to blur_fill (a reframe
    failure — deps/detector/geometry). Intermediates live under drafts/ and are always cleaned."""
    import reframe
    content = C.DRAFTS / f".content_{rank:02d}.mp4"
    tracked = C.DRAFTS / f".tracked_{rank:02d}.mp4"
    fps = int(cfg.get("output_fps", OUTPUT_FPS) or OUTPUT_FPS)
    try:
        C.run_cmd(build_content_cmd(source, start, end, segments, cold_open, content, cfg,
                                    has_audio, n_audio), desc=f"track p1/3 (content) {out_path.name}")
        if not reframe.track_reframe(content, tracked, cfg, detectors, fps=fps):
            return False
        C.run_cmd(build_overlay_cmd(tracked, content, caption_png, watermark_png, ass_path,
                                    out_path, cfg, has_audio, caption_y=caption_y),
                  desc=f"track p3/3 (overlay) {out_path.name}")
        return True
    finally:
        content.unlink(missing_ok=True)
        tracked.unlink(missing_ok=True)


# --- misc ----------------------------------------------------------------------
def slugify(text, n=40):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "clip").lower()).strip("-")
    return (s[:n].strip("-") or "clip")


def _durations(sources):
    cache = {}
    for s in sources:
        cache[s["source"]] = s.get("duration_sec") or C.ffprobe_duration(C.ROOT / s["source"])
    return cache


def run(state):
    C.require_exe("ffmpeg")
    caps = C.load_json(C.CAPTIONS_JSON)
    if not caps:
        C.fail("campaign/captions.json missing — run the captions stage first.")
    rules = C.load_json(C.RULES_JSON) or {}
    banned = list(rules.get("banned_words", C.DEFAULT_BANNED_WORDS)) + list(rules.get("banned_topics", []))
    if C.load_knowledge():
        C.log("loaded campaign/knowledge.md for campaign context.")
    moments = C.load_json(C.MOMENTS_JSON) or {"sources": []}
    durations = _durations(moments.get("sources", []))
    # Per-source word timings (index.py's whisper word_timestamps), mapped through each
    # clip's segment reorder to drive the karaoke subtitles. Absent on a moments.json built
    # before this feature — subtitles then no-op (re-run index to populate them).
    words_by_source = {s["source"]: s.get("words", []) for s in moments.get("sources", [])}

    cfg = state.get("config", {})
    cmin = float(cfg.get("clip_min_seconds", 15))
    cmax = float(cfg.get("clip_max_seconds", 45))
    pre = float(cfg.get("story_pre_seconds", 20))
    post = float(cfg.get("story_post_seconds", 15))
    layout = str(cfg.get("layout", "auto")).lower()   # auto | track | blur_fill | crop_fill
    # TRACK-mode face/person reframe (OpenShorts-style). Only load the heavy detectors when the
    # layout can actually use them (auto/track) AND the deps import; otherwise every clip uses
    # blur_fill exactly as before. Detectors load ONCE and are reused across all clips.
    detectors = None
    if layout in ("auto", "track"):
        import reframe
        if reframe.deps_available():
            detectors = reframe.Detectors(cfg)
            if not detectors.usable:
                detectors = None
        if detectors is None:
            C.warn(f"TRACK: layout='{layout}' requested but reframe is unavailable — "
                   "falling back to blur_fill for all clips.")
    emoji_in_caption = bool(cfg.get("emoji_in_caption", True))
    # Hook vs subtitle are visually DISTINCT layers. The HOOK is the plated scroll-stopper
    # at the top (Pillow PNG, larger text, denser plate — hook_font_scale / hook_plate_opacity).
    # The SUBTITLES are word-level karaoke lower-center (ASS/libass, build_ass) — a different
    # KIND of element, not a smaller plate.
    hook_font_scale = float(cfg.get("hook_font_scale", 1.06))     # >1 = larger/bolder hook
    # HOOK plate/outline preset (A|B|C|D). Constant across clips (white text, fixed TOP) so the
    # plate/outline choice can be judged on its own — the karaoke keeps its per-clip variety.
    hook_style = resolve_hook_style(cfg)
    C.log(f"== hook style: {str(cfg.get('hook_style', DEFAULT_HOOK_STYLE)).upper()} "
          f"(plate_opacity={hook_style['plate_opacity']}, stroke={hook_style['stroke']}) ==")
    # Karaoke subtitles need per-word timings from index (whisper) → off in offline mode
    # (index skips whisper) and when moments.json predates the feature (no words stored).
    subs_on = bool(cfg.get("subtitles_enabled", True)) and not C.offline_mode()
    if subs_on and not any(words_by_source.values()):
        C.warn("subtitles enabled but moments.json has no word timings — re-run the index "
               "stage to populate them. Shipping clips WITHOUT karaoke subtitles.")
        subs_on = False
    C.log(f"== cut layout mode: {layout} "
          f"({'whole frame on blurred bg, no crop' if layout != 'crop_fill' else 'COVER + center-crop'}) "
          f"| emoji_in_caption={emoji_in_caption} | subtitles={'on' if subs_on else 'off'} ==")
    # Watermark is applied unless the campaign explicitly confirms none is required
    # (rules.json watermark_required=false). This both proceeds without a watermark AND
    # guards against burning a STALE/wrong watermark left in assets/ from a prior campaign.
    if rules.get("watermark_required") is False:
        watermark = None
        C.log("watermark: not required for this campaign (rules.watermark_required=false) — "
              "skipping; no watermark will be burned in.")
    else:
        watermark = find_watermark(cfg)
        C.log(f"watermark: {watermark.name}")

    C.DRAFTS.mkdir(parents=True, exist_ok=True)
    camp_tag = C.campaign_tag(caps.get("campaign"))   # filename prefix so each clip names its campaign
    ranked_clips = sorted(caps["clips"], key=lambda c: (c.get("score") or 0), reverse=True)
    audio_cache = {}
    dim_cache = {}

    # FIX 5 — DUPLICATE GUARD: no two OUTPUT clips may cover overlapping SOURCE time. Clips are
    # best-first, so keep the higher-scored clip and DROP any later one whose (final) window
    # overlaps it. This is the last-line guarantee (belt-and-suspenders over select's spread
    # check) that a near-duplicate never reaches drafts/. Windows are computed once here.
    clips, kept_windows = [], []
    for c in ranked_clips:
        duration = durations.get(c["source"]) or C.ffprobe_duration(C.ROOT / c["source"])
        s, e = clip_bounds(c, duration, cmin, cmax, pre, post)
        dup = next((w for w in kept_windows
                    if w[0] == c["source"] and not (e <= w[1] or s >= w[2])), None)
        if dup:
            C.log(f"dedup: dropping {c['moment_id']} (window {s:.0f}-{e:.0f}s) — overlaps a "
                  f"higher-scored clip's window {dup[1]:.0f}-{dup[2]:.0f}s.")
            continue
        kept_windows.append((c["source"], s, e))
        clips.append((c, s, e))
    if len(clips) < len(ranked_clips):
        C.log(f"dedup: dropped {len(ranked_clips) - len(clips)} overlapping clip(s); "
              f"{len(clips)} unique clip(s) to render.")

    manifest = []
    for (c, start, end) in clips:
        # final rules gate (defensive — the gauntlet already filtered)
        for field in ("caption", "tiktok_caption", "shorts_title"):
            from captions import banned_hit
            if banned_hit(c.get(field, ""), banned):
                C.fail(f"banned word slipped into {field} for clip {c['moment_id']} — aborting.")

        src_path = C.ROOT / c["source"]
        if c["source"] not in audio_cache:
            audio_cache[c["source"]] = C.audio_stream_count(src_path)
        n_audio = audio_cache[c["source"]]
        has_audio = n_audio > 0

        keeps = dead_air_keeps(src_path, start, end, n_audio)

        # COLD-OPEN (FIX 1/2/4) — CONDITIONAL on a genuine single sharp peak with real payoff
        # separation; see plan_cold_open. Flat-high clips and edge/too-soon peaks play straight.
        cold_rel, reason = plan_cold_open(src_path, c["source"], start, end, keeps, cmax, cfg,
                                          hook_moment=c.get("hook_moment"))
        cold_open = cold_rel is not None
        C.log(f"cold-open [{c['moment_id']}]: {reason}")
        segments = ([cold_rel] if cold_open else []) + keeps
        segments = cap_segments(segments, cmax)      # total playtime <= clip_max_seconds

        # MIN-LENGTH GATE (1c): require >= clip_min_seconds of ACTUAL content, EXCLUDING the
        # cold-open teaser. dead-air trims and source-edge clamping can shrink a nominal window
        # well below the minimum; a ~4-6s stub is not postable, so DROP the moment rather than
        # ship it. This is where clip_min_seconds is truly enforced (clip_bounds only sizes the
        # window; the played content after trimming is what actually matters).
        teaser_dur = (segments[0][1] - segments[0][0]) if cold_open else 0.0
        body_content = sum(b - a for a, b in segments) - teaser_dur
        if body_content + 1e-6 < cmin:
            C.log(f"drop [{c['moment_id']}]: only {body_content:.1f}s of content after dead-air "
                  f"trim (< {cmin:g}s minimum) — skipping stub clip.")
            continue
        rank = len(manifest) + 1          # contiguous output rank (drops leave no numbering gaps)

        score_i = int(round(c.get("score") or 0))
        name = f"{camp_tag}_{rank:02d}_{score_i:03d}_{slugify(c['caption'])}.mp4"
        out_path = C.DRAFTS / name
        cap_png = C.DRAFTS / f".cap_{rank:02d}.png"
        # PER-CLIP KARAOKE STYLE (deterministic, seeded by moment id → stable on re-cut): accent
        # colour + active-word emphasis mode. The HOOK is intentionally CONSTANT — plain white text
        # at the fixed TOP position — with only its plate/outline set by the hook_style preset.
        style = resolve_clip_style(cfg, c["moment_id"])
        render_caption_png(c["caption"], cap_png, emoji=emoji_in_caption,
                           font_scale=hook_font_scale, text_color="white",
                           max_font=int(cfg.get("hook_max_font_size", HOOK_MAX_FONT)), **hook_style)

        # --- LAYOUT: resolve TRACK vs GENERAL (blur_fill) for THIS clip ---
        # blur_fill/crop_fill are forced GENERAL; track forces TRACK; auto samples the clip and
        # picks TRACK only for a single clear subject (else blur_fill). eff_layout also drives
        # the karaoke placement (TRACK fills the frame → no lower letterbox band).
        eff_layout = layout if layout in ("blur_fill", "crop_fill") else "blur_fill"
        use_track, track_reason = False, ""
        if detectors is not None:
            if layout == "track":
                use_track, track_reason = True, "forced (layout=track)"
            else:
                use_track, track_reason = reframe.decide_track(
                    src_path, cfg, detectors, window=(start, end))
            eff_layout = "track" if use_track else "blur_fill"
        C.log(f"  layout [{c['moment_id']}]: "
              + (f"TRACK — {track_reason}" if use_track
                 else "GENERAL blur_fill" + (f" — {track_reason}" if track_reason else "")))

        # KARAOKE SUBTITLES: take the clip's source words, map them through the SAME
        # `segments` compose plays (cold-open reorder + dead-air trims), and write an ASS
        # file whose per-word events land on the final output timeline. build_compose_cmd
        # burns it last (after the blur-fill), so it stays synced to the reordered edit.
        ass_path = None
        if subs_on and has_audio:
            src_words = [w for w in words_by_source.get(c["source"], [])
                         if float(w["end"]) > start and float(w["start"]) < end]
            events = map_words_to_output(src_words, start, segments)
            # Non-speech sound labels (captions stage, Groq-inferred) mapped through the SAME
            # segments so a '*scream*' lands on the beat in the FINAL timeline, in a word gap.
            fx_events = []
            for fx in (c.get("sound_fx") or []):
                mapped = map_words_to_output(
                    [{"word": fx["label"], "start": float(fx["start"]), "end": float(fx["end"])}],
                    start, segments)
                fx_events += [{"label": e["word"], "start": e["start"], "end": e["end"]}
                              for e in mapped]
            # Pin the karaoke line into the lower letterbox band BELOW the footage, using the
            # same blur_fill geometry compose renders (source aspect drives the band height).
            if c["source"] not in dim_cache:
                dim_cache[c["source"]] = probe_dimensions(src_path)
            sw, sh = dim_cache[c["source"]]
            top_y = subtitle_top_y(cfg, sw, sh, layout=eff_layout)
            ass_file = C.DRAFTS / f".sub_{rank:02d}.ass"
            ass_path = build_ass(events, cfg, banned, ass_file, top_y, fx_events, style=style)
        subtitled = ass_path is not None

        # RENDER: TRACK uses the 3-pass reframe (content → crop → overlay); on any reframe
        # failure it falls back to the normal blur_fill single pass. Everything else (hook,
        # karaoke, watermark, cold-open, CFR) is identical across both.
        if use_track:
            if not compose_track(src_path, start, end, segments, cold_open, cap_png, watermark,
                                 out_path, cfg, has_audio, n_audio, ass_path, detectors, rank,
                                 caption_y=CAPTION_TOP_Y):
                eff_layout = "blur_fill"
                C.warn(f"  {c['moment_id']}: TRACK reframe failed → blur_fill fallback.")
                compose(src_path, start, end, segments, cold_open, cap_png, watermark, out_path,
                        cfg, has_audio, n_audio, ass_path=ass_path, caption_y=CAPTION_TOP_Y)
        else:
            compose(src_path, start, end, segments, cold_open, cap_png, watermark, out_path,
                    cfg, has_audio, n_audio, ass_path=ass_path, caption_y=CAPTION_TOP_Y)
        cap_png.unlink(missing_ok=True)
        if ass_path:
            ass_path.unlink(missing_ok=True)
        C.log(f"  {'cold-open ' if cold_open else ''}{'subtitled ' if subtitled else ''}"
              f"[{eff_layout.upper()}] cut {name}")

        manifest.append({
            "filename": name, "caption": c["caption"], "variant": c.get("variant"),
            "source": c["source"], "source_start": start, "source_end": end,
            "layout": eff_layout,
            "accent": "#" + style["accent_rgb"], "emphasis": style["emphasis"],
            "hook_style": str(cfg.get("hook_style", DEFAULT_HOOK_STYLE)).upper(),
            "cold_open": cold_open, "dead_air_trimmed": len(keeps) > 1,
            "subtitles": subtitled, "score": c.get("score"),
            "tiktok_caption": c["tiktok_caption"], "shorts_title": c["shorts_title"],
            "reels_hashtags": c["reels_hashtags"],
            "suggested_post_window": c["suggested_post_window"],
        })

    C.save_json(C.DRAFTS_MANIFEST, {"campaign": caps.get("campaign"),
                                    "created_at": C.now_iso(), "clips": manifest})
    C.mark_stage(state, "cut", delivered=len(manifest))
    C.log(f"cut done: {len(manifest)} clip(s) in drafts/ (best first).")


if __name__ == "__main__":
    run(C.load_state())
