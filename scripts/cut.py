"""CUT + DELIVER stage.

Per selected clip: extract the moment, tighten dead air, render to vertical
1080x1920, burn the flzsh caption at top (safe-zone aware), overlay the mandatory
watermark, and write drafts/NN_score_slug.mp4 (best first) + drafts/manifest.json.

Captions are rendered to a transparent PNG with Pillow (bold white, black outline,
top-center, <=2 lines, auto font-size) and overlaid by ffmpeg — this dodges ffmpeg
drawtext font/escaping issues and is identical across OSes. Watermark is the campaign
PNG from campaign/assets/. Fail loud if anything essential is missing.
"""
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont, ImageFilter

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
# blurred/black dead space. We center the subtitle plate at ~1440: below the footage frame,
# clear of the top hook caption, above the bottom ~18-20% TikTok caption/UI zone (~1536+),
# and — at 640px wide, centered (x 220-860) — inside the right action-rail notch (~x<870)
# and above the bottom-right watermark. So subtitles never sit on the footage, the
# watermark, or the platform UI.
SUBTITLE_BOX_W = 640
SUBTITLE_CENTER_Y = 1440      # ~75% down: in the lower black band, off the video frame

# --- cold-open restructure (the biggest hook lever) ----------------------------
COLD_OPEN_DUR = 2.0           # target length of the peak teaser (spec: 1.5–2.5s)
COLD_OPEN_MIN = 1.5           # never shorter than this or it reads as a glitch
COLD_OPEN_MIN_SETUP = 3.0     # peak must sit >= this many s past the natural start,
COLD_OPEN_MIN_PAYOFF = 2.0    # and leave >= this much clip after it, else play in order
MOTION_RADIUS = 1.0           # scan ±1s around the peak for the highest-motion frame
FADE_FRAMES = 2               # 2-frame fade on the cold-open->setup cut (reads intentional)
ASSUMED_FPS = 30.0            # fade duration basis when the true fps is unknown
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"   # per-clip audio normalization (social target)


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


# Plates hug the text — kept tight so neither caption reads as a bulky block. The HOOK is
# the scroll-stopper (bigger text, denser plate); SUBTITLES are quieter readability support
# (smaller text, lighter/tighter plate). Each treatment is independently tunable via config:
# hook_font_scale / hook_plate_opacity vs subtitle_font_scale / subtitle_plate_opacity.
CAPTION_PLATE_PAD_X = 20
CAPTION_PLATE_PAD_Y = 9
CAPTION_PLATE_RADIUS = 18

SUBTITLE_PLATE_PAD_X = 14
SUBTITLE_PLATE_PAD_Y = 6
SUBTITLE_PLATE_RADIUS = 14


def render_caption_png(text, out_path, box_w=CAPTION_BOX_W, max_lines=2, stroke=2,
                       emoji=True, plate_opacity=105, font_scale=1.0):
    """The HOOK caption — the scroll-stopper, so it is the PROMINENT of the two text layers
    (larger/bolder than the burned subtitles). Clean, understated (Title Case, emoji as
    punctuation): white text on a SEMI-TRANSPARENT DARK PLATE so it stays legible on ANY
    background — bright, dark, or busy — with only a THIN (or no) outline; the plate does the
    contrast work, not a heavy stroke. `plate_opacity` (0-255, 0 = no plate), `font_scale`
    (>1 = larger/bolder-looking) and `stroke` (0 = no outline) are the tunable knobs.

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
    inner = box_w - 2 * CAPTION_PLATE_PAD_X - 16
    smax = max(int(round(84 * font_scale)), 34)   # hook runs larger than the subtitles
    smin = max(int(round(30 * font_scale)), 20)
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
    pad_x, pad_y = CAPTION_PLATE_PAD_X, CAPTION_PLATE_PAD_Y
    img_h = line_h * len(lines) + 2 * pad_y
    img = Image.new("RGBA", (box_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Semi-transparent dark plate hugging the widest line — guarantees white-on-anything
    # contrast without a loud outline. plate_opacity<=0 disables it.
    maxw = max((sum(c["w"] for c in ln) for ln in lines), default=0.0)
    if plate_opacity > 0 and maxw > 0:
        pw = min(float(box_w), maxw + 2 * pad_x)
        px0 = (box_w - pw) / 2
        d.rounded_rectangle([px0, 0, px0 + pw, img_h], radius=CAPTION_PLATE_RADIUS,
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
                d.text((x, y), c["s"], font=c["font"], fill="white",
                       stroke_width=stroke, stroke_fill="black")
            else:
                d.text((x, y), c["s"], font=c["font"], fill="white")
            x += c["w"]
        y += line_h
    img.save(out_path)
    return out_path


# --- burned karaoke-style subtitles --------------------------------------------
# TIMING APPROACH: we RE-TRANSCRIBE the finished 30s clip rather than remapping source
# word timestamps. The clip is restructured (cold-open reorder + dead-air trims), so
# source timings would desync; a fresh 30s whisper pass is edit-proof and takes seconds
# on CPU. Subtitles are a SEPARATE layer from the hook caption and are gated off in
# offline mode (no whisper) and by config `subtitles_enabled=false`.
_sub_model = None


def _subtitle_model():
    global _sub_model
    if _sub_model is None:
        from faster_whisper import WhisperModel        # lazy: cut.py must import w/o it
        C.log("loading faster-whisper 'base' for subtitle timing (CPU, int8)…")
        _sub_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _sub_model


def transcribe_clip_words(clip_path):
    """Clip-relative word timings from a whisper pass on the FINAL clip. Best-effort:
    returns [] (ship without subtitles) if whisper is missing or transcription fails."""
    try:
        model = _subtitle_model()
    except ImportError:
        C.warn("faster-whisper not installed — skipping burned subtitles.")
        return []
    try:
        segments, _info = model.transcribe(str(clip_path), word_timestamps=True)
        words = []
        for seg in segments:
            for w in (seg.words or []):
                t = (w.word or "").strip()
                if t:
                    words.append({"word": t, "start": float(w.start), "end": float(w.end)})
        return words
    except Exception as e:
        C.warn(f"subtitle transcription failed ({e}) — shipping clip without subtitles.")
        return []


def _mask_banned_word(word, banned):
    """Mask any campaign-banned word before it can be burned into a subtitle
    ('bet' -> 'b**', 'gambling' -> 'g*******'). Surrounding punctuation is preserved;
    clean words pass through untouched. A banned word in a subtitle = campaign violation,
    so this is the FIRST line of defense (cut.py re-checks the whole chunk after)."""
    from captions import banned_hit
    m = re.match(r"^(\W*)(.*?)(\W*)$", word, re.S)
    pre, core, post = m.group(1), m.group(2), m.group(3)
    if core and banned_hit(core, banned):
        core = (core[0] + "*" * (len(core) - 1)) if len(core) > 1 else "*"
    return pre + core + post


def _chunk_words(words, banned, size=3, max_gap=0.7):
    """Group scrubbed words into short synced phrase chunks (<= `size` words; a new
    chunk also starts after a >max_gap pause). Each chunk carries clip-relative start/end."""
    chunks, cur = [], []
    for w in words:
        tok = _mask_banned_word(w["word"].strip(), banned)
        if not tok:
            continue
        if cur and (len(cur) >= size or w["start"] - cur[-1]["end"] > max_gap):
            chunks.append(_finish_chunk(cur)); cur = []
        cur.append({"tok": tok, "start": w["start"], "end": w["end"]})
    if cur:
        chunks.append(_finish_chunk(cur))
    return chunks


def _finish_chunk(cur):
    return {"text": " ".join(c["tok"] for c in cur).strip(),
            "start": round(cur[0]["start"], 3), "end": round(cur[-1]["end"], 3)}


def _draw_text_lines(d, lines, box_w, y0, line_h, fill, stroke=0, stroke_fill="black", x_off=0):
    """Draw already-wrapped text `lines` centered in `box_w`, top-down from y0."""
    y = y0
    for ln in lines:
        total = sum(c["w"] for c in ln)
        x = (box_w - total) / 2 + x_off
        for c in ln:
            if stroke > 0:
                d.text((x, y), c["s"], font=c["font"], fill=fill,
                       stroke_width=stroke, stroke_fill=stroke_fill)
            else:
                d.text((x, y), c["s"], font=c["font"], fill=fill)
            x += c["w"]
        y += line_h


def render_subtitle_png(text, out_path, cfg, box_w=SUBTITLE_BOX_W, max_lines=2):
    """A subtitle phrase chunk — DELIBERATELY a different KIND of element from the plated
    hook, not just a smaller version of it. Default is clean white text FLOATING with a
    soft drop shadow + thin outline (NO box), so it reads as plain spoken-word text while
    the hook stays the plated headline. `subtitle_plate` (config, default FALSE) brings the
    dark plate back if wanted. Smaller than the hook (`subtitle_font_scale`). Returns the
    PNG height so the caller can vertically center it in the lower-band safe zone."""
    plate_on = bool(cfg.get("subtitle_plate", False))            # default: NO plate — floats
    plate_opacity = int(cfg.get("subtitle_plate_opacity", 70))
    font_scale = float(cfg.get("subtitle_font_scale", 0.82))     # smaller than the hook text
    stroke = int(cfg.get("caption_outline_width", 2))            # thin outline for legibility
    # Soft drop-shadow (only when no plate) — carries legibility on bright OR dark footage
    # without a box. Down-offset + blur reads as a natural shadow, not a second outline.
    sh_blur = float(cfg.get("subtitle_shadow_blur", 4))
    sh_dx = int(cfg.get("subtitle_shadow_dx", 0))
    sh_dy = int(cfg.get("subtitle_shadow_dy", 4))
    sh_op = int(cfg.get("subtitle_shadow_opacity", 200))
    font_path = find_bold_font()
    from captions import titlecase           # Title Case burned subtitles (same as captions)
    text = titlecase(strip_to_ascii(text or "").strip()) or " "
    scratch = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    pad_x, pad_y = SUBTITLE_PLATE_PAD_X, SUBTITLE_PLATE_PAD_Y
    inner = box_w - 2 * pad_x - 12
    smax = max(int(round(60 * font_scale)), 26)
    smin = max(int(round(28 * font_scale)), 16)
    size, lines, tf = smax, [[]], None
    while size >= smin:
        tf = ImageFont.truetype(font_path, size)
        lines = _wrap_cells(_cells(text, tf, None, scratch), inner, max_lines)
        if len(_wrap_cells(_cells(text, tf, None, scratch), inner, max_lines + 1)) <= max_lines:
            break
        size -= 4
    ascent, descent = tf.getmetrics()
    line_h = ascent + descent + 6
    maxw = max((sum(c["w"] for c in ln) for ln in lines), default=0.0)

    # Reserve margin for the soft-shadow bleed (blur + offset) when there's no plate, so
    # the shadow isn't clipped at the PNG edges.
    sh_margin = 0 if plate_on else int(round(sh_blur * 2 + max(abs(sh_dx), abs(sh_dy)) + stroke))
    top = pad_y + sh_margin
    img_h = line_h * len(lines) + 2 * pad_y + 2 * sh_margin
    img = Image.new("RGBA", (box_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if plate_on and plate_opacity > 0 and maxw > 0:
        pw = min(float(box_w), maxw + 2 * pad_x)
        px0 = (box_w - pw) / 2
        d.rounded_rectangle([px0, top - pad_y, px0 + pw, top - pad_y + line_h * len(lines) + 2 * pad_y],
                            radius=SUBTITLE_PLATE_RADIUS, fill=(0, 0, 0, plate_opacity))
    elif not plate_on and maxw > 0:
        # SOFT DROP SHADOW (no box): draw the phrase black on its own layer (offset), blur
        # it, and composite under the white text.
        shadow = Image.new("RGBA", (box_w, img_h), (0, 0, 0, 0))
        _draw_text_lines(ImageDraw.Draw(shadow), lines, box_w, top + sh_dy, line_h,
                         fill=(0, 0, 0, sh_op), stroke=0, x_off=sh_dx)
        img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(sh_blur)))
        d = ImageDraw.Draw(img)

    _draw_text_lines(d, lines, box_w, top, line_h, fill="white",
                     stroke=stroke, stroke_fill="black")
    img.save(out_path)
    return img_h


def _subtitle_overlay_cmd(body_path, out_path, chunks, cfg, has_audio):
    center_y = int(cfg.get("subtitle_center_y", SUBTITLE_CENTER_Y))
    threads = str(max(1, int(cfg.get("ffmpeg_threads", 2) or 2)))
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-filter_complex_threads", threads, "-threads", threads, "-i", str(body_path)]
    for ch in chunks:
        cmd += ["-i", str(ch["_png"])]
    fc, prev = [], "[0:v]"
    for i, ch in enumerate(chunks):
        y = int(center_y - ch["_h"] / 2)
        out = "[vout]" if i == len(chunks) - 1 else f"[sv{i}]"
        # eof_action=repeat keeps the single-frame PNG available for the whole clip;
        # enable gates it to the phrase's [start,end]. Audio is copied (no re-encode).
        fc.append(f"{prev}[{i + 1}:v]overlay=x=(W-w)/2:y={y}:eof_action=repeat:"
                  f"enable='between(t,{ch['start']:.3f},{ch['end']:.3f})'{out}")
        prev = out
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "0:a?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    return cmd


def burn_subtitles(body_path, out_path, cfg, banned, tag, has_audio):
    """Transcribe the FINAL clip, chunk into short synced phrases, scrub banned words,
    and overlay karaoke-style subtitle PNGs into `out_path`. Returns True if subtitles
    were burned, False if there was nothing to burn (body moved to out_path as-is)."""
    if not has_audio:
        os.replace(body_path, out_path); return False
    chunks = _chunk_words(transcribe_clip_words(body_path), banned)
    if not chunks:
        os.replace(body_path, out_path); return False

    from captions import banned_hit
    for i, ch in enumerate(chunks):
        png = C.DRAFTS / f".sub_{tag:02d}_{i:03d}.png"
        ch["_h"] = render_subtitle_png(ch["text"], png, cfg)
        ch["_png"] = png
        # DEFENSE-IN-DEPTH: same gauntlet pattern as captions — a banned word must never
        # reach a burned subtitle. Masking above should have caught it; abort if not.
        if banned_hit(ch["text"], banned):
            for c in chunks:
                if c.get("_png"):
                    c["_png"].unlink(missing_ok=True)
            C.fail(f"banned word slipped into a subtitle chunk ({ch['text']!r}) — aborting.")

    C.run_cmd(_subtitle_overlay_cmd(body_path, out_path, chunks, cfg, has_audio),
              desc=f"burning {len(chunks)} subtitle chunk(s) into {out_path.name}")
    for ch in chunks:
        ch["_png"].unlink(missing_ok=True)
    return True


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


def build_compose_cmd(source, start, end, segments, cold_open, caption_png, watermark_png,
                      out_path, cfg, has_audio, n_audio=1):
    """`segments` are clip-relative (a, b) spans played in order. When cold_open is set,
    segment 0 is the peak teaser (opened on the payoff) and segment 1 is the setup —
    a 2-frame fade straddles that cut so it reads as intentional. Remaining segments are
    the dead-air-trimmed body.

    Audio: ALL source tracks are merged (amix) so no audio is ever lost — this VOD keeps
    the commentary/action on a 2nd track, and first-track-only left clips silent."""
    wm_scale = cfg.get("watermark_scale", 0.18)
    wm_margin = cfg.get("watermark_margin", 40)
    wm_w = int(W * wm_scale)
    trimming = cold_open or len(segments) > 1
    fd = FADE_FRAMES / ASSUMED_FPS

    fc = []
    audio_label = None                        # filter label OR raw stream feeding loudnorm
    # Build a reusable audio source. A filter OUTPUT label is single-use, so when we
    # merge tracks we asplit the mix into one copy per consumer (segment). With a single
    # track we just reuse the [0:a] input pad (which IS multi-use) as before.
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
            vf = f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS"
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

    # Per-clip audio peak/loudness normalization to a consistent social target.
    if has_audio:
        fc.append(f"{audio_label}{LOUDNORM}[aout];")

    # Downscale the source to max_source_height BEFORE any blur-fill/scale/overlay work, so the
    # WHOLE filtergraph runs on <=720p frames. 4K (3840x2160) through split+scale+overlay
    # exhausts RAM ("Cannot allocate memory -12"); the final output is 1080x1920 regardless, so
    # a 720 source loses nothing visible. Quoted min() protects the inner comma; -2 keeps the
    # width even for yuv420p; a source already <= max_h is left unchanged.
    max_src_h = int(cfg.get("max_source_height", 720) or 720)
    fc.append(f"{vsrc}scale=-2:'min(ih,{max_src_h})'[dsrc];")
    vsrc = "[dsrc]"

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
    if watermark_png:
        fc.append(f"[base][1:v]overlay=(W-w)/2:{CAPTION_TOP_Y}:eof_action=repeat[cap];")
        fc.append(f"[2:v]scale={wm_w}:-2[wm];")
        fc.append(f"[cap][wm]overlay=W-w-{wm_margin}:H-h-{wm_margin + 20}:eof_action=repeat[vout]")
    else:
        fc.append(f"[base][1:v]overlay=(W-w)/2:{CAPTION_TOP_Y}:eof_action=repeat[vout]")

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
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    return cmd


def compose(source, start, end, segments, cold_open, caption_png, watermark_png, out_path,
            cfg, has_audio, n_audio=1):
    cmd = build_compose_cmd(source, start, end, segments, cold_open, caption_png,
                            watermark_png, out_path, cfg, has_audio, n_audio)
    C.run_cmd(cmd, desc=f"cutting {out_path.name}")


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

    cfg = state.get("config", {})
    cmin = float(cfg.get("clip_min_seconds", 15))
    cmax = float(cfg.get("clip_max_seconds", 45))
    pre = float(cfg.get("story_pre_seconds", 20))
    post = float(cfg.get("story_post_seconds", 15))
    layout = str(cfg.get("layout", "blur_fill")).lower()
    emoji_in_caption = bool(cfg.get("emoji_in_caption", True))
    # Hook vs subtitle are visually DISTINCT and independently tunable. The HOOK is the
    # scroll-stopper (larger text, denser plate); SUBTITLES recede as readability support
    # (smaller text, lighter plate — read inside render_subtitle_png). Both plates hug the
    # text and run at reduced opacity so neither is a bulky block. `plate_opacity` (legacy,
    # shared) is superseded by hook_/subtitle_ specific knobs.
    hook_plate_opacity = int(cfg.get("hook_plate_opacity", 105))  # dark plate alpha (0-255; 0=off)
    hook_font_scale = float(cfg.get("hook_font_scale", 1.06))     # >1 = larger/bolder hook
    caption_outline = int(cfg.get("caption_outline_width", 2))  # text stroke px (0 = none)
    # Burned subtitles need a whisper pass on each final clip → off in offline mode.
    subs_on = bool(cfg.get("subtitles_enabled", True)) and not C.offline_mode()
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
    clips = sorted(caps["clips"], key=lambda c: (c.get("score") or 0), reverse=True)
    audio_cache = {}
    manifest = []
    for rank, c in enumerate(clips, 1):
        # final rules gate (defensive — the gauntlet already filtered)
        for field in ("caption", "tiktok_caption", "shorts_title"):
            from captions import banned_hit
            if banned_hit(c.get(field, ""), banned):
                C.fail(f"banned word slipped into {field} for clip {c['moment_id']} — aborting.")

        src_path = C.ROOT / c["source"]
        duration = durations.get(c["source"]) or C.ffprobe_duration(src_path)
        if c["source"] not in audio_cache:
            audio_cache[c["source"]] = C.audio_stream_count(src_path)
        n_audio = audio_cache[c["source"]]
        has_audio = n_audio > 0

        start, end = clip_bounds(c, duration, cmin, cmax, pre, post)
        keeps = dead_air_keeps(src_path, start, end, n_audio)

        # COLD-OPEN: if the payoff peak is separable from the setup (sits well past the
        # start and leaves real clip after it), open on the peak, then cut back to the
        # setup. Otherwise the clip plays chronologically.
        cold_rel = None
        peak = c.get("peak")
        if peak is not None:
            try:
                peakf = float(peak)
            except (TypeError, ValueError):
                peakf = None
            if (peakf is not None and (peakf - start) >= COLD_OPEN_MIN_SETUP
                    and (end - peakf) >= COLD_OPEN_MIN_PAYOFF):
                cold_rel = cold_open_window(src_path, peakf, start, end)
        cold_open = cold_rel is not None
        segments = ([cold_rel] if cold_open else []) + keeps
        segments = cap_segments(segments, cmax)      # total playtime <= clip_max_seconds

        score_i = int(round(c.get("score") or 0))
        name = f"{rank:02d}_{score_i:03d}_{slugify(c['caption'])}.mp4"
        out_path = C.DRAFTS / name
        cap_png = C.DRAFTS / f".cap_{rank:02d}.png"
        render_caption_png(c["caption"], cap_png, emoji=emoji_in_caption,
                           stroke=caption_outline, plate_opacity=hook_plate_opacity,
                           font_scale=hook_font_scale)
        subtitled = False
        if subs_on:
            # Compose to a temp body, THEN transcribe + burn subtitles into out_path so
            # the whisper pass sees the final (cut/reordered) edit.
            body = C.DRAFTS / f".body_{rank:02d}.mp4"
            compose(src_path, start, end, segments, cold_open, cap_png, watermark, body,
                    cfg, has_audio, n_audio)
            subtitled = burn_subtitles(body, out_path, cfg, banned, rank, has_audio)
            body.unlink(missing_ok=True)
        else:
            compose(src_path, start, end, segments, cold_open, cap_png, watermark, out_path,
                    cfg, has_audio, n_audio)
        cap_png.unlink(missing_ok=True)
        C.log(f"  {'cold-open ' if cold_open else ''}{'subtitled ' if subtitled else ''}cut {name}")

        manifest.append({
            "filename": name, "caption": c["caption"], "variant": c.get("variant"),
            "source": c["source"], "source_start": start, "source_end": end,
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
