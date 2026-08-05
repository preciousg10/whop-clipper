"""Self-test. Runs offline (CLIPPER_OFFLINE=1) so it needs no Groq key or model.

Always runs the ffmpeg-independent checks:
  T1  intake extracts banned words + required elements into rules.json
  T2  caption gauntlet kills banned candidates; word-boundary is respected
  T3  PIL renders a caption PNG

If ffmpeg is installed it also runs the full pipeline on a synthetic 60s video
(color bars + 3 tone bursts) + a dummy watermark + a brief containing a banned word:
  T4  index->select->captions->cut yields clips + manifest, watermark applied,
      and NO caption contains a banned word.

Exits non-zero if any check FAILS. ffmpeg-dependent checks report SKIP (not fail)
when ffmpeg is absent.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["CLIPPER_OFFLINE"] = "1"      # must be set before importing stages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C                     # noqa: E402

# --- SANDBOX (data-loss guard) -------------------------------------------------
# The self-test writes synthetic fixtures (synthetic.mp4, dummy watermark, brief)
# and deletes campaign files between tests. It must NEVER touch the user's real
# campaign/ data — doing so has wiped multi-hour VODs and cached transcripts.
# So: redirect EVERY data path in `common` into an isolated temp directory BEFORE
# importing any stage module (stages read paths via `C.<NAME>` at call time, and a
# couple capture them at import time — e.g. selectclips.POSTED — so the redirect
# has to happen first). Nothing below this point can reach the real campaign/.
_REAL_CAMPAIGN = C.CAMPAIGN            # remember it only to prove we moved off it
_SANDBOX = Path(tempfile.mkdtemp(prefix="clipper_selftest_"))


def _redirect_common_paths(base):
    """Point all of common's data paths inside `base` (mirrors common.py's layout)."""
    C.ROOT = base
    C.CAMPAIGN = base / "campaign"
    C.FOOTAGE = C.CAMPAIGN / "footage"
    C.ASSETS = C.CAMPAIGN / "assets"
    C.DOCS = C.CAMPAIGN / "docs"
    C.OTHER = C.CAMPAIGN / "other"
    C.TRANSCRIPTS = C.CAMPAIGN / "transcripts"
    C.DRAFTS = base / "drafts"
    C.MEMORY = base / "memory"
    C.STATE_PATH = base / "state.json"
    C.BRIEF_MD = C.CAMPAIGN / "brief.md"
    C.RULES_JSON = C.CAMPAIGN / "rules.json"
    C.KNOWLEDGE_MD = C.CAMPAIGN / "knowledge.md"
    C.CAMPAIGN_MANIFEST = C.CAMPAIGN / "manifest.json"
    C.MOMENTS_JSON = C.CAMPAIGN / "moments.json"
    C.SELECTED_JSON = C.CAMPAIGN / "selected.json"
    C.CAPTIONS_JSON = C.CAMPAIGN / "captions.json"
    C.DRAFTS_MANIFEST = C.DRAFTS / "manifest.json"


_redirect_common_paths(_SANDBOX)
atexit.register(lambda: shutil.rmtree(_SANDBOX, ignore_errors=True))

from PIL import Image, ImageDraw       # noqa: E402
import captions as captions_stage      # noqa: E402
import cut as cut_stage                # noqa: E402

WORK = C.ROOT / "selftest_work"        # under the sandbox now (C.ROOT is redirected)
BANNED_CUSTOM = "unicorn"
results = []


def record(name, ok, detail=""):
    results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def skip(name, detail):
    results.append((name, "SKIP", detail))
    print(f"  [SKIP] {name} — {detail}")


def has_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_watermark(path):
    img = Image.new("RGBA", (240, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, 239, 89], radius=16, fill=(255, 40, 90, 210))
    d.text((24, 30), "WTF", fill="white")
    img.save(path)


def make_brief(path):
    path.write_text(
        "WTF Leagues clipping campaign.\n"
        "Pay: $1.50 per 1K views. Budget $50,000.\n"
        "Platform: TikTok, Reels, Shorts. Vertical 9:16, 10-40 seconds.\n"
        "Watermark PNG mandatory on every clip. Tag #WTFLeagues.\n"
        f"Banned words: {BANNED_CUSTOM}, gamble, casino.\n"
        "Submission: paste the post link in the Whop campaign page.\n",
        encoding="utf-8",
    )


def make_synth_video(path):
    C.run_cmd([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=60",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=60",
        "-filter_complex",
        "[1:a]volume=0.1,volume=enable='between(t,10,11)+between(t,25,26)+"
        "between(t,40,41)':volume=20[a]",
        "-map", "0:v", "-map", "[a]", "-t", "60",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ])


def _assert_sandboxed():
    """Refuse to delete anything unless every path lives under the temp sandbox.
    Belt-and-suspenders: if a future edit forgets to redirect a path, this aborts
    the run loudly instead of wiping the user's real campaign/ again."""
    sb = str(_SANDBOX.resolve())
    for p in (C.CAMPAIGN, C.FOOTAGE, C.ASSETS, C.DOCS, C.OTHER, C.TRANSCRIPTS,
              C.DRAFTS, C.MEMORY, C.STATE_PATH, C.RULES_JSON):
        if not str(Path(p).resolve()).startswith(sb):
            C.fail(f"SAFETY ABORT: self-test path {p} is outside the sandbox {sb}. "
                   "Refusing to touch real data.")


def reset_campaign():
    _assert_sandboxed()
    for p in (C.RULES_JSON, C.CAMPAIGN_MANIFEST, C.BRIEF_MD, C.KNOWLEDGE_MD, C.MOMENTS_JSON,
              C.SELECTED_JSON, C.CAPTIONS_JSON, C.STATE_PATH):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    # Clear directory CONTENTS (don't remove the dirs — OneDrive can lock them).
    for d in (C.FOOTAGE, C.ASSETS, C.DOCS, C.OTHER, C.TRANSCRIPTS, C.DRAFTS):
        if not d.exists():
            continue
        for f in d.iterdir():
            try:
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink()
            except Exception:
                pass
    C.ensure_dirs()


def run_intake(brief, links, campaign):
    import intake
    sys.argv = ["intake.py", "--brief", str(brief), "--campaign", campaign]
    for ln in links:
        sys.argv += ["--link", str(ln)]
    intake.main()


# --- tests ---------------------------------------------------------------------
def t1_intake_rules(footage_path, watermark_path, brief_path):
    reset_campaign()
    run_intake(brief_path, [footage_path, watermark_path], "WTF Leagues Selftest")
    rules = C.load_json(C.RULES_JSON) or {}
    banned = set(rules.get("banned_words", []))
    ok = BANNED_CUSTOM in banned and "gamble" in banned and "casino" in banned
    record("T1 rules.json extracts banned words", ok, f"banned={sorted(banned)}")
    types = {r["type"] for r in rules.get("required_elements", [])}
    record("T1 required elements (watermark + hashtag)",
           "watermark" in types and "hashtag" in types, f"types={sorted(types)}")
    man = C.load_json(C.CAMPAIGN_MANIFEST) or {}
    kinds = {d["kind"] for d in man.get("downloads", [])}
    record("T1 manifest has footage + asset", {"footage", "asset"} <= kinds, f"kinds={sorted(kinds)}")


def t2_gauntlet():
    banned = [BANNED_CUSTOM, "gamble", "stake"]
    cands = ["clean caption 😭", "i love my unicorn 🦄", "gamble responsibly ✌️", "big stake energy"]
    survivors, killed = captions_stage.gauntlet(cands, banned)
    killed_caps = {k["caption"] for k in killed}
    ok_kill = "i love my unicorn 🦄" in killed_caps and "gamble responsibly ✌️" in killed_caps
    ok_survive = all(captions_stage.banned_hit(s, banned) is None for s in survivors) and \
        "clean caption 😭" in survivors
    record("T2 gauntlet kills banned candidates", ok_kill and ok_survive,
           f"survivors={survivors}")
    # word boundary: 'stakeholder' must NOT trip 'stake'
    boundary_ok = captions_stage.banned_hit("stakeholder meeting notes", ["stake"]) is None
    record("T2 banned-word boundary (stakeholder != stake)", boundary_ok)


def t3_caption_png():
    out = WORK / "caption.png"
    cut_stage.render_caption_png("bro lost the whole run in 10 seconds 😭✌️", out)
    ok = out.exists() and out.stat().st_size > 0
    record("T3 PIL renders caption PNG", ok, f"{out.name} {out.stat().st_size if out.exists() else 0}B")


def t5_watermark_selection():
    _assert_sandboxed()
    C.ensure_dirs()
    for p in C.ASSETS.glob("*"):
        p.unlink()
    make_watermark(C.ASSETS / "Safezones TikTok.png")      # reference guide (excluded)
    make_watermark(C.ASSETS / "Example Placement.png")     # reference guide (excluded)
    make_watermark(C.ASSETS / "SPONGEBOB WATERMARK.png")   # the real one
    chosen = cut_stage.find_watermark({})
    record("T5 auto-picks real watermark (ignores safezone/example)",
           chosen.name == "SPONGEBOB WATERMARK.png", chosen.name)
    chosen2 = cut_stage.find_watermark({"watermark_file": "SPONGEBOB WATERMARK.png"})
    record("T5 config watermark_file is honored",
           chosen2.name == "SPONGEBOB WATERMARK.png", chosen2.name)
    for p in C.ASSETS.glob("*"):
        p.unlink()


def t6_routing():
    import download as DL
    ok = (DL.classify_file("clip.mp4") == "footage" and
          DL.classify_file("WATERMARK.PNG") == "asset" and
          DL.classify_file("guide.pdf") == "doc" and
          DL.classify_file("brand.docx") == "doc" and
          DL.classify_file("telemetry.bin") == "other")
    record("T6 file routing (video/image/doc/other by extension)", ok)


def t7_intake_analyst(footage, watermark, brief):
    reset_campaign()
    doc = WORK / "brand_guide.md"
    doc.write_text("# Brand guide\n\nExtra rules that live only in this doc.\n"
                   "Banned words: zebra\n"
                   "Full guide: https://example.com/guide\n"
                   "Watermark must be bottom-right on every clip.\n", encoding="utf-8")
    other = WORK / "telemetry.bin"
    other.write_bytes(b"\x00\x01\x02\x03")
    run_intake(brief, [footage, watermark, doc, other], "Analyst Test")

    rules = C.load_json(C.RULES_JSON) or {}
    banned = set(rules.get("banned_words", []))
    record("T7 doc content read into rules (zebra came from the doc)", "zebra" in banned, sorted(banned))
    kn = C.KNOWLEDGE_MD.read_text(encoding="utf-8") if C.KNOWLEDGE_MD.exists() else ""
    record("T7 knowledge.md written listing the doc", "brand_guide.md" in kn)
    man = C.load_json(C.CAMPAIGN_MANIFEST) or {}
    kinds = {d["kind"] for d in man.get("downloads", [])}
    record("T7 doc + other kinds routed correctly", {"doc", "other"} <= kinds, sorted(kinds))
    unused = any(u.startswith("UNUSED") for d in man.get("downloads", []) for u in d.get("usage", []))
    record("T7 unknown file flagged UNUSED (nothing silently ignored)", unused)
    record("T7 url harvested from doc", any("example.com" in u for u in man.get("harvested_urls", [])))
    amb = rules.get("ambiguities", [])
    record("T7 ambiguities flag the unused file + LLM-skipped",
           any("telemetry.bin" in a for a in amb) and any("LLM extraction skipped" in a for a in amb))


def t4_full_pipeline(synth, watermark_path, brief_path):
    reset_campaign()
    run_intake(brief_path, [synth, watermark_path], "WTF Leagues Selftest")
    import run as run_mod
    sys.argv = ["run.py", "--force", "--no-cleanup"]
    run_mod.main()

    drafts = list(C.DRAFTS.glob("*.mp4"))
    record("T4 clips produced", len(drafts) >= 1, f"{len(drafts)} mp4(s)")
    man = C.load_json(C.DRAFTS_MANIFEST) or {}
    record("T4 drafts/manifest.json present", bool(man.get("clips")),
           f"{len(man.get('clips', []))} entries")
    rules = C.load_json(C.RULES_JSON) or {}
    banned = rules.get("banned_words", [])
    clean = all(captions_stage.banned_hit(c["caption"], banned) is None and
                captions_stage.banned_hit(c["tiktok_caption"], banned) is None
                for c in man.get("clips", []))
    record("T4 no banned words in any caption", clean)
    record("T4 watermark asset resolved", cut_stage.find_watermark() is not None)
    record("T4 clips are non-empty", all(p.stat().st_size > 0 for p in drafts))


def main():
    print("== clipper self-test (offline) ==")
    print(f"   sandbox: {_SANDBOX}  (real campaign/ is never touched)")
    WORK.mkdir(parents=True, exist_ok=True)
    watermark = WORK / "watermark.png"
    brief = WORK / "brief.txt"
    make_watermark(watermark)
    make_brief(brief)

    # Real synthetic mp4 (color bars + 3 tone bursts) for ALL paths when ffmpeg is
    # present. Only fall back to a placeholder when ffmpeg is absent (T4 is skipped
    # then anyway, and intake skips ffprobe with no ffprobe on PATH).
    synth = None
    if has_ffmpeg():
        synth = WORK / "synthetic.mp4"
        make_synth_video(synth)
        footage_for_t1 = synth
    else:
        footage_for_t1 = WORK / "placeholder.mp4"
        footage_for_t1.write_bytes(b"\x00")

    t1_intake_rules(footage_for_t1, watermark, brief)
    t2_gauntlet()
    t3_caption_png()
    t5_watermark_selection()
    t6_routing()
    t7_intake_analyst(footage_for_t1, watermark, brief)

    if synth is not None:
        t4_full_pipeline(synth, watermark, brief)
    else:
        for n in ("T4 clips produced", "T4 drafts/manifest.json present",
                  "T4 no banned words in any caption", "T4 watermark applied"):
            skip(n, "ffmpeg not installed — run `python scripts/selftest.py` on a box with ffmpeg")

    print("\n== summary ==")
    for name, status, detail in results:
        print(f"  {status:4}  {name}")
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\n{'FAILED' if failed else 'OK'}: "
          f"{sum(r[1]=='PASS' for r in results)} passed, "
          f"{len(failed)} failed, {sum(r[1]=='SKIP' for r in results)} skipped.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
