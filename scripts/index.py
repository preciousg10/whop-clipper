"""INDEX stage — build a moment index of the ENTIRE source.

Two moment sources, per instructions.md:
  - transcript (faster-whisper 'base', CPU, word timestamps)
  - audio-spike detection (RMS over 1s windows; windows >2 std above the mean are
    screams/laughs/chaos = candidate moments)

Long-VOD handling: audio is processed in time CHUNKS. Each chunk is transcribed +
RMS-scanned, then checkpointed to disk and to state.json, so a crash mid-VOD resumes
with zero redone work. Progress prints as "transcribed 47/180 min".

Offline/degraded mode (CLIPPER_OFFLINE=1) skips whisper (spikes only) so the pipeline
is testable without downloading a model.
"""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CHUNK_SEC = 300          # 5-minute chunks (checkpoint granularity)
SR = 16000               # analysis sample rate
WIN_SEC = 1.0            # RMS window
SPIKE_STD = 2.0          # flag windows > mean + SPIKE_STD*std
CLEANUP_TMP = True
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".flv", ".m2ts"}


# --- footage selection (prefer safe-margin) ------------------------------------
def choose_footage(manifest):
    footage = [d for d in manifest.get("downloads", []) if d.get("kind") == "footage"]
    safe = [d for d in footage if d.get("safe_margin")]
    if safe:
        C.log(f"safe-margin footage present — indexing {len(safe)} safe version(s), "
              f"skipping {len(footage) - len(safe)} non-safe.")
        return safe
    return footage


def discover_footage(manifest):
    """Manifest footage + any video files sitting in campaign/footage/ that the
    manifest doesn't know about. A large VOD dropped into footage/ AFTER intake ran
    (so it's absent from manifest.json) must still be indexed, not silently skipped —
    that exact gap left a 35 GB stream untranscribed. Discovered files are logged
    loudly and indexed as non-safe sources."""
    chosen = choose_footage(manifest)
    # Every footage basename the manifest knows (safe AND non-safe), so we never
    # re-index a non-safe variant that choose_footage intentionally dropped.
    known = {os.path.basename(d.get("path", "")).lower()
             for d in manifest.get("downloads", []) if d.get("kind") == "footage"}
    extra = []
    for p in sorted(C.FOOTAGE.glob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        if p.name.lower() in known:
            continue
        C.warn(f"footage on disk but NOT in manifest.json — indexing it anyway: {p.name}")
        extra.append({"path": str(p.relative_to(C.ROOT)), "safe_margin": False})
    return chosen + extra


# --- transcription -------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            C.fail("faster-whisper not installed. pip install -r requirements.txt "
                   "(or run offline with CLIPPER_OFFLINE=1 to skip transcription).")
        C.log("loading faster-whisper 'base' (CPU, int8) — first load downloads the model…")
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def _extract_full_audio(source, wav_path):
    """Extract the ENTIRE first audio stream to a 16kHz mono PCM wav in ONE pass.
    Audio-only is tiny (~0.6 GB for a 5h VOD) regardless of the video size, so we never
    seek or decode the multi-GB video per chunk. `-vn` skips video decode entirely;
    `0:a:0?` picks the first audio stream (VODs often carry several) and won't error if
    a source has none. Writes to a .tmp then renames, so the final wav exists only once
    fully extracted — a killed extraction re-runs cleanly on resume."""
    tmp = wav_path.with_suffix(".wav.tmp")
    n = max(1, C.audio_stream_count(source))
    # -f wav is REQUIRED: the .tmp extension gives ffmpeg no format to infer from.
    if n >= 2:
        # MERGE every audio track (normalize=0 = sum, so a track isn't halved and a
        # silent track contributes nothing) — commentary/action lives on a 2nd track on
        # this VOD, and first-track-only left ~20s of clips silent.
        labels = "".join(f"[0:a:{k}]" for k in range(n))
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(source),
               "-filter_complex", f"{labels}amix=inputs={n}:normalize=0[a]",
               "-map", "[a]", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
               "-f", "wav", str(tmp)]
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn",
               "-map", "0:a:0?", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
               "-f", "wav", str(tmp)]
    C.run_cmd(cmd)
    os.replace(tmp, wav_path)


def _read_wav_slice(wav_path, start_sec, dur_sec):
    """Read only [start_sec, start_sec+dur_sec) from the wav via a frame seek — never
    loads the whole (up to ~0.6 GB) audio file into memory. ~9.6 MB per 5-min chunk."""
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        w.setpos(min(int(start_sec * sr), w.getnframes()))
        frames = w.readframes(int(dur_sec * sr))
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32)


def _rms_per_second(audio):
    win = int(SR * WIN_SEC)
    n = len(audio) // win
    if n == 0:
        return np.array([])
    a = audio[:n * win].reshape(n, win)
    return np.sqrt((a ** 2).mean(axis=1) + 1e-9)


def _transcribe_audio(audio, offset):
    """Transcribe an in-memory 16kHz mono chunk. faster-whisper accepts a float32 numpy
    array in [-1, 1], so we normalize the int16-scaled samples here (we already hold the
    samples — no need to re-open a file).

    Returns (segments, words): segment-level text (as before) AND per-word timings
    {word,start,end}. Word timings power the karaoke-style burned subtitles in cut.py,
    which maps them through the cold-open reorder onto the FINAL clip timeline — so we
    persist them at INDEX time (one whisper pass) instead of re-transcribing every clip.
    Both are offset to ABSOLUTE source seconds (chunk offset + segment/word time)."""
    model = _get_model()
    segments, _info = model.transcribe(audio / 32768.0, word_timestamps=True)
    segs, words = [], []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            segs.append({"start": round(offset + seg.start, 2),
                         "end": round(offset + seg.end, 2),
                         "text": text})
        for w in (seg.words or []):
            wt = (w.word or "").strip()
            if not wt:
                continue
            words.append({"word": wt,
                          "start": round(offset + float(w.start), 3),
                          "end": round(offset + float(w.end), 3)})
    return segs, words


# --- moment construction -------------------------------------------------------
def _spike_moments(source, rms):
    if rms.size == 0:
        return []
    mean, std = float(rms.mean()), float(rms.std())
    if std <= 0:
        return []
    thr = mean + SPIKE_STD * std
    hot = rms > thr
    moments, i, n = [], 0, len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]:
                j += 1
            seg = rms[i:j]
            z = float((seg.max() - mean) / std)
            # peak = absolute second of the loudest window in the spike (the punchline
            # / crash / scream). Cold-open restructuring opens the clip here.
            peak = float(i + int(seg.argmax()))
            moments.append({"source": source, "start": float(i), "end": float(j),
                            "type": "audio_spike", "intensity": round(z, 2),
                            "peak": peak, "text": ""})
            i = j
        else:
            i += 1
    return moments


def _energy_peak(rms, start, end):
    """Absolute second of the loudest RMS window inside [start, end) — the most-
    emphasized point of a speech moment. `rms` is the per-second RMS array. Returns None
    if there's no RMS (e.g. offline mode never builds a transcript anyway)."""
    if rms is None or len(rms) == 0:
        return None
    lo = max(0, min(int(np.floor(start)), len(rms) - 1))
    hi = max(lo + 1, min(int(np.ceil(end)), len(rms)))
    window = rms[lo:hi]
    if len(window) == 0:
        return None
    return float(lo + int(np.argmax(window)))


def _speech_moments(source, transcript, rms=None):
    out = []
    for seg in transcript:
        # Give EVERY speech moment an energy peak (loudest second in its span) so the
        # cut stage's cold-open can fire on speech clips too — and so an over-length
        # merged moment is centered on its peak instead of clamped from the start.
        peak = _energy_peak(rms, seg["start"], seg["end"])
        out.append({"source": source, "start": seg["start"], "end": seg["end"],
                    "type": "speech", "intensity": round(len(seg["text"]) / 10.0, 2),
                    "peak": peak, "text": seg["text"]})
    return out


# --- per-source indexing (chunked + resumable) ---------------------------------
def _partial_paths(name):
    return (C.TRANSCRIPTS / f"{name}.transcript.json",
            C.TRANSCRIPTS / f"{name}.rms.json",
            C.TRANSCRIPTS / f"{name}.words.json")


def index_source(entry, state, do_transcribe):
    source = C.ROOT / entry["path"]
    name = os.path.splitext(os.path.basename(entry["path"]))[0]
    duration = entry.get("duration_sec") or C.ffprobe_duration(source)
    chunks_total = max(1, int(np.ceil(duration / CHUNK_SEC)))

    idx_state = state["stages"].setdefault("index", {"done": False, "sources": {}})
    ss = idx_state["sources"].setdefault(name, {"chunks_total": chunks_total, "chunks_done": 0})
    ss["chunks_total"] = chunks_total

    tr_path, rms_path, words_path = _partial_paths(name)
    # Only REUSE the on-disk partials when we're genuinely resuming (chunks_done > 0). On a
    # fresh start — a brand-new source or a --force re-run (which pops the stage so chunks_done
    # resets to 0) — the old partials are stale: appending to them DOUBLES the transcript/words
    # (each re-run walks every chunk again). Start empty and let the per-chunk save overwrite.
    resuming = ss["chunks_done"] > 0
    transcript = (C.load_json(tr_path, default=[]) or []) if resuming else []
    rms_vals = (C.load_json(rms_path, default=[]) or []) if resuming else []
    words = (C.load_json(words_path, default=[]) or []) if resuming else []

    # Extract the whole audio track ONCE to a small wav (resumes reuse it — it's only
    # deleted after every chunk is done), then walk it in CHUNK_SEC slices. The multi-GB
    # video is opened a single time for audio, never per chunk.
    audio_wav = C.TRANSCRIPTS / f"{name}.audio.wav"
    if ss["chunks_done"] < chunks_total and not audio_wav.exists():
        C.log(f"  {name}: extracting audio track → 16kHz mono wav (one pass over the source)…")
        _extract_full_audio(source, audio_wav)

    for ci in range(ss["chunks_done"], chunks_total):
        start = ci * CHUNK_SEC
        dur = min(CHUNK_SEC, duration - start)
        audio = _read_wav_slice(audio_wav, start, dur)
        rms_vals.extend([round(float(v), 2) for v in _rms_per_second(audio)])
        if do_transcribe:
            segs, chunk_words = _transcribe_audio(audio, start)
            transcript.extend(segs)
            words.extend(chunk_words)
        # checkpoint after each chunk
        C.save_json(tr_path, transcript)
        C.save_json(rms_path, rms_vals)
        C.save_json(words_path, words)
        ss["chunks_done"] = ci + 1
        C.save_state(state)
        done_min = int((ci + 1) * CHUNK_SEC / 60)
        total_min = int(np.ceil(duration / 60))
        C.log(f"  {name}: {'transcribed' if do_transcribe else 'scanned'} "
              f"{min(done_min, total_min)}/{total_min} min")
    if CLEANUP_TMP and audio_wav.exists():
        audio_wav.unlink()

    rms = np.array(rms_vals, dtype=np.float32)
    moments = (_spike_moments(entry["path"], rms)
               + _speech_moments(entry["path"], transcript, rms))
    return {"source": entry["path"], "duration_sec": round(duration, 2),
            "safe_margin": entry.get("safe_margin", False),
            "transcript": transcript, "words": words, "moments": moments}


def run(state):
    C.ensure_dirs()
    manifest = C.load_json(C.CAMPAIGN_MANIFEST)
    if not manifest:
        C.fail("campaign/manifest.json missing — run intake.py first.")
    footage = discover_footage(manifest)
    if not footage:
        raise C.NothingUsable("no footage to index for this campaign.")

    do_transcribe = not C.offline_mode()
    if not do_transcribe:
        C.warn("offline mode — skipping whisper transcription (audio spikes only).")

    # ROBUSTNESS (Unit 2a): one bad file must not fail the whole stage. We SKIP a footage file
    # that can't be indexed — no audio stream, or corrupt/unreadable (transcript + audio spikes
    # both come from audio, so a video-only / broken file has nothing to index) — and keep going.
    # The stage COMPLETES as long as at least ONE file indexed; it hard-fails only if ZERO did.
    sources, all_moments, skipped = [], [], []
    for entry in footage:
        src_path = C.ROOT / entry["path"]
        base = os.path.basename(entry["path"])
        size_gb = (src_path.stat().st_size / 1e9) if src_path.exists() else 0.0

        if not src_path.exists():
            C.warn(f"  SKIP {base}: file missing on disk — not indexable.")
            skipped.append((base, "missing on disk"))
            continue
        # Cheap probe FIRST: no readable audio stream = no-audio or corrupt -> skip cleanly.
        try:
            n_audio = C.audio_stream_count(src_path)
        except SystemExit:
            n_audio = None                 # ffprobe unavailable -> unknown; let index_source try
        if n_audio == 0:
            C.warn(f"  SKIP {base}: no readable audio stream (no-audio or corrupt) — "
                   f"nothing to transcribe or detect. Skipping, continuing with the rest.")
            skipped.append((base, "no audio / unreadable"))
            continue

        try:
            duration = entry.get("duration_sec") or C.ffprobe_duration(src_path)
            dur_txt = f"{duration / 60:.1f} min"
        except SystemExit:
            C.warn(f"  SKIP {base}: duration unreadable (corrupt) — skipping.")
            skipped.append((base, "duration unreadable"))
            continue
        C.log(f"indexing {base}  ({size_gb:.2f} GB, {dur_txt}) …")
        try:
            src = index_source(entry, state, do_transcribe)
            sources.append(src)
            all_moments.extend(src["moments"])
            C.log(f"  OK {base}: {len(src['moments'])} moment(s), "
                  f"{len(src['transcript'])} transcript segment(s).")
        except SystemExit as e:            # C.fail() inside a source: skip it, keep going
            C.warn(f"  SKIP {base}: indexing error ({e.code}) — skipping this file.")
            skipped.append((base, f"index error {e.code}"))
        except Exception as e:
            import traceback
            C.warn(f"  SKIP {base}: {e.__class__.__name__}: {e} — skipping this file.")
            traceback.print_exc()
            skipped.append((base, f"{e.__class__.__name__}: {e}"))

    # assign global ids, newest-intensity first is handled later by select
    for i, m in enumerate(all_moments):
        m["id"] = f"m{i:04d}"

    C.save_json(C.MOMENTS_JSON, {"created_at": C.now_iso(),
                                 "sources": sources, "moments": all_moments})
    if not sources:
        # ZERO usable sources — nothing to clip. NothingUsable so the orchestrator can
        # auto-advance to the next campaign (or dead-end loud without --auto-advance).
        detail = "; ".join(f"{n} ({e})" for n, e in skipped) or "no footage files"
        raise C.NothingUsable(f"index produced ZERO usable sources — every footage file was "
                              f"skipped (no audio / corrupt / error): {detail}.")
    C.mark_stage(state, "index", moments=len(all_moments), sources=len(sources),
                 skipped=len(skipped))
    if skipped:
        detail = ", ".join(f"{n} ({e})" for n, e in skipped)
        C.warn(f"index completed with {len(skipped)} file(s) SKIPPED as not indexable "
               f"(indexed {len(sources)} OK): {detail}")
    C.log(f"index done: {len(all_moments)} moments across {len(sources)} source(s).")


if __name__ == "__main__":
    try:
        run(C.load_state())
    except C.NothingUsable as e:
        C.fail(str(e))            # standalone: still fail loud (only run.py --auto-advance moves on)
