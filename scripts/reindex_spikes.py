"""Recompute audio-spike moments on the MERGED audio, REUSING the cached whisper
transcript (no 5-hour re-transcription).

Why this exists: the original moments.json spikes were detected on the first audio
track only — but this VOD carries the race action/commentary on a SECOND track, so the
"loudest" moments were crowd hype / countdowns, not the action. This rescans RMS on all
tracks merged (index._extract_full_audio already amixes them), keeps the transcript as
is, and rewrites moments.json. Then re-run select -> captions -> cut.

    python scripts/reindex_spikes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import index as IX


def _rms_for_source(src_abs, name, duration):
    """Extract merged audio once, then RMS-per-second in chunks (never loads the full
    ~0.6 GB wav into memory). Returns the per-second RMS list."""
    audio_wav = C.TRANSCRIPTS / f"{name}.audio.wav"
    if not audio_wav.exists():
        C.log(f"  {name}: extracting MERGED audio (one pass)…")
        IX._extract_full_audio(src_abs, audio_wav)
    rms_vals = []
    chunks = max(1, int(np.ceil(duration / IX.CHUNK_SEC)))
    for ci in range(chunks):
        start = ci * IX.CHUNK_SEC
        dur = min(IX.CHUNK_SEC, duration - start)
        audio = IX._read_wav_slice(audio_wav, start, dur)
        rms_vals.extend([round(float(v), 2) for v in IX._rms_per_second(audio)])
    if IX.CLEANUP_TMP and audio_wav.exists():
        audio_wav.unlink()
    return rms_vals


def main():
    C.ensure_dirs()
    data = C.load_json(C.MOMENTS_JSON)
    if not data:
        C.fail("campaign/moments.json missing — nothing to reindex.")
    C.save_json(C.MOMENTS_JSON.with_suffix(".json.bak"), data)   # safety backup

    all_moments, out_sources = [], []
    for s in data.get("sources", []):
        rel = s["source"]
        src_abs = C.ROOT / rel
        name = os.path.splitext(os.path.basename(rel))[0]
        duration = s.get("duration_sec") or C.ffprobe_duration(src_abs)
        n_audio = C.audio_stream_count(src_abs)
        C.log(f"reindex: {name}  ({n_audio} audio track(s){' — MERGING' if n_audio >= 2 else ''}, "
              f"{duration / 60:.1f} min)")

        rms = _rms_for_source(src_abs, name, duration)
        tr_path, rms_path = IX._partial_paths(name)
        C.save_json(rms_path, rms)                               # overwrite spike source
        transcript = C.load_json(tr_path, default=[]) or []       # REUSE cached transcript

        rms_arr = np.array(rms, dtype=np.float32)
        spikes = IX._spike_moments(rel, rms_arr)
        speech = IX._speech_moments(rel, transcript)
        C.log(f"  {name}: {len(spikes)} spike + {len(speech)} speech moment(s) "
              f"(transcript reused, {len(transcript)} segs).")
        out_sources.append({"source": rel, "duration_sec": round(duration, 2),
                            "safe_margin": s.get("safe_margin", False),
                            "transcript": transcript, "moments": spikes + speech})
        all_moments.extend(spikes + speech)

    for i, m in enumerate(all_moments):
        m["id"] = f"m{i:04d}"
    C.save_json(C.MOMENTS_JSON, {"created_at": C.now_iso(),
                                 "sources": out_sources, "moments": all_moments})
    C.log(f"reindex done: {len(all_moments)} moments across {len(out_sources)} source(s). "
          f"(old moments.json backed up to moments.json.bak)")


if __name__ == "__main__":
    main()
