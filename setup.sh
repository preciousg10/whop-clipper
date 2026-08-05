#!/usr/bin/env bash
# Clipper setup. Requires Python 3.12 (3.11 works) and ffmpeg.
set -euo pipefail

echo "== clipper setup =="

# 1. ffmpeg / ffprobe must exist (used for probing, audio decode, cutting).
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg/ffprobe not found on PATH."
  echo "  macOS:  brew install ffmpeg"
  echo "  Debian: sudo apt-get install -y ffmpeg"
  echo "  Windows: winget install Gyan.FFmpeg   (or scoop install ffmpeg)"
  exit 1
fi

# 2. Python venv + deps.
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup done. Next:"
echo "  export GROQ_API_KEY=...            # required for real runs"
echo "  python scripts/intake.py --brief brief.txt --links links.txt"
echo "  python scripts/run.py"
echo
echo "Offline self-test (no Groq / no model download needed):"
echo "  python scripts/selftest.py"
