"""TRACK-mode vertical reframe — face/person-tracked 16:9 → 9:16 crop (OpenShorts-style).

An ALTERNATIVE to the blur_fill layout: instead of shrinking the whole 16:9 frame onto a
blurred background (subject tiny), TRACK crops a 9:16 window that follows the primary subject so
the streamer FILLS the vertical frame like a real clipper's edit. All local, free, CPU-only:

  detection : MediaPipe Face Detection (BlazeFace, Tasks API) — fast; YOLOv8n person detection
              as a fallback when no face is visible (turned away / far). Models are downloaded
              ONCE into models/ (blaze_face_short_range.tflite ~230KB, yolov8n.pt ~6MB).
  stabilize : SmoothedCameraman — a "heavy tripod". It HOLDS the crop still while the subject
              stays inside a center safe zone and only pans (slowly, capped) when they leave it,
              so the crop never jitters on small head movements (mirrors OpenShorts).
  mode      : decide_track() samples the clip — a single clear subject on a landscape source →
              TRACK; group shot / no clear subject / already-vertical → GENERAL (blur_fill).

This module ONLY produces the reframed 1080x1920 video (video-only). cut.py composites the hook,
karaoke subtitles and watermark on top afterwards (a later ffmpeg pass), so every other feature
is untouched. Heavy install (mediapipe/ultralytics/opencv/torch) — if it's missing or a model
can't load we WARN LOUDLY and the caller falls back to blur_fill (never a silent break).
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

W, H = 1080, 1920                       # output canvas (matches cut.py)
TARGET_AR = 9.0 / 16.0                  # crop aspect (portrait)
MODEL_DIR = C.ROOT / "models"
BLAZE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
BLAZE_PATH = MODEL_DIR / "blaze_face_short_range.tflite"
YOLO_PATH = MODEL_DIR / "yolov8n.pt"


def deps_available():
    """True if the TRACK toolchain imports. Warn ONCE (with the pip line) if not, so a missing
    heavy dep degrades to blur_fill loudly instead of crashing the whole cut stage."""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import mediapipe  # noqa: F401
        import ultralytics  # noqa: F401
        return True
    except Exception as e:
        if not getattr(deps_available, "_warned", False):
            C.warn(f"TRACK reframe unavailable ({e.__class__.__name__}: {e}). Install with:\n"
                   "    pip install opencv-python mediapipe ultralytics\n"
                   "  Falling back to blur_fill for all clips.")
            deps_available._warned = True
        return False


# --- detectors (lazy, cached; each returns None on failure and we degrade) ------------------
class Detectors:
    """Holds the MediaPipe face detector and the YOLO person detector, loaded lazily. Either may
    be None (model missing/offline) — as long as ONE works we can center on a subject."""

    def __init__(self, cfg):
        self.cfg = cfg or {}
        self._face = self._load_face()
        self._yolo = self._load_yolo()
        if self._face is None and self._yolo is None:
            C.warn("TRACK: neither a face nor a person detector could load — TRACK disabled "
                   "(all clips use blur_fill). Check the model downloads / network.")

    @property
    def usable(self):
        return self._face is not None or self._yolo is not None

    def _load_face(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            MODEL_DIR.mkdir(exist_ok=True)
            if not BLAZE_PATH.exists():
                C.log("TRACK: downloading BlazeFace model (one-time, ~230KB)…")
                urllib.request.urlretrieve(BLAZE_URL, BLAZE_PATH)
            opts = vision.FaceDetectorOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(BLAZE_PATH)),
                min_detection_confidence=float(self.cfg.get("track_face_conf", 0.5)))
            self._mp = mp
            return vision.FaceDetector.create_from_options(opts)
        except Exception as e:
            C.warn(f"TRACK: MediaPipe face detector unavailable ({e.__class__.__name__}: {e}) "
                   "— using YOLO person detection only.")
            return None

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            MODEL_DIR.mkdir(exist_ok=True)
            if not YOLO_PATH.exists():
                C.log("TRACK: fetching YOLOv8n weights (one-time, ~6MB)…")
                m = YOLO("yolov8n.pt")                     # ultralytics downloads to cwd
                stray = Path("yolov8n.pt")
                if stray.exists():
                    stray.replace(YOLO_PATH)
                    return YOLO(str(YOLO_PATH))
                return m
            return YOLO(str(YOLO_PATH))
        except Exception as e:
            C.warn(f"TRACK: YOLO person detector unavailable ({e.__class__.__name__}: {e}).")
            return None

    def faces(self, frame_bgr):
        """List of (cx, cy, area) for detected faces (BlazeFace), largest-first. [] if none."""
        if self._face is None:
            return []
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        out = []
        for d in (self._face.detect(img).detections or []):
            b = d.bounding_box
            out.append((b.origin_x + b.width / 2.0, b.origin_y + b.height / 2.0,
                        float(b.width * b.height)))
        out.sort(key=lambda t: t[2], reverse=True)
        return out

    def persons(self, frame_bgr):
        """List of (cx, cy, area) for detected persons (YOLO class 0), largest-first. [] if none."""
        if self._yolo is None:
            return []
        conf = float(self.cfg.get("track_person_conf", 0.4))
        r = self._yolo.predict(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
        out = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, float((x2 - x1) * (y2 - y1))))
        out.sort(key=lambda t: t[2], reverse=True)
        return out

    def subject(self, frame_bgr):
        """Primary subject center for CENTERING: the largest face if any (precise on the
        streamer), else the largest person. Returns (cx, cy) or None."""
        f = self.faces(frame_bgr)
        if f:
            return (f[0][0], f[0][1])
        p = self.persons(frame_bgr)
        if p:
            return (p[0][0], p[0][1])
        return None

    def subject_box(self, frame_bgr):
        """Primary subject BOUNDING BOX for FRAMING: (cx, cy, w, h). Prefers the largest PERSON
        (head+torso — the natural unit to frame around); falls back to expanding the largest FACE
        into an approximate head+shoulders box. None if nothing is detected. This box drives the
        crop SIZE (how much padding to leave around the subject) — see track_subject_scale."""
        if self._yolo is not None:
            conf = float(self.cfg.get("track_person_conf", 0.4))
            r = self._yolo.predict(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
            best = None
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                a = (x2 - x1) * (y2 - y1)
                if best is None or a > best[0]:
                    best = (a, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)
            if best:
                return best[1:]
        f = self.faces(frame_bgr)
        if f:
            cx, cy, area = f[0]
            fh = area ** 0.5                          # face box ~square → side length
            # head+shoulders ≈ a face ~3.2x tall / ~2.2x wide, centered a bit below the face.
            return (cx, cy + fh * 0.9, fh * 2.2, fh * 3.2)
        return None


# --- stabilizer (heavy tripod) --------------------------------------------------------------
class SmoothedCameraman:
    """Holds a horizontal crop center that stays STILL while the subject is inside a center safe
    zone, and pans slowly (eased + speed-capped) only when the subject leaves it — so small head
    movements never move the crop (no jitter), but a real walk is followed. Mirrors OpenShorts'
    SmoothedCameraman."""

    def __init__(self, frame_w, crop_w, safe_ratio=0.35, smooth=0.12, max_pan=12.0):
        self.frame_w = float(frame_w)
        self.half = crop_w / 2.0
        self.safe = crop_w * safe_ratio / 2.0          # half-width of the dead zone
        self.smooth = float(smooth)
        self.max_pan = float(max_pan)
        self.cx = frame_w / 2.0                          # start centered

    def update(self, subject_x):
        """Advance one frame toward `subject_x` (or hold if None / inside the safe zone). Returns
        the integer crop LEFT edge (x0), clamped so the crop stays inside the frame."""
        if subject_x is not None and abs(subject_x - self.cx) > self.safe:
            delta = (subject_x - self.cx) * self.smooth
            delta = max(-self.max_pan, min(self.max_pan, delta))   # cap pan speed
            self.cx += delta
        self.cx = max(self.half, min(self.frame_w - self.half, self.cx))
        return int(round(self.cx - self.half))


# --- mode auto-detection --------------------------------------------------------------------
def decide_track(video_path, cfg, detectors, n_samples=14, window=None):
    """Sample the clip and decide TRACK vs GENERAL. TRACK needs a landscape source we can crop
    AND a single clear subject in most sampled frames; a group shot, a subject-less landscape
    (gameplay/screenshare), or an already-vertical source → GENERAL. Returns
    (use_track: bool, reason: str). `window=(start,end)` samples that source time-range via MSEC
    seeks (so we can decide on the SOURCE without first building the content clip); otherwise
    samples evenly across the whole file."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "could not open clip for subject analysis → blur_fill"
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    crop_w = int(round(fh * TARGET_AR))
    if fw <= 0 or fh <= 0 or crop_w >= fw:
        cap.release()
        return False, f"source not landscape enough ({fw}x{fh}) → blur_fill"
    single = multi = none = 0
    for i in range(n_samples):
        frac = (i + 0.5) / n_samples
        if window:
            cap.set(cv2.CAP_PROP_POS_MSEC, (window[0] + frac * (window[1] - window[0])) * 1000.0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, frame = cap.read()
        if not ok:
            continue
        # count PEOPLE for grouping (robust to turned-away faces); fall back to faces.
        people = detectors.persons(frame)
        n = len(people) if people else len(detectors.faces(frame))
        if n == 0:
            none += 1
        elif n == 1:
            single += 1
        else:
            multi += 1
    cap.release()
    seen = single + multi + none
    if seen == 0:
        return False, "no frames sampled → blur_fill"
    sf, mf, nf = single / seen, multi / seen, none / seen
    min_single = float(cfg.get("track_min_single_frac", 0.5))
    max_multi = float(cfg.get("track_max_multi_frac", 0.3))
    max_none = float(cfg.get("track_max_none_frac", 0.5))
    stats = f"single={sf:.0%} multi={mf:.0%} none={nf:.0%}"
    if sf >= min_single and mf <= max_multi and nf <= max_none:
        return True, f"single clear subject ({stats}) → TRACK"
    if mf > max_multi:
        return False, f"group shot ({stats}) → blur_fill"
    if nf > max_none:
        return False, f"no clear subject ({stats}) → blur_fill"
    return False, f"subject not consistent ({stats}) → blur_fill"


def _measure_subject(cap, detectors, n=14):
    """Sample the clip and return the MEDIAN subject bbox (cx, cy, h) — used to size the crop
    ONCE per clip so the zoom stays fixed (no per-frame pulsing). None if no subject sampled.
    Rewinds the capture to the start when done."""
    import cv2
    import numpy as np
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    hs, cxs, cys = [], [], []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, fr = cap.read()
        if not ok:
            continue
        b = detectors.subject_box(fr)
        if b:
            cxs.append(b[0]); cys.append(b[1]); hs.append(b[3])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not hs:
        return None
    return float(np.median(cxs)), float(np.median(cys)), float(np.median(hs))


def _plan_crop(cfg, fw, fh, subj):
    """Compute the FIXED per-clip crop geometry from the subject bbox + track_subject_scale.

    track_subject_scale = the fraction of the OUTPUT height the subject should fill (~0.6 = head
    +shoulders+room; lower = looser, higher = tighter). We size a 9:16 window so the subject fills
    that fraction. If the needed window fits inside the source we crop it directly (edge-to-edge).
    If the subject is so close that filling the frame edge-to-edge would be TIGHTER than the target
    (a webcam close-up), we can't crop 'wider than the source', so we scale a full-height 9:16 crop
    DOWN and letterbox it on a blurred background — giving the looser framing the crop alone can't.

    Returns dict: {letterbox, crop_w, crop_h, y0, fg_h}."""
    target = min(0.95, max(0.30, float(cfg.get("track_subject_scale", 0.60))))
    _, cy, bbox_h = subj
    scale = target * H / max(1.0, bbox_h)              # output px per source px
    crop_w_want = int(round(W / scale))                # 9:16 source window that hits the target
    crop_h_want = int(round(H / scale))
    if crop_h_want <= fh and crop_w_want <= fw:
        # fits: crop the window directly, positioned on the subject with a little headroom.
        crop_h, crop_w = crop_h_want, crop_w_want
        y0 = int(round(min(max(0.0, cy - crop_h * 0.47), fh - crop_h)))
        return {"letterbox": False, "crop_w": crop_w, "crop_h": crop_h, "y0": y0, "fg_h": H}
    # too close to reach the target by cropping → full-height crop, scaled down + blurred bars.
    crop_w = min(fw, crop_w_want)
    crop_h = fh
    fg_h = min(H, int(round(fh * scale)))              # foreground height (< H → letterbox)
    return {"letterbox": True, "crop_w": crop_w, "crop_h": crop_h, "y0": 0, "fg_h": fg_h}


# --- the reframe pass ------------------------------------------------------------------------
def track_reframe(content_mp4, out_mp4, cfg, detectors, fps=30):
    """Read the 16:9 clip, follow the subject with SmoothedCameraman, and write a 1080x1920
    video-only mp4 at CFR `fps`. The crop SIZE is fixed per clip from the subject bbox +
    track_subject_scale (loose/tight zoom); only the horizontal center pans. Detection runs every
    `track_detect_every` frames (reusing the last center between). Returns True, or False to fall back."""
    import cv2
    import numpy as np  # noqa: F401
    cap = cv2.VideoCapture(str(content_mp4))
    if not cap.isOpened():
        C.warn(f"TRACK: could not open {content_mp4} — falling back to blur_fill.")
        return False
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fh <= 0 or fw <= 0:
        cap.release()
        return False
    subj = _measure_subject(cap, detectors)
    if subj is None:                                   # no subject found — nothing to track
        subj = (fw / 2.0, fh / 2.0, fh * TARGET_AR / 0.6)   # neutral: ~full-height framing
    plan = _plan_crop(cfg, fw, fh, subj)
    crop_w, crop_h, y0, letterbox, fg_h = (
        plan["crop_w"], plan["crop_h"], plan["y0"], plan["letterbox"], plan["fg_h"])
    if crop_w >= fw and not letterbox:                 # nothing to crop horizontally → not useful
        cap.release()
        return False
    cam = SmoothedCameraman(
        fw, crop_w,
        safe_ratio=float(cfg.get("track_safe_zone", 0.35)),
        smooth=float(cfg.get("track_smooth", 0.12)),
        max_pan=float(cfg.get("track_max_pan", 12.0)))
    detect_every = max(1, int(cfg.get("track_detect_every", 3)))
    fg_y = (H - fg_h) // 2                              # letterbox: vertical offset of the fg

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(fps), str(out_mp4)],
        stdin=subprocess.PIPE)
    last_subject = None
    fi = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % detect_every == 0:
                s = detectors.subject(frame)
                if s is not None:
                    last_subject = s[0]                  # subject center x (crop size is fixed)
            x0 = cam.update(last_subject)
            region = frame[y0:y0 + crop_h, x0:x0 + crop_w]
            if not letterbox:
                out = cv2.resize(region, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                fg = cv2.resize(region, (W, fg_h), interpolation=cv2.INTER_LINEAR)
                # blurred letterbox bg — blur at LOW res then upscale (visually identical for a
                # blurred bg, ~10x cheaper than gblur at 1080x1920, which was the bottleneck).
                small = cv2.resize(region, (135, 240), interpolation=cv2.INTER_LINEAR)
                small = cv2.GaussianBlur(small, (0, 0), sigmaX=6)
                out = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
                out[fg_y:fg_y + fg_h, 0:W] = fg
            proc.stdin.write(out.tobytes())
            fi += 1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0 or not Path(out_mp4).exists():
        C.warn("TRACK: reframe encode failed — falling back to blur_fill.")
        return False
    frac = subj[2] * fg_h / (crop_h * H)               # subject bbox height as a fraction of output
    C.log(f"    TRACK reframe: {fi} frames in {time.time() - t0:.1f}s "
          f"(crop {crop_w}x{crop_h}{'+letterbox' if letterbox else ''}→{W}x{H}, subject ~{frac:.0%} "
          f"of height, target {float(cfg.get('track_subject_scale', 0.60)):.0%}).")
    return True
