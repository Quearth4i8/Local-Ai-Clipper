"""Subject-aware reframing: turn a wide clip into 9:16 / 4:5 / 1:1.

A centre crop is wrong most of the time - the person talking is rarely in the
middle of the frame. So we sample the clip, find the subject in each sampled
frame, smooth that into a camera path, and hand FFmpeg a time-varying crop.

Everything is local. Face detection uses the Haar cascades that ship inside the
OpenCV wheel, so there is no model download. If OpenCV is missing entirely the
module degrades to a static centre crop rather than failing.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .audio_mix import build_music_graph

log = logging.getLogger("clipfinder.reframe")

# name -> (width, height, label, platforms)
FORMATS: Dict[str, Tuple[int, int, str, str]] = {
    "9:16": (1080, 1920, "Vertical", "TikTok · Reels · Shorts"),
    "4:5":  (1080, 1350, "Portrait", "Instagram feed"),
    "1:1":  (1080, 1080, "Square", "Facebook · LinkedIn"),
    "16:9": (1920, 1080, "Original", "YouTube · X"),
}
DEFAULT_FORMAT = "9:16"

# libavutil's expression parser carries a fixed stack (STACK_SIZE 100 in
# eval.c). Measured against this FFmpeg build: a crop expression accepts at most
# 98 operands, whether they are nested if()s or a flat sum - beyond that the
# filter dies with "Failed to configure input pad" / EINVAL(-22). A fast-cut
# vlog easily produces more camera moves than that, so the path is capped well
# under the limit. 80 moves is a framing change every 0.75s in a 60s clip,
# which is far more than anything reads as motion on screen.
MAX_PATH_SEGMENTS = 80


def format_spec(name: str) -> Tuple[int, int, str, str]:
    return FORMATS.get(name, FORMATS[DEFAULT_FORMAT])


def safe_tag(name: str) -> str:
    return name.replace(":", "x")


def subject_tracking_available() -> Dict[str, Any]:
    """Whether we can actually follow the subject, and why not if we cannot."""
    try:
        import cv2
    except ImportError:
        return {"ok": False, "reason": "OpenCV is not installed - exports will use a "
                                       "centre crop. Fix: pip install "
                                       "\"opencv-python-headless<5\""}
    if not _cascades():
        return {"ok": False, "version": cv2.__version__,
                "reason": f"OpenCV {cv2.__version__} ships no face cascades "
                          "(5.x removed them) - exports will use a centre crop. "
                          "Fix: pip install \"opencv-python-headless<5\""}
    return {"ok": True, "version": cv2.__version__, "reason": None}


# ---------------------------------------------------------------------------
# Subject tracking
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    t: float                  # seconds from clip start
    cx: Optional[float]       # subject centre, 0..1 of source width (None = unknown)
    cy: Optional[float]
    weight: float = 0.0       # detection confidence-ish (face area)
    cut: bool = False         # a shot change happens at this sample


@dataclass
class CameraPath:
    """Piecewise-linear crop centre over time, in 0..1 of source width."""
    segments: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # (t0, t1, cx_at_t0, cx_at_t1)
    detections: int = 0
    frames: int = 0
    cuts: int = 0

    @property
    def coverage(self) -> float:
        return (self.detections / self.frames) if self.frames else 0.0


def _cascades():
    """Load the Haar cascades bundled with the OpenCV wheel.

    OpenCV 5.0 dropped CascadeClassifier and ships no cascade XMLs, so this
    returns nothing there and the caller falls back to a centre crop. Pin
    opencv-python-headless<5 to keep subject tracking (requirements do).
    """
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):
        log.warning("OpenCV %s has no CascadeClassifier - install "
                    "opencv-python-headless<5 for subject tracking", cv2.__version__)
        return []
    try:
        base = Path(cv2.data.haarcascades)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml",
                 "haarcascade_profileface.xml"):
        p = base / name
        if p.exists():
            c = cv2.CascadeClassifier(str(p))
            if not c.empty():
                out.append(c)
    if not out:
        log.warning("No Haar cascades found in the OpenCV wheel")
    return out


def sample_subject(ffmpeg_bin: str, src: str, start: float, end: float,
                   sample_fps: float = 3.0, work_width: int = 480,
                   on_log=None) -> List[Sample]:
    """Extract low-res frames for [start, end] and locate the subject in each."""
    duration = max(0.1, end - start)
    tmp = Path(tempfile.mkdtemp(prefix="reframe_"))
    try:
        args = [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}", "-i", str(src), "-t", f"{duration:.3f}",
            "-vf", f"fps={sample_fps},scale={work_width}:-2",
            "-q:v", "4", str(tmp / "f_%05d.jpg"),
        ]
        creation = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        proc = subprocess.run(args, capture_output=True, creationflags=creation)
        frames = sorted(tmp.glob("f_*.jpg"))
        if proc.returncode != 0 or not frames:
            log.warning("reframe: frame extraction produced nothing")
            return []

        try:
            import cv2
            import numpy as np
        except ImportError:
            if on_log:
                on_log("OpenCV not installed - reframing falls back to a centre crop. "
                       "Install it with: pip install opencv-python-headless")
            return []

        cascades = _cascades()
        samples: List[Sample] = []
        diffs: List[float] = []
        prev_gray = None

        for i, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Frame-to-frame difference; the cut threshold is decided later,
            # because it has to adapt to how much this video moves.
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diffs.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
            else:
                diffs.append(0.0)
            prev_gray = gray
            cut = False

            # --- faces ---
            best = None
            for c in cascades:
                try:
                    found = c.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                               minSize=(max(24, w // 22),) * 2)
                except Exception:  # noqa: BLE001
                    continue
                for (x, y, fw, fh) in found:
                    area = fw * fh
                    if best is None or area > best[0]:
                        best = (area, x + fw / 2.0, y + fh / 2.0)

            t = i / sample_fps
            if best:
                samples.append(Sample(t=t, cx=best[1] / w, cy=best[2] / h,
                                      weight=best[0] / float(w * h), cut=cut))
            else:
                samples.append(Sample(t=t, cx=None, cy=None, cut=cut))

        _mark_cuts(samples, diffs, on_log=on_log)
        return samples
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _mark_cuts(samples: List[Sample], diffs: List[float], on_log=None) -> None:
    """Decide which samples are real shot changes.

    A fixed absolute threshold does not work. Frames sampled a third of a second
    apart in handheld footage differ enormously without any cut happening, so a
    fixed threshold flags almost every sample - measured 175 "cuts" in 68s on a
    vlog, which made the crop jitter at 3 Hz and read as stutter.

    Instead the threshold adapts: a cut has to stand well clear of how much this
    particular clip normally changes between samples. If the result is still
    implausible, the footage is simply too busy to segment and we report no cuts
    at all, letting the smoothing carry the whole clip.
    """
    if not samples or len(diffs) != len(samples):
        return
    body = sorted(d for d in diffs[1:] if d > 0)
    if not body:
        return
    median = body[len(body) // 2]
    threshold = max(20.0, median * 3.0)

    flagged = [i for i, d in enumerate(diffs) if i > 0 and d > threshold]
    ratio = len(flagged) / max(1, len(samples))
    if ratio > 0.35:
        # More than a third "cut" means we are measuring motion, not edits.
        if on_log:
            on_log(f"Reframe: footage too busy to detect cuts reliably "
                   f"({ratio * 100:.0f}% of samples flagged) - relying on smoothing")
        return
    for i in flagged:
        samples[i].cut = True


def build_path(samples: Sequence[Sample], duration: float,
               smooth: float = 0.35, move_threshold: float = 0.035,
               segment_seconds: float = 0.7) -> CameraPath:
    """Turn noisy per-frame detections into a smooth, cut-aware camera path.

    - Gaps (no face) hold the last known position instead of snapping to centre.
    - The smoothing resets on a shot change, so the crop jumps on a cut the way
      a human editor would, instead of sliding across the frame.
    - A shot where the subject barely moves collapses to a single static crop:
      a locked-off frame looks intentional, a drifting one looks broken.
    """
    path = CameraPath(frames=len(samples))
    if not samples:
        path.segments = [(0.0, max(0.1, duration), 0.5, 0.5)]
        return path

    known = [s for s in samples if s.cx is not None]
    path.detections = len(known)
    path.cuts = sum(1 for s in samples if s.cut)
    xs_known = sorted(s.cx for s in known)
    median_x = xs_known[len(xs_known) // 2] if xs_known else 0.5

    # When the subject is only visible in a minority of frames - action montages,
    # wide shots, faces turned away - any path we build is mostly guesswork, and
    # a guessed camera that moves looks far worse than one that holds still.
    if path.coverage < 0.35:
        path.segments = [(0.0, max(0.1, duration) + 0.5, median_x, median_x)]
        return path

    # 1. fill gaps + exponential smoothing, reset at cuts
    filled: List[Tuple[float, float]] = []
    cur = known[0].cx if known else 0.5
    for s in samples:
        if s.cut and s.cx is not None:
            cur = s.cx          # new shot, and we can see the subject: jump to it
        elif s.cx is not None:
            cur = (1.0 - smooth) * cur + smooth * s.cx
        # No detection: HOLD. Snapping to a global average mid-clip is a visible
        # lurch to a position nothing in the frame justifies.
        filled.append((s.t, cur))

    # 2. split at cuts
    shots: List[List[Tuple[float, float]]] = [[]]
    for s, pt in zip(samples, filled):
        if s.cut and shots[-1]:
            shots.append([])
        shots[-1].append(pt)
    shots = [sh for sh in shots if sh]

    # 3. per shot: static if nearly still, otherwise piecewise linear
    segs: List[Tuple[float, float, float, float]] = []
    for sh in shots:
        t0 = sh[0][0]
        t1 = sh[-1][0]
        xs = [x for _, x in sh]
        if t1 <= t0:
            t1 = t0 + 0.2
        if (max(xs) - min(xs)) < move_threshold or len(sh) < 3:
            mid = sorted(xs)[len(xs) // 2]
            segs.append((t0, t1, mid, mid))
            continue
        step = max(2, int(round(segment_seconds * len(sh) / max(0.1, t1 - t0))))
        i = 0
        while i < len(sh) - 1:
            j = min(i + step, len(sh) - 1)
            segs.append((sh[i][0], sh[j][0], sh[i][1], sh[j][1]))
            i = j

    # 4. close gaps so the expression always has a value
    segs.sort()
    fixed: List[Tuple[float, float, float, float]] = []
    for a, b, x0, x1 in segs:
        if fixed and a > fixed[-1][1]:
            pa, pb, px0, px1 = fixed[-1]
            fixed[-1] = (pa, a, px0, px1)
        fixed.append((a, b, x0, x1))
    if fixed:
        first = fixed[0]
        if first[0] > 0:
            fixed[0] = (0.0, first[1], first[2], first[3])
        last = fixed[-1]
        if last[1] < duration:
            fixed[-1] = (last[0], duration + 0.5, last[2], last[3])
    else:
        fixed = [(0.0, duration + 0.5, fallback, fallback)]

    path.segments = _merge_segments(fixed)
    return path


def _merge_segments(segs: List[Tuple[float, float, float, float]],
                    same: float = 0.006, max_segments: int = MAX_PATH_SEGMENTS,
                    min_dwell: float = 0.45
                    ) -> List[Tuple[float, float, float, float]]:
    """Collapse neighbouring segments that hold effectively the same framing.

    Two crops 0.6% of the frame apart are indistinguishable, so keeping both
    only lengthens the filter expression and adds imperceptible jitter.

    `min_dwell` is the important one: a crop that only holds a position for a
    fraction of a second before moving again reads as stutter, not as camera
    work. Short segments are absorbed into their neighbour.
    """
    if not segs:
        return segs
    out: List[Tuple[float, float, float, float]] = [segs[0]]
    for t0, t1, x0, x1 in segs[1:]:
        pt0, pt1, px0, px1 = out[-1]
        flat_prev = abs(px1 - px0) < same
        flat_cur = abs(x1 - x0) < same
        if flat_prev and flat_cur and abs(x0 - px1) < same:
            out[-1] = (pt0, t1, px0, px0)   # extend the held frame
        elif (t1 - t0) < min_dwell and flat_prev and flat_cur:
            # Too brief to be a deliberate move: keep the previous framing.
            out[-1] = (pt0, t1, px0, px1)
        else:
            out.append((t0, t1, x0, x1))

    # Hard safety net: if a pathological clip still yields a huge path, drop the
    # least significant transitions rather than emit a monstrous expression.
    while len(out) > max_segments:
        deltas = [(abs(out[i][3] - out[i][2]) + abs(out[i][2] - out[i - 1][3]), i)
                  for i in range(1, len(out))]
        _, idx = min(deltas)
        pt0, _, px0, _ = out[idx - 1]
        _, nt1, _, nx1 = out[idx]
        out[idx - 1] = (pt0, nt1, px0, nx1)
        del out[idx]
    return out


# ---------------------------------------------------------------------------
# FFmpeg filter construction
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".") or "0"


def crop_x_expression(path: CameraPath, src_w: int, crop_w: int) -> str:
    """Expression giving the crop's left edge at time t.

    Written as a FLAT SUM of gated terms rather than nested if()s. libavutil's
    expression parser has a fixed stack of 100 (STACK_SIZE in eval.c), so a
    nested chain dies with EINVAL at exactly 100 segments - and a fast-cut vlog
    easily produces more than that. Measured: 99 nested if() parse, 100 fail.

    Each term is `gte(t,t0)*lt(t,t1)*value`, so only one is ever non-zero and
    the nesting depth stays constant no matter how many segments there are.
    """
    span = max(1, src_w - crop_w)

    def left_at(cx: float) -> float:
        return min(float(span), max(0.0, cx * src_w - crop_w / 2.0))

    segments = path.segments
    if not segments:
        return _fmt(span / 2.0)
    # Defensive: a path built elsewhere must still respect the parser's limit.
    if len(segments) > MAX_PATH_SEGMENTS:
        segments = _merge_segments(segments, max_segments=MAX_PATH_SEGMENTS)

    terms: List[str] = []
    last = len(segments) - 1
    for i, (t0, t1, x0, x1) in enumerate(segments):
        a, b = left_at(x0), left_at(x1)
        # Ramp only over a span long enough to divide by safely.
        if abs(a - b) < 0.75 or (t1 - t0) < 0.05:
            value = _fmt((a + b) / 2.0)
        else:
            value = (f"({_fmt(a)}+({_fmt(b - a)})*(t-{_fmt(t0)})"
                     f"/{_fmt(t1 - t0)})")
        # Half-open gates so adjacent segments never both fire. The final
        # segment stays open-ended to cover any frame past the last sample.
        if i == 0 and i == last:
            return value
        if i == last:
            terms.append(f"gte(t,{_fmt(t0)})*{value}")
        elif i == 0:
            terms.append(f"lt(t,{_fmt(t1)})*{value}")
        else:
            terms.append(f"gte(t,{_fmt(t0)})*lt(t,{_fmt(t1)})*{value}")
    return "+".join(terms)


# name -> (x expression, y expression), relative to the overlay input "W"/"H"
# (output frame) and the watermark's own "w"/"h", with `M` substituted for the
# margin in pixels.
WATERMARK_POSITIONS: Dict[str, Tuple[str, str]] = {
    "top_left":     ("M", "M"),
    "top_right":    ("W-w-M", "M"),
    "bottom_left":  ("M", "H-h-M"),
    "bottom_right": ("W-w-M", "H-h-M"),
    "center":       ("(W-w)/2", "(H-h)/2"),
}


def _watermark_overlay(base_chain: str, out_w: int, out_h: int,
                       watermark: Dict[str, Any]) -> str:
    """Wrap `base_chain` (a plain -vf chain on input 0) into a filter_complex
    graph that also scales input 1 (the watermark image) and overlays it.

    Returns a full filter_complex string ending in an output pad named [vout].
    """
    scale = max(0.02, min(0.9, float(watermark.get("scale", 0.18) or 0.18)))
    opacity = max(0.05, min(1.0, float(watermark.get("opacity", 0.85) or 0.85)))
    margin_ratio = max(0.0, min(0.4, float(watermark.get("margin", 0.04) or 0.04)))
    wm_w = max(2, int(round(out_w * scale)))
    margin_px = max(0, int(round(min(out_w, out_h) * margin_ratio)))
    x_expr, y_expr = WATERMARK_POSITIONS.get(
        str(watermark.get("position", "top_right")), WATERMARK_POSITIONS["top_right"])
    x_expr = x_expr.replace("M", str(margin_px))
    y_expr = y_expr.replace("M", str(margin_px))

    return (
        f"[0:v]{base_chain}[base];"
        f"[1:v]scale={wm_w}:-1,format=rgba,colorchannelmixer=aa={opacity:.3f}[wm];"
        f"[base][wm]overlay=x={x_expr}:y={y_expr}:format=auto[vout]"
    )


def build_filter(src_w: int, src_h: int, fmt: str, path: Optional[CameraPath],
                 layout: str = "crop", subtitle_file: Optional[str] = None,
                 blur_strength: int = 28,
                 watermark: Optional[Dict[str, Any]] = None,
                 music: Optional[Dict[str, Any]] = None,
                 clip_duration: float = 0.0,
                 music_offset: float = 0.0,
                 music_fade_in: bool = True,
                 music_fade_out: bool = True,
                 ) -> Tuple[str, int, int, bool, List[str], str]:
    """Full filter chain for one clip.

    Returns (filter, out_w, out_h, complex, extra_inputs, audio_map):
    - `complex` is True when `filter` is a filter_complex graph (a watermark
      or music track is active, either of which needs another input) rather
      than a plain single-input -vf chain - the caller must pass it through
      accordingly.
    - `extra_inputs` is the ordered list of additional -i paths the graph
      references (watermark image, then music file, in that order) - the
      caller must add them as FFmpeg inputs in exactly this order.
    - `audio_map` is what to pass to -map for the audio stream: "0:a?" for
      an untouched passthrough, or "[aout]" for the music-mixed output pad.

    Captions are appended LAST so they are drawn on the final frame - sizing and
    position must follow the output aspect, not the source's.
    """
    out_w, out_h, _, _ = format_spec(fmt)
    src_w = max(2, int(src_w or 1920))
    src_h = max(2, int(src_h or 1080))
    target = out_w / out_h
    source = src_w / src_h
    parts: List[str] = []

    if layout == "fit_blur":
        # Whole frame kept, letterboxed onto a blurred enlargement of itself.
        # Right choice when two people sit far apart and no crop holds both.
        parts.append(
            f"split=2[bg][fg];"
            f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},gblur=sigma={blur_strength}[bgb];"
            f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
        )
    elif abs(source - target) < 0.02:
        parts.append(f"scale={out_w}:{out_h}")
    elif source > target:
        # source is wider: crop horizontally, track the subject
        crop_w = max(2, int(round(src_h * target)))
        crop_w -= crop_w % 2
        crop_w = min(crop_w, src_w)
        if path and path.segments:
            expr = crop_x_expression(path, src_w, crop_w)
            parts.append(f"crop={crop_w}:{src_h}:x='{expr}':y=0")
        else:
            parts.append(f"crop={crop_w}:{src_h}:x=(iw-{crop_w})/2:y=0")
        parts.append(f"scale={out_w}:{out_h}")
    else:
        # source is taller than the target: crop vertically, bias to the top
        # third where heads live rather than the exact middle
        crop_h = max(2, int(round(src_w / target)))
        crop_h -= crop_h % 2
        crop_h = min(crop_h, src_h)
        parts.append(f"crop={src_w}:{crop_h}:x=0:y=(ih-{crop_h})*0.35")
        parts.append(f"scale={out_w}:{out_h}")

    parts.append("setsar=1")
    if subtitle_file:
        parts.append(f"subtitles={subtitle_file}")
    chain = ",".join(parts)

    has_wm = bool(watermark and watermark.get("enabled") and watermark.get("path"))
    has_music = bool(music and music.get("enabled") and music.get("path"))

    extra_inputs: List[str] = []
    graph_parts: List[str] = []
    audio_map = "0:a?"

    if has_wm:
        graph_parts.append(_watermark_overlay(chain, out_w, out_h, watermark))
        extra_inputs.append(watermark["path"])
    elif has_music:
        # No watermark, but music still needs filter_complex mode - give the
        # video its own explicit output pad too.
        graph_parts.append(f"[0:v]{chain}[vout]")

    if has_music:
        idx = len(extra_inputs) + 1
        graph_parts.append(build_music_graph(
            music, clip_duration, idx, offset=music_offset,
            fade_in=music_fade_in, fade_out=music_fade_out))
        extra_inputs.append(music["path"])
        audio_map = "[aout]"

    if graph_parts:
        return ";".join(graph_parts), out_w, out_h, True, extra_inputs, audio_map
    return chain, out_w, out_h, False, extra_inputs, audio_map


def plan_reframe(ffmpeg_bin: str, src: str, start: float, end: float,
                 fmt: str, src_w: int, src_h: int, layout: str = "crop",
                 sample_fps: float = 3.0, on_log=None) -> Optional[CameraPath]:
    """Only analyse the video when the chosen format actually needs a crop."""
    out_w, out_h, _, _ = format_spec(fmt)
    if layout == "fit_blur":
        return None
    if abs((src_w or 1920) / (src_h or 1080) - out_w / out_h) < 0.02:
        return None
    if (src_w or 1920) / (src_h or 1080) <= out_w / out_h:
        return None   # vertical crop path, no tracking needed
    samples = sample_subject(ffmpeg_bin, src, start, end, sample_fps=sample_fps,
                             on_log=on_log)
    if not samples:
        return None
    path = build_path(samples, end - start)
    if on_log:
        on_log(f"Reframe {fmt}: subject found in {path.coverage * 100:.0f}% of "
               f"{path.frames} sampled frames, {path.cuts} shot change(s), "
               f"{len(path.segments)} camera move(s)")
    return path
