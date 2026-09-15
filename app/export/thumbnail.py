"""Auto-generated clip thumbnails: pick the strongest detected frame (usually
the clearest face shot, via the same face detector reframing uses) and burn
the clip's hook line on top, styled like a real short-form cover image.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..video.ffmpeg_tools import FFmpeg
from ..video.reframe import build_filter, sample_subject

log = logging.getLogger("clipfinder.thumbnail")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:  # pragma: no cover - optional dependency
    PIL_OK = False

# Filenames that exist on a stock Windows install, tried in order. A bold,
# heavy face reads best on a thumbnail - Impact first, then the safest bolds.
_WINDOWS_FONT_CANDIDATES = ["impact.ttf", "arialbd.ttf", "segoeuib.ttf", "verdanab.ttf"]


def pil_available() -> bool:
    return PIL_OK


def resolve_font(fonts_dir: Optional[Path] = None, preferred_name: str = "") -> Optional[Path]:
    """Best-effort path to a real font FILE - Pillow needs one, unlike libass
    which resolves family names through the system font database."""
    if fonts_dir and fonts_dir.is_dir():
        candidates = list(fonts_dir.glob("*.[ot]t[fc]"))
        if preferred_name:
            for p in candidates:
                if p.stem.lower() == preferred_name.strip().lower():
                    return p
        if candidates:
            return candidates[0]
    win_fonts = Path("C:/Windows/Fonts")
    if win_fonts.is_dir():
        for name in _WINDOWS_FONT_CANDIDATES:
            p = win_fonts / name
            if p.exists():
                return p
    return None


def pick_best_frame_time(ffmpeg_bin: str, video_path: str, start: float, end: float,
                         on_log: Optional[Callable[[str], None]] = None) -> float:
    """Best-guess timestamp for a thumbnail: the sampled frame with the
    strongest detected face, or ~18% into the clip when none was found -
    avoids frame 0, which is often mid-blink or a transition."""
    try:
        samples = sample_subject(ffmpeg_bin, video_path, start, end,
                                 sample_fps=2.0, on_log=on_log)
    except Exception as exc:  # noqa: BLE001
        if on_log:
            on_log(f"Thumbnail frame sampling failed, using a fallback frame: {exc}")
        samples = []
    best = max((s for s in samples if s.cx is not None), key=lambda s: s.weight, default=None)
    if best:
        return start + best.t
    return start + max(0.3, (end - start) * 0.18)


def _hex_rgb(value: str, default: tuple) -> tuple:
    try:
        h = value.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:  # noqa: BLE001
        return default


def _wrap_text(draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.FreeTypeFont",
              max_width: float) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if not cur or draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def compose_text(frame_path: Path, text: str, out_path: Path, *,
                 font_path: Optional[Path] = None, text_color: str = "#FFFFFF",
                 outline_color: str = "#000000", uppercase: bool = True) -> Path:
    """Overlay bold, outlined, word-wrapped text over the lower third of an
    already-cropped/watermarked frame (ffmpeg does the crop/watermark; this
    only adds the text, which needs real layout control ffmpeg's drawtext
    does not give us easily)."""
    if not PIL_OK:
        raise RuntimeError("Pillow is not installed - run: pip install Pillow")

    img = Image.open(frame_path).convert("RGBA")
    w, h = img.size
    label = (text or "").strip()
    if uppercase:
        label = label.upper()

    if label:
        # Darken the lower third with a soft gradient so text stays legible
        # over any footage, without flattening the whole frame.
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        band_top = int(h * 0.55)
        for y in range(band_top, h):
            alpha = int(165 * (y - band_top) / max(1, h - band_top))
            od.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        max_w = w * 0.9
        size = int(h * 0.13)
        min_size = max(10, int(h * 0.045))
        font, lines = None, [label]
        while size >= min_size:
            try:
                font = (ImageFont.truetype(str(font_path), size) if font_path
                        else ImageFont.load_default(size=size))
            except Exception:  # noqa: BLE001
                font = ImageFont.load_default()
            lines = _wrap_text(draw, label, font, max_w)
            line_h = size * 1.15
            if len(lines) <= 3 and line_h * len(lines) <= h * 0.42:
                break
            size -= max(2, size // 14)

        stroke_w = max(2, size // 14)
        line_h = size * 1.15
        y = h - int(h * 0.06) - line_h * len(lines)
        for line in lines:
            tw = draw.textlength(line, font=font)
            x = (w - tw) / 2
            draw.text((x, y), line, font=font, fill=_hex_rgb(text_color, (255, 255, 255)),
                      stroke_width=stroke_w, stroke_fill=_hex_rgb(outline_color, (0, 0, 0)))
            y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, quality=92)
    return out_path


def generate_clip_thumbnail(
    ff: FFmpeg, video_path: str, thumb_t: float, fmt: str,
    video_width: int, video_height: int, layout: str,
    out_path: Path, text: str, *,
    watermark: Optional[Dict[str, Any]] = None,
    font_path: Optional[Path] = None,
    text_color: str = "#FFFFFF",
    outline_color: str = "#000000",
    uppercase: bool = True,
) -> Path:
    """One JPG: a still frame cropped/scaled the same way the video export
    is (plus watermark, via the same filter_complex build used for video),
    with the hook line composited on top by Pillow."""
    vf, _, _, use_fc, _, _ = build_filter(video_width, video_height, fmt, None,
                                          layout=layout, watermark=watermark)
    raw = out_path.with_name(out_path.stem + "_raw.jpg")
    ff.frame(video_path, str(raw), thumb_t, vf=vf,
            extra_input=watermark["path"] if watermark else None,
            filter_complex=use_fc)
    try:
        compose_text(raw, text, out_path, font_path=font_path, text_color=text_color,
                    outline_color=outline_color, uppercase=uppercase)
    finally:
        try:
            raw.unlink()
        except OSError:
            pass
    return out_path
