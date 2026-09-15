"""JSON / CSV / plain-text export, plus optional FFmpeg clip cutting."""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..models import Word, fmt_ts
from ..video.ffmpeg_tools import FFmpeg
from ..video.reframe import (FORMATS, build_filter, format_spec, plan_reframe,
                             safe_tag)
from ..video.trimming import is_noop, plan_keep_spans, total_duration
from .captions import build_ass, words_in_range
from .thumbnail import (generate_clip_thumbnail, pick_best_frame_time, pil_available,
                        resolve_font)

CSV_COLUMNS = ["rank", "start", "end", "start_tc", "end_tc", "duration", "score",
               "type", "title", "reason", "hook", "payoff", "emotion", "curiosity",
               "standalone", "editability"]


def safe_stem(name: str, limit: int = 60) -> str:
    stem = re.sub(r"[^\w\s.-]", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return (stem[:limit] or "clips")


def normalize_watermark(watermark: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """None unless a watermark is actually usable - enabled, with an image
    that exists on disk. Callers can then just check truthiness."""
    if not watermark or not watermark.get("enabled"):
        return None
    path = str(watermark.get("path") or "").strip()
    if not path or not Path(path).is_file():
        return None
    return {**watermark, "path": path}


def normalize_tighten(tighten: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """None unless jump-cut tightening is actually enabled - callers can just
    check truthiness."""
    if not tighten or not tighten.get("enabled"):
        return None
    return {
        "remove_fillers": bool(tighten.get("remove_fillers", True)),
        "min_gap": float(tighten.get("min_gap", 0.6) or 0.6),
        "keep_pause": float(tighten.get("keep_pause", 0.35) or 0.35),
    }


def normalize_thumbnail(thumbnail: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """None unless thumbnails are enabled AND Pillow is actually installed."""
    if not thumbnail or not thumbnail.get("enabled"):
        return None
    if not pil_available():
        return None
    return {
        "text_color": str(thumbnail.get("text_color", "#FFFFFF") or "#FFFFFF"),
        "outline_color": str(thumbnail.get("outline_color", "#000000") or "#000000"),
        "uppercase": bool(thumbnail.get("uppercase", True)),
    }


def normalize_music(music: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """None unless a background track is actually usable - enabled, with a
    file that exists on disk. Callers can then just check truthiness."""
    if not music or not music.get("enabled"):
        return None
    path = str(music.get("path") or "").strip()
    if not path or not Path(path).is_file():
        return None
    return {
        "enabled": True,
        "path": path,
        "volume": float(music.get("volume", 0.15) or 0.15),
        "fade_seconds": float(music.get("fade_seconds", 0.6) or 0.6),
    }


def _render_segment(
    ff: FFmpeg, video_path: str, seg_start: float, seg_end: float, fmt: str,
    video_width: int, video_height: int, layout: str, sample_fps: float,
    burn_captions: bool, words: Sequence[Word], captions_cfg: Optional[Dict[str, Any]],
    watermark: Optional[Dict[str, Any]], encoder: str, tmp_dir: Path, seg_name: str,
    out_path: Path, music: Optional[Dict[str, Any]] = None, music_offset: float = 0.0,
    music_fade_in: bool = True, music_fade_out: bool = True,
    on_log: Optional[Callable[[str], None]] = None,
) -> Path:
    """Render ONE continuous [seg_start, seg_end) span through reframe, plus
    optional burned captions, watermark and background music, to `out_path`.

    Shared building block: a viral compilation and a tightened (jump-cut)
    single clip are both "several independent time-spans, rendered and then
    concatenated into one file" - this is that one render step. `music_offset`
    is how far into the song's endless loop this span starts, so a caller
    stitching several spans of one clip/compilation can keep the music
    continuous across the cuts instead of restarting it each time.
    """
    path = plan_reframe(ff.ffmpeg, video_path, seg_start, seg_end, fmt,
                        video_width, video_height, layout=layout,
                        sample_fps=sample_fps, on_log=on_log)
    sub_name = None
    if burn_captions and words:
        _, out_w, out_h, _, _, _ = build_filter(video_width, video_height, fmt, path, layout=layout)
        write_ass_for_clip(words, seg_start, seg_end, out_w, out_h,
                           captions_cfg, tmp_dir / f"{seg_name}.ass")
        sub_name = f"{seg_name}.ass"
    vf, _, _, use_fc, extra_inputs, audio_map = build_filter(
        video_width, video_height, fmt, path, layout=layout, subtitle_file=sub_name,
        watermark=watermark, music=music, clip_duration=seg_end - seg_start,
        music_offset=music_offset, music_fade_in=music_fade_in, music_fade_out=music_fade_out,
    )
    ff.render(video_path, str(out_path), seg_start, seg_end, vf=vf, work_dir=str(tmp_dir),
             encoder=encoder, extra_inputs=extra_inputs, filter_complex=use_fc,
             audio_map=audio_map)
    return out_path


def _row(clip: Dict[str, Any]) -> Dict[str, Any]:
    scores = clip.get("scores", {})
    return {
        "rank": clip.get("rank"),
        "start": clip.get("start"),
        "end": clip.get("end"),
        "start_tc": clip.get("start_tc"),
        "end_tc": clip.get("end_tc"),
        "duration": clip.get("duration"),
        "score": clip.get("score"),
        "type": clip.get("type"),
        "title": clip.get("title"),
        "reason": " ".join(str(clip.get("reason", "")).split()),
        **{k: scores.get(k, "") for k in
           ("hook", "payoff", "emotion", "curiosity", "standalone", "editability")},
    }


def to_json(result: Dict[str, Any]) -> str:
    video = result.get("video", {})
    payload = {
        "video": video.get("filename"),
        "video_path": video.get("path"),
        "duration": video.get("duration"),
        "language": result.get("language"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "Local AI Clip Finder",
        "stats": result.get("stats", {}),
        "clips": [
            {
                "rank": c.get("rank"),
                "start": c.get("start"),
                "end": c.get("end"),
                "duration": c.get("duration"),
                "start_tc": c.get("start_tc"),
                "end_tc": c.get("end_tc"),
                "score": c.get("score"),
                "scores": c.get("scores"),
                "confidence": c.get("confidence"),
                "type": c.get("type"),
                "title": c.get("title"),
                "reason": c.get("reason"),
                "verdict": c.get("verdict"),
                "transcript": c.get("transcript"),
            }
            for c in result.get("clips", [])
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def to_csv(result: Dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for clip in result.get("clips", []):
        writer.writerow(_row(clip))
    return buf.getvalue()


def to_text(result: Dict[str, Any]) -> str:
    video = result.get("video", {})
    lines: List[str] = [
        "LOCAL AI CLIP FINDER - results",
        f"Video    : {video.get('filename')}",
        f"Duration : {video.get('duration_tc')}",
        f"Language : {result.get('language')}",
        f"Clips    : {len(result.get('clips', []))}",
        "=" * 66,
        "",
    ]
    for c in result.get("clips", []):
        s = c.get("scores", {})
        lines += [
            f"#{c.get('rank')} - {c.get('score')}/100   [{c.get('type')}]",
            f"{c.get('start_tc')} -> {c.get('end_tc')}   ({c.get('duration')}s)",
            f"Title: {c.get('title')}",
            (f"Hook {s.get('hook')}/25 | Payoff {s.get('payoff')}/25 | "
             f"Emotion {s.get('emotion')}/15 | Curiosity {s.get('curiosity')}/15 | "
             f"Standalone {s.get('standalone')}/10 | Editability {s.get('editability')}/10"),
            f"Why: {c.get('reason')}",
        ] + ([f"Ranking verdict: {c.get('verdict')}"] if c.get("verdict") else []) + [
            "Transcript:",
            "  " + " ".join(str(c.get("transcript", "")).split()),
            "-" * 66,
            "",
        ]
    return "\n".join(lines)


def write_all(result: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    stem = safe_stem(Path(result.get("video", {}).get("filename", "clips")).stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(output_dir) / f"{stem}_{stamp}"
    base.parent.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for suffix, content in (("json", to_json(result)), ("csv", to_csv(result)),
                            ("txt", to_text(result))):
        path = base.with_suffix("." + suffix)
        path.write_text(content, encoding="utf-8-sig" if suffix == "csv" else "utf-8")
        written[suffix] = str(path)
    return written


# ---------------------------------------------------------------------------
# Optional: actually cut the clips
# ---------------------------------------------------------------------------
def export_clips(
    ff: FFmpeg,
    video_path: str,
    clips: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    mode: str = "copy",
    crf: int = 20,
    preset: str = "veryfast",
    pad_start: float = 0.15,
    pad_end: float = 0.35,
    video_duration: float = 0.0,
    captions: Optional[Dict[str, Any]] = None,
    video_width: int = 0,
    video_height: int = 0,
    encoder: str = "auto",
    fonts_dir: Optional[str] = None,
    formats: Optional[Sequence[str]] = None,
    layout: str = "crop",
    sample_fps: float = 3.0,
    watermark: Optional[Dict[str, Any]] = None,
    tighten: Optional[Dict[str, Any]] = None,
    thumbnail: Optional[Dict[str, Any]] = None,
    music: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    stem = safe_stem(Path(video_path).stem)
    folder = Path(output_dir) / f"{stem}_clips"
    folder.mkdir(parents=True, exist_ok=True)
    burn_captions = bool(captions and captions.get("enabled"))
    wanted = [f for f in (formats or ["16:9"]) if f in FORMATS] or ["16:9"]
    watermark = normalize_watermark(watermark)
    tighten = normalize_tighten(tighten)
    thumbnail = normalize_thumbnail(thumbnail)
    music = normalize_music(music)
    font_path = resolve_font(Path(fonts_dir) if fonts_dir else None,
                             str((captions or {}).get("font", ""))) if thumbnail else None

    written: List[Dict[str, Any]] = []
    total = len(clips) * len(wanted)
    step = 0

    for n, clip in enumerate(clips, start=1):
        rank = clip.get("rank", n)
        start = max(0.0, float(clip["start"]) - pad_start)
        end = float(clip["end"]) + pad_end
        if video_duration:
            end = min(end, video_duration)
        title = safe_stem(str(clip.get("title", "")).strip(), 40) or clip.get("type", "clip")
        words = _clip_words(clip)

        keep_spans = [(start, end)]
        if tighten:
            keep_spans = plan_keep_spans(words, start, end, remove_fillers=tighten["remove_fillers"],
                                         min_gap=tighten["min_gap"], keep_pause=tighten["keep_pause"])
        tightened = tighten is not None and not is_noop(keep_spans, start, end)
        if tightened and on_log:
            cut = (end - start) - total_duration(keep_spans)
            on_log(f"Clip #{rank}: tightened {cut:.1f}s of filler/dead air "
                   f"across {len(keep_spans)} segment(s)")

        thumb_t = pick_best_frame_time(ff.ffmpeg, video_path, start, end, on_log=on_log) \
            if thumbnail else None
        thumb_text = str(clip.get("hook_line") or clip.get("title") or "")

        for fmt in wanted:
            step += 1
            tag = safe_tag(fmt)
            out = folder / f"clip_{int(rank):02d}_{title}_{tag}.mp4"
            if on_progress:
                on_progress(step, total, out.name)

            native = fmt == "16:9" and abs((video_width or 1920) / (video_height or 1080)
                                           - 16 / 9) < 0.02
            # Fast path: original aspect, no captions, no watermark, no music,
            # no jump cuts -> lossless stream copy.
            if native and not (burn_captions and words) and not watermark and not music \
                    and not tightened:
                ff.cut(video_path, str(out), start, end, mode=mode, crf=crf, preset=preset)
                written.append(_record(rank, out, start, end, fmt, False, layout))
            else:
                tmp = Path(tempfile.mkdtemp(prefix="clipfmt_"))
                try:
                    if not tightened:
                        _render_segment(ff, video_path, start, end, fmt, video_width,
                                        video_height, layout, sample_fps, burn_captions,
                                        words, captions, watermark, encoder, tmp, "c", out,
                                        music=music, on_log=on_log)
                    else:
                        seg_paths = []
                        elapsed = 0.0
                        for i, (a, b) in enumerate(keep_spans):
                            seg_paths.append(_render_segment(
                                ff, video_path, a, b, fmt, video_width, video_height,
                                layout, sample_fps, burn_captions, words, captions,
                                watermark, encoder, tmp, f"seg{i}", tmp / f"seg{i}.mp4",
                                music=music, music_offset=elapsed, music_fade_in=(i == 0),
                                music_fade_out=(i == len(keep_spans) - 1), on_log=on_log))
                            elapsed += b - a
                        ff.concat([str(p) for p in seg_paths], str(out))
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

                written.append(_record(rank, out, start, end, fmt,
                                       bool(burn_captions and words), layout,
                                       tightened=tightened))

            if thumbnail and thumb_t is not None:
                try:
                    thumb_out = out.with_name(out.stem + "_thumb.jpg")
                    generate_clip_thumbnail(
                        ff, video_path, thumb_t, fmt, video_width, video_height, layout,
                        thumb_out, thumb_text, watermark=watermark, font_path=font_path,
                        text_color=thumbnail["text_color"], outline_color=thumbnail["outline_color"],
                        uppercase=thumbnail["uppercase"],
                    )
                    written[-1]["thumbnail"] = str(thumb_out)
                except Exception as exc:  # noqa: BLE001
                    if on_log:
                        on_log(f"Thumbnail for clip #{rank} ({fmt}) failed: {exc}")
    return written


def export_compilation(
    ff: FFmpeg,
    video_path: str,
    clips: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    name: str = "compilation",
    pad_start: float = 0.15,
    pad_end: float = 0.35,
    video_duration: float = 0.0,
    captions: Optional[Dict[str, Any]] = None,
    video_width: int = 0,
    video_height: int = 0,
    encoder: str = "auto",
    formats: Optional[Sequence[str]] = None,
    layout: str = "crop",
    sample_fps: float = 3.0,
    watermark: Optional[Dict[str, Any]] = None,
    tighten: Optional[Dict[str, Any]] = None,
    music: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Render an ordered list of clips as segments and stitch them into ONE
    video per format - a single "best of" cut instead of one file per clip.

    Every segment is re-encoded to the same target resolution/layout (no
    native stream-copy fast path here, unlike `export_clips`) so the final
    concat is a plain, lossless stream copy regardless of the source's own
    codec quirks.
    """
    stem = safe_stem(Path(video_path).stem)
    folder = Path(output_dir) / f"{stem}_compilations"
    folder.mkdir(parents=True, exist_ok=True)
    burn_captions = bool(captions and captions.get("enabled"))
    wanted = [f for f in (formats or ["9:16"]) if f in FORMATS] or ["9:16"]
    watermark = normalize_watermark(watermark)
    tighten = normalize_tighten(tighten)
    music = normalize_music(music)

    written: List[Dict[str, Any]] = []
    total = len(clips) * len(wanted)
    step = 0

    for fmt in wanted:
        tmp = Path(tempfile.mkdtemp(prefix="compfmt_"))
        try:
            seg_paths: List[Path] = []
            # One continuous music bed under the WHOLE compilation, not one
            # per source clip - `elapsed` tracks position in the song's loop
            # across every clip so it never restarts mid-compilation.
            elapsed = 0.0
            for n, clip in enumerate(clips, start=1):
                step += 1
                if on_progress:
                    on_progress(step, total, f"{fmt} · segment {n}/{len(clips)}")
                start = max(0.0, float(clip["start"]) - pad_start)
                end = float(clip["end"]) + pad_end
                if video_duration:
                    end = min(end, video_duration)
                words = _clip_words(clip)

                spans = [(start, end)]
                if tighten:
                    spans = plan_keep_spans(words, start, end,
                                            remove_fillers=tighten["remove_fillers"],
                                            min_gap=tighten["min_gap"],
                                            keep_pause=tighten["keep_pause"])
                for i, (a, b) in enumerate(spans):
                    seg_out = tmp / f"seg_{n:03d}_{i}.mp4"
                    _render_segment(ff, video_path, a, b, fmt, video_width, video_height,
                                    layout, sample_fps, burn_captions, words, captions,
                                    watermark, encoder, tmp, f"c{n}_{i}", seg_out,
                                    music=music, music_offset=elapsed,
                                    music_fade_in=(n == 1 and i == 0),
                                    music_fade_out=(n == len(clips) and i == len(spans) - 1),
                                    on_log=on_log)
                    seg_paths.append(seg_out)
                    elapsed += b - a

            tag = safe_tag(fmt)
            out = folder / f"{safe_stem(name, 40)}_{tag}.mp4"
            ff.concat([str(p) for p in seg_paths], str(out))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        written.append({
            "path": str(out),
            "format": fmt,
            "clip_count": len(clips),
            "duration": round(sum(float(c["end"]) - float(c["start"]) + pad_start + pad_end
                                  for c in clips), 1),
            "order": [{"rank": c.get("rank"), "title": c.get("title"),
                      "type": c.get("type")} for c in clips],
        })
    return written


def _record(rank: Any, out: Path, start: float, end: float, fmt: str,
            captions: bool, layout: str, tightened: bool = False) -> Dict[str, Any]:
    w, h, label, platforms = format_spec(fmt)
    return {"rank": rank, "path": str(out), "start": start, "end": end,
            "start_tc": fmt_ts(start), "end_tc": fmt_ts(end),
            "captions": captions, "format": fmt, "layout": layout,
            "width": w, "height": h, "label": label, "platforms": platforms,
            "tightened": tightened}


def _clip_words(clip: Dict[str, Any]) -> List[Word]:
    """Words are stored compactly as [start, end, text] triples."""
    out: List[Word] = []
    for item in clip.get("words") or []:
        try:
            if isinstance(item, dict):
                out.append(Word(start=float(item["start"]), end=float(item["end"]),
                                text=str(item.get("text", ""))))
            else:
                out.append(Word(start=float(item[0]), end=float(item[1]), text=str(item[2])))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def write_ass_for_clip(words: Sequence[Word], start: float, end: float,
                       width: int, height: int, cfg: Dict[str, Any],
                       path: Path) -> Path:
    """Build the .ass for one clip, with times rebased to the cut point."""
    inside = words_in_range(words, start, end)
    ass = build_ass(inside, width or 1080, height or 1920, cfg, time_offset=start)
    path.parent.mkdir(parents=True, exist_ok=True)
    # libass reads UTF-8; BOM keeps non-ASCII (accents) safe across tools.
    path.write_text(ass, encoding="utf-8-sig")
    return path
