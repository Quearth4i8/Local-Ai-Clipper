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
from .captions import build_ass, words_in_range

CSV_COLUMNS = ["rank", "start", "end", "start_tc", "end_tc", "duration", "score",
               "type", "title", "reason", "hook", "payoff", "emotion", "curiosity",
               "standalone", "editability"]


def safe_stem(name: str, limit: int = 60) -> str:
    stem = re.sub(r"[^\w\s.-]", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return (stem[:limit] or "clips")


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
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    stem = safe_stem(Path(video_path).stem)
    folder = Path(output_dir) / f"{stem}_clips"
    folder.mkdir(parents=True, exist_ok=True)
    burn_captions = bool(captions and captions.get("enabled"))

    written: List[Dict[str, Any]] = []
    total = len(clips)
    for n, clip in enumerate(clips, start=1):
        rank = clip.get("rank", n)
        start = max(0.0, float(clip["start"]) - pad_start)
        end = float(clip["end"]) + pad_end
        if video_duration:
            end = min(end, video_duration)
        title = safe_stem(str(clip.get("title", "")).strip(), 40) or clip.get("type", "clip")
        out = folder / f"clip_{int(rank):02d}_{title}.mp4"
        if on_progress:
            on_progress(n, total, out.name)

        words = _clip_words(clip)
        if burn_captions and words:
            # Build the subtitle in a temp dir, not next to the exports: a failed
            # burn used to leave a stray .ass sitting in the output folder.
            tmp = Path(tempfile.mkdtemp(prefix="clipcap_"))
            try:
                ass_path = write_ass_for_clip(words, start, end, video_width,
                                              video_height, captions, tmp / "c.ass")
                ff.burn(video_path, str(out), start, end, str(ass_path),
                        encoder=encoder, fonts_dir=fonts_dir)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            ff.cut(video_path, str(out), start, end, mode=mode, crf=crf, preset=preset)

        written.append({"rank": rank, "path": str(out), "start": start, "end": end,
                        "start_tc": fmt_ts(start), "end_tc": fmt_ts(end),
                        "captions": bool(burn_captions and words)})
    return written


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
