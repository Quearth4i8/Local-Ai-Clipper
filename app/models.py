"""Shared data structures for the whole pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

CLIP_TYPES = [
    "funny",
    "educational",
    "controversial",
    "emotional",
    "story",
    "surprising",
    "opinion",
    "practical_advice",
    "inspirational",
    "confrontation",
    "reveal",
    "other",
]


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Sentence:
    """One natural speech unit: a sentence, or a clause ended by a real pause."""
    idx: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    pause_before: float = 0.0
    pause_after: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "pause_before": round(self.pause_before, 3),
            "pause_after": round(self.pause_after, 3),
            "words": [asdict(w) for w in self.words],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Sentence":
        return Sentence(
            idx=d["idx"],
            start=d["start"],
            end=d["end"],
            text=d["text"],
            words=[Word(**w) for w in d.get("words", [])],
            pause_before=d.get("pause_before", 0.0),
            pause_after=d.get("pause_after", 0.0),
        )


@dataclass
class Transcript:
    language: str
    duration: float
    sentences: List[Sentence] = field(default_factory=list)
    raw_segments: List[Dict[str, Any]] = field(default_factory=list)

    def text_between(self, start_idx: int, end_idx: int) -> str:
        lo = max(0, start_idx)
        hi = min(len(self.sentences) - 1, end_idx)
        return " ".join(s.text for s in self.sentences[lo: hi + 1]).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "duration": self.duration,
            "sentences": [s.to_dict() for s in self.sentences],
            "raw_segments": self.raw_segments,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Transcript":
        return Transcript(
            language=d.get("language", "en"),
            duration=d.get("duration", 0.0),
            sentences=[Sentence.from_dict(s) for s in d.get("sentences", [])],
            raw_segments=d.get("raw_segments", []),
        )


@dataclass
class Candidate:
    """A potential clip. Boundaries always land on sentence edges."""
    start_idx: int
    end_idx: int
    start: float
    end: float
    text: str = ""

    # scoring
    scores: Dict[str, float] = field(default_factory=dict)   # per-category, raw LLM values
    overall: float = 0.0
    heuristic: float = 0.0
    confidence: float = 0.0

    # descriptive
    clip_type: str = "other"
    title: str = ""
    reason: str = ""       # why pass 1 liked it
    verdict: str = ""      # pass 2's comparative judgement
    hook_line: str = ""

    # provenance / bookkeeping
    source: str = "llm"          # llm | heuristic | merged
    block_index: int = -1
    boundary_optimized: bool = False
    merged_from: int = 1
    rank: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, other: "Candidate") -> float:
        """Intersection-over-union on the time axis."""
        inter = max(0.0, min(self.end, other.end) - max(self.start, other.start))
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "start_tc": fmt_ts(self.start),
            "end_tc": fmt_ts(self.end),
            "score": round(self.overall, 1),
            "scores": {k: round(float(v), 1) for k, v in self.scores.items()},
            "confidence": round(self.confidence, 2),
            "type": self.clip_type,
            "title": self.title,
            "reason": self.reason,
            "verdict": self.verdict,
            "hook_line": self.hook_line,
            "transcript": self.text,
            "heuristic": round(self.heuristic, 1),
            "source": self.source,
            "boundary_optimized": self.boundary_optimized,
            "merged_from": self.merged_from,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
        }


@dataclass
class VideoInfo:
    path: str
    filename: str
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    size_bytes: int = 0
    vcodec: str = ""
    acodec: str = ""
    has_audio: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "duration": round(self.duration, 2),
            "duration_tc": fmt_ts(self.duration),
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}" if self.width else "unknown",
            "fps": round(self.fps, 2),
            "size_mb": round(self.size_bytes / 1_048_576, 1),
            "vcodec": self.vcodec,
            "acodec": self.acodec,
            "has_audio": self.has_audio,
        }


@dataclass
class AnalysisResult:
    video: VideoInfo
    clips: List[Candidate]
    language: str
    stats: Dict[str, Any] = field(default_factory=dict)
    transcript_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "language": self.language,
            "stats": self.stats,
            "clips": [c.to_dict() for c in self.clips],
        }
