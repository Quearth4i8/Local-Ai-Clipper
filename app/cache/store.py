"""Transcript + result caching keyed by video fingerprint.

Re-analysing the same file never re-transcribes it unless you ask for it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Transcript, VideoInfo


def fingerprint(path: str, sample_bytes: int = 4 * 1024 * 1024) -> str:
    """Fast, stable id: size + mtime + head/tail content hash."""
    p = Path(path)
    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())
    h.update(p.name.encode("utf-8", "ignore"))
    with open(p, "rb") as fh:
        h.update(fh.read(sample_bytes))
        if stat.st_size > sample_bytes * 2:
            fh.seek(-sample_bytes, os.SEEK_END)
            h.update(fh.read(sample_bytes))
    return h.hexdigest()[:20]


# Sub-folders of cache/ that are not per-video entries.
RESERVED_DIRS = {"models", "thumbs"}

# Incremented when the transcription pipeline changes what it produces.
TRANSCRIPT_VERSION = 2


class CacheStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def dir_for(self, video_hash: str) -> Path:
        d = self.root / video_hash
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _transcript_key(whisper_model: str, language: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "_"
                       for ch in f"{whisper_model}_{language or 'auto'}")
        # Version tag: bump when transcription output changes materially, so an
        # improved pass is not masked by an older cached transcript.
        # v2 = gap-filling pass (recovers passages Whisper skipped).
        return f"transcript_v{TRANSCRIPT_VERSION}_{safe}.json"

    # -------------------------------------------------------------- metadata
    def write_metadata(self, video_hash: str, info: VideoInfo,
                       extra: Optional[Dict[str, Any]] = None) -> None:
        data = {"video": info.to_dict(), "cached_at": time.time()}
        if extra:
            data.update(extra)
        (self.dir_for(video_hash) / "metadata.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def read_metadata(self, video_hash: str) -> Optional[Dict[str, Any]]:
        p = self.dir_for(video_hash) / "metadata.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ------------------------------------------------------------ transcript
    def transcript_path(self, video_hash: str, whisper_model: str, language: str) -> Path:
        return self.dir_for(video_hash) / self._transcript_key(whisper_model, language)

    def load_transcript(self, video_hash: str, whisper_model: str,
                        language: str) -> Optional[Transcript]:
        p = self.transcript_path(video_hash, whisper_model, language)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        # An empty transcript is a failed run, not a result. Ignoring it here
        # means a retry actually re-transcribes instead of replaying the failure.
        if not data.get("raw_segments"):
            return None
        return Transcript(
            language=data.get("language", "en"),
            duration=float(data.get("duration", 0.0)),
            raw_segments=data.get("raw_segments", []),
        )

    def save_transcript(self, video_hash: str, whisper_model: str, language: str,
                        transcript: Transcript) -> Optional[Path]:
        if not transcript.raw_segments:
            return None   # never cache a failed transcription
        p = self.transcript_path(video_hash, whisper_model, language)
        payload = {
            "language": transcript.language,
            "duration": transcript.duration,
            "whisper_model": whisper_model,
            "created_at": time.time(),
            "raw_segments": transcript.raw_segments,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return p

    # --------------------------------------------------------------- results
    def save_result(self, video_hash: str, result: Dict[str, Any]) -> Path:
        p = self.dir_for(video_hash) / "last_analysis.json"
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def load_result(self, video_hash: str) -> Optional[Dict[str, Any]]:
        p = self.dir_for(video_hash) / "last_analysis.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ----------------------------------------------------------- maintenance
    def entries(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not d.is_dir() or d.name in RESERVED_DIRS:
                continue
            meta = self.read_metadata(d.name) or {}
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            out.append({
                "hash": d.name,
                "video": (meta.get("video") or {}).get("filename", "?"),
                "size_mb": round(size / 1_048_576, 1),
                "cached_at": meta.get("cached_at", 0),
            })
        return out

    def clear(self, video_hash: Optional[str] = None) -> int:
        """Drop cached transcripts. Never deletes the downloaded Whisper models."""
        import shutil
        if video_hash:
            targets = [] if video_hash in RESERVED_DIRS else [self.root / video_hash]
        else:
            targets = [d for d in self.root.iterdir()
                       if d.is_dir() and d.name not in RESERVED_DIRS]
        removed = 0
        for t in targets:
            if t.exists() and t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
                removed += 1
        return removed
