"""Local performance tracking: log how exported clips actually did, and
surface which score categories correlate with real engagement so scoring
weights can be tuned from evidence instead of guesswork.

Everything lives in one SQLite file next to the transcript cache - no
network calls, no analytics SDK, nothing leaves the machine.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCORE_KEYS = ("hook", "payoff", "emotion", "curiosity", "standalone", "editability")

# Comments/shares are stronger algorithmic signals than a raw view, so they
# count for more in the synthetic "engagement" figure insights are built on.
ENGAGEMENT_WEIGHTS = {"views": 1.0, "likes": 3.0, "comments": 5.0, "shares": 8.0}

MIN_SAMPLE = 5  # entries needed (with a usable video-relative signal) before insights are shown

SCHEMA = """
CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_hash TEXT,
    video_name TEXT,
    rank TEXT,
    title TEXT,
    clip_type TEXT,
    duration REAL,
    scores TEXT,
    overall REAL,
    virality REAL,
    platform TEXT,
    views INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    rating REAL,
    notes TEXT,
    posted_at TEXT,
    logged_at REAL
);
"""


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


class PerformanceStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ CRUD
    def log(self, entry: Dict[str, Any]) -> int:
        scores = entry.get("scores") or {}
        row = (
            entry.get("video_hash"), entry.get("video_name"), str(entry.get("rank", "")),
            entry.get("title"), entry.get("clip_type"), float(entry.get("duration", 0) or 0),
            json.dumps({k: float(scores.get(k, 0) or 0) for k in SCORE_KEYS}),
            float(entry.get("overall", 0) or 0), float(entry.get("virality", 0) or 0),
            str(entry.get("platform", "other") or "other"),
            int(entry.get("views", 0) or 0), int(entry.get("likes", 0) or 0),
            int(entry.get("comments", 0) or 0), int(entry.get("shares", 0) or 0),
            entry.get("rating"), entry.get("notes"), entry.get("posted_at"),
            time.time(),
        )
        cur = self._conn.execute(
            "INSERT INTO performance (video_hash, video_name, rank, title, clip_type, "
            "duration, scores, overall, virality, platform, views, likes, comments, "
            "shares, rating, notes, posted_at, logged_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row,
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list(self, video_hash: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        q = "SELECT * FROM performance"
        args: tuple = ()
        if video_hash:
            q += " WHERE video_hash = ?"
            args = (video_hash,)
        q += " ORDER BY logged_at DESC LIMIT ?"
        args = args + (limit,)
        cur = self._conn.execute(q, args)
        cols = [d[0] for d in cur.description]
        out = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            try:
                d["scores"] = json.loads(d.get("scores") or "{}")
            except (TypeError, ValueError):
                d["scores"] = {}
            d["engagement"] = self._engagement(d)
            out.append(d)
        return out

    def delete(self, entry_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM performance WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # --------------------------------------------------------------- insight
    @staticmethod
    def _engagement(row: Dict[str, Any]) -> float:
        return sum(ENGAGEMENT_WEIGHTS[k] * float(row.get(k, 0) or 0) for k in ENGAGEMENT_WEIGHTS)

    def insights(self, current_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        rows = self.list(limit=5000)
        if not rows:
            return {"ready": False, "count": 0, "need": MIN_SAMPLE, "reason": "No performance logged yet."}

        # Engagement is only comparable WITHIN one source video (view counts
        # scale with channel size/reach, not clip quality) - so it's z-scored
        # per video_hash, and only videos with 2+ logged clips carry a signal.
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_video.setdefault(r.get("video_hash") or "unknown", []).append(r)

        signal_rows: List[Dict[str, Any]] = []
        for group in by_video.values():
            if len(group) < 2:
                continue
            vals = [r["engagement"] for r in group]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            sd = var ** 0.5
            if sd <= 0:
                continue
            for r, v in zip(group, vals):
                r["z"] = (v - mean) / sd
                signal_rows.append(r)

        if len(signal_rows) < MIN_SAMPLE:
            return {
                "ready": False, "count": len(rows), "with_signal": len(signal_rows),
                "need": MIN_SAMPLE,
                "reason": (f"{len(rows)} clip(s) logged, but only {len(signal_rows)} sit in a "
                          f"source video with 2+ logged clips - need at least {MIN_SAMPLE} of "
                          "those before a trend is more than noise."),
            }

        categories: Dict[str, Dict[str, Any]] = {}
        for key in (*SCORE_KEYS, "overall", "virality"):
            if key in SCORE_KEYS:
                xs = [float((r.get("scores") or {}).get(key, 0) or 0) for r in signal_rows]
            else:
                xs = [float(r.get(key, 0) or 0) for r in signal_rows]
            ys = [r["z"] for r in signal_rows]
            corr = round(_pearson(xs, ys), 3)
            if corr >= 0.3:
                note = "higher scores here tracked with better performance"
            elif corr <= -0.3:
                note = "higher scores here tracked with WORSE performance"
            else:
                note = "no clear signal yet"
            categories[key] = {"correlation": corr, "note": note}

        by_type: Dict[str, List[float]] = {}
        for r in signal_rows:
            by_type.setdefault(r.get("clip_type") or "other", []).append(r["z"])
        type_ranking = sorted(
            ({"clip_type": t, "avg_z": round(sum(v) / len(v), 2), "count": len(v)}
             for t, v in by_type.items()),
            key=lambda x: -x["avg_z"],
        )

        suggested_weights = self._suggest_weights(categories, current_weights)

        return {
            "ready": True, "count": len(rows), "with_signal": len(signal_rows),
            "categories": categories, "by_clip_type": type_ranking,
            "suggested_weights": suggested_weights,
        }

    @staticmethod
    def _suggest_weights(categories: Dict[str, Dict[str, Any]],
                         current: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Gently nudge each weight toward its measured correlation - capped at
        +/-25% per round so one batch of data can't swing scoring wildly.
        Renormalised back to the current total so the rubric still reads out
        of the same scale.
        """
        base = dict(current or {"hook": 25, "payoff": 25, "emotion": 15,
                                "curiosity": 15, "standalone": 10, "editability": 10})
        total = sum(base.values()) or 100.0
        nudged = {}
        for key in SCORE_KEYS:
            corr = categories.get(key, {}).get("correlation", 0.0)
            # Only act on a correlation past the "clear signal" threshold -
            # nudging weights on noise would just make scoring less stable.
            corr = corr if abs(corr) >= 0.3 else 0.0
            factor = 1.0 + max(-0.25, min(0.25, corr))
            nudged[key] = max(1.0, base.get(key, 10.0) * factor)
        scale = total / (sum(nudged.values()) or 1.0)
        return {k: round(v * scale, 1) for k, v in nudged.items()}
