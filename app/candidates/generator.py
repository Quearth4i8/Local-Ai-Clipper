"""Heuristic candidate window generation.

Cheap, deterministic, runs in a fraction of a second on a 3-hour transcript.
Its job is to point the LLM at the interesting 10% of the video, and to act as
a standalone fallback if no LLM is reachable.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..models import Candidate, Sentence
from .heuristics import SignalProfile, dominant_type, profile_all

TERMINAL = re.compile(r'[.!?…。！？؟]["\'»”\)\]]*\s*$')


def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))


class WindowScorer:
    """Turns a (start_idx, end_idx) span into a 0-100 'worth looking at' score."""

    def __init__(self, sentences: List[Sentence], profiles: List[SignalProfile],
                 topic: Sequence[float], ideal_duration: float = 45.0):
        self.s = sentences
        self.p = profiles
        self.topic = topic
        self.ideal = ideal_duration

    def parts(self, i: int, j: int) -> Dict[str, float]:
        s, p = self.s, self.p
        span = s[j].end - s[i].start
        if span <= 0:
            return {}
        n = j - i + 1
        head = range(i, min(j, i + 2) + 1)
        tail_start = i + max(1, int(n * 0.55))
        tail = range(min(tail_start, j), j + 1)

        # --- hook: does the opening grab attention -------------------------
        hook = max((p[k].hook for k in head), default=0.0)
        hook = max(hook, 0.6 * max((p[k].curiosity for k in head), default=0.0))
        hook = max(hook, 0.5 * max((p[k].opinion for k in head), default=0.0))
        hook *= 1.0 - 0.45 * p[i].weak_start
        hook += 0.15 * self.topic[i] if i < len(self.topic) else 0.0

        # --- payoff: does it land somewhere --------------------------------
        payoff = max((p[k].payoff for k in tail), default=0.0)
        payoff = max(payoff, 0.75 * max((p[k].value for k in tail), default=0.0))
        payoff = max(payoff, 0.6 * max((p[k].emotion for k in tail), default=0.0))

        # --- emotion / curiosity ------------------------------------------
        emotion = max((p[k].emotion for k in range(i, j + 1)), default=0.0)
        emotion = 0.7 * emotion + 0.3 * (sum(p[k].emotion for k in range(i, j + 1)) / n)
        curiosity = max((p[k].curiosity for k in range(i, min(j, i + int(n * 0.7)) + 1)),
                        default=0.0)
        curiosity = max(curiosity, 0.5 * max((p[k].story for k in range(i, j + 1)), default=0.0))

        # --- standalone: can a stranger follow it --------------------------
        standalone = 1.0 - 0.6 * p[i].weak_start
        if i < len(self.topic):
            standalone = min(1.0, standalone + 0.25 * self.topic[i])
        first_words = " ".join(s[i].text.split()[:6]).lower()
        if re.match(r"^(and|but|so|because|which|that'?s why|et|mais|donc|parce)", first_words):
            standalone *= 0.75

        # --- editability: clean in / clean out -----------------------------
        edit = 0.35
        edit += min(0.3, s[i].pause_before / 1.2 * 0.3)
        edit += min(0.25, s[j].pause_after / 1.2 * 0.25)
        if TERMINAL.search(s[j].text):
            edit += 0.15
        else:
            edit -= 0.10
        edit = max(0.0, min(1.0, edit))

        # --- shape modifiers ------------------------------------------------
        duration_fit = _gauss(span, self.ideal, 22.0)
        words = sum(s[k].word_count for k in range(i, j + 1))
        wps = words / span
        density = min(1.0, wps / 2.6) if wps < 2.6 else max(0.5, 1.0 - (wps - 2.6) / 4)

        return {
            "hook": min(1.0, hook),
            "payoff": min(1.0, payoff),
            "emotion": min(1.0, emotion),
            "curiosity": min(1.0, curiosity),
            "standalone": min(1.0, max(0.0, standalone)),
            "editability": edit,
            "duration_fit": duration_fit,
            "density": density,
        }

    def score(self, i: int, j: int) -> float:
        f = self.parts(i, j)
        if not f:
            return 0.0
        core = (25 * f["hook"] + 25 * f["payoff"] + 15 * f["emotion"]
                + 15 * f["curiosity"] + 10 * f["standalone"] + 10 * f["editability"])
        return core * (0.72 + 0.18 * f["duration_fit"] + 0.10 * f["density"])


def generate_windows(
    sentences: List[Sentence],
    language: str,
    min_duration: float = 20.0,
    max_duration: float = 90.0,
    ideal_duration: float = 45.0,
    topic: Optional[Sequence[float]] = None,
    profiles: Optional[List[SignalProfile]] = None,
    limit: int = 400,
) -> Tuple[List[Candidate], List[SignalProfile], WindowScorer]:
    """Enumerate every sentence-aligned window in [min, max] and keep the best."""
    if profiles is None:
        profiles = profile_all(sentences, language)
    if topic is None:
        topic = [0.0] * len(sentences)

    scorer = WindowScorer(sentences, profiles, topic, ideal_duration)
    n = len(sentences)
    scored: List[Tuple[float, int, int]] = []

    for i in range(n):
        # skip openings that are obviously unusable as a first line
        if profiles[i].weak_start >= 0.95:
            continue
        j = i
        while j < n and sentences[j].end - sentences[i].start < min_duration:
            j += 1
        while j < n and sentences[j].end - sentences[i].start <= max_duration:
            scored.append((scorer.score(i, j), i, j))
            j += 1

    scored.sort(reverse=True)
    kept = _suppress(scored, sentences, limit=limit, iou=0.6)

    out: List[Candidate] = []
    for value, i, j in kept:
        out.append(Candidate(
            start_idx=i, end_idx=j,
            start=sentences[i].start, end=sentences[j].end,
            text=" ".join(s.text for s in sentences[i:j + 1]),
            heuristic=round(value, 2),
            clip_type=dominant_type(profiles[i:j + 1]),
            source="heuristic",
        ))
    return out, profiles, scorer


def _suppress(scored: List[Tuple[float, int, int]], sentences: List[Sentence],
              limit: int, iou: float) -> List[Tuple[float, int, int]]:
    """Greedy non-maximum suppression on the time axis."""
    kept: List[Tuple[float, int, int]] = []
    spans: List[Tuple[float, float]] = []
    for value, i, j in scored:
        a, b = sentences[i].start, sentences[j].end
        clash = False
        for (x, y) in spans:
            inter = max(0.0, min(b, y) - max(a, x))
            union = max(b, y) - min(a, x)
            if union > 0 and inter / union > iou:
                clash = True
                break
        if clash:
            continue
        kept.append((value, i, j))
        spans.append((a, b))
        if len(kept) >= limit:
            break
    return kept


def hotspots_in_block(candidates: List[Candidate], start_idx: int, end_idx: int,
                      top: int = 6) -> List[Candidate]:
    """The heuristic's best guesses inside one transcript block, for prompt hints."""
    inside = [c for c in candidates if c.start_idx >= start_idx and c.end_idx <= end_idx]
    inside.sort(key=lambda c: c.heuristic, reverse=True)
    return inside[:top]
