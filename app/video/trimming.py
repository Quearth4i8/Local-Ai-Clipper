"""Tighten a clip by removing filler words and shortening long pauses - a
jump-cut edit built entirely from the word-level transcript, no re-listening
needed.

Conservative by design: a pause shorter than `min_gap` is normal speech
rhythm and is never touched. Only the EXCESS above `keep_pause` on a LONGER
pause is removed - the pause itself doesn't vanish, which would read as a
jarring splice, it just gets shorter.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from ..models import Word

FILLER_WORDS = {"um", "umm", "ummm", "uh", "uhh", "uhm", "erm", "hmm", "hm", "mhm", "huh"}


def is_filler_word(text: str) -> bool:
    return text.strip(" .,!?-…").lower() in FILLER_WORDS


def plan_keep_spans(
    words: Sequence[Word],
    start: float,
    end: float,
    *,
    remove_fillers: bool = True,
    min_gap: float = 0.6,
    keep_pause: float = 0.35,
    pad: float = 0.06,
    min_span: float = 0.12,
) -> List[Tuple[float, float]]:
    """The sub-ranges of [start, end] (absolute video time) to KEEP, after
    cutting filler words and shortening pauses longer than `min_gap` down to
    `keep_pause`. Returns [(start, end)] unchanged when there is nothing to
    trim - callers can treat that as "no jump cuts needed".
    """
    in_range = sorted((w for w in words if w.end > start and w.start < end),
                      key=lambda w: w.start)
    if not in_range:
        # No word-level evidence for this span at all - trimming blind would
        # mean guessing where speech is, which is exactly backwards.
        return [(start, end)]

    cuts: List[Tuple[float, float]] = []
    if remove_fillers:
        for w in in_range:
            if is_filler_word(w.text):
                a, b = max(start, w.start - pad), min(end, w.end + pad)
                if b > a:
                    cuts.append((a, b))

    half = max(0.0, keep_pause) / 2.0
    prev_end = start
    for w in in_range:
        gap = w.start - prev_end
        if gap > min_gap:
            a, b = prev_end + half, w.start - half
            if b > a:
                cuts.append((a, b))
        prev_end = max(prev_end, w.end)
    if end - prev_end > min_gap:
        a, b = prev_end + half, end - half
        if b > a:
            cuts.append((a, b))

    if not cuts:
        return [(start, end)]

    cuts.sort()
    merged: List[Tuple[float, float]] = [cuts[0]]
    for a, b in cuts[1:]:
        pa, pb = merged[-1]
        if a <= pb + 0.02:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))

    keep: List[Tuple[float, float]] = []
    cursor = start
    for a, b in merged:
        if a - cursor >= min_span:
            keep.append((cursor, a))
        cursor = max(cursor, b)
    if end - cursor >= min_span:
        keep.append((cursor, end))

    return keep or [(start, end)]


def total_duration(spans: Sequence[Tuple[float, float]]) -> float:
    return sum(b - a for a, b in spans)


def is_noop(spans: Sequence[Tuple[float, float]], start: float, end: float) -> bool:
    return len(spans) == 1 and abs(spans[0][0] - start) < 1e-6 and abs(spans[0][1] - end) < 1e-6
