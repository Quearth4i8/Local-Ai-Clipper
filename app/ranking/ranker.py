"""Deduplication, pass-2 head-to-head ranking, and diversity selection."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..llm.base import LLMBackend, LLMError
from ..llm.json_repair import as_float, as_int, parse_list
from ..llm.prompts import PASS2_SCHEMA, SYSTEM, pass2_prompt
from ..models import Candidate, Sentence, fmt_ts

log = logging.getLogger("clipfinder.ranking")


# ---------------------------------------------------------------------------
# 1. Overlap removal / merging
# ---------------------------------------------------------------------------
def dedupe(cands: List[Candidate], sentences: List[Sentence],
           iou_threshold: float = 0.4, max_duration: float = 90.0,
           merge_ceiling: float = 0.75) -> List[Candidate]:
    """Collapse candidates that describe the same moment.

    High overlap  -> keep the better one.
    Partial overlap that still fits the duration budget -> merge into the union,
    which is what turns 01:20-02:00 / 01:24-02:08 / 01:29-02:15 into one clip.
    """
    ordered = sorted(cands, key=lambda c: (-c.overall, c.start))
    kept: List[Candidate] = []

    for cand in ordered:
        target: Optional[Candidate] = None
        best_iou = 0.0
        for existing in kept:
            iou = cand.overlap(existing)
            if iou > best_iou:
                best_iou, target = iou, existing
        if target is None or best_iou < iou_threshold:
            kept.append(cand)
            continue

        target.merged_from += 1
        union_start_idx = min(target.start_idx, cand.start_idx)
        union_end_idx = max(target.end_idx, cand.end_idx)
        union_span = sentences[union_end_idx].end - sentences[union_start_idx].start

        if best_iou < merge_ceiling and union_span <= max_duration:
            target.start_idx, target.end_idx = union_start_idx, union_end_idx
            target.start = sentences[union_start_idx].start
            target.end = sentences[union_end_idx].end
            target.text = " ".join(s.text for s in
                                   sentences[union_start_idx:union_end_idx + 1]).strip()
            target.source = "merged"
            for key in list(target.scores) or []:
                target.scores[key] = max(target.scores.get(key, 0.0),
                                         cand.scores.get(key, 0.0))
            target.overall = max(target.overall, cand.overall)
            target.heuristic = max(target.heuristic, cand.heuristic)
        # Keep the richer description if the loser had one and the winner did not.
        if not target.reason and cand.reason:
            target.reason = cand.reason
        if not target.title and cand.title:
            target.title = cand.title
        target.confidence = max(target.confidence, cand.confidence * 0.9)

    kept.sort(key=lambda c: -c.overall)
    return kept


# ---------------------------------------------------------------------------
# 2. Pass 2 - compare candidates against each other
# ---------------------------------------------------------------------------
def _excerpt(text: str, limit: int = 620) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.6)].rsplit(" ", 1)[0]
    tail = text[-int(limit * 0.35):].split(" ", 1)[-1]
    return f"{head} […] {tail}"


def second_pass(
    backend: LLMBackend,
    cands: List[Candidate],
    language: str,
    target_count: int,
    pool_size: int = 18,
    on_log: Optional[Callable[[str], None]] = None,
) -> List[Candidate]:
    pool = sorted(cands, key=lambda c: -c.overall)[:max(2, pool_size)]
    if len(pool) < 2:
        return pool

    entries: List[Dict[str, Any]] = []
    for n, c in enumerate(pool, start=1):
        entries.append({
            "id": n,
            "start_tc": fmt_ts(c.start),
            "end_tc": fmt_ts(c.end),
            "duration": c.duration,
            "type": c.clip_type,
            "score": c.overall,
            "title": c.title or "(untitled)",
            "excerpt": _excerpt(c.text),
        })

    prompt = pass2_prompt(entries, target_count, language)
    try:
        raw = backend.complete(SYSTEM, prompt, PASS2_SCHEMA, on_log=on_log)
    except LLMError as exc:
        if on_log:
            on_log(f"Pass 2 unavailable ({exc}); keeping pass-1 ranking.")
        return pool

    ranked = parse_list(raw, "ranked")
    if not ranked:
        if on_log:
            on_log("Pass 2 returned nothing parsable; keeping pass-1 ranking.")
        return pool

    by_id = {n: c for n, c in enumerate(pool, start=1)}
    kept_ids = [as_int(r.get("id"), -1) for r in ranked]

    # Collect absorptions first so a clip can't be both a keeper and a duplicate.
    # A clip is only a duplicate if it actually shares screen time with the
    # keeper - otherwise the model is just expressing a preference, and the
    # right response is to demote it, not to delete a whole distinct moment.
    absorbed: set[int] = set()
    demoted: set[int] = set()
    for row in ranked:
        rid = as_int(row.get("id"), -1)
        keeper = by_id.get(rid)
        for other in (row.get("absorbed") or []):
            oid = as_int(other, -1)
            if oid not in by_id or oid == rid:
                continue
            if keeper is not None and by_id[oid].overlap(keeper) <= 0.0:
                demoted.add(oid)
            else:
                absorbed.add(oid)
    absorbed -= {rid for rid in kept_ids[:target_count] if rid in by_id}

    out: List[Candidate] = []
    handled: set[int] = set()

    for row in ranked:
        rid = as_int(row.get("id"), -1)
        cand = by_id.get(rid)
        if cand is None or rid in absorbed or rid in handled:
            continue
        handled.add(rid)
        for other in (row.get("absorbed") or []):
            oid = as_int(other, -1)
            if oid in absorbed:
                cand.merged_from += by_id[oid].merged_from
        final = as_float(row.get("final_score", cand.overall), cand.overall)
        if 0 < final <= 10 and cand.overall > 20:   # model answered on a 0-10 scale
            final *= 10
        # Trust pass 2, but do not let one hallucinated number erase pass 1.
        cand.overall = round(max(0.0, min(100.0, 0.65 * final + 0.35 * cand.overall)), 1)
        new_title = str(row.get("title", "") or "").strip().strip('"')
        if len(new_title) > 4:
            cand.title = new_title[:120]
        verdict = str(row.get("verdict", "") or "").strip()
        if verdict:
            # Kept separate from `reason`: pass 2 judges the clip against its
            # rivals, so its wording can legitimately disagree with pass 1.
            cand.verdict = verdict[:400]
            if not cand.reason:
                cand.reason = verdict
        new_type = str(row.get("clip_type", "") or "").strip()
        if new_type:
            from ..scoring.scorer import normalise_type
            cand.clip_type = normalise_type(new_type)
        cand.confidence = min(1.0, cand.confidence + 0.1)
        out.append(cand)

    # Anything the model silently dropped, or wanted gone without an overlap to
    # justify it, still deserves a place at the bottom of the list.
    for n, c in by_id.items():
        if n not in absorbed and n not in handled:
            c.overall = round(c.overall * (0.80 if n in demoted else 0.92), 1)
            out.append(c)
    out.sort(key=lambda c: -c.overall)
    return out


# ---------------------------------------------------------------------------
# 3. Diversity-aware final selection
# ---------------------------------------------------------------------------
def _similarity(a: Candidate, b: Candidate, video_duration: float) -> float:
    sim = 0.0
    if a.clip_type == b.clip_type:
        sim += 0.55
    gap = abs((a.start + a.end) / 2 - (b.start + b.end) / 2)
    span = max(1.0, video_duration)
    sim += 0.45 * max(0.0, 1.0 - gap / (span * 0.25))
    return min(1.0, sim)


def select_diverse(cands: List[Candidate], target_count: int,
                   video_duration: float, lam: float = 0.7,
                   min_gap: float = 5.0) -> List[Candidate]:
    """Maximal-marginal-relevance pick: quality first, variety as the tie-breaker."""
    pool = sorted(cands, key=lambda c: -c.overall)
    if not pool:
        return []
    lam = max(0.0, min(1.0, lam))
    chosen: List[Candidate] = [pool[0]]
    remaining = pool[1:]

    while remaining and len(chosen) < target_count:
        best, best_value = None, -1e9
        for cand in remaining:
            if any(cand.overlap(c) > 0.05 for c in chosen):
                continue
            if any(cand.start < c.end + min_gap and c.start < cand.end + min_gap
                   for c in chosen):
                continue
            penalty = max(_similarity(cand, c, video_duration) for c in chosen)
            value = lam * (cand.overall / 100.0) - (1.0 - lam) * penalty
            if value > best_value:
                best, best_value = cand, value
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)

    chosen.sort(key=lambda c: -c.overall)
    for n, c in enumerate(chosen, start=1):
        c.rank = n
    return chosen


# ---------------------------------------------------------------------------
# 4. Compilation ordering - arrange several clips into one retention-optimised cut
# ---------------------------------------------------------------------------
def _virality_of(clip: Dict[str, Any]) -> float:
    v = clip.get("virality")
    return float(v) if v else float(clip.get("score", 0.0))


def _avoid_same_type_adjacent(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Swap a clip forward when it shares a type with the one right before it -
    back-to-back "confrontation" clips feel repetitive even when both are strong."""
    out = list(clips)
    for i in range(1, len(out)):
        if out[i].get("type") != out[i - 1].get("type"):
            continue
        for j in range(i + 1, len(out)):
            if out[j].get("type") != out[i - 1].get("type"):
                out[i], out[j] = out[j], out[i]
                break
    return out


def arrange_for_retention(clips: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order a set of already-selected clips into the shape of one compilation
    video built to hold attention start to finish.

    Opens on the single most attention-grabbing clip (the scroll-stopper),
    closes on the biggest payoff (the reward for watching to the end), and
    zig-zags the middle between higher- and lower-energy moments instead of
    letting virality decline in a straight line, which is what causes viewers
    to drop off partway through a compilation.
    """
    pool = list(clips)
    if len(pool) <= 2:
        return sorted(pool, key=lambda c: -_virality_of(c))

    ranked = sorted(pool, key=lambda c: -_virality_of(c))
    opener, rest = ranked[0], ranked[1:]
    closer = max(rest, key=lambda c: (c.get("scores") or {}).get("payoff", 0.0))
    rest = [c for c in rest if c is not closer]

    rest.sort(key=lambda c: -_virality_of(c))
    middle: List[Dict[str, Any]] = []
    lo, hi = 0, len(rest) - 1
    take_high = True
    while lo <= hi:
        if take_high:
            middle.append(rest[lo]); lo += 1
        else:
            middle.append(rest[hi]); hi -= 1
        take_high = not take_high

    return [opener, *_avoid_same_type_adjacent(middle), closer]


def type_breakdown(cands: Sequence[Candidate]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cands:
        out[c.clip_type] = out.get(c.clip_type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
