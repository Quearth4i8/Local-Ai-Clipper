"""Pass 1 (discover + score) and boundary optimisation."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..candidates.generator import WindowScorer
from ..candidates.heuristics import is_filler, is_transition
from ..llm.base import LLMBackend, LLMError
from ..llm.json_repair import as_float, as_int, parse_json, parse_list
from ..llm.prompts import (BOUNDARY_SCHEMA, PASS1_SCHEMA, SYSTEM, boundary_prompt,
                           pass1_prompt)
from ..models import CLIP_TYPES, Candidate, Sentence

log = logging.getLogger("clipfinder.scoring")

# The rubric maxima the model is asked to score against.
CATEGORY_MAX = {"hook": 25, "payoff": 25, "emotion": 15,
                "curiosity": 15, "standalone": 10, "editability": 10}


def normalise_type(value: Any) -> str:
    t = str(value or "other").strip().lower().replace(" ", "_").replace("-", "_")
    if t in CLIP_TYPES:
        return t
    aliases = {
        "humor": "funny", "humour": "funny", "comedy": "funny", "joke": "funny",
        "drole": "funny", "education": "educational", "educative": "educational",
        "informative": "educational", "tip": "practical_advice", "tips": "practical_advice",
        "advice": "practical_advice", "conseil": "practical_advice",
        "controversial_opinion": "controversial", "hot_take": "controversial",
        "debate": "confrontation", "argument": "confrontation", "clash": "confrontation",
        "personal_story": "story", "anecdote": "story", "histoire": "story",
        "emotion": "emotional", "inspiring": "inspirational", "motivation": "inspirational",
        "surprise": "surprising", "shocking": "surprising", "twist": "reveal",
        "revelation": "reveal", "insight": "educational",
    }
    for key, mapped in aliases.items():
        if key in t:
            return mapped
    return "other"


def compute_overall(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Rescale per-category values onto the configured weights (always /100)."""
    total = 0.0
    for key, cap in CATEGORY_MAX.items():
        raw = max(0.0, min(float(cap), float(scores.get(key, 0.0))))
        total += (raw / cap) * float(weights.get(key, cap))
    scale = sum(weights.values()) or 100.0
    return round(total * 100.0 / scale, 2)


def blend_scores(llm_score: float, heuristic: float, blend: float) -> float:
    blend = max(0.0, min(1.0, blend))
    return round((1.0 - blend) * llm_score + blend * heuristic, 2)


# Clip types that reliably pull replies, quote-posts and "well actually"
# comments - the engagement rage bait runs on - beyond what their raw scores
# already capture.
VIRALITY_TYPE_BONUS = {
    "controversial": 12.0, "confrontation": 12.0, "reveal": 8.0,
    "surprising": 8.0, "emotional": 5.0, "funny": 4.0,
}


def compute_virality(scores: Dict[str, float], clip_type: str) -> float:
    """How hard a clip grabs a scroll and how likely it is to get people
    replying/arguing - a different axis than `overall`, which rewards a clean,
    well-rounded clip. This leans on emotion and curiosity (what makes someone
    stop and comment) more than editability or standalone-ness, and adds a
    bonus for clip types that are divisive by nature."""
    hook_n = scores.get("hook", 0.0) / CATEGORY_MAX["hook"] * 100
    payoff_n = scores.get("payoff", 0.0) / CATEGORY_MAX["payoff"] * 100
    emotion_n = scores.get("emotion", 0.0) / CATEGORY_MAX["emotion"] * 100
    curiosity_n = scores.get("curiosity", 0.0) / CATEGORY_MAX["curiosity"] * 100
    base = 0.30 * hook_n + 0.20 * payoff_n + 0.28 * emotion_n + 0.22 * curiosity_n
    bonus = VIRALITY_TYPE_BONUS.get(clip_type, 0.0)
    return round(min(100.0, base + bonus), 1)


# ---------------------------------------------------------------------------
# Index / duration hygiene
# ---------------------------------------------------------------------------
def _fit_duration(sentences: List[Sentence], i: int, j: int,
                  min_d: float, max_d: float) -> Optional[Tuple[int, int]]:
    """Nudge a span onto sentence edges until it respects the duration limits."""
    n = len(sentences)
    i = max(0, min(n - 1, i))
    j = max(0, min(n - 1, j))
    if j < i:
        i, j = j, i

    guard = 0
    while sentences[j].end - sentences[i].start < min_d and guard < 400:
        guard += 1
        # grow on whichever side has the smaller pause (keeps cuts natural)
        grow_end = j + 1 < n
        grow_start = i > 0
        if grow_end and grow_start:
            if sentences[j].pause_after <= sentences[i].pause_before:
                j += 1
            else:
                i -= 1
        elif grow_end:
            j += 1
        elif grow_start:
            i -= 1
        else:
            break

    guard = 0
    while sentences[j].end - sentences[i].start > max_d and guard < 400 and j > i:
        guard += 1
        # trim from whichever end is weaker: prefer trimming the tail
        if sentences[j].pause_before >= sentences[i].pause_after:
            j -= 1
        else:
            i += 1

    span = sentences[j].end - sentences[i].start
    if span < min_d * 0.6 or span > max_d * 1.35:
        return None
    return i, j


def build_candidate(sentences: List[Sentence], i: int, j: int) -> Candidate:
    return Candidate(
        start_idx=i, end_idx=j,
        start=sentences[i].start, end=sentences[j].end,
        text=" ".join(s.text for s in sentences[i:j + 1]).strip(),
    )


def trim_edges(cand: Candidate, sentences: List[Sentence], min_duration: float,
               max_trim: int = 3) -> bool:
    """Deterministically shave dead weight off both ends of a clip.

    The LLM reliably finds the right *moment* but often leaves the host's
    "anyway, let's move on" attached to the tail, or opens on a bare "yeah".
    Those are unambiguous, so a rule handles them better than another LLM call.
    """
    i, j = cand.start_idx, cand.end_idx
    changed = False

    for _ in range(max_trim):
        if j <= i:
            break
        text = sentences[j].text
        if not (is_transition(text) or is_filler(text)):
            break
        if sentences[j - 1].end - sentences[i].start < min_duration:
            break
        j -= 1
        changed = True

    for _ in range(max_trim):
        if i >= j:
            break
        if not is_filler(sentences[i].text):
            break
        if sentences[j].end - sentences[i + 1].start < min_duration:
            break
        i += 1
        changed = True

    if changed:
        cand.start_idx, cand.end_idx = i, j
        cand.start, cand.end = sentences[i].start, sentences[j].end
        cand.text = " ".join(s.text for s in sentences[i:j + 1]).strip()
    return changed


def apply_padding(cand: Candidate, sentences: List[Sentence],
                  lead: float = 0.25, tail: float = 0.40) -> None:
    """Breathe a little air into the cut without eating a neighbour's speech."""
    s0, s1 = sentences[cand.start_idx], sentences[cand.end_idx]
    cand.start = max(0.0, s0.start - min(lead, s0.pause_before * 0.6))
    cand.end = s1.end + min(tail, max(0.0, s1.pause_after) * 0.6)


# ---------------------------------------------------------------------------
# PASS 1
# ---------------------------------------------------------------------------
def analyze_block(
    backend: LLMBackend,
    sentences: List[Sentence],
    block: Tuple[int, int],
    language: str,
    clips_cfg: Dict[str, Any],
    weights: Dict[str, float],
    hints: Sequence[Candidate] = (),
    window_scorer: Optional[WindowScorer] = None,
    heuristic_blend: float = 0.15,
    on_log: Optional[Callable[[str], None]] = None,
    max_results: int = 5,
) -> List[Candidate]:
    start_idx, end_idx = block
    min_d = float(clips_cfg.get("min_duration", 20))
    max_d = float(clips_cfg.get("max_duration", 90))

    prompt = pass1_prompt(sentences, start_idx, end_idx, min_d, max_d, language,
                          hints=hints, max_results=max_results)
    try:
        raw = backend.complete(SYSTEM, prompt, PASS1_SCHEMA, on_log=on_log)
    except LLMError as exc:
        if on_log:
            on_log(f"Block {start_idx}-{end_idx}: LLM failed ({exc}); using heuristics only.")
        return []

    items = parse_list(raw, "candidates")
    if not items and on_log:
        on_log(f"Block {start_idx}-{end_idx}: no parsable candidates returned.")

    out: List[Candidate] = []
    for item in items:
        i = as_int(item.get("start_index", item.get("start", -1)), -1)
        j = as_int(item.get("end_index", item.get("end", -1)), -1)
        if i < 0 or j < 0:
            continue
        # Model occasionally answers with seconds instead of indices.
        if i > len(sentences) or j > len(sentences):
            continue
        # Keep it inside the block it was shown (allow 2 lines of slack).
        if j < start_idx - 2 or i > end_idx + 2:
            continue
        fitted = _fit_duration(sentences, i, j, min_d, max_d)
        if not fitted:
            continue
        i, j = fitted

        cand = build_candidate(sentences, i, j)
        cand.scores = {k: max(0.0, min(float(CATEGORY_MAX[k]),
                                       as_float(item.get(k), 0.0)))
                       for k in CATEGORY_MAX}
        llm_score = compute_overall(cand.scores, weights)
        if window_scorer:
            cand.heuristic = round(window_scorer.score(i, j), 2)
        cand.overall = blend_scores(llm_score, cand.heuristic, heuristic_blend)
        cand.clip_type = normalise_type(item.get("clip_type"))
        cand.title = str(item.get("suggested_title", "") or "").strip().strip('"')[:120]
        cand.reason = str(item.get("reason", "") or "").strip()
        cand.hook_line = str(item.get("hook_line", "") or "").strip()[:200]
        cand.confidence = max(0.0, min(1.0, as_float(item.get("confidence", 0.7), 0.7)))
        cand.block_index = start_idx
        cand.source = "llm"
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# BOUNDARY OPTIMISATION
# ---------------------------------------------------------------------------
def optimize_boundaries(
    backend: LLMBackend,
    sentences: List[Sentence],
    cand: Candidate,
    clips_cfg: Dict[str, Any],
    window_scorer: Optional[WindowScorer] = None,
    language: str = "en",
    on_log: Optional[Callable[[str], None]] = None,
) -> Candidate:
    min_d = float(clips_cfg.get("min_duration", 20))
    max_d = float(clips_cfg.get("max_duration", 90))
    ctx = int(clips_cfg.get("boundary_context_sentences", 5))

    ctx_start = max(0, cand.start_idx - ctx)
    ctx_end = min(len(sentences) - 1, cand.end_idx + ctx)
    prompt = boundary_prompt(sentences, ctx_start, ctx_end,
                             cand.start_idx, cand.end_idx, min_d, max_d,
                             language=language, current_title=cand.title)
    try:
        raw = backend.complete(SYSTEM, prompt, BOUNDARY_SCHEMA, on_log=on_log)
    except LLMError:
        return cand

    data = parse_json(raw)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return cand

    i = as_int(data.get("start_index", cand.start_idx), cand.start_idx)
    j = as_int(data.get("end_index", cand.end_idx), cand.end_idx)
    if not (ctx_start <= i <= ctx_end and ctx_start <= j <= ctx_end) or j < i:
        return cand

    fitted = _fit_duration(sentences, i, j, min_d, max_d)
    if not fitted:
        return cand
    i, j = fitted

    # A merged or re-cut clip often no longer matches the title pass 1 wrote.
    new_title = str(data.get("title", "") or "").strip().strip('"')
    moved = (i, j) != (cand.start_idx, cand.end_idx)
    if len(new_title) > 4 and (moved or cand.source == "merged" or not cand.title):
        cand.title = new_title[:120]

    if not moved:
        cand.boundary_optimized = True
        return cand

    old_heur = cand.heuristic
    cand.start_idx, cand.end_idx = i, j
    cand.start, cand.end = sentences[i].start, sentences[j].end
    cand.text = " ".join(s.text for s in sentences[i:j + 1]).strip()
    cand.boundary_optimized = True
    if window_scorer:
        cand.heuristic = round(window_scorer.score(i, j), 2)
        # A better shape earns a small bump; a worse one a small penalty.
        delta = (cand.heuristic - old_heur) * 0.15
        cand.overall = round(max(0.0, min(100.0, cand.overall + delta)), 2)
    return cand


# ---------------------------------------------------------------------------
# Heuristic-only fallback (no LLM reachable)
# ---------------------------------------------------------------------------
def heuristic_only_scores(cand: Candidate, scorer: WindowScorer,
                          weights: Dict[str, float]) -> Candidate:
    f = scorer.parts(cand.start_idx, cand.end_idx)
    if not f:
        return cand
    cand.scores = {k: round(f.get(k, 0.0) * CATEGORY_MAX[k], 1) for k in CATEGORY_MAX}
    cand.overall = compute_overall(cand.scores, weights)
    cand.heuristic = round(scorer.score(cand.start_idx, cand.end_idx), 2)
    cand.confidence = 0.35
    cand.source = "heuristic"
    if not cand.reason:
        cand.reason = ("Selected by the offline pattern detector (no local LLM was "
                       "reachable): strong opening signal and a resolving end.")
    return cand
