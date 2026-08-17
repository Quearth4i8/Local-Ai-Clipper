"""Turn raw Whisper output into clean, natural speech units.

Every clip boundary in this app lands on a Sentence edge, which is what keeps
the exported clips from starting or ending mid-word.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..models import Sentence, Transcript, Word

SENTENCE_END = re.compile(r'[.!?…。！？؟]+["\'»”\)\]]*$')
CLAUSE_END = re.compile(r'[,;:—–-]$')

# Abbreviations that end in a period but do not end a sentence.
ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "inc.", "ltd.", "co.", "no.", "fig.", "approx.",
    "m.", "mme.", "mlle.", "dr.", "pr.", "cf.", "ex.", "env.", "art.",
}

DEFAULTS = dict(
    pause_break=0.65,      # a gap this long is treated as a real boundary
    hard_pause=1.10,       # always break here
    min_words=4,           # never emit a fragment shorter than this...
    max_words=48,          # ...and never let one run longer than this
    min_duration=0.8,
)


def _is_sentence_end(token: str) -> bool:
    if not SENTENCE_END.search(token):
        return False
    return token.lower() not in ABBREV


def _flatten_words(transcript: Transcript) -> List[Word]:
    words: List[Word] = []
    for seg in transcript.raw_segments:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                text = (w.get("text") or "").strip()
                if text:
                    words.append(Word(start=float(w["start"]), end=float(w["end"]),
                                      text=text, prob=float(w.get("prob", 1.0))))
        else:
            # No word timestamps (word_timestamps disabled or model gave none):
            # spread the segment's words evenly across its span.
            tokens = (seg.get("text") or "").split()
            if not tokens:
                continue
            start, end = float(seg["start"]), float(seg["end"])
            step = (end - start) / len(tokens)
            for i, tok in enumerate(tokens):
                words.append(Word(start=start + i * step,
                                  end=start + (i + 1) * step, text=tok))
    words.sort(key=lambda w: w.start)
    return words


def build_sentences(transcript: Transcript, **overrides) -> List[Sentence]:
    opts = {**DEFAULTS, **overrides}
    words = _flatten_words(transcript)
    if not words:
        return []

    sentences: List[Sentence] = []
    buf: List[Word] = []

    def flush(force: bool = False):
        nonlocal buf
        if not buf:
            return
        text = " ".join(w.text for w in buf).strip()
        text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
        if not text:
            buf = []
            return
        # Merge runts into the previous sentence rather than emitting noise.
        too_small = (len(buf) < 2 and (buf[-1].end - buf[0].start) < 0.4)
        if too_small and sentences and not force:
            prev = sentences[-1]
            prev.text = (prev.text + " " + text).strip()
            prev.words.extend(buf)
            prev.end = buf[-1].end
        else:
            sentences.append(Sentence(
                idx=len(sentences),
                start=buf[0].start,
                end=buf[-1].end,
                text=text,
                words=list(buf),
            ))
        buf = []

    for i, word in enumerate(words):
        buf.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.start - word.end) if nxt else 999.0
        span = word.end - buf[0].start
        n = len(buf)

        if nxt is None:
            break
        if gap >= opts["hard_pause"] and n >= 2:
            flush()
        elif _is_sentence_end(word.text) and n >= opts["min_words"] and span >= opts["min_duration"]:
            flush()
        elif gap >= opts["pause_break"] and n >= opts["min_words"]:
            flush()
        elif n >= opts["max_words"]:
            # Prefer breaking on a clause marker near the cap.
            cut = 0
            for back in range(1, min(9, max(2, n - 3))):
                if CLAUSE_END.search(buf[-back].text):
                    cut = back
                    break
            if cut > 1:
                tail = buf[-(cut - 1):]
                del buf[-(cut - 1):]
                flush()
                buf = tail
            else:
                flush()
    flush(force=True)

    # Fill in the pause fields - used heavily by the editability heuristic.
    for i, s in enumerate(sentences):
        s.idx = i
        s.pause_before = round(s.start - sentences[i - 1].end, 3) if i else s.start
        s.pause_after = round(sentences[i + 1].start - s.end, 3) if i + 1 < len(sentences) else 2.0
        s.pause_before = max(0.0, s.pause_before)
        s.pause_after = max(0.0, s.pause_after)
    return sentences


# --------------------------------------------------------------------------
# Topic structure
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

STOPWORDS = set("""
the a an and or but so then that this these those there here what when where who whom which why how
is are was were be been being am do does did doing have has had having will would can could should
may might must shall i you he she it we they me him her us them my your his its our their
of in on at to for with from by as about into over after before under between out up down off again
not no nor only own same than too very just now also really actually thing things like know think
le la les un une des du de et ou mais donc car ne pas plus moins que qui quoi dont ou est sont etait
etaient avoir etre fait faire dans sur pour avec sans sous chez vers par ce cet cette ces mon ton son
notre votre leur nous vous ils elles je tu il elle on y en au aux se sa ses lui leurs tout tous toute
comme quand meme aussi tres bien alors apres avant encore deja parce
""".split())


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in STOPWORDS]


def topic_shift_scores(sentences: List[Sentence], window: int = 6) -> List[float]:
    """Lexical-cohesion dip detection (a light TextTiling).

    Returns one value per sentence: how strongly a new topic seems to start there.
    Used to bias clip starts toward the beginning of a thought.
    """
    n = len(sentences)
    if n < window * 2:
        return [0.0] * n
    toks = [set(_tokens(s.text)) for s in sentences]
    scores = [0.0] * n
    for i in range(window, n - window):
        left: set = set()
        right: set = set()
        for j in range(i - window, i):
            left |= toks[j]
        for j in range(i, i + window):
            right |= toks[j]
        if not left or not right:
            continue
        overlap = len(left & right) / min(len(left), len(right))
        scores[i] = 1.0 - overlap
    # normalise to 0..1
    hi = max(scores) or 1.0
    return [round(s / hi, 3) for s in scores]


def split_into_blocks(sentences: List[Sentence], block_seconds: float,
                      overlap_seconds: float) -> List[Tuple[int, int]]:
    """Chunk the transcript into overlapping (start_idx, end_idx) blocks for pass 1."""
    if not sentences:
        return []
    blocks: List[Tuple[int, int]] = []
    i = 0
    n = len(sentences)
    while i < n:
        t0 = sentences[i].start
        j = i
        while j + 1 < n and sentences[j + 1].end - t0 < block_seconds:
            j += 1
        blocks.append((i, j))
        if j >= n - 1:
            break
        # step back by the overlap
        target = sentences[j].end - overlap_seconds
        k = j
        while k > i and sentences[k].start > target:
            k -= 1
        i = max(i + 1, k)
    return blocks


def sentence_at(sentences: List[Sentence], t: float) -> int:
    """Index of the sentence containing (or nearest to) time t."""
    if not sentences:
        return 0
    lo, hi = 0, len(sentences) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sentences[mid].end < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def snap_range(sentences: List[Sentence], start: float, end: float) -> Tuple[int, int]:
    """Convert a raw time range to the sentence indices that cover it."""
    a = sentence_at(sentences, start)
    b = sentence_at(sentences, end)
    if b < a:
        b = a
    # If the chosen end sentence barely overlaps the range, drop it.
    if b > a and sentences[b].start > end + 0.5:
        b -= 1
    return a, b
