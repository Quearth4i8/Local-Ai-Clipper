"""Animated word-by-word captions (the "Opus Clip / CapCut" look).

Whisper already gives us per-word timestamps, which is exactly what karaoke
captions need. We turn those into an ASS subtitle file where, for every word,
one Dialogue event redraws the whole phrase with just that word scaled up and
eased into the highlight colour. libass renders it; FFmpeg burns it in.

Why one event per word instead of ASS's built-in \\k karaoke tag: \\k can only
sweep the primary colour. It cannot scale a single word, and scaling is what
makes the effect read as "alive" rather than as a subtitle track.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..models import Word

# ---------------------------------------------------------------------------
# Colour handling
# ---------------------------------------------------------------------------
def hex_to_ass(colour: str, alpha: int = 0) -> str:
    """#RRGGBB (or #AARRGGBB) -> ASS &HAABBGGRR. ASS stores colour as BGR."""
    c = str(colour or "").strip().lstrip("#")
    if len(c) == 8:                      # AARRGGBB
        alpha = int(c[0:2], 16)
        c = c[2:]
    if len(c) == 3:                      # RGB shorthand
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        c = "FFFFFF"
    r, g, b = c[0:2], c[2:4], c[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


# ---------------------------------------------------------------------------
# Phrase grouping
# ---------------------------------------------------------------------------
@dataclass
class Phrase:
    words: List[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


_PUNCT_END = re.compile(r'[.!?…:;][\'"»”\)\]]*$')


def group_words(words: Sequence[Word], max_words: int = 4,
                max_duration: float = 2.4, gap_break: float = 0.55,
                max_chars: int = 0) -> List[Phrase]:
    """Split a clip's words into short on-screen phrases.

    Breaks on: line width, word count, elapsed time, sentence-ending
    punctuation, or a real pause. Keeping phrases short is what stops the line
    reflowing wildly when the active word grows.

    `max_chars` guards the physical width: four short words and four long ones
    are very different on screen, and a word count alone lets "COMPLETELY
    UNDERSTANDABLE ARRANGEMENT" run past both edges of the frame.
    """
    phrases: List[Phrase] = []
    current = Phrase()

    def width_of(extra: str) -> int:
        return sum(len(w.text) + 1 for w in current.words) + len(extra)

    for w in words:
        text = (w.text or "").strip()
        if not text:
            continue
        if current.words:
            gap = w.start - current.words[-1].end
            too_long = len(current.words) >= max_words
            too_slow = (w.end - current.words[0].start) > max_duration
            too_wide = max_chars > 0 and width_of(text) > max_chars
            if too_long or too_slow or too_wide or gap >= gap_break:
                phrases.append(current)
                current = Phrase()
        current.words.append(w)
        if _PUNCT_END.search(text) and len(current.words) >= 2:
            phrases.append(current)
            current = Phrase()

    if current.words:
        phrases.append(current)
    return [p for p in phrases if p.words]


def fit_chars_per_line(width: int, font_size: int, margin_h: int) -> int:
    """How many characters actually fit across the frame.

    Heavy display faces average roughly 0.58 em per glyph; being a little
    conservative is fine because a too-narrow line just breaks earlier.
    """
    usable = max(1, int(width) - 2 * int(margin_h))
    per_char = max(1.0, float(font_size) * 0.58)
    return max(8, int(usable / per_char))


# ---------------------------------------------------------------------------
# ASS generation
# ---------------------------------------------------------------------------
def _ts(seconds: float) -> str:
    """ASS timestamp: H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _clean(text: str, uppercase: bool, strip_punct: bool) -> str:
    t = (text or "").strip()
    if strip_punct:
        t = re.sub(r'[,.!?;:"“”«»]+$', "", t)
    if uppercase:
        t = t.upper()
    # Braces are ASS override delimiters and would corrupt the line.
    return t.replace("{", "(").replace("}", ")")


DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "font": "Segoe UI Black",
    "font_size_ratio": 0.070,     # x min(width, height)
    "base_color": "#FFFFFF",
    "highlight_color": "#22C55E",
    "outline_color": "#000000",
    "outline_ratio": 0.09,        # x font size
    "shadow_ratio": 0.05,
    "position": "bottom",         # bottom | center
    "margin_v_ratio": 0.16,       # x height, from the chosen edge
    "max_words": 4,
    "max_chars": 0,          # 0 = fit automatically to the frame width
    "max_duration": 2.4,
    "uppercase": True,
    "strip_punctuation": True,
    "highlight_scale": 118,       # % size of the active word
    "animation_ms": 130,          # ease-in duration for scale + colour
    "fade_ms": 90,
    "highlight_outline_color": "",   # blank = same as outline_color
    # --- sync tuning -------------------------------------------------------
    "time_offset": -0.05,   # seconds added to every caption time; negative = earlier
    "lead_in_max": 0.12,    # max seconds a switch may be pulled early into a pause
    "tail_hold": 0.10,      # keep the last word lit this long after it ends
}


def word_timeline(words: Sequence[Word], lead_in_max: float = 0.12,
                  tail_hold: float = 0.10) -> List[float]:
    """When each word lights up, plus a final time for when the phrase ends.

    Returns n+1 boundaries for n words.

    Whisper's word *start* times carry the most error (measured p10..p90 spread
    of 226 ms on real audio), and a late start is what reads as "the caption is
    behind". When there is a pause between two words, both endpoints of that
    pause are estimates, so switching at the middle of the gap averages the two
    errors instead of trusting the noisier one - and it costs nothing visually,
    because nobody is speaking during the gap anyway.
    """
    n = len(words)
    if n == 0:
        return []
    times: List[float] = []
    for k, w in enumerate(words):
        if k == 0:
            # Bring the phrase up slightly before the first word is spoken.
            times.append(w.start - lead_in_max)
        else:
            gap = w.start - words[k - 1].end
            pull = min(max(gap, 0.0) * 0.5, lead_in_max)
            times.append(w.start - pull)
    times.append(words[-1].end + tail_hold)

    # Keep it strictly increasing after the shifts.
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 0.04
    return times


def build_ass(words: Sequence[Word], width: int, height: int,
              cfg: Optional[Dict[str, Any]] = None,
              time_offset: float = 0.0) -> str:
    """Render an .ass file for one clip.

    `time_offset` is subtracted from every word time, converting source-video
    timestamps into clip-relative ones.
    """
    o = {**DEFAULTS, **(cfg or {})}

    base_dim = max(1, min(int(width or 1080), int(height or 1920)))
    font_size = max(12, round(base_dim * float(o["font_size_ratio"])))
    outline = max(1, round(font_size * float(o["outline_ratio"])))
    shadow = max(0, round(font_size * float(o["shadow_ratio"])))
    margin_v = max(0, round(int(height or 1920) * float(o["margin_v_ratio"])))
    margin_h = max(0, round(int(width or 1080) * 0.06))
    alignment = 5 if str(o["position"]).lower() == "center" else 2
    if alignment == 5:
        margin_v = 0

    base_col = hex_to_ass(o["base_color"])
    hl_col = hex_to_ass(o["highlight_color"])
    out_col = hex_to_ass(o["outline_color"])
    hl_out_col = hex_to_ass(o["highlight_outline_color"] or o["outline_color"])
    scale = int(o["highlight_scale"])
    anim = max(0, int(o["animation_ms"]))
    fade = max(0, int(o["fade_ms"]))
    # animation_ms 0 (or scale 100) means no pop at all - snap straight to the
    # highlighted state.
    colour_ease = anim > 0 and scale != 100

    head = f"""[Script Info]
; Generated by Local AI Clip Finder
ScriptType: v4.00+
PlayResX: {int(width or 1080)}
PlayResY: {int(height or 1920)}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{o['font']},{font_size},{base_col},{base_col},{out_col},&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: List[str] = []
    # Cap the line by real width, not just word count, so captions never run
    # past the edges of the frame.
    max_chars = int(o.get("max_chars", 0) or 0)
    if max_chars <= 0:
        max_chars = fit_chars_per_line(int(width or 1080), font_size, margin_h)
    phrases = group_words(words, int(o["max_words"]), float(o["max_duration"]),
                          max_chars=max_chars)

    sync = float(o["time_offset"])
    lead_in_max = max(0.0, float(o["lead_in_max"]))
    tail_hold = max(0.0, float(o["tail_hold"]))

    # Build every phrase's timeline first, then make sure no two phrases can be
    # on screen at once. tail_hold + lead_in_max can easily exceed a short pause
    # between phrases, and because both draw at the same position libass would
    # render one on top of the other - unreadable.
    timelines = [word_timeline(p.words, lead_in_max, tail_hold) for p in phrases]
    for i in range(len(timelines) - 1):
        cur, nxt = timelines[i], timelines[i + 1]
        speech_end = phrases[i].words[-1].end
        # Never bring the next phrase up before the current one stops talking.
        if nxt[0] < speech_end:
            nxt[0] = min(speech_end, nxt[1] - 0.04)
        # ...and always retire the current phrase before the next appears.
        if cur[-1] > nxt[0]:
            cur[-1] = max(cur[-2] + 0.05, nxt[0] - 0.02)

    for idx, phrase in enumerate(phrases):
        rendered = [_clean(w.text, bool(o["uppercase"]), bool(o["strip_punctuation"]))
                    for w in phrase.words]
        n = len(phrase.words)
        marks = timelines[idx]

        for k, w in enumerate(phrase.words):
            # marks[k] -> marks[k+1] is when this word owns the highlight.
            start = marks[k] - time_offset + sync
            end = marks[k + 1] - time_offset + sync
            if end <= start:
                end = start + 0.08
            if end <= 0:
                continue
            start = max(0.0, start)

            # Ease no longer than the word itself, or the pop looks laggy.
            dur_ms = max(1, int((end - start) * 1000))
            ease = min(anim, max(40, int(dur_ms * 0.6)))

            parts: List[str] = []
            for i, text in enumerate(rendered):
                if not text:
                    continue
                if i == k:
                    # Colour is applied INSTANTLY and only the scale is eased.
                    # Animating the colour instead leaves the active word
                    # looking inactive for the whole ease, which swallows short
                    # words entirely ("I" lasts 160ms).
                    # Inline colour overrides take a trailing '&' - libass is
                    # lenient about it, other ASS renderers are not.
                    colour = f"\\c{hl_col}&\\3c{hl_out_col}&"
                    if colour_ease:
                        pop = (f"{colour}\\fscx100\\fscy100"
                               f"\\t(0,{ease},\\fscx{scale}\\fscy{scale})")
                    else:
                        pop = (f"{colour}\\fscx{scale}\\fscy{scale}"
                               if scale != 100 else colour)
                    parts.append(f"{{{pop}}}{text}{{\\r}}")
                else:
                    parts.append(text)
            if not parts:
                continue

            effects = ""
            if fade:
                fade_in = fade if k == 0 else 0
                fade_out = fade if k == n - 1 else 0
                if fade_in or fade_out:
                    effects = f"{{\\fad({fade_in},{fade_out})}}"

            lines.append(
                f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{effects}{' '.join(parts)}"
            )

    return head + "\n".join(lines) + "\n"


def words_in_range(words: Sequence[Word], start: float, end: float) -> List[Word]:
    """Words that overlap [start, end], clipped to the range."""
    out: List[Word] = []
    for w in words:
        if w.end <= start or w.start >= end:
            continue
        out.append(Word(start=max(w.start, start), end=min(w.end, end),
                        text=w.text, prob=w.prob))
    return out
