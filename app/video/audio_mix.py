"""Background music: quietly mixed under a clip's own audio.

The music input is looped indefinitely (`aloop`) so any song length works,
then a window is trimmed out of that infinite loop at the position this
render actually needs - which lets a jump-cut clip built from several
segments (see trimming.py) pull consecutive windows and keep the music
continuous across the cuts, instead of restarting it at every segment.
"""
from __future__ import annotations

from typing import Any, Dict


def build_music_graph(music: Dict[str, Any], duration: float, input_idx: int, *,
                      offset: float = 0.0, fade_in: bool = True,
                      fade_out: bool = True) -> str:
    """Filter_complex fragment: main audio + a quiet, looped, faded music bed,
    mixed down to one output pad `[aout]`. `offset` is how far into the SONG's
    endless loop this window starts - callers stitching several segments of
    one clip pass the cumulative elapsed time so the music doesn't jump.
    """
    volume = max(0.0, min(1.0, float(music.get("volume", 0.15) or 0.15)))
    duration = max(0.05, duration)
    fade = max(0.0, min(2.0, float(music.get("fade_seconds", 0.6) or 0.6)))
    fade = min(fade, max(0.02, duration / 2 - 0.02))

    fades = []
    if fade_in:
        fades.append(f"afade=t=in:st=0:d={fade:.3f}")
    if fade_out:
        fades.append(f"afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}")
    fade_chain = ("," + ",".join(fades)) if fades else ""

    return (
        f"[0:a]volume=1.0[amain];"
        f"[{input_idx}:a]aloop=loop=-1:size=2e9,"
        f"atrim=start={offset:.3f}:end={offset + duration:.3f},"
        f"asetpts=PTS-STARTPTS{fade_chain},"
        f"volume={volume:.3f}[amus];"
        f"[amain][amus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
    )
