"""Defensive JSON extraction for local LLM output.

7B models running at q4 will occasionally emit a stray comment, a trailing
comma, a smart quote, or simply run out of tokens mid-array. None of that
should kill a 40-minute analysis run, so we salvage what we can.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_LINE_COMMENT = re.compile(r"(^|\s)//[^\n\"]*$", re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SMART = {"“": '"', "”": '"', "‘": "'", "’": "'",
          "«": '"', "»": '"', "–": "-", "—": "-"}


def _strip_fences(text: str) -> str:
    m = _FENCE.search(text)
    return m.group(1) if m else text


def _balanced_slice(text: str, opener: str, closer: str) -> Optional[str]:
    start = text.find(opener)
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # truncated - caller will try to salvage


def _basic_fixes(text: str) -> str:
    for bad, good in _SMART.items():
        text = text.replace(bad, good)
    text = _BLOCK_COMMENT.sub("", text)
    text = _LINE_COMMENT.sub(r"\1", text)
    text = _TRAILING_COMMA.sub(r"\1", text)
    # Bare NaN / Infinity / python literals
    text = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "0", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def _salvage_objects(text: str) -> List[Dict[str, Any]]:
    """Pull complete {...} objects out of a truncated response.

    A cut-off `{"candidates":[{...},{...}` never closes its outer brace, so if
    nothing is found at the top level we unwrap one array layer and retry.
    """
    out = _objects_at_top_level(text)
    if out:
        return out
    cursor = 0
    for _ in range(3):
        idx = text.find("[", cursor)
        if idx == -1:
            break
        out = _objects_at_top_level(text[idx + 1:])
        if out:
            return out
        cursor = idx + 1
    return []


def _objects_at_top_level(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    depth, in_str, esc, start = 0, False, False, -1
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = text[start:i + 1]
                for attempt in (chunk, _basic_fixes(chunk)):
                    try:
                        obj = json.loads(attempt)
                        if isinstance(obj, dict):
                            out.append(obj)
                        break
                    except json.JSONDecodeError:
                        continue
                start = -1
    return out


def parse_json(text: str, expect_key: Optional[str] = None) -> Optional[Any]:
    """Best-effort parse. Returns None only when nothing usable is recoverable."""
    if not text:
        return None
    raw = _strip_fences(text.strip())

    for attempt in (raw, _basic_fixes(raw)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        sliced = _balanced_slice(raw, opener, closer)
        if not sliced:
            continue
        for attempt in (sliced, _basic_fixes(sliced)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    # Last resort: harvest whatever complete objects exist.
    objects = _salvage_objects(_basic_fixes(raw))
    if objects:
        if expect_key:
            merged: List[Dict[str, Any]] = []
            for obj in objects:
                if expect_key in obj and isinstance(obj[expect_key], list):
                    merged.extend(x for x in obj[expect_key] if isinstance(x, dict))
            if merged:
                return {expect_key: merged}
            return {expect_key: objects}
        return objects
    return None


def parse_list(text: str, key: str) -> List[Dict[str, Any]]:
    """Parse a response expected to be {key: [ {...}, ... ]} (or a bare list)."""
    data = parse_json(text, expect_key=key)
    if data is None:
        return []
    if isinstance(data, dict):
        for candidate_key in (key, "candidates", "clips", "results", "items", "moments"):
            value = data.get(candidate_key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        # single object that looks like one item
        if any(k in data for k in ("start", "start_index", "hook", "id")):
            return [data]
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            value = value.strip().split()[0].replace(",", ".")
            value = re.sub(r"[^0-9.\-]", "", value) or default
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    return int(round(as_float(value, default)))
