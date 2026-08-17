"""Configuration loading / merging / persistence."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

DEFAULTS: Dict[str, Any] = {
    "general": {
        "language": "auto",
        "cache_dir": "cache",
        "output_dir": "output",
        "ffmpeg_path": "",
        "ffprobe_path": "",
    },
    "whisper": {
        "model": "small",
        "device": "auto",
        "compute_type": "auto",
        "beam_size": 5,
        "vad_filter": True,
        "vad_min_silence_ms": 500,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "cpu_threads": 0,
    },
    "llm": {
        "backend": "ollama",
        "model": "qwen2.5:7b-instruct",
        "base_url": "http://127.0.0.1:11434",
        "api_key": "",
        "temperature": 0.25,
        "top_p": 0.9,
        "num_ctx": 8192,
        "max_tokens": 3072,
        "request_timeout": 600,
        "keep_alive": "10m",
        "unload_after_analysis": True,
        "use_json_schema": True,
        "max_retries": 3,
    },
    "clips": {
        "min_duration": 20,
        "max_duration": 90,
        "target_count": 12,
        "ideal_duration": 45,
        "block_seconds": 420,
        "block_overlap_seconds": 45,
        "max_blocks": 0,
        "max_candidates_pass1": 60,
        "boundary_optimization": True,
        "boundary_candidates": 20,
        "boundary_context_sentences": 5,
        "overlap_iou_threshold": 0.4,
        "final_pool_size": 18,
        "diversity_lambda": 0.7,
        "min_gap_between_clips": 5,
    },
    "scoring": {
        "weights": {
            "hook": 25,
            "payoff": 25,
            "emotion": 15,
            "curiosity": 15,
            "standalone": 10,
            "editability": 10,
        },
        "heuristic_blend": 0.15,
    },
    "export": {
        "mode": "copy",
        "reencode_crf": 20,
        "reencode_preset": "veryfast",
        "padding_start": 0.15,
        "padding_end": 0.35,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8420,
        "open_browser": True,
    },
}

SCORE_KEYS = ("hook", "payoff", "emotion", "curiosity", "standalone", "editability")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Thin dict wrapper with dotted access and disk persistence."""

    def __init__(self, data: Dict[str, Any], path: Path | None = None):
        self.data = data
        self.path = path or CONFIG_PATH

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        value = self.data.get(name, {})
        return value if isinstance(value, dict) else {}

    def __getitem__(self, name: str) -> Any:
        return self.data[name]

    # -- paths -------------------------------------------------------------
    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (ROOT / p)

    @property
    def cache_dir(self) -> Path:
        p = self._resolve(self.get("general.cache_dir", "cache"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir(self) -> Path:
        p = self._resolve(self.get("general.output_dir", "output"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- weights -----------------------------------------------------------
    @property
    def weights(self) -> Dict[str, float]:
        w = dict(DEFAULTS["scoring"]["weights"])
        w.update({k: float(v) for k, v in self.section("scoring").get("weights", {}).items()
                  if k in SCORE_KEYS})
        return w

    # -- persistence -------------------------------------------------------
    def update(self, patch: Dict[str, Any]) -> None:
        self.data = _deep_merge(self.data, patch)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.data, fh, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else CONFIG_PATH
    user: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    return Config(_deep_merge(DEFAULTS, user), cfg_path)
