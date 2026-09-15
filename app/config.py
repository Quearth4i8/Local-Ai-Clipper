"""Configuration loading / merging / persistence."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

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
        # Re-listen to stretches Whisper returned nothing for but that still
        # have audio in them. Whisper (turbo especially) sometimes skips whole
        # passages, which shows up as a clip with no captions over speech.
        "fill_gaps": True,
        "gap_min_seconds": 2.5,
        "gap_noise_db": -38.0,
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
        "encoder": "auto",
    },
    "reframe": {
        # Which aspect ratios "Export video clips" produces.
        "formats": ["9:16"],
        "layout": "crop",        # crop = track the subject | fit_blur = whole frame on a blurred bed
        "sample_fps": 3.0,       # frames per second analysed for the subject
        "smooth": 0.35,          # 0..1, higher = snappier camera
        "move_threshold": 0.035, # below this the shot gets one static crop
        "blur_strength": 28,
    },
    "captions": {
        "enabled": True,
        "font": "Arial Rounded MT Bold",
        "fonts_dir": "assets/fonts",
        "font_size_ratio": 0.070,
        "base_color": "#FFFFFF",
        "highlight_color": "#22C55E",
        "outline_color": "#000000",
        "highlight_outline_color": "",
        "outline_ratio": 0.09,
        "shadow_ratio": 0.05,
        "position": "bottom",
        "margin_v_ratio": 0.16,
        "max_words": 4,
        "max_chars": 0,
        "max_duration": 2.4,
        "uppercase": True,
        "strip_punctuation": True,
        "highlight_scale": 118,
        "animation_ms": 130,
        "fade_ms": 90,
        "time_offset": -0.05,
        "lead_in_max": 0.12,
        "tail_hold": 0.10,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8420,
        "open_browser": True,
    },
    "campaign": {
        # Free-text brief (naming rules, tone, required mentions/hashtags, ...)
        # the metadata generator is instructed to follow when exporting clips.
        "rules": "",
        "generate_metadata": False,
    },
    "watermark": {
        # Optional logo/bug burned into the top of exported clips - only used
        # when a campaign requires one. Off and pathless by default.
        "enabled": False,
        "path": "",
        "opacity": 0.85,       # 0..1
        "scale": 0.18,         # watermark width, as a fraction of the output frame width
        "margin": 0.04,        # gap from the frame edge, as a fraction of min(width, height)
        "position": "top_right",   # top_right | top_left | bottom_right | bottom_left | center
    },
    "tightening": {
        # Jump-cut editing: shorten long pauses and drop filler words ("um",
        # "uh"...) using the word-level transcript already captured by Whisper.
        # A pause under min_gap is never touched - only the excess above
        # keep_pause on a LONGER pause is removed, so pacing stays natural.
        "enabled": False,
        "remove_fillers": True,
        "min_gap": 0.6,        # seconds - pauses shorter than this are left alone
        "keep_pause": 0.35,    # seconds - how much of a long pause survives the cut
    },
    "thumbnail": {
        # Auto-generated cover image per exported clip: the strongest detected
        # frame (usually the best face shot) with the hook line burned on top.
        "enabled": False,
        "text_color": "#FFFFFF",
        "outline_color": "#000000",
        "uppercase": True,
    },
    "music": {
        # Background music, mixed in quietly under the clip's own audio - off
        # and pathless by default. The track loops if shorter than the clip
        # and stays continuous across a tightened clip's internal jump cuts.
        "enabled": False,
        "path": "",
        "volume": 0.15,        # 0..1, loudness of the music relative to itself (not a mix ratio)
        "fade_seconds": 0.6,   # fade-in/out at the very start/end of the clip
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

    @property
    def performance_db(self) -> Path:
        """Local SQLite file tracking how exported clips actually performed."""
        return self.cache_dir / "performance.sqlite3"

    @property
    def fonts_dir(self) -> Optional[Path]:
        """Drop-in folder for caption fonts; None when it holds no font files."""
        raw = self.get("captions.fonts_dir", "")
        if not raw:
            return None
        p = self._resolve(raw)
        if p.is_dir() and any(p.glob("*.[ot]t[fc]")):
            return p
        return None

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
        self._sanitize()

    def _sanitize(self) -> None:
        """Drop anything that does not belong, so a bad write can't accumulate.

        scoring.weights in particular must only ever hold the six known keys
        with numeric values.
        """
        # reframe.formats must be a non-empty list of known aspect-ratio ids.
        # A stray value here reaches the browser and then fails request
        # validation on export, which is a confusing way to find a typo.
        rf = self.data.get("reframe")
        if isinstance(rf, dict):
            from .video.reframe import FORMATS
            raw = rf.get("formats")
            if not isinstance(raw, list):
                raw = [raw]
            clean: list = []
            for item in raw:
                key = str(item).strip()
                if key in FORMATS and key not in clean:
                    clean.append(key)
            rf["formats"] = clean or list(DEFAULTS["reframe"]["formats"])
            if str(rf.get("layout", "crop")) not in ("crop", "fit_blur"):
                rf["layout"] = "crop"

        wm = self.data.get("watermark")
        if isinstance(wm, dict):
            positions = ("top_left", "top_right", "bottom_left", "bottom_right", "center")
            if str(wm.get("position", "top_right")) not in positions:
                wm["position"] = "top_right"
            for key, lo, hi in (("opacity", 0.05, 1.0), ("scale", 0.02, 0.9), ("margin", 0.0, 0.4)):
                try:
                    wm[key] = max(lo, min(hi, float(wm.get(key, DEFAULTS["watermark"][key]))))
                except (TypeError, ValueError):
                    wm[key] = DEFAULTS["watermark"][key]

        mu = self.data.get("music")
        if isinstance(mu, dict):
            for key, lo, hi in (("volume", 0.0, 1.0), ("fade_seconds", 0.0, 2.0)):
                try:
                    mu[key] = max(lo, min(hi, float(mu.get(key, DEFAULTS["music"][key]))))
                except (TypeError, ValueError):
                    mu[key] = DEFAULTS["music"][key]

        tg = self.data.get("tightening")
        if isinstance(tg, dict):
            for key, lo, hi in (("min_gap", 0.15, 5.0), ("keep_pause", 0.0, 3.0)):
                try:
                    tg[key] = max(lo, min(hi, float(tg.get(key, DEFAULTS["tightening"][key]))))
                except (TypeError, ValueError):
                    tg[key] = DEFAULTS["tightening"][key]
            # A kept pause longer than the trigger threshold would be a no-op
            # dressed up as a feature - keep it strictly shorter.
            if tg["keep_pause"] >= tg["min_gap"]:
                tg["keep_pause"] = max(0.0, tg["min_gap"] - 0.1)

        weights = self.data.get("scoring", {}).get("weights")
        if isinstance(weights, dict):
            clean = {}
            for key, value in weights.items():
                if key not in SCORE_KEYS:
                    continue
                try:
                    clean[key] = float(value)
                except (TypeError, ValueError):
                    clean[key] = float(DEFAULTS["scoring"]["weights"][key])
            for key in SCORE_KEYS:
                clean.setdefault(key, float(DEFAULTS["scoring"]["weights"][key]))
            self.data["scoring"]["weights"] = clean

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
    cfg = Config(_deep_merge(DEFAULTS, user), cfg_path)
    cfg._sanitize()
    return cfg
