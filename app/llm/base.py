"""Local LLM backend abstraction.

Two backends ship with the app:
  * ollama            - the Ollama server (default)
  * openai_compatible - llama.cpp server, LM Studio, vLLM, text-generation-webui

Adding another backend = subclass LLMBackend and register it in get_backend().
No cloud provider is referenced anywhere.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("clipfinder.llm")


class LLMError(RuntimeError):
    pass


class LLMBackend(ABC):
    name = "base"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model = cfg.get("model", "")
        self.base_url = str(cfg.get("base_url", "")).rstrip("/")
        self.temperature = float(cfg.get("temperature", 0.25))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.num_ctx = int(cfg.get("num_ctx", 8192))
        self.max_tokens = int(cfg.get("max_tokens", 3072))
        self.timeout = int(cfg.get("request_timeout", 600))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.use_schema = bool(cfg.get("use_json_schema", True))

    # -- to implement -------------------------------------------------------
    @abstractmethod
    def _complete(self, system: str, user: str,
                  schema: Optional[Dict[str, Any]]) -> str: ...

    @abstractmethod
    def health(self) -> Dict[str, Any]: ...

    def unload(self) -> None:
        """Optional: release the model from VRAM."""

    def warmup(self) -> bool:
        """Force the model into VRAM before the first real call.

        Loading a 7B model off disk takes 20-90s and would otherwise be hidden
        inside the first analysis call, looking exactly like a frozen progress bar.
        Returns True if a load was actually triggered.
        """
        return False

    # -- shared -------------------------------------------------------------
    def complete(self, system: str, user: str,
                 schema: Optional[Dict[str, Any]] = None,
                 on_log: Optional[Callable[[str], None]] = None) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                started = time.time()
                text = self._complete(system, user, schema if self.use_schema else None)
                log.debug("LLM call %.1fs, %d chars out", time.time() - started, len(text or ""))
                if text and text.strip():
                    return text
                last_error = LLMError("empty response")
            except Exception as exc:  # noqa: BLE001 - surface anything the server does
                last_error = exc
                msg = f"LLM attempt {attempt}/{self.max_retries} failed: {exc}"
                log.warning(msg)
                if on_log and attempt == 1:
                    on_log(msg)
            if attempt < self.max_retries:
                time.sleep(min(4.0, 1.5 * attempt))
        raise LLMError(str(last_error) if last_error else "LLM call failed")


def get_backend(cfg: Dict[str, Any]) -> LLMBackend:
    backend = str(cfg.get("backend", "ollama")).lower().replace("-", "_")
    if backend in ("ollama",):
        from .ollama_backend import OllamaBackend
        return OllamaBackend(cfg)
    if backend in ("openai_compatible", "llamacpp", "llama_cpp", "lmstudio", "vllm", "openai"):
        from .openai_backend import OpenAICompatibleBackend
        return OpenAICompatibleBackend(cfg)
    raise LLMError(
        f"Unknown llm.backend '{backend}'. Use 'ollama' or 'openai_compatible'."
    )


def list_local_models(cfg: Dict[str, Any]) -> List[str]:
    try:
        return get_backend(cfg).health().get("models", [])
    except Exception:
        return []
