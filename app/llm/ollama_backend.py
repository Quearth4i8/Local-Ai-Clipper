"""Ollama backend (http://127.0.0.1:11434) - fully local, no key required."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from .base import LLMBackend, LLMError


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.base_url = (self.base_url or "http://127.0.0.1:11434").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.keep_alive = cfg.get("keep_alive", "10m")

    # ---------------------------------------------------------------- health
    def health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=8)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
        except requests.RequestException as exc:
            return {
                "ok": False,
                "backend": "ollama",
                "url": self.base_url,
                "models": [],
                "error": f"Cannot reach Ollama at {self.base_url} ({exc.__class__.__name__}). "
                         "Start it with 'ollama serve'.",
            }
        installed = self._resolve_model(models)
        return {
            "ok": True,
            "backend": "ollama",
            "url": self.base_url,
            "models": models,
            "model": self.model,
            "model_installed": installed is not None,
            "error": None if installed else
                     f"Model '{self.model}' is not installed. Run: ollama pull {self.model}",
        }

    def _resolve_model(self, models: List[str]) -> Optional[str]:
        """Accept 'qwen2.5:7b-instruct' when the tag is 'qwen2.5:7b-instruct-q4_K_M'."""
        if self.model in models:
            return self.model
        base = self.model.split(":")[0]
        for m in models:
            if m == self.model or m.startswith(self.model):
                return m
        for m in models:
            if m.split(":")[0] == base:
                return m
        return None

    # ------------------------------------------------------------ completion
    def _complete(self, system: str, user: str,
                  schema: Optional[Dict[str, Any]]) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        payload["format"] = schema if schema else "json"

        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        if resp.status_code == 400 and schema:
            # Older Ollama builds only understand format="json".
            payload["format"] = "json"
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)

        if resp.status_code == 404:
            raise LLMError(
                f"Ollama does not have model '{self.model}'. Run: ollama pull {self.model}"
            )
        if resp.status_code >= 400:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama returned a non-JSON envelope") from exc
        return (data.get("message") or {}).get("content", "") or ""

    # --------------------------------------------------------------- warmup
    def warmup(self) -> bool:
        """Ask Ollama to load the model with an empty prompt (no generation)."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "", "stream": False,
                      "keep_alive": self.keep_alive,
                      "options": {"num_ctx": self.num_ctx}},
                timeout=min(self.timeout, 600),
            )
            return resp.status_code < 400
        except requests.RequestException:
            return False   # not fatal - the first real call will load it instead

    # ---------------------------------------------------------------- unload
    def unload(self) -> None:
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=20,
            )
        except requests.RequestException:
            pass
