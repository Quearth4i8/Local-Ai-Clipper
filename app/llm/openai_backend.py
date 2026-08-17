"""OpenAI-compatible LOCAL servers: llama.cpp `server`, LM Studio, vLLM, TGW.

This talks to a server you run on your own machine. It is here so you can swap
inference engines, not so you can call a cloud service - point base_url at
localhost and leave api_key empty.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .base import LLMBackend, LLMError


class OpenAICompatibleBackend(LLMBackend):
    name = "openai_compatible"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.base_url = (self.base_url or "http://127.0.0.1:8080/v1").rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = str(cfg.get("api_key", "") or "")

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=8)
            resp.raise_for_status()
            models = [m.get("id", "") for m in resp.json().get("data", [])]
        except requests.RequestException as exc:
            return {
                "ok": False, "backend": self.name, "url": self.base_url, "models": [],
                "error": f"Cannot reach local server at {self.base_url} ({exc.__class__.__name__}).",
            }
        return {
            "ok": True, "backend": self.name, "url": self.base_url,
            "models": models, "model": self.model,
            "model_installed": (not models) or (self.model in models),
            "error": None,
        }

    def _complete(self, system: str, user: str,
                  schema: Optional[Dict[str, Any]]) -> str:
        payload: Dict[str, Any] = {
            "model": self.model or "local-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "clip_analysis", "strict": True, "schema": schema},
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                 headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"Local LLM request failed: {exc}") from exc

        if resp.status_code >= 400 and "response_format" in payload:
            payload.pop("response_format", None)  # server may not support it at all
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                 headers=self._headers(), timeout=self.timeout)
        if resp.status_code >= 400:
            raise LLMError(f"Local LLM HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Local LLM returned no choices")
        return (choices[0].get("message") or {}).get("content", "") or ""
