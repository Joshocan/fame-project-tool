from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaHTTP:
    """
    Simple Ollama text generation client via HTTP.

    Env:
      - OLLAMA_HOST default http://127.0.0.1:11434
      - OLLAMA_LLM_MODEL default gpt-oss:120b-cloud
    """
    model: str = "gpt-oss:120b-cloud"
    host: str = "http://127.0.0.1:11434"
    timeout_s: int = 300
    api_key: str = ""
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"

    def __post_init__(self) -> None:
        # Prefer LLM-specific host; fallback to shared OLLAMA_HOST
        self.host = os.getenv("OLLAMA_LLM_HOST", os.getenv("OLLAMA_HOST", self.host)).rstrip("/")
        self.model = os.getenv("OLLAMA_LLM_MODEL", self.model).strip()
        key = os.getenv("OLLAMA_API_KEY", "").strip()
        key_file = os.getenv("OLLAMA_API_KEY_FILE", "").strip()
        if not key and key_file:
            try:
                key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
            except Exception:
                key = ""
        self.api_key = key
        self.auth_header = os.getenv("OLLAMA_AUTH_HEADER", self.auth_header).strip() or "Authorization"
        self.auth_scheme = os.getenv("OLLAMA_AUTH_SCHEME", self.auth_scheme).strip()

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        headers: Dict[str, str] = {}
        if self.api_key:
            if self.auth_scheme:
                headers[self.auth_header] = f"{self.auth_scheme} {self.api_key}"
            else:
                headers[self.auth_header] = self.api_key

        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        out = data.get("response", "")
        return (out or "").strip()


def assert_ollama_running(host: Optional[str] = None) -> None:
    h = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    try:
        r = requests.get(f"{h}/api/tags", timeout=2)
        if not (200 <= r.status_code < 400):
            raise RuntimeError(f"Ollama not healthy: {r.status_code}")
    except Exception as e:
        raise RuntimeError(
            f"❌ Ollama is not reachable at {h}. Start Ollama first.\n"
            f"Details: {e}"
        )
