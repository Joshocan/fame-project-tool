from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from fame.exceptions import LLMTimeoutError, LLMHTTPError, format_error


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
    timeout_s: int = 500
    api_key: str = ""
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    retries: int = 3
    retry_delay_s: float = 5.0

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
        self.retries = int(os.getenv("OLLAMA_RETRIES", self.retries))
        self.retry_delay_s = float(os.getenv("OLLAMA_RETRY_DELAY", self.retry_delay_s))

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
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
                if not r.ok:
                    detail = ""
                    try:
                        detail = r.json().get("error", "")
                    except Exception:
                        detail = r.text
                    raise LLMHTTPError(self.host, self.model, r.status_code, detail)

                data = r.json()
                out = data.get("response", "")
                return (out or "").strip()
            except requests.exceptions.ReadTimeout as e:
                last_exc = LLMTimeoutError(self.host, self.model, self.timeout_s)
            except requests.exceptions.RequestException as e:
                last_exc = LLMHTTPError(self.host, self.model, -1, detail=str(e))
            except LLMHTTPError as e:
                last_exc = e

            if attempt < self.retries:
                time.sleep(self.retry_delay_s)
                continue
            raise last_exc


def assert_ollama_running(host: Optional[str] = None) -> None:
    h = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    try:
        r = requests.get(f"{h}/api/tags", timeout=500)
        if not (200 <= r.status_code < 400):
            raise RuntimeError(f"Ollama not healthy: {r.status_code}")
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Ollama is not reachable at {h}. Start Ollama first.\n"
            f"Details: {e}"
        )
