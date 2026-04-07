from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


@dataclass
class JudgeClient:
    model: str
    base_url: str
    api_key_env: str
    temperature: float
    max_tokens: int
    timeout_s: int

    def _get_api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip()

    def _retry_cfg(self) -> Tuple[int, float]:
        """Return (retries, delay_seconds) for proprietary model HTTP calls."""
        retries = int(os.getenv("JUDGE_RETRIES", "3") or 3)
        delay = float(os.getenv("JUDGE_RETRY_DELAY", "5") or 5.0)
        return max(retries, 1), max(delay, 0.0)

    def _post_with_retries(self, url: str, *, headers: dict, json_payload: dict):
        """POST with simple retry on request/HTTP errors."""
        retries, delay = self._retry_cfg()
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=json_payload, timeout=self.timeout_s)
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as exc:  # network or HTTP errors
                last_exc = exc
                if attempt < retries:
                    time.sleep(delay)
                else:
                    raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Unexpected failure in _post_with_retries")

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        raise NotImplementedError
