from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

import requests


class Embedder:
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


@dataclass
class OllamaEmbedder(Embedder):
    """
    Embedding via Ollama server HTTP API.
    Requires Ollama server running (local or remote).

    Env vars supported:
      - OLLAMA_HOST  (default: http://127.0.0.1:11434)
      - OLLAMA_EMBED_MODEL (default: nomic-embed-text)
    """
    model: str = "nomic-embed-text"
    host: str = "http://127.0.0.1:11434"
    timeout_s: int = 120

    def __post_init__(self) -> None:
        env_host = os.getenv("OLLAMA_HOST")
        env_model = os.getenv("OLLAMA_EMBED_MODEL")
        if env_host:
            self.host = env_host.rstrip("/")
        if env_model:
            self.model = env_model.strip()

    def _embed_one(self, text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"Ollama returned invalid embedding payload: {data}")
        return [float(x) for x in emb]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        # Simple sequential embedding (safe + easy to debug)
        out: List[List[float]] = []
        for t in texts:
            t = (t or "").strip()
            if not t:
                out.append([])
                continue
            out.append(self._embed_one(t))
        return out
