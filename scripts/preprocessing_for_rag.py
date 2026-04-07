#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from fame.utils.runtime import workspace
from fame.ingestion.pipeline import ingest_and_prepare
from fame.vectorization.pipeline import index_all_chunks
from fame.services.chroma_service import start_chroma
from fame.services.ollama_service import _ollama_bin, start_ollama, verify_running, pull_models


# ----------------------------
# Health checks
# ----------------------------

def chroma_healthy(host: str, port: int) -> bool:
    endpoints = [
        f"http://{host}:{port}/api/v1/heartbeat",
        f"http://{host}:{port}/api/v2/heartbeat",
        f"http://{host}:{port}/heartbeat",
        f"http://{host}:{port}/api/v1/version",
        f"http://{host}:{port}/",
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=2)
            if 200 <= r.status_code < 400:
                return True
        except Exception:
            pass
    return False


def ensure_chroma_running(chroma_path: Path, host: str, port: int, timeout_s: int) -> int:
    """
    Ensure Chroma is reachable. If not, start it and wait.
    Returns PID (0 if already running and we didn't start it).
    """
    if chroma_healthy(host, port):
        print(f"SUCCESS: Chroma already running at http://{host}:{port}")
        return 0

    print("🔧 Chroma not reachable — starting via chroma_service...")
    pid = start_chroma(
        path=str(chroma_path),
        host=host,
        port=port,
        timeout_s=timeout_s,
        force_restart=False,  # important: don't kill if user has it running elsewhere
    )
    return pid


def ensure_ollama_running(log_dir: Path, timeout_s: int) -> int:
    """
    Ensure Ollama is reachable. If not, start it.
    Returns PID (0 if already running).
    """
    try:
        verify_running()
        print("SUCCESS: Ollama already running.")
        return 0
    except Exception:
        print("🔧 Ollama not reachable — attempting to start locally...")
        # start_ollama needs local ollama binary
        pid = start_ollama(log_dir=str(log_dir), timeout_s=timeout_s, force_restart=False)
        return pid


def ensure_ollama_embed_model(model: str, host: str) -> None:
    """
    Best-effort pull of embedding model.

    If host is remote (not localhost/127.0.0.1), we avoid auto-pull because the
    model catalog is controlled by that service. In that case we simply warn.
    """
    host_lower = host.lower()
    is_local = any(h in host_lower for h in ["127.0.0.1", "localhost"])

    if not is_local:
        print(
            "WARN:  Remote Ollama host detected; skipping auto-pull of embedding model.\n"
            f"    Ensure '{model}' is available on {host} or set OLLAMA_EMBED_MODEL to a model present there."
        )
        return

    bin_path = _ollama_bin()
    if not bin_path:
        print(
            "WARN:  Cannot auto-pull embedding model because 'ollama' binary is not found.\n"
            f"    Please pull manually (local host):\n"
            f"      ollama pull {model}\n"
        )
        return

    print(f"⬇️  Ensuring Ollama embedding model is available: {model}")
    pull_models([model])


# ----------------------------
# Main pipeline
# ----------------------------

def main() -> None:
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    # ---- Config ----
    chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_timeout = int(os.getenv("CHROMA_STARTUP_TIMEOUT", "120"))
    chroma_path = Path(os.getenv("CHROMA_PATH", str(paths.vector_db))).expanduser().resolve()

    ollama_timeout = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "60"))
    ollama_log_dir = Path(os.getenv("OLLAMA_LOG_DIR", str(paths.base_dir / "data" / "ollama"))).expanduser().resolve()
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()

    # ---- Print ----
    print("\n==================== PREPROCESSING FOR RAG ====================")
    print(f"BASE_DIR           : {paths.base_dir}")
    print(f"RAW_DIR            : {paths.raw_data}")
    print(f"CHUNKS_OUT_DIR     : {paths.processed_data / 'chunks'}")
    print(f"CHROMA_MODE        : {os.getenv('CHROMA_MODE', 'persistent')}")
    print(f"CHROMA_PATH        : {chroma_path}")
    print(f"CHROMA_HOST:PORT   : {chroma_host}:{chroma_port}")
    print(f"OLLAMA_HOST        : {os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}")
    print(f"OLLAMA_EMBED_MODEL : {embed_model}")
    print("===============================================================\n")

    # ---- Ensure services ----
    chroma_pid = ensure_chroma_running(chroma_path, chroma_host, chroma_port, chroma_timeout)
    ollama_pid = ensure_ollama_running(ollama_log_dir, ollama_timeout)
    ensure_ollama_embed_model(embed_model, os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))

    # ---- Ingestion ----
    print("\n🧩 Running ingestion pipeline...")
    ingest_res = ingest_and_prepare(raw_dir=paths.raw_data, out_dir=paths.processed_data / "chunks")
    print(f"SUCCESS: Ingestion done: processed={len(ingest_res['processed'])}, skipped={len(ingest_res['skipped'])}")

    if not ingest_res["processed"]:
        print("WARN:  No chunks.json produced. Ensure there is at least one PDF in data/raw.")
        sys.exit(2)

    # ---- Vectorization ----
    print("\n🧠 Running vectorization (indexing) pipeline...")
    vec_res = index_all_chunks(
        chunks_dir=paths.processed_data / "chunks",
        batch_size=int(os.getenv("VEC_BATCH_SIZE", "24")),
        collection_prefix=os.getenv("COLLECTION_PREFIX", ""),
    )
    print("SUCCESS: Vectorization done.")
    print(vec_res)

    print("\nSUCCESS: PREPROCESSING FOR RAG COMPLETE.")
    if chroma_pid:
        print(f"Chroma PID started by this run: {chroma_pid}")
    if ollama_pid:
        print(f"Ollama PID started by this run: {ollama_pid}")


if __name__ == "__main__":
    main()
