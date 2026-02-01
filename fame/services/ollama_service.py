from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, List


def _ollama_bin() -> str:
    p = os.getenv("OLLAMA_BIN", "").strip()
    if p and Path(p).exists():
        return p
    # fallback to PATH
    return _which("ollama") or ""


def _which(cmd: str) -> Optional[str]:
    try:
        out = subprocess.run(["bash", "-lc", f"command -v {cmd}"], capture_output=True, text=True)
        p = out.stdout.strip()
        return p if p else None
    except Exception:
        return None


def _run(args: List[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def _is_ollama_running() -> bool:
    """
    Check if ollama server responds. Ollama exposes /api/tags when running.
    """
    try:
        r = _run(["curl", "-fsS", "http://127.0.0.1:11434/api/tags"], check=False)
        return r.returncode == 0
    except Exception:
        return False


def _pkill_ollama_serve() -> None:
    """
    Kill any running 'ollama serve' processes (best-effort, macOS/Linux).
    """
    if _which("pkill"):
        _run(["pkill", "-f", "ollama serve"], check=False)
        time.sleep(1)
        # Second pass (sometimes multiple)
        _run(["pkill", "-f", "ollama serve"], check=False)
    else:
        print(" pkill not found; cannot auto-stop existing ollama serve processes.")


def stop_existing(pid_file: Path) -> None:
    """
    Stop ollama serve via pidfile, then pkill safety net.
    """
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            print(f"⚠️  Stopping previous Ollama PID from file: {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # If still alive, kill
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            except OSError:
                pass
        except Exception:
            pass
        finally:
            pid_file.unlink(missing_ok=True)

    # Safety net
    _pkill_ollama_serve()


def start_ollama(
    log_dir: str,
    timeout_s: int = 60,
    force_restart: bool = True,
) -> int:
    """
    Start ollama server and wait until it responds.
    Returns PID of background process.
    """
    
    ollama_exe = _which("ollama") or "/usr/local/bin/ollama" or "/opt/homebrew/bin/ollama"
    if not Path(ollama_exe).exists():
        raise RuntimeError(
            "Ollama executable not found.\n"
            "Install Ollama (macOS app) and set OLLAMA_BIN to the executable path, e.g.\n"
            "  export OLLAMA_BIN=/Applications/Ollama.app/Contents/MacOS/ollama"
            "Install Ollama from https://ollama.com\n"
            "Then verify in terminal: which ollama && ollama --version\n"
            f"Checked PATH and also tried: {ollama_exe}"
        )



    log_path = Path(log_dir).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "ollama_serve.log"
    pid_file = log_path / "ollama_serve.pid"

    if force_restart:
        print("🧹 Cleaning up any existing Ollama server processes...")
        stop_existing(pid_file)
        time.sleep(1)
    else:
        if _is_ollama_running():
            raise RuntimeError("Ollama appears to already be running on http://127.0.0.1:11434")

    # Start server in background
    log_file.write_text("", encoding="utf-8")
    print("Starting Ollama server: ollama serve")
    with log_file.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(["ollama", "serve"], stdout=lf, stderr=lf)

    pid_file.write_text(str(proc.pid), encoding="utf-8")

    # Wait for readiness
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = _run(["tail", "-n", "200", str(log_file)], check=False).stdout
            raise RuntimeError(
                f"Ollama exited early with code {proc.returncode}\n"
                f"Log file: {log_file}\n"
                f"--- Log tail ---\n{tail}\n"
            )

        if _is_ollama_running():
            print("Ollama is up at http://127.0.0.1:11434")
            return proc.pid

        time.sleep(1)

    tail = _run(["tail", "-n", "200", str(log_file)], check=False).stdout
    raise RuntimeError(
        f"Timed out waiting for Ollama readiness after {timeout_s}s\n"
        f"Log file: {log_file}\n"
        f"--- Log tail ---\n{tail}\n"
    )


def pull_models(models: List[str]) -> None:
    """
    Pull each model via 'ollama pull <model>'.
    """
    for m in models:
        m = m.strip()
        if not m:
            continue
        print(f"Pulling model: {m}")
        r = _run(["ollama", "pull", m], check=False)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to pull model '{m}'.\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}")
        print(f"✅ Model '{m}' pulled (or already present).")


def list_models() -> str:
    r = _run(["ollama", "list"], check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Failed to list models.\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}")
    return r.stdout


def verify_models_available(required: List[str]) -> None:
    out = list_models()
    print("--- Ollama List Output ---")
    print(out)
    print("--------------------------")

    missing = [m for m in required if m not in out]
    if missing:
        raise RuntimeError(f"❌ Missing models in 'ollama list': {missing}")
    print(f"✅ All required models are available: {required}")


def setup_ollama(
    embedding_model: str,
    llm_model: str,
    log_dir: str,
    timeout_s: int = 60,
    force_restart: bool = True,
) -> int:
    """
    Full setup: start server, pull models, verify.
    Returns server PID.
    """
    pid = start_ollama(log_dir=log_dir, timeout_s=timeout_s, force_restart=force_restart)
    pull_models([embedding_model, llm_model])
    verify_models_available([embedding_model, llm_model])
    print(f"✅ Ollama server running and models ready: '{embedding_model}', '{llm_model}'")
    return pid


def stop_ollama(log_dir: str) -> None:
    log_path = Path(log_dir).expanduser().resolve()
    pid_file = log_path / "ollama_serve.pid"
    stop_existing(pid_file)
    print("✅ Ollama stopped (best-effort).")


if __name__ == "__main__":
    # Env-controlled config (so bash scripts can run it easily)
    base_dir = os.getenv("FAME_BASE_DIR", str(Path.home() / "Desktop" / "FAME_project"))
    log_dir = os.getenv("OLLAMA_LOG_DIR", str(Path(base_dir) / "data" / "ollama"))

    embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    llm_model = os.getenv("OLLAMA_LLM_MODEL", "").strip()

    # If not provided, let user know clearly (no guessing)
    if not llm_model:
        raise RuntimeError(
            "❌ OLLAMA_LLM_MODEL is not set.\n"
            "Example:\n"
            "  export OLLAMA_LLM_MODEL=llama3.1:8b\n"
            "  python -m fame.services.ollama_service"
        )

    timeout_s = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "60"))
    force_restart = os.getenv("OLLAMA_FORCE_RESTART", "1").strip() not in ("0", "false", "False")

    print("Starting Ollama service manager...")
    print(f"Log dir         : {log_dir}")
    print(f"Embedding model : {embedding_model}")
    print(f"LLM model       : {llm_model}")
    print(f"Force restart   : {force_restart}")

    pid = setup_ollama(
        embedding_model=embedding_model,
        llm_model=llm_model,
        log_dir=log_dir,
        timeout_s=timeout_s,
        force_restart=force_restart,
    )
    print(f"✅ Ollama ready. PID={pid}")

