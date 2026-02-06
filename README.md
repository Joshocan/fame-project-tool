# Feature Argumentation Modelling Environment  (FAME) Project

## Setup (Automated)

Use the setup script to create the virtual environment and install all requirements.

macOS/Linux:
```bash
./scripts/initial_setup.sh
```

Windows (PowerShell):
```powershell
.\scripts\bootstrap.ps1
```

### Running scripts and tests

macOS/Linux:
```bash
./scripts/run_ingestion.sh
./scripts/run_tests.sh tests/test_ingestion.py -v
```

Windows (PowerShell):
```powershell
.\scripts\bootstrap.ps1
.venv\Scripts\python scripts\preprocessing_for_rag.py
.venv\Scripts\python -m pytest tests\test_ingestion.py -v
```

## Ollama API Key (if your Ollama server is secured)

Local Ollama does not require an API key by default. If your Ollama server is behind an auth proxy or a secured/managed deployment, obtain the API key from your hosting provider/admin and set:

macOS/Linux:
```bash
export OLLAMA_API_KEY="your_key_here"
```

Windows (PowerShell):
```powershell
$env:OLLAMA_API_KEY="your_key_here"
```

This key is used for both embeddings and generation requests.

Tip: if you store the key in `api_keys/ollama_key.txt`, you can load it like this:

macOS/Linux:
```bash
export OLLAMA_API_KEY="$(cat api_keys/ollama_key.txt)"
```

Windows (PowerShell):
```powershell
$env:OLLAMA_API_KEY = Get-Content -Raw api_keys/ollama_key.txt
```

### Local embeddings + cloud LLM (recommended)

If your Ollama cloud account does not support embeddings, you can split hosts:

macOS/Linux:
```bash
export OLLAMA_EMBED_HOST="http://127.0.0.1:11434"
export OLLAMA_LLM_HOST="https://ollama.com"
export OLLAMA_API_KEY_FILE=api_keys/ollama_key.txt
```

If no API key is provided, the LLM defaults to local Ollama.
