# Feature Argumentation Modelling Environment (FAME)

FAME builds feature models from documents using Retrieval‑Augmented (RAG) and Non‑RAG pipelines.

## 1) Initial setup (once)
- See [SETUP.md](SETUP.md) for details.
- TL;DR (macOS/Linux):
  ```bash
  ./scripts/initial_setup.sh
  ```

## 2) End‑to‑end launcher (recommended)
Runs optional preprocessing, then lets you pick RAG / Non‑RAG and SS / IS variants.

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_fame.py
```

## 3) Evaluation helpers
- Coverage (semantic recall vs ground truth):
  ```bash
  PYTHONPATH=$(pwd) .venv/bin/python scripts/coverage_fm.py \
    --gt data/ground_truth/federation.xml \
    --pred results/rag/ss-rgfm/fm/your_model.xml
  ```
- Conformance/XSD check:
  ```bash
  PYTHONPATH=$(pwd) .venv/bin/python scripts/check_wellformed.py \
    --xml results/rag/ss-rgfm/fm/your_model.xml \
    --xsd prompts/specifications/feature_model_featureide.xsd
  ```

## 4) Run individual steps
All commands assume the venv is active (`source .venv/bin/activate`) and `PYTHONPATH=$(pwd)`.

### Preprocessing (ingest + vectorize)
```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/preprocessing_for_rag.py
```

### Single‑Stage Non‑RAG (SS‑NonRAG)
```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_ss_nonrag.py --interactive
```

### Iterative Non‑RAG (IS‑NonRAG)
```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_is_nonrag.py --interactive
```

### Single‑Stage RAG (SS‑RGFM)
```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_ss_rag.py --interactive
```

### Iterative RAG (IS‑RGFM)
```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/run_is_rag.py --interactive
```

## 5) Ollama host hints
- Embeddings typically run on a local Ollama host: `OLLAMA_EMBED_HOST=http://127.0.0.1:11434`.
- LLM generation can use cloud: `OLLAMA_LLM_HOST=https://ollama.com`.
Set these before running preprocessing/pipelines if you split hosts.

## 6) Logs & outputs
- Results: `results/` (FM XMLs, prompts, stats, etc.)
- Logs: `results/logs/fame.log` (structured JSON)

For more setup detail, read [SETUP.md](SETUP.md).
