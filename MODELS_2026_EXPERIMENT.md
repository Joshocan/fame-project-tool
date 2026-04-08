# MODELS 2026 Experiment Guide

This document describes the experiment workflow used for the paper:

- `Automated Feature Model Extraction from Textual Artefacts using Large Language Models`
- target venue: `MODELS '26`

Important: this is the `MODELS 2026` experiment guide, not `MODELS 2006`.

Before running this workflow, complete the environment setup in:

- `SETUP.md`

## 1) Experiment scope

The paper evaluates:

- four pipelines:
  - `SS-NonRAG`
  - `IS-NonRAG`
  - `SS-RAG`
  - `IS-RAG`
- five LLM families:
  - `gpt-oss:120b-cloud`
  - `deepseek-v3.2:cloud`
  - `glm-4.7:cloud`
  - `gpt-4.1`
  - `gemini-3.1-pro-preview`
- dataset:
  - `14` primary papers under `data/raw/`
  - ground-truth FM: `data/ground_truth/federation.xml`
- evaluation dimensions:
  - semantic precision / recall / F1
  - semantic coverage
  - well-formedness / conformance
  - satisfiability
  - top-`k` ranking of candidate FMs

## 2) Experiment defaults

Paper-level defaults reflected in the current code/config:

- LLM temperature: `0.2`
- RAG embedding model: `nomic-embed-text`
- evaluation embedding model: `all-mpnet-base-v2`
- vector store: `Chroma`
- Non-RAG chunk budget:
  - `max_chunk_chars = 6000`
- top-FM overall score:
  - `0.6 * semantic_f1 + 0.4 * coverage_score`
- top-FM eligibility:
  - well-formed / XSD-valid
  - no duplicate feature names
  - optional satisfiability gate when `--require-sat` is enabled

## 3) Environment assumptions

All commands below assume:

```bash
source $HOME/.venvs/fame/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH=$(pwd)
```

For stable runs, keep at least these outside OneDrive:

- Python virtual environments
- Chroma persistent storage
- temporary runtime / analysis copies

## 4) Reproduce the experiment

### 4.1 Preprocess documents for RAG

```bash
python scripts/preprocessing_for_rag.py
```

This ingests the `data/raw/` papers, creates cleaned chunks, and builds the Chroma collections.

### 4.2 Run the four pipeline families

Interactive mode:

```bash
python scripts/run_ss_nonrag.py --interactive
python scripts/run_is_nonrag.py --interactive
python scripts/run_ss_rag.py --interactive
python scripts/run_is_rag.py --interactive
```

Non-interactive mode:

```bash
python scripts/run_ss_nonrag.py --root-feature "Model Federation" --domain "Model Driven Engineering"
python scripts/run_is_nonrag.py --root-feature "Model Federation" --domain "Model Driven Engineering"
python scripts/run_ss_rag.py --root-feature "Model Federation" --domain "Model Driven Engineering"
python scripts/run_is_rag.py --root-feature "Model Federation" --domain "Model Driven Engineering"
```

Relevant outputs:

- `results/non_rag/ss-nonrag/`
- `results/non_rag/is-nonrag/`
- `results/rag/ss-rgfm/`
- `results/rag/is-rgfm/`

### 4.3 Rank top candidate feature models

Generate `top_1`, `top_3`, and `top_5` for all four pipelines:

```bash
python scripts/rank_top_fm.py \
  --all-pipelines \
  --gt-path data/ground_truth/federation.xml \
  --xsd-path prompts/specifications/feature_model_featureide.xsd \
  --top-fm 1

python scripts/rank_top_fm.py \
  --all-pipelines \
  --gt-path data/ground_truth/federation.xml \
  --xsd-path prompts/specifications/feature_model_featureide.xsd \
  --top-fm 3

python scripts/rank_top_fm.py \
  --all-pipelines \
  --gt-path data/ground_truth/federation.xml \
  --xsd-path prompts/specifications/feature_model_featureide.xsd \
  --top-fm 5
```

Require satisfiability in ranking:

```bash
python scripts/rank_top_fm.py \
  --all-pipelines \
  --gt-path data/ground_truth/federation.xml \
  --xsd-path prompts/specifications/feature_model_featureide.xsd \
  --top-fm 5 \
  --require-sat
```

Ranking outputs:

- `results/rag/ss-rgfm/top_fm/top_<k>/`
- `results/rag/is-rgfm/top_fm/top_<k>/`
- `results/non_rag/ss-nonrag/top_fm/top_<k>/`
- `results/non_rag/is-nonrag/top_fm/top_<k>/`

### 4.4 Build the overall evaluation dataset

```bash
python scripts/build_overall_pipeline_data.py \
  --gt data/ground_truth/federation.xml \
  --out-dir results/analysis \
  --label overall_four_pipelines \
  --require-sat
```

Main outputs:

- `results/analysis/overall_four_pipelines/overall_pipeline_runs_enriched.csv`
- `results/analysis/overall_four_pipelines/overall_pipeline_summary.csv`
- `results/analysis/overall_four_pipelines/overall_pipeline_summary_wf_only.csv`

### 4.5 Compare full-run performance against top-k performance

```bash
python scripts/compare_overall_vs_topk.py --verbose
```

Optional pooled summary:

```bash
python scripts/compare_overall_vs_topk.py --include-pooled --verbose
```

Outputs:

- `results/analysis/overall_four_pipelines/overall_vs_topk_comparison.csv`
- `results/analysis/overall_four_pipelines/overall_vs_topk_comparison.md`

### 4.6 Generate paper plots

```bash
python scripts/plot_eval_overall_capability.py \
  --data-dir results/analysis/overall_four_pipelines \
  --verbose

python scripts/plot_eval_rag_vs_nonrag.py \
  --data-dir results/analysis/overall_four_pipelines \
  --verbose

python scripts/plot_eval_iteration.py \
  --data-dir results/analysis/overall_four_pipelines \
  --verbose

python scripts/plot_eval_validity.py \
  --data-dir results/analysis/overall_four_pipelines \
  --verbose

python scripts/plot_eval_model_comparison.py \
  --data-dir results/analysis/overall_four_pipelines \
  --verbose
```

## 5) Retrieval-budget analysis

The paper studies retrieval depth `k` for SS-RAG and distinguishes:

- centralized retrieval: `one_collection`
- federated retrieval: `per_source`

Relevant tooling:

```bash
python scripts/ablate_ss_rag_k.py --interactive
python scripts/build_ss_k_rag_ablation_data.py
python scripts/plot_ss_rag_ablation.py
python scripts/ss_k_ablation_plots.py
```

Selected budgets in the manuscript:

- centralized / one-collection: `k = 70`
- federated / per-source: `k = 15`

## 6) Single-FM evaluation helpers

Coverage:

```bash
python scripts/coverage_fm.py \
  --gt data/ground_truth/federation.xml \
  --pred results/rag/ss-rgfm/fm/your_model.xml
```

Well-formedness / XSD conformance:

```bash
python scripts/check_wellformed.py \
  --xml results/rag/ss-rgfm/fm/your_model.xml \
  --xsd prompts/specifications/feature_model_featureide.xsd
```

Duplicate feature names:

```bash
python scripts/check_feature_duplicates.py \
  --xml results/rag/ss-rgfm/fm/your_model.xml
```

## 7) Main output directories

- generated FMs: `results/**/fm/`
- pipeline reports / metadata: `results/**/reports/`
- top-ranked FMs: `results/**/top_fm/`
- overall analysis dataset: `results/analysis/overall_four_pipelines/`
- curated final FMs: `final_fm/`
