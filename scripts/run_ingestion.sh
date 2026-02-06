#!/usr/bin/env bash
set -euo pipefail

echo "=== FAME Ingestion Stage ==="

# Ensure we run from repo root (directory containing 'fame/')
REPO_ROOT="${FAME_BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export FAME_BASE_DIR="$REPO_ROOT"

echo "Base dir: $FAME_BASE_DIR"

cd "$REPO_ROOT"

VENV_DIR="${REPO_ROOT}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -r "${REPO_ROOT}/config/requirements.txt"

"${VENV_DIR}/bin/python" -m scripts.run_ingestion

echo "✅ Ingestion stage finished."
