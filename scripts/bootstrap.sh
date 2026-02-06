#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REPO_ROOT}/config/requirements.txt"

"${VENV_DIR}/bin/python" - <<'PY'
import nltk
nltk.download("punkt", quiet=True)
PY

echo "✅ Bootstrap complete (macOS/Linux)."
