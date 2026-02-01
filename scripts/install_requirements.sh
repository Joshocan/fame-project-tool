#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 - <<'PY'
import nltk
nltk.download("punkt", quiet=True)
PY

echo "✅ Requirements installed and NLTK punkt downloaded."


