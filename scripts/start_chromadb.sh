#!/usr/bin/env bash
set -euo pipefail

# pick python
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "❌ Python not found."
  exit 1
fi

echo "Using Python: $PY ($($PY --version))"
$PY -m fame.services.chroma_service
