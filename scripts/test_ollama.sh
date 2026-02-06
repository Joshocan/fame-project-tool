#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL="${OLLAMA_LLM_MODEL:-gpt-oss:120b-cloud}"

KEY=""
if [[ -n "${OLLAMA_API_KEY:-}" ]]; then
  KEY="${OLLAMA_API_KEY}"
elif [[ -n "${OLLAMA_API_KEY_FILE:-}" && -f "${OLLAMA_API_KEY_FILE}" ]]; then
  KEY="$(cat "${OLLAMA_API_KEY_FILE}")"
elif [[ -f "${PROJECT_ROOT}/api_keys/ollama_key.txt" ]]; then
  KEY="$(cat "${PROJECT_ROOT}/api_keys/ollama_key.txt")"
fi

AUTH_HEADER="${OLLAMA_AUTH_HEADER:-Authorization}"
AUTH_SCHEME="${OLLAMA_AUTH_SCHEME:-Bearer}"

echo "Testing Ollama at ${HOST} with model '${MODEL}'"

if [[ -n "${KEY}" ]]; then
  if [[ -n "${AUTH_SCHEME}" ]]; then
    AUTH_VALUE="${AUTH_SCHEME} ${KEY}"
  else
    AUTH_VALUE="${KEY}"
  fi
  curl -i -sS -X POST "${HOST}/api/generate" \
    -H "${AUTH_HEADER}: ${AUTH_VALUE}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"ping\",\"stream\":false}" \
    | head -n 40
else
  curl -i -sS -X POST "${HOST}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"ping\",\"stream\":false}" \
    | head -n 40
fi
