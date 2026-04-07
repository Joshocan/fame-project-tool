Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
$venvDir = Join-Path $repoRoot ".venv"

if (-Not (Test-Path $venvDir)) {
  python -m venv $venvDir
}

$python = Join-Path $venvDir "Scripts/python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $repoRoot "config/requirements.txt")

& $python - <<'PY'
import nltk
nltk.download("punkt", quiet=True)
PY

Write-Output "SUCCESS: Bootstrap complete (Windows)."
