# Feature Arguementation Modelling Environment  (FAME) Project

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
