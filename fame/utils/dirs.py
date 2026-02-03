from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


# ---------------------------------------------------------
# Core path registry (NO side effects)
# ---------------------------------------------------------

@dataclass(frozen=True)
class FamePaths:
    base_dir: Path

    # input / author-controlled
    raw_data: Path
    prompts: Path
    specification: Path
    notebooks: Path
    tests: Path

    # intermediate / generated
    processed_data: Path
    vector_db: Path
    vector_db_algo1: Path

    # results (generated)
    results: Path
    ss_fm: Path
    ms_fm: Path

    # non-rag umbrella (generated) - kept for compatibility/convenience
    non_root: Path
    non_fm: Path
    non_context: Path
    non_reports: Path
    non_runs: Path

    # non-rag variants (generated)
    non_ss_root: Path
    non_ss_fm: Path
    non_ss_context: Path
    non_ss_reports: Path
    non_ss_runs: Path

    non_is_root: Path
    non_is_fm: Path
    non_is_context: Path
    non_is_reports: Path
    non_is_runs: Path

    # reports & validation (generated)
    reports: Path
    validation: Path
    success: Path
    failed: Path
    images: Path
    chunk_reports: Path

    # llm-as-judge (generated)
    judge_fm: Path
    judge_features: Path
    judge_constraints: Path
    judge_hierarchy: Path
    judge_reports: Path

    # ground truth (may be provided or generated)
    ground_truth: Path


# ---------------------------------------------------------
# Base directory resolution
# ---------------------------------------------------------

def resolve_base_dir() -> Path:
    """
    Resolution order:
    1) FAME_BASE_DIR environment variable (explicit, recommended)
    2) Repository root (auto-detected from this file location)
    """
    env = os.getenv("FAME_BASE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    # fame/utils/dirs.py -> fame/utils -> fame -> repo root
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Build paths (pure function, NO mkdir)
# ---------------------------------------------------------

def build_paths(base_dir: Optional[Path] = None) -> FamePaths:
    base = (base_dir or resolve_base_dir()).expanduser().resolve()

    data = base / "data"
    results = base / "results"

    non_root = results / "non_rag"
    non_ss_root = non_root / "ss-nonrag"
    non_is_root = non_root / "is-nonrag"

    return FamePaths(
        base_dir=base,

        # input / author-controlled
        raw_data=data / "raw",
        prompts=base / "prompts",
        specification=base / "prompts" / "specification",
        notebooks=base / "notebooks",
        tests=base / "tests",

        # intermediate / generated
        processed_data=data / "processed" / "algorithm_1",
        vector_db=data / "chroma_db",
        vector_db_algo1=data / "chroma_db_algorithm_1",

        # results (generated)
        results=results,
        ss_fm=results / "rag" / "ss-rgfm" / "fm",
        ms_fm=results / "rag" / "ms-rgfm" / "fm",

        # non-rag umbrella (generated) - kept for compatibility
        non_root=non_root,
        non_fm=non_root / "fm",
        non_context=non_root / "context",
        non_reports=non_root / "reports",
        non_runs=non_root / "runs",

        # non-rag variants (generated)
        non_ss_root=non_ss_root,
        non_ss_fm=non_ss_root / "fm",
        non_ss_context=non_ss_root / "context",
        non_ss_reports=non_ss_root / "reports",
        non_ss_runs=non_ss_root / "runs",

        non_is_root=non_is_root,
        non_is_fm=non_is_root / "fm",
        non_is_context=non_is_root / "context",
        non_is_reports=non_is_root / "reports",
        non_is_runs=non_is_root / "runs",

        # reports & validation (generated)
        reports=results / "rag" / "ss-rgfm" / "reports",
        validation=results / "rag" / "ss-rgfm" / "reports" / "validation",
        success=results / "rag" / "ss-rgfm" / "reports" / "validation" / "success",
        failed=results / "rag" / "ss-rgfm" / "reports" / "validation" / "failed",
        images=results / "rag" / "ss-rgfm" / "images",
        chunk_reports=results / "rag" / "ss-rgfm" / "chunk_retrieval_reports",

        # llm-as-judge (generated)
        judge_fm=results / "llm_judge" / "ms-rgfm" / "fm",
        judge_features=results / "llm_judge" / "ms-rgfm" / "features",
        judge_constraints=results / "llm_judge" / "ms-rgfm" / "constraints",
        judge_hierarchy=results / "llm_judge" / "ms-rgfm" / "hierarchy",
        judge_reports=results / "llm_judge" / "ms-rgfm" / "reports",

        # ground truth (may be provided or generated)
        ground_truth=results / "ground_truth",
    )


# ---------------------------------------------------------
# On-demand directory creation
# ---------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create a single directory if missing. Safe to call repeatedly."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> Dict[str, Path]:
    """Create multiple directories explicitly. Returns a mapping for logging/debugging."""
    created: Dict[str, Path] = {}
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        created[str(p)] = p
    return created


# ---------------------------------------------------------
# Stage-scoped helpers
# ---------------------------------------------------------

def ensure_for_stage(stage: str, p: FamePaths) -> Dict[str, Path]:
    """
    Create only the directories needed for a given pipeline stage.
    """
    stage = stage.lower().strip()
    created: Dict[str, Path] = {}

    def mk(label: str, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        created[label] = path

    if stage in ("raw", "ingest"):
        mk("RAW_DATA", p.raw_data)

    elif stage in ("prompts", "specification"):
        mk("PROMPTS", p.prompts)
        mk("SPECIFICATION", p.specification)

    elif stage in ("notebooks",):
        mk("NOTEBOOKS", p.notebooks)

    elif stage in ("tests", "test"):
        mk("TESTS", p.tests)

    elif stage in ("preprocess", "chunk"):
        mk("PROCESSED_DATA", p.processed_data)
        mk("CHUNK_REPORTS", p.chunk_reports)

    elif stage in ("vectorize", "embed", "chroma"):
        mk("VECTOR_DB", p.vector_db)
        mk("VECTOR_DB_ALGO1", p.vector_db_algo1)

    elif stage in ("ss-rgfm", "extract_ss"):
        mk("SS_FM", p.ss_fm)
        mk("REPORTS", p.reports)
        mk("IMAGES", p.images)

    elif stage in ("ms-rgfm", "extract_ms"):
        mk("MS_FM", p.ms_fm)

    # umbrella: create both non-rag variants + umbrella folders
    elif stage in ("non-rag", "non_rag", "nonrag"):
        mk("NON_ROOT", p.non_root)
        mk("NON_FM", p.non_fm)
        mk("NON_CONTEXT", p.non_context)
        mk("NON_REPORTS", p.non_reports)
        mk("NON_RUNS", p.non_runs)

        mk("NON_SS_FM", p.non_ss_fm)
        mk("NON_SS_CONTEXT", p.non_ss_context)
        mk("NON_SS_REPORTS", p.non_ss_reports)
        mk("NON_SS_RUNS", p.non_ss_runs)

        mk("NON_IS_FM", p.non_is_fm)
        mk("NON_IS_CONTEXT", p.non_is_context)
        mk("NON_IS_REPORTS", p.non_is_reports)
        mk("NON_IS_RUNS", p.non_is_runs)

    # specific non-rag variant stages
    elif stage in ("ss-nonrag", "ss_nonrag", "ss-non-rag"):
        mk("NON_SS_FM", p.non_ss_fm)
        mk("NON_SS_CONTEXT", p.non_ss_context)
        mk("NON_SS_REPORTS", p.non_ss_reports)
        mk("NON_SS_RUNS", p.non_ss_runs)

    elif stage in ("is-nonrag", "is_nonrag", "is-non-rag", "iter-nonrag", "iter_nonrag"):
        mk("NON_IS_FM", p.non_is_fm)
        mk("NON_IS_CONTEXT", p.non_is_context)
        mk("NON_IS_REPORTS", p.non_is_reports)
        mk("NON_IS_RUNS", p.non_is_runs)

    elif stage in ("validate",):
        mk("VALIDATION", p.validation)
        mk("SUCCESS", p.success)
        mk("FAILED", p.failed)

    elif stage in ("judge", "llm_judge"):
        mk("JUDGE_FM", p.judge_fm)
        mk("JUDGE_FEATURES", p.judge_features)
        mk("JUDGE_CONSTRAINTS", p.judge_constraints)
        mk("JUDGE_HIERARCHY", p.judge_hierarchy)
        mk("JUDGE_REPORTS", p.judge_reports)

    elif stage in ("ground_truth", "ground-truth"):
        mk("GROUND_TRUTH", p.ground_truth)

    else:
        raise ValueError(f"Unknown stage '{stage}'")

    return created


# ---------------------------------------------------------
# Convenience: print resolved paths (NO mkdir by default)
# ---------------------------------------------------------

def print_paths(p: FamePaths) -> None:
    """Print all registered paths for debugging. Does NOT create directories."""
    print("\n=== FAME Paths ===")
    print(f"BASE_DIR: {p.base_dir}")
    for k, v in vars(p).items():
        if k != "base_dir":
            print(f"{k}: {v}")
    print("==================\n")
