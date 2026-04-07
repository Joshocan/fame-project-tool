# tests/test_is_nonrag.py
from __future__ import annotations

import os
from pathlib import Path
import pytest

from fame.utils.dirs import build_paths, ensure_for_stage
from fame.nonrag.is_pipeline import ISNonRagConfig, run_is_nonrag


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, system=None, temperature: float = 0.2) -> str:
        self.calls += 1
        # minimal valid-ish XML placeholder
        return f"<featureModel><root>FM_iter_{self.calls}</root></featureModel>"


@pytest.mark.integration
def test_is_nonrag_writes_iteration_outputs() -> None:
    paths = build_paths()
    ensure_for_stage("is-nonrag", paths)
    ensure_for_stage("preprocess", paths)

    chunks_dir = paths.processed_data / "chunks"
    files = sorted(chunks_dir.glob("*.chunks.json"))
    if len(files) < 1:
        pytest.skip("No *.chunks.json found. Run ingestion first to create chunks.")

    # Use only first 1-2 sources to keep test fast
    use_files = files[:2]

    cfg = ISNonRagConfig(
        root_feature="Model Federation",
        domain="Model-Driven Engineering",
        chunks_files=use_files,
        max_delta_chars=10_000,
        max_delta_chunks=10,
    )

    out = run_is_nonrag(cfg, llm=FakeLLM())

    final_xml = Path(out["final_xml"])
    meta = Path(out["meta"])

    assert final_xml.exists()
    assert meta.exists()

    # Ensure iteration artifacts exist
    run_id = out["run_id"]
    iter_prompts = list(paths.non_is_runs.glob(f"{run_id}_iter*.prompt.txt"))
    iter_xmls = list(paths.non_is_runs.glob(f"{run_id}_iter*.xml"))
    iter_deltas = list(paths.non_is_context.glob(f"{run_id}_iter*.delta.txt"))

    assert len(iter_prompts) == len(use_files)
    assert len(iter_xmls) == len(use_files)
    assert len(iter_deltas) == len(use_files)
