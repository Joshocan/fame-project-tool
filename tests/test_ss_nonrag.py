# tests/test_ss_nonrag.py
from __future__ import annotations

from pathlib import Path
import pytest

from fame.utils.dirs import build_paths, ensure_for_stage
from fame.nonrag.ss_pipeline import SSNonRagConfig, run_ss_nonrag


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.model = "fake-llm"

    def generate(self, prompt: str, *, system=None, temperature: float = 0.2) -> str:
        self.calls += 1
        return f"<featureModel><root>FM_ss_{self.calls}</root></featureModel>"


@pytest.mark.integration
def test_ss_nonrag_writes_outputs() -> None:
    paths = build_paths()
    ensure_for_stage("ss-nonrag", paths)
    ensure_for_stage("preprocess", paths)

    chunks_dir = paths.processed_data / "chunks"
    files = sorted(chunks_dir.glob("*.chunks.json"))
    if len(files) < 1:
        pytest.skip("No *.chunks.json found. Run ingestion first to create chunks.")

    cfg = SSNonRagConfig(
        root_feature="Model Federation",
        domain="Model-Driven Engineering",
        chunks_files=files[:1],
        max_total_chars=20_000,
        max_chunks=10,
        max_chunk_chars=2_000,
    )

    out = run_ss_nonrag(cfg, llm_client=FakeLLM())

    context_file = Path(out["context"])
    prompt_file = Path(out["prompt"])
    meta_file = Path(out["meta"])
    fm_file = Path(out["fm_xml"])

    assert context_file.exists()
    assert prompt_file.exists()
    assert meta_file.exists()
    assert fm_file.exists()

    # Modified prompt should be saved
    modified_prompts = sorted(paths.modified_prompts.glob("ss_nonrag_*-prompt.txt"))
    assert modified_prompts, "Expected modified prompt in results/modified_prompts"

    # Basic content sanity checks
    assert context_file.read_text(encoding="utf-8").strip()
    assert prompt_file.read_text(encoding="utf-8").strip()
    assert fm_file.read_text(encoding="utf-8").strip()
