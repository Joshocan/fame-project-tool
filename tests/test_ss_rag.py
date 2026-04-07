from __future__ import annotations

from pathlib import Path
import pytest

from fame.utils.dirs import build_paths, ensure_for_stage
from fame.rag.ss_pipeline import SSRGFMConfig, run_ss_rgfm


class FakeRetriever:
    def __init__(self) -> None:
        self.query = None
        self.calls = 0

    def retrieve(self, root_feature: str, domain: str, collections, n_results_per_collection: int, max_total_results: int):
        self.calls += 1
        self.query = f"{root_feature} {domain}"

        class Res:
            def __init__(self, q): self.query = q; self.chunks = [{"source": "dummy", "text": "content"}]
        return Res(self.query)

    def to_prompt_evidence(self, res, max_total_chars: int, max_chunk_chars: int) -> str:
        return "EVIDENCE: dummy"


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.model = "fake-llm"
        self.host = "http://fake"

    def generate(self, prompt: str, *, system=None, temperature: float = 0.2) -> str:
        self.calls += 1
        return f"<featureModel><root>FM_{self.calls}</root></featureModel>"


@pytest.mark.integration
def test_ss_rag_writes_outputs(tmp_path: Path) -> None:
    paths = build_paths()
    ensure_for_stage("ss-rgfm", paths)
    ensure_for_stage("preprocess", paths)
    ensure_for_stage("vectorize", paths)

    chunks_dir = paths.processed_data / "chunks"
    files = sorted(chunks_dir.glob("*.chunks.json"))
    if len(files) < 1:
        pytest.skip("No *.chunks.json found. Run ingestion first to create chunks.")

    cfg = SSRGFMConfig(
        root_feature="Model Federation",
        domain="Model-Driven Engineering",
        chunks_files=files[:1],
        prompt_path=None,
        collection_mode="per_source",
    )

    out = run_ss_rgfm(cfg, llm=FakeLLM(), retriever=FakeRetriever(), skip_vectorize=True)

    fm_file = Path(out["fm_xml"])
    prompt_file = Path(out["prompt"])
    evidence_file = Path(out["evidence"])
    meta_file = Path(out["meta"])

    assert fm_file.exists()
    assert prompt_file.exists()
    assert evidence_file.exists()
    assert meta_file.exists()

    assert fm_file.read_text(encoding="utf-8").strip()
    assert prompt_file.read_text(encoding="utf-8").strip()
    assert evidence_file.read_text(encoding="utf-8").strip()
