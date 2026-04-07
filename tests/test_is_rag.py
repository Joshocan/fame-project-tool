from __future__ import annotations

from pathlib import Path
import pytest

from fame.utils.dirs import build_paths, ensure_for_stage
from fame.rag.is_pipeline import ISRgfmConfig, run_is_rgfm


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, *, root_feature, domain, collections, n_results_per_collection, max_total_results):
        self.calls += 1

        class Res:
            def __init__(self, q, col):
                self.query = q
                self.chunks = [
                    type("C", (), {"collection": col, "chunk_id": "c1", "text": "text1", "metadata": {}, "distance": 0.1}),
                ]
        return Res(f"{root_feature} {domain}", collections[0])

    def to_prompt_evidence(self, result, max_total_chars: int, max_chunk_chars: int) -> str:
        return "EVIDENCE: text1"


class FakeLLM:
    def __init__(self) -> None:
        self.model = "fake-llm"
        self.host = "http://fake"
        self.calls = 0

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        self.calls += 1
        return f"<featureModel><root>FM_{self.calls}</root></featureModel>"


@pytest.mark.integration
def test_is_rag_writes_outputs() -> None:
    paths = build_paths()
    ensure_for_stage("is-rgfm", paths)
    ensure_for_stage("preprocess", paths)

    chunks_dir = paths.processed_data / "chunks"
    files = sorted(chunks_dir.glob("*.chunks.json"))
    if len(files) < 1:
        pytest.skip("No *.chunks.json found. Run ingestion first to create chunks.")

    cfg = ISRgfmConfig(
        root_feature="Model Federation",
        domain="Model-Driven Engineering",
        chunks_files=files[:1],
    )

    out = run_is_rgfm(cfg, llm=FakeLLM(), retriever=FakeRetriever())

    final_xml = Path(out["final_xml"])
    meta_file = Path(out["meta"])

    assert final_xml.exists()
    assert meta_file.exists()
    assert final_xml.read_text(encoding="utf-8").strip()
