from __future__ import annotations

from typing import Any, Dict, List


def partition_and_chunk(cleaned_text: str, source_filename: str) -> List[Dict[str, Any]]:
    """
    Use unstructured.partition_text + chunk_by_title to create chunks.
    Returns JSON-serializable dict chunks.
    """
    try:
        from unstructured.partition.text import partition_text
        from unstructured.chunking.title import chunk_by_title
    except Exception as e:
        raise RuntimeError(
            "unstructured is required for chunking. Install: python -m pip install -U unstructured"
        ) from e

    elements = partition_text(text=cleaned_text)
    chunks = chunk_by_title(elements)

    out: List[Dict[str, Any]] = []
    for i, el in enumerate(chunks):
        text = getattr(el, "text", "") or ""
        meta = getattr(el, "metadata", None)

        # metadata can be a dataclass-like object; keep it JSON-safe
        meta_dict: Dict[str, Any] = {}
        if meta is not None:
            # best-effort: many unstructured metadata objects implement to_dict()
            if hasattr(meta, "to_dict"):
                try:
                    meta_dict = meta.to_dict()  # type: ignore[attr-defined]
                except Exception:
                    meta_dict = {}
            else:
                # fallback: pick common attrs if present
                for k in ("filename", "page_number", "category", "coordinates", "languages"):
                    if hasattr(meta, k):
                        meta_dict[k] = getattr(meta, k)

        out.append(
            {
                "chunk_id": f"{source_filename}::chunk::{i}",
                "source": source_filename,
                "text": text,
                "metadata": meta_dict,
            }
        )

    return out
