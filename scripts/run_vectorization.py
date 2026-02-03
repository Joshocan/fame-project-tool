#!/usr/bin/env python3
from __future__ import annotations

import os
from fame.vectorization.pipeline import index_all_chunks
from fame.utils.runtime import workspace


def main() -> None:
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    print("=== FAME Vectorization Stage ===")
    print(f"Base dir     : {paths.base_dir}")
    print(f"Chunks dir   : {paths.processed_data / 'chunks'}")
    print(f"Vector DB dir: {paths.vector_db}")
    print("===============================")

    res = index_all_chunks(batch_size=int(os.getenv("VEC_BATCH_SIZE", "24")))
    print("\n✅ Vectorization complete.")
    print(res)


if __name__ == "__main__":
    main()
