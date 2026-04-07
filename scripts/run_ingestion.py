#!/usr/bin/env python3
"""
Run the FAME ingestion + preparation pipeline.

This script:
  - resolves the FAME workspace
  - runs ingestion over data/raw
  - produces chunked JSON under data/processed/chunks
"""

from __future__ import annotations

import os
from pathlib import Path

from fame.ingestion.pipeline import ingest_and_prepare
from fame.utils.runtime import workspace


def main() -> None:
    # Resolve workspace (creates required dirs for preprocess stage)
    ws = workspace("preprocess", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths

    print("=== FAME Ingestion Stage ===")
    print(f"Base dir       : {paths.base_dir}")
    print(f"Raw data dir   : {paths.raw_data}")
    print(f"Processed dir  : {paths.processed_data}")
    print("============================")

    result = ingest_and_prepare(
        raw_dir=paths.raw_data,
        out_dir=paths.processed_data / "chunks",
    )

    print("\n=== Ingestion Result ===")
    print(f"Processed files: {len(result['processed'])}")
    print(f"Skipped files  : {len(result['skipped'])}")

    if result["processed"]:
        print("\nSample output:")
        print(f"  {result['processed'][0]}")

    if result["skipped"]:
        print("\nSkipped inputs:")
        for s in result["skipped"]:
            print(f"  - {s}")

    print("\nSUCCESS: Ingestion stage completed successfully.")


if __name__ == "__main__":
    main()
