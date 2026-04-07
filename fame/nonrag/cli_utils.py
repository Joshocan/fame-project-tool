from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def prompt_choice(title: str, options: Tuple[str, ...]) -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    while True:
        choice = input("Select option: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Try again.")


def load_key_file(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None


def default_high_level_features():
    return {
        "Structural": (
            "Covers the artefacts, links, and structural features of Model Federation, "
            "including model composition, graph structure, formalism, domain, and technological context."
        ),
        "Operational": (
            "Includes the processes, operations, and triggers in Model Federation, focusing on querying, "
            "synchronisation, validation, and management of federated models and links."
        ),
        "Intentional": (
            "Addresses the objectives and goals of Model Federation, such as traceability, consistency assurance, "
            "transformation management, composition, and conceptual elicitation."
        ),
    }
