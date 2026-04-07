from __future__ import annotations

import re


class UnresolvedPlaceholdersError(ValueError):
    def __init__(self, placeholders):
        super().__init__(f"Unreplaced prompt placeholders: {', '.join(sorted(placeholders))}")
        self.placeholders = placeholders


def assert_no_placeholders(text: str) -> None:
    """
    Raises UnresolvedPlaceholdersError if any {placeholder} or {{PLACEHOLDER}} remain.
    """
    missing = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
    missing.update(re.findall(r"\{([a-zA-Z0-9_]+)\}", text))
    if missing:
        raise UnresolvedPlaceholdersError(missing)
