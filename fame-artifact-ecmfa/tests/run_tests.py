#!/usr/bin/env python3
"""
Artifact test suite. Run from the repository root:

    python3 tests/run_tests.py

Asserts three things:
  1. scenario/scenario.xmi is well-formed (no condition violated).
  2. Every file in tests/seeded/ violates the condition named by its filename.
  3. scenario/correspondences.corr round-trips through the notation tools.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fame_validate  # noqa: E402
import corr_notation  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = RESET = ""

passed = failed = 0


def report(ok, label, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"{GREEN}PASS{RESET} {label}")
    else:
        failed += 1
        print(f"{RED}FAIL{RESET} {label}  {detail}")


# --- 1. the clean scenario validates ---------------------------------------
scenario = os.path.join(ROOT, "scenario", "scenario.xmi")
vs = fame_validate.check(scenario)
report(not vs, "scenario/scenario.xmi is well-formed",
       "; ".join(f"{v.cond} on {v.obj_id}" for v in vs))

# --- 2. every seed violates the condition its filename names ---------------
seeded = os.path.join(ROOT, "tests", "seeded")
for fn in sorted(os.listdir(seeded)):
    if not fn.endswith(".xmi"):
        continue
    expected = fn[:-4]
    vs = fame_validate.check(os.path.join(seeded, fn))
    names = {v.cond for v in vs}
    extra = names - {expected}
    label = f"{fn} violates {expected}"
    if extra:
        label += f"  (also, by cascade: {', '.join(sorted(extra))})"
    report(expected in names, label, f"got {sorted(names) or 'no violations'}")

# --- 3. the notation round-trips -------------------------------------------
corr_file = os.path.join(ROOT, "scenario", "correspondences.corr")
try:
    parsed = corr_notation.parse_file(corr_file)
    reserialised = corr_notation.to_text(parsed)
    reparsed = corr_notation.parse_text(reserialised)
    report(parsed == reparsed, "correspondences.corr parses and re-serialises identically")

    from_xmi = corr_notation.from_xmi(scenario)
    report(from_xmi == parsed,
           "correspondences.corr agrees with scenario/scenario.xmi",
           "notation and XMI describe different correspondence models")
except Exception as exc:  # noqa: BLE001
    report(False, "notation round-trip", repr(exc))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
