#!/usr/bin/env python3
"""
Provenance resolvability check for condition B1.

For every SourceArtefact declared in an instance, confirm that its `uri`
resolves to a file in artefacts/. For every ProvenanceLink, confirm that its
`evidence` names a declared SourceArtefact.

    python3 tools/check_provenance.py scenario/scenario.xmi

SCOPE. This establishes that a provenance reference is *resolvable*, which is
what condition B1 as stated requires ("at least one resolvable reference to a
source artefact"). It does NOT establish that the cited artefact supports the
element it justifies, nor does it calibrate the confidence value. Those are
open problems, stated as such in the paper's limitations; nothing in this
artifact should be read as evidence that they are solved.
"""

import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(path):
    base = os.path.dirname(os.path.abspath(path))
    root = ET.parse(path).getroot()

    artefacts = {}
    for a in root.iter("artefacts"):
        artefacts[a.get("id")] = a.get("uri")

    problems = []
    for aid, uri in sorted(artefacts.items()):
        if not uri:
            problems.append(f"artefact {aid} has no uri")
            continue
        target = os.path.normpath(os.path.join(base, uri))
        if not os.path.isfile(target):
            problems.append(f"artefact {aid}: uri does not resolve -> {uri}")

    cited = set()
    for link in root.iter("justifiedBy"):
        ev = link.get("evidence")
        cited.add(ev)
        if ev not in artefacts:
            problems.append(f"provenance link cites undeclared artefact '{ev}'")

    unused = set(artefacts) - cited
    return problems, artefacts, cited, unused


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for path in sys.argv[1:]:
        problems, artefacts, cited, unused = check(path)
        print(f"{path}")
        print(f"  {len(artefacts)} artefact(s) declared, {len(cited)} cited by provenance links")
        if unused:
            print(f"  note: declared but never cited: {', '.join(sorted(unused))}")
        if problems:
            rc = 1
            for p in problems:
                print(f"  FAIL {p}")
        else:
            print("  OK   every cited artefact is declared and every uri resolves")
    return rc


if __name__ == "__main__":
    sys.exit(main())
