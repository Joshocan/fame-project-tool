#!/usr/bin/env python3
"""
FAME reference validator and resolver.

Two jobs:

1.  A dependency-free re-implementation of every invariant in
    constraints/constraints.ocl, so that the seeded-violation test set can be
    checked on any machine without an Eclipse/EMF installation. The OCL
    document remains normative; this is a cross-check, and tests/run_tests.py
    asserts the two agree on which condition each seeded file violates.

2.  The *resolver* referred to in Sections 6.4 and 6.5 of the paper. Conditions
    E5 (endpoint references resolve) and E7 (roles iff directed) are the two
    conditions that intra-resource OCL cannot decide, because they require
    loading the participating system models. They are decided here.

Usage:
    python3 tools/fame_validate.py scenario/scenario.xmi
    python3 tools/fame_validate.py --quiet tests/seeded/C2_*.xmi

Exit code 0 if well-formed, 1 if any condition is violated.
"""

import sys
import xml.etree.ElementTree as ET

TAU = 0.8  # condition C3 threshold, matches constraints.ocl line 6

DIRECTED_KINDS = {"refinement", "generalization", "dependency"}

# XML tag -> FAME metatype for containment features
TAG_TYPE = {
    "components": "Component",
    "exposes": "Interface",
    "provides": "Capability",
    "decomposesInto": "Operation",
    "concepts": "DataConcept",
    "attributes": "Attribute",
    "relationships": "Relationship",
    "constraintRules": "ConstraintRule",
    "correspondences": "Correspondence",
}
NON_ELEMENT_TAGS = {"justifiedBy", "artefacts", "relates", "ends", "conditionedBy"}


class Violation:
    def __init__(self, cond, obj_id, detail):
        self.cond, self.obj_id, self.detail = cond, obj_id, detail

    def __str__(self):
        return f"  {self.cond:<38} {self.obj_id or '(model)':<28} {self.detail}"


class Element:
    """A FAME ModelElement lifted out of the XMI."""

    def __init__(self, node, metatype, owner=None):
        self.node, self.metatype, self.owner = node, metatype, owner
        g = node.get
        self.id = g("id")
        self.name = g("name")
        self.status = g("status")
        self.kind = g("kind")
        self.confidence = float(g("confidence")) if g("confidence") is not None else None
        self.corroboration = int(g("corroborationCount")) if g("corroborationCount") is not None else None
        self.links = [(p.get("role"), p.get("evidence")) for p in node.findall("justifiedBy")]
        self.children = []


def parse(path):
    root = ET.parse(path).getroot()
    systems, corrmodels, all_elements = [], [], []

    def walk(node, owner):
        for child in node:
            if not isinstance(child.tag, str) or child.tag in NON_ELEMENT_TAGS:
                continue
            mt = TAG_TYPE.get(child.tag)
            if mt is None:
                continue
            el = Element(child, mt, owner)
            all_elements.append(el)
            if owner is not None:
                owner.children.append(el)
            walk(child, el)

    for top in root:
        if not isinstance(top.tag, str):
            continue
        tag = top.tag.split("}")[-1]
        if tag == "SystemModel":
            el = Element(top, "SystemModel")
            systems.append(el)
            all_elements.append(el)
            walk(top, el)
        elif tag == "CorrespondenceModel":
            corrmodels.append(top)
            walk(top, None)
    return root, systems, corrmodels, all_elements


def tree_of(sysmodel):
    out, stack = [], [sysmodel]
    while stack:
        e = stack.pop()
        out.append(e)
        stack.extend(e.children)
    return out


def check(path):
    root, systems, corrmodels, elements = parse(path)
    v = []
    corr_elements = [e for e in elements if e.metatype == "Correspondence"]

    # ---- A. structural -----------------------------------------------------
    for sm in systems:
        ids = [e.id for e in tree_of(sm)]
        dupes = {i for i in ids if i is not None and ids.count(i) > 1}
        for d in sorted(dupes):
            v.append(Violation("A1_UniqueIds", sm.id, f"duplicate id '{d}'"))

    for e in elements:
        if not e.name:
            v.append(Violation("A2_NonEmptyName", e.id, "name is empty or absent"))

    for sm in systems:
        local = {x.id for x in tree_of(sm)}
        for r in [x for x in tree_of(sm) if x.metatype == "Relationship"]:
            for end in ("source", "target"):
                ref = r.node.get(end)
                if ref and ref not in local:
                    v.append(Violation("A3_LocalEndpoints", r.id, f"{end} '{ref}' outside {sm.id}"))

    by_id = {e.id: e for e in elements if e.id}
    for a in [e for e in elements if e.metatype == "Attribute"]:
        if a.owner is None or a.owner.metatype != "DataConcept":
            owner = a.owner.metatype if a.owner else "none"
            v.append(Violation("A5_OwnedAttribute", a.id, f"owner is {owner}, not DataConcept"))

    for r in [e for e in elements if e.metatype == "Relationship"]:
        if r.kind == "specialization":
            s, t = by_id.get(r.node.get("source")), by_id.get(r.node.get("target"))
            if s and t and s.metatype != t.metatype:
                v.append(Violation("A6_TypeConsistentSpecialization", r.id,
                                   f"{s.metatype} specialises {t.metatype}"))

    # ---- B. provenance -----------------------------------------------------
    for e in elements:
        if not e.links:
            v.append(Violation("B1_EverythingJustified", e.id, "no ProvenanceLink"))
        if e.confidence is None or not (0.0 <= e.confidence <= 1.0):
            v.append(Violation("B2_BoundedConfidence", e.id, f"confidence={e.confidence}"))
        if e.links and not any(r == "primary" for r, _ in e.links):
            v.append(Violation("B3_PrimaryEvidence", e.id, "no primary link"))

    # ---- C. corroboration --------------------------------------------------
    for e in elements:
        supporting = {ev for r, ev in e.links if r != "contradicting"}
        if e.corroboration is None or e.corroboration < 0:
            v.append(Violation("C1_ConsistentCounting", e.id, f"count={e.corroboration}"))
        elif e.corroboration < len(supporting):
            v.append(Violation("C1_ConsistentCounting", e.id,
                               f"count={e.corroboration} < {len(supporting)} distinct artefacts"))
        if any(r == "contradicting" for r, _ in e.links) and e.confidence is not None and e.confidence >= 1.0:
            v.append(Violation("C2_ContradictionCapsConfidence", e.id,
                               f"contradicting link with confidence={e.confidence}"))
        if e.confidence is not None and e.confidence >= TAU and (e.corroboration or 0) < 2:
            v.append(Violation("C3_HighConfidenceNeedsCorroboration", e.id,
                               f"confidence={e.confidence} >= {TAU} but count={e.corroboration}"))

    # ---- T. trust lifecycle ------------------------------------------------
    for e in elements:
        if e.status == "trusted":
            ok = (e.links and any(r == "primary" for r, _ in e.links)
                  and e.confidence is not None and 0.0 <= e.confidence <= 1.0
                  and (e.corroboration or -1) >= 0)
            if not ok:
                v.append(Violation("T1_TrustedMeansEvidenced", e.id,
                                   "status=trusted but B/C preconditions unmet"))

    # ---- D. logical --------------------------------------------------------
    for c in [e for e in elements if e.metatype == "ConstraintRule"]:
        if c.node.get("formal") == "true" and not (c.node.get("expression") or "").strip():
            v.append(Violation("D1_FormalHasExpression", c.id, "formal=true with empty expression"))
        for ref in (c.node.get("constrains") or "").split():
            if ref not in by_id:
                v.append(Violation("D2_TargetsExist", c.id, f"constrains unknown element '{ref}'"))

    # ---- E. correspondence -------------------------------------------------
    sys_by_id = {s.id: s for s in systems}
    for cm in corrmodels:
        parts = cm.findall("relates")
        refs = [p.get("ref") for p in parts]
        if len(parts) < 2 or len(set(refs)) != len(refs):
            v.append(Violation("E1_RelatesTwoModels", None,
                               f"relates {len(parts)} model(s), refs={refs}"))
        alias_by_index = {i: p.get("alias") for i, p in enumerate(parts)}
        ref_by_index = {i: p.get("ref") for i, p in enumerate(parts)}

        def part_index(end):
            im = end.get("inModel") or ""
            return int(im.split("@relates.")[-1]) if "@relates." in im else None

        for c in cm.findall("correspondences"):
            cid = c.get("id")
            ends = c.findall("ends")
            directed = c.get("directed") == "true"
            kind = c.get("kind")

            if len(ends) < 2:
                v.append(Violation("E2_AtLeastTwoEnds", cid, f"{len(ends)} end(s)"))
            if len({part_index(e) for e in ends}) < 2:
                v.append(Violation("E3_CrossesModels", cid, "all ends in one participating model"))
            seen = [(part_index(e), e.get("elementRef")) for e in ends]
            if len(set(seen)) != len(seen):
                v.append(Violation("E4_DistinctEndpoints", cid, "an endpoint appears twice"))
            if (kind in DIRECTED_KINDS) != directed:
                v.append(Violation("E6_DirectionMatchesKind", cid,
                                   f"kind={kind} but directed={directed}"))
            if directed and any(e.get("role") is None for e in ends):
                v.append(Violation("E7_RolesIffDirected", cid, "directed end without a role"))
            if not directed and any(e.get("role") is not None for e in ends):
                v.append(Violation("E7b_NoRolesIfUndirected", cid, "undirected end carries a role"))

            # E5 -- the resolver. Requires the participating system models.
            for e in ends:
                idx, ref = part_index(e), e.get("elementRef")
                target_model = sys_by_id.get(ref_by_index.get(idx))
                if target_model is None:
                    v.append(Violation("E5_EndpointsResolve", cid,
                                       f"participating model '{ref_by_index.get(idx)}' not loaded"))
                elif ref not in {x.id for x in tree_of(target_model)}:
                    v.append(Violation("E5_EndpointsResolve", cid,
                                       f"'{ref}' not found in {target_model.id}"))

        # ---- G1 internal consistency
        def endset(c):
            return frozenset((part_index(e), e.get("elementRef")) for e in c.findall("ends"))

        cs = cm.findall("correspondences")
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if endset(cs[i]) == endset(cs[j]):
                    kinds = {cs[i].get("kind"), cs[j].get("kind")}
                    if kinds == {"equivalence", "mismatch"}:
                        v.append(Violation("G1_InternalConsistency",
                                           f"{cs[i].get('id')}/{cs[j].get('id')}",
                                           "same endpoints asserted equivalence and mismatch"))
    return v


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in sys.argv
    if not args:
        print(__doc__)
        return 2
    failed = False
    for path in args:
        vs = check(path)
        if vs:
            failed = True
            print(f"FAIL {path}  ({len(vs)} violation(s))")
            if not quiet:
                for x in vs:
                    print(x)
        else:
            print(f"OK   {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
