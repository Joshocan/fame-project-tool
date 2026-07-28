#!/usr/bin/env python3
"""
Derive metamodels/fame-delegates.ecore from metamodels/fame.ecore by embedding
the invariants of constraints/constraints.ocl as EMF OCL validation delegates,
so that "Validate" in the Eclipse model editor checks them with no extra setup.

    python3 tools/make_delegates.py

The Complete OCL document remains normative. This generator only transcribes
it. The `allElements()` helper of constraints.ocl has no delegate equivalent
and is inlined into A1 and A3.

IMPORTANT: the generated file has NOT been loaded in Eclipse by the generator.
Open it once in the Ecore editor, run Validate on scenario/scenario.xmi, and
confirm the invariants fire before releasing the artifact. See README section
"Verifying the delegate metamodel".
"""

import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "metamodels", "fame.ecore")
DST = os.path.join(HERE, "metamodels", "fame-delegates.ecore")

ALL = ("self->asSet()->closure(e | e.oclContents()->selectByKind(ModelElement))"
       "->including(self)")

INVARIANTS = {
    "SystemModel": {
        "A1_UniqueIds": f"{ALL}->isUnique(e | e.id)",
        "A3_LocalEndpoints": (
            f"self.relationships->forAll(r | {ALL}->includes(r.source) and "
            f"{ALL}->includes(r.target))"),
    },
    "ModelElement": {
        "A2_NonEmptyName": "self.name <> null and self.name.size() > 0",
        "B1_EverythingJustified": "self.justifiedBy->size() >= 1",
        "B2_BoundedConfidence": "self.confidence >= 0.0 and self.confidence <= 1.0",
        "B3_PrimaryEvidence":
            "self.justifiedBy->exists(p | p.role = ProvenanceRole::primary)",
        "C1_ConsistentCounting": (
            "self.corroborationCount >= 0 and self.corroborationCount >= "
            "self.justifiedBy->select(p | p.role <> ProvenanceRole::contradicting)"
            "->collect(p | p.evidence)->asSet()->size()"),
        "C2_ContradictionCapsConfidence": (
            "self.justifiedBy->exists(p | p.role = ProvenanceRole::contradicting) "
            "implies self.confidence < 1.0"),
        "C3_HighConfidenceNeedsCorroboration":
            "self.confidence >= 0.8 implies self.corroborationCount >= 2",
        "T1_TrustedMeansEvidenced": (
            "self.status = TrustStatus::trusted implies (self.justifiedBy->size() >= 1 "
            "and self.justifiedBy->exists(p | p.role = ProvenanceRole::primary) "
            "and self.confidence >= 0.0 and self.confidence <= 1.0 "
            "and self.corroborationCount >= 0)"),
    },
    "Attribute": {
        "A5_OwnedAttribute": "self.oclContainer().oclIsKindOf(DataConcept)",
    },
    "Relationship": {
        "A6_TypeConsistentSpecialization": (
            "self.kind = RelationKind::specialization implies "
            "self.source.oclType() = self.target.oclType()"),
    },
    "ConstraintRule": {
        "D1_FormalHasExpression":
            "self.formal implies (self.expression <> null and self.expression.size() > 0)",
        "D2_TargetsExist": (
            "self.constrains->forAll(e | e.oclContainer() <> null or "
            "e.oclIsKindOf(SystemModel))"),
    },
    "CorrespondenceModel": {
        "E1_RelatesTwoModels":
            "self.relates->size() >= 2 and self.relates->isUnique(p | p.ref)",
        "G1_InternalConsistency": (
            "self.correspondences->forAll(c1, c2 | (c1 <> c2 and "
            "c1.ends->collect(e | Tuple{m = e.inModel.ref, el = e.elementRef})->asSet() = "
            "c2.ends->collect(e | Tuple{m = e.inModel.ref, el = e.elementRef})->asSet()) "
            "implies not (Set{c1.kind, c2.kind} = "
            "Set{CorrespondenceKind::equivalence, CorrespondenceKind::mismatch}))"),
    },
    "Correspondence": {
        "E2_AtLeastTwoEnds": "self.ends->size() >= 2",
        "E3_CrossesModels": "self.ends->collect(e | e.inModel)->asSet()->size() >= 2",
        "E4_DistinctEndpoints":
            "self.ends->isUnique(e | Tuple{m = e.inModel.ref, el = e.elementRef})",
        "E6_DirectionMatchesKind": (
            "(Set{CorrespondenceKind::refinement, CorrespondenceKind::generalization, "
            "CorrespondenceKind::dependency}"
            "->includes(self.kind)) = self.directed"),
        "E7_RolesIffDirected":
            "self.directed implies self.ends->forAll(e | e.role <> null)",
        "E7b_NoRolesIfUndirected":
            "(not self.directed) implies self.ends->forAll(e | e.role = null)",
    },
}

ECORE_ANN = "http://www.eclipse.org/emf/2002/Ecore"
OCL_ANN = "http://www.eclipse.org/emf/2002/Ecore/OCL/Pivot"


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def annotations_for(cls, indent):
    invs = INVARIANTS[cls]
    pad = " " * indent
    out = [f'{pad}<eAnnotations source="{ECORE_ANN}">',
           f'{pad}  <details key="constraints" value="{" ".join(invs)}"/>',
           f"{pad}</eAnnotations>",
           f'{pad}<eAnnotations source="{OCL_ANN}">']
    for name, body in invs.items():
        out.append(f'{pad}  <details key="{name}" value="{esc(body)}"/>')
    out.append(f"{pad}</eAnnotations>")
    return "\n".join(out)


def main():
    with open(SRC, encoding="utf-8") as fh:
        text = fh.read()

    # package-level: declare the OCL delegate provider
    text = text.replace(
        '<eSubpackages name="core" nsURI="http://fame/core/1.0" nsPrefix="famecore">',
        '<eSubpackages name="core" nsURI="http://fame/core/1.0" nsPrefix="famecore">\n'
        f'    <eAnnotations source="{ECORE_ANN}">\n'
        f'      <details key="validationDelegates" value="{OCL_ANN}"/>\n'
        "    </eAnnotations>", 1)
    text = text.replace(
        '<eSubpackages name="correspondence" nsURI="http://fame/correspondence/1.0"\n'
        '      nsPrefix="famecorr">',
        '<eSubpackages name="correspondence" nsURI="http://fame/correspondence/1.0"\n'
        '      nsPrefix="famecorr">\n'
        f'    <eAnnotations source="{ECORE_ANN}">\n'
        f'      <details key="validationDelegates" value="{OCL_ANN}"/>\n'
        "    </eAnnotations>", 1)

    injected = 0
    for cls in INVARIANTS:
        pattern = re.compile(
            r'(<eClassifiers xsi:type="ecore:EClass" name="' + cls + r'"[^>]*?)(/>|>)',
            re.DOTALL)

        def repl(m):
            nonlocal injected
            injected += 1
            head, close = m.group(1), m.group(2)
            body = annotations_for(cls, 6)
            if close == "/>":
                return head + ">\n" + body + "\n    </eClassifiers>"
            return head + ">\n" + body

        text, n = pattern.subn(repl, text, count=1)
        if n == 0:
            print(f"  WARNING: class {cls} not found in fame.ecore")

    header = ("<!--\n"
              "  GENERATED by tools/make_delegates.py from fame.ecore + constraints.ocl.\n"
              "  Do not edit by hand; edit constraints.ocl and regenerate.\n"
              "  NOT YET VERIFIED IN ECLIPSE - see README, 'Verifying the delegate metamodel'.\n"
              "-->\n")
    text = text.replace("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
                        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + header, 1)

    with open(DST, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(DST, HERE)}  ({injected} classes annotated, "
          f"{sum(len(v) for v in INVARIANTS.values())} invariants)")


if __name__ == "__main__":
    main()
