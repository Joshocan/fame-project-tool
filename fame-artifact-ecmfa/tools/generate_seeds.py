#!/usr/bin/env python3
"""
Generate the seeded-violation test set from scenario/scenario.xmi.

Each seed is the running scenario with exactly one deliberate corruption,
targeting one named well-formedness condition. Regenerate with:

    python3 tools/generate_seeds.py

Conditions A4 (acyclic containment) and A5 (owned attributes) are NOT seeded:
both are guaranteed by EMF containment semantics and by the Ecore structure
itself, so no well-typed instance can violate them. See the note in
tests/expected-violations.md.
"""

import copy
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "scenario", "scenario.xmi")
OUT = os.path.join(HERE, "tests", "seeded")

NS = {"famecore": "http://fame/core/1.0", "famecorr": "http://fame/correspondence/1.0"}
for p, u in NS.items():
    ET.register_namespace(p, u)
ET.register_namespace("xmi", "http://www.omg.org/XMI")
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")


def load():
    return ET.parse(SRC).getroot()


def find_id(root, ident):
    for e in root.iter():
        if e.get("id") == ident:
            return e
    raise KeyError(ident)


def corr(root, cid):
    return find_id(root, cid)


def cm(root):
    for e in root:
        if isinstance(e.tag, str) and e.tag.endswith("CorrespondenceModel"):
            return e
    raise KeyError("CorrespondenceModel")


# --- seed definitions: name -> (mutation, human description) ----------------

def s_A1(r):
    find_id(r, "M1.Patient").set("id", "M1.LabRequest")


def s_A2(r):
    find_id(r, "M1.LabRequest").set("name", "")


def s_A3(r):
    find_id(r, "M1.isFor").set("target", "M2.TestOrder")


def s_A6(r):
    find_id(r, "M1.servedBy").set("kind", "specialization")


def s_B1(r):
    e = find_id(r, "M1.Patient")
    for link in e.findall("justifiedBy"):
        e.remove(link)


def s_B2(r):
    find_id(r, "M1.Patient").set("confidence", "1.7")


def s_B3(r):
    e = find_id(r, "M1.Patient")
    for link in e.findall("justifiedBy"):
        if link.get("role") == "primary":
            link.set("role", "corroborating")


def s_C1(r):
    find_id(r, "M1.LabRequest").set("corroborationCount", "1")


def s_C2(r):
    find_id(r, "M2.ValidatedTest").set("confidence", "1.0")


def s_C3(r):
    find_id(r, "M2.ValidatedTest").set("corroborationCount", "1")


def s_T1(r):
    e = find_id(r, "M1.Practitioner")
    for link in e.findall("justifiedBy"):
        if link.get("role") == "primary":
            link.set("role", "corroborating")
    e.set("status", "trusted")


def s_D1(r):
    find_id(r, "M1.OnePatient").set("expression", "")


def s_D2(r):
    find_id(r, "M1.OnePatient").set("constrains", "M1.LabRequest M1.NoSuchElement")


def s_E1(r):
    parts = cm(r).findall("relates")
    parts[1].set("ref", "M1")


def s_E2(r):
    c = corr(r, "C3")
    c.remove(c.findall("ends")[1])


def s_E3(r):
    # paper Section 6.4: retarget both ends of C5 into the laboratory model
    ends = corr(r, "C5").findall("ends")
    ends[1].set("elementRef", "M2.Specimen")
    ends[1].set("inModel", "/3/@relates.1")


def s_E4(r):
    ends = corr(r, "C2").findall("ends")
    ends[2].set("elementRef", "M2.TestOrder")


def s_E5(r):
    corr(r, "C5").findall("ends")[0].set("elementRef", "M2.ValidatedTestX")


def s_E6(r):
    corr(r, "C1").set("kind", "refinement")


def s_E7(r):
    del corr(r, "C2").findall("ends")[0].attrib["role"]


def s_E7b(r):
    corr(r, "C5").findall("ends")[0].set("role", "abstracted")


def s_G1(r):
    c5 = corr(r, "C5")
    clone = copy.deepcopy(c5)
    clone.set("id", "C6")
    clone.set("name", "validatedtest-mismatches-observation")
    clone.set("kind", "mismatch")
    cm(r).append(clone)


SEEDS = [
    ("A1_UniqueIds", s_A1, "M1.Patient given the id of M1.LabRequest"),
    ("A2_NonEmptyName", s_A2, "M1.LabRequest name emptied"),
    ("A3_LocalEndpoints", s_A3, "M1.isFor retargeted to an element of M2"),
    ("A6_TypeConsistentSpecialization", s_A6, "M1.servedBy (DataConcept to Component) made a specialization"),
    ("B1_EverythingJustified", s_B1, "all provenance links removed from M1.Patient"),
    ("B2_BoundedConfidence", s_B2, "M1.Patient confidence set to 1.7"),
    ("B3_PrimaryEvidence", s_B3, "M1.Patient primary link demoted to corroborating"),
    ("C1_ConsistentCounting", s_C1, "M1.LabRequest count lowered below its 3 distinct artefacts"),
    ("C2_ContradictionCapsConfidence", s_C2, "M2.ValidatedTest confidence restored to 1.0 (paper Figure 3)"),
    ("C3_HighConfidenceNeedsCorroboration", s_C3, "M2.ValidatedTest count lowered to 1 at confidence 0.86"),
    ("T1_TrustedMeansEvidenced", s_T1, "M1.Practitioner marked trusted with no primary evidence"),
    ("D1_FormalHasExpression", s_D1, "M1.OnePatient formal=true with empty expression"),
    ("D2_TargetsExist", s_D2, "M1.OnePatient constrains a non-existent element"),
    ("E1_RelatesTwoModels", s_E1, "participating models M1 and M2 collapsed to the same ref"),
    ("E2_AtLeastTwoEnds", s_E2, "one end removed from C3"),
    ("E3_CrossesModels", s_E3, "both ends of C5 retargeted into M2"),
    ("E4_DistinctEndpoints", s_E4, "C2's two refined ends made identical"),
    ("E5_EndpointsResolve", s_E5, "C5 end points at a non-existent element of M2"),
    ("E6_DirectionMatchesKind", s_E6, "C1 kind changed to refinement while left undirected"),
    ("E7_RolesIffDirected", s_E7, "role removed from an end of the directed C2"),
    ("E7b_NoRolesIfUndirected", s_E7b, "role added to an end of the undirected C5"),
    ("G1_InternalConsistency", s_G1, "C6 asserts mismatch over C5's endpoint set"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for old in os.listdir(OUT):
        if old.endswith(".xmi"):
            os.remove(os.path.join(OUT, old))
    for name, mutate, desc in SEEDS:
        root = load()
        mutate(root)
        path = os.path.join(OUT, name + ".xmi")
        ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
        print(f"wrote {name}.xmi  ({desc})")
    print(f"\n{len(SEEDS)} seeded instances written to tests/seeded/")


if __name__ == "__main__":
    main()
