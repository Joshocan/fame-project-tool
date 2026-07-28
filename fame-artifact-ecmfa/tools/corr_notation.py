#!/usr/bin/env python3
"""
Parser and serialiser for the typed correspondence notation of Section 6.3.

Implements the grammar of Figure 6:

    CorrModel ::= 'models' (Alias '=' ModelId)+ Corr*
    Corr      ::= CorrId ':' Kind
                  End (('~' | '->' | ',') End)+
                  ('when' STRING ('must' | 'should'))*
                  'evidence'   ArtefactId (',' ArtefactId)*
                  'confidence' REAL
                  'status'     ('candidate'|'validated'|'trusted'|'rejected')
    End       ::= Alias '::' ElementId ('[' Role ']')?

Two deviations from Figure 6 as printed, both deliberate:

  * ',' is accepted as an end separator. The paper's own C2 example uses it
    for the third end of a ternary correspondence, but the printed grammar
    lists only '~' and '->'. The grammar in the paper should be corrected.
  * Parsing enforces the typing rules of Section 6.4 (E2, E3, E4, E6, E7) at
    parse time, so a statement that does not type is rejected with the
    condition named, exactly as the corresponding XMI instance would be.

The notation is a *projection* of the metamodel, not a bijection: it carries
kind, ends, roles, compatibility conditions, evidence, confidence and status,
but not `name`, `corroborationCount`, or a condition's `formal` flag. from_xmi()
therefore projects onto the notation's vocabulary before comparison.

Usage:
    python3 tools/corr_notation.py check   scenario/correspondences.corr
    python3 tools/corr_notation.py fromxmi scenario/scenario.xmi
"""

import re
import sys
import xml.etree.ElementTree as ET

KINDS_UNDIRECTED = {"equivalence", "overlap", "mismatch"}
KINDS_DIRECTED = {"refinement", "generalization", "dependency"}
KINDS = KINDS_UNDIRECTED | KINDS_DIRECTED
ROLES_FOR = {
    "refinement": {"abstracted", "refined"},
    "generalization": {"abstracted", "refined"},
    "dependency": {"required", "dependent"},
}
STATUSES = {"candidate", "validated", "trusted", "rejected"}


class NotationError(Exception):
    pass


# ---------------------------------------------------------------- parsing --

_END = re.compile(r"^([A-Za-z_]\w*)::([\w.]+)(?:\[(\w+)\])?$")


def _qualify(local, model_ref):
    """Notation writes element names local to the aliased model (gp::LabRequest);
    the XMI carries model-qualified ids (M1.LabRequest). Add the prefix back."""
    return local if local.startswith(model_ref + ".") else f"{model_ref}.{local}"


def _localise(element_id, model_ref):
    """Inverse of _qualify."""
    prefix = model_ref + "."
    return element_id[len(prefix):] if element_id.startswith(prefix) else element_id


def parse_text(text):
    lines = []
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if line:
            lines.append(line)
    if not lines or not lines[0].startswith("models"):
        raise NotationError("a correspondence model must open with a 'models' declaration")

    models = {}
    decls = re.findall(r"(\w+)\s*=\s*([\w.]+)", lines[0][len("models"):])
    if not decls:
        raise NotationError(f"malformed model declaration line: {lines[0]}")
    for alias, ref in decls:
        models[alias] = ref
    if len(models) < 2 or len(set(models.values())) < 2:
        raise NotationError("E1_RelatesTwoModels: fewer than two distinct participating models")

    corrs, current = [], None
    for line in lines[1:]:
        head = re.match(r"^(\w+)\s*:\s*(\w+)$", line)
        if head:
            if current:
                corrs.append(_finish(current, models))
            cid, kind = head.group(1), head.group(2)
            if kind not in KINDS:
                raise NotationError(f"{cid}: '{kind}' is not one of the six correspondence kinds")
            current = {"id": cid, "kind": kind, "ends": [], "when": [],
                       "evidence": [], "confidence": None, "status": None}
            continue
        if current is None:
            raise NotationError(f"clause outside any correspondence: '{line}'")

        if line.startswith("when"):
            m = re.match(r"^when\s+'([^']*)'\s+(must|should)$", line)
            if not m:
                raise NotationError(f"{current['id']}: malformed 'when' clause: {line}")
            current["when"].append((m.group(1), m.group(2)))
        elif line.startswith("evidence"):
            current["evidence"] = [a.strip() for a in line[len("evidence"):].split(",") if a.strip()]
        elif line.startswith("confidence"):
            current["confidence"] = float(line.split(None, 1)[1])
        elif line.startswith("status"):
            st = line.split(None, 1)[1].strip()
            if st not in STATUSES:
                raise NotationError(f"{current['id']}: '{st}' is not a TrustStatus literal")
            current["status"] = st
        else:
            for tok in re.split(r"~|->|,", line):
                tok = tok.strip()
                if not tok:
                    continue
                m = _END.match(tok)
                if not m:
                    raise NotationError(f"{current['id']}: malformed end '{tok}'")
                alias, elem, role = m.groups()
                if alias not in models:
                    raise NotationError(f"{current['id']}: undeclared model alias '{alias}'")
                current["ends"].append({"model": alias,
                                        "element": _qualify(elem, models[alias]),
                                        "role": role})
    if current:
        corrs.append(_finish(current, models))
    return {"models": models, "corrs": corrs}


def _finish(c, models):
    cid, kind, ends = c["id"], c["kind"], c["ends"]
    if len(ends) < 2:
        raise NotationError(f"E2_AtLeastTwoEnds: {cid} has {len(ends)} end(s)")
    if len({e["model"] for e in ends}) < 2:
        raise NotationError(f"E3_CrossesModels: {cid} does not span two participating models")
    keys = [(e["model"], e["element"]) for e in ends]
    if len(set(keys)) != len(keys):
        raise NotationError(f"E4_DistinctEndpoints: {cid} repeats an endpoint")
    directed = kind in KINDS_DIRECTED
    if directed and any(e["role"] is None for e in ends):
        raise NotationError(f"E7_RolesIffDirected: {cid} is directed but an end has no role")
    if not directed and any(e["role"] is not None for e in ends):
        raise NotationError(f"E7b_NoRolesIfUndirected: {cid} is undirected but an end carries a role")
    if directed:
        allowed = ROLES_FOR[kind]
        for e in ends:
            if e["role"] not in allowed:
                raise NotationError(
                    f"E6_DirectionMatchesKind: role '{e['role']}' is not valid for kind {kind}")
    if c["confidence"] is None or not (0.0 <= c["confidence"] <= 1.0):
        raise NotationError(f"B2_BoundedConfidence: {cid} confidence={c['confidence']}")
    if not c["evidence"]:
        raise NotationError(f"B1_EverythingJustified: {cid} cites no evidence")
    if c["status"] is None:
        raise NotationError(f"{cid}: missing status clause")
    c["directed"] = directed
    return c


def parse_file(path):
    with open(path, encoding="utf-8") as fh:
        return parse_text(fh.read())


# ------------------------------------------------------------ serialising --

def to_text(model):
    out = ["models " + "  ".join(f"{a} = {r}" for a, r in model["models"].items()), ""]
    for c in model["corrs"]:
        out.append(f"{c['id']} : {c['kind']}")
        sep = " ~ "
        rendered = [
            f"{e['model']}::{_localise(e['element'], model['models'][e['model']])}"
            + (f"[{e['role']}]" if e["role"] else "")
            for e in c["ends"]
        ]
        if c["directed"]:
            out.append("    " + rendered[0] + " -> " + ", ".join(rendered[1:]))
        else:
            out.append("    " + sep.join(rendered))
        for expr, modality in c["when"]:
            out.append(f"    when '{expr}' {modality}")
        out.append("    evidence   " + ", ".join(c["evidence"]))
        out.append(f"    confidence {c['confidence']}")
        out.append(f"    status     {c['status']}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------- XMI projection --

def from_xmi(path):
    root = ET.parse(path).getroot()
    cm = None
    for top in root:
        if isinstance(top.tag, str) and top.tag.endswith("CorrespondenceModel"):
            cm = top
    if cm is None:
        raise NotationError("no CorrespondenceModel in " + path)

    parts = cm.findall("relates")
    models = {p.get("alias"): p.get("ref") for p in parts}
    alias_at = {i: p.get("alias") for i, p in enumerate(parts)}

    corrs = []
    for c in cm.findall("correspondences"):
        ends = []
        for e in c.findall("ends"):
            idx = int((e.get("inModel") or "").split("@relates.")[-1])
            ends.append({"model": alias_at[idx], "element": e.get("elementRef"),
                         "role": e.get("role")})
        corrs.append({
            "id": c.get("id"),
            "kind": c.get("kind"),
            "directed": c.get("directed") == "true",
            "ends": ends,
            "when": [(w.get("expression"), "must" if w.get("mustHold") == "true" else "should")
                     for w in c.findall("conditionedBy")],
            "evidence": [p.get("evidence") for p in c.findall("justifiedBy")],
            "confidence": float(c.get("confidence")),
            "status": c.get("status"),
        })
    return {"models": models, "corrs": corrs}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd, path = sys.argv[1], sys.argv[2]
    try:
        if cmd == "check":
            m = parse_file(path)
            print(f"OK   {path}: {len(m['corrs'])} correspondence(s) over "
                  f"{len(m['models'])} models, all typing rules satisfied")
        elif cmd == "fromxmi":
            print(to_text(from_xmi(path)))
        else:
            print(__doc__)
            return 2
    except NotationError as exc:
        print(f"FAIL {path}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
