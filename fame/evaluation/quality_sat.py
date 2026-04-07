from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET


@dataclass
class SATQuality:
    satisfiable: Optional[bool]
    dead_features: Optional[list[str]]
    core_features: Optional[list[str]]
    products_count: Optional[int]


FEATURE_TAGS = {"feature", "and", "or", "alt"}
FORMULA_TAGS = {"var", "not", "imp", "eq", "conj", "disj"}


def _load_sat_backend():
    try:
        from pysat.formula import IDPool
        from pysat.solvers import Solver
    except Exception as e:
        raise ImportError(
            "SAT quality analysis requires python-sat. Install with: pip install python-sat[pblib,aiger]"
        ) from e
    return IDPool, Solver


class _Encoder:
    def __init__(self) -> None:
        IDPool, _ = _load_sat_backend()
        self.pool = IDPool()
        self.clauses: list[list[int]] = []
        self.features: list[str] = []

    def var(self, name: str) -> int:
        if name not in self.features:
            self.features.append(name)
        return int(self.pool.id(name))

    def aux(self, prefix: str = "aux") -> int:
        return int(self.pool.id(f"{prefix}_{self.pool.top + 1}"))

    def add(self, *lits: int) -> None:
        self.clauses.append([int(l) for l in lits])


def _named_children(node: ET.Element) -> list[ET.Element]:
    return [ch for ch in node if ch.tag in FEATURE_TAGS]


def _child_var(node: ET.Element, enc: _Encoder) -> tuple[Optional[int], str]:
    name = (node.attrib.get("name") or "").strip()
    if name:
        return enc.var(name), name
    return None, ""


def _encode_tree(node: ET.Element, enc: _Encoder, *, parent_lit: Optional[int]) -> None:
    node_lit, _ = _child_var(node, enc)
    current_parent = node_lit if node_lit is not None else parent_lit

    if parent_lit is not None and node_lit is not None:
        # child -> parent
        enc.add(-node_lit, parent_lit)
        # mandatory child under an AND/feature parent
        if node.attrib.get("mandatory", "").lower() == "true":
            enc.add(-parent_lit, node_lit)

    children = _named_children(node)
    child_lits = []
    for ch in children:
        lit, _ = _child_var(ch, enc)
        if lit is not None:
            child_lits.append(lit)

    if current_parent is not None and child_lits:
        if node.tag in {"feature", "and"}:
            pass
        elif node.tag == "or":
            # parent selected => at least one child
            enc.add(-current_parent, *child_lits)
            for lit in child_lits:
                enc.add(-lit, current_parent)
        elif node.tag == "alt":
            enc.add(-current_parent, *child_lits)
            for lit in child_lits:
                enc.add(-lit, current_parent)
            for i in range(len(child_lits)):
                for j in range(i + 1, len(child_lits)):
                    enc.add(-child_lits[i], -child_lits[j])

    for ch in children:
        _encode_tree(ch, enc, parent_lit=current_parent)


def _formula_lit(node: ET.Element, enc: _Encoder) -> int:
    tag = node.tag
    if tag == "var":
        name = (node.text or "").strip()
        if not name:
            raise ValueError("Empty <var> in constraint")
        return enc.var(name)
    if tag == "not":
        if len(node) != 1:
            raise ValueError("<not> must have exactly one child")
        child = _formula_lit(node[0], enc)
        aux = enc.aux("not")
        enc.add(-aux, -child)
        enc.add(aux, child)
        return aux
    if tag in {"conj", "disj"}:
        if len(node) < 1:
            raise ValueError(f"<{tag}> must have children")
        lits = [_formula_lit(ch, enc) for ch in node if ch.tag in FORMULA_TAGS]
        if not lits:
            raise ValueError(f"<{tag}> contains no formula children")
        aux = enc.aux(tag)
        if tag == "conj":
            for lit in lits:
                enc.add(-aux, lit)
            enc.add(aux, *[-lit for lit in lits])
        else:
            enc.add(-aux, *lits)
            for lit in lits:
                enc.add(aux, -lit)
        return aux
    if tag == "imp":
        if len(node) != 2:
            raise ValueError("<imp> must have exactly two children")
        left = _formula_lit(node[0], enc)
        right = _formula_lit(node[1], enc)
        aux = enc.aux("imp")
        # aux <-> (~left or right)
        enc.add(-aux, -left, right)
        enc.add(aux, left)
        enc.add(aux, -right)
        return aux
    if tag == "eq":
        if len(node) != 2:
            raise ValueError("<eq> must have exactly two children")
        left = _formula_lit(node[0], enc)
        right = _formula_lit(node[1], enc)
        aux = enc.aux("eq")
        # aux <-> (left <-> right)
        enc.add(-aux, -left, right)
        enc.add(-aux, left, -right)
        enc.add(aux, left, right)
        enc.add(aux, -left, -right)
        return aux
    raise ValueError(f"Unsupported constraint tag: <{tag}>")


def _encode_constraints(root: ET.Element, enc: _Encoder) -> None:
    constraints = root.find("constraints")
    if constraints is None:
        return
    for rule in constraints.findall("rule"):
        formula_nodes = [ch for ch in rule if ch.tag in FORMULA_TAGS]
        if not formula_nodes:
            continue
        if len(formula_nodes) != 1:
            raise ValueError("<rule> must contain exactly one formula root")
        enc.add(_formula_lit(formula_nodes[0], enc))


def _resolve_named_root(node: ET.Element) -> ET.Element:
    current = node
    while not (current.attrib.get("name") or "").strip():
        children = [ch for ch in current if ch.tag in FEATURE_TAGS]
        if len(children) != 1:
            break
        current = children[0]
    return current


def _build_cnf(xml_path: Path) -> tuple[_Encoder, list[int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        raise ValueError("<struct> not found in feature model")

    top = [ch for ch in struct if ch.tag in FEATURE_TAGS]
    if len(top) != 1:
        raise ValueError("Expected exactly one root feature/group in <struct>")

    enc = _Encoder()
    root_node = _resolve_named_root(top[0])
    root_name = (root_node.attrib.get("name") or "").strip()
    if not root_name:
        raise ValueError("Root feature/group is missing a name")
    root_lit = enc.var(root_name)
    enc.add(root_lit)

    _encode_tree(root_node, enc, parent_lit=None)
    _encode_constraints(root, enc)
    feature_vars = [enc.var(name) for name in enc.features]
    return enc, feature_vars


def _solve(clauses: Iterable[list[int]], assumptions: Optional[list[int]] = None) -> bool:
    _, Solver = _load_sat_backend()
    with Solver(name="g3") as solver:
        for clause in clauses:
            solver.add_clause(list(clause))
        return bool(solver.solve(assumptions=assumptions or []))


def _count_models(clauses: list[list[int]], feature_vars: list[int]) -> int:
    _, Solver = _load_sat_backend()
    count = 0
    with Solver(name="g3") as solver:
        for clause in clauses:
            solver.add_clause(list(clause))
        while solver.solve():
            count += 1
            model = set(solver.get_model())
            blocking = [(-v if v in model else v) for v in feature_vars]
            solver.add_clause(blocking)
    return count


def analyze_sat_quality(xml_path: str | Path, *, compute_products: bool = False) -> SATQuality:
    xml_path = Path(xml_path).expanduser()
    enc, feature_vars = _build_cnf(xml_path)

    satisfiable = _solve(enc.clauses)
    if not satisfiable:
        return SATQuality(
            satisfiable=False,
            dead_features=None,
            core_features=None,
            products_count=0 if compute_products else None,
        )

    dead: list[str] = []
    core: list[str] = []
    for name in enc.features:
        lit = enc.var(name)
        if not _solve(enc.clauses, assumptions=[lit]):
            dead.append(name)
        if not _solve(enc.clauses, assumptions=[-lit]):
            core.append(name)

    products = None
    if compute_products:
        products = _count_models(enc.clauses, feature_vars)

    return SATQuality(
        satisfiable=True,
        dead_features=sorted(dead),
        core_features=sorted(core),
        products_count=products,
    )
