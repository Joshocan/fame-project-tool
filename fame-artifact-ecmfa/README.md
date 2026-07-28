# FAME artifact

Companion artifact for *Construction Before Alignment: Trustworthy Models and
Correspondences for Semantic Interoperability* (Ocansey, Lamo, Rutle, Rabbi).

It contains the two metamodels, the well-formedness conditions as OCL, the
running healthcare scenario of Section 2, the seeded-violation test set, and
tools that decide the conditions intra-resource OCL cannot.

**Quick check** (no Eclipse needed, Python 3.8+, no dependencies):

```
python3 tests/run_tests.py
```

Expected: `25 passed, 0 failed`.

---

## Claim-to-file map

Every checkable claim in the paper, the file that carries it, and the command
that verifies it.

| § | Claim | File | Verify |
|---|---|---|---|
| 4.1–4.3 | Core metamodel: trust-bearing root, structural/behavioural layers, closed vs extensible value spaces | `metamodels/fame.ecore` (subpackage `core`) | open in Ecore editor |
| 4.4 | Conditions A1–A6, B1–B3, C1–C3, T1, D1–D2 as OCL invariants | `constraints/constraints.ocl` | `python3 tools/fame_validate.py scenario/scenario.xmi` |
| 4.4 | Every seedable invariant is rejected with its name reported | `tests/seeded/`, `tests/expected-violations.md` | `python3 tests/run_tests.py` |
| 4.4, Fig. 3 | C2 rejects `ValidatedTest` at confidence 1.0 with a contradicting link | `tests/seeded/C2_ContradictionCapsConfidence.xmi` | `python3 tools/fame_validate.py tests/seeded/C2_*.xmi` |
| 2, Fig. 1 | Three system models M1–M3 and correspondences C1–C5 | `scenario/scenario.xmi` | `python3 tools/fame_validate.py scenario/scenario.xmi` |
| 4.1, B1 | Every element cites resolvable evidence | `artefacts/`, `scenario/scenario.xmi` | `python3 tools/check_provenance.py scenario/scenario.xmi` |
| 6.1–6.2 | Correspondences are `ModelElement`s; n-ary ends; kind-labelled | `metamodels/fame.ecore` (subpackage `correspondence`) | open in Ecore editor |
| 6.3, Fig. 6 | Typed notation; checking a statement and validating the instance are the same judgement | `scenario/correspondences.corr`, `tools/corr_notation.py` | `python3 tools/corr_notation.py check scenario/correspondences.corr` |
| 6.3 | The notation and the metamodel instance agree | both of the above | `python3 tests/run_tests.py` (last two assertions) |
| 6.4 | Conditions E1–E7, G1 | `constraints/constraints.ocl` | `python3 tests/run_tests.py` |
| 6.4, 6.5 | E5 and E7 decided by the resolver | `tools/fame_validate.py` | `python3 tools/fame_validate.py tests/seeded/E5_*.xmi` |
| 6.5 | Invariants also available as EMF validation delegates | `metamodels/fame-delegates.ecore` | **unverified, see below** |

Two rows in Section 6.5 of the paper have **no** entry here, because the
machinery does not exist: the SAT/SMT solver for D3, and the update
operations for T2 and G2. See "Known gaps".

---

## Layout

```
metamodels/
  fame.ecore              core + correspondence metamodels (normative)
  fame-delegates.ecore    generated: same, with OCL validation delegates
constraints/
  constraints.ocl         all invariants, Complete OCL (normative)
scenario/
  scenario.xmi            M1, M2, M3 and the C1..C5 correspondence model
  correspondences.corr    the same correspondences in the Section 6.3 notation
artefacts/                synthetic source corpus cited by provenance links
tests/
  seeded/                 22 instances, one deliberate violation each
  expected-violations.md  what each seed violates, and what is not seedable
  run_tests.py            the suite
tools/
  fame_validate.py        reference validator + the resolver (E5, E7)
  corr_notation.py        notation parser, serialiser, XMI projection
  generate_seeds.py       regenerates tests/seeded/
  make_delegates.py       regenerates fame-delegates.ecore
  check_provenance.py     B1 resolvability check
figures/                  core-mm and corr-mm, vector versions
```

---

## Running the checks in Eclipse

Tested configuration: Eclipse Modeling Tools 2024-03, EMF 2.36, OCL 6.20.

1. Import the repository as an existing project.
2. Right-click `constraints/constraints.ocl` → **OCL → Load Document**, and
   select `metamodels/fame.ecore` when prompted for the metamodel.
3. Open `scenario/scenario.xmi` in the sample Ecore editor.
4. Select the root, then **Edit → Validate**. Expect no diagnostics.
5. Repeat on any file in `tests/seeded/`. Expect the diagnostic to name the
   invariant matching the filename.

### Verifying the delegate metamodel

`metamodels/fame-delegates.ecore` was generated mechanically by
`tools/make_delegates.py` and **has not been opened in Eclipse**. Before
releasing this artifact:

1. Open it in the Ecore editor and confirm it loads without errors.
2. Point `scenario/scenario.xmi`'s `xsi:schemaLocation` at it.
3. Run **Validate**; confirm the invariants fire with no OCL parse errors.
4. If any invariant fails to parse, fix it in `constraints/constraints.ocl`
   and rerun `python3 tools/make_delegates.py`.

Until step 3 passes, the paper should not claim the delegate form exists.

---

## The Python tools

`tools/fame_validate.py` re-implements every OCL invariant so the test set can
be checked without Eclipse. The OCL document stays normative; the Python is a
cross-check, and `tests/run_tests.py` asserts the two agree on which condition
each seed violates.

It also *is* the resolver referred to in Sections 6.4 and 6.5. Conditions E5
(endpoint references resolve) and E7 (roles iff directed) need the
participating system models loaded, which intra-resource OCL cannot do. They
are decided here.

`tools/corr_notation.py` parses the Section 6.3 grammar, enforces its typing
rules at parse time, and projects an XMI correspondence model back onto the
notation. The round-trip is asserted by the test suite.

The notation is a **projection**, not a bijection: it carries kind, ends,
roles, compatibility conditions, evidence, confidence and status, but not
`name`, `corroborationCount`, or a condition's `formal` flag. The paper's
claim that each notation clause maps onto one metamodel construct holds in
that direction; it does not claim the reverse, and the reverse is false.

---

## Known gaps

Stated plainly, because the paper currently implies otherwise in places
(see `PAPER-INCONSISTENCIES.md`, item 8).

1. **No SAT/SMT solver.** Condition D3 (satisfiability of a model's formal
   constraint rules) is not decided anywhere in this artifact. Only the
   necessary typing condition D1 is checked.
2. **No update operations.** Conditions T2 (monotone promotion) and G2
   (preservation under update) are properties of operations, not snapshots.
   Neither is implemented, so Theorem 1 is not exercised computationally.
3. **No construction pipeline.** The LLM-based construction, shortlisting,
   validation and repair loop of Section 5 is not in this artifact. It
   belongs to the enabling paper and should be a separate archived record,
   cross-linked from this one.
4. **Provenance resolvability is not provenance faithfulness.**
   `check_provenance.py` confirms that a cited artefact exists. It cannot
   confirm that the artefact supports the element citing it, and nothing
   here calibrates the `confidence` values, which are authored. This is the
   framework's main open problem and should be stated as such in the paper.
5. **The artefact corpus is synthetic.** Written for this artifact, not
   extracted from real clinical systems.

---

## Reproducing derived files

```
python3 tools/generate_seeds.py                    # tests/seeded/
python3 tools/make_delegates.py                    # metamodels/fame-delegates.ecore
python3 tools/corr_notation.py fromxmi scenario/scenario.xmi > scenario/correspondences.corr
```

## Licence

Models, constraints, scenario, artefact corpus and documentation:
CC-BY-4.0. Tools under `tools/` and `tests/`: Apache-2.0. See `LICENSE`.
