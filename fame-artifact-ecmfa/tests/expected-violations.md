# Seeded-violation test set

Each file in `seeded/` is `scenario/scenario.xmi` with exactly one deliberate
corruption. The filename names the condition the corruption targets.

Regenerate: `python3 tools/generate_seeds.py`
Check:      `python3 tests/run_tests.py`

## Reading the "also fires" column

Some conditions are not independent. `T1` embeds `B1`, `B3` and the `C`
conditions by construction, so any instance violating `T1` also violates one
of them; `E5` resolution depends on the participating-model declarations that
`E1` constrains. Where a seed trips more than one condition, the cascade is
recorded rather than engineered away, because engineering it away would mean
misrepresenting the dependency between the conditions.

15 of the 22 seeds are isolated (exactly one condition). 7 cascade.

| Seed file | Corruption | Also fires |
|---|---|---|
| `A1_UniqueIds.xmi` | `M1.Patient` given the id of `M1.LabRequest` | A3, D2, E5 |
| `A2_NonEmptyName.xmi` | `M1.LabRequest` name emptied | — |
| `A3_LocalEndpoints.xmi` | `M1.isFor` retargeted to `M2.TestOrder` | — |
| `A6_TypeConsistentSpecialization.xmi` | `M1.servedBy` (DataConcept → Component) made a specialization | — |
| `B1_EverythingJustified.xmi` | all provenance links removed from `M1.Patient` | T1 |
| `B2_BoundedConfidence.xmi` | `M1.Patient` confidence set to 1.7 | T1 |
| `B3_PrimaryEvidence.xmi` | `M1.Patient` primary link demoted to corroborating | T1 |
| `C1_ConsistentCounting.xmi` | `M1.LabRequest` count lowered below its 3 distinct artefacts | C3 |
| `C2_ContradictionCapsConfidence.xmi` | `M2.ValidatedTest` confidence restored to 1.0 | — |
| `C3_HighConfidenceNeedsCorroboration.xmi` | `M2.ValidatedTest` count lowered to 1 at confidence 0.86 | — |
| `T1_TrustedMeansEvidenced.xmi` | `M1.Practitioner` marked trusted with no primary evidence | B3 |
| `D1_FormalHasExpression.xmi` | `M1.OnePatient` formal=true with empty expression | — |
| `D2_TargetsExist.xmi` | `M1.OnePatient` constrains a non-existent element | — |
| `E1_RelatesTwoModels.xmi` | participating models M1 and M2 collapsed to one ref | E5 |
| `E2_AtLeastTwoEnds.xmi` | one end removed from C3 | E3 |
| `E3_CrossesModels.xmi` | both ends of C5 retargeted into M2 | — |
| `E4_DistinctEndpoints.xmi` | C2's two refined ends made identical | — |
| `E5_EndpointsResolve.xmi` | C5 end points at a non-existent element of M2 | — |
| `E6_DirectionMatchesKind.xmi` | C1 kind changed to refinement, left undirected | — |
| `E7_RolesIffDirected.xmi` | role removed from an end of the directed C2 | — |
| `E7b_NoRolesIfUndirected.xmi` | role added to an end of the undirected C5 | — |
| `G1_InternalConsistency.xmi` | C6 asserts mismatch over C5's endpoint set | — |

`C2_ContradictionCapsConfidence.xmi` is the instance shown in Figure 3 of the
paper.

## Conditions that are not seeded, and why

| Condition | Why no seed exists |
|---|---|
| **A4** acyclic containment | Guaranteed by EMF containment semantics: an object has at most one container and cannot transitively contain itself. No well-typed instance can violate it. |
| **A5** owned attributes | Guaranteed by the Ecore structure: `attributes` is a containment feature declared only on `DataConcept`, so an `Attribute` has no other possible container. The OCL invariant is vacuously true. |
| **D3** satisfiability | Semi-formal. Delegated to a SAT/SMT solver, which this artifact does not include. See "Known gaps" in the top-level README. |
| **T2** monotone promotion | Operation-level, not a property of a snapshot. Requires the update operations, which this artifact does not include. |
| **G2** preservation under update | Operation-level, as above. Theorem 1 covers the endpoint-preserving case. |

The paper currently states that "for every invariant, a seeded violation of
the running example is rejected with the invariant reported by name"
(Section 4.4). That is true for every invariant a well-typed instance can
violate, but not for A4 and A5, which no instance can violate. The sentence
should be qualified accordingly.
