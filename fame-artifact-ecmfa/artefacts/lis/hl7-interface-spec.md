# Laboratory information system, HL7 v2 interface specification

## Inbound: ORM^O01 (order message)
Creates a **TestOrder**. One inbound order yields one TestOrder record and,
separately, one **Specimen** record per sample drawn. The specimen carries
its own accession number and is the unit tracked through the analyser.

`Specimen` **fulfils** the `TestOrder` it was drawn for.

## Concepts
- `TestOrder`      - the laboratory's record of work requested.
- `Specimen`       - the physical sample, separately identified.
- `LaboratoryTest` - a single assay on the laboratory catalogue.
- `SubjectOfCare`  - the person the specimen was drawn from.
- `ValidatedTest`  - carried in the OBX segment of the outbound ORU^R01.

## Outbound: ORU^R01 (result message)
Carries validated results. In this interface a validated test is **one
message segment among several** in the result message; it is not itself a
document.
