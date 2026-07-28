# Integration mapping notes

## GP request to laboratory order
A GP `LabRequest` does not map one-to-one. It becomes a laboratory
`TestOrder` **plus** one `Specimen`. Any implementation must preserve
specimen identity across the split; collapsing the two loses the accession
number and breaks result routing.

## Requested test to laboratory test
`RequestedTest` and `LaboratoryTest` denote the same catalogue assay under
different names. Direct exchange is safe.

## Validated test to observation
The laboratory's `ValidatedTest` and the EHR's `Observation` carry the same
clinical fact. Exchange is safe only where the Observation is contained in a
DiagnosticReport; otherwise the reportability semantics differ.

## Patient
GP `Patient`, laboratory `SubjectOfCare` and EHR `PatientRecord` denote the
same individual. See the patient identity note.
