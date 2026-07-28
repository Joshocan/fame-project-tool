# EHR FHIR profile extract

## Observation
Profile of FHIR R5 `Observation`. A single clinical finding or measurement.
An Observation recorded from laboratory work is expected to be referenced by
a `DiagnosticReport`; a standalone Observation is accepted but is not
considered reportable.

## DiagnosticReport
Groups Observations produced by one diagnostic act. **contains** one or more
Observations.

## Clinician
Any professional recorded on an encounter, including nursing and allied
health staff. Broader than the GP system's authorising physician.

## PatientRecord
The aggregated record for one subject across contributing providers.
