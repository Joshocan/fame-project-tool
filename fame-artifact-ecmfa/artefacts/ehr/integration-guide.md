# EHR integration guide (extract)

## 3. Receiving laboratory results
Results arriving from a laboratory are stored as Observations. Consumers
should not treat an Observation as reportable unless it is contained in a
DiagnosticReport.

## 5. Actors
The `Clinician` element on an encounter records who was present, not who
authorised any particular order. Systems mapping an external "practitioner"
onto Clinician must confirm the intended semantics first.

## 7. Patient identity
The record key is the national identifier where present. Contributing
providers may additionally carry local identifiers.
