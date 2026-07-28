# Patient identity across the three systems

All three systems represent the same individual:

- GP:  `Patient`        keyed by practice patient number
- LIS: `SubjectOfCare`  keyed by laboratory subject id
- EHR: `PatientRecord`  keyed by national identifier

Correlation is reliable only where the national identifier is present on all
three records. Where it is absent, correlation falls back to demographic
matching and is not safe for automated exchange. This is the compatibility
condition attached to the patient identity correspondence.
