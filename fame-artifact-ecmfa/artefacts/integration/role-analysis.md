# Actor role analysis

The GP system's `Practitioner` is the physician who authorised a request.
The EHR's `Clinician` is any professional recorded on an encounter.

The extensions overlap but neither contains the other: a nurse may be a
Clinician without ever being a Practitioner, and a referring physician from
outside the practice may authorise without appearing on any encounter.

**Conclusion.** These must not be identified. An integration that maps one
onto the other will silently attribute authorisation to staff who did not
authorise. Record as a mismatch; reconcile explicitly before exchanging any
actor-level data.
