# GP System REST API, v3

Base: `/api/v3`. All payloads JSON.

## POST /lab-requests
Creates a **LabRequest**. A lab request is the document a practitioner issues
to order laboratory work for a patient. Fields:

| field         | type   | required | notes                                  |
|---------------|--------|----------|----------------------------------------|
| `orderId`     | String | yes      | request identifier, unique per practice |
| `patientId`   | String | yes      | reference to a **Patient**              |
| `authorisedBy`| String | yes      | reference to a **Practitioner**         |
| `tests`       | array  | yes      | one or more **RequestedTest** entries   |

A LabRequest **isFor** exactly one Patient.

## GET /patients/{id}
Returns a **Patient**: the person receiving care. Identity is the practice
patient number plus, where recorded, the national identifier.

## Practitioner
The authorising physician. Only a Practitioner may sign a LabRequest.
This is narrower than a general "clinical staff member".

## Interfaces
The GP system (`GPSystem`) exposes this REST interface (`gp-rest-api`),
which exchanges LabRequest and Patient representations.
