# Synthetic artefact corpus

The provenance links in `scenario/scenario.xmi` cite these files. They are
**synthetic**: written for this artifact so that every `SourceArtefact.uri`
resolves and condition B1 can be checked end to end, rather than reproduced
from any real vendor documentation, which could not be redistributed.

They are deliberately partial, heterogeneous in form, and in one place
mutually inconsistent (the laboratory procedure manual contradicts the HL7
interface specification about `ValidatedTest`), because those are the
properties Section 3.1 of the paper attributes to real corpora.

`python3 tools/check_provenance.py` verifies that every cited URI resolves.
Note what this does and does not establish: it shows the reference is
resolvable, not that the artefact supports the element it justifies. That
gap is discussed in the paper's limitations.
