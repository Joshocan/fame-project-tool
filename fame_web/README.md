# FAME Web

Lightweight web scaffold for the FAME experiment flow:

1. discover or inspect a GitHub healthcare repository
2. review and select source artefacts
3. choose a pipeline preset
4. create a run
5. inspect mock output, validation, and graph export placeholders

## Current scope

This scaffold is intentionally thin:

- mock repository catalog
- local JSON persistence in `storage/`
- pipeline preset selection
- mock output generation
- placeholder graph / Neo4j review

It does **not** yet call:

- real GitHub APIs
- real Claude / Anthropic APIs
- ChromaDB
- Neo4j
- the existing FAME CLI scripts

## Run locally

```bash
cd fame_web
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## Next integration steps

1. Replace mock repo catalog with GitHub search + repository tree fetch.
2. Persist artefact bundles with pinned commit SHAs from real repositories.
3. Add preprocessing status and CLI orchestration adapters.
4. Replace mock run execution with the existing FAME scripts.
5. Add ChromaDB and Neo4j integration paths.
