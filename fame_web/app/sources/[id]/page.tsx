import { notFound } from "next/navigation";
import { ArtefactSelector } from "@/components/ArtefactSelector";
import { extractUploadedSourceArtefacts, fetchUploadedSource } from "@/lib/sources";

export default async function UploadedSourcePage({
  params
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const source = await fetchUploadedSource(id);
  if (!source) {
    notFound();
  }

  const artefacts = await extractUploadedSourceArtefacts(source);

  return (
    <div className="stack">
      <section className="panel stack">
        <p className="eyebrow">Uploaded source</p>
        <h2>{source.displayName}</h2>
        <p className="muted">
          Archive: {source.archiveFilename} · Created: {new Date(source.createdAt).toLocaleString()}
        </p>
        <p className="muted">Commit surrogate: {source.commitSha}</p>
      </section>

      <section className="panel stack">
        <h3>How uploaded repository artefacts are extracted</h3>
        <ul className="list">
          <li>The uploaded zip is stored server-side.</li>
          <li>The archive is unzipped into a dedicated extraction directory.</li>
          <li>The extracted file tree is scanned recursively.</li>
          <li>The same deterministic artefact heuristics used for GitHub repos are applied locally.</li>
        </ul>
      </section>

      {artefacts.length > 0 ? (
        <ArtefactSelector
          repoKey={`upload/${source.displayName}`}
          commitSha={source.commitSha}
          sourceType="upload"
          sourceId={source.id}
          artefacts={artefacts}
        />
      ) : (
        <section className="panel stack">
          <h3>No candidate artefacts found</h3>
          <p className="muted">
            The archive was extracted, but no files matched the current artefact heuristics strongly enough.
          </p>
        </section>
      )}
    </div>
  );
}
