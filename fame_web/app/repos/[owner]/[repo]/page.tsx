import { ArtefactSelector } from "@/components/ArtefactSelector";
import { fetchRepository, fetchRepositoryArtefacts } from "@/lib/github";

const extractionRules = [
  "Fetch repository metadata and pin the default-branch commit SHA.",
  "Retrieve the recursive Git tree from GitHub.",
  "Prefer README, docs, API specs, workflow guides, and schema-like files.",
  "Exclude generated assets, binaries, build outputs, and dependency trees.",
  "Require human review before an artefact enters the model-construction bundle."
];

export default async function RepoDetailPage({
  params
}: {
  params: Promise<{ owner: string; repo: string }>;
}) {
  const { owner, repo } = await params;
  const repoSummary = await fetchRepository(owner, repo);
  const artefacts = await fetchRepositoryArtefacts(owner, repo);

  return (
    <div className="stack">
      <section className="panel stack">
        <p className="eyebrow">Step 1 · Choose repo</p>
        <h2>
          {repoSummary?.owner}/{repoSummary?.name}
        </h2>
        <p>{repoSummary?.description}</p>
        <p className="muted">
          Domain: {repoSummary?.domain} · Default branch: {repoSummary?.defaultBranch} · Commit: {repoSummary?.commitSha}
        </p>
        <p className="muted">
          Stars: {repoSummary?.stars} {repoSummary?.topics.length ? `· Topics: ${repoSummary?.topics.join(", ")}` : ""}
        </p>
      </section>

      <section className="panel stack">
        <h3>How artefacts are extracted</h3>
        <p className="muted">
          The source layer now queries GitHub directly, retrieves the repository tree, and applies deterministic heuristics before showing candidate artefacts for human review.
        </p>
        <ul className="list">
          {extractionRules.map((rule) => (
            <li key={rule}>{rule}</li>
          ))}
        </ul>
      </section>

      {artefacts.length > 0 ? (
        <ArtefactSelector
          repoKey={`${repoSummary?.owner}/${repoSummary?.name}`}
          commitSha={repoSummary?.commitSha ?? "pending-github-integration"}
          artefacts={artefacts}
        />
      ) : (
        <section className="panel stack">
          <h3>No candidate artefacts found</h3>
          <p className="muted">
            GitHub retrieval succeeded, but no files matched the current extraction heuristics strongly enough. Relax the heuristics or inspect the repository manually.
          </p>
        </section>
      )}
    </div>
  );
}
