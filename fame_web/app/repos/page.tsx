import { RepoUrlForm } from "@/components/RepoUrlForm";
import { RepositoryUploadForm } from "@/components/RepositoryUploadForm";

export default async function ReposPage() {
  return (
    <div className="stack">
      <section className="panel stack">
        <h2>Repository discovery</h2>
        <RepoUrlForm />
      </section>

      <section className="panel stack">
        <h2>Uploaded repository source</h2>
        <p className="muted">
          Upload a repository zip archive. FAME Web will extract it server-side and run the same artefact heuristics over the unpacked files.
        </p>
        <RepositoryUploadForm />
      </section>
    </div>
  );
}
