import Link from "next/link";
import type { Route } from "next";
import { listContainers } from "@/lib/storage";
import { CreateContainerButton } from "@/components/CreateContainerButton";

export default async function ContainersPage() {
  const containers = await listContainers();

  return (
    <div className="stack">
      <div className="panel stack">
        <div className="row spaceBetween">
          <div>
            <p className="eyebrow">Step 1 · Systems</p>
            <h2>Artefact systems</h2>
            <p className="muted">
              Each system holds a set of uploaded artefacts for a FAME
              experiment. Create a system, upload your files, then proceed to
              pipeline selection.
            </p>
          </div>
          <CreateContainerButton />
        </div>
      </div>

      {containers.length === 0 ? (
        <div className="panel stack">
          <p className="muted">No systems yet. Create one to get started.</p>
          <CreateContainerButton />
        </div>
      ) : (
        <div className="grid">
          {containers.map((container) => (
            <Link
              key={container.id}
              href={`/containers/${container.id}` as Route}
              className="card selectable stack"
            >
              <strong>{container.name}</strong>
              <p className="muted">
                {container.files.length} file
                {container.files.length !== 1 ? "s" : ""}
              </p>
              <p className="muted" style={{ fontSize: "0.78rem" }}>
                Created {new Date(container.createdAt).toLocaleString()}
              </p>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
