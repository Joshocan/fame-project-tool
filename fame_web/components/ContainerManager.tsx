"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { Route } from "next";
import { ContainerRecord } from "@/lib/models";

type Props = {
  container: ContainerRecord;
};

export function ContainerManager({ container: initial }: Props) {
  const router = useRouter();
  const [container, setContainer] = useState(initial);
  const [editingName, setEditingName] = useState(false);
  const [nameInput, setNameInput] = useState(initial.name);
  const [selected, setSelected] = useState<string[]>(initial.files.map((f) => f.id));
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function saveName() {
    const trimmed = nameInput.trim();
    if (!trimmed || trimmed === container.name) {
      setEditingName(false);
      setNameInput(container.name);
      return;
    }
    const res = await fetch(`/api/containers/${container.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: trimmed })
    });
    if (res.ok) {
      const updated = (await res.json()) as ContainerRecord;
      setContainer(updated);
      setNameInput(updated.name);
    }
    setEditingName(false);
  }

  async function uploadFiles(files: FileList) {
    setUploading(true);
    setError("");
    const formData = new FormData();
    for (const file of files) formData.append("files", file);

    const res = await fetch(`/api/containers/${container.id}/files`, {
      method: "POST",
      body: formData
    });
    setUploading(false);

    if (!res.ok) {
      const payload = await res.json().catch(() => ({ error: "Upload failed." }));
      setError(payload.error ?? "Upload failed.");
      return;
    }

    const updated = (await res.json()) as ContainerRecord;
    setContainer(updated);
    setSelected(updated.files.map((f) => f.id));
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function deleteFile(fileId: string) {
    const res = await fetch(`/api/containers/${container.id}/files/${fileId}`, {
      method: "DELETE"
    });
    if (res.ok) {
      const updated = (await res.json()) as ContainerRecord;
      setContainer(updated);
      setSelected((s) => s.filter((id) => id !== fileId));
    }
  }

  function toggle(fileId: string) {
    setSelected((s) =>
      s.includes(fileId) ? s.filter((id) => id !== fileId) : [...s, fileId]
    );
  }

  function toggleAll(checked: boolean) {
    setSelected(checked ? container.files.map((f) => f.id) : []);
  }

  async function proceed() {
    if (selected.length === 0) {
      setError("Select at least one artefact to continue.");
      return;
    }
    setSubmitting(true);
    setError("");

    const res = await fetch("/api/bundles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repoKey: container.name,
        commitSha: `container-${container.id}`,
        selectedArtefactPaths: selected,
        sourceType: "container",
        sourceId: container.id
      })
    });

    setSubmitting(false);
    if (!res.ok) {
      setError("Failed to create bundle.");
      return;
    }

    const { id } = (await res.json()) as { id: string };
    router.push(`/runs/new?bundleId=${id}` as Route);
  }

  const allSelected =
    container.files.length > 0 && selected.length === container.files.length;

  return (
    <div className="stack">
      {/* Name */}
      <div className="panel stack">
        {editingName ? (
          <div className="row" style={{ gap: "0.5rem" }}>
            <input
              className="input"
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") saveName();
                if (e.key === "Escape") {
                  setEditingName(false);
                  setNameInput(container.name);
                }
              }}
              autoFocus
            />
            <button className="button primary" onClick={saveName}>Save</button>
            <button
              className="button"
              onClick={() => {
                setEditingName(false);
                setNameInput(container.name);
              }}
            >
              Cancel
            </button>
          </div>
        ) : (
          <div className="row spaceBetween">
            <h2>{container.name}</h2>
            <button className="button ghost" onClick={() => setEditingName(true)}>
              Rename
            </button>
          </div>
        )}
        <p className="muted">
          Created {new Date(container.createdAt).toLocaleString()} &middot;{" "}
          {container.files.length} file{container.files.length !== 1 ? "s" : ""}
        </p>
      </div>

      {/* Upload */}
      <div className="panel stack">
        <h3>Upload artefacts</h3>
        <p className="muted">
          Add PDF, text, markdown, or other artefact files. Multiple files can be
          selected at once.
        </p>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="input"
          onChange={(e) => {
            if (e.target.files?.length) uploadFiles(e.target.files);
          }}
        />
        {uploading && <p className="muted">Uploading...</p>}
      </div>

      {/* File list */}
      <div className="panel stack">
        <div className="row spaceBetween">
          <h3>Artefacts in this system</h3>
          <span className="badge">{selected.length} / {container.files.length} selected</span>
        </div>

        {container.files.length === 0 ? (
          <p className="muted">No artefacts yet — upload files above.</p>
        ) : (
          <>
            <label className="row" style={{ gap: "0.45rem" }}>
              <input
                type="checkbox"
                checked={allSelected}
                onChange={(e) => toggleAll(e.currentTarget.checked)}
              />
              <span>Select / deselect all</span>
            </label>

            <div className="stack" style={{ gap: "0.5rem", marginTop: "0.25rem" }}>
              {container.files.map((file) => (
                <div
                  key={file.id}
                  className="artefactRow row spaceBetween"
                  style={{ cursor: "default" }}
                >
                  <label
                    className="row"
                    style={{ gap: "0.6rem", flex: 1, cursor: "pointer" }}
                  >
                    <input
                      type="checkbox"
                      checked={selected.includes(file.id)}
                      onChange={() => toggle(file.id)}
                    />
                    <div>
                      <strong className="artefactPath">{file.filename}</strong>
                      <p className="muted artefactMeta">
                        {(file.size / 1024).toFixed(1)} KB &middot;{" "}
                        {new Date(file.uploadedAt).toLocaleString()}
                      </p>
                    </div>
                  </label>
                  <button
                    className="button ghost"
                    style={{ color: "var(--error)", flexShrink: 0 }}
                    onClick={() => deleteFile(file.id)}
                    title="Remove artefact"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="row">
        <button
          type="button"
          className="button primary"
          onClick={proceed}
          disabled={submitting || selected.length === 0}
        >
          {submitting ? "Creating bundle..." : "Continue to pipeline selection"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
