"use client";

import type { Route } from "next";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

export function RepositoryUploadForm() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose a repository zip archive.");
      return;
    }

    setSubmitting(true);
    setError("");

    const formData = new FormData();
    formData.append("archive", file);

    const response = await fetch("/api/sources/upload", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({ error: "Upload failed." }));
      setSubmitting(false);
      setError(payload.error ?? "Upload failed.");
      return;
    }

    const payload = (await response.json()) as { id: string };
    router.push(`/sources/${payload.id}` as Route);
  }

  return (
    <form onSubmit={onSubmit} className="stack">
      <label className="label" htmlFor="repo-archive">
        Upload repository archive (.zip)
      </label>
      <input
        id="repo-archive"
        type="file"
        accept=".zip,application/zip"
        className="input"
        onChange={(event) => setFile(event.target.files?.[0] ?? null)}
      />
      <button type="submit" className="button primary" disabled={submitting}>
        {submitting ? "Uploading..." : "Upload and extract"}
      </button>
      {error ? <p className="error">{error}</p> : null}
    </form>
  );
}
