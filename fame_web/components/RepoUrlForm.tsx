"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

function parseGithubUrl(value: string) {
  try {
    const url = new URL(value);
    const segments = url.pathname.split("/").filter(Boolean);
    if (url.hostname.includes("github.com") && segments.length >= 2) {
      return { owner: segments[0], repo: segments[1].replace(/\.git$/, "") };
    }
  } catch {
    // no-op
  }
  return null;
}

export function RepoUrlForm() {
  const router = useRouter();
  const [value, setValue] = useState("");
  const [error, setError] = useState("");

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const parsed = parseGithubUrl(value);
    if (!parsed) {
      setError("Enter a valid GitHub repository URL.");
      return;
    }
    setError("");
    router.push(`/repos/${parsed.owner}/${parsed.repo}`);
  }

  return (
    <form onSubmit={onSubmit} className="stack">
      <label className="label" htmlFor="repo-url">
        Paste GitHub repository URL
      </label>
      <div className="row">
        <input
          id="repo-url"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="https://github.com/owner/repo"
          className="input"
        />
        <button type="submit" className="button primary">
          Inspect repo
        </button>
      </div>
      {error ? <p className="error">{error}</p> : null}
    </form>
  );
}
