"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { PipelinePreset } from "@/lib/models";

type Props = {
  bundleId: string;
  presets: PipelinePreset[];
};

export function PipelineForm({ bundleId, presets }: Props) {
  const router = useRouter();
  const [pipelineId, setPipelineId] = useState<string>(presets[0]?.id ?? "");
  const [runCount, setRunCount] = useState(3);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bundleId,
        pipelineId,
        runCount
      })
    });

    if (!response.ok) {
      setSubmitting(false);
      setError("Failed to create run.");
      return;
    }

    const payload = (await response.json()) as { id: string };
    router.push(`/runs/${payload.id}`);
  }

  return (
    <form onSubmit={onSubmit} className="stack">
      <div className="grid">
        {presets.map((preset) => (
          <label key={preset.id} className={`card selectable ${pipelineId === preset.id ? "selected" : ""}`}>
            <div className="row">
              <input
                type="radio"
                name="pipeline"
                value={preset.id}
                checked={pipelineId === preset.id}
                onChange={() => setPipelineId(preset.id)}
              />
              <div>
                <strong>{preset.name}</strong>
                <p className="muted">{preset.description}</p>
              </div>
            </div>
          </label>
        ))}
      </div>

      <label className="label" htmlFor="run-count">
        Number of repeated runs
      </label>
      <input
        id="run-count"
        className="input small"
        min={1}
        max={10}
        type="number"
        value={runCount}
        onChange={(event) => setRunCount(Number(event.target.value))}
      />

      <button type="submit" className="button primary" disabled={submitting}>
        {submitting ? "Preparing run..." : "Prepare data and create run"}
      </button>
      {error ? <p className="error">{error}</p> : null}
    </form>
  );
}
