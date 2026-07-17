"use client";

import { useState } from "react";
import { RunRecord } from "@/lib/models";

const tabs = ["summary", "raw", "validation", "graph"] as const;
type Tab = (typeof tabs)[number];

export function RunTabs({ run }: { run: RunRecord }) {
  const [active, setActive] = useState<Tab>("summary");

  return (
    <div className="stack">
      <div className="row tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            className={`tab ${active === tab ? "active" : ""}`}
            onClick={() => setActive(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {active === "summary" ? (
        <div className="panel">
          <h3>Run summary</h3>
          <ul className="list">
            <li>Status: {run.status}</li>
            <li>Pipeline preset: {run.pipelineId}</li>
            <li>Repeated runs requested: {run.runCount}</li>
            <li>Created: {new Date(run.createdAt).toLocaleString()}</li>
          </ul>
        </div>
      ) : null}

      {active === "raw" ? (
        <div className="panel">
          <h3>Raw JSON model</h3>
          <pre className="codeBlock">
            {JSON.stringify(run.output ?? { message: "No output yet." }, null, 2)}
          </pre>
        </div>
      ) : null}

      {active === "validation" ? (
        <div className="panel">
          <h3>Validation</h3>
          <pre className="codeBlock">
            {JSON.stringify(run.validation ?? { message: "No validation result yet." }, null, 2)}
          </pre>
        </div>
      ) : null}

      {active === "graph" ? (
        <div className="panel">
          <h3>Graph / Neo4j export</h3>
          <p className="muted">
            {run.graphExport?.note ?? "Neo4j export has not been prepared yet."}
          </p>
          <div className="graphPlaceholder">
            {run.output?.nodes.map((node) => (
              <div key={node.id} className="graphNode">
                <strong>{node.name}</strong>
                <span>{node.type}</span>
              </div>
            )) ?? "No graph projection yet."}
          </div>
        </div>
      ) : null}
    </div>
  );
}
