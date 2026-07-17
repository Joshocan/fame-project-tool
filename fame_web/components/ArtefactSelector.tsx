"use client";

import { useMemo, useState, useEffect } from "react";
import { Artefact } from "@/lib/models";

type Props = {
  repoKey: string;
  commitSha: string;
  artefacts: Artefact[];
  sourceType?: "github" | "upload";
  sourceId?: string;
};

type FullContentPayload = {
  path: string;
  content: string;
};

export function ArtefactSelector({ repoKey, commitSha, artefacts, sourceType = "github", sourceId }: Props) {
  const [selected, setSelected] = useState<string[]>(
    artefacts.filter((item) => item.recommended).map((item) => item.path)
  );
  const [activePath, setActivePath] = useState<string>(artefacts[0]?.path ?? "");
  const [hoveredPath, setHoveredPath] = useState<string | null>(null);
  const [contentState, setContentState] = useState<{ loading: boolean; error: string; payload?: FullContentPayload }>({
    loading: false,
    error: ""
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const selectedCount = useMemo(() => selected.length, [selected]);
  const totalSize = useMemo(
    () => artefacts.filter((item) => selected.includes(item.path)).reduce((sum, item) => sum + item.size, 0),
    [artefacts, selected]
  );
  const viewerPath = hoveredPath ?? activePath;
  const activeArtefact = useMemo(
    () => artefacts.find((item) => item.path === viewerPath) ?? artefacts[0],
    [artefacts, viewerPath]
  );
  const allPaths = useMemo(() => artefacts.map((item) => item.path), [artefacts]);
  const allSelected = artefacts.length > 0 && selected.length === artefacts.length;

  useEffect(() => {
    setSelected((current) => {
      const currentSet = new Set(current);
      const filtered = artefacts.map((item) => item.path).filter((path) => currentSet.has(path));
      return filtered.length ? filtered : artefacts.filter((item) => item.recommended).map((item) => item.path);
    });
    setActivePath((current) => {
      if (artefacts.length && !artefacts.some((item) => item.path === current)) {
        return artefacts[0].path;
      }
      return current;
    });
  }, [artefacts]);

  function toggle(path: string) {
    setSelected((current) =>
      current.includes(path) ? current.filter((item) => item !== path) : [...current, path]
    );
  }

  function toggleAll(checked: boolean) {
    setSelected(checked ? allPaths : []);
  }

  useEffect(() => {
    if (!activeArtefact) return;
    const params = new URLSearchParams({
      sourceType,
      path: activeArtefact.path,
      commitSha,
      repoKey
    });
    if (sourceId) params.set("sourceId", sourceId);

    let cancelled = false;
    setContentState({ loading: true, error: "" });

    fetch(`/api/artefacts/content?${params.toString()}`)
      .then(async (response) => {
        if (!response.ok) {
          const payload = await response.json().catch(() => ({ error: "Failed to load artefact content." }));
          throw new Error(payload.error ?? "Failed to load artefact content.");
        }
        return response.json() as Promise<FullContentPayload>;
      })
      .then((payload) => {
        if (!cancelled) setContentState({ loading: false, error: "", payload });
      })
      .catch(() => {
        if (!cancelled) {
          setContentState({
            loading: false,
            error: "",
            payload: {
              path: activeArtefact.path,
              content: activeArtefact.preview
            }
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeArtefact, commitSha, repoKey, sourceId, sourceType]);

  async function createBundle() {
    if (selected.length === 0) {
      setError("Select at least one artefact.");
      return;
    }

    setSubmitting(true);
    setError("");

    const response = await fetch("/api/bundles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repoKey,
        commitSha,
        selectedArtefactPaths: selected,
        sourceType,
        sourceId
      })
    });

    if (!response.ok) {
      setSubmitting(false);
      setError("Failed to create artefact bundle.");
      return;
    }

    const payload = (await response.json()) as { id: string };
    window.location.href = `/runs/new?bundleId=${payload.id}`;
  }

  return (
    <div className="stack">
      <div className="panel">
        <div className="row spaceBetween wrap">
          <div>
            <h3>Review candidate artefacts</h3>
            <p className="muted">
              All candidate artefacts are shown in ranked order. Hover or focus an artefact to review its content, then explicitly select the files to include in the bundle.
            </p>
          </div>
          <div className="stack compactStats">
            <div className="badge">{artefacts.length} candidates</div>
            <div className="badge">{selectedCount} selected</div>
            <div className="badge">{totalSize} bytes</div>
          </div>
        </div>
        <div className="row wrap" style={{ marginTop: "0.9rem" }}>
          <label className="row" style={{ gap: "0.45rem" }}>
            <input
              type="checkbox"
              checked={allSelected}
              onChange={(event) => toggleAll(event.currentTarget.checked)}
            />
            <span>Select / deselect all</span>
          </label>
        </div>
      </div>

      <div className="artefactReviewLayout">
        <div className="artefactList panel">
          <div className="artefactListHeader">
            <strong>Ranked artefact list</strong>
          </div>
          <div className="artefactListBody">
            {artefacts.map((artefact, index) => {
              const checked = selected.includes(artefact.path);
              const active = activePath === artefact.path;
              return (
                <button
                  key={artefact.path}
                  type="button"
                  className={`artefactRow ${active ? "active" : ""}`}
                  onMouseEnter={() => setHoveredPath(artefact.path)}
                  onMouseLeave={() => setHoveredPath(null)}
                  onFocus={() => setHoveredPath(artefact.path)}
                  onBlur={() => setHoveredPath(null)}
                  onClick={() => setActivePath(artefact.path)}
                >
                  <div className="artefactRowTop">
                    <span className="artefactRank">#{index + 1}</span>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(event) => {
                        event.stopPropagation();
                        toggle(artefact.path);
                      }}
                      onClick={(event) => event.stopPropagation()}
                    />
                  </div>
                  <strong className="artefactPath">{artefact.path}</strong>
                  <p className="muted artefactMeta">
                    {artefact.category} · {artefact.extension} · {artefact.sourceKind} · score {artefact.score.toFixed(2)}
                  </p>
                  <div className="tokenRow">
                    {artefact.recommended ? <span className="pill good">recommended</span> : <span className="pill neutral">optional</span>}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="artefactViewer panel">
          {activeArtefact ? (
            <div className="stack viewerWrap">
              <div className="stack viewerHeader">
                <div className="row spaceBetween wrap">
                  <strong>{activeArtefact.path}</strong>
                  <span className="muted">{activeArtefact.size} bytes</span>
                </div>
                <p className="muted">
                  {activeArtefact.category} · {activeArtefact.extension} · {activeArtefact.sourceKind} · score {activeArtefact.score.toFixed(2)}
                </p>
                <div className="detailBlock">
                  <strong>Why extracted</strong>
                  <p className="muted">{activeArtefact.extractionReason}</p>
                </div>
                {activeArtefact.heuristicSignals?.length ? (
                  <div className="detailBlock">
                    <strong>Heuristic signals</strong>
                    <div className="tokenRow">
                      {activeArtefact.heuristicSignals.map((signal) => (
                        <span key={signal} className="token">{signal}</span>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="viewerContent">
                {contentState.loading ? (
                  <p className="muted">Loading full artefact content...</p>
                ) : (
                  <pre className="codeBlock fullContentBlock">{contentState.payload?.content ?? activeArtefact.preview}</pre>
                )}
              </div>
            </div>
          ) : (
            <p className="muted">No artefact selected.</p>
          )}
        </div>
      </div>

      <div className="row wrap">
        <button type="button" className="button primary" onClick={createBundle} disabled={submitting}>
          {submitting ? "Creating bundle..." : "Continue to pipeline selection"}
        </button>
      </div>
      {error ? <p className="error">{error}</p> : null}
    </div>
  );
}
