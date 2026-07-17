import Link from "next/link";

const workflowSteps = [
  "Create a system — assigned a default name (e.g. System_1) that you can rename any time.",
  "Upload your artefact files — PDFs, text, markdown, or other documents.",
  "Select the artefacts to include and choose a pipeline preset.",
  "Run the construction flow and inspect the model output.",
  "Add or remove artefacts from the system and re-run as needed."
];

export default function HomePage() {
  return (
    <div className="stack">
      <div className="panel stack">
        <p className="eyebrow">Artefact-driven model construction</p>
        <h2>FAME Web</h2>
        <p className="muted">
          Upload artefacts into a named container, select a pipeline preset,
          and run the FAME construction flow — end to end in the browser.
        </p>
        <div className="row" style={{ marginTop: "0.25rem" }}>
          <Link href="/containers" className="button primary">Browse systems</Link>
          <Link href="/runs/new" className="button">Configure a run</Link>
        </div>
      </div>

      <div className="panel stack">
        <h3>Experiment workflow</h3>
        <ol className="stepList">
          {workflowSteps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </div>

      <div className="panel stack">
        <h3>Pipeline presets</h3>
        <ul className="list">
          <li>Non-RAG / Single-stage — bundle artefacts directly into one Claude extraction pass</li>
          <li>Non-RAG / Iterative — direct prompting with iterative refinement</li>
          <li>RAG / Single-stage — chunk, index, then retrieve for one extraction pass</li>
          <li>RAG / Iterative — chunk, retrieve, and refine over multiple prompts</li>
        </ul>
      </div>
    </div>
  );
}
