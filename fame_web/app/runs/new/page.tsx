import Link from "next/link";
import { PipelineForm } from "@/components/PipelineForm";
import { pipelinePresets } from "@/lib/pipelines";
import { getBundle, getContainer } from "@/lib/storage";

export default async function NewRunPage({
  searchParams
}: {
  searchParams?: Promise<{ bundleId?: string }>;
}) {
  const params = (await searchParams) ?? {};
  const bundle = params.bundleId ? await getBundle(params.bundleId) : undefined;

  if (!bundle) {
    return (
      <div className="panel stack">
        <h2>No artefact bundle selected</h2>
        <p className="muted">
          Open a system, select your artefacts, then click &ldquo;Continue to
          pipeline selection&rdquo; to arrive here.
        </p>
        <Link href="/containers" className="button primary">
          Go to systems
        </Link>
      </div>
    );
  }

  let description = "";
  if (bundle.sourceType === "container" && bundle.sourceId) {
    const container = await getContainer(bundle.sourceId);
    description = container ? `System: ${container.name}` : "Artefact system";
  }

  return (
    <div className="stack">
      <section className="panel stack">
        <p className="eyebrow">Step 3 · Choose pipeline</p>
        <h2>Configure pipeline</h2>
        <p>
          Source: <strong>{bundle.repoKey}</strong>
        </p>
        <p className="muted">
          Artefacts selected: {bundle.selectedArtefactPaths.length}
        </p>
        {description ? <p className="muted">{description}</p> : null}
      </section>

      <section className="panel stack">
        <h3>Next workflow stages</h3>
        <ol className="list">
          <li>Step 4 · Preprocessing: normalize, parse, and optionally chunk/index the selected artefacts.</li>
          <li>Step 5 · Model construction: run Claude over the prepared bundle.</li>
          <li>Step 6 · Validation: structural, semantic, and logical checks.</li>
          <li>Step 7 · Review outputs: raw JSON, validation report, graph rendering, and run comparison.</li>
        </ol>
      </section>

      <PipelineForm bundleId={bundle.id} presets={pipelinePresets} />
    </div>
  );
}
