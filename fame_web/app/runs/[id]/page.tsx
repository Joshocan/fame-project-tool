import { notFound } from "next/navigation";
import { RunTabs } from "@/components/RunTabs";
import { buildCliPlan } from "@/lib/cli";
import { getBundle, getRun } from "@/lib/storage";

export default async function RunDetailPage({
  params
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const run = await getRun(id);
  if (!run) {
    notFound();
  }

  const bundle = await getBundle(run.bundleId);
  const cliPlan = bundle ? buildCliPlan(bundle, run.pipelineId) : undefined;

  return (
    <div className="stack">
      <section className="panel stack">
        <p className="eyebrow">Run</p>
        <h2>{run.id}</h2>
        <p className="muted">
          Bundle: {bundle?.repoKey ?? run.bundleId} · Status: {run.status}
        </p>
        <p className="muted">
          This page is wired for the lightweight demo. Replace the mock execution with the
          existing CLI orchestration next.
        </p>
        {cliPlan ? (
          <div className="panel">
            <h3>CLI handoff placeholder</h3>
            <p className="muted">Working directory: {cliPlan.workingDirectory}</p>
            <p className="muted">Execution command: {cliPlan.executionCommand}</p>
            <ul className="list">
              {cliPlan.preprocessingSteps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <RunTabs run={run} />
    </div>
  );
}
