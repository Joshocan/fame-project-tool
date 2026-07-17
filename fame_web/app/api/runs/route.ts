import { NextRequest, NextResponse } from "next/server";
import { buildInitialWorkflowSteps } from "@/lib/execution";
import { getPipelinePreset } from "@/lib/pipelines";
import { createRun, getBundle } from "@/lib/storage";

export async function POST(request: NextRequest) {
  const body = (await request.json()) as {
    bundleId?: string;
    pipelineId?: string;
    runCount?: number;
  };

  if (!body.bundleId || !body.pipelineId) {
    return NextResponse.json(
      { error: "bundleId and pipelineId are required." },
      { status: 400 }
    );
  }

  const bundle = await getBundle(body.bundleId);
  if (!bundle) {
    return NextResponse.json({ error: "Bundle not found." }, { status: 404 });
  }

  if (!getPipelinePreset(body.pipelineId)) {
    return NextResponse.json({ error: "Pipeline preset not found." }, { status: 404 });
  }

  const run = await createRun({
    bundleId: bundle.id,
    pipelineId: body.pipelineId,
    runCount: body.runCount ?? 3,
    status: "preprocessing",
    steps: buildInitialWorkflowSteps(bundle, body.pipelineId)
  });

  return NextResponse.json({ id: run.id });
}
