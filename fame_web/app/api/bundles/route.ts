import { NextRequest, NextResponse } from "next/server";
import { createBundle } from "@/lib/storage";

export async function POST(request: NextRequest) {
  const body = (await request.json()) as {
    repoKey?: string;
    commitSha?: string;
    selectedArtefactPaths?: string[];
    sourceType?: "github" | "upload";
    sourceId?: string;
  };

  if (!body.repoKey || !body.commitSha || !body.selectedArtefactPaths?.length) {
    return NextResponse.json(
      { error: "repoKey, commitSha, and selectedArtefactPaths are required." },
      { status: 400 }
    );
  }

  const bundle = await createBundle({
    repoKey: body.repoKey,
    commitSha: body.commitSha,
    selectedArtefactPaths: body.selectedArtefactPaths,
    sourceType: body.sourceType,
    sourceId: body.sourceId
  });

  return NextResponse.json({ id: bundle.id });
}
