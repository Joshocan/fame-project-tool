import { NextResponse } from "next/server";
import { fetchRepositoryArtefactContent } from "@/lib/github";
import { fetchUploadedSourceArtefactContent } from "@/lib/sources";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const sourceType = searchParams.get("sourceType") ?? "github";
  const filePath = searchParams.get("path");

  if (!filePath) {
    return NextResponse.json({ error: "path is required." }, { status: 400 });
  }

  try {
    if (sourceType === "upload") {
      const sourceId = searchParams.get("sourceId");
      if (!sourceId) {
        return NextResponse.json({ error: "sourceId is required for uploaded sources." }, { status: 400 });
      }
      const payload = await fetchUploadedSourceArtefactContent(sourceId, filePath);
      return NextResponse.json({ path: filePath, content: payload.content });
    }

    const repoKey = searchParams.get("repoKey");
    const commitSha = searchParams.get("commitSha");
    if (!repoKey || !commitSha) {
      return NextResponse.json({ error: "repoKey and commitSha are required for GitHub sources." }, { status: 400 });
    }

    const [owner, name] = repoKey.split("/");
    const payload = await fetchRepositoryArtefactContent(owner, name, filePath, commitSha);
    return NextResponse.json({ path: filePath, content: payload.content });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load artefact content.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
