import { NextResponse } from "next/server";
import { createUploadedSourceFromArchive } from "@/lib/sources";

export async function POST(request: Request) {
  const formData = await request.formData();
  const archive = formData.get("archive");

  if (!(archive instanceof File)) {
    return NextResponse.json({ error: "Upload a repository zip archive." }, { status: 400 });
  }

  if (!archive.name.toLowerCase().endsWith(".zip")) {
    return NextResponse.json({ error: "Only .zip repository archives are supported right now." }, { status: 400 });
  }

  const bytes = Buffer.from(await archive.arrayBuffer());
  const source = await createUploadedSourceFromArchive(archive.name, bytes);
  return NextResponse.json({ id: source.id });
}
