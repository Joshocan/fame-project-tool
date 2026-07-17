import { promises as fs } from "node:fs";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { ContainerFile } from "@/lib/models";
import { addFileToContainer, createStorageId, getContainer, getContainerFilesDir } from "@/lib/storage";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const container = await getContainer(id);
  if (!container) return NextResponse.json({ error: "Container not found." }, { status: 404 });

  const formData = await request.formData();
  const files = formData.getAll("files");

  if (!files.length) {
    return NextResponse.json({ error: "No files provided." }, { status: 400 });
  }

  const uploadDir = getContainerFilesDir(id);
  await fs.mkdir(uploadDir, { recursive: true });

  let updated = container;
  for (const entry of files) {
    if (!(entry instanceof File)) continue;
    const fileId = createStorageId();
    const safeName = entry.name.replace(/[^a-zA-Z0-9._-]/g, "_");
    const storedPath = path.join(uploadDir, `${fileId}-${safeName}`);
    const bytes = Buffer.from(await entry.arrayBuffer());
    await fs.writeFile(storedPath, bytes);

    const containerFile: ContainerFile = {
      id: fileId,
      filename: entry.name,
      storedPath,
      size: entry.size,
      uploadedAt: new Date().toISOString()
    };

    updated = (await addFileToContainer(id, containerFile)) ?? updated;
  }

  return NextResponse.json(updated);
}
