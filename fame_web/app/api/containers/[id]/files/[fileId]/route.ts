import { promises as fs } from "node:fs";
import { NextRequest, NextResponse } from "next/server";
import { getContainer, removeFileFromContainer } from "@/lib/storage";

export async function DELETE(_request: NextRequest, { params }: { params: Promise<{ id: string; fileId: string }> }) {
  const { id, fileId } = await params;
  const container = await getContainer(id);
  if (!container) return NextResponse.json({ error: "Container not found." }, { status: 404 });

  const file = container.files.find((f) => f.id === fileId);
  if (!file) return NextResponse.json({ error: "File not found." }, { status: 404 });

  try {
    await fs.unlink(file.storedPath);
  } catch {
    // file may already be gone
  }

  const updated = await removeFileFromContainer(id, fileId);
  return NextResponse.json(updated);
}
