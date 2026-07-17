import { NextRequest, NextResponse } from "next/server";
import { getContainer, updateContainer } from "@/lib/storage";

export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const container = await getContainer(id);
  if (!container) return NextResponse.json({ error: "Container not found." }, { status: 404 });
  return NextResponse.json(container);
}

export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const body = (await request.json()) as { name?: string };
  if (!body.name?.trim()) {
    return NextResponse.json({ error: "name is required." }, { status: 400 });
  }
  const updated = await updateContainer(id, { name: body.name.trim() });
  if (!updated) return NextResponse.json({ error: "Container not found." }, { status: 404 });
  return NextResponse.json(updated);
}
