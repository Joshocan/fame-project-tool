import { NextResponse } from "next/server";
import { createContainer, listContainers } from "@/lib/storage";

export async function GET() {
  const containers = await listContainers();
  return NextResponse.json(containers);
}

export async function POST() {
  const container = await createContainer();
  return NextResponse.json(container, { status: 201 });
}
