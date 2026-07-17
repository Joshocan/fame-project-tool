import { notFound } from "next/navigation";
import { getContainer } from "@/lib/storage";
import { ContainerManager } from "@/components/ContainerManager";

export default async function ContainerPage({
  params
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const container = await getContainer(id);
  if (!container) notFound();

  return (
    <div className="stack">
      <section className="panel">
        <p className="eyebrow">Step 2 · Artefact curation — System</p>
      </section>
      <ContainerManager container={container} />
    </div>
  );
}
