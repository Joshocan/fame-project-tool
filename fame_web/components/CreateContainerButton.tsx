"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { Route } from "next";
import { ContainerRecord } from "@/lib/models";

export function CreateContainerButton() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  async function create() {
    setLoading(true);
    const res = await fetch("/api/containers", { method: "POST" });
    if (res.ok) {
      const container = (await res.json()) as ContainerRecord;
      router.push(`/containers/${container.id}` as Route);
    } else {
      setLoading(false);
    }
  }

  return (
    <button className="button primary" onClick={create} disabled={loading}>
      {loading ? "Creating..." : "New system"}
    </button>
  );
}
