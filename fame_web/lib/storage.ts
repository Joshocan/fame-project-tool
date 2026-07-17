import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { BundleRecord, ContainerFile, ContainerRecord, RunRecord, UploadedSource } from "./models";

const storageDir = path.join(process.cwd(), "storage");
const uploadsDir = path.join(storageDir, "uploads");
const containersDir = path.join(storageDir, "containers");
const bundlesFile = path.join(storageDir, "bundles.json");
const runsFile = path.join(storageDir, "runs.json");
const sourcesFile = path.join(storageDir, "sources.json");
const containersFile = path.join(storageDir, "containers.json");

async function ensureFile(filePath: string) {
  await fs.mkdir(storageDir, { recursive: true });
  try {
    await fs.access(filePath);
  } catch {
    await fs.writeFile(filePath, "[]\n", "utf8");
  }
}

const fileLocks = new Map<string, Promise<unknown>>();

function withFileLock<T>(filePath: string, task: () => Promise<T>): Promise<T> {
  const prev = fileLocks.get(filePath) ?? Promise.resolve();
  const next = prev.then(task, task);
  fileLocks.set(
    filePath,
    next.catch(() => undefined).finally(() => {
      if (fileLocks.get(filePath) === next) fileLocks.delete(filePath);
    })
  );
  return next;
}

async function readCollection<T>(filePath: string): Promise<T[]> {
  await ensureFile(filePath);
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw) as T[];
}

async function writeCollection<T>(filePath: string, rows: T[]) {
  await ensureFile(filePath);
  await fs.writeFile(filePath, `${JSON.stringify(rows, null, 2)}\n`, "utf8");
}

async function mutateCollection<T, R>(
  filePath: string,
  mutator: (rows: T[]) => Promise<{ rows: T[]; result: R }> | { rows: T[]; result: R }
): Promise<R> {
  return withFileLock(filePath, async () => {
    const rows = await readCollection<T>(filePath);
    const { rows: nextRows, result } = await mutator(rows);
    await writeCollection(filePath, nextRows);
    return result;
  });
}

export function getUploadsRoot() {
  return uploadsDir;
}

export function getContainerFilesDir(containerId: string) {
  return path.join(containersDir, containerId);
}

export function createStorageId() {
  return randomUUID();
}

export async function listContainers(): Promise<ContainerRecord[]> {
  return readCollection<ContainerRecord>(containersFile);
}

export async function getContainer(id: string): Promise<ContainerRecord | undefined> {
  const containers = await listContainers();
  return containers.find((c) => c.id === id);
}

export async function createContainer(name?: string): Promise<ContainerRecord> {
  return mutateCollection<ContainerRecord, ContainerRecord>(containersFile, (containers) => {
    const systemName = name ?? `System_${containers.length + 1}`;
    const now = new Date().toISOString();
    const container: ContainerRecord = {
      id: createStorageId(),
      name: systemName,
      files: [],
      createdAt: now,
      updatedAt: now
    };
    return { rows: [...containers, container], result: container };
  });
}

export async function updateContainer(id: string, patch: Partial<Pick<ContainerRecord, "name">>): Promise<ContainerRecord | undefined> {
  return mutateCollection<ContainerRecord, ContainerRecord | undefined>(containersFile, (containers) => {
    const idx = containers.findIndex((c) => c.id === id);
    if (idx === -1) return { rows: containers, result: undefined };
    const next = [...containers];
    next[idx] = { ...next[idx], ...patch, updatedAt: new Date().toISOString() };
    return { rows: next, result: next[idx] };
  });
}

export async function addFileToContainer(containerId: string, file: ContainerFile): Promise<ContainerRecord | undefined> {
  return mutateCollection<ContainerRecord, ContainerRecord | undefined>(containersFile, (containers) => {
    const idx = containers.findIndex((c) => c.id === containerId);
    if (idx === -1) return { rows: containers, result: undefined };
    const next = [...containers];
    next[idx] = {
      ...next[idx],
      files: [...next[idx].files, file],
      updatedAt: new Date().toISOString()
    };
    return { rows: next, result: next[idx] };
  });
}

export async function removeFileFromContainer(containerId: string, fileId: string): Promise<ContainerRecord | undefined> {
  return mutateCollection<ContainerRecord, ContainerRecord | undefined>(containersFile, (containers) => {
    const idx = containers.findIndex((c) => c.id === containerId);
    if (idx === -1) return { rows: containers, result: undefined };
    const next = [...containers];
    next[idx] = {
      ...next[idx],
      files: next[idx].files.filter((f) => f.id !== fileId),
      updatedAt: new Date().toISOString()
    };
    return { rows: next, result: next[idx] };
  });
}

export async function listSources(): Promise<UploadedSource[]> {
  return readCollection<UploadedSource>(sourcesFile);
}

export async function getSource(id: string): Promise<UploadedSource | undefined> {
  const sources = await listSources();
  return sources.find((source) => source.id === id);
}

export async function createSource(input: Omit<UploadedSource, "id" | "createdAt">) {
  return mutateCollection<UploadedSource, UploadedSource>(sourcesFile, (sources) => {
    const source: UploadedSource = {
      id: createStorageId(),
      createdAt: new Date().toISOString(),
      ...input
    };
    return { rows: [...sources, source], result: source };
  });
}

export async function listBundles(): Promise<BundleRecord[]> {
  return readCollection<BundleRecord>(bundlesFile);
}

export async function getBundle(id: string): Promise<BundleRecord | undefined> {
  const bundles = await listBundles();
  return bundles.find((bundle) => bundle.id === id);
}

export async function createBundle(input: Omit<BundleRecord, "id" | "createdAt">) {
  return mutateCollection<BundleRecord, BundleRecord>(bundlesFile, (bundles) => {
    const bundle: BundleRecord = {
      id: createStorageId(),
      createdAt: new Date().toISOString(),
      ...input
    };
    return { rows: [...bundles, bundle], result: bundle };
  });
}

export async function listRuns(): Promise<RunRecord[]> {
  return readCollection<RunRecord>(runsFile);
}

export async function getRun(id: string): Promise<RunRecord | undefined> {
  const runs = await listRuns();
  return runs.find((run) => run.id === id);
}

export async function createRun(input: Omit<RunRecord, "id" | "createdAt" | "updatedAt">) {
  return mutateCollection<RunRecord, RunRecord>(runsFile, (runs) => {
    const now = new Date().toISOString();
    const run: RunRecord = {
      id: createStorageId(),
      createdAt: now,
      updatedAt: now,
      ...input
    };
    return { rows: [...runs, run], result: run };
  });
}

export async function updateRun(id: string, patch: Partial<RunRecord>) {
  return mutateCollection<RunRecord, RunRecord | undefined>(runsFile, (runs) => {
    const index = runs.findIndex((run) => run.id === id);
    if (index === -1) return { rows: runs, result: undefined };
    const next = [...runs];
    next[index] = { ...next[index], ...patch, updatedAt: new Date().toISOString() };
    return { rows: next, result: next[index] };
  });
}
