import { promises as fs } from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { extractArtefactCandidates } from "./artefact-extraction";
import { Artefact, UploadedSource } from "./models";
import { createSource, getSource, getUploadsRoot, listSources } from "./storage";

const execFileAsync = promisify(execFile);
const PREVIEW_LIMIT = 280;

async function walk(root: string, current = ""): Promise<Array<{ path: string; type: string; size?: number; absolutePath: string }>> {
  const dir = path.join(root, current);
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const results: Array<{ path: string; type: string; size?: number; absolutePath: string }> = [];

  for (const entry of entries) {
    const rel = current ? path.posix.join(current, entry.name) : entry.name;
    const absolutePath = path.join(root, rel);
    if (entry.isDirectory()) {
      results.push(...(await walk(root, rel)));
      continue;
    }
    if (entry.isFile()) {
      const stat = await fs.stat(absolutePath);
      results.push({ path: rel, type: "blob", size: stat.size, absolutePath });
    }
  }

  return results;
}

function sanitizeBaseName(name: string): string {
  return name.replace(/\.zip$/i, "").replace(/[^a-zA-Z0-9._-]+/g, "-");
}

function summarize(text: string): string {
  return text.replace(/\s+/g, " ").trim().slice(0, PREVIEW_LIMIT) || "No text preview available.";
}

async function unzipArchive(archivePath: string, extractedPath: string) {
  await fs.mkdir(extractedPath, { recursive: true });
  await execFileAsync("/usr/bin/unzip", ["-oq", archivePath, "-d", extractedPath]);
}

export async function createUploadedSourceFromArchive(fileName: string, bytes: Buffer) {
  const uploadsRoot = getUploadsRoot();
  await fs.mkdir(uploadsRoot, { recursive: true });

  const tempId = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const baseName = sanitizeBaseName(fileName || "uploaded-repository");
  const sourceDir = path.join(uploadsRoot, tempId);
  const archivePath = path.join(sourceDir, `${baseName}.zip`);
  const extractedPath = path.join(sourceDir, "extracted");

  await fs.mkdir(sourceDir, { recursive: true });
  await fs.writeFile(archivePath, bytes);
  await unzipArchive(archivePath, extractedPath);

  const source = await createSource({
    kind: "upload",
    displayName: baseName,
    archiveFilename: fileName,
    archivePath,
    extractedPath,
    commitSha: `uploaded-${tempId}`
  });

  return source;
}

export async function listUploadedSources() {
  return listSources();
}

export async function fetchUploadedSource(sourceId: string) {
  return getSource(sourceId);
}

export async function fetchUploadedSourceArtefactContent(sourceId: string, filePath: string) {
  const source = await getSource(sourceId);
  if (!source) {
    throw new Error("Uploaded source not found.");
  }
  const absolutePath = path.join(source.extractedPath, filePath);
  const content = await fs.readFile(absolutePath, "utf8");
  return {
    content,
    preview: summarize(content)
  };
}

export async function extractUploadedSourceArtefacts(source: UploadedSource): Promise<Artefact[]> {
  const tree = await walk(source.extractedPath);
  const candidates = extractArtefactCandidates(tree);
  const withPreview = await Promise.all(
    candidates.map(async (candidate) => {
      const absolutePath = tree.find((entry) => entry.path === candidate.path)?.absolutePath;
      let preview = "Preview unavailable from uploaded archive.";
      if (absolutePath) {
        try {
          const text = await fs.readFile(absolutePath, "utf8");
          preview = summarize(text);
        } catch {
          preview = "Binary or non-text file preview unavailable.";
        }
      }
      return {
        ...candidate,
        preview
      };
    })
  );

  return withPreview;
}
