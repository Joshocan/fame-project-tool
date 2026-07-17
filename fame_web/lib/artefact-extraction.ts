import { Artefact, ArtefactCategory } from "./models";

type TreeEntry = {
  path: string;
  type: string;
  size?: number;
};

type Candidate = Omit<Artefact, "preview">;

const EXCLUDED_SEGMENTS = [
  "node_modules",
  "dist",
  "build",
  ".git",
  "coverage",
  "vendor",
  "target",
  "__pycache__"
];

const EXCLUDED_SUFFIXES = [
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".svg",
  ".pdf",
  ".lock",
  ".class",
  ".jar",
  ".zip"
];

const STRUCTURED_EXTENSIONS = new Set([".yaml", ".yml", ".json", ".xml"]);
const TEXT_EXTENSIONS = new Set([".md", ".rst", ".yaml", ".yml", ".json", ".xml", ".txt"]);

function extensionOf(path: string): string {
  const idx = path.lastIndexOf(".");
  return idx >= 0 ? path.slice(idx).toLowerCase() : "";
}

function sourceKindForExtension(extension: string): Artefact["sourceKind"] {
  if (extension === ".md" || extension === ".rst") return "markdown";
  if (extension === ".yaml" || extension === ".yml") return "yaml";
  if (extension === ".json") return "json";
  if (extension === ".xml") return "xml";
  return "text";
}

function scoreAndClassify(path: string): {
  category: ArtefactCategory;
  score: number;
  signals: string[];
  reason: string;
} | null {
  const lower = path.toLowerCase();
  const extension = extensionOf(path);

  if (EXCLUDED_SEGMENTS.some((segment) => lower.split("/").includes(segment))) return null;
  if (EXCLUDED_SUFFIXES.some((suffix) => lower.endsWith(suffix))) return null;

  const signals: string[] = [];
  let score = 0;
  let category: ArtefactCategory = "docs";

  if (lower === "readme.md" || lower.endsWith("/readme.md")) {
    category = "readme";
    score += 0.95;
    signals.push("root-readme");
  }
  if (lower.includes("docs/")) {
    score += 0.2;
    signals.push("docs-path");
  }
  if (lower.includes("api") || lower.includes("openapi") || lower.includes("swagger") || lower.includes("rest")) {
    category = "api";
    score += 0.35;
    signals.push("api-keyword");
  }
  if (lower.includes("schema") || lower.includes("profile") || STRUCTURED_EXTENSIONS.has(extension)) {
    if (category !== "api") category = "schema";
    score += 0.28;
    signals.push("schema-signal");
  }
  if (lower.includes("workflow") || lower.includes("manual") || lower.includes("guide") || lower.includes("lab") || lower.includes("order")) {
    category = "workflow";
    score += 0.25;
    signals.push("workflow-signal");
  }
  if (lower.includes("example") || lower.includes("sample")) {
    category = "example";
    score += 0.18;
    signals.push("example-signal");
  }
  if (lower.includes("fhir") || lower.includes("hl7") || lower.includes("patient") || lower.includes("observation")) {
    score += 0.15;
    signals.push("healthcare-signal");
  }
  if (TEXT_EXTENSIONS.has(extension)) {
    score += 0.08;
    signals.push("text-like-extension");
  }

  if (score < 0.18) return null;

  const reason = `Selected from repository tree using path and filename heuristics: ${signals
    .slice(0, 4)
    .join(", ")}.`;
  return {
    category,
    score: Math.min(score, 0.99),
    signals,
    reason
  };
}

export function extractArtefactCandidates(tree: TreeEntry[]): Candidate[] {
  const results: Candidate[] = [];

  for (const entry of tree) {
    if (entry.type !== "blob") continue;
    const classified = scoreAndClassify(entry.path);
    if (!classified) continue;

    const extension = extensionOf(entry.path);
    results.push({
      path: entry.path,
      category: classified.category,
      size: entry.size ?? 0,
      score: Number(classified.score.toFixed(2)),
      extension,
      sourceKind: sourceKindForExtension(extension),
      recommended: classified.score >= 0.4,
      extractionReason: classified.reason,
      heuristicSignals: classified.signals
    });
  }

  return results.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
}
