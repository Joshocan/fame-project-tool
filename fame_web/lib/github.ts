import { promises as fs } from "node:fs";
import path from "node:path";
import { extractArtefactCandidates } from "./artefact-extraction";
import { Artefact, RepoSummary } from "./models";

type GitHubRepo = {
  name: string;
  owner: { login: string };
  description: string | null;
  stargazers_count: number;
  topics?: string[];
  default_branch: string;
};

type BranchResponse = {
  name: string;
  commit: { sha: string };
};

type TreeEntry = {
  path: string;
  type: string;
  size?: number;
};

type TreeResponse = {
  truncated: boolean;
  tree: TreeEntry[];
};

type ContentsResponse = {
  content?: string;
  encoding?: string;
};

const API_ROOT = "https://api.github.com";
const USER_AGENT = "fame-web-local";
const MAX_PREVIEW_CHARS = 240;

function parseRepoUrl(value: string): { owner: string; name: string } | null {
  try {
    const url = new URL(value);
    const segments = url.pathname.split("/").filter(Boolean);
    if (url.hostname.includes("github.com") && segments.length >= 2) {
      return { owner: segments[0], name: segments[1].replace(/\.git$/, "") };
    }
  } catch {
    // ignore invalid URL
  }
  return null;
}

let cachedToken: string | null | undefined;

async function readGitHubToken(): Promise<string | null> {
  if (cachedToken !== undefined) {
    return cachedToken;
  }

  if (process.env.GITHUB_TOKEN) {
    cachedToken = process.env.GITHUB_TOKEN.trim();
    return cachedToken || null;
  }

  const tokenPath = path.resolve(process.cwd(), "../api_keys/github_key.txt");
  try {
    const token = (await fs.readFile(tokenPath, "utf8")).trim();
    cachedToken = token || null;
    return cachedToken;
  } catch {
    cachedToken = null;
    return null;
  }
}

async function githubFetch<T>(pathname: string): Promise<T> {
  const token = await readGitHubToken();
  const response = await fetch(`${API_ROOT}${pathname}`, {
    headers: {
      Accept: "application/vnd.github+json",
      "User-Agent": USER_AGENT,
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    cache: "no-store"
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`GitHub API ${response.status} for ${pathname}: ${body}`);
  }

  return response.json() as Promise<T>;
}

function inferDomain(repo: GitHubRepo): string {
  const haystack = [repo.name, repo.description ?? "", ...(repo.topics ?? [])].join(" ").toLowerCase();
  if (haystack.includes("fhir")) return "FHIR";
  if (haystack.includes("ehr") || haystack.includes("emr")) return "EHR/EMR";
  if (haystack.includes("lab") || haystack.includes("laboratory")) return "Laboratory";
  if (haystack.includes("api")) return "Healthcare API";
  return "Healthcare Software";
}

function summarizeContent(raw: string): string {
  return raw.replace(/\s+/g, " ").trim().slice(0, MAX_PREVIEW_CHARS) || "No text preview available.";
}

function decodeContent(content: string, encoding?: string): string {
  if (encoding === "base64") {
    return Buffer.from(content, "base64").toString("utf8");
  }
  return content;
}

export async function searchRepositories(_query?: string): Promise<RepoSummary[]> {
  return [];
}

export async function fetchRepository(owner: string, name: string): Promise<RepoSummary | undefined> {
  const repo = await githubFetch<GitHubRepo>(`/repos/${owner}/${name}`);
  const branch = await githubFetch<BranchResponse>(`/repos/${owner}/${name}/branches/${repo.default_branch}`);

  return {
    owner: repo.owner.login,
    name: repo.name,
    description: repo.description ?? "No repository description provided.",
    domain: inferDomain(repo),
    stars: repo.stargazers_count,
    topics: repo.topics ?? [],
    defaultBranch: repo.default_branch,
    commitSha: branch.commit.sha
  };
}

export async function fetchRepositoryArtefactContent(owner: string, name: string, filePath: string, ref: string) {
  const encodedPath = filePath.split("/").map(encodeURIComponent).join("/");
  const contentResponse = await githubFetch<ContentsResponse>(
    `/repos/${owner}/${name}/contents/${encodedPath}?ref=${ref}`
  );
  const raw = contentResponse.content ? decodeContent(contentResponse.content, contentResponse.encoding) : "";
  return {
    content: raw,
    preview: summarizeContent(raw)
  };
}

export async function fetchRepositoryArtefacts(owner: string, name: string): Promise<Artefact[]> {
  const repo = await fetchRepository(owner, name);
  if (!repo) return [];

  const treeResponse = await githubFetch<TreeResponse>(`/repos/${owner}/${name}/git/trees/${repo.commitSha}?recursive=1`);
  const candidates = extractArtefactCandidates(treeResponse.tree);

  return Promise.all(
    candidates.map(async (candidate) => {
      try {
        const { preview } = await fetchRepositoryArtefactContent(owner, name, candidate.path, repo.commitSha);
        return {
          ...candidate,
          preview
        };
      } catch {
        return {
          ...candidate,
          preview: "Preview unavailable from GitHub contents API."
        };
      }
    })
  );
}

export async function parseRepositoryFromUrl(url: string): Promise<RepoSummary | undefined> {
  const parsed = parseRepoUrl(url);
  if (!parsed) {
    return undefined;
  }
  return fetchRepository(parsed.owner, parsed.name);
}
