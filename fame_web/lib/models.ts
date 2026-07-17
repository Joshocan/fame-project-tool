export type RepoSummary = {
  owner: string;
  name: string;
  description: string;
  domain: string;
  stars: number;
  topics: string[];
  defaultBranch: string;
  commitSha: string;
};

export type UploadedSource = {
  id: string;
  kind: "upload";
  displayName: string;
  archiveFilename: string;
  archivePath: string;
  extractedPath: string;
  createdAt: string;
  commitSha: string;
};

export type ArtefactCategory =
  | "readme"
  | "docs"
  | "api"
  | "schema"
  | "example"
  | "workflow";

export type Artefact = {
  path: string;
  category: ArtefactCategory;
  size: number;
  preview: string;
  score: number;
  extension?: string;
  sourceKind?: "markdown" | "yaml" | "json" | "xml" | "text";
  recommended?: boolean;
  extractionReason?: string;
  heuristicSignals?: string[];
};

export type ContainerFile = {
  id: string;
  filename: string;
  storedPath: string;
  size: number;
  uploadedAt: string;
};

export type ContainerRecord = {
  id: string;
  name: string;
  files: ContainerFile[];
  createdAt: string;
  updatedAt: string;
};

export type BundleRecord = {
  id: string;
  repoKey: string;
  commitSha: string;
  selectedArtefactPaths: string[];
  sourceType?: "github" | "upload" | "container";
  sourceId?: string;
  createdAt: string;
};

export type PipelinePreset = {
  id: string;
  name: string;
  retrievalMode: "rag" | "non-rag";
  promptingMode: "single-stage" | "iterative";
  description: string;
};

export type ValidationSummary = {
  schemaValid: boolean;
  metamodelConformance: "pass" | "warning";
  notes: string[];
};

export type ModelOutput = {
  systemName: string;
  modelId: string;
  nodes: Array<{
    id: string;
    type: "System" | "Capability" | "DataConcept" | "Interface" | "Constraint";
    name: string;
    description: string;
  }>;
  edges: Array<{
    source: string;
    type:
      | "contains"
      | "consumes"
      | "produces"
      | "dependsOn"
      | "usesInterface"
      | "constrains";
    target: string;
    description: string;
  }>;
};

export type RunStatus =
  | "created"
  | "preprocessing"
  | "ready"
  | "running"
  | "validating"
  | "completed"
  | "failed";

export type WorkflowStepStatus = "pending" | "active" | "completed";

export type WorkflowStep = {
  key: "preprocessing" | "construction" | "validation" | "review";
  title: string;
  status: WorkflowStepStatus;
  details: string[];
};

export type RunRecord = {
  id: string;
  bundleId: string;
  pipelineId: string;
  runCount: number;
  status: RunStatus;
  createdAt: string;
  updatedAt: string;
  steps: WorkflowStep[];
  output?: ModelOutput;
  validation?: ValidationSummary;
  graphExport?: {
    neo4jReady: boolean;
    note: string;
  };
  promptId?: string;
  promptSnapshotPath?: string;
  workspaceDir?: string;
  logsPath?: string;
  outputPath?: string;
  cliCommand?: string;
  exitCode?: number;
  errorMessage?: string;
};

export type PromptTemplate = {
  id: string;
  label: string;
  path: string;
  formalism: "feature-model" | "generic-relation" | "custom";
  description?: string;
};

export type PipelineRunInput = {
  run: RunRecord;
  bundle: BundleRecord;
  containerFilesDir?: string;
  workspaceDir: string;
  promptPath: string;
  rootFeature?: string;
  domain?: string;
};

export type PipelineRunResult = {
  output?: ModelOutput;
  outputPath?: string;
  cliCommand: string;
  exitCode: number;
  workspaceDir: string;
};
