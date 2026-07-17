import path from "node:path";
import { BundleRecord } from "./models";
import { getPipelinePreset } from "./pipelines";

export type CliPlan = {
  workingDirectory: string;
  preprocessingSteps: string[];
  executionCommand: string;
};

export function buildCliPlan(bundle: BundleRecord, pipelineId: string): CliPlan {
  const preset = getPipelinePreset(pipelineId);
  const preprocessingSteps =
    preset?.retrievalMode === "rag"
      ? [
          "normalize selected artefacts",
          "chunk artefact text",
          "build ChromaDB collection"
        ]
      : ["normalize selected artefacts", "prepare direct prompt bundle"];

  return {
    workingDirectory: path.resolve(process.cwd(), ".."),
    preprocessingSteps,
    executionCommand:
      preset?.id === "rag-single"
        ? "python scripts/run_ss_rag.py --interactive"
        : preset?.id === "rag-iterative"
          ? "python scripts/run_is_rag.py --interactive"
          : preset?.id === "nonrag-iterative"
            ? "python scripts/run_is_nonrag.py --interactive"
            : "python scripts/run_ss_nonrag.py --interactive"
  };
}
