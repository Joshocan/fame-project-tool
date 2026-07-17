import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import path from "node:path";
import { ModelOutput, PipelineRunInput, PipelineRunResult, PromptTemplate } from "./models";
import { getPipelinePreset } from "./pipelines";

const REPO_ROOT = process.env.FAME_REPO_ROOT
  ? path.resolve(process.env.FAME_REPO_ROOT)
  : path.resolve(process.cwd(), "..");

const PYTHON = process.env.FAME_PYTHON ?? "python";

const scriptForPreset: Record<string, string> = {
  "nonrag-single": "scripts/run_ss_nonrag.py",
  "nonrag-iterative": "scripts/run_is_nonrag.py",
  "rag-single": "scripts/run_ss_rag.py",
  "rag-iterative": "scripts/run_is_rag.py"
};

const defaultPromptForPreset: Record<string, string> = {
  "nonrag-single": "prompts/fm_extraction_prompt.txt",
  "nonrag-iterative": "prompts/fm_iterated_prompt.txt",
  "rag-single": "prompts/fm_extraction_prompt.txt",
  "rag-iterative": "prompts/fm_iterated_prompt.txt"
};

export const promptCatalog: PromptTemplate[] = [
  {
    id: "fm-extraction",
    label: "Feature Model — extraction",
    path: "prompts/fm_extraction_prompt.txt",
    formalism: "feature-model",
    description: "Single-pass feature model extraction prompt."
  },
  {
    id: "fm-iterated",
    label: "Feature Model — iterative",
    path: "prompts/fm_iterated_prompt.txt",
    formalism: "feature-model",
    description: "Iterative refinement of the feature model."
  }
];

export function getPromptById(id: string): PromptTemplate | undefined {
  return promptCatalog.find((p) => p.id === id);
}

export function defaultPromptForPipeline(pipelineId: string): string {
  return defaultPromptForPreset[pipelineId] ?? "prompts/fm_extraction_prompt.txt";
}

export class PipelineError extends Error {
  constructor(
    message: string,
    readonly context: { workspaceDir: string; exitCode?: number }
  ) {
    super(message);
    this.name = "PipelineError";
  }
}

export interface PipelineRunner {
  run(input: PipelineRunInput): Promise<PipelineRunResult>;
}

class SubprocessRunner implements PipelineRunner {
  async run(input: PipelineRunInput): Promise<PipelineRunResult> {
    const preset = getPipelinePreset(input.run.pipelineId);
    if (!preset) {
      throw new PipelineError(`Unknown pipeline preset: ${input.run.pipelineId}`, {
        workspaceDir: input.workspaceDir
      });
    }
    const script = scriptForPreset[preset.id];
    if (!script) {
      throw new PipelineError(`No script mapped for preset: ${preset.id}`, {
        workspaceDir: input.workspaceDir
      });
    }

    await fs.mkdir(input.workspaceDir, { recursive: true });
    const chunksDir = path.join(input.workspaceDir, "chunks");
    const outputDir = path.join(input.workspaceDir, "output");
    await fs.mkdir(chunksDir, { recursive: true });
    await fs.mkdir(outputDir, { recursive: true });

    const promptSnapshot = path.join(input.workspaceDir, "prompt.txt");
    const promptSource = path.isAbsolute(input.promptPath)
      ? input.promptPath
      : path.resolve(REPO_ROOT, input.promptPath);
    await fs.copyFile(promptSource, promptSnapshot);

    const args: string[] = [
      script,
      "--prompt-path", promptSnapshot,
      "--chunks-dir", chunksDir,
      "--root-feature", input.rootFeature ?? "System",
      "--domain", input.domain ?? "generic",
      "--run-tag", input.run.id
    ];

    const logsPath = path.join(input.workspaceDir, "logs.jsonl");
    const logStream = await fs.open(logsPath, "a");

    const env: NodeJS.ProcessEnv = {
      ...process.env,
      FAME_OUTPUT_DIR: outputDir,
      FAME_RUN_ID: input.run.id
    };
    if (input.containerFilesDir) {
      env.FAME_INPUT_FILES_DIR = input.containerFilesDir;
    }

    const cliCommand = [PYTHON, ...args].join(" ");
    const started = Date.now();
    await logStream.write(
      JSON.stringify({ t: started, s: "system", line: `spawn: ${cliCommand}` }) + "\n"
    );

    const proc = spawn(PYTHON, args, { cwd: REPO_ROOT, env });

    const append = (stream: "stdout" | "stderr") => (chunk: Buffer) => {
      logStream.write(
        JSON.stringify({ t: Date.now(), s: stream, line: chunk.toString() }) + "\n"
      );
    };
    proc.stdout.on("data", append("stdout"));
    proc.stderr.on("data", append("stderr"));

    const exitCode: number = await new Promise((resolve, reject) => {
      proc.on("error", reject);
      proc.on("close", (code) => resolve(code ?? -1));
    });

    await logStream.write(
      JSON.stringify({
        t: Date.now(),
        s: "system",
        line: `exit ${exitCode} in ${Date.now() - started}ms`
      }) + "\n"
    );
    await logStream.close();

    if (exitCode !== 0) {
      throw new PipelineError(`Pipeline exited with code ${exitCode}`, {
        workspaceDir: input.workspaceDir,
        exitCode
      });
    }

    const outputPath = path.join(outputDir, "model.json");
    let output: ModelOutput | undefined;
    try {
      const raw = await fs.readFile(outputPath, "utf8");
      output = JSON.parse(raw) as ModelOutput;
    } catch {
      output = undefined;
    }

    return {
      output,
      outputPath,
      cliCommand,
      exitCode,
      workspaceDir: input.workspaceDir
    };
  }
}

export const runner: PipelineRunner = new SubprocessRunner();

export function runWorkspaceDir(runId: string): string {
  return path.join(process.cwd(), "storage", "runs", runId);
}
