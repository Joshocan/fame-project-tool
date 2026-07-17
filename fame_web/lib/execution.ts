import { getPipelinePreset } from "./pipelines";
import { BundleRecord, ModelOutput, RunRecord, ValidationSummary, WorkflowStep } from "./models";

function buildModel(repoKey: string): ModelOutput {
  return {
    modelId: `model-${repoKey.replace("/", "-")}`,
    systemName: repoKey,
    nodes: [
      { id: "sys1", type: "System", name: repoKey, description: "Recovered healthcare software system." },
      { id: "cap1", type: "Capability", name: "ManageClinicalData", description: "Handles core healthcare information." },
      { id: "data1", type: "DataConcept", name: "PatientRecord", description: "Patient-oriented data concept." },
      { id: "if1", type: "Interface", name: "ServiceAPI", description: "External integration interface." }
    ],
    edges: [
      { source: "sys1", type: "contains", target: "cap1", description: "System capability." },
      { source: "cap1", type: "produces", target: "data1", description: "Produces or manages patient data." },
      { source: "cap1", type: "usesInterface", target: "if1", description: "Exposes an interface." }
    ]
  };
}

export function buildInitialWorkflowSteps(bundle: BundleRecord, pipelineId: string): WorkflowStep[] {
  const pipeline = getPipelinePreset(pipelineId);
  const artefactCount = bundle.selectedArtefactPaths.length;

  return [
    {
      key: "preprocessing",
      title: "Preprocessing",
      status: "active",
      details: [
        `${artefactCount} selected artefacts pinned to commit ${bundle.commitSha}`,
        pipeline?.retrievalMode === "rag"
          ? "RAG preset selected: normalize text, chunk artefacts, build retrieval index."
          : "Non-RAG preset selected: normalize text and assemble direct prompt bundle."
      ]
    },
    {
      key: "construction",
      title: "Model construction",
      status: "pending",
      details: [
        "Claude extraction has not started yet.",
        pipeline ? `Prompting mode: ${pipeline.promptingMode}.` : "Pipeline metadata unavailable."
      ]
    },
    {
      key: "validation",
      title: "Validation",
      status: "pending",
      details: ["Schema, metamodel, and trust checks have not run yet."]
    },
    {
      key: "review",
      title: "Review outputs",
      status: "pending",
      details: ["Raw JSON, validation report, and graph projection will appear here."]
    }
  ];
}

function completedValidation(bundle: BundleRecord, pipelineId: string): ValidationSummary {
  const pipeline = getPipelinePreset(pipelineId);
  return {
    schemaValid: true,
    metamodelConformance: pipeline?.retrievalMode === "rag" ? "pass" : "warning",
    notes: [
      `Selected artefacts: ${bundle.selectedArtefactPaths.length}`,
      pipeline ? `Pipeline preset: ${pipeline.name}` : "Pipeline preset missing in catalog.",
      "Mock output only. Replace with CLI execution for real runs.",
      "At this stage, the run demonstrates why model selection and validation are necessary before trusting outputs."
    ]
  };
}

export function advanceMockRun(run: RunRecord, bundle: BundleRecord): Partial<RunRecord> {
  const pipeline = getPipelinePreset(run.pipelineId);

  if (run.status === "preprocessing") {
    return {
      status: "ready",
      steps: run.steps.map((step) =>
        step.key === "preprocessing"
          ? {
              ...step,
              status: "completed",
              details: [...step.details, "Artefacts normalized and preprocessing plan prepared."]
            }
          : step.key === "construction"
            ? {
                ...step,
                status: "active",
                details: [
                  `Ready to call Claude for ${bundle.repoKey}.`,
                  pipeline ? `Using preset ${pipeline.name}.` : "Pipeline metadata unavailable."
                ]
              }
            : step
      )
    };
  }

  if (run.status === "ready") {
    return {
      status: "running",
      steps: run.steps.map((step) =>
        step.key === "construction"
          ? {
              ...step,
              status: "active",
              details: [
                `Running Claude extraction over ${bundle.selectedArtefactPaths.length} curated artefacts.`,
                pipeline?.retrievalMode === "rag"
                  ? "Retrieval context injected into model construction prompt."
                  : "Direct artefact bundle sent without retrieval index."
              ]
            }
          : step
      )
    };
  }

  if (run.status === "running") {
    return {
      status: "validating",
      steps: run.steps.map((step) =>
        step.key === "construction"
          ? {
              ...step,
              status: "completed",
              details: [...step.details, "Candidate model JSON returned."]
            }
          : step.key === "validation"
            ? {
                ...step,
                status: "active",
                details: [
                  "Checking JSON structure, node and edge typing, and metamodel conformance.",
                  "Preparing trust-oriented review notes."
                ]
              }
            : step
      )
    };
  }

  if (run.status === "validating") {
    const output = buildModel(bundle.repoKey);
    return {
      status: "completed",
      output: {
        ...output,
        modelId: `${output.modelId}-${run.pipelineId}`,
        systemName: output.systemName
      },
      validation: completedValidation(bundle, run.pipelineId),
      graphExport: {
        neo4jReady: true,
        note: "Graph export placeholder ready. Wire this to Neo4j import in the next increment."
      },
      steps: run.steps.map((step) => {
        if (step.key === "validation") {
          return {
            ...step,
            status: "completed",
            details: [
              "Schema and structural checks passed.",
              pipeline?.retrievalMode === "rag"
                ? "Metamodel conformance stronger under retrieved context."
                : "Metamodel conformance flagged as warning to reflect weaker grounding."
            ]
          };
        }
        if (step.key === "review") {
          return {
            ...step,
            status: "completed",
            details: [
              "Raw JSON available.",
              "Validation report available.",
              "Graph / Neo4j projection placeholder available."
            ]
          };
        }
        return step;
      })
    };
  }

  return {};
}
