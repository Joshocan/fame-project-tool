import { PipelinePreset } from "./models";

export const pipelinePresets: PipelinePreset[] = [
  {
    id: "nonrag-single",
    name: "Non-RAG / Single-stage",
    retrievalMode: "non-rag",
    promptingMode: "single-stage",
    description: "Bundle artefacts directly into a single Claude extraction pass."
  },
  {
    id: "nonrag-iterative",
    name: "Non-RAG / Iterative",
    retrievalMode: "non-rag",
    promptingMode: "iterative",
    description: "Direct artefact prompting with iterative refinement across steps."
  },
  {
    id: "rag-single",
    name: "RAG / Single-stage",
    retrievalMode: "rag",
    promptingMode: "single-stage",
    description: "Chunk and index selected artefacts, then retrieve context for one extraction pass."
  },
  {
    id: "rag-iterative",
    name: "RAG / Iterative",
    retrievalMode: "rag",
    promptingMode: "iterative",
    description: "Chunk, retrieve, and refine the model over multiple prompts."
  }
];

export function getPipelinePreset(id: string) {
  return pipelinePresets.find((preset) => preset.id === id);
}
