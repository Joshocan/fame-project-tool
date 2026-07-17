export const appConfig = {
  githubTokenConfigured: Boolean(process.env.GITHUB_TOKEN),
  anthropicKeyConfigured: Boolean(process.env.ANTHROPIC_API_KEY),
  chromaPath: process.env.FAME_CHROMA_PATH ?? "../data/chroma_db",
  neo4jUri: process.env.NEO4J_URI ?? "",
  resultsRoot: process.env.FAME_RESULTS_ROOT ?? "../results"
};
