// ═══════════════════════════════════════════
// Single source of truth for all external links
// Update here, changes everywhere.
// ═══════════════════════════════════════════

const EC_DOI = "10.5281/zenodo.19343570";
const TRANSFORMER_DOI = "10.5281/zenodo.19344277";

export const LINKS = {
  // Papers
  ecDoi: `https://doi.org/${EC_DOI}`,
  ecDoiShort: `doi:${EC_DOI}`,
  transformerDoi: `https://doi.org/${TRANSFORMER_DOI}`,
  transformerDoiShort: `doi:${TRANSFORMER_DOI}`,

  // Repos
  paper: "https://github.com/abgnydn/webgpu-kernel-fusion",
  transformerPaper: "https://github.com/abgnydn/webgpu-transformer-fusion",
  repo: "https://github.com/abgnydn/gpubench",
  research: "https://kernelfusion.dev",
  site: "https://gpubench.dev",
} as const;

export const STATS = {
  webgpuOverPytorch: "159",
  cudaOverPytorch: "720",
  jaxOverPytorch: "172",
  tritonOverPytorch: "27",
  browserOverhead: "1.92",
  fusionAblation: "2.18",
} as const;
