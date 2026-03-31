// ═══════════════════════════════════════════
// Single source of truth for all external links
// Update here, changes everywhere.
// ═══════════════════════════════════════════

export const LINKS = {
  doi: "https://doi.org/10.5281/zenodo.19342888",
  doiShort: "doi:10.5281/zenodo.19342888",
  github: "https://github.com/abgnydn/webgpu-kernel-fusion",
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
