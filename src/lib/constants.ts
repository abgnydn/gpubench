// ═══════════════════════════════════════════════════════════
// Project-specific links for gpubench.dev
//
// Cross-site data lives in ./sites.ts — synced from
// ~/sites-shared/sites.ts. DO NOT duplicate SITES URLs here.
// ═══════════════════════════════════════════════════════════

import { SITES } from "./sites";

export { SITES, CROSSLINKS, AUTHOR, SAME_AS } from "./sites";
export type { SiteKey, SiteInfo } from "./sites";

// Concept DOIs that auto-resolve to the latest published version on Zenodo
// (currently v6 / v2 as of 2026-05-04). Stable across future bumps.
const EC_DOI = "10.5281/zenodo.19331833";
const TRANSFORMER_DOI = "10.5281/zenodo.19344276";

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
  research: SITES.kernelfusion.url,
  site: SITES.gpubench.url,

  // WebGPU-DNA
  webgpuDnaSite: SITES.webgpudna.url,
  webgpuDnaRepo: SITES.webgpudna.githubRepo!,

  // Neuropulse
  neuropulseSite: SITES.neuropulse.url,
  neuropulseRepo: SITES.neuropulse.githubRepo!,
} as const;

export const STATS = {
  // Paper numbers (2 machines: M2 Pro, T4)
  webgpuOverPytorch: "159",
  cudaOverPytorch: "720",
  jaxOverPytorch: "172",
  tritonOverPytorch: "27",
  browserOverhead: "1.92",
  fusionAblation: "2.18",
  // Real-world numbers — medians from gpubench DB snapshot 2026-05-04.
  // Means are not used because Safari-on-macOS produces measurement
  // artifacts on the unfused baseline (peaks like 79,021× are real
  // numbers but describe Safari's WebGPU stalling, not Apple Silicon
  // performance). Medians filter those out cleanly without an explicit
  // outlier rule, and stay stable as the DB grows.
  // 92 unique devices, 7 GPU vendors, 890 total runs.
  appleMedianSpeedup: "71",
  adrenoMedianSpeedup: "20",
  nvidiaMedianSpeedup: "56",
  armMedianSpeedup: "55",
  applePeakSpeedup: "226",
  nvidiaPeakSpeedup: "402",
  adrenoPeakSpeedup: "103",
  totalDevices: "92",
  totalRuns: "890",
  vendorCount: "7",
  mobileTokensPerSecAvg: "15000",
  mobileTokensPerSecPeak: "213000",
} as const;
