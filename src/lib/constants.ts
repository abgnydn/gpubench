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
