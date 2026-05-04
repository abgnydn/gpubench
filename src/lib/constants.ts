// ═══════════════════════════════════════════════════════════
// Project-specific links for gpubench.dev
//
// Cross-site data lives in ./sites.ts — synced from
// ~/sites-shared/sites.ts. DO NOT duplicate SITES URLs here.
// ═══════════════════════════════════════════════════════════

import { SITES } from "./sites";

export { SITES, CROSSLINKS, AUTHOR, SAME_AS } from "./sites";
export type { SiteKey, SiteInfo } from "./sites";

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
  research: SITES.kernelfusion.url,
  site: SITES.gpubench.url,

  // Zero-TVM — nested pages under zerotvm.com
  zerotvmSite: SITES.zerotvm.url,
  zerotvmRepo: SITES.zerotvm.githubRepo!,
  zerotvmChat: `${SITES.zerotvm.url}/zero-tvm.html`,
  zerotvmDocs: `${SITES.zerotvm.url}/docs.html`,
  zerotvmArchitecture: `${SITES.zerotvm.url}/architecture.html`,
  zerotvmDispatchViz: `${SITES.zerotvm.url}/demo.html`,
  zerotvmWebllmBench: `${SITES.zerotvm.url}/webllm-bench.html`,
  zerotvmCompilerChat: `${SITES.zerotvm.url}/compiler-chat.html`,
  zerotvmShaderDump: `${SITES.zerotvm.url}/dump.html`,
  zerotvmShaderSource: `${SITES.zerotvm.url}/shaders.html`,
  zerotvmValidate: `${SITES.zerotvm.url}/validate.html`,

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
  // Real-world numbers (592 devices, 7 GPU vendors)
  appleAvgSpeedup: "2865",
  adrenoAvgSpeedup: "623",
  nvidiaAvgSpeedup: "79",
  armAvgSpeedup: "56",
  totalRuns: "592",
  mobileTokensPerSecAvg: "15000",
  mobileTokensPerSecPeak: "213000",
} as const;
