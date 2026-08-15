// ═══════════════════════════════════════════════════════════════════════
// sites.ts — cross-project links & metadata. DO NOT EDIT.
//
// This is a copy of sites-shared/generated/sites.json, which build-sites.mjs
// renders from facts.json. Every number below traces to a fact id there.
// Hand-editing this file is how one inverted comparison reached four live
// surfaces. To change anything: fix facts.json or sites.source.json, run
// `node bin/build-sites.mjs` in sites-shared, then resync with
//   cp ~/dev/sites-shared/generated/sites.json <this file's data>
// keeping the interface and CROSSLINKS/AUTHOR/SAME_AS tail below intact.
//
// Schema:
//   SITES[key]: url, domain, tagline, category, badge, stats, githubRepo
//   CROSSLINKS[key]: which sites to render as "see also" on that site
//   SAME_AS: flat URL list for JSON-LD `sameAs` (Person/Organization)
//   AUTHOR: name/email/github/linkedin for schema.org Person
// ═══════════════════════════════════════════════════════════════════════

export interface SiteInfo {
  readonly key: string;
  readonly url: string;
  readonly domain: string;
  readonly name: string;
  readonly tagline: string;
  readonly shortDesc: string;
  readonly category: string;
  readonly badge?: "flagship" | "research" | "applied" | "companion" | "personal";
  readonly githubRepo?: string;
  readonly stats?: readonly { value: string; label: string }[];
  readonly cta?: string;
}

export const SITES = {
  kernelfusion: {
    key: "kernelfusion",
    url: "https://kernelfusion.dev",
    domain: "kernelfusion.dev",
    name: "kernelfusion.dev",
    tagline: "Single-kernel fusion — sequential GPU dispatch chains collapsed into one dispatch.",
    shortDesc:
      "The research line. Two published preprints and one npm SDK. The theory the applied projects build on.",
    category: "Theory",
    badge: "research",
    githubRepo: "https://github.com/abgnydn/kernelfusion",
    stats: [
      { value: "DOI-archived", label: "two preprints on Zenodo" },
      { value: "@wgpu-fusion/core", label: "npm SDK" },
      { value: "WGSL", label: "hand-written kernels" },
    ],
    cta: "Read the research →",
  },
  webgpudna: {
    key: "webgpudna",
    url: "https://webgpudna.com",
    domain: "webgpudna.com",
    name: "webgpudna.com",
    tagline: "Geant4-DNA Monte Carlo in the browser. One fused dispatch per primary.",
    shortDesc:
      "Electron track-structure simulation ported from the CNRS/IN2P3 Geant4-DNA toolkit to WebGPU. One thread per primary, full 10 keV history in a single for-loop. Radiolysis chemistry and DNA damage scoring live in a browser tab.",
    category: "Radiobiology",
    badge: "flagship",
    githubRepo: "https://github.com/abgnydn/webgpu-dna",
    stats: [
      { value: "71", label: "unit tests pass" },
      { value: "Geant4-DNA", label: "physics ported" },
      { value: "validated", label: "radiolysis vs Karamitros 2011" },
    ],
    cta: "See the simulation →",
  },
  zerotvm: {
    key: "zerotvm",
    url: "https://zerotvm.com",
    domain: "zerotvm.com",
    name: "zerotvm.com",
    tagline: "Phi-3-mini in the browser on 51 hand-written WGSL kernels, no compiler.",
    shortDesc:
      "Phi-3-mini (3.8B) running end-to-end on 51 hand-written WGSL files, replacing the 85 TVM-autotuned shaders WebLLM needs. 69.55 tok/s against WebLLM's 59.95 on identical weights in the same session, +16%.",
    category: "LLM inference",
    badge: "applied",
    githubRepo: "https://github.com/abgnydn/zero-tvm",
    stats: [
      { value: "85", label: "TVM shaders replaced" },
      { value: "51", label: "WGSL files" },
      { value: "69.55", label: "tok/s (M2 Max)" },
    ],
    cta: "Run it live →",
  },
  gpubench: {
    key: "gpubench",
    url: "https://gpubench.dev",
    domain: "gpubench.dev",
    name: "gpubench.dev",
    tagline: "How fast is your GPU in the browser?",
    shortDesc:
      "Open WebGPU compute benchmarks — Rastrigin, N-body, Monte Carlo Pi, RL environments. 794 runs published, no cherry-picking.",
    category: "Benchmarks",
    badge: "companion",
    githubRepo: "https://github.com/abgnydn/gpubench",
    stats: [
      { value: "794", label: "runs published" },
      { value: "public", label: "every run downloadable" },
    ],
    cta: "Benchmark your GPU →",
  },
  neuropulse: {
    key: "neuropulse",
    url: "https://neuropulse.live",
    domain: "neuropulse.live",
    name: "neuropulse.live",
    tagline: "Watch Phi-3-mini think — every tensor rendered live from WebGPU.",
    shortDesc:
      "A real forward pass of Phi-3-mini visualised tensor-by-tensor. 3.8 billion parameters, your GPU, your browser — every glow is a live activation read back from WebGPU. Zero server, zero API key.",
    category: "Visualization",
    badge: "applied",
    githubRepo: "https://github.com/abgnydn/neuropulse",
    stats: [
      { value: "Phi-3-mini", label: "3.8B params, rendered live" },
      { value: "every tensor", label: "rendered 1:1" },
      { value: "offline", label: "zero server calls" },
    ],
    cta: "Watch it think →",
  },
  barisgunaydin: {
    key: "barisgunaydin",
    url: "https://barisgunaydin.com",
    domain: "barisgunaydin.com",
    name: "barisgunaydin.com",
    tagline: "Independent researcher — kernel fusion, GPU compute, browser-native ML.",
    shortDesc: "Personal site and project hub.",
    category: "Personal",
    badge: "personal",
    cta: "About →",
  },
  markview: {
    key: "markview",
    url: "https://markview.ai",
    domain: "markview.ai",
    name: "markview.ai",
    tagline: "Markdown editor.",
    shortDesc: "Markdown editor and desktop app.",
    category: "Utility",
    githubRepo: "https://github.com/abgnydn/markview",
    cta: "Open →",
  },
  safenpm: {
    key: "safenpm",
    url: "https://safenpm.dev",
    domain: "safenpm.dev",
    name: "safenpm.dev",
    tagline: "npm install, but you see what you're installing.",
    shortDesc:
      "Inspect npm packages before adding them to your project — dependency tree, post-install scripts, license, maintainer, file list. Pre-install sanity check for the registry.",
    category: "Tooling",
    githubRepo: "https://github.com/abgnydn/safenpm",
    cta: "Inspect a package →",
  },
  fusedlora: {
    key: "fusedlora",
    url: "https://fused-lora.pages.dev",
    domain: "fused-lora.pages.dev",
    name: "fused-lora",
    tagline: "LoRA fine-tuning of BitNet b1.58 2B4T, live in a browser tab.",
    shortDesc:
      "First public demo of fine-tuning a ternary LLM entirely in the browser via WebGPU. ~86 ms/step on M-series laptops. Adapters emit as 4 MB .flora files. Includes a Phi-3 LoRA reference.",
    category: "LLM training",
    badge: "research",
    githubRepo: "https://github.com/abgnydn/fused-lora",
    cta: "Fine-tune in browser →",
  },
  webgpuDna: {
    key: "webgpuDna",
    url: "https://webgpu-dna.pages.dev",
    domain: "webgpu-dna.pages.dev",
    name: "webgpu-dna (secondary)",
    tagline: "Secondary Cloudflare Pages deploy for webgpudna (preview/experimental surface).",
    shortDesc:
      "Separate CF Pages project used alongside the canonical webgpudna.com — kept live intentionally. Not a distinct product; same codebase.",
    category: "Radiobiology",
    githubRepo: "https://github.com/abgnydn/webgpu-dna",
    cta: "Open preview →",
  },
  webgpuq: {
    key: "webgpuq",
    url: "https://webgpu-q.vercel.app",
    domain: "webgpu-q.vercel.app",
    name: "webgpu-q",
    tagline: "Full-featured quantum circuit simulator in a single browser tab.",
    shortDesc:
      "Statevector + MPS quantum simulator running on commodity hardware via WebGPU compute. Six-level research ladder from bandwidth-bound statevector through MPS, kernel fusion, WebRTC swarm, IBM hardware cross-verify, to chemistry/VQE. No CUDA, no install.",
    category: "Quantum",
    badge: "research",
    cta: "Open the simulator →",
  },
} as const satisfies Record<string, SiteInfo>;

export type SiteKey = keyof typeof SITES;

export const AUTHOR = {
  name: "Ahmet Baris Gunaydin",
  email: "hi@barisgunaydin.com",
  github: "https://github.com/abgnydn",
  linkedin: "https://www.linkedin.com/in/abgnydn/",
} as const;

// Flat list for JSON-LD Person/Organization `sameAs` — every canonical URL
// where this identity shows up.
export const SAME_AS: readonly string[] = [
  SITES.barisgunaydin.url,
  SITES.kernelfusion.url,
  SITES.gpubench.url,
  SITES.zerotvm.url,
  SITES.webgpudna.url,
  SITES.neuropulse.url,
  SITES.webgpuq.url,
  SITES.markview.url,
  SITES.safenpm.url,
  SITES.fusedlora.url,
  AUTHOR.github,
  AUTHOR.linkedin,
];

// Which sibling sites each project surfaces in its "see also" area.
// Order matters — first entry gets the featured / flagship slot where the
// rendering host elevates one card.
export const CROSSLINKS: Record<SiteKey, readonly SiteKey[]> = {
  kernelfusion:  ["webgpudna", "gpubench", "zerotvm", "neuropulse", "webgpuq", "barisgunaydin"],
  webgpudna:     ["kernelfusion", "gpubench", "zerotvm", "neuropulse", "webgpuq", "barisgunaydin"],
  zerotvm:       ["kernelfusion", "webgpudna", "gpubench", "neuropulse", "webgpuq", "barisgunaydin"],
  gpubench:      ["kernelfusion", "webgpudna", "zerotvm", "neuropulse", "webgpuq", "barisgunaydin"],
  neuropulse:    ["kernelfusion", "webgpudna", "zerotvm", "gpubench", "webgpuq", "barisgunaydin"],
  barisgunaydin: ["kernelfusion", "webgpudna", "gpubench", "zerotvm", "neuropulse", "webgpuq", "markview", "safenpm"],
  markview:      ["barisgunaydin", "kernelfusion"],
  safenpm:       ["barisgunaydin", "kernelfusion"],
  fusedlora:     ["kernelfusion", "zerotvm", "barisgunaydin"],
  webgpuq:       ["kernelfusion", "webgpudna", "gpubench", "zerotvm", "neuropulse", "barisgunaydin"],
  webgpuDna:     ["webgpudna"],
};
