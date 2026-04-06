"use client";

const DEMOS = [
  {
    id: "flappy",
    icon: "🐦",
    name: "Flappy Evolution",
    description: "50 neural networks learn to play Flappy Bird. GPU evaluates 4,096 birds per dispatch via fused kernel.",
    tags: ["Kernel Fusion", "WebRTC P2P", "OpenAI ES"],
    metrics: [
      { label: "Population", value: "4,096" },
      { label: "NN params", value: "22" },
      { label: "Genome size", value: "88 B" },
    ],
    href: "/demos/flappy.html",
    featured: true,
  },
  {
    id: "rastrigin-p2p",
    icon: "📊",
    name: "P2P Rastrigin Scaling",
    description: "2,000-dimensional multimodal optimization. Open multiple tabs to form an island-model swarm via WebRTC.",
    tags: ["Island Model", "WebRTC P2P", "Diversity Injection"],
    metrics: [
      { label: "Dimensions", value: "2,000" },
      { label: "Population", value: "4,096" },
      { label: "Improvement", value: "+28.8%" },
    ],
    href: "/demos/rastrigin-p2p.html",
    featured: false,
  },
  {
    id: "petase",
    icon: "🧬",
    name: "PETase Enzyme Evolution",
    description: "Distributed island-model evolution of PETase plastic-degrading enzyme. Islands specialize on stability vs activity.",
    tags: ["Protein Design", "Multi-Objective", "Island Model"],
    metrics: [
      { label: "Evolvable positions", value: "65" },
      { label: "Fitness components", value: "4" },
      { label: "Islands", value: "3" },
    ],
    href: "/demos/petase-p2p.html",
    featured: false,
  },
  {
    id: "neoantigen",
    icon: "💉",
    name: "Multi-Allele Neoantigen",
    description: "Cancer neoantigen design across HLA alleles. Each island evolves peptides for a different MHC-I allele.",
    tags: ["Cancer Research", "Multi-Allele", "Island Model"],
    metrics: [
      { label: "HLA alleles", value: "3" },
      { label: "Peptide length", value: "9-mer" },
      { label: "Anchor discovery", value: "Auto" },
    ],
    href: "/demos/neoantigen-p2p.html",
    featured: false,
  },
];

const PAPER_RESULTS = [
  { number: "28.8%", label: "fitness improvement (4 P2P islands, shared GPU)" },
  { number: "14.6%", label: "improvement on independent RTX 3090 GPUs" },
  { number: "97%", label: "performance maintained at 50% Byzantine nodes" },
  { number: "20 KB/s", label: "total bandwidth for 4-node mesh" },
  { number: "10\u00D7", label: "speed gap bridged by elite injection (iPhone vs Mac)" },
  { number: "82%", label: "of full-mesh benefit with k4-regular topology (74% fewer edges)" },
];

export default function SwarmPage() {
  return (
    <div className="min-h-screen">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.05)_0%,_transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,_rgba(74,222,128,0.03)_0%,_transparent_50%)]" />
      </div>

      <header className="max-w-3xl mx-auto px-6 pt-8 pb-6 text-center">
        {/* Logo */}
        <div className="flex items-center justify-center gap-2 mb-6">
          <svg className="w-5 h-5 text-bench-accent animate-spin-slow" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="4 4" />
            <circle cx="12" cy="12" r="4" fill="currentColor" />
          </svg>
          <span className="font-bold text-bench-text">WebGPU Bench</span>
        </div>

        {/* Tab Switcher */}
        <div className="inline-flex rounded-lg bg-bench-surface border border-bench-border p-1 mb-8">
          <a href="/" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            GPU Compute
          </a>
          <a href="/transformer" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            Transformer Fusion
          </a>
          <span className="px-4 py-2 rounded-md text-sm font-medium bg-bench-accent/10 text-bench-accent">
            Distributed P2P
          </span>
        </div>

        <h1 className="text-3xl md:text-4xl font-bold tracking-tight leading-[1.1] mb-3">
          Distributed Evolution{" "}
          <span className="bg-gradient-to-r from-bench-green to-bench-accent bg-clip-text text-transparent">
            via WebRTC
          </span>
        </h1>
        <p className="text-base text-bench-muted max-w-lg mx-auto">
          Browser tabs serve as evolutionary islands, exchanging elite genomes peer-to-peer.
          GPU kernel fusion + WebRTC data channels. Zero install.
        </p>
      </header>

      <main className="max-w-3xl mx-auto px-6 pb-24 space-y-6">

        {/* Paper Results */}
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full bg-bench-accent/10 text-bench-accent">
              Paper 3
            </span>
            <span className="text-xs text-bench-muted">Browser-to-Browser Evolutionary Computation</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {PAPER_RESULTS.map((r) => (
              <div key={r.label} className="bg-bench-bg rounded-lg p-2.5 text-center">
                <div className="text-lg font-extrabold text-bench-accent">{r.number}</div>
                <div className="text-[9px] text-bench-muted mt-0.5 leading-tight">{r.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Demos */}
        <div className="flex items-center gap-3 mt-8">
          <div className="flex-1 h-px bg-bench-border" />
          <span className="text-[10px] text-bench-muted font-medium uppercase tracking-widest">Live Demos</span>
          <div className="flex-1 h-px bg-bench-border" />
        </div>

        {DEMOS.map((demo) => (
          <div
            key={demo.id}
            className={`card transition-all ${demo.featured ? "ring-1 ring-bench-accent/20" : ""}`}
          >
            {demo.featured && (
              <div className="flex items-center gap-2 mb-3">
                <span className="text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full bg-bench-green/10 text-bench-green">
                  Featured
                </span>
                <span className="text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full bg-bench-accent/10 text-bench-accent">
                  Kernel Fusion + P2P
                </span>
              </div>
            )}

            <div className="flex items-start gap-4">
              <div className={`
                w-11 h-11 rounded-lg flex items-center justify-center text-xl
                ${demo.featured ? "bg-bench-accent/10 border border-bench-accent/20" : "bg-bench-surface border border-bench-border/50"}
              `}>
                {demo.icon}
              </div>

              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-bench-text text-[15px] mb-1">{demo.name}</h3>
                <p className="text-xs text-bench-muted/80 leading-relaxed mb-3">{demo.description}</p>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-2 mb-3">
                  {demo.metrics.map((m) => (
                    <div key={m.label} className="bg-bench-bg rounded-md px-2 py-1.5 text-center">
                      <div className="text-sm font-bold text-bench-accent">{m.value}</div>
                      <div className="text-[8px] text-bench-muted mt-0.5">{m.label}</div>
                    </div>
                  ))}
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {demo.tags.map((tag) => (
                    <span key={tag} className="text-[9px] text-bench-muted bg-bench-bg rounded px-2 py-0.5 border border-bench-border/50">
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Action */}
                <div className="flex gap-2">
                  <a
                    href={demo.href}
                    className="text-xs font-medium px-4 py-1.5 rounded-md bg-bench-accent/10 text-bench-accent hover:bg-bench-accent/20 transition"
                  >
                    Launch Demo
                  </a>
                  <span className="text-xs font-medium px-3 py-1.5 rounded-md bg-bench-border/30 text-bench-muted">
                    Open in 2+ tabs for P2P
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* How it works */}
        <div className="card mt-8">
          <h2 className="text-base font-bold mb-4">How Distributed Evolution Works</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-bench-bg rounded-lg p-4">
              <div className="text-2xl mb-2">🖥️</div>
              <h3 className="text-sm font-semibold mb-1">1. GPU Evolution</h3>
              <p className="text-[11px] text-bench-muted leading-relaxed">
                Each browser tab runs a fused WebGPU kernel that evaluates 4,096 neural networks in a single GPU dispatch. No per-step overhead.
              </p>
            </div>
            <div className="bg-bench-bg rounded-lg p-4">
              <div className="text-2xl mb-2">🔗</div>
              <h3 className="text-sm font-semibold mb-1">2. WebRTC Discovery</h3>
              <p className="text-[11px] text-bench-muted leading-relaxed">
                Tabs discover each other via a 130-line signaling relay. WebRTC data channels form a direct P2P mesh. No server after handshake.
              </p>
            </div>
            <div className="bg-bench-bg rounded-lg p-4">
              <div className="text-2xl mb-2">🧬</div>
              <h3 className="text-sm font-semibold mb-1">3. Elite Exchange</h3>
              <p className="text-[11px] text-bench-muted leading-relaxed">
                Best genomes flow between islands at 88 bytes each. Different islands explore different fitness landscapes. Diversity injection beats isolation by 28.8%.
              </p>
            </div>
          </div>
        </div>

        {/* Architecture diagram (text) */}
        <div className="card">
          <h2 className="text-base font-bold mb-3">Architecture</h2>
          <div className="bg-bench-bg rounded-lg p-4 font-mono text-[11px] text-bench-muted leading-relaxed">
            <div className="flex flex-col items-center gap-1">
              <div className="flex gap-8">
                <span className="text-bench-accent">Tab 1 (GPU)</span>
                <span className="text-bench-muted">{"◄──WebRTC──►"}</span>
                <span className="text-bench-green">Tab 2 (GPU)</span>
              </div>
              <div className="text-bench-muted/50">│ genome exchange │</div>
              <div className="flex gap-8">
                <span className="text-bench-muted/50">{"     ▼"}</span>
                <span className="text-bench-muted/50">{"                    ▼"}</span>
              </div>
              <div className="flex gap-4">
                <span className="text-bench-yellow">Signaling Relay</span>
                <span className="text-bench-muted/50">(handshake only, ~130 LOC)</span>
              </div>
            </div>
          </div>
          <div className="flex gap-2 mt-3">
            <a
              href="https://kernelfusion.dev"
              className="text-xs font-medium px-3 py-1.5 rounded-md bg-bench-accent/10 text-bench-accent hover:bg-bench-accent/20 transition"
            >
              Paper 1: Kernel Fusion
            </a>
            <a
              href="https://kernelfusion.dev"
              className="text-xs font-medium px-3 py-1.5 rounded-md bg-bench-green/10 text-bench-green hover:bg-bench-green/20 transition"
            >
              Paper 3: WebRTC P2P
            </a>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center pt-8">
          <p className="text-xs text-bench-muted">
            Part of the{" "}
            <a href="https://kernelfusion.dev" className="text-bench-accent hover:underline">kernelfusion.dev</a>
            {" "}research project.{" "}
            <a href="https://github.com/abgnydn/the-swarm" className="text-bench-accent hover:underline">Source code on GitHub</a>.
          </p>
        </div>
      </main>
    </div>
  );
}
