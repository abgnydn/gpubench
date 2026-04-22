import type { Metadata } from "next";
import { LINKS } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Why This Matters — WebGPU Bench",
  description: "We proved browsers can run GPU computation at near-native speed. No download. No account. No expensive hardware.",
};

const barriers = [
  { icon: "\u{1F512}", title: "Hardware lock-in", desc: "CUDA only runs on NVIDIA GPUs. If you have a Mac, an AMD card, or an Intel integrated chip \u2014 you're out." },
  { icon: "\u{1F9E9}", title: "Software complexity", desc: "Python environments, CUDA drivers, cuDNN versions, framework dependencies. One mismatch and nothing works." },
  { icon: "\u{1F4B0}", title: "Cost", desc: "Cloud GPU instances cost $2\u20134/hour. A university lab without GPU budget simply can't participate." },
];

const personas = [
  { icon: "\u{1F393}", title: "The student who can't afford a GPU cluster", desc: "A computer science student in Lagos, Bangalore, or rural anywhere can now run real GPU-accelerated experiments on their $400 laptop. Open a browser tab, not a grant application." },
  { icon: "\u{1F469}\u200D\u{1F3EB}", title: "The teacher who wants to show, not just tell", desc: "Instead of slides about parallel computing, show it running live in the classroom. Every student's laptop becomes a GPU workstation. No lab setup, no admin permissions." },
  { icon: "\u{1F52C}", title: "The researcher who wants reproducible science", desc: "\u201CTo reproduce our results, open this URL.\u201D Not \u201Cinstall Python 3.10.12, CUDA 12.1, cuDNN 8.9.7, match the exact driver version.\u201D Reproducibility should be one click." },
  { icon: "\u{1F680}", title: "The startup that can't justify cloud bills yet", desc: "A biotech team in Nairobi running molecular screening. A fintech in Sao Paulo backtesting strategies. Their scientists' laptops ARE the compute cluster. No AWS required." },
];

const steps = [
  { title: "Your GPU is already powerful", desc: "The graphics chip in your laptop \u2014 whether it's Apple Silicon, AMD, Intel, or NVIDIA \u2014 is a parallel processor with thousands of cores. It spends most of its time idle." },
  { title: "Browsers can now talk to it", desc: "WebGPU is a new web standard (shipping in Chrome since 2023) that lets web pages run computation directly on your GPU. Not just graphics \u2014 real number crunching." },
  { title: "We figured out how to make it fast", desc: "Instead of sending thousands of small tasks to the GPU one by one (like PyTorch does), we pack the entire computation into a single instruction. One dispatch instead of 22,500. The browser only adds 48% overhead vs native, and it's still faster than PyTorch." },
  { title: "We proved it with real benchmarks", desc: "30 independent runs per experiment. Statistical tests. Comparisons against 8 systems on 2 hardware platforms. Not hype \u2014 evidence." },
];

const beforeAfter = [
  { before: "\u201CReproduce our results\u201D \u2192 Install Python 3.10.12 \u2192 Install CUDA 12.1 \u2192 Install cuDNN 8.9.7 \u2192 Match driver version \u2192 Debug for 3 hours \u2192 Maybe it works", after: "Click this link. Results run in your browser. Verified in 30 seconds." },
  { before: "\u201CI need GPU compute\u201D \u2192 AWS account \u2192 GPU instance ($2\u20134/hr) \u2192 DevOps setup \u2192 $5,000/month bill \u2192 Need funding before building", after: "Users' own GPUs do the work. Server cost: $0. Ship on day one." },
  { before: "\u201CToday we'll learn parallel computing\u201D \u2192 Show slides \u2192 Students nod \u2192 Nobody runs anything because the school has no GPU lab", after: "\u201COpen this URL on your laptop.\u201D 30 students run GPU computation simultaneously. They see it, touch it, modify it." },
  { before: "\u201CRun screening on patient data\u201D \u2192 6-month legal review to upload to cloud \u2192 HIPAA/GDPR audit \u2192 $200K contract \u2192 Finally start work", after: "Data never leaves the laptop. GPU compute runs in the browser. Compliance by architecture, not by contract." },
];

export default function WhyPage() {
  return (
    <div className="min-h-screen">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.05)_0%,_transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,_rgba(168,85,247,0.04)_0%,_transparent_50%)]" />
      </div>

      <div className="max-w-3xl mx-auto px-6">

        {/* Nav */}
        <nav className="flex items-center justify-between py-6">
          <a href="/" className="flex items-center gap-2 text-sm text-bench-muted hover:text-bench-text transition">
            <svg className="w-4 h-4 text-bench-accent" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2.5" strokeDasharray="60 28" strokeLinecap="round"/>
              <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
            </svg>
            <span className="font-semibold text-bench-text">WebGPU Bench</span>
          </a>
          <a href="/" className="text-sm text-bench-accent hover:underline">Run benchmarks &rarr;</a>
        </nav>

        {/* Hero */}
        <header className="text-center pt-16 pb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-bench-accent/5 border border-bench-accent/10 text-bench-accent text-xs font-medium mb-8">
            Research preprint
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight leading-[1.1] mb-6">
            Your browser just became a{" "}
            <span className="bg-gradient-to-r from-bench-accent to-purple-400 bg-clip-text text-transparent">supercomputer</span>
          </h1>
          <p className="text-lg text-bench-muted max-w-lg mx-auto">
            We proved that browsers can run GPU computation at near-native speed.
            No download. No account. No expensive hardware. Just open a page.
          </p>
        </header>

        {/* The Problem */}
        <section className="py-16 border-t border-bench-border/50">
          <h2 className="text-2xl font-bold mb-3">GPU computing has a gate</h2>
          <p className="text-bench-muted mb-8 max-w-lg">
            For decades, running computation on a graphics card meant navigating three barriers
            that kept most of the world locked out.
          </p>
          <div className="grid sm:grid-cols-3 gap-4">
            {barriers.map((b) => (
              <div key={b.title} className="card">
                <div className="text-2xl mb-3">{b.icon}</div>
                <h3 className="font-semibold mb-2">{b.title}</h3>
                <p className="text-sm text-bench-muted leading-relaxed">{b.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Stats */}
        <section className="py-16 border-t border-bench-border/50">
          <h2 className="text-2xl font-bold mb-3">We proved there&apos;s another way</h2>
          <p className="text-bench-muted mb-8 max-w-lg">
            WebGPU is a new browser standard that gives web pages direct access to your graphics card.
            We showed it&apos;s fast enough for real scientific computation.
          </p>
          <div className="grid grid-cols-3 gap-4">
            {[
              { number: "2,865\u00D7", label: "Apple Silicon average\n592 real-world devices" },
              { number: "623\u00D7", label: "Android phones average\nQualcomm Adreno" },
              { number: "0", label: "things to install\njust open Chrome" },
            ].map((s) => (
              <div key={s.number} className="card text-center py-8">
                <div className="text-3xl md:text-4xl font-extrabold text-bench-accent mb-2">{s.number}</div>
                <div className="text-xs text-bench-muted whitespace-pre-line">{s.label}</div>
              </div>
            ))}
          </div>

          <div className="mt-8 border-l-2 border-bench-accent/30 pl-6 py-4">
            <p className="text-bench-muted italic leading-relaxed">
              &ldquo;For fitness functions with sequential dependencies &mdash; common in reinforcement learning,
              financial simulation, and control systems &mdash; custom compute shaders dramatically
              outperform framework-based GPU code.&rdquo;
            </p>
            <p className="text-bench-accent text-sm mt-3">From the preprint, Section 4.2</p>
          </div>
        </section>

        {/* Who it's for */}
        <section className="py-16 border-t border-bench-border/50">
          <h2 className="text-2xl font-bold mb-3">Who this is for</h2>
          <p className="text-bench-muted mb-8 max-w-lg">
            This isn&apos;t about replacing data centers. It&apos;s about giving GPU access to
            the people who never had it.
          </p>
          <div className="space-y-5">
            {personas.map((p) => (
              <div key={p.title} className="flex gap-4 items-start">
                <div className="w-12 h-12 rounded-xl bg-bench-surface border border-bench-border flex items-center justify-center text-xl flex-shrink-0">
                  {p.icon}
                </div>
                <div>
                  <h3 className="font-semibold mb-1">{p.title}</h3>
                  <p className="text-sm text-bench-muted leading-relaxed">{p.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* How it works */}
        <section className="py-16 border-t border-bench-border/50">
          <h2 className="text-2xl font-bold mb-3">How it works (simply)</h2>
          <p className="text-bench-muted mb-8 max-w-lg">
            The technical details are in the preprint. Here&apos;s the intuition.
          </p>
          <div className="space-y-8">
            {steps.map((s, i) => (
              <div key={s.title} className="flex gap-5 items-start">
                <div className="w-10 h-10 rounded-lg bg-bench-accent/10 border border-bench-accent/20 flex items-center justify-center text-bench-accent font-bold text-sm flex-shrink-0">
                  {i + 1}
                </div>
                <div>
                  <h3 className="font-semibold mb-1">{s.title}</h3>
                  <p className="text-sm text-bench-muted leading-relaxed">{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Before / After */}
        <section className="py-16 border-t border-bench-border/50">
          <h2 className="text-2xl font-bold mb-3">What actually changes</h2>
          <p className="text-bench-muted mb-8 max-w-lg">
            This isn&apos;t theoretical. Here&apos;s what&apos;s different tomorrow.
          </p>
          <div className="space-y-4">
            {beforeAfter.map((ba, i) => (
              <div key={i} className="card grid sm:grid-cols-[1fr_auto_1fr] gap-4 items-start">
                <div>
                  <div className="text-[10px] uppercase tracking-widest text-bench-muted/50 mb-2">Before</div>
                  <p className="text-sm text-bench-muted leading-relaxed">{ba.before}</p>
                </div>
                <div className="text-bench-accent text-xl pt-4 hidden sm:block">&rarr;</div>
                <div>
                  <div className="text-[10px] uppercase tracking-widest text-bench-accent mb-2">After</div>
                  <p className="text-sm text-bench-text leading-relaxed font-medium">{ba.after}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* The key finding */}
        <section className="py-16 border-t border-bench-border/50">
          <h2 className="text-2xl font-bold mb-6">The result that surprised us</h2>
          <div className="border-l-2 border-bench-accent/30 pl-6 py-2 mb-6">
            <p className="text-lg text-bench-muted italic leading-relaxed">
              The paper measured 159&ndash;720&times; on two machines.
              Then 592 people ran it on their own devices, and the numbers got bigger.
            </p>
          </div>
          <p className="text-sm text-bench-muted leading-relaxed mb-4">
            In the paper, we tested across four GPU APIs on two hardware platforms. On a Tesla T4:
            hand-fused CUDA 720&times;, JAX lax.scan 172&times;, Triton 27&times;. On an M2 Pro:
            WebGPU 159&times; over PyTorch MPS. The pattern was consistent: fusion eliminates dispatch overhead.
          </p>
          <p className="text-sm text-bench-muted leading-relaxed">
            Since publishing, 592 devices have confirmed this &mdash; and the real-world numbers are larger. Apple Silicon
            averages 2,865&times;. Qualcomm Adreno (the chip in most Android phones) averages 623&times;. NVIDIA desktops
            average 79&times;.
          </p>
        </section>

        {/* Why are the real-world numbers bigger? */}
        <section className="py-16 border-t border-bench-border/50">
          <h2 className="text-2xl font-bold mb-6">Why are the real-world numbers bigger?</h2>
          <div className="space-y-4 text-sm text-bench-muted leading-relaxed">
            <p>
              The papers measured on 2 machines: an Apple M2 Pro laptop and a Tesla T4 server. Both are fast GPUs
              with efficient command dispatching. That&apos;s why the speedups were &ldquo;only&rdquo; 159&ndash;720&times;.
            </p>
            <p>
              Real-world devices include phones, tablets, Chromebooks, and laptops with integrated GPUs &mdash; hardware
              that was never designed for compute workloads. These GPUs have much worse dispatch overhead.
            </p>
            <p>
              Kernel fusion eliminates dispatch overhead. So the worse a device is at dispatching, the more it benefits.
              NVIDIA desktop GPUs (good dispatching) see ~79&times;. Apple Silicon laptops see ~2,865&times;. Android phones
              see ~623&times;. <strong className="text-bench-text">The devices that need fusion most, benefit from it most.</strong>
            </p>
          </div>
        </section>

        {/* CTA */}
        <section className="py-20 text-center">
          <h2 className="text-2xl font-bold mb-4">See it for yourself</h2>
          <p className="text-bench-muted mb-8">
            Run real GPU benchmarks on your hardware, right now, in your browser.
          </p>
          <div className="flex flex-wrap gap-3 justify-center">
            <a href="/" className="btn-primary">Run the Benchmark</a>
            <a href="/results" className="btn-secondary">All Results (Open Data)</a>
            <a href={LINKS.ecDoi} className="btn-secondary">Read the Preprint</a>
          </div>
          <p className="text-xs text-bench-muted/50 mt-4">
            Every result from every device is public. No cherry-picking. Verify any claim yourself.
          </p>
        </section>

      </div>

      {/* Footer */}
      <footer className="border-t border-bench-border/50">
        <div className="max-w-3xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-1.5 text-sm">
              <svg className="w-4 h-4 text-bench-accent" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2.5" strokeDasharray="60 28" strokeLinecap="round"/>
                <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
              </svg>
              <span className="font-semibold">WebGPU Bench</span>
            </div>
            <div className="flex items-center gap-5 text-sm text-bench-muted">
              <a href={LINKS.repo} className="hover:text-bench-text transition">GitHub</a>
              <a href="/privacy" className="hover:text-bench-text transition">Privacy</a>
              <span>MIT License</span>
            </div>
          </div>
          <p className="text-xs text-bench-muted/50 text-center mt-6">
            Built by Ahmet Baris Gunaydin &middot; All computation runs locally on your GPU
          </p>
        </div>
      </footer>
    </div>
  );
}
