"use client";

import { useState, useEffect, useCallback } from "react";
import { BenchmarkCard } from "@/components/benchmark-card";
import { GpuInfoCard } from "@/components/gpu-info-card";
import { ResultsSummary } from "@/components/results-summary";
import { RecentRuns } from "@/components/recent-runs";
import type { GpuInfo } from "@/lib/gpu-detect";
import type { BenchmarkResult } from "@/lib/benchmark-runner";
import { LINKS } from "@/lib/constants";
import { PaperCard } from "@/components/paper-card";

type BenchStatus = "idle" | "warmup" | "running" | "done";

interface BenchState {
  status: BenchStatus;
  progress: number;
  result?: BenchmarkResult;
}

const BENCHMARKS = [
  {
    key: "rastrigin",
    name: "Rastrigin",
    description: "Standard optimization benchmark, embarrassingly parallel (POP=4096, DIM=2000)",
    icon: "\u{1F4CA}",
    populationSize: 4096,
    dimensions: 2000,
    warmupIterations: 5,
    benchmarkIterations: 50,
  },
  {
    key: "nbody",
    name: "N-Body Simulation",
    description: "Gravitational physics, 512 bodies, 200 timesteps fused (SEQUENTIAL)",
    icon: "\u{1F30C}",
    populationSize: 512,
    dimensions: 4,
    warmupIterations: 3,
    benchmarkIterations: 30,
  },
  {
    key: "acrobot",
    name: "Acrobot-v1",
    description: "Standard Gym RL, double pendulum, 500 steps with RK4 physics (SEQUENTIAL)",
    icon: "\u{1F3AF}",
    populationSize: 4096,
    dimensions: 163,
    warmupIterations: 3,
    benchmarkIterations: 30,
  },
  {
    key: "mountaincar",
    name: "MountainCar-v0",
    description: "Standard Gym RL, 200 timesteps, linear policy (SEQUENTIAL)",
    icon: "\u{26F0}\uFE0F",
    populationSize: 4096,
    dimensions: 9,
    warmupIterations: 5,
    benchmarkIterations: 50,
  },
  {
    key: "montecarlo",
    name: "Monte Carlo Pi",
    description: "Classic parallel estimation, 100K samples per worker (PARALLEL)",
    icon: "\u{1F3B2}",
    populationSize: 4096,
    dimensions: 1,
    warmupIterations: 5,
    benchmarkIterations: 50,
  },
] as const;

export default function HomePage() {
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [gpuLoading, setGpuLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [benchStates, setBenchStates] = useState<Record<string, BenchState>>({
    rastrigin: { status: "idle", progress: 0 },
    nbody: { status: "idle", progress: 0 },
    acrobot: { status: "idle", progress: 0 },
    mountaincar: { status: "idle", progress: 0 },
    montecarlo: { status: "idle", progress: 0 },
  });

  useEffect(() => {
    const detect = async () => {
      try {
        const { detectGPU } = await import("@/lib/gpu-detect");
        const info = await detectGPU();
        setGpuInfo(info);
      } catch {
        setGpuInfo({ supported: false, adapterName: "", vendor: "", architecture: "", maxBufferSize: 0, maxComputeWorkgroupSize: [0, 0, 0], maxComputeInvocationsPerWorkgroup: 0, features: [] });
      } finally {
        setGpuLoading(false);
      }
    };
    detect();
  }, []);

  const runBenchmarks = useCallback(async () => {
    if (running || !gpuInfo?.supported) return;
    setRunning(true);
    setSubmitted(false);

    setBenchStates({
      rastrigin: { status: "idle", progress: 0 },
      nbody: { status: "idle", progress: 0 },
      acrobot: { status: "idle", progress: 0 },
      mountaincar: { status: "idle", progress: 0 },
      montecarlo: { status: "idle", progress: 0 },
    });

    try {
      const { BenchmarkRunner } = await import("@/lib/benchmark-runner");
      const { RASTRIGIN_SHADER, NBODY_SHADER, ACROBOT_SHADER, MOUNTAINCAR_SHADER, MONTECARLO_SHADER } = await import("@/lib/shaders");

      const runner = new BenchmarkRunner();
      await runner.init();

      const shaders: Record<string, string> = {
        rastrigin: RASTRIGIN_SHADER,
        nbody: NBODY_SHADER,
        acrobot: ACROBOT_SHADER,
        mountaincar: MOUNTAINCAR_SHADER,
        montecarlo: MONTECARLO_SHADER,
      };

      for (const bench of BENCHMARKS) {
        const shader = shaders[bench.key];
        if (!shader) continue;

        setBenchStates((prev) => ({
          ...prev,
          [bench.key]: { status: "warmup", progress: 0 },
        }));

        await new Promise((r) => setTimeout(r, 100));

        setBenchStates((prev) => ({
          ...prev,
          [bench.key]: { status: "running", progress: 0 },
        }));

        const result = await runner.run(
          {
            name: bench.name,
            shader,
            populationSize: bench.populationSize,
            dimensions: bench.dimensions,
            warmupIterations: bench.warmupIterations,
            benchmarkIterations: bench.benchmarkIterations,
          },
          (pct) => {
            setBenchStates((prev) => ({
              ...prev,
              [bench.key]: { ...prev[bench.key]!, status: "running", progress: pct },
            }));
          }
        );

        setBenchStates((prev) => ({
          ...prev,
          [bench.key]: { status: "done", progress: 100, result },
        }));
      }

      runner.destroy();
    } catch (err) {
      console.error("Benchmark failed:", err);
    } finally {
      setRunning(false);
    }
  }, [running, gpuInfo]);

  const allDone = Object.values(benchStates).every((s) => s.status === "done");
  const completedResults = Object.values(benchStates)
    .filter((s): s is BenchState & { result: BenchmarkResult } => s.status === "done" && !!s.result)
    .map((s) => s.result);

  // Auto-submit results when all benchmarks complete
  useEffect(() => {
    if (!allDone || submitted || completedResults.length === 0 || !gpuInfo?.supported) return;
    setSubmitted(true);

    const find = (name: string) => completedResults.find((r) => r.name === name);
    const rastrigin = find("Rastrigin");
    const nbody = find("N-Body Simulation");
    const acrobot = find("Acrobot-v1");
    const mountaincar = find("MountainCar-v0");
    const montecarlo = find("Monte Carlo Pi");

    const geoMean = Math.round(
      Math.pow(
        completedResults.reduce((acc, r) => acc * Math.max(r.throughput, 1), 1),
        1 / completedResults.length
      )
    );

    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

    fetch("/api/results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gpuName: gpuInfo.adapterName,
        gpuVendor: gpuInfo.vendor,
        gpuArch: gpuInfo.architecture,
        maxBuffer: gpuInfo.maxBufferSize,
        features: gpuInfo.features.length,
        maxWorkgroupX: gpuInfo.maxComputeWorkgroupSize[0],
        maxWorkgroupY: gpuInfo.maxComputeWorkgroupSize[1],
        maxWorkgroupZ: gpuInfo.maxComputeWorkgroupSize[2],
        maxInvocations: gpuInfo.maxComputeInvocationsPerWorkgroup,
        backend: gpuInfo.architecture || "",
        devicePixelRatio: window.devicePixelRatio || 1,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
        isMobile,
        // Legacy columns (backward compat)
        parallel: rastrigin?.throughput ?? null,
        sequential: nbody?.throughput ?? null,
        matrix: montecarlo?.throughput ?? null,
        score: geoMean,
        // New per-benchmark columns
        rastrigin: rastrigin?.throughput ?? null,
        nbody: nbody?.throughput ?? null,
        acrobot: acrobot?.throughput ?? null,
        mountaincar: mountaincar?.throughput ?? null,
        montecarlo: montecarlo?.throughput ?? null,
        // Timing stats
        rastriginMean: rastrigin?.meanTime ?? null,
        rastriginMin: rastrigin?.minTime ?? null,
        rastriginMax: rastrigin?.maxTime ?? null,
        rastriginStd: rastrigin?.stdDev ?? null,
        nbodyMean: nbody?.meanTime ?? null,
        nbodyMin: nbody?.minTime ?? null,
        nbodyMax: nbody?.maxTime ?? null,
        nbodyStd: nbody?.stdDev ?? null,
        acrobotMean: acrobot?.meanTime ?? null,
        acrobotMin: acrobot?.minTime ?? null,
        acrobotMax: acrobot?.maxTime ?? null,
        acrobotStd: acrobot?.stdDev ?? null,
        mountaincarMean: mountaincar?.meanTime ?? null,
        mountaincarMin: mountaincar?.minTime ?? null,
        mountaincarMax: mountaincar?.maxTime ?? null,
        mountaincarStd: mountaincar?.stdDev ?? null,
        montecarloMean: montecarlo?.meanTime ?? null,
        montecarloMin: montecarlo?.minTime ?? null,
        montecarloMax: montecarlo?.maxTime ?? null,
        montecarloStd: montecarlo?.stdDev ?? null,
      }),
    }).catch(() => { /* silent fail */ });
  }, [allDone, submitted, completedResults, gpuInfo]);

  return (
    <div className="min-h-screen">
      {/* Subtle gradient background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.05)_0%,_transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,_rgba(168,85,247,0.04)_0%,_transparent_50%)]" />
      </div>

      {/* Header */}
      <header className="max-w-3xl mx-auto px-6 pt-12 pb-6 text-center">
        <div className="flex items-center justify-center gap-2 mb-6">
          <svg className="w-5 h-5 text-bench-accent" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2.5" strokeDasharray="60 28" strokeLinecap="round"/>
            <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
          </svg>
          <span className="font-bold text-bench-text">WebGPU Bench</span>
        </div>

        {/* Tab Switcher */}
        <div className="inline-flex rounded-lg bg-bench-surface border border-bench-border p-1 mb-8">
          <span className="px-4 py-2 rounded-md text-sm font-medium bg-bench-accent/10 text-bench-accent">
            GPU Compute
          </span>
          <a href="/transformer" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            Transformer Fusion
          </a>
          <a href="/swarm" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            Distributed P2P
          </a>
          <a href="/demos/zerotvm-chat.html" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            Zero-TVM
          </a>
          <a href="/results" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            All Results
          </a>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-[1.1] mb-4">
          How fast is your GPU
          <br />
          <span className="bg-gradient-to-r from-bench-accent to-purple-400 bg-clip-text text-transparent">
            in the browser?
          </span>
        </h1>
        <p className="text-lg text-bench-muted max-w-md mx-auto">
          Real WebGPU compute benchmarks. No install, no account.
          Just click Run.
        </p>
      </header>

      <main className="max-w-3xl mx-auto px-6 pb-24 space-y-5">
        {/* GPU Info */}
        <GpuInfoCard info={gpuInfo} loading={gpuLoading} />

        {/* Divider */}
        <div className="flex items-center gap-4 py-2">
          <div className="flex-1 h-px bg-bench-border" />
          <span className="text-xs text-bench-muted font-medium uppercase tracking-widest">Benchmarks</span>
          <div className="flex-1 h-px bg-bench-border" />
        </div>

        {/* Benchmark Cards */}
        <div className="space-y-3">
          {BENCHMARKS.map((bench) => {
            const state = benchStates[bench.key]!;
            return (
              <BenchmarkCard
                key={bench.key}
                name={bench.name}
                description={bench.description}
                icon={bench.icon}
                status={state.status}
                progress={state.progress}
                result={state.result}
              />
            );
          })}
        </div>

        {/* Run Button + Consent */}
        <div className="space-y-3 pt-2">
          <button
            className="btn-primary w-full text-base py-4 relative overflow-hidden group"
            disabled={!gpuInfo?.supported || running || gpuLoading}
            onClick={runBenchmarks}
          >
            {running && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-0 animate-[shimmer_2s_infinite]" />
            )}
            {running ? (
              <>
                <svg className="w-4 h-4 animate-spin-slow" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="2" opacity="0.3"/>
                  <path d="M14.5 8a6.5 6.5 0 00-6.5-6.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                Running Benchmarks...
              </>
            ) : allDone ? (
              <>
                <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none"><path d="M2 8l4 4 8-8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                Run Again
              </>
            ) : (
              <>
                <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none"><path d="M5 3l8 5-8 5V3z" fill="currentColor"/></svg>
                Run All Benchmarks
              </>
            )}
          </button>
          <p className="text-[11px] text-bench-muted/60 text-center leading-relaxed">
            By clicking Run, your GPU model and benchmark results are saved anonymously.
            No personal information is collected.{" "}
            <a href="/privacy" className="text-bench-accent/60 hover:text-bench-accent transition">
              Privacy policy
            </a>
          </p>
        </div>

        {/* Results Summary */}
        {allDone && (
          <div className="pt-4">
            <ResultsSummary results={completedResults} gpuName={gpuInfo?.adapterName} />
          </div>
        )}

        {/* Recent Runs Feed */}
        <RecentRuns className="pt-4" />

        <PaperCard
          title="Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders"
          description="487 real-world devices: 4,081× avg on Apple Silicon, 826× on phones. Originally measured at 159× (WebGPU, M2 Pro) and 720× (CUDA, T4) in the paper."
          doi={LINKS.ecDoi}
          doiLabel={LINKS.ecDoiShort}
        />

        {/* Open data */}
        <div className="card border-bench-accent/10 mt-4">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-bench-accent flex-shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>
            <div className="flex-1">
              <h3 className="font-semibold text-bench-text text-sm mb-1">Every result is public</h3>
              <p className="text-xs text-bench-muted leading-relaxed mb-2">
                We don&apos;t cherry-pick. Every benchmark run from every device is published &mdash; GPU name, score, browser,
                OS, timestamp. No data is hidden. Verify any claim yourself.
              </p>
              <a href="/results" className="text-xs font-medium text-bench-accent hover:underline">
                Browse all results &rarr;
              </a>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-bench-border/50">
        <div className="max-w-3xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-1.5 text-sm">
              <svg className="w-4 h-4 text-bench-accent" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2.5" strokeDasharray="60 28" strokeLinecap="round"/>
                <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
              </svg>
              <span className="font-semibold text-bench-text">WebGPU Bench</span>
            </div>
            <div className="flex items-center gap-5 text-sm text-bench-muted">
              <a href="/results" className="hover:text-bench-text transition">All Results</a>
              <a href="/why" className="hover:text-bench-text transition">Why this matters</a>
              <a href={LINKS.research} className="hover:text-bench-text transition">Research</a>
              <a href={LINKS.repo} className="hover:text-bench-text transition">GitHub</a>
              <a href="/privacy" className="hover:text-bench-text transition">Privacy</a>
              <span>MIT License</span>
            </div>
          </div>
          <p className="text-xs text-bench-muted/50 text-center mt-6">
            All computation runs locally on your GPU. Only anonymous hardware stats are collected.
          </p>
        </div>
      </footer>
    </div>
  );
}
