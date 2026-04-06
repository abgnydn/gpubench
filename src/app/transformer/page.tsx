"use client";

import { useState, useCallback, useEffect } from "react";
import { GpuInfoCard } from "@/components/gpu-info-card";
import type { GpuInfo } from "@/lib/gpu-detect";
import { LINKS } from "@/lib/constants";
import { PaperCard } from "@/components/paper-card";

interface BenchResult {
  label: string;
  D?: number;
  layers: number;
  dispatches: number;
  unfused: { mean_ms: number; std_ms: number; tokens_per_sec: number };
  fused: { mean_ms: number; std_ms: number; tokens_per_sec: number };
  parallel?: { mean_ms: number; std_ms: number; tokens_per_sec: number };
  speedup: number;
  parSpeedup?: number;
  error?: string;
}

type Status = "idle" | "running" | "done";

export default function TransformerPage() {
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [gpuLoading, setGpuLoading] = useState(true);
  const [status, setStatus] = useState<Status>("idle");
  const [log, setLog] = useState<string[]>([]);
  const [results, setResults] = useState<BenchResult[]>([]);
  const [gpuName, setGpuName] = useState("");
  const [currentConfig, setCurrentConfig] = useState("");

  const ALL_CONFIGS = [
    { key: "d32l1", label: "D=32, L=1", D: 32, layers: 1, default: true },
    { key: "d32l4", label: "D=32, L=4", D: 32, layers: 4, default: true },
    { key: "d64l1", label: "D=64, L=1", D: 64, layers: 1, default: true },
    { key: "d64l4", label: "D=64, L=4", D: 64, layers: 4, default: true },
    { key: "d128l1", label: "D=128, L=1", D: 128, layers: 1, default: false },
    { key: "d128l4", label: "D=128, L=4", D: 128, layers: 4, default: false },
  ];

  const [selectedConfigs, setSelectedConfigs] = useState<Set<string>>(
    new Set(ALL_CONFIGS.filter((c) => c.default).map((c) => c.key))
  );

  const toggleConfig = (key: string) => {
    setSelectedConfigs((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

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

  const addLog = useCallback((msg: string) => {
    setLog((prev) => [...prev, msg]);
  }, []);

  const runBenchmark = useCallback(async () => {
    if (status === "running" || !gpuInfo?.supported) return;
    setStatus("running");
    setLog([]);
    setResults([]);

    try {
      const mod = await import("@/lib/transformer-bench");

      // Build filtered config list from selected checkboxes
      const selectedLabels = new Set(
        ALL_CONFIGS.filter((c) => selectedConfigs.has(c.key)).map((c) => c.label)
      );
      const filteredConfigs = mod.CONFIGS.filter((c: { label: string }) => selectedLabels.has(c.label));

      const result = await mod.runSweepWithConfigs(filteredConfigs,
        (msg: string) => {
          addLog(msg);
          if (msg.startsWith("---")) setCurrentConfig(msg.replace(/---/g, "").trim());
        },
        (row: BenchResult) => {
          setResults((prev) => [...prev, row]);
          if (!row.error && row.unfused) {
            fetch("/api/transformer-results", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                gpuName: gpuInfo?.adapterName || "Unknown",
                gpuVendor: gpuInfo?.vendor || "",
                gpuArch: gpuInfo?.architecture || "",
                config: row.label,
                layers: row.layers,
                dModel: row.D || 0,
                dispatches: row.dispatches,
                unfusedMs: row.unfused.mean_ms,
                fused1tMs: row.fused?.mean_ms ?? null,
                parallelMs: row.parallel?.mean_ms ?? null,
                speedup1t: row.speedup,
                speedupParallel: row.parSpeedup ?? null,
                tokensPerSec: row.parallel?.tokens_per_sec ?? row.fused?.tokens_per_sec ?? null,
                screenWidth: window.screen.width,
                screenHeight: window.screen.height,
                isMobile: /Android|iPhone|iPad|iPod/i.test(navigator.userAgent),
              }),
            }).catch(() => {});
          }
        }
      );
      setGpuName(result.gpuName || "Unknown");
    } catch (err) {
      addLog(`ERROR: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setStatus("done");
    }
  }, [status, gpuInfo, addLog, selectedConfigs]);

  return (
    <div className="min-h-screen">
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.05)_0%,_transparent_50%)]" />
      </div>

      {/* Header */}
      <header className="max-w-3xl mx-auto px-6 pt-12 pb-4 text-center">
        <div className="flex items-center justify-center gap-2 mb-6">
          <svg className="w-5 h-5 text-bench-accent" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2.5" strokeDasharray="60 28" strokeLinecap="round"/>
            <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
          </svg>
          <span className="font-bold text-bench-text">WebGPU Bench</span>
        </div>

        {/* Tab Switcher */}
        <div className="inline-flex rounded-lg bg-bench-surface border border-bench-border p-1 mb-8">
          <a href="/" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            GPU Compute
          </a>
          <span className="px-4 py-2 rounded-md text-sm font-medium bg-bench-accent/10 text-bench-accent">
            Transformer Fusion
          </span>
          <a href="/swarm" className="px-4 py-2 rounded-md text-sm font-medium text-bench-muted hover:text-bench-text transition">
            Distributed P2P
          </a>
        </div>

        <h1 className="text-3xl md:text-4xl font-bold tracking-tight leading-[1.1] mb-3">
          Transformer Fusion{" "}
          <span className="bg-gradient-to-r from-bench-accent to-purple-400 bg-clip-text text-transparent">
            Benchmark
          </span>
        </h1>
        <p className="text-bench-muted max-w-xl mx-auto mb-2">
          Fused vs unfused autoregressive decoding. Single-threaded and parallel (64 threads).
          6 configs + sequence scaling.
        </p>
      </header>

      <main className="max-w-3xl mx-auto px-6 pb-20 space-y-5">
        <GpuInfoCard info={gpuInfo} loading={gpuLoading} />

        {gpuName && (
          <div className="card flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-bench-green" />
            <span className="text-sm font-medium">{gpuName}</span>
          </div>
        )}

        {/* Config Selector */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold">Configurations</h3>
            <div className="flex gap-2">
              <button
                className="text-[10px] text-bench-accent hover:underline"
                onClick={() => setSelectedConfigs(new Set(ALL_CONFIGS.map((c) => c.key)))}
              >Select all</button>
              <button
                className="text-[10px] text-bench-muted hover:underline"
                onClick={() => setSelectedConfigs(new Set(ALL_CONFIGS.filter((c) => c.default).map((c) => c.key)))}
              >Reset</button>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {ALL_CONFIGS.map((c) => (
              <label
                key={c.key}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition text-sm ${
                  selectedConfigs.has(c.key)
                    ? "border-bench-accent/40 bg-bench-accent/5 text-bench-text"
                    : "border-bench-border/50 text-bench-muted hover:border-bench-border"
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedConfigs.has(c.key)}
                  onChange={() => toggleConfig(c.key)}
                  className="accent-[#22d3ee] w-3.5 h-3.5"
                />
                {c.label}
                {!c.default && <span className="text-[9px] text-bench-yellow ml-auto">slow</span>}
              </label>
            ))}
          </div>
        </div>

        {/* Run Button */}
        <div className="space-y-3 pt-2">
          <button
            className="btn-primary w-full text-base py-4"
            disabled={!gpuInfo?.supported || status === "running" || gpuLoading}
            onClick={runBenchmark}
          >
            {status === "running" ? (
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4 animate-spin-slow" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="2" opacity="0.3"/>
                  <path d="M14.5 8a6.5 6.5 0 00-6.5-6.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                {currentConfig ? `Running: ${currentConfig}` : "Initializing..."} ({results.length}/{selectedConfigs.size})
              </span>
            ) : status === "done" ? `Run Again (${results.length} results)` : "Run Full Sweep"}
          </button>
          <p className="text-[11px] text-bench-muted/60 text-center">
            By clicking Run, anonymous GPU stats and results are saved.{" "}
            <a href="/privacy" className="text-bench-accent/60 hover:text-bench-accent transition">Privacy policy</a>
          </p>
        </div>

        {/* Results Table */}
        {results.length > 0 && (
          <div className="card overflow-x-auto">
            <h2 className="text-lg font-semibold mb-4">Results</h2>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-bench-border text-bench-muted text-xs">
                  <th className="text-left py-2 pr-3">Config</th>
                  <th className="text-right py-2 px-2">Dispatches</th>
                  <th className="text-right py-2 px-2">Unfused</th>
                  <th className="text-right py-2 px-2">Fused 1T</th>
                  <th className="text-right py-2 px-2">Parallel</th>
                  <th className="text-right py-2 px-2">1T</th>
                  <th className="text-right py-2 pl-2">Parallel</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-b border-bench-border/30">
                    {r.error ? (
                      <td colSpan={7} className="py-2 text-bench-red text-xs">{r.label}: {r.error}</td>
                    ) : (
                      <>
                        <td className="py-2 pr-3 font-medium text-bench-text">{r.label}</td>
                        <td className="py-2 px-2 text-right text-bench-muted">{r.dispatches}</td>
                        <td className="py-2 px-2 text-right text-bench-muted">{r.unfused.mean_ms.toFixed(0)}ms</td>
                        <td className="py-2 px-2 text-right">{r.fused.mean_ms.toFixed(1)}ms</td>
                        <td className="py-2 px-2 text-right text-bench-accent">{r.parallel ? `${r.parallel.mean_ms.toFixed(1)}ms` : "—"}</td>
                        <td className="py-2 px-2 text-right font-semibold text-bench-green">{r.speedup.toFixed(1)}×</td>
                        <td className="py-2 pl-2 text-right font-bold text-bench-accent">{r.parSpeedup ? `${r.parSpeedup.toFixed(0)}×` : "—"}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Log */}
        {log.length > 0 && (
          <div className="card">
            <h2 className="text-sm font-semibold mb-3 text-bench-muted">Live Log</h2>
            <pre className="text-xs text-bench-muted/70 font-mono max-h-64 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {log.join("\n")}
            </pre>
          </div>
        )}

        <PaperCard
          title="Single-Kernel Fusion for Autoregressive Transformer Decoding via WebGPU Compute Shaders"
          description="Fusing the entire autoregressive decoding loop into a single GPU dispatch achieves 66–458× over unfused dispatch. The parallel kernel beats PyTorch MPS by 7.5–161× at all tested sizes."
          doi={LINKS.transformerDoi}
          doiLabel={LINKS.transformerDoiShort}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-bench-border/50">
        <div className="max-w-3xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-bench-muted">
          <span className="font-semibold text-bench-text">WebGPU Bench</span>
          <div className="flex gap-5">
            <a href="/why" className="hover:text-bench-text transition">Why this matters</a>
            <a href={LINKS.research} className="hover:text-bench-text transition">Research</a>
            <a href="/privacy" className="hover:text-bench-text transition">Privacy</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
