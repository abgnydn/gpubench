"use client";

import { useState } from "react";

interface BenchmarkResult {
  name: string;
  throughput: number;
  meanTime: number;
}

interface ResultsSummaryProps {
  results: BenchmarkResult[];
}

function getGrade(score: number): { label: string; color: string; gradient: string; percentile: string } {
  if (score >= 800) return { label: "Exceptional", color: "text-bench-accent", gradient: "from-bench-accent to-cyan-300", percentile: "Top 5%" };
  if (score >= 500) return { label: "Excellent", color: "text-bench-green", gradient: "from-bench-green to-emerald-300", percentile: "Top 15%" };
  if (score >= 300) return { label: "Good", color: "text-bench-green", gradient: "from-bench-green to-emerald-300", percentile: "Top 40%" };
  if (score >= 150) return { label: "Average", color: "text-bench-yellow", gradient: "from-bench-yellow to-amber-300", percentile: "Top 60%" };
  if (score >= 50) return { label: "Below Average", color: "text-bench-yellow", gradient: "from-bench-yellow to-amber-300", percentile: "Top 80%" };
  return { label: "Low", color: "text-bench-red", gradient: "from-bench-red to-rose-300", percentile: "Bottom 20%" };
}

export function ResultsSummary({ results }: ResultsSummaryProps) {
  const [copied, setCopied] = useState(false);

  if (results.length === 0) return null;

  const geoMean = Math.round(
    Math.pow(
      results.reduce((acc, r) => acc * Math.max(r.throughput, 1), 1),
      1 / results.length
    )
  );

  const grade = getGrade(geoMean);

  const shareText = [
    `WebGPU Bench Score: ${geoMean.toLocaleString()}`,
    "",
    ...results.map((r) => `  ${r.name}: ${r.throughput.toLocaleString()} gen/s (${r.meanTime.toFixed(2)}ms/gen)`),
    "",
    "Run yours: gpubench.dev",
  ].join("\n");

  const handleCopy = async () => {
    await navigator.clipboard.writeText(shareText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="card border-bench-accent/20 overflow-hidden relative">
      {/* Background glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-32 bg-bench-accent/5 blur-3xl rounded-full pointer-events-none" />

      <div className="relative text-center">
        <p className="text-xs text-bench-muted uppercase tracking-widest mb-3">Your GPU Score</p>

        <div className={`text-6xl font-extrabold bg-gradient-to-r ${grade.gradient} bg-clip-text text-transparent mb-2`}>
          {geoMean.toLocaleString()}
        </div>

        <div className="flex items-center justify-center gap-3 mb-8">
          <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${grade.color} bg-current/10 border border-current/20`}>
            <span className={`w-1.5 h-1.5 rounded-full bg-current`} />
            {grade.label}
          </span>
          <span className="text-xs text-bench-muted">{grade.percentile}</span>
        </div>

        <div className="grid grid-cols-3 gap-3 mb-8">
          {results.map((r) => (
            <div key={r.name} className="bg-bench-bg/50 rounded-lg p-4 border border-bench-border/30">
              <div className="text-[11px] text-bench-muted/60 mb-2 truncate">{r.name}</div>
              <div className="text-xl font-bold text-bench-text">
                {r.throughput.toLocaleString()}
              </div>
              <div className="text-[10px] text-bench-muted/50 mt-0.5">gen/s</div>
            </div>
          ))}
        </div>

        <button
          onClick={handleCopy}
          className="btn-secondary inline-flex items-center gap-2 text-sm"
        >
          {copied ? (
            <>
              <svg className="w-4 h-4 text-bench-green" viewBox="0 0 16 16" fill="none"><path d="M3 8l3 3 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
              Copied!
            </>
          ) : (
            <>
              <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.5"/><path d="M3 11V3a2 2 0 012-2h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
              Copy Results
            </>
          )}
        </button>

        <p className="text-[10px] text-bench-muted/40 mt-6">
          Powered by WebGPU Bench &mdash; all computation ran locally on your GPU
        </p>
      </div>
    </div>
  );
}
