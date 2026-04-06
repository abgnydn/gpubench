"use client";

import { ShareButtons } from "./share-buttons";

interface BenchmarkResult {
  name: string;
  throughput: number;
  meanTime: number;
}

interface ResultsSummaryProps {
  results: BenchmarkResult[];
  gpuName?: string;
}

function getGrade(score: number): { label: string; color: string; gradient: string; percentile: string } {
  if (score >= 800) return { label: "Exceptional", color: "text-bench-accent", gradient: "from-bench-accent to-cyan-300", percentile: "Top 5%" };
  if (score >= 500) return { label: "Excellent", color: "text-bench-green", gradient: "from-bench-green to-emerald-300", percentile: "Top 15%" };
  if (score >= 300) return { label: "Good", color: "text-bench-green", gradient: "from-bench-green to-emerald-300", percentile: "Top 40%" };
  if (score >= 150) return { label: "Average", color: "text-bench-yellow", gradient: "from-bench-yellow to-amber-300", percentile: "Top 60%" };
  if (score >= 50) return { label: "Below Average", color: "text-bench-yellow", gradient: "from-bench-yellow to-amber-300", percentile: "Top 80%" };
  return { label: "Low", color: "text-bench-red", gradient: "from-bench-red to-rose-300", percentile: "Bottom 20%" };
}

function cleanGpuName(raw: string | undefined): string {
  if (!raw) return "Unknown GPU";
  const cleaned = raw
    .replace(/ANGLE \(.*?\)/g, "")
    .replace(/Direct3D11.*$/g, "")
    .replace(/\(0x[0-9A-Fa-f]+\)/g, "")
    .replace(/\s*vs_\d+_\d+\s*ps_\d+_\d+/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
  return cleaned || "Unknown GPU";
}

function formatShareText(
  gpuName: string,
  geoMean: number,
  grade: { label: string; percentile: string },
  results: BenchmarkResult[],
): string {
  const gpu = cleanGpuName(gpuName);
  // Highest-throughput benchmark as the single "flex" number people can read
  const peak = results.reduce<BenchmarkResult | null>(
    (best, r) => (!best || r.throughput > best.throughput ? r : best),
    null,
  );
  const peakLine = peak
    ? `Fastest test: ${peak.throughput.toLocaleString()} runs/sec on ${peak.name}`
    : "";

  return [
    `I just tested my GPU speed in a browser tab. No install, no CUDA.`,
    "",
    `Device:  ${gpu}`,
    `Score:   ${geoMean.toLocaleString()} — ${grade.label} (${grade.percentile} of tested devices)`,
    peakLine,
    "",
    `It runs real compute (physics sims, optimization, Monte Carlo) on your GPU through the browser. Test yours:`,
    `https://gpubench.dev`,
  ]
    .filter((l) => l !== "")
    .join("\n")
    // restore intentional blank lines (markers)
    .replace(/(Device:.*$)/m, "\n$1")
    .replace(/(It runs real compute.*$)/m, "\n$1");
}

export function ResultsSummary({ results, gpuName }: ResultsSummaryProps) {
  if (results.length === 0) return null;

  const geoMean = Math.round(
    Math.pow(
      results.reduce((acc, r) => acc * Math.max(r.throughput, 1), 1),
      1 / results.length
    )
  );

  const grade = getGrade(geoMean);

  const shareText = formatShareText(gpuName ?? "", geoMean, grade, results);

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

        <div className="flex justify-center">
          <ShareButtons
            text={shareText}
            url="https://gpubench.dev"
            title={`WebGPU Bench Score: ${geoMean.toLocaleString()}`}
          />
        </div>

        <p className="text-[10px] text-bench-muted/40 mt-6">
          Powered by WebGPU Bench &mdash; all computation ran locally on your GPU
        </p>
      </div>
    </div>
  );
}
