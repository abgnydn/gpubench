"use client";

interface BenchmarkResult {
  throughput: number;
  meanTime: number;
  minTime: number;
  maxTime: number;
  stdDev: number;
}

interface BenchmarkCardProps {
  name: string;
  description: string;
  icon: string;
  status: "idle" | "warmup" | "running" | "done";
  progress: number;
  result?: BenchmarkResult;
}

const statusConfig = {
  idle: { label: "Ready", dot: "bg-bench-border", text: "text-bench-muted", bg: "bg-bench-border/30" },
  warmup: { label: "Warming up", dot: "bg-bench-yellow", text: "text-bench-yellow", bg: "bg-bench-yellow/10" },
  running: { label: "Running", dot: "bg-bench-accent", text: "text-bench-accent", bg: "bg-bench-accent/10" },
  done: { label: "Complete", dot: "bg-bench-green", text: "text-bench-green", bg: "bg-bench-green/10" },
};

export function BenchmarkCard({ name, description, icon, status, progress, result }: BenchmarkCardProps) {
  const s = statusConfig[status];
  const isActive = status === "warmup" || status === "running";

  return (
    <div className={`
      card transition-all duration-500 relative overflow-hidden
      ${isActive ? "border-bench-accent/30 shadow-[0_0_30px_rgba(34,211,238,0.06)]" : ""}
      ${status === "done" ? "border-bench-green/20" : ""}
    `}>
      {/* Subtle animated border glow when running */}
      {isActive && (
        <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-bench-accent/5 via-transparent to-bench-accent/5 animate-pulse-glow pointer-events-none" />
      )}

      <div className="relative">
        {/* Header row */}
        <div className="flex items-start justify-between mb-1">
          <div className="flex items-center gap-3">
            <div className={`
              w-10 h-10 rounded-lg flex items-center justify-center text-lg
              ${status === "done" ? "bg-bench-green/10" : isActive ? "bg-bench-accent/10" : "bg-bench-surface"}
              border border-bench-border/50 transition-colors duration-500
            `}>
              {icon}
            </div>
            <div>
              <h3 className="font-semibold text-bench-text text-[15px]">{name}</h3>
              <p className="text-xs text-bench-muted/70 mt-0.5">{description}</p>
            </div>
          </div>
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium ${s.bg} ${s.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${s.dot} ${isActive ? "animate-pulse-glow" : ""}`} />
            {s.label}
          </div>
        </div>

        {/* Progress bar */}
        {isActive && (
          <div className="mt-4 mb-1">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] text-bench-muted">Progress</span>
              <span className="text-[11px] font-mono text-bench-accent">{progress}%</span>
            </div>
            <div className="h-1 rounded-full bg-bench-border/50 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-bench-accent to-cyan-300 transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Results */}
        {status === "done" && result && (
          <div className="mt-4 pt-4 border-t border-bench-border/50">
            <div className="flex items-baseline gap-2 mb-3">
              <span className="text-3xl font-bold bg-gradient-to-r from-bench-green to-emerald-300 bg-clip-text text-transparent">
                {result.throughput.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
              <span className="text-sm text-bench-muted">gen/s</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {[
                { label: "mean", value: `${result.meanTime.toFixed(2)}ms` },
                { label: "min", value: `${result.minTime.toFixed(2)}ms` },
                { label: "max", value: `${result.maxTime.toFixed(2)}ms` },
                { label: "std", value: `${result.stdDev.toFixed(2)}ms` },
              ].map((stat) => (
                <div key={stat.label} className="bg-bench-bg/50 rounded-md px-2.5 py-2 text-center">
                  <div className="text-xs font-mono font-medium text-bench-text/80">{stat.value}</div>
                  <div className="text-[10px] text-bench-muted/60 mt-0.5">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
