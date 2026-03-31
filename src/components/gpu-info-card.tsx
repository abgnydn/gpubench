"use client";

interface GpuInfo {
  supported: boolean;
  adapterName: string;
  vendor: string;
  architecture: string;
  maxBufferSize: number;
  maxComputeWorkgroupSize: [number, number, number];
  maxComputeInvocationsPerWorkgroup: number;
  features: string[];
}

interface GpuInfoCardProps {
  info: GpuInfo | null;
  loading: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(1)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(0)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export function GpuInfoCard({ info, loading }: GpuInfoCardProps) {
  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-3 h-3 rounded-full bg-bench-border animate-pulse" />
          <div className="h-4 w-32 bg-bench-border rounded animate-pulse" />
        </div>
        <div className="h-7 w-56 bg-bench-border rounded animate-pulse mb-4" />
        <div className="grid grid-cols-2 gap-2.5">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-14 bg-bench-border/50 rounded-lg animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (!info || !info.supported) {
    return (
      <div className="card border-bench-red/30 bg-bench-red/[0.02]">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-lg bg-bench-red/10 flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-bench-red" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M7 7l6 6M13 7l-6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-bench-red mb-1">WebGPU Not Supported</h3>
            <p className="text-sm text-bench-muted leading-relaxed">
              Your browser doesn&apos;t support WebGPU compute shaders. Try Chrome 113+ or Edge 113+ on desktop.
              Mobile browsers have limited WebGPU support.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const displayName = info.adapterName
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

  const showSubtitle = info.vendor && !info.adapterName.toLowerCase().includes(info.vendor.toLowerCase());

  return (
    <div className="card border-bench-green/20 bg-bench-green/[0.01]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-bench-green opacity-40" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-bench-green" />
          </span>
          <span className="text-sm font-medium text-bench-green">WebGPU Ready</span>
        </div>
        <span className="text-[11px] font-medium text-bench-accent bg-bench-accent/10 px-2.5 py-1 rounded-full">
          {info.features.length} features
        </span>
      </div>

      <h3 className="text-xl font-bold text-bench-text mb-0.5">{displayName || "Unknown GPU"}</h3>
      {showSubtitle && (
        <p className="text-sm text-bench-muted mb-4">
          {info.vendor} &mdash; {info.architecture}
        </p>
      )}
      {!showSubtitle && <div className="mb-4" />}

      <div className="grid grid-cols-2 gap-2.5">
        {[
          { label: "Max Buffer", value: formatBytes(info.maxBufferSize), icon: "M4 8h12M4 4h16M4 12h8" },
          { label: "Max Workgroup", value: info.maxComputeWorkgroupSize.join(" \u00D7 "), icon: "M4 4h4v4H4zM12 4h4v4h-4zM8 8h4v4H8z" },
          { label: "Max Invocations", value: String(info.maxComputeInvocationsPerWorkgroup), icon: "M8 2v4M2 8h4M14 8h4M8 14v4" },
          { label: "Features", value: `${info.features.length} supported`, icon: "M4 8l3 3 6-6" },
        ].map((stat) => (
          <div key={stat.label} className="bg-bench-bg/80 rounded-lg p-3 border border-bench-border/30">
            <div className="flex items-center gap-1.5 mb-1.5">
              <svg className="w-3 h-3 text-bench-muted/50" viewBox="0 0 16 16" fill="none">
                <path d={stat.icon} stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span className="text-[10px] text-bench-muted/60 uppercase tracking-wider">{stat.label}</span>
            </div>
            <div className="text-sm font-medium text-bench-text/90">{stat.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
