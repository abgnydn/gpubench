"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { LINKS } from "@/lib/constants";
import { ShareButtons } from "@/components/share-buttons";
import { TabSwitcher } from "@/components/tab-switcher";

type SortDir = "asc" | "desc";
type Tab = "compute" | "transformer" | "demos";

interface ComputeRow {
  gpu_name: string;
  gpu_vendor: string;
  gpu_arch: string;
  score: number;
  rastrigin_gps: number | null;
  nbody_gps: number | null;
  acrobot_gps: number | null;
  mountaincar_gps: number | null;
  montecarlo_gps: number | null;
  browser: string;
  os: string;
  is_mobile: boolean;
  created_at: string;
}

interface TransformerRow {
  gpu_name: string;
  gpu_vendor: string;
  gpu_arch: string;
  config: string;
  layers: number;
  d_model: number;
  dispatches: number;
  unfused_ms: number | null;
  fused_1t_ms: number | null;
  parallel_ms: number | null;
  speedup_1t: number | null;
  speedup_parallel: number | null;
  tokens_per_sec: number | null;
  browser: string;
  os: string;
  is_mobile: boolean;
  created_at: string;
}

interface DeviceRow {
  device_id: string;
  device_name: string;
  gpu: string;
  workload: string;
  fitness: number | null;
  gen: number | null;
  speed: number | null;
  browser: string;
  os: string;
  is_mobile: boolean;
  created_at: string;
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function shortBrowser(raw: string): string {
  return raw?.split(" ")[0] ?? raw;
}

function fmtNum(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function SortHeader({
  label,
  field,
  currentSort,
  currentDir,
  onSort,
}: {
  label: string;
  field: string;
  currentSort: string;
  currentDir: SortDir;
  onSort: (field: string) => void;
}) {
  const active = currentSort === field;
  return (
    <th
      className="px-2 py-2 text-left text-[10px] uppercase tracking-wider text-bench-muted cursor-pointer hover:text-bench-text select-none whitespace-nowrap"
      onClick={() => onSort(field)}
    >
      {label}
      {active && (
        <span className="ml-1 text-bench-accent">
          {currentDir === "asc" ? "\u25B2" : "\u25BC"}
        </span>
      )}
    </th>
  );
}

export default function ResultsPage() {
  const [tab, setTab] = useState<Tab>("compute");
  const [computeRows, setComputeRows] = useState<ComputeRow[]>([]);
  const [transformerRows, setTransformerRows] = useState<TransformerRow[]>([]);
  const [computeTotal, setComputeTotal] = useState(0);
  const [transformerTotal, setTransformerTotal] = useState(0);
  const [computePage, setComputePage] = useState(1);
  const [transformerPage, setTransformerPage] = useState(1);
  const [computePages, setComputePages] = useState(1);
  const [transformerPages, setTransformerPages] = useState(1);
  const [demoRows, setDemoRows] = useState<DeviceRow[]>([]);
  const [demoTotal, setDemoTotal] = useState(0);
  const [demoPage, setDemoPage] = useState(1);
  const [demoPages, setDemoPages] = useState(1);
  // Breakdown by workload (reports + unique devices). Fetched from the
  // aggregate /api/device endpoint (no ?all) so it reflects the whole
  // dataset, not just the visible page. Used to surface per-demo counts
  // in the stat cards above the table — currently split out for
  // Zero-TVM because it's a meaningfully different workload (LLM decode
  // tok/s vs the evolutionary gen/s the P2P demos report).
  const [workloadStats, setWorkloadStats] = useState<
    Array<{ workload: string; reports: number; devices: number }>
  >([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState("created_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const fetchCompute = useCallback(async (page: number, sort?: string, dir?: string) => {
    try {
      const s = sort ?? sortField;
      const d = dir ?? sortDir;
      const res = await fetch(`/api/results?all=true&page=${page}&sort=${s}&dir=${d}`);
      const json = await res.json();
      setComputeRows(json.rows ?? []);
      setComputeTotal(json.total ?? 0);
      setComputePages(json.totalPages ?? 1);
    } catch { /* */ }
  }, [sortField, sortDir]);

  const fetchTransformer = useCallback(async (page: number, sort?: string, dir?: string) => {
    try {
      const s = sort ?? sortField;
      const d = dir ?? sortDir;
      const res = await fetch(`/api/transformer-results?all=true&page=${page}&sort=${s}&dir=${d}`);
      const json = await res.json();
      setTransformerRows(json.rows ?? []);
      setTransformerTotal(json.total ?? 0);
      setTransformerPages(json.totalPages ?? 1);
    } catch { /* */ }
  }, [sortField, sortDir]);

  const fetchDemos = useCallback(async (page: number, sort?: string, dir?: string) => {
    try {
      const s = sort ?? sortField;
      const d = dir ?? sortDir;
      const res = await fetch(`/api/device?all=true&page=${page}&sort=${s}&dir=${d}`);
      const json = await res.json();
      setDemoRows(json.rows ?? []);
      setDemoTotal(json.total ?? 0);
      setDemoPages(json.totalPages ?? 1);
    } catch { /* */ }
  }, [sortField, sortDir]);

  // One-shot aggregate over all workloads — independent of pagination.
  // Used only to populate the per-workload stat cards at the top.
  const fetchWorkloadStats = useCallback(async () => {
    try {
      const res = await fetch(`/api/device`);
      const json = await res.json();
      setWorkloadStats(json.workloads ?? []);
    } catch { /* */ }
  }, []);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchCompute(1),
      fetchTransformer(1),
      fetchDemos(1),
      fetchWorkloadStats(),
    ]).finally(() => setLoading(false));
  }, [fetchCompute, fetchTransformer, fetchDemos, fetchWorkloadStats]);

  const handleSort = (field: string) => {
    const newDir = sortField === field ? (sortDir === "asc" ? "desc" : "asc") : "desc";
    setSortField(field);
    setSortDir(newDir as SortDir);
    // Refetch current tab from server with new sort
    if (tab === "compute") fetchCompute(computePage, field, newDir);
    else if (tab === "transformer") fetchTransformer(transformerPage, field, newDir);
    else fetchDemos(demoPage, field, newDir);
  };


  const filteredCompute = useMemo(() => {
    const q = search.toLowerCase();
    return q
      ? computeRows.filter(
          (r) =>
            r.gpu_name?.toLowerCase().includes(q) ||
            r.gpu_vendor?.toLowerCase().includes(q) ||
            r.browser?.toLowerCase().includes(q) ||
            r.os?.toLowerCase().includes(q),
        )
      : computeRows;
  }, [computeRows, search]);

  const filteredTransformer = useMemo(() => {
    const q = search.toLowerCase();
    return q
      ? transformerRows.filter(
          (r) =>
            r.gpu_name?.toLowerCase().includes(q) ||
            r.gpu_vendor?.toLowerCase().includes(q) ||
            r.browser?.toLowerCase().includes(q) ||
            r.os?.toLowerCase().includes(q),
        )
      : transformerRows;
  }, [transformerRows, search]);

  const filteredDemos = useMemo(() => {
    const q = search.toLowerCase();
    return q
      ? demoRows.filter(
          (r) =>
            r.gpu?.toLowerCase().includes(q) ||
            r.workload?.toLowerCase().includes(q) ||
            r.device_name?.toLowerCase().includes(q) ||
            r.browser?.toLowerCase().includes(q) ||
            r.os?.toLowerCase().includes(q),
        )
      : demoRows;
  }, [demoRows, search]);

  const handleComputePage = (p: number) => {
    setComputePage(p);
    fetchCompute(p);
  };
  const handleTransformerPage = (p: number) => {
    setTransformerPage(p);
    fetchTransformer(p);
  };
  const handleDemoPage = (p: number) => {
    setDemoPage(p);
    fetchDemos(p);
  };

  const csvExport = () => {
    if (tab === "compute") {
      const header = "GPU,Vendor,Score,Rastrigin,N-Body,Acrobot,MountainCar,MonteCarlo,Browser,OS,Mobile,Timestamp";
      const rows = filteredCompute.map((r) =>
        [r.gpu_name, r.gpu_vendor, r.score, r.rastrigin_gps, r.nbody_gps, r.acrobot_gps, r.mountaincar_gps, r.montecarlo_gps, shortBrowser(r.browser), r.os, r.is_mobile, r.created_at]
          .map((v) => `"${v ?? ""}"`)
          .join(","),
      );
      const blob = new Blob([header + "\n" + rows.join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "gpubench-compute-results.csv";
      a.click();
    } else if (tab === "transformer") {
      const header = "GPU,Vendor,Config,D,Layers,Unfused_ms,Fused_ms,Parallel_ms,Speedup_1T,Speedup_Parallel,Tokens/s,Browser,OS,Mobile,Timestamp";
      const rows = filteredTransformer.map((r) =>
        [r.gpu_name, r.gpu_vendor, r.config, r.d_model, r.layers, r.unfused_ms, r.fused_1t_ms, r.parallel_ms, r.speedup_1t, r.speedup_parallel, r.tokens_per_sec, shortBrowser(r.browser), r.os, r.is_mobile, r.created_at]
          .map((v) => `"${v ?? ""}"`)
          .join(","),
      );
      const blob = new Blob([header + "\n" + rows.join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "gpubench-transformer-results.csv";
      a.click();
    } else {
      const header = "Device,GPU,Workload,Fitness,Generation,Speed_gen_s,Browser,OS,Mobile,Timestamp";
      const rows = filteredDemos.map((r) =>
        [r.device_name, r.gpu, r.workload, r.fitness, r.gen, r.speed, shortBrowser(r.browser), r.os, r.is_mobile, r.created_at]
          .map((v) => `"${v ?? ""}"`)
          .join(","),
      );
      const blob = new Blob([header + "\n" + rows.join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "gpubench-demo-sessions.csv";
      a.click();
    }
  };

  const allTotal = computeTotal + transformerTotal + demoTotal;
  const shareText = `WebGPU Bench: ${allTotal} benchmark runs from real devices. Full dataset, sortable, filterable, downloadable as CSV.`;

  return (
    <div className="min-h-screen">
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.05)_0%,_transparent_50%)]" />
      </div>

      <div className="max-w-6xl mx-auto px-6">
        {/* Nav — shared chrome across all top-level pages */}
        <header className="pt-8 pb-2 text-center">
          <div className="flex items-center justify-center gap-2 mb-6">
            <svg className="w-5 h-5 text-bench-accent animate-spin-slow" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2.5" strokeDasharray="60 28" strokeLinecap="round"/>
              <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
            </svg>
            <span className="font-bold text-bench-text">WebGPU Bench</span>
          </div>
          <TabSwitcher />
        </header>

        {/* Header */}
        <header className="pb-6">
          <h1 className="text-3xl font-extrabold tracking-tight mb-2">All Results</h1>
          <p className="text-bench-muted text-sm">
            Every benchmark run from every device. {allTotal > 0
              ? `${allTotal.toLocaleString()} runs total.`
              : "Loading..."}
          </p>
        </header>

        {/* Aggregate stats */}
        {!loading && (() => {
          // Split demo runs into Zero-TVM vs the rest (P2P evolution demos).
          // Server-side aggregate so it's accurate over the whole table,
          // not just the visible page.
          const zerotvmRuns = workloadStats
            .filter((w) => w.workload.toLowerCase().includes("zero-tvm"))
            .reduce((acc, w) => acc + (Number(w.reports) || 0), 0);
          const p2pRuns = Math.max(0, demoTotal - zerotvmRuns);
          return (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
              <div className="card text-center py-4">
                <div className="text-2xl font-extrabold text-bench-accent">{allTotal.toLocaleString()}</div>
                <div className="text-[10px] text-bench-muted mt-1">Total runs</div>
              </div>
              <div className="card text-center py-4">
                <div className="text-2xl font-extrabold text-bench-accent">{computeTotal.toLocaleString()}</div>
                <div className="text-[10px] text-bench-muted mt-1">GPU Compute</div>
              </div>
              <div className="card text-center py-4">
                <div className="text-2xl font-extrabold text-bench-accent">{transformerTotal.toLocaleString()}</div>
                <div className="text-[10px] text-bench-muted mt-1">Transformer Fusion</div>
              </div>
              <div className="card text-center py-4">
                <div className="text-2xl font-extrabold text-bench-accent">{p2pRuns.toLocaleString()}</div>
                <div className="text-[10px] text-bench-muted mt-1">P2P Demos</div>
              </div>
              <div className="card text-center py-4">
                <div className="text-2xl font-extrabold text-bench-accent">{zerotvmRuns.toLocaleString()}</div>
                <div className="text-[10px] text-bench-muted mt-1">Zero-TVM</div>
              </div>
            </div>
          );
        })()}

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          {/* Tab switcher */}
          <div className="inline-flex rounded-lg bg-bench-surface border border-bench-border p-1">
            <button
              onClick={() => setTab("compute")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                tab === "compute" ? "bg-bench-accent/10 text-bench-accent" : "text-bench-muted hover:text-bench-text"
              }`}
            >
              GPU Compute
            </button>
            <button
              onClick={() => setTab("transformer")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                tab === "transformer" ? "bg-bench-accent/10 text-bench-accent" : "text-bench-muted hover:text-bench-text"
              }`}
            >
              Transformer Fusion
            </button>
            <button
              onClick={() => setTab("demos")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                tab === "demos"
                  ? "bg-bench-accent/10 text-bench-accent"
                  : "text-bench-muted hover:text-bench-text"
              }`}
            >
              Demos
            </button>
          </div>

          {/* Search */}
          <input
            type="text"
            placeholder="Search GPU, vendor, browser..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 min-w-[200px] px-3 py-1.5 rounded-md text-xs bg-bench-surface border border-bench-border text-bench-text placeholder:text-bench-muted/50 focus:outline-none focus:border-bench-accent/50"
          />

          {/* Actions */}
          <button onClick={csvExport} className="btn-secondary text-xs py-1.5 px-3">
            Export CSV
          </button>
          <ShareButtons
            text={shareText}
            url="https://gpubench.dev/results"
            title="WebGPU Bench — All Results"
          />
        </div>

        {/* Table */}
        {loading ? (
          <div className="card text-center py-16 text-bench-muted">Loading results...</div>
        ) : tab === "compute" ? (
          <>
            <div className="rounded-lg border border-bench-border overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-bench-surface border-b border-bench-border">
                  <tr>
                    <SortHeader label="GPU" field="gpu_name" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Vendor" field="gpu_vendor" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Score" field="score" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Rastrigin" field="rastrigin_gps" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="N-Body" field="nbody_gps" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Acrobot" field="acrobot_gps" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="MtnCar" field="mountaincar_gps" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="MC Pi" field="montecarlo_gps" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <th className="px-2 py-2 text-left text-[10px] uppercase tracking-wider text-bench-muted">Browser</th>
                    <th className="px-2 py-2 text-left text-[10px] uppercase tracking-wider text-bench-muted">OS</th>
                    <SortHeader label="When" field="created_at" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                  </tr>
                </thead>
                <tbody>
                  {filteredCompute.map((r, i) => (
                    <tr
                      key={`${r.created_at}-${i}`}
                      className={`${i % 2 === 0 ? "bg-bench-bg/30" : "bg-bench-surface"} hover:bg-bench-accent/5 transition-colors`}
                    >
                      <td className="px-2 py-1.5 font-medium text-bench-text truncate max-w-[180px]">
                        {r.gpu_name}
                        {r.is_mobile && <span className="ml-1 text-[9px] px-1 py-0.5 rounded bg-bench-accent/10 text-bench-accent">mobile</span>}
                      </td>
                      <td className="px-2 py-1.5 text-bench-muted">{r.gpu_vendor}</td>
                      <td className="px-2 py-1.5 text-bench-accent font-bold tabular-nums">{Math.round(r.score)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.rastrigin_gps)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.nbody_gps)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.acrobot_gps)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.mountaincar_gps)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.montecarlo_gps)}</td>
                      <td className="px-2 py-1.5 text-bench-muted/60">{shortBrowser(r.browser)}</td>
                      <td className="px-2 py-1.5 text-bench-muted/60">{r.os}</td>
                      <td className="px-2 py-1.5 text-bench-muted/40 tabular-nums text-[10px]">{timeAgo(r.created_at)}</td>
                    </tr>
                  ))}
                  {filteredCompute.length === 0 && (
                    <tr><td colSpan={11} className="px-4 py-8 text-center text-bench-muted">No results found</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            {/* Pagination */}
            {computePages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-4">
                <button
                  disabled={computePage <= 1}
                  onClick={() => handleComputePage(computePage - 1)}
                  className="btn-secondary text-xs py-1 px-2 disabled:opacity-30"
                >
                  Prev
                </button>
                <span className="text-xs text-bench-muted">
                  Page {computePage} of {computePages}
                </span>
                <button
                  disabled={computePage >= computePages}
                  onClick={() => handleComputePage(computePage + 1)}
                  className="btn-secondary text-xs py-1 px-2 disabled:opacity-30"
                >
                  Next
                </button>
              </div>
            )}
          </>
        ) : tab === "transformer" ? (
          <>
            <div className="rounded-lg border border-bench-border overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-bench-surface border-b border-bench-border">
                  <tr>
                    <SortHeader label="GPU" field="gpu_name" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Config" field="config" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="D" field="d_model" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Layers" field="layers" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Unfused ms" field="unfused_ms" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Fused ms" field="fused_1t_ms" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Parallel ms" field="parallel_ms" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="1T\u00D7" field="speedup_1t" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Par\u00D7" field="speedup_parallel" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Tok/s" field="tokens_per_sec" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <th className="px-2 py-2 text-left text-[10px] uppercase tracking-wider text-bench-muted">Browser</th>
                    <SortHeader label="When" field="created_at" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                  </tr>
                </thead>
                <tbody>
                  {filteredTransformer.map((r, i) => (
                    <tr
                      key={`${r.created_at}-${i}`}
                      className={`${i % 2 === 0 ? "bg-bench-bg/30" : "bg-bench-surface"} hover:bg-bench-accent/5 transition-colors`}
                    >
                      <td className="px-2 py-1.5 font-medium text-bench-text truncate max-w-[180px]">
                        {r.gpu_name}
                        {r.is_mobile && <span className="ml-1 text-[9px] px-1 py-0.5 rounded bg-bench-accent/10 text-bench-accent">mobile</span>}
                      </td>
                      <td className="px-2 py-1.5 text-bench-muted">{r.config}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{r.d_model}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{r.layers}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.unfused_ms)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.fused_1t_ms)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.parallel_ms)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-accent font-bold">{fmtNum(r.speedup_1t)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-accent font-bold">{fmtNum(r.speedup_parallel)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.tokens_per_sec)}</td>
                      <td className="px-2 py-1.5 text-bench-muted/60">{shortBrowser(r.browser)} {r.os}</td>
                      <td className="px-2 py-1.5 text-bench-muted/40 tabular-nums text-[10px]">{timeAgo(r.created_at)}</td>
                    </tr>
                  ))}
                  {filteredTransformer.length === 0 && (
                    <tr><td colSpan={12} className="px-4 py-8 text-center text-bench-muted">No results found</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            {transformerPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-4">
                <button
                  disabled={transformerPage <= 1}
                  onClick={() => handleTransformerPage(transformerPage - 1)}
                  className="btn-secondary text-xs py-1 px-2 disabled:opacity-30"
                >
                  Prev
                </button>
                <span className="text-xs text-bench-muted">
                  Page {transformerPage} of {transformerPages}
                </span>
                <button
                  disabled={transformerPage >= transformerPages}
                  onClick={() => handleTransformerPage(transformerPage + 1)}
                  className="btn-secondary text-xs py-1 px-2 disabled:opacity-30"
                >
                  Next
                </button>
              </div>
            )}
          </>
        ) : tab === "demos" ? (
          <>
            <div className="rounded-lg border border-bench-border overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-bench-surface border-b border-bench-border">
                  <tr>
                    <SortHeader label="GPU" field="gpu" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Device" field="device_name" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Workload" field="workload" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Fitness" field="fitness" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Gen / Tok" field="gen" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <SortHeader label="Rate /s" field="speed" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                    <th className="px-2 py-2 text-left text-[10px] uppercase tracking-wider text-bench-muted">Browser</th>
                    <th className="px-2 py-2 text-left text-[10px] uppercase tracking-wider text-bench-muted">OS</th>
                    <SortHeader label="When" field="created_at" currentSort={sortField} currentDir={sortDir} onSort={handleSort} />
                  </tr>
                </thead>
                <tbody>
                  {filteredDemos.map((r, i) => (
                    <tr
                      key={`${r.created_at}-${r.device_id}-${i}`}
                      className={`${i % 2 === 0 ? "bg-bench-bg/30" : "bg-bench-surface"} hover:bg-bench-accent/5 transition-colors`}
                    >
                      <td className="px-2 py-1.5 font-medium text-bench-text truncate max-w-[180px]">{r.gpu}</td>
                      <td className="px-2 py-1.5 text-bench-muted">
                        {r.device_name}
                        {r.is_mobile && <span className="ml-1 text-[9px] px-1 py-0.5 rounded bg-bench-accent/10 text-bench-accent">mobile</span>}
                      </td>
                      <td className="px-2 py-1.5 text-bench-muted">
                        {r.workload?.toLowerCase().includes("zero-tvm") ? (
                          <span className="inline-flex items-center gap-1">
                            <span className="text-[9px] px-1 py-0.5 rounded bg-bench-green/10 text-bench-green font-semibold uppercase tracking-wider">LLM</span>
                            {r.workload}
                          </span>
                        ) : (
                          r.workload
                        )}
                      </td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-accent font-bold">{fmtNum(r.fitness)}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{r.gen?.toLocaleString() ?? "—"}</td>
                      <td className="px-2 py-1.5 tabular-nums text-bench-muted">{fmtNum(r.speed)}</td>
                      <td className="px-2 py-1.5 text-bench-muted/60">{shortBrowser(r.browser)}</td>
                      <td className="px-2 py-1.5 text-bench-muted/60">{r.os}</td>
                      <td className="px-2 py-1.5 text-bench-muted/40 tabular-nums text-[10px]">{timeAgo(r.created_at)}</td>
                    </tr>
                  ))}
                  {filteredDemos.length === 0 && (
                    <tr><td colSpan={9} className="px-4 py-8 text-center text-bench-muted">No demo sessions found</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            {demoPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-4">
                <button
                  disabled={demoPage <= 1}
                  onClick={() => handleDemoPage(demoPage - 1)}
                  className="btn-secondary text-xs py-1 px-2 disabled:opacity-30"
                >
                  Prev
                </button>
                <span className="text-xs text-bench-muted">
                  Page {demoPage} of {demoPages}
                </span>
                <button
                  disabled={demoPage >= demoPages}
                  onClick={() => handleDemoPage(demoPage + 1)}
                  className="btn-secondary text-xs py-1 px-2 disabled:opacity-30"
                >
                  Next
                </button>
              </div>
            )}
          </>
        ) : null}

        <p className="text-[10px] text-bench-muted/40 text-center mt-8 mb-4">
          All data is anonymous. Only GPU model, benchmark throughput, browser, and OS are stored. No personal data.
        </p>
      </div>

      {/* Footer */}
      <footer className="border-t border-bench-border/50 mt-8">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-1.5 text-sm">
              <svg className="w-4 h-4 text-bench-accent" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2.5" strokeDasharray="60 28" strokeLinecap="round"/>
                <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
              </svg>
              <span className="font-semibold text-bench-text">WebGPU Bench</span>
            </div>
            <div className="flex items-center gap-5 text-sm text-bench-muted">
              <a href="/" className="hover:text-bench-text transition">Run Benchmarks</a>
              <a href="/why" className="hover:text-bench-text transition">Why this matters</a>
              <a href={LINKS.research} className="hover:text-bench-text transition">Research</a>
              <a href={LINKS.repo} className="hover:text-bench-text transition">GitHub</a>
              <a href="/privacy" className="hover:text-bench-text transition">Privacy</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
