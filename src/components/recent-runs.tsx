"use client";

import { useEffect, useState } from "react";

interface Run {
  gpu_name: string;
  score: number;
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
  return raw.split(" ")[0] ?? raw;
}

interface RecentRunsProps {
  apiUrl?: string;
  className?: string;
}

export function RecentRuns({ apiUrl = "/api/results", className = "" }: RecentRunsProps) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const fetchRuns = async () => {
      try {
        const res = await fetch(apiUrl, { cache: "no-store" });
        if (!res.ok) return;
        const json = await res.json();
        if (!cancelled && Array.isArray(json.recent)) {
          setRuns(json.recent.slice(0, 10));
          setTotal(json.total ?? 0);
        }
      } catch {
        /* silent */
      }
    };
    fetchRuns();
    const interval = setInterval(fetchRuns, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [apiUrl]);

  if (runs.length === 0) return null;

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-bench-green animate-pulse" />
          <h3 className="text-xs font-medium text-bench-muted uppercase tracking-widest">
            Recent Runs
          </h3>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-bench-muted/60">
            {total.toLocaleString()} total
          </span>
          <a
            href="/results"
            className="text-[10px] text-bench-accent hover:underline"
          >
            View all &rarr;
          </a>
        </div>
      </div>
      <div className="rounded-lg border border-bench-border overflow-hidden">
        {runs.map((run, i) => (
          <div
            key={`${run.created_at}-${i}`}
            className={`flex items-center gap-3 px-3 py-2 text-xs ${
              i > 0 ? "border-t border-bench-border/50" : ""
            } ${i % 2 === 0 ? "bg-bench-surface" : "bg-bench-bg/30"}`}
          >
            <span className="text-bench-text font-medium truncate min-w-0 flex-1">
              {run.gpu_name}
            </span>
            <span className="text-bench-accent font-bold tabular-nums w-12 text-right">
              {Math.round(run.score).toLocaleString()}
            </span>
            <span className="text-bench-muted/60 w-16 truncate hidden sm:block">
              {shortBrowser(run.browser)} {run.os}
            </span>
            {run.is_mobile && (
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-bench-accent/10 text-bench-accent">
                mobile
              </span>
            )}
            <span className="text-bench-muted/40 tabular-nums w-14 text-right text-[10px]">
              {timeAgo(run.created_at)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
