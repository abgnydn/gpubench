import { sql } from "@vercel/postgres";
import { NextResponse } from "next/server";

const rateLimit = new Map<string, { count: number; resetAt: number }>();
function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const entry = rateLimit.get(ip);
  if (!entry || now > entry.resetAt) {
    rateLimit.set(ip, { count: 1, resetAt: now + 60_000 });
    return false;
  }
  entry.count++;
  return entry.count > 10;
}

function parseUA(ua: string): { browser: string; os: string } {
  let browser = "Unknown";
  let os = "Unknown";
  if (ua.includes("Chrome/")) {
    const match = /Chrome\/([\d.]+)/.exec(ua);
    browser = match ? `Chrome ${match[1]}` : "Chrome";
  } else if (ua.includes("Firefox/")) { browser = "Firefox"; }
  else if (ua.includes("Safari/") && !ua.includes("Chrome")) { browser = "Safari"; }
  if (ua.includes("Mac OS X")) os = "macOS";
  else if (ua.includes("Windows")) os = "Windows";
  else if (ua.includes("Linux")) os = "Linux";
  else if (ua.includes("Android")) os = "Android";
  return { browser, os };
}

function num(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return v;
}

function str(v: unknown, max = 500): string {
  if (typeof v !== "string") return "";
  return v.slice(0, max);
}

export async function POST(request: Request) {
  try {
    const forwarded = request.headers.get("x-forwarded-for");
    const ip = forwarded?.split(",")[0]?.trim() ?? "unknown";
    if (isRateLimited(ip)) {
      return NextResponse.json({ error: "Too many requests" }, { status: 429 });
    }

    const body = await request.json() as Record<string, unknown>;
    const gpuName = str(body["gpuName"]);
    if (!gpuName) {
      return NextResponse.json({ error: "Invalid" }, { status: 400 });
    }

    const ua = request.headers.get("user-agent") ?? "";
    const { browser, os } = parseUA(ua);
    const id = crypto.randomUUID();

    await sql`
      INSERT INTO transformer_runs (
        id, gpu_name, gpu_vendor, gpu_arch, browser, os,
        config, layers, d_model, dispatches,
        unfused_ms, fused_1t_ms, parallel_ms,
        speedup_1t, speedup_parallel,
        tokens_per_sec, screen_width, screen_height, is_mobile
      ) VALUES (
        ${id}, ${gpuName}, ${str(body["gpuVendor"])}, ${str(body["gpuArch"])},
        ${browser}, ${os},
        ${str(body["config"])}, ${num(body["layers"])}, ${num(body["dModel"])},
        ${num(body["dispatches"])},
        ${num(body["unfusedMs"])}, ${num(body["fused1tMs"])}, ${num(body["parallelMs"])},
        ${num(body["speedup1t"])}, ${num(body["speedupParallel"])},
        ${num(body["tokensPerSec"])},
        ${num(body["screenWidth"]) ?? 0}, ${num(body["screenHeight"]) ?? 0},
        ${body["isMobile"] === true}
      )
    `;

    return NextResponse.json({ ok: true, id });
  } catch (err) {
    console.error("Failed to save:", err);
    return NextResponse.json({ error: "Failed to save" }, { status: 500 });
  }
}

function bucketVendor(name: string, vendor: string): string {
  const n = (name || "").toLowerCase();
  const v = (vendor || "").toLowerCase();
  if (n.includes("apple") || v.includes("apple")) return "Apple Silicon";
  if (n.includes("adreno") || v.includes("qualcomm")) return "Qualcomm Adreno";
  if (n.includes("mali") || v.includes("arm")) return "ARM Mali";
  if (n.includes("nvidia") || n.includes("rtx") || n.includes("geforce") || v.includes("nvidia")) return "NVIDIA";
  if (n.includes("radeon") || n.includes("amd") || v.includes("amd")) return "AMD";
  if (n.includes("intel") || v.includes("intel")) return "Intel";
  return "Other";
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const all = searchParams.get("all") === "true";

    if (all) {
      const page = Math.max(1, Number(searchParams.get("page") ?? 1));
      const limit = 100;
      const offset = (page - 1) * limit;

      const ALLOWED_SORT = new Set([
        "created_at", "gpu_name", "config", "d_model", "layers",
        "unfused_ms", "fused_1t_ms", "parallel_ms",
        "speedup_1t", "speedup_parallel", "tokens_per_sec",
      ]);
      const sortParam = searchParams.get("sort") ?? "created_at";
      const sortCol = ALLOWED_SORT.has(sortParam) ? sortParam : "created_at";
      const sortDir = searchParams.get("dir") === "asc" ? "ASC" : "DESC";

      const totalResult = await sql`SELECT COUNT(*) as count FROM transformer_runs`;
      const total = Number(totalResult.rows[0]?.["count"] ?? 0);

      const rows = await sql.query(
        `SELECT gpu_name, gpu_vendor, gpu_arch, config, layers, d_model, dispatches,
                unfused_ms, fused_1t_ms, parallel_ms,
                speedup_1t, speedup_parallel, tokens_per_sec,
                browser, os, is_mobile, created_at
         FROM transformer_runs
         ORDER BY ${sortCol} ${sortDir} NULLS LAST
         LIMIT $1 OFFSET $2`,
        [limit, offset],
      );

      const response = NextResponse.json({
        total,
        page,
        totalPages: Math.ceil(total / limit),
        rows: rows.rows,
      });
      response.headers.set("Cache-Control", "public, s-maxage=30, stale-while-revalidate=60");
      response.headers.set("Access-Control-Allow-Origin", "*");
      return response;
    }

    const totalResult = await sql`SELECT COUNT(*) as count FROM transformer_runs`;
    const total = Number(totalResult.rows[0]?.["count"] ?? 0);

    const topGpus = await sql`
      SELECT gpu_name, COUNT(*) as runs, ROUND(AVG(speedup_parallel)::numeric, 1) as avg_speedup
      FROM transformer_runs WHERE speedup_parallel IS NOT NULL
      GROUP BY gpu_name ORDER BY avg_speedup DESC LIMIT 10
    `;

    const allRows = await sql`
      SELECT gpu_name, gpu_vendor, speedup_parallel, tokens_per_sec, is_mobile, browser
      FROM transformer_runs
      WHERE speedup_parallel IS NOT NULL
    `;

    const byVendor: Record<string, { total: number; sum: number; peak: number }> = {};
    let mobileCount = 0;
    let tpsSum = 0;
    let tpsCount = 0;
    let tpsPeak = 0;
    const browserCounts: Record<string, number> = {};

    for (const row of allRows.rows) {
      const speedup = Number(row["speedup_parallel"] ?? 0);
      const tps = Number(row["tokens_per_sec"] ?? 0);
      const bucket = bucketVendor(
        String(row["gpu_name"] ?? ""),
        String(row["gpu_vendor"] ?? ""),
      );
      if (!byVendor[bucket]) byVendor[bucket] = { total: 0, sum: 0, peak: 0 };
      byVendor[bucket].total++;
      byVendor[bucket].sum += speedup;
      if (speedup > byVendor[bucket].peak) byVendor[bucket].peak = speedup;
      if (row["is_mobile"] === true) mobileCount++;
      if (Number.isFinite(tps) && tps > 0) {
        tpsSum += tps;
        tpsCount++;
        if (tps > tpsPeak) tpsPeak = tps;
      }
      const browser = String(row["browser"] ?? "Unknown").split(" ")[0] ?? "Unknown";
      browserCounts[browser] = (browserCounts[browser] ?? 0) + 1;
    }

    const vendorAggregates = Object.entries(byVendor)
      .map(([name, v]) => ({
        name,
        runs: v.total,
        avgSpeedup: Math.round(v.sum / Math.max(v.total, 1)),
        peakSpeedup: Math.round(v.peak),
      }))
      .sort((a, b) => b.avgSpeedup - a.avgSpeedup);

    const response = NextResponse.json({
      total,
      topGpus: topGpus.rows,
      vendors: vendorAggregates,
      mobile: {
        runs: mobileCount,
        avgTokensPerSec: tpsCount ? Math.round(tpsSum / tpsCount) : 0,
        peakTokensPerSec: Math.round(tpsPeak),
      },
      browsers: browserCounts,
    });
    response.headers.set("Cache-Control", "public, s-maxage=30, stale-while-revalidate=60");
    response.headers.set("Access-Control-Allow-Origin", "*");
    response.headers.set("Access-Control-Allow-Methods", "GET, OPTIONS");
    return response;
  } catch {
    const fallback = NextResponse.json({
      total: 0,
      topGpus: [],
      vendors: [],
      mobile: { runs: 0, avgTokensPerSec: 0, peakTokensPerSec: 0 },
      browsers: {},
    });
    fallback.headers.set("Access-Control-Allow-Origin", "*");
    return fallback;
  }
}

export async function OPTIONS() {
  const response = new NextResponse(null, { status: 204 });
  response.headers.set("Access-Control-Allow-Origin", "*");
  response.headers.set("Access-Control-Allow-Methods", "GET, OPTIONS");
  response.headers.set("Access-Control-Allow-Headers", "Content-Type");
  return response;
}
