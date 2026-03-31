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

export async function GET() {
  try {
    const totalResult = await sql`SELECT COUNT(*) as count FROM transformer_runs`;
    const total = Number(totalResult.rows[0]?.["count"] ?? 0);

    const topGpus = await sql`
      SELECT gpu_name, COUNT(*) as runs, ROUND(AVG(speedup_parallel)::numeric, 1) as avg_speedup
      FROM transformer_runs WHERE speedup_parallel IS NOT NULL
      GROUP BY gpu_name ORDER BY avg_speedup DESC LIMIT 10
    `;

    const response = NextResponse.json({ total, topGpus: topGpus.rows });
    response.headers.set("Cache-Control", "public, s-maxage=30, stale-while-revalidate=60");
    return response;
  } catch {
    return NextResponse.json({ total: 0, topGpus: [] });
  }
}
