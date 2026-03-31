import { sql } from "@vercel/postgres";
import { NextResponse } from "next/server";

interface BenchmarkSubmission {
  gpuName: string;
  gpuVendor: string;
  gpuArch: string;
  maxBuffer: number;
  features: number;
  parallel: number | null;
  sequential: number | null;
  matrix: number | null;
  score: number;
}

// Simple rate limit: 10 submissions per minute per IP
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

function isValidSubmission(body: unknown): body is BenchmarkSubmission {
  if (typeof body !== "object" || body === null) return false;
  const b = body as Record<string, unknown>;
  return (
    typeof b["gpuName"] === "string" && (b["gpuName"] as string).length < 500 &&
    typeof b["gpuVendor"] === "string" && (b["gpuVendor"] as string).length < 500 &&
    typeof b["gpuArch"] === "string" && (b["gpuArch"] as string).length < 500 &&
    typeof b["maxBuffer"] === "number" && Number.isFinite(b["maxBuffer"] as number) &&
    typeof b["features"] === "number" && Number.isFinite(b["features"] as number) &&
    typeof b["score"] === "number" && (b["score"] as number) >= 0 && (b["score"] as number) < 1_000_000 &&
    (b["parallel"] === null || (typeof b["parallel"] === "number" && Number.isFinite(b["parallel"] as number))) &&
    (b["sequential"] === null || (typeof b["sequential"] === "number" && Number.isFinite(b["sequential"] as number))) &&
    (b["matrix"] === null || (typeof b["matrix"] === "number" && Number.isFinite(b["matrix"] as number)))
  );
}

function parseUA(ua: string): { browser: string; os: string } {
  let browser = "Unknown";
  let os = "Unknown";

  if (ua.includes("Chrome/")) {
    const match = /Chrome\/([\d.]+)/.exec(ua);
    browser = match ? `Chrome ${match[1]}` : "Chrome";
  } else if (ua.includes("Firefox/")) {
    browser = "Firefox";
  } else if (ua.includes("Safari/") && !ua.includes("Chrome")) {
    browser = "Safari";
  }

  if (ua.includes("Mac OS X")) os = "macOS";
  else if (ua.includes("Windows")) os = "Windows";
  else if (ua.includes("Linux")) os = "Linux";
  else if (ua.includes("CrOS")) os = "ChromeOS";
  else if (ua.includes("Android")) os = "Android";

  return { browser, os };
}

export async function POST(request: Request) {
  try {
    const forwarded = request.headers.get("x-forwarded-for");
    const ip = forwarded?.split(",")[0]?.trim() ?? "unknown";
    if (isRateLimited(ip)) {
      return NextResponse.json({ error: "Too many requests" }, { status: 429 });
    }

    const body: unknown = await request.json();

    if (!isValidSubmission(body)) {
      return NextResponse.json({ error: "Invalid submission" }, { status: 400 });
    }

    const ua = request.headers.get("user-agent") ?? "";
    const { browser, os } = parseUA(ua);
    const id = crypto.randomUUID();

    await sql`
      INSERT INTO benchmark_runs (id, gpu_name, gpu_vendor, gpu_arch, max_buffer, features, browser, os, parallel_gps, sequential_gps, matrix_gps, score)
      VALUES (${id}, ${body.gpuName}, ${body.gpuVendor}, ${body.gpuArch}, ${body.maxBuffer}, ${body.features}, ${browser}, ${os}, ${body.parallel}, ${body.sequential}, ${body.matrix}, ${body.score})
    `;

    return NextResponse.json({ ok: true, id });
  } catch (err) {
    console.error("Failed to save benchmark:", err);
    return NextResponse.json({ error: "Failed to save" }, { status: 500 });
  }
}

export async function GET() {
  try {
    const totalResult = await sql`SELECT COUNT(*) as count FROM benchmark_runs`;
    const total = Number(totalResult.rows[0]?.["count"] ?? 0);

    const avgResult = await sql`
      SELECT
        ROUND(AVG(parallel_gps)::numeric, 1) as avg_parallel,
        ROUND(AVG(sequential_gps)::numeric, 1) as avg_sequential,
        ROUND(AVG(matrix_gps)::numeric, 1) as avg_matrix,
        ROUND(AVG(score)::numeric, 0) as avg_score
      FROM benchmark_runs
    `;

    const topGpus = await sql`
      SELECT gpu_name, gpu_vendor, gpu_arch, COUNT(*) as runs, ROUND(AVG(score)::numeric, 0) as avg_score
      FROM benchmark_runs
      GROUP BY gpu_name, gpu_vendor, gpu_arch
      ORDER BY avg_score DESC
      LIMIT 10
    `;

    const recentResult = await sql`
      SELECT gpu_name, score, browser, os, created_at
      FROM benchmark_runs
      ORDER BY created_at DESC
      LIMIT 20
    `;

    const response = NextResponse.json({
      total,
      averages: avgResult.rows[0] ?? {},
      topGpus: topGpus.rows,
      recent: recentResult.rows,
    });

    response.headers.set("Cache-Control", "public, s-maxage=30, stale-while-revalidate=60");
    return response;
  } catch {
    return NextResponse.json({ total: 0, averages: {}, topGpus: [], recent: [] });
  }
}
