import { sql } from "@/lib/db";
import { NextResponse } from "next/server";

// Rate limiting
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
  else if (ua.includes("CrOS")) os = "ChromeOS";
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

function bool(v: unknown): boolean {
  return v === true;
}

export async function POST(request: Request) {
  try {
    const forwarded = request.headers.get("x-forwarded-for");
    const ip = forwarded?.split(",")[0]?.trim() ?? "unknown";
    if (isRateLimited(ip)) {
      return NextResponse.json({ error: "Too many requests" }, { status: 429 });
    }

    const body = await request.json() as Record<string, unknown>;
    if (typeof body !== "object" || body === null) {
      return NextResponse.json({ error: "Invalid" }, { status: 400 });
    }

    const gpuName = str(body["gpuName"]);
    const gpuVendor = str(body["gpuVendor"]);
    const gpuArch = str(body["gpuArch"]);
    const score = num(body["score"]);
    if (!gpuName || score === null || score < 0 || score > 1_000_000) {
      return NextResponse.json({ error: "Invalid submission" }, { status: 400 });
    }

    const ua = request.headers.get("user-agent") ?? "";
    const { browser, os } = parseUA(ua);
    const id = crypto.randomUUID();

    await sql`
      INSERT INTO benchmark_runs (
        id, gpu_name, gpu_vendor, gpu_arch, max_buffer, features, browser, os,
        parallel_gps, sequential_gps, matrix_gps, score,
        rastrigin_gps, nbody_gps, acrobot_gps, mountaincar_gps, cartpole_gps, montecarlo_gps,
        max_workgroup_x, max_workgroup_y, max_workgroup_z, max_invocations,
        backend, device_pixel_ratio, screen_width, screen_height, is_mobile,
        rastrigin_mean, rastrigin_min, rastrigin_max, rastrigin_std,
        nbody_mean, nbody_min, nbody_max, nbody_std,
        acrobot_mean, acrobot_min, acrobot_max, acrobot_std,
        mountaincar_mean, mountaincar_min, mountaincar_max, mountaincar_std,
        cartpole_mean, cartpole_min, cartpole_max, cartpole_std,
        montecarlo_mean, montecarlo_min, montecarlo_max, montecarlo_std
      ) VALUES (
        ${id}, ${gpuName}, ${gpuVendor}, ${gpuArch},
        ${num(body["maxBuffer"]) ?? 0}, ${num(body["features"]) ?? 0},
        ${browser}, ${os},
        ${num(body["parallel"])}, ${num(body["sequential"])}, ${num(body["matrix"])}, ${score},
        ${num(body["rastrigin"])}, ${num(body["nbody"])}, ${num(body["acrobot"])},
        ${num(body["mountaincar"])}, ${num(body["cartpole"])}, ${num(body["montecarlo"])},
        ${num(body["maxWorkgroupX"]) ?? 0}, ${num(body["maxWorkgroupY"]) ?? 0},
        ${num(body["maxWorkgroupZ"]) ?? 0}, ${num(body["maxInvocations"]) ?? 0},
        ${str(body["backend"])}, ${num(body["devicePixelRatio"]) ?? 1},
        ${num(body["screenWidth"]) ?? 0}, ${num(body["screenHeight"]) ?? 0},
        ${bool(body["isMobile"])},
        ${num(body["rastriginMean"])}, ${num(body["rastriginMin"])}, ${num(body["rastriginMax"])}, ${num(body["rastriginStd"])},
        ${num(body["nbodyMean"])}, ${num(body["nbodyMin"])}, ${num(body["nbodyMax"])}, ${num(body["nbodyStd"])},
        ${num(body["acrobotMean"])}, ${num(body["acrobotMin"])}, ${num(body["acrobotMax"])}, ${num(body["acrobotStd"])},
        ${num(body["mountaincarMean"])}, ${num(body["mountaincarMin"])}, ${num(body["mountaincarMax"])}, ${num(body["mountaincarStd"])},
        ${num(body["cartpoleMean"])}, ${num(body["cartpoleMin"])}, ${num(body["cartpoleMax"])}, ${num(body["cartpoleStd"])},
        ${num(body["montecarloMean"])}, ${num(body["montecarloMin"])}, ${num(body["montecarloMax"])}, ${num(body["montecarloStd"])}
      )
    `;

    return NextResponse.json({ ok: true, id });
  } catch (err) {
    console.error("Failed to save:", err);
    return NextResponse.json({ error: "Failed to save" }, { status: 500 });
  }
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
        "created_at", "score", "gpu_name", "gpu_vendor",
        "rastrigin_gps", "nbody_gps", "acrobot_gps", "mountaincar_gps", "cartpole_gps", "montecarlo_gps",
      ]);
      const sortParam = searchParams.get("sort") ?? "created_at";
      const sortCol = ALLOWED_SORT.has(sortParam) ? sortParam : "created_at";
      const sortDir = searchParams.get("dir") === "asc" ? "ASC" : "DESC";

      const totalResult = await sql`SELECT COUNT(*) as count FROM benchmark_runs`;
      const total = Number(totalResult.rows[0]?.["count"] ?? 0);

      const rows = await sql.query(
        `SELECT gpu_name, gpu_vendor, gpu_arch, score,
                rastrigin_gps, nbody_gps, acrobot_gps, mountaincar_gps, cartpole_gps, montecarlo_gps,
                browser, os, is_mobile, created_at
         FROM benchmark_runs
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

    const totalResult = await sql`SELECT COUNT(*) as count FROM benchmark_runs`;
    const total = Number(totalResult.rows[0]?.["count"] ?? 0);

    // Aggregations use MEDIAN (percentile_cont 0.5), not mean.
    // Means in this dataset are right-skewed by a long tail of high-end devices
    // (e.g., Acrobot mean 105 vs median 40 — 2.6× skew). Medians describe the
    // typical device experience. JSON field names retain the `avg_` prefix
    // for back-compat with the frontend; rename pending.
    const avgResult = await sql`
      SELECT
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY rastrigin_gps)::numeric, 1) as avg_rastrigin,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY nbody_gps)::numeric, 1) as avg_nbody,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY acrobot_gps)::numeric, 1) as avg_acrobot,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY mountaincar_gps)::numeric, 1) as avg_mountaincar,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY cartpole_gps)::numeric, 1) as avg_cartpole,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY montecarlo_gps)::numeric, 1) as avg_montecarlo,
        ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY score)::numeric, 0) as avg_score
      FROM benchmark_runs
    `;

    const topGpus = await sql`
      SELECT gpu_name, gpu_vendor, gpu_arch, COUNT(*) as runs,
             ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY score)::numeric, 0) as avg_score
      FROM benchmark_runs
      GROUP BY gpu_name, gpu_vendor, gpu_arch
      ORDER BY avg_score DESC
      LIMIT 10
    `;

    const recentResult = await sql`
      SELECT gpu_name, score, browser, os, backend, is_mobile, created_at
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
    response.headers.set("Access-Control-Allow-Origin", "*");
    return response;
  } catch (err) {
    console.error("[api/results] GET failed:", err);
    return NextResponse.json({ total: 0, averages: {}, topGpus: [], recent: [] });
  }
}
