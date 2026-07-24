import { sql } from "@/lib/db";
import { NextResponse } from "next/server";
import { createRateLimiter, getClientIp, num, parseUA, str } from "@/lib/api-utils";

const isRateLimited = createRateLimiter(30); // generous — demos POST every 5s

export async function POST(request: Request) {
  try {
    if (isRateLimited(getClientIp(request))) {
      return NextResponse.json({ error: "Too many requests" }, { status: 429 });
    }

    const body = (await request.json()) as Record<string, unknown>;
    const deviceId = str(body["id"]) || crypto.randomUUID();
    const deviceName = str(body["name"]);
    const gpu = str(body["gpu"]);
    const workload = str(body["workload"]);
    const fitness = num(body["fitness"]);
    const gen = num(body["gen"]);
    const speed = num(body["speed"]);

    if (!workload) {
      return NextResponse.json({ error: "workload required" }, { status: 400 });
    }

    const ua = request.headers.get("user-agent") ?? "";
    const { browser, os } = parseUA(ua);
    const isMobile =
      /Android|iPhone|iPad|iPod/i.test(ua) || deviceName === "Mobile";

    await sql`
      INSERT INTO device_sessions (
        device_id, device_name, gpu, workload,
        fitness, gen, speed,
        browser, os, is_mobile
      ) VALUES (
        ${deviceId}, ${deviceName}, ${gpu}, ${workload},
        ${fitness}, ${gen}, ${speed},
        ${browser}, ${os}, ${isMobile}
      )
    `;

    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error("Device telemetry error:", err);
    return NextResponse.json({ error: "Failed" }, { status: 500 });
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const all = searchParams.get("all") === "true";

  try {
    if (all) {
      const page = Math.max(1, Number(searchParams.get("page") ?? 1));
      const limit = 100;
      const offset = (page - 1) * limit;

      const ALLOWED_SORT = new Set([
        "created_at", "gpu", "workload", "fitness", "gen", "speed", "device_name",
      ]);
      const sortParam = searchParams.get("sort") ?? "created_at";
      const sortCol = ALLOWED_SORT.has(sortParam) ? sortParam : "created_at";
      const sortDir = searchParams.get("dir") === "asc" ? "ASC" : "DESC";

      // P2P evolution demos are the only workload class shown in the
      // results UI now (the Zero-TVM iframed mirror was retired). Keep
      // the filter closed-set so callers can't sneak arbitrary LIKE
      // patterns through.
      const filterParam = searchParams.get("filter");
      let whereSQL = "";
      if (filterParam === "p2p") {
        whereSQL = "WHERE workload NOT ILIKE '%zero-tvm%'";
      }

      const totalResult = await sql.query(
        `SELECT COUNT(*) as count FROM device_sessions ${whereSQL}`,
      );
      const total = Number(totalResult.rows[0]?.["count"] ?? 0);

      const rows = await sql.query(
        `SELECT device_id, device_name, gpu, workload,
                fitness, gen, speed,
                browser, os, is_mobile, created_at
         FROM device_sessions
         ${whereSQL}
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
      response.headers.set(
        "Cache-Control",
        "public, s-maxage=10, stale-while-revalidate=30",
      );
      response.headers.set("Access-Control-Allow-Origin", "*");
      return response;
    }

    // Default: aggregate stats + recent
    const totalResult = await sql`SELECT COUNT(*) as count FROM device_sessions`;
    const total = Number(totalResult.rows[0]?.["count"] ?? 0);

    const uniqueDevices = await sql`SELECT COUNT(DISTINCT device_id) as count FROM device_sessions`;
    const devices = Number(uniqueDevices.rows[0]?.["count"] ?? 0);

    // Median, not mean — keeps `avg_speed` field name for back-compat.
    const byWorkload = await sql`
      SELECT workload, COUNT(*) as reports,
             COUNT(DISTINCT device_id) as devices,
             ROUND(MAX(fitness)::numeric, 4) as peak_fitness,
             ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY speed)::numeric, 1) as avg_speed,
             MAX(gen) as max_gen
      FROM device_sessions
      GROUP BY workload
      ORDER BY reports DESC
    `;

    const recent = await sql`
      SELECT device_name, gpu, workload, fitness, gen, speed, browser, os, is_mobile, created_at
      FROM device_sessions
      ORDER BY created_at DESC
      LIMIT 20
    `;

    const response = NextResponse.json({
      total,
      devices,
      workloads: byWorkload.rows,
      recent: recent.rows,
    });
    response.headers.set(
      "Cache-Control",
      "public, s-maxage=10, stale-while-revalidate=30",
    );
    response.headers.set("Access-Control-Allow-Origin", "*");
    return response;
  } catch (err) {
    // Previously this catch silently returned the aggregate-shape empty
    // payload for BOTH branches, which masked schema/connection failures as
    // "no data." Log loudly, return a 500, and match the shape of whichever
    // branch was requested so clients don't crash on `json.rows`/`json.recent`.
    console.error("Device GET error:", err);
    if (all) {
      return NextResponse.json(
        { total: 0, page: 1, totalPages: 0, rows: [], error: "Failed" },
        { status: 500 },
      );
    }
    return NextResponse.json(
      { total: 0, devices: 0, workloads: [], recent: [], error: "Failed" },
      { status: 500 },
    );
  }
}

export async function OPTIONS() {
  const response = new NextResponse(null, { status: 204 });
  response.headers.set("Access-Control-Allow-Origin", "*");
  response.headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  response.headers.set("Access-Control-Allow-Headers", "Content-Type");
  return response;
}
