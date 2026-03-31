import { sql } from "@vercel/postgres";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const authHeader = request.headers.get("authorization");
  const secret = process.env["SETUP_SECRET"];

  if (!secret) {
    return NextResponse.json({ error: "SETUP_SECRET not configured" }, { status: 500 });
  }

  if (authHeader !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const test = await sql`SELECT 1 as connected`;
    if (!test.rows[0]) {
      return NextResponse.json({ error: "DB connection failed" }, { status: 500 });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: "DB connection failed", detail: message }, { status: 500 });
  }

  try {
    // Add new columns to existing table (safe — IF NOT EXISTS equivalent via ALTER)
    const migrations = [
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS rastrigin_gps REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS nbody_gps REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS acrobot_gps REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mountaincar_gps REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_gps REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS max_workgroup_x INT DEFAULT 0`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS max_workgroup_y INT DEFAULT 0`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS max_workgroup_z INT DEFAULT 0`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS max_invocations INT DEFAULT 0`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS backend TEXT DEFAULT ''`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS device_pixel_ratio REAL DEFAULT 1`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS screen_width INT DEFAULT 0`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS screen_height INT DEFAULT 0`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS is_mobile BOOLEAN DEFAULT false`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS rastrigin_mean REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS rastrigin_min REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS rastrigin_max REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS rastrigin_std REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS nbody_mean REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS nbody_min REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS nbody_max REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS nbody_std REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS acrobot_mean REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS acrobot_min REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS acrobot_max REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS acrobot_std REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mountaincar_mean REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mountaincar_min REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mountaincar_max REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mountaincar_std REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_mean REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_min REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_max REAL`,
      `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_std REAL`,
    ];

    for (const m of migrations) {
      await sql.query(m);
    }

    return NextResponse.json({ ok: true, message: "Migrations applied", columns_added: migrations.length });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: "Migration failed", detail: message }, { status: 500 });
  }
}
