import { sql } from "@vercel/postgres";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const authHeader = request.headers.get("authorization");
  const secret = process.env["SETUP_SECRET"];

  if (!secret) {
    return NextResponse.json({ error: "SETUP_SECRET not configured" }, { status: 500 });
  }

  if (authHeader !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized", hint: "Use: Authorization: Bearer <SETUP_SECRET>" }, { status: 401 });
  }

  // Check DB connection first
  try {
    const test = await sql`SELECT 1 as connected`;
    if (!test.rows[0]) {
      return NextResponse.json({ error: "Database connection failed — no rows returned" }, { status: 500 });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({
      error: "Database connection failed",
      detail: message,
      hint: "Check POSTGRES_URL env var is set in Vercel"
    }, { status: 500 });
  }

  // Create table
  try {
    await sql`
      CREATE TABLE IF NOT EXISTS benchmark_runs (
        id            TEXT PRIMARY KEY,
        created_at    TIMESTAMP DEFAULT NOW(),
        gpu_name      TEXT NOT NULL,
        gpu_vendor    TEXT NOT NULL,
        gpu_arch      TEXT NOT NULL DEFAULT '',
        max_buffer    BIGINT NOT NULL DEFAULT 0,
        features      INT NOT NULL DEFAULT 0,
        browser       TEXT NOT NULL DEFAULT '',
        os            TEXT NOT NULL DEFAULT '',
        parallel_gps  REAL,
        sequential_gps REAL,
        matrix_gps    REAL,
        score         INT NOT NULL DEFAULT 0
      )
    `;

    return NextResponse.json({ ok: true, message: "Table created successfully" });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: "Table creation failed", detail: message }, { status: 500 });
  }
}
