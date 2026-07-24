import { sql } from "@/lib/db";
import { MIGRATIONS } from "@/lib/migrations";
import { NextResponse } from "next/server";

// Migrations are a side effect, so this is a POST. (It used to be a GET —
// bearer-protected, but side-effecting GETs are prefetcher/link-scanner bait.)
export async function POST(request: Request) {
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
    for (const m of MIGRATIONS) {
      await sql.query(m);
    }

    return NextResponse.json({ ok: true, message: "Migrations applied", columns_added: MIGRATIONS.length });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: "Migration failed", detail: message }, { status: 500 });
  }
}
