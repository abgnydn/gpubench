// Runs the idempotent schema migrations against the Neon DB.
// Invoked at the end of `npm run build`, so on Vercel every deploy migrates
// BEFORE the new serverless functions go live (the build environment already
// has STORAGE_POSTGRES_URL — no secret ever leaves Vercel).
// Local/CI builds have no DB env and skip cleanly.

import { MIGRATIONS } from "../src/lib/migrations.js";

const url = process.env.STORAGE_POSTGRES_URL;
if (!url) {
  console.log("[migrate] STORAGE_POSTGRES_URL not set — skipping (local/CI build)");
  process.exit(0);
}

process.env.POSTGRES_URL = url;
const { sql } = await import("@vercel/postgres");

for (const statement of MIGRATIONS) {
  await sql.query(statement);
}
console.log(`[migrate] applied ${MIGRATIONS.length} idempotent migrations`);
