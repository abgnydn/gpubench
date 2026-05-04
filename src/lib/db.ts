// Single source of truth for the Postgres connection.
//
// `@vercel/postgres` lazy-loads its connection string from POSTGRES_URL on
// first use. Our DB is provisioned through the Vercel Marketplace (Neon),
// which injects STORAGE_POSTGRES_URL — so we mirror it into POSTGRES_URL
// once at module load. The legacy POSTGRES_URL set previously seeded on
// this project pointed at a dropped Vercel Postgres instance and was the
// reason the live API silently returned empty data.

const live = process.env["STORAGE_POSTGRES_URL"];
if (!live) {
  throw new Error(
    "STORAGE_POSTGRES_URL is not set. The Neon Marketplace integration injects this on Vercel; locally pull it with `vercel env pull .env.local`.",
  );
}
process.env["POSTGRES_URL"] = live;
process.env["POSTGRES_URL_NON_POOLING"] =
  process.env["STORAGE_POSTGRES_URL_NON_POOLING"] ?? live;

export { sql } from "@vercel/postgres";
