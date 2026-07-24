// Single source of truth for the Postgres connection.
//
// `@vercel/postgres` lazy-loads its connection string from POSTGRES_URL on
// first use. Our DB is provisioned through the Vercel Marketplace (Neon),
// which injects STORAGE_POSTGRES_URL — so we mirror it into POSTGRES_URL
// on first query. The legacy POSTGRES_URL previously seeded on this project
// pointed at a dropped Vercel Postgres instance and was the reason the live
// API silently returned empty data.
//
// The env check is deliberately lazy (not at module load): API route modules
// get evaluated during `next build`, and a fresh clone without Vercel env
// should still build and run the non-DB parts of the site.

import { sql as vercelSql } from "@vercel/postgres";

let envChecked = false;

function ensureEnv(): void {
  if (envChecked) return;
  const live = process.env["STORAGE_POSTGRES_URL"];
  if (!live) {
    throw new Error(
      "STORAGE_POSTGRES_URL is not set. The Neon Marketplace integration injects this on Vercel; locally pull it with `vercel env pull .env.local`.",
    );
  }
  process.env["POSTGRES_URL"] = live;
  process.env["POSTGRES_URL_NON_POOLING"] =
    process.env["STORAGE_POSTGRES_URL_NON_POOLING"] ?? live;
  envChecked = true;
}

// `sql` is used both as a tagged template (apply) and via `sql.query` (get) —
// the proxy runs the env check on first touch either way.
export const sql: typeof vercelSql = new Proxy(vercelSql, {
  apply(target, thisArg, args) {
    ensureEnv();
    return Reflect.apply(target, thisArg, args);
  },
  get(target, prop, receiver) {
    ensureEnv();
    const value = Reflect.get(target, prop, receiver);
    return typeof value === "function" ? value.bind(target) : value;
  },
});
