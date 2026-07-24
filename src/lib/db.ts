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
import { MIGRATIONS } from "./migrations";

let envChecked = false;

function ensureEnv(): void {
  if (envChecked) return;
  const live = process.env["STORAGE_POSTGRES_URL"];
  if (!live) {
    throw new Error(
      "STORAGE_POSTGRES_URL is not set. It lives in the deployment platform's secrets; locally pull it into .env.local.",
    );
  }
  process.env["POSTGRES_URL"] = live;
  process.env["POSTGRES_URL_NON_POOLING"] =
    process.env["STORAGE_POSTGRES_URL_NON_POOLING"] ?? live;
  envChecked = true;
}

// Self-healing schema: deploys are decoupled from migrations (no git
// integration, secrets only exist on the worker), so when new code hits a
// database that hasn't been migrated yet, the first "column/relation does
// not exist" error replays the idempotent migration list once and retries.
// After that first heal the schema is current for the lifetime of the DB.
let healAttempted = false;

function isSchemaError(err: unknown): boolean {
  return err instanceof Error && /does not exist/.test(err.message);
}

async function withSchemaHeal<T>(exec: () => Promise<T>): Promise<T> {
  try {
    return await exec();
  } catch (err) {
    if (!isSchemaError(err) || healAttempted) throw err;
    healAttempted = true;
    console.warn("[db] schema out of date — applying idempotent migrations");
    for (const m of MIGRATIONS) {
      await vercelSql.query(m);
    }
    return await exec();
  }
}

// `sql` is used both as a tagged template (apply) and via `sql.query` (get) —
// the proxy runs the env check on first touch either way, and every query
// goes through the schema-heal wrapper.
export const sql: typeof vercelSql = new Proxy(vercelSql, {
  apply(target, thisArg, args) {
    ensureEnv();
    return withSchemaHeal(() => Reflect.apply(target, thisArg, args));
  },
  get(target, prop, receiver) {
    ensureEnv();
    const value = Reflect.get(target, prop, receiver);
    if (prop === "query" && typeof value === "function") {
      return (...args: unknown[]) =>
        withSchemaHeal(() => (value as (...a: unknown[]) => Promise<unknown>).apply(target, args));
    }
    return typeof value === "function" ? value.bind(target) : value;
  },
});
