// D1-backed data layer.
//
// Production runs on Cloudflare Workers (OpenNext) with the D1 binding `DB`
// — that database holds all historical runs. The old @vercel/postgres layer
// in this repo predated the (previously unpushed) D1 port and never matched
// production; deploying it took the data APIs down on 2026-07-24.
//
// The exported `sql` keeps the @vercel/postgres call shape — tagged template
// plus `sql.query(text, [$1-style params])` — so route code is unchanged.
// Postgres-only SQL (percentile_cont, ::numeric, ILIKE) is not supported;
// those aggregations live in JS in the routes.
//
// Schema is self-ensuring: once per isolate, tables are created if missing
// and columns are diffed via PRAGMA table_info and added (SQLite has no
// ADD COLUMN IF NOT EXISTS). Deploys never need a separate migration step.

import { getCloudflareContext } from "@opennextjs/cloudflare";
import { TABLES } from "./db-schema";

type Bindable = string | number | boolean | null | undefined;

interface D1Rows {
  results?: Record<string, unknown>[];
}
interface D1Prepared {
  bind(...values: unknown[]): D1Prepared;
  all(): Promise<D1Rows>;
}
export interface D1Like {
  prepare(query: string): D1Prepared;
}

function getDb(): D1Like {
  const { env } = getCloudflareContext() as unknown as { env: Record<string, unknown> };
  const db = env["DB"] as D1Like | undefined;
  if (!db) {
    throw new Error(
      "D1 binding `DB` is not available. Outside the Worker (plain `next dev`) the data APIs have no database; deploy config lives in wrangler.jsonc.",
    );
  }
  return db;
}

let ensured: Promise<void> | null = null;

function ensureSchema(db: D1Like): Promise<void> {
  ensured ??= (async () => {
    for (const [name, table] of Object.entries(TABLES)) {
      await db.prepare(table.create).all();
      const info = await db.prepare(`PRAGMA table_info(${name})`).all();
      const existing = new Set((info.results ?? []).map((r) => String(r["name"])));
      for (const [col, type] of Object.entries(table.columns)) {
        if (!existing.has(col)) {
          await db.prepare(`ALTER TABLE ${name} ADD COLUMN ${col} ${type}`).all();
        }
      }
    }
  })().catch((err: unknown) => {
    ensured = null; // retry on the next query instead of caching the failure
    throw err;
  });
  return ensured;
}

async function run(text: string, params: Bindable[]): Promise<{ rows: Record<string, unknown>[] }> {
  const db = getDb();
  await ensureSchema(db);
  const bound = params.map((p) =>
    p === undefined ? null : typeof p === "boolean" ? (p ? 1 : 0) : p,
  );
  const res = await db.prepare(text).bind(...bound).all();
  return { rows: res.results ?? [] };
}

function tag(strings: TemplateStringsArray, ...values: Bindable[]) {
  let text = strings[0] ?? "";
  for (let i = 0; i < values.length; i++) {
    text += `?${i + 1}${strings[i + 1] ?? ""}`;
  }
  return run(text, values);
}

export const sql = Object.assign(tag, {
  // $1/$2 placeholders translate directly to SQLite's ?1/?2.
  query: (text: string, params: Bindable[] = []) =>
    run(text.replace(/\$(\d+)/g, "?$1"), params),
});

/** Used by POST /api/setup — any query triggers the schema self-ensure. */
export async function ensureDatabase(): Promise<void> {
  await run("SELECT 1 AS ok", []);
}
