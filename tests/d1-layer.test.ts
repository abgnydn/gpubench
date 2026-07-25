/**
 * End-to-end test of the D1 data layer and API routes against a real SQLite
 * database (node:sqlite) standing in for D1 — same dialect, same semantics.
 * Verifies: schema self-ensure (fresh CREATE + ALTER upgrade of a legacy
 * table), the sql shim ($n translation, boolean coercion), submission
 * validation, and the JS median aggregations.
 */
import { describe, it, expect, vi, beforeAll } from "vitest";
import type { DatabaseSync } from "node:sqlite";

vi.mock("@opennextjs/cloudflare", async () => {
  const { DatabaseSync } = await import("node:sqlite");
  const sqlite = new DatabaseSync(":memory:");
  (globalThis as Record<string, unknown>)["__testSqlite"] = sqlite;

  class FakePrepared {
    private params: unknown[] = [];
    constructor(private text: string) {}
    bind(...values: unknown[]) {
      this.params = values;
      return this;
    }
    async all() {
      const stmt = sqlite.prepare(this.text);
      if (/^\s*(SELECT|PRAGMA)/i.test(this.text)) {
        return { results: stmt.all(...(this.params as never[])) };
      }
      stmt.run(...(this.params as never[]));
      return { results: [] };
    }
  }
  const fakeDb = { prepare: (q: string) => new FakePrepared(q) };
  return { getCloudflareContext: () => ({ env: { DB: fakeDb } }) };
});

const sqlite = () => (globalThis as Record<string, unknown>)["__testSqlite"] as DatabaseSync;

function jsonRequest(url: string, body: unknown): Request {
  return new Request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

const validSubmission = {
  gpuName: "Test GPU 9000",
  gpuVendor: "testcorp",
  gpuArch: "arch-1",
  score: 100,
  benchVersion: 2,
  rastrigin: 100,
  rastriginMean: 10,
  rastriginMin: 9,
  rastriginMax: 12,
  rastriginStd: 0.5,
  rastriginBatched: 400,
  isMobile: true,
};

beforeAll(async () => {
  // Force the (lazy) module mock so the in-memory SQLite handle exists.
  await import("@opennextjs/cloudflare");
  // Simulate production: transformer_runs exists with the PRE-v2 column set,
  // so ensureSchema must upgrade it via ALTER TABLE (benchmark_runs and
  // device_sessions don't exist at all and take the CREATE path).
  sqlite().exec(`CREATE TABLE transformer_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    gpu_name TEXT NOT NULL, gpu_vendor TEXT DEFAULT '', gpu_arch TEXT DEFAULT '',
    browser TEXT DEFAULT '', os TEXT DEFAULT '', config TEXT DEFAULT '',
    layers INTEGER DEFAULT 0, d_model INTEGER DEFAULT 0, dispatches INTEGER DEFAULT 0,
    unfused_ms REAL, fused_1t_ms REAL, parallel_ms REAL,
    speedup_1t REAL, speedup_parallel REAL, tokens_per_sec REAL,
    screen_width INTEGER DEFAULT 0, screen_height INTEGER DEFAULT 0, is_mobile INTEGER DEFAULT 0
  )`);
});

describe("results route on D1", () => {
  it("accepts a plausible submission and stores it", async () => {
    const { POST } = await import("@/app/api/results/route");
    const res = await POST(jsonRequest("http://test/api/results", validSubmission));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);

    const row = sqlite().prepare("SELECT gpu_name, score, bench_version, rastrigin_batched_gps, is_mobile FROM benchmark_runs").get() as Record<string, unknown>;
    expect(row["gpu_name"]).toBe("Test GPU 9000");
    expect(row["bench_version"]).toBe(2);
    expect(row["rastrigin_batched_gps"]).toBe(400);
    expect(row["is_mobile"]).toBe(1); // boolean coerced to SQLite integer
  });

  it("rejects structurally implausible stats with 400", async () => {
    const { POST } = await import("@/app/api/results/route");
    const res = await POST(jsonRequest("http://test/api/results", {
      ...validSubmission,
      rastriginMin: 50, // min > mean — impossible
    }));
    expect(res.status).toBe(400);
    const count = sqlite().prepare("SELECT COUNT(*) AS c FROM benchmark_runs").get() as { c: number };
    expect(count.c).toBe(1); // nothing extra stored
  });

  it("aggregates medians in GET, ignoring NULL columns like SQL does", async () => {
    const { POST, GET } = await import("@/app/api/results/route");
    // second plausible run with a different score → median of [100, 300] = 200
    await POST(jsonRequest("http://test/api/results", {
      ...validSubmission, score: 300, rastrigin: 200, rastriginMean: 5,
      rastriginMin: 4, rastriginMax: 7, rastriginStd: 0.2, rastriginBatched: 800,
    }));
    // legacy-style row: score only, every per-benchmark column NULL — must
    // not drag per-benchmark medians toward 0 (Number(null) === 0 trap)
    await POST(jsonRequest("http://test/api/results", {
      gpuName: "Legacy GPU", gpuVendor: "old", gpuArch: "", score: 200,
    }));
    const res = await GET(new Request("http://test/api/results"));
    const body = await res.json();
    expect(body.total).toBe(3);
    expect(body.averages.avg_score).toBe(200);
    expect(body.averages.avg_rastrigin).toBe(150); // NULLs ignored
    expect(body.topGpus[0].runs).toBe(2);
    expect(body.recent).toHaveLength(3);
  });

  it("paginates with translated $n placeholders", async () => {
    const { GET } = await import("@/app/api/results/route");
    const res = await GET(new Request("http://test/api/results?all=true&sort=score&dir=desc"));
    const body = await res.json();
    expect(body.total).toBe(3);
    expect(body.rows[0].score).toBe(300);
  });
});

describe("transformer route on D1", () => {
  it("schema self-ensure added the v2 columns to the legacy table", async () => {
    // Trigger any query so ensureSchema runs (results tests already did),
    // then check the legacy transformer_runs gained the new columns.
    const cols = sqlite().prepare("PRAGMA table_info(transformer_runs)").all() as { name: string }[];
    const names = new Set(cols.map((c) => c.name));
    expect(names.has("unfused_batched_ms")).toBe(true);
    expect(names.has("speedup_batched")).toBe(true);
    expect(names.has("equiv_max_diff")).toBe(true);
  });

  it("stores a run and aggregates medians + mobile count", async () => {
    const { POST, GET } = await import("@/app/api/transformer-results/route");
    const post = await POST(jsonRequest("http://test/api/transformer-results", {
      gpuName: "Test GPU 9000", gpuVendor: "apple", config: "D=32, L=1",
      layers: 1, dModel: 32, dispatches: 256,
      unfusedMs: 500, unfusedBatchedMs: 40, fused1tMs: 10, parallelMs: 5,
      speedup1t: 50, speedupParallel: 100, speedupBatched: 4,
      equivMaxDiff: 0.0001, tokensPerSec: 12800, isMobile: true,
    }));
    expect(post.status).toBe(200);

    const res = await GET(new Request("http://test/api/transformer-results"));
    const body = await res.json();
    expect(body.total).toBe(1);
    expect(body.topGpus[0].gpu_name).toBe("Test GPU 9000");
    expect(body.topGpus[0].avg_speedup).toBe(100);
    expect(body.vendors[0].name.toLowerCase()).toContain("apple");
    expect(body.mobile.runs).toBe(1); // 0/1 integer flag counted correctly
  });

  it("row listing includes the v2 columns", async () => {
    const { GET } = await import("@/app/api/transformer-results/route");
    const res = await GET(new Request("http://test/api/transformer-results?all=true"));
    const body = await res.json();
    expect(body.rows[0].unfused_batched_ms).toBe(40);
    expect(body.rows[0].equiv_max_diff).toBeCloseTo(0.0001);
  });
});

describe("device route on D1", () => {
  it("stores telemetry and aggregates workloads without percentile_cont", async () => {
    const { POST, GET } = await import("@/app/api/device/route");
    for (const speed of [10, 20, 30]) {
      const res = await POST(jsonRequest("http://test/api/device", {
        id: `dev-${speed}`, name: "tester", gpu: "Test GPU", workload: "rastrigin-p2p",
        fitness: speed / 100, gen: speed, speed, isMobile: false,
      }));
      expect(res.status).toBe(200);
    }
    const res = await GET(new Request("http://test/api/device"));
    const body = await res.json();
    expect(body.total).toBe(3);
    expect(body.devices).toBe(3);
    expect(body.workloads[0].workload).toBe("rastrigin-p2p");
    expect(body.workloads[0].avg_speed).toBe(20); // median of 10/20/30
    expect(body.workloads[0].max_gen).toBe(30);
  });
});
