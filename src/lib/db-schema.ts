// Desired D1 (SQLite) schema. `create` builds a fresh database; `columns`
// is diffed against PRAGMA table_info so missing columns are added to an
// existing database (SQLite has no ADD COLUMN IF NOT EXISTS). Every column
// type in `columns` must be legal in ALTER TABLE ADD COLUMN — constant
// defaults only, no CURRENT_TIMESTAMP.

export interface TableSpec {
  create: string;
  columns: Record<string, string>;
}

const BENCH_STAT_COLUMNS: Record<string, string> = {};
for (const bench of ["rastrigin", "nbody", "acrobot", "mountaincar", "cartpole", "montecarlo"]) {
  BENCH_STAT_COLUMNS[`${bench}_gps`] = "REAL";
  BENCH_STAT_COLUMNS[`${bench}_mean`] = "REAL";
  BENCH_STAT_COLUMNS[`${bench}_min`] = "REAL";
  BENCH_STAT_COLUMNS[`${bench}_max`] = "REAL";
  BENCH_STAT_COLUMNS[`${bench}_std`] = "REAL";
  // v2 protocol: batched-submit throughput
  BENCH_STAT_COLUMNS[`${bench}_batched_gps`] = "REAL";
}

export const TABLES: Record<string, TableSpec> = {
  benchmark_runs: {
    create: `CREATE TABLE IF NOT EXISTS benchmark_runs (
      id TEXT PRIMARY KEY,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      gpu_name TEXT NOT NULL,
      gpu_vendor TEXT DEFAULT '',
      gpu_arch TEXT DEFAULT '',
      max_buffer INTEGER DEFAULT 0,
      features INTEGER DEFAULT 0,
      browser TEXT DEFAULT '',
      os TEXT DEFAULT '',
      parallel_gps REAL,
      sequential_gps REAL,
      matrix_gps REAL,
      score INTEGER,
      bench_version INTEGER DEFAULT 1,
      ${Object.entries(BENCH_STAT_COLUMNS).map(([c, t]) => `${c} ${t}`).join(",\n      ")},
      max_workgroup_x INTEGER DEFAULT 0,
      max_workgroup_y INTEGER DEFAULT 0,
      max_workgroup_z INTEGER DEFAULT 0,
      max_invocations INTEGER DEFAULT 0,
      backend TEXT DEFAULT '',
      device_pixel_ratio REAL DEFAULT 1,
      screen_width INTEGER DEFAULT 0,
      screen_height INTEGER DEFAULT 0,
      is_mobile INTEGER DEFAULT 0
    )`,
    columns: {
      bench_version: "INTEGER DEFAULT 1",
      ...BENCH_STAT_COLUMNS,
      max_workgroup_x: "INTEGER DEFAULT 0",
      max_workgroup_y: "INTEGER DEFAULT 0",
      max_workgroup_z: "INTEGER DEFAULT 0",
      max_invocations: "INTEGER DEFAULT 0",
      backend: "TEXT DEFAULT ''",
      device_pixel_ratio: "REAL DEFAULT 1",
      screen_width: "INTEGER DEFAULT 0",
      screen_height: "INTEGER DEFAULT 0",
      is_mobile: "INTEGER DEFAULT 0",
    },
  },
  transformer_runs: {
    create: `CREATE TABLE IF NOT EXISTS transformer_runs (
      id TEXT PRIMARY KEY,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      gpu_name TEXT NOT NULL,
      gpu_vendor TEXT DEFAULT '',
      gpu_arch TEXT DEFAULT '',
      browser TEXT DEFAULT '',
      os TEXT DEFAULT '',
      config TEXT DEFAULT '',
      layers INTEGER DEFAULT 0,
      d_model INTEGER DEFAULT 0,
      dispatches INTEGER DEFAULT 0,
      unfused_ms REAL,
      unfused_batched_ms REAL,
      fused_1t_ms REAL,
      parallel_ms REAL,
      speedup_1t REAL,
      speedup_parallel REAL,
      speedup_batched REAL,
      equiv_max_diff REAL,
      tokens_per_sec REAL,
      screen_width INTEGER DEFAULT 0,
      screen_height INTEGER DEFAULT 0,
      is_mobile INTEGER DEFAULT 0
    )`,
    columns: {
      unfused_batched_ms: "REAL",
      speedup_batched: "REAL",
      equiv_max_diff: "REAL",
    },
  },
  device_sessions: {
    create: `CREATE TABLE IF NOT EXISTS device_sessions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      device_id TEXT NOT NULL,
      device_name TEXT DEFAULT '',
      gpu TEXT DEFAULT '',
      workload TEXT NOT NULL,
      fitness REAL,
      gen INTEGER,
      speed REAL,
      browser TEXT DEFAULT '',
      os TEXT DEFAULT '',
      is_mobile INTEGER DEFAULT 0
    )`,
    columns: {},
  },
};
