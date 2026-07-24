// Idempotent schema migrations, shared by:
//   - scripts/migrate.mjs  — runs during `npm run build` on Vercel, so every
//     deploy self-migrates before the new functions go live
//   - /api/setup           — manual trigger (bearer-protected POST)
// Every statement must stay idempotent (IF NOT EXISTS) — the list is replayed
// in full on each run.

export const MIGRATIONS = [
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS rastrigin_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS nbody_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS acrobot_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mountaincar_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS cartpole_gps REAL`,
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
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS cartpole_mean REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS cartpole_min REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS cartpole_max REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS cartpole_std REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_mean REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_min REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_max REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_std REAL`,
  // v2 measurement protocol: batched-submit timing + protocol version.
  // v1 rows (bench_version=1) time one dispatch per submit and must not
  // be aggregated with v2 rows.
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS bench_version INT DEFAULT 1`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS rastrigin_batched_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS nbody_batched_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS acrobot_batched_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS mountaincar_batched_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS cartpole_batched_gps REAL`,
  `ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS montecarlo_batched_gps REAL`,
  // Device telemetry from P2P demos
  `CREATE TABLE IF NOT EXISTS device_sessions (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT NOW(),
    device_id       TEXT NOT NULL,
    device_name     TEXT DEFAULT '',
    gpu             TEXT DEFAULT '',
    workload        TEXT NOT NULL,
    fitness         REAL,
    gen             INT,
    speed           REAL,
    browser         TEXT DEFAULT '',
    os              TEXT DEFAULT '',
    is_mobile       BOOLEAN DEFAULT false
  )`,
  // Transformer benchmark table
  `CREATE TABLE IF NOT EXISTS transformer_runs (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT NOW(),
    gpu_name        TEXT NOT NULL,
    gpu_vendor      TEXT NOT NULL DEFAULT '',
    gpu_arch        TEXT NOT NULL DEFAULT '',
    browser         TEXT NOT NULL DEFAULT '',
    os              TEXT NOT NULL DEFAULT '',
    config          TEXT DEFAULT '',
    layers          INT DEFAULT 0,
    d_model         INT DEFAULT 0,
    dispatches      INT DEFAULT 0,
    unfused_ms      REAL,
    fused_1t_ms     REAL,
    parallel_ms     REAL,
    speedup_1t      REAL,
    speedup_parallel REAL,
    tokens_per_sec  REAL,
    screen_width    INT DEFAULT 0,
    screen_height   INT DEFAULT 0,
    is_mobile       BOOLEAN DEFAULT false
  )`,
  // Transformer bench: fair single-submit baseline + equivalence check
  // (after the CREATE so the ALTERs always have a table to hit).
  `ALTER TABLE transformer_runs ADD COLUMN IF NOT EXISTS unfused_batched_ms REAL`,
  `ALTER TABLE transformer_runs ADD COLUMN IF NOT EXISTS speedup_batched REAL`,
  `ALTER TABLE transformer_runs ADD COLUMN IF NOT EXISTS equiv_max_diff REAL`,
];
