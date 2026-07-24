# gpubench.dev

## Goal

Browser-native WebGPU benchmark suite with public, unfiltered results from
every device that runs it. The measurement arm of the kernel-fusion research
line — the numbers the preprints cite live in this DB. Every result published,
no cherry-picking.

## Architecture

Next.js 16 with App Router + `/api/*` routes backed by Neon Postgres
(`STORAGE_POSTGRES_URL`). Deploys to **Cloudflare Workers via OpenNext** —
config lives in-repo (`wrangler.jsonc` + `open-next.config.ts`); secrets
(`STORAGE_POSTGRES_URL`, `SETUP_SECRET`) live on the worker and persist
across deploys. Schema migrations are self-healing: any "column does not
exist" error replays the idempotent list in `src/lib/migrations.js` once
(see `src/lib/db.ts`), so deploy order never matters.

- `src/app/page.tsx` — main benchmark page. `BENCHMARKS` array (top of file)
  drives the card grid. Companion-projects section at the bottom renders
  from `CROSSLINKS.gpubench` with `COMPANION_FLAGSHIP` (kernelfusion) as a
  hero card + 3-col grid of adjacent siblings.
- `src/app/results/` — all-runs table with filtering.
- `src/app/transformer/` — separate transformer-fusion benchmark flow.
- `src/app/swarm/` — evolutionary-compute demo.
- `src/app/zerotvm/` — zero-tvm companion comparison page.
- `src/app/api/*/route.ts` — POST endpoints that insert benchmark results
  into Postgres. GET endpoints that return aggregates for the UI.
- `src/lib/benchmark-runner.ts` — WGSL dispatch + timing logic.
- `src/lib/gpu-detect.ts` — adapter info gathering (vendor, arch, limits).
- `src/lib/sites.ts` — synced from `~/sites-shared/sites.ts`.
- `src/components/benchmark-card.tsx`, `results-summary.tsx`,
  `recent-runs.tsx`, `paper-card.tsx` — UI primitives.

### Companion-projects convention

`CROSSLINKS.gpubench[0]` is the flagship (currently `kernelfusion`, the
research line). Rendered as a hero card with stats. The remaining three
(`webgpudna`, `zerotvm`, `neuropulse`) fill a 3-col grid via
`CATEGORY_BADGE` static class-string map.

## Commands

```bash
npm install
npm run dev          # Next dev server at localhost:3000
npm run build        # Next.js production build
npm run typecheck    # tsc --noEmit
npm run lint         # eslint src/
npm run test         # vitest run
npm run check        # typecheck + lint + test
npm run sync:zerotvm # one-off: pull zerotvm benchmark data
```

Deploy (staged): `npx opennextjs-cloudflare build && npx wrangler versions upload`
→ smoke-test the preview URL → `npx wrangler versions deploy` to promote.
Direct: `npx opennextjs-cloudflare build && npx wrangler deploy`.

## Cross-site context

`src/lib/sites.ts` is synced from `~/sites-shared/sites.ts`.

## Known gaps

- TS config has `strict: true` + `noUncheckedIndexedAccess: true`, so lookups
  like `SITES[CROSSLINKS.gpubench[0]]` need a non-null assertion (`!`) on the
  indexer — see `COMPANION_FLAGSHIP` in `page.tsx`.
- `src/lib/shader-gen.js` has DIVERGED from `~/sites-shared/shader-gen.js`
  (residual bindings on unfused Attn/FFN, parallel-fused scratch fix,
  `layerOffsets` export). Backport to sites-shared before running the sync
  script again, or the fixes get overwritten.
- The worker has an unused D1 binding (`DB`) provisioned for a possible
  future Postgres → D1 migration; the app still uses Neon Postgres.
