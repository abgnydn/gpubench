# gpubench.dev

## Goal

Browser-native WebGPU benchmark suite with public, unfiltered results from
every device that runs it. The measurement arm of the kernel-fusion research
line — the numbers the preprints cite live in this DB. Every result published,
no cherry-picking.

## Architecture

Next.js 16 with App Router + `/api/*` serverless functions backed by Vercel
Postgres. **This is the only non-static site** in the constellation — it
can't deploy to Cloudflare Pages because it needs server runtime for
`/api/device`, `/api/results`, `/api/setup`, `/api/transformer-results`.
Deploys to Vercel.

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

Deploy: `node ~/sites-shared/deploy.mjs gpubench` → `vercel deploy --prod`.

## Cross-site context

`src/lib/sites.ts` is synced from `~/sites-shared/sites.ts`.

## Known gaps

- TS config has `strict: true` + `noUncheckedIndexedAccess: true`, so lookups
  like `SITES[CROSSLINKS.gpubench[0]]` need a non-null assertion (`!`) on the
  indexer — see `COMPANION_FLAGSHIP` in `page.tsx`.
- `src/lib/shader-gen.js` has two pre-existing unused-variable lint errors
  (same file as kernelfusion — shared via the sync).
- Several `tests/transformer-*.test.*` files are empty placeholders that
  vitest reports as "no test suite found". Harmless but noisy.
- Not yet migrated to Cloudflare Pages — the only site still on Vercel
  because of `/api/*`. Future: port endpoints to CF Workers + D1 for a full
  constellation migration.
