#!/usr/bin/env node
/**
 * Mirror the current zerotvm.com chat bundle into public/demos/zerotvm/
 * and wire it into the gpubench telemetry + embed-mode conventions.
 *
 * Why this exists: we want the /zerotvm tab's chat UI to auto-track
 * whatever zerotvm.com ships (new copy, new numbers, new kernels),
 * but we also want Zero-TVM decode tok/s to flow into gpubench's
 * /results leaderboard. Upstream dropped the `reportDevice` call in
 * the Apr 2026 bundle and doesn't have an `?embed=1` flag, so we
 * can't iframe zerotvm.com directly without losing both.
 *
 * The script:
 *   1. Fetches the upstream HTML + every asset it references
 *   2. Writes them under public/demos/zerotvm/ (same-origin → our
 *      telemetry POST to /api/device resolves to gpubench)
 *   3. Injects an embed-mode block into <head> that:
 *        - adds `html.embed` class pre-paint when ?embed=1
 *        - hides upstream's own header/back-link/footer when embedded
 *          (the outer Next page already owns that chrome)
 *        - loads /demos/device-telemetry.js (defines window.reportDevice)
 *        - MutationObserves #stats and calls reportDevice whenever
 *          upstream writes "N tok/s" into it — fallback for the
 *          dropped telemetry call in the main bundle
 *
 * Run: `node scripts/sync-zerotvm.mjs`
 * Or:  `npm run sync:zerotvm`
 *
 * Idempotent: wipes OUT_DIR before each run. Safe to run repeatedly;
 * review `git diff public/demos/zerotvm/` before committing.
 */

import { mkdir, writeFile, rm } from "node:fs/promises";
import { join, dirname } from "node:path";

const UPSTREAM = "https://zerotvm.com";
const ENTRY = "/zero-tvm.html";
const OUT_DIR = "public/demos/zerotvm";

async function fetchText(path) {
  const url = UPSTREAM + path;
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.text();
}

async function fetchBuf(path) {
  const url = UPSTREAM + path;
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

/**
 * Collect every absolute-path reference inside src="..." and href="...".
 * Exclude the entry HTML itself and navigation links (roots, .html files
 * other than the entry) — those will be rewritten to upstream URLs in
 * rewriteHtml() rather than fetched locally, so "back to zerotvm.com"
 * keeps working.
 */
function collectAssets(html) {
  const rx = /(?:src|href)="(\/[^"#?]+)"/g;
  const out = new Set();
  for (const m of html.matchAll(rx)) {
    const p = m[1];
    if (p === "/") continue; // home link — leave for upstream
    if (p.endsWith(".html")) continue; // other zerotvm pages — leave for upstream
    out.add(p);
  }
  return out;
}

/**
 * Rewrite paths:
 *   - Known local assets  → /demos/zerotvm<path>      (served from our origin)
 *   - Other absolute paths → https://zerotvm.com<path> (keep upstream links live)
 *   - relative / data / #anchors → untouched
 */
function rewriteHtml(html, downloaded) {
  return html.replace(
    /(src|href)="(\/[^"#]*)"/g,
    (_full, attr, path) => {
      if (downloaded.has(path)) {
        return `${attr}="/demos/zerotvm${path}"`;
      }
      return `${attr}="${UPSTREAM}${path}"`;
    },
  );
}

/**
 * Block that lives inside <head>. Order matters:
 *   1. pre-paint class toggle (before CSS parse → no flash)
 *   2. embed-mode CSS
 *   3. telemetry shim (defines window.reportDevice)
 *   4. stats-scraper (calls reportDevice when #stats updates)
 *
 * The stats-scraper MutationObserver is intentionally tolerant: it
 * matches any "<number> tok/s" in the text, throttles to once every
 * 5s via the shim's own rate-limiter, and bails cleanly if #stats
 * doesn't exist (upstream may rename the element in the future).
 */
const EMBED_HEAD = `
  <!-- === EMBED MODE (injected by scripts/sync-zerotvm.mjs) === -->
  <script>
    // Pre-paint class toggle — no flash of upstream chrome when we
    // embed this file inside the /zerotvm Next page.
    if (new URLSearchParams(location.search).get("embed") === "1") {
      document.documentElement.classList.add("embed");
    }
  </script>
  <style>
    /* Upstream's own header + back-link + footer duplicate the
       gpubench nav when embedded. Hide them, but keep #stats and
       #badge so the user still sees load + tok/s feedback. */
    html.embed .back-link,
    html.embed .model-info,
    html.embed .header-spacer,
    html.embed footer,
    html.embed .app-footer { display: none !important; }
    html.embed .app-header { justify-content: flex-end !important; }
    html.embed,
    html.embed body { min-height: 100% !important; height: 100%; }
  </style>
  <!-- gpubench telemetry: defines window.reportDevice(workload,fitness,gen,speed). -->
  <script src="/demos/device-telemetry.js"></script>
  <script>
    // Fallback telemetry wiring. Upstream's Apr 2026 bundle removed
    // the direct reportDevice() call, so we MutationObserve the
    // #stats element (which the chat populates with "N tok/s · …"
    // during decode) and forward whatever number it shows.
    // device-telemetry.js rate-limits at 5s, so we don't flood.
    window.addEventListener("DOMContentLoaded", () => {
      const stats = document.getElementById("stats");
      if (!stats || typeof window.reportDevice !== "function") return;
      const obs = new MutationObserver(() => {
        const m = /([\\d.]+)\\s*tok\\/s/i.exec(stats.textContent || "");
        if (!m) return;
        const speed = Number(m[1]);
        if (!Number.isFinite(speed) || speed <= 0) return;
        window.reportDevice("Zero-TVM Phi-3-mini", null, null, speed);
      });
      obs.observe(stats, { childList: true, characterData: true, subtree: true });
    });
  </script>
  <!-- === /EMBED MODE === -->
`;

function injectHead(html) {
  // Most reliable anchor — every upstream HTML has a closing </head>.
  if (!html.includes("</head>")) {
    throw new Error("upstream HTML missing </head> — cannot inject embed block");
  }
  return html.replace("</head>", EMBED_HEAD + "\n</head>");
}

async function main() {
  console.log(`→ Fetching ${UPSTREAM}${ENTRY}`);
  const html = await fetchText(ENTRY);

  const assets = collectAssets(html);
  console.log(`→ ${assets.size} assets referenced`);

  // Fresh snapshot each run — if upstream renames or drops a file,
  // we don't want a stale copy lingering under public/demos/zerotvm/.
  await rm(OUT_DIR, { recursive: true, force: true });
  await mkdir(OUT_DIR, { recursive: true });

  let totalBytes = 0;
  for (const path of assets) {
    try {
      const buf = await fetchBuf(path);
      const dest = join(OUT_DIR, path); // e.g. public/demos/zerotvm/assets/foo.js
      await mkdir(dirname(dest), { recursive: true });
      await writeFile(dest, buf);
      totalBytes += buf.length;
      console.log(`  ✓ ${path.padEnd(50)} ${(buf.length / 1024).toFixed(1)} KB`);
    } catch (e) {
      console.error(`  ✗ ${path}: ${e.message}`);
      throw e;
    }
  }

  const rewritten = rewriteHtml(html, assets);
  const finalHtml = injectHead(rewritten);
  await writeFile(join(OUT_DIR, "zero-tvm.html"), finalHtml);

  console.log(
    `→ Wrote ${assets.size + 1} files, ${(totalBytes / 1024 / 1024).toFixed(2)} MB total`,
  );
  console.log(`→ Chat served at /demos/zerotvm/zero-tvm.html?embed=1`);
  console.log(`\nReview \`git diff public/demos/zerotvm/\` before committing.`);
}

main().catch((err) => {
  console.error("\nsync failed:", err);
  process.exit(1);
});
