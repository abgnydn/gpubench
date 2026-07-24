/**
 * Transformer Fusion Benchmark — Multi-Config Sweep
 *
 * Tests fused vs unfused across:
 *   D_MODEL:  32, 64, 128
 *   Layers:   1, 2, 4
 *   SEQ_LEN:  64
 *   Heads:    2 (fixed)
 *   FFN:      4× D_MODEL
 *
 * Three baselines per config, all computing the SAME function on the SAME
 * packed weights (verified by a numerical equivalence check before timing):
 *   - unfused (decode):   per-token autoregressive loop, no KV cache,
 *                         one submit per layer — the worst-case dispatch pattern
 *   - unfused (1 submit): full-sequence forward, all 4·NL passes encoded into
 *                         a single command buffer — isolates dispatch overhead
 *                         from submit overhead
 *   - fused:              everything in one dispatch
 */

import {
  generateConfig, generateFusedShader, generateParallelFusedShader, generateF16FusedShader,
  generateUnfusedLN, generateUnfusedAttn, generateUnfusedFFN, layerOffsets
} from './shader-gen.js';

// Central config — change once, applies everywhere
export const BENCH = { WARMUP: 3, RUNS: 10 };

export async function initWebGPU() {
  if (!navigator.gpu) throw new Error('WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No GPU adapter found');
  const info = adapter.info || await adapter.requestAdapterInfo?.() || {};
  const features = [];
  if (adapter.features.has('shader-f16')) features.push('shader-f16');
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    }
  });
  return { device, info };
}

function randWeights(n, D) {
  const w = new Float32Array(n);
  const s = Math.sqrt(2.0 / D);
  for (let i = 0; i < n; i++) {
    const u1 = Math.random() || 1e-9;
    const u2 = Math.random();
    w[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * s;
  }
  return w;
}

function randEmb(SL, D) {
  const e = new Float32Array(SL * D);
  for (let i = 0; i < e.length; i++) e[i] = (Math.random() - 0.5) * 0.1;
  return e;
}

// One packed weight tensor shared by every variant, so fused/unfused/parallel
// are comparable AND checkable for numerical equivalence.
function makePackedWeights(cfg) {
  const { D, NL, perLayer, totalWeights } = cfg;
  const packed = randWeights(totalWeights, D);
  const o = layerOffsets(cfg);
  for (let l = 0; l < NL; l++) {
    const base = l * perLayer;
    for (let i = 0; i < D; i++) { packed[base + o.LG + i] = 1.0; }
    for (let i = 0; i < D; i++) { packed[base + o.LB + i] = 0.0; }
    for (let i = 0; i < D; i++) { packed[base + o.LG2 + i] = 1.0; }
    for (let i = 0; i < D; i++) { packed[base + o.LB2 + i] = 0.0; }
  }
  return packed;
}

function stats(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const mean = times.reduce((s, t) => s + t, 0) / times.length;
  const std = Math.sqrt(times.reduce((s, t) => s + (t - mean) ** 2, 0) / times.length);
  return { median, mean, std };
}

function toResult(times, SL, dispatches) {
  const { median, mean, std } = stats(times);
  return {
    median_ms: median, mean_ms: mean, std_ms: std,
    tokens_per_sec: (SL / median) * 1000,
    total_dispatches: dispatches, all_times: times, n: times.length,
  };
}

async function readBack(device, srcBuf, floatCount) {
  const rb = device.createBuffer({
    size: floatCount * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(srcBuf, 0, rb, 0, floatCount * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return data;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

// ========== UNFUSED (shared setup for decode / single-submit / verify) ==========
// Dataflow per layer, mirroring the fused kernel exactly:
//   xBuf  --LN1-->  lnBuf
//   lnBuf --Attn (residual: xBuf)-->  x2Buf        (x2 = x + Wo·attn(ln1))
//   x2Buf --LN2-->  lnBuf
//   lnBuf --FFN (residual: x2Buf)-->  xBuf         (x  = x2 + FFN(ln2))
// so xBuf holds the residual stream again at the start of the next layer.
function setupUnfused(device, cfg, packed, emb) {
  const { D, DF, SL, NL, perLayer } = cfg;
  const o = layerOffsets(cfg);

  const lnPipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code: generateUnfusedLN(cfg) }), entryPoint: 'main' }
  });
  const attnPipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code: generateUnfusedAttn(cfg) }), entryPoint: 'main' }
  });
  const ffnPipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code: generateUnfusedFFN(cfg) }), entryPoint: 'main' }
  });

  const bufSize = SL * D * 4;
  const buffers = [];
  const mk = (size, usage) => {
    const b = device.createBuffer({ size, usage });
    buffers.push(b);
    return b;
  };

  const xBuf = mk(bufSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const lnBuf = mk(bufSize, GPUBufferUsage.STORAGE);
  const x2Buf = mk(bufSize, GPUBufferUsage.STORAGE);
  const uniformBuf = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  // Per-layer weight buffers sliced from the same packed tensor the fused
  // kernel uses (the old version generated fresh random weights per buffer
  // and reused one set across all layers — not comparable, not verifiable).
  const upload = (slice) => {
    const b = mk(slice.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(b, 0, slice);
    return b;
  };

  const layers = [];
  for (let l = 0; l < NL; l++) {
    const base = l * perLayer;
    const slice = (off, len) => packed.subarray(base + off, base + off + len);
    const gamma1 = upload(slice(o.LG, D));
    const beta1 = upload(slice(o.LB, D));
    const wq = upload(slice(o.WQ, D * D));
    const wk = upload(slice(o.WK, D * D));
    const wv = upload(slice(o.WV, D * D));
    const wo = upload(slice(o.WO, D * D));
    const gamma2 = upload(slice(o.LG2, D));
    const beta2 = upload(slice(o.LB2, D));
    const w1 = upload(slice(o.W1, D * DF));
    const b1 = upload(slice(o.B1, DF));
    const w2 = upload(slice(o.W2, DF * D));
    const b2 = upload(slice(o.B2, D));

    const ln1BG = device.createBindGroup({
      layout: lnPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuf } },
        { binding: 1, resource: { buffer: lnBuf } },
        { binding: 2, resource: { buffer: gamma1 } },
        { binding: 3, resource: { buffer: beta1 } },
        { binding: 4, resource: { buffer: uniformBuf } },
      ]
    });
    const attnBG = device.createBindGroup({
      layout: attnPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: lnBuf } },
        { binding: 1, resource: { buffer: x2Buf } },
        { binding: 2, resource: { buffer: wq } },
        { binding: 3, resource: { buffer: wk } },
        { binding: 4, resource: { buffer: wv } },
        { binding: 5, resource: { buffer: wo } },
        { binding: 6, resource: { buffer: uniformBuf } },
        { binding: 7, resource: { buffer: xBuf } },
      ]
    });
    const ln2BG = device.createBindGroup({
      layout: lnPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: x2Buf } },
        { binding: 1, resource: { buffer: lnBuf } },
        { binding: 2, resource: { buffer: gamma2 } },
        { binding: 3, resource: { buffer: beta2 } },
        { binding: 4, resource: { buffer: uniformBuf } },
      ]
    });
    const ffnBG = device.createBindGroup({
      layout: ffnPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: lnBuf } },
        { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: w1 } },
        { binding: 3, resource: { buffer: b1 } },
        { binding: 4, resource: { buffer: w2 } },
        { binding: 5, resource: { buffer: b2 } },
        { binding: 6, resource: { buffer: uniformBuf } },
        { binding: 7, resource: { buffer: x2Buf } },
      ]
    });
    layers.push({ ln1BG, attnBG, ln2BG, ffnBG });
  }

  const wg = Math.ceil(SL / 64);

  const encodeLayer = (enc, lg) => {
    const p1 = enc.beginComputePass(); p1.setPipeline(lnPipe); p1.setBindGroup(0, lg.ln1BG); p1.dispatchWorkgroups(wg); p1.end();
    const p2 = enc.beginComputePass(); p2.setPipeline(attnPipe); p2.setBindGroup(0, lg.attnBG); p2.dispatchWorkgroups(wg); p2.end();
    const p3 = enc.beginComputePass(); p3.setPipeline(lnPipe); p3.setBindGroup(0, lg.ln2BG); p3.dispatchWorkgroups(wg); p3.end();
    const p4 = enc.beginComputePass(); p4.setPipeline(ffnPipe); p4.setBindGroup(0, lg.ffnBG); p4.dispatchWorkgroups(wg); p4.end();
  };

  return {
    // Restore the embedding into the residual stream. Runs are destructive
    // (xBuf is overwritten), so every timed run starts from the same state.
    async reset() {
      device.queue.writeBuffer(xBuf, 0, emb);
      await device.queue.onSubmittedWorkDone();
    },

    // Worst-case dispatch pattern: per-token autoregressive loop, no KV cache,
    // one submit per layer. O(SL²) recompute — mirrors naive uncached decode.
    async runDecode() {
      let totalDispatches = 0;
      for (let t = 1; t <= SL; t++) {
        device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([t, 0, 0, 0]));
        for (const lg of layers) {
          const enc = device.createCommandEncoder();
          encodeLayer(enc, lg);
          device.queue.submit([enc.finish()]);
          totalDispatches += 4;
        }
      }
      await device.queue.onSubmittedWorkDone();
      return totalDispatches;
    },

    // Fair baseline: same kernels, full-sequence forward, everything encoded
    // into ONE command buffer with ONE submit. 4·NL dispatches vs fused's 1.
    async runFullSequence() {
      device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([SL, 0, 0, 0]));
      const enc = device.createCommandEncoder();
      for (const lg of layers) encodeLayer(enc, lg);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      return 4 * NL;
    },

    readOutput: () => readBack(device, xBuf, SL * D),
    destroy() { for (const b of buffers) b.destroy(); },
  };
}

// ========== FUSED / PARALLEL / F16 (shared setup) ==========
function setupSingleKernel(device, cfg, packed, emb, code, extraOutFloats = 0) {
  const { D, SL, NL } = cfg;
  const pipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' }
  });

  const buffers = [];
  const mk = (size, usage) => {
    const b = device.createBuffer({ size, usage });
    buffers.push(b);
    return b;
  };

  const wBuf = mk(packed.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const eBuf = mk(SL * D * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const kvBuf = mk(NL * SL * D * 2 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const oBuf = mk((SL * D + extraOutFloats) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  const uBuf = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  device.queue.writeBuffer(wBuf, 0, packed);
  device.queue.writeBuffer(eBuf, 0, emb);
  device.queue.writeBuffer(uBuf, 0, new Uint32Array([SL, 0, 0, 0]));

  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wBuf } },
      { binding: 1, resource: { buffer: eBuf } },
      { binding: 2, resource: { buffer: kvBuf } },
      { binding: 3, resource: { buffer: oBuf } },
      { binding: 4, resource: { buffer: uBuf } },
    ]
  });

  return {
    async run() {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe); pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1);
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    },
    readOutput: () => readBack(device, oBuf, SL * D),
    destroy() { for (const b of buffers) b.destroy(); },
  };
}

async function timeRuns(runFn, warmup, runs, beforeEach) {
  for (let i = 0; i < warmup; i++) {
    if (beforeEach) await beforeEach();
    await runFn();
  }
  const times = [];
  for (let i = 0; i < runs; i++) {
    if (beforeEach) await beforeEach();
    const t0 = performance.now();
    await runFn();
    times.push(performance.now() - t0);
  }
  return times;
}

// ========== EQUIVALENCE CHECK ==========
// Runs each variant once on the same weights + embeddings and compares outputs.
// This is what makes the speedup numbers meaningful: without it, "fused vs
// unfused" could be comparing kernels that compute different functions
// (which is exactly what an earlier version of this benchmark did).
export async function verifyEquivalence(device, cfg, packed, emb) {
  const { DF } = cfg;

  const fused = setupSingleKernel(device, cfg, packed, emb, generateFusedShader(cfg));
  await fused.run();
  const fusedOut = await fused.readOutput();
  fused.destroy();

  const unfused = setupUnfused(device, cfg, packed, emb);
  await unfused.reset();
  await unfused.runFullSequence();
  const unfusedOut = await unfused.readOutput();
  unfused.destroy();

  const parallel = setupSingleKernel(device, cfg, packed, emb, generateParallelFusedShader(cfg, 64), DF);
  await parallel.run();
  const parallelOut = await parallel.readOutput();
  parallel.destroy();

  return {
    unfusedDiff: maxAbsDiff(fusedOut, unfusedOut),
    parallelDiff: maxAbsDiff(fusedOut, parallelOut),
    sample: fusedOut[0],
  };
}

// Absolute tolerance for f32 kernels that differ only in summation order.
export const EQUIV_TOLERANCE = 1e-2;

// ========== BENCHES ==========
async function benchUnfused(setup, SL, mode, warmup = BENCH.WARMUP, runs = BENCH.RUNS) {
  const runFn = mode === 'decode' ? () => setup.runDecode() : () => setup.runFullSequence();
  let dispatches = 0;
  const times = await timeRuns(async () => { dispatches = await runFn(); }, warmup, runs, () => setup.reset());
  return toResult(times, SL, dispatches);
}

async function benchSingleKernel(device, cfg, packed, emb, code, extraOutFloats = 0, warmup = BENCH.WARMUP, runs = BENCH.RUNS) {
  const setup = setupSingleKernel(device, cfg, packed, emb, code, extraOutFloats);
  try {
    const times = await timeRuns(() => setup.run(), warmup, runs);
    return toResult(times, cfg.SL, 1);
  } finally {
    setup.destroy();
  }
}

// ========== SWEEP ==========
// Main sweep: D × Layers
export const CONFIGS = [
  { D: 32,  heads: 2, ffn: 4, seq: 64, layers: 1, label: 'D=32, L=1' },
  { D: 32,  heads: 2, ffn: 4, seq: 64, layers: 4, label: 'D=32, L=4' },
  { D: 64,  heads: 2, ffn: 4, seq: 64, layers: 1, label: 'D=64, L=1' },
  { D: 64,  heads: 2, ffn: 4, seq: 64, layers: 4, label: 'D=64, L=4' },
  { D: 128, heads: 2, ffn: 4, seq: 64, layers: 1, label: 'D=128, L=1' },
  { D: 128, heads: 2, ffn: 4, seq: 64, layers: 4, label: 'D=128, L=4' },
];

// Sequence length scaling: fixed D=32, L=1, vary SEQ
export const SEQ_CONFIGS = [
  { D: 32, heads: 2, ffn: 4, seq: 16,  layers: 1, label: 'SEQ=16' },
  { D: 32, heads: 2, ffn: 4, seq: 64,  layers: 1, label: 'SEQ=64' },
  { D: 32, heads: 2, ffn: 4, seq: 128, layers: 1, label: 'SEQ=128' },
];

export async function runSweep(log, onResult) {
  return runSweepWithConfigs(CONFIGS, log, onResult);
}

export async function runSweepWithConfigs(configs, log, onResult) {
  log('Initializing WebGPU...');
  const { device, info } = await initWebGPU();
  const gpuName = `${info.vendor || '?'} ${info.architecture || ''} — ${info.device || info.description || 'detected'}`;
  log(`GPU: ${gpuName}`);
  log('All variants share one weight tensor; outputs are cross-checked before timing.');
  log('Speedups are computed from MEDIAN times (means are skewed by browser stalls).\n');

  const results = [];

  for (const c of configs) {
    const cfg = generateConfig(c.D, c.heads, c.ffn, c.seq, c.layers);
    const dispatches = 4 * c.seq * c.layers;

    log(`--- ${c.label} (${dispatches} dispatches vs 1) ---`);

    let unfusedSetup = null;
    try {
      const packed = makePackedWeights(cfg);
      const emb = randEmb(cfg.SL, cfg.D);

      log('  Verifying all variants compute the same output...');
      const equiv = await verifyEquivalence(device, cfg, packed, emb);
      const equivOk = equiv.unfusedDiff <= EQUIV_TOLERANCE && equiv.parallelDiff <= EQUIV_TOLERANCE;
      log(`  max|Δ| vs fused: unfused ${equiv.unfusedDiff.toExponential(1)}, parallel ${equiv.parallelDiff.toExponential(1)} — ${equivOk ? 'OK' : 'MISMATCH (speedups below are NOT comparable)'}`);

      unfusedSetup = setupUnfused(device, cfg, packed, emb);

      log(`  Unfused decode: per-token, no KV cache, 1 submit/layer (N=${BENCH.RUNS})...`);
      const unfused = await benchUnfused(unfusedSetup, cfg.SL, 'decode');
      log(`  ${unfused.median_ms.toFixed(1)} ms median (${unfused.mean_ms.toFixed(1)} ± ${unfused.std_ms.toFixed(1)}) | ${unfused.tokens_per_sec.toFixed(0)} tok/s`);

      log(`  Unfused 1-submit: full sequence, ${4 * c.layers} dispatches, one submit (N=${BENCH.RUNS})...`);
      const unfusedBatched = await benchUnfused(unfusedSetup, cfg.SL, 'full');
      log(`  ${unfusedBatched.median_ms.toFixed(1)} ms median (${unfusedBatched.mean_ms.toFixed(1)} ± ${unfusedBatched.std_ms.toFixed(1)}) | ${unfusedBatched.tokens_per_sec.toFixed(0)} tok/s`);

      unfusedSetup.destroy();
      unfusedSetup = null;

      log(`  Fused-1T (N=${BENCH.RUNS})...`);
      const fused = await benchSingleKernel(device, cfg, packed, emb, generateFusedShader(cfg));
      log(`  ${fused.median_ms.toFixed(1)} ms median (${fused.mean_ms.toFixed(1)} ± ${fused.std_ms.toFixed(1)}) | ${fused.tokens_per_sec.toFixed(0)} tok/s`);

      log(`  Fused-parallel (N=${BENCH.RUNS})...`);
      const parallel = await benchSingleKernel(device, cfg, packed, emb, generateParallelFusedShader(cfg, 64), cfg.DF);
      log(`  ${parallel.median_ms.toFixed(1)} ms median (${parallel.mean_ms.toFixed(1)} ± ${parallel.std_ms.toFixed(1)}) | ${parallel.tokens_per_sec.toFixed(0)} tok/s`);

      let f16 = null;
      if (device.features.has('shader-f16')) {
        log(`  Fused-f16 (N=${BENCH.RUNS})...`);
        f16 = await benchSingleKernel(device, cfg, packed, emb, generateF16FusedShader(cfg));
        log(`  ${f16.median_ms.toFixed(1)} ms median (${f16.mean_ms.toFixed(1)} ± ${f16.std_ms.toFixed(1)}) | ${f16.tokens_per_sec.toFixed(0)} tok/s`);
      } else {
        log(`  f16: not supported on this GPU`);
      }

      const speedup = unfused.median_ms / fused.median_ms;
      const batchedSpeedup = unfusedBatched.median_ms / fused.median_ms;
      const parSpeedup = unfused.median_ms / parallel.median_ms;
      log(`  Speedup (median): fused-1T ${speedup.toFixed(1)}× vs decode | ${batchedSpeedup.toFixed(1)}× vs 1-submit | parallel ${parSpeedup.toFixed(1)}×${f16 ? ' | f16 ' + (unfused.median_ms / f16.median_ms).toFixed(1) + '×' : ''}\n`);

      const row = {
        ...c, unfused, unfusedBatched, fused, parallel, f16,
        speedup, parSpeedup, batchedSpeedup,
        equivUnfusedDiff: equiv.unfusedDiff, equivParallelDiff: equiv.parallelDiff, equivOk,
        dispatches,
      };
      results.push(row);
      if (onResult) onResult(row);
    } catch (e) {
      log(`  ERROR: ${e.message}\n`);
      results.push({ ...c, error: e.message });
      if (onResult) onResult({ ...c, error: e.message });
    } finally {
      if (unfusedSetup) unfusedSetup.destroy();
    }
  }

  // Sequence length scaling (skipped in public benchmark)
  const seqResults = [];

  log(`\n========== FULL RESULTS (N=${BENCH.RUNS}, median [mean ± std]) ==========\n`);
  log('Config            | Dispatches | Unfused decode (ms)  | Unfused 1-submit (ms)| Fused-1T (ms)        | Parallel (ms)        | vs decode | vs 1-submit');
  log('------------------|------------|----------------------|----------------------|----------------------|----------------------|-----------|------------');
  for (const r of results) {
    if (r.error) {
      log(`${r.label.padEnd(17)} | ${String(r.dispatches || '?').padStart(10)} | ERROR: ${r.error}`);
    } else {
      const fmt = (b) => `${b.median_ms.toFixed(1)} [${b.mean_ms.toFixed(1)} ± ${b.std_ms.toFixed(1)}]`;
      log(`${r.label.padEnd(17)} | ${String(r.dispatches).padStart(10)} | ${fmt(r.unfused).padStart(20)} | ${fmt(r.unfusedBatched).padStart(20)} | ${fmt(r.fused).padStart(20)} | ${fmt(r.parallel).padStart(20)} | ${(r.speedup.toFixed(1) + '×').padStart(9)} | ${(r.batchedSpeedup.toFixed(1) + '×').padStart(10)}`);
    }
  }

  // F16 results if available
  const f16Results = results.filter(r => r.f16 && !r.f16.error);
  if (f16Results.length > 0) {
    log('\nF16 Results       | Fused-f32 (ms)       | Fused-f16 (ms)       | f16 speedup');
    log('------------------|----------------------|----------------------|-----------');
    for (const r of f16Results) {
      const ff = `${r.fused.median_ms.toFixed(1)} ± ${r.fused.std_ms.toFixed(1)}`;
      const f16f = `${r.f16.median_ms.toFixed(1)} ± ${r.f16.std_ms.toFixed(1)}`;
      const f16sp = (r.fused.median_ms / r.f16.median_ms).toFixed(2);
      log(`${r.label.padEnd(17)} | ${ff.padStart(20)} | ${f16f.padStart(20)} | ${f16sp}×`);
    }
  }

  if (seqResults.length > 0) {
    log('\nSEQ Scaling       | Dispatches | Unfused (ms)         | Fused (ms)           | Speedup');
    log('------------------|------------|----------------------|----------------------|--------');
    for (const r of seqResults) {
      if (r.error) continue;
      const uf = `${r.unfused.median_ms.toFixed(1)} ± ${r.unfused.std_ms.toFixed(1)}`;
      const ff = `${r.fused.median_ms.toFixed(1)} ± ${r.fused.std_ms.toFixed(1)}`;
      log(`${r.label.padEnd(17)} | ${String(r.dispatches).padStart(10)} | ${uf.padStart(20)} | ${ff.padStart(20)} | ${r.speedup.toFixed(1)}×`);
    }
  }

  return { results, seqResults, gpuName };
}
