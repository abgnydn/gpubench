/**
 * Transformer Fusion Benchmark — Multi-Config Sweep
 *
 * Tests fused vs unfused across:
 *   D_MODEL:  32, 64, 128
 *   Layers:   1, 2, 4
 *   SEQ_LEN:  64
 *   Heads:    2 (fixed)
 *   FFN:      4× D_MODEL
 */

import {
  generateConfig, generateFusedShader, generateParallelFusedShader, generateF16FusedShader,
  generateUnfusedLN, generateUnfusedAttn, generateUnfusedFFN
} from './shader-gen.js';

// Central config — change once, applies everywhere
export const BENCH = { WARMUP: 3, RUNS: 10 };  // Reduced for exploratory run with D=256/512

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

// ========== UNFUSED ==========
async function benchUnfused(device, cfg, warmup = BENCH.WARMUP, runs = BENCH.RUNS) {
  const { D, DF, SL, NL } = cfg;
  const lnCode = generateUnfusedLN(cfg);
  const attnCode = generateUnfusedAttn(cfg);
  const ffnCode = generateUnfusedFFN(cfg);

  const lnPipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code: lnCode }), entryPoint: 'main' }
  });
  const attnPipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code: attnCode }), entryPoint: 'main' }
  });
  const ffnPipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code: ffnCode }), entryPoint: 'main' }
  });

  const bufSize = SL * D * 4;
  const wSize = D * D * 4;

  const inputBuf = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuf = device.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const gammaBuf = device.createBuffer({ size: D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const betaBuf = device.createBuffer({ size: D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const wqBuf = device.createBuffer({ size: wSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const wkBuf = device.createBuffer({ size: wSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const wvBuf = device.createBuffer({ size: wSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const woBuf = device.createBuffer({ size: wSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const w1Buf = device.createBuffer({ size: D * DF * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const b1Buf = device.createBuffer({ size: DF * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const w2Buf = device.createBuffer({ size: DF * D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const b2Buf = device.createBuffer({ size: D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const uniformBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const emb = randEmb(SL, D);
  device.queue.writeBuffer(inputBuf, 0, emb);
  device.queue.writeBuffer(gammaBuf, 0, new Float32Array(D).fill(1.0));
  device.queue.writeBuffer(betaBuf, 0, new Float32Array(D).fill(0.0));
  device.queue.writeBuffer(wqBuf, 0, randWeights(D * D, D));
  device.queue.writeBuffer(wkBuf, 0, randWeights(D * D, D));
  device.queue.writeBuffer(wvBuf, 0, randWeights(D * D, D));
  device.queue.writeBuffer(woBuf, 0, randWeights(D * D, D));
  device.queue.writeBuffer(w1Buf, 0, randWeights(D * DF, D));
  device.queue.writeBuffer(b1Buf, 0, new Float32Array(DF).fill(0.0));
  device.queue.writeBuffer(w2Buf, 0, randWeights(DF * D, D));
  device.queue.writeBuffer(b2Buf, 0, new Float32Array(D).fill(0.0));

  const lnBG = device.createBindGroup({
    layout: lnPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: gammaBuf } },
      { binding: 3, resource: { buffer: betaBuf } },
      { binding: 4, resource: { buffer: uniformBuf } },
    ]
  });
  const attnBG = device.createBindGroup({
    layout: attnPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outputBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: wqBuf } },
      { binding: 3, resource: { buffer: wkBuf } },
      { binding: 4, resource: { buffer: wvBuf } },
      { binding: 5, resource: { buffer: woBuf } },
      { binding: 6, resource: { buffer: uniformBuf } },
    ]
  });
  const ffnBG = device.createBindGroup({
    layout: ffnPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: w1Buf } },
      { binding: 3, resource: { buffer: b1Buf } },
      { binding: 4, resource: { buffer: w2Buf } },
      { binding: 5, resource: { buffer: b2Buf } },
      { binding: 6, resource: { buffer: uniformBuf } },
    ]
  });

  const wg = Math.ceil(SL / 64);

  async function run() {
    let totalDispatches = 0;
    for (let t = 1; t <= SL; t++) {
      device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([t, 0, 0, 0]));
      for (let layer = 0; layer < NL; layer++) {
        const enc = device.createCommandEncoder();
        const p1 = enc.beginComputePass(); p1.setPipeline(lnPipe); p1.setBindGroup(0, lnBG); p1.dispatchWorkgroups(wg); p1.end();
        const p2 = enc.beginComputePass(); p2.setPipeline(attnPipe); p2.setBindGroup(0, attnBG); p2.dispatchWorkgroups(wg); p2.end();
        const p3 = enc.beginComputePass(); p3.setPipeline(lnPipe); p3.setBindGroup(0, lnBG); p3.dispatchWorkgroups(wg); p3.end();
        const p4 = enc.beginComputePass(); p4.setPipeline(ffnPipe); p4.setBindGroup(0, ffnBG); p4.dispatchWorkgroups(wg); p4.end();
        device.queue.submit([enc.finish()]);
        totalDispatches += 4;
      }
    }
    await device.queue.onSubmittedWorkDone();
    return totalDispatches;
  }

  for (let i = 0; i < warmup; i++) await run();
  const times = [];
  let dispatches = 0;
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    dispatches = await run();
    times.push(performance.now() - t0);
  }
  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const mean = times.reduce((s, t) => s + t, 0) / times.length;
  const std = Math.sqrt(times.reduce((s, t) => s + (t - mean) ** 2, 0) / times.length);
  return { median_ms: median, mean_ms: mean, std_ms: std, tokens_per_sec: (SL / median) * 1000, total_dispatches: dispatches, all_times: times, n: times.length };
}

// ========== FUSED ==========
async function benchFused(device, cfg, warmup = BENCH.WARMUP, runs = BENCH.RUNS) {
  const { D, SL, NL, totalWeights, perLayer } = cfg;
  const code = generateFusedShader(cfg);

  const pipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' }
  });

  const wBuf = device.createBuffer({ size: totalWeights * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const eBuf = device.createBuffer({ size: SL * D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const kvBuf = device.createBuffer({ size: NL * SL * D * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const oBuf = device.createBuffer({ size: SL * D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const uBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const packed = randWeights(totalWeights, D);
  // Set LN gammas=1, betas=0 for each layer
  for (let l = 0; l < NL; l++) {
    const base = l * perLayer;
    for (let i = 0; i < D; i++) { packed[base + i] = 1.0; }                     // ln1_gamma
    for (let i = 0; i < D; i++) { packed[base + D + i] = 0.0; }                 // ln1_beta
    const ln2gOff = 4 * D * D + 2 * D;
    for (let i = 0; i < D; i++) { packed[base + ln2gOff + i] = 1.0; }           // ln2_gamma
    for (let i = 0; i < D; i++) { packed[base + ln2gOff + D + i] = 0.0; }       // ln2_beta
  }

  device.queue.writeBuffer(wBuf, 0, packed);
  device.queue.writeBuffer(eBuf, 0, randEmb(SL, D));
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

  async function run() {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  for (let i = 0; i < warmup; i++) await run();
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await run();
    times.push(performance.now() - t0);
  }
  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const mean = times.reduce((s, t) => s + t, 0) / times.length;
  const std = Math.sqrt(times.reduce((s, t) => s + (t - mean) ** 2, 0) / times.length);
  return { median_ms: median, mean_ms: mean, std_ms: std, tokens_per_sec: (SL / median) * 1000, total_dispatches: 1, all_times: times, n: times.length };
}

// ========== PARALLEL FUSED ==========
async function benchParallelFused(device, cfg, warmup = BENCH.WARMUP, runs = BENCH.RUNS) {
  const { D, SL, NL, DF, totalWeights, perLayer } = cfg;
  const code = generateParallelFusedShader(cfg, 64);

  const pipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' }
  });

  const wBuf = device.createBuffer({ size: totalWeights * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const eBuf = device.createBuffer({ size: SL * D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const kvBuf = device.createBuffer({ size: NL * SL * D * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  // Output buffer needs to be large enough for FFN hidden (DF) temp storage too
  const oBufSize = Math.max(SL * D, DF) * 4;
  const oBuf = device.createBuffer({ size: oBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const uBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const packed = randWeights(totalWeights, D);
  for (let l = 0; l < NL; l++) {
    const base = l * perLayer;
    for (let i = 0; i < D; i++) { packed[base + i] = 1.0; }
    for (let i = 0; i < D; i++) { packed[base + D + i] = 0.0; }
    const ln2gOff = 4 * D * D + 2 * D;
    for (let i = 0; i < D; i++) { packed[base + ln2gOff + i] = 1.0; }
    for (let i = 0; i < D; i++) { packed[base + ln2gOff + D + i] = 0.0; }
  }

  device.queue.writeBuffer(wBuf, 0, packed);
  device.queue.writeBuffer(eBuf, 0, randEmb(SL, D));
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

  async function run() {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);  // 1 workgroup of 64 threads
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  for (let i = 0; i < warmup; i++) await run();
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await run();
    times.push(performance.now() - t0);
  }
  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const mean = times.reduce((s, t) => s + t, 0) / times.length;
  const std = Math.sqrt(times.reduce((s, t) => s + (t - mean) ** 2, 0) / times.length);
  return { median_ms: median, mean_ms: mean, std_ms: std, tokens_per_sec: (SL / median) * 1000, total_dispatches: 1, all_times: times, n: times.length };
}

// ========== F16 FUSED ==========
async function benchF16Fused(device, cfg, warmup = BENCH.WARMUP, runs = BENCH.RUNS) {
  const { D, SL, NL, totalWeights, perLayer } = cfg;

  // Check f16 support
  if (!device.features.has('shader-f16')) {
    return { error: 'shader-f16 not supported' };
  }

  const code = generateF16FusedShader(cfg);
  const pipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' }
  });

  const wBuf = device.createBuffer({ size: totalWeights * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const eBuf = device.createBuffer({ size: SL * D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const kvBuf = device.createBuffer({ size: NL * SL * D * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const oBuf = device.createBuffer({ size: SL * D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const uBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const packed = randWeights(totalWeights, D);
  for (let l = 0; l < NL; l++) {
    const base = l * perLayer;
    for (let i = 0; i < D; i++) { packed[base + i] = 1.0; }
    for (let i = 0; i < D; i++) { packed[base + D + i] = 0.0; }
    const ln2gOff = 4 * D * D + 2 * D;
    for (let i = 0; i < D; i++) { packed[base + ln2gOff + i] = 1.0; }
    for (let i = 0; i < D; i++) { packed[base + ln2gOff + D + i] = 0.0; }
  }

  device.queue.writeBuffer(wBuf, 0, packed);
  device.queue.writeBuffer(eBuf, 0, randEmb(SL, D));
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

  async function run() {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  for (let i = 0; i < warmup; i++) await run();
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await run();
    times.push(performance.now() - t0);
  }
  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const mean = times.reduce((s, t) => s + t, 0) / times.length;
  const std = Math.sqrt(times.reduce((s, t) => s + (t - mean) ** 2, 0) / times.length);
  return { median_ms: median, mean_ms: mean, std_ms: std, tokens_per_sec: (SL / median) * 1000, total_dispatches: 1, all_times: times, n: times.length };
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
  log(`GPU: ${gpuName}\n`);

  const results = [];

  for (const c of configs) {
    const cfg = generateConfig(c.D, c.heads, c.ffn, c.seq, c.layers);
    const dispatches = 4 * c.seq * c.layers;

    log(`--- ${c.label} (${dispatches} dispatches vs 1) ---`);

    try {
      log(`  Unfused (N=${BENCH.RUNS})...`);
      const unfused = await benchUnfused(device, cfg);
      log(`  ${unfused.mean_ms.toFixed(1)} ± ${unfused.std_ms.toFixed(1)} ms | ${unfused.tokens_per_sec.toFixed(0)} tok/s`);

      log(`  Fused-1T (N=${BENCH.RUNS})...`);
      const fused = await benchFused(device, cfg);
      log(`  ${fused.mean_ms.toFixed(1)} ± ${fused.std_ms.toFixed(1)} ms | ${fused.tokens_per_sec.toFixed(0)} tok/s`);

      log(`  Fused-parallel (N=${BENCH.RUNS})...`);
      const parallel = await benchParallelFused(device, cfg);
      log(`  ${parallel.mean_ms.toFixed(1)} ± ${parallel.std_ms.toFixed(1)} ms | ${parallel.tokens_per_sec.toFixed(0)} tok/s`);

      let f16 = null;
      if (device.features.has('shader-f16')) {
        log(`  Fused-f16 (N=${BENCH.RUNS})...`);
        f16 = await benchF16Fused(device, cfg);
        if (f16.error) {
          log(`  f16: ${f16.error}`);
          f16 = null;
        } else {
          log(`  ${f16.mean_ms.toFixed(1)} ± ${f16.std_ms.toFixed(1)} ms | ${f16.tokens_per_sec.toFixed(0)} tok/s`);
        }
      } else {
        log(`  f16: not supported on this GPU`);
      }

      const speedup = unfused.mean_ms / fused.mean_ms;
      const parSpeedup = unfused.mean_ms / parallel.mean_ms;
      log(`  Speedup: fused-1T ${speedup.toFixed(1)}× | parallel ${parSpeedup.toFixed(1)}×${f16 ? ' | f16 ' + (unfused.mean_ms / f16.mean_ms).toFixed(1) + '×' : ''}\n`);

      const row = { ...c, unfused, fused, parallel, f16, speedup, parSpeedup, dispatches };
      results.push(row);
      if (onResult) onResult(row);
    } catch (e) {
      log(`  ERROR: ${e.message}\n`);
      results.push({ ...c, error: e.message });
      if (onResult) onResult({ ...c, error: e.message });
    }
  }

  // Sequence length scaling (skipped in public benchmark)
  const seqResults = [];

  log(`\n========== FULL RESULTS (N=${BENCH.RUNS}, mean ± std) ==========\n`);
  log('Config            | Dispatches | Unfused (ms)         | Fused-1T (ms)        | Parallel (ms)        | Fused-1T  | Parallel');
  log('------------------|------------|----------------------|----------------------|----------------------|-----------|--------');
  for (const r of results) {
    if (r.error) {
      log(`${r.label.padEnd(17)} | ${String(r.dispatches || '?').padStart(10)} | ERROR: ${r.error}`);
    } else {
      const uf = `${r.unfused.mean_ms.toFixed(1)} ± ${r.unfused.std_ms.toFixed(1)}`;
      const ff = `${r.fused.mean_ms.toFixed(1)} ± ${r.fused.std_ms.toFixed(1)}`;
      const pf = r.parallel ? `${r.parallel.mean_ms.toFixed(1)} ± ${r.parallel.std_ms.toFixed(1)}` : 'N/A';
      log(`${r.label.padEnd(17)} | ${String(r.dispatches).padStart(10)} | ${uf.padStart(20)} | ${ff.padStart(20)} | ${pf.padStart(20)} | ${r.speedup.toFixed(1)}×`.padEnd(9) + ` | ${r.parSpeedup.toFixed(1)}×`);
    }
  }

  // F16 results if available
  const f16Results = results.filter(r => r.f16 && !r.f16.error);
  if (f16Results.length > 0) {
    log('\nF16 Results       | Fused-f32 (ms)       | Fused-f16 (ms)       | f16 speedup');
    log('------------------|----------------------|----------------------|-----------');
    for (const r of f16Results) {
      const ff = `${r.fused.mean_ms.toFixed(1)} ± ${r.fused.std_ms.toFixed(1)}`;
      const f16f = `${r.f16.mean_ms.toFixed(1)} ± ${r.f16.std_ms.toFixed(1)}`;
      const f16sp = (r.fused.mean_ms / r.f16.mean_ms).toFixed(2);
      log(`${r.label.padEnd(17)} | ${ff.padStart(20)} | ${f16f.padStart(20)} | ${f16sp}×`);
    }
  }

  if (seqResults.length > 0) {
    log('\nSEQ Scaling       | Dispatches | Unfused (ms)         | Fused (ms)           | Speedup');
    log('------------------|------------|----------------------|----------------------|--------');
    for (const r of seqResults) {
      if (r.error) continue;
      const uf = `${r.unfused.mean_ms.toFixed(1)} ± ${r.unfused.std_ms.toFixed(1)}`;
      const ff = `${r.fused.mean_ms.toFixed(1)} ± ${r.fused.std_ms.toFixed(1)}`;
      log(`${r.label.padEnd(17)} | ${String(r.dispatches).padStart(10)} | ${uf.padStart(20)} | ${ff.padStart(20)} | ${r.speedup.toFixed(1)}×`);
    }
  }

  return { results, seqResults, gpuName };
}
