export interface BenchmarkConfig {
  name: string;
  shader: string;
  populationSize: number;
  /** Elements per individual in the input buffer (4 for n-body, 1 minimum). */
  dimensions: number;
  /** Uniform buffer contents, e.g. [pop, dim] or [pop, genomeSize, steps].
   *  Must match the shader's params layout — nothing is inferred from the
   *  shader source. */
  params: number[];
  /** "random-floats": f32 in [-1, 1]. "random-seeds": u32 PRNG seeds. */
  input: "random-floats" | "random-seeds";
  warmupIterations: number;
  benchmarkIterations: number;
}

export interface BenchmarkResult {
  name: string;
  /** Dispatches/sec with one dispatch per submit (v1 protocol — includes
   *  per-submit browser/driver overhead; comparable with historical data). */
  throughput: number;
  meanTime: number;
  minTime: number;
  maxTime: number;
  stdDev: number;
  /** Per-dispatch ms when BATCH dispatches share one command buffer + submit.
   *  Much closer to pure GPU time — the v1 numbers are dominated by submit
   *  overhead on fast GPUs. */
  batchedMeanTime: number;
  batchedThroughput: number;
  iterations: number;
  populationSize: number;
  dimensions: number;
}

/** Bumped when the measurement protocol changes; stored with each run so
 *  aggregate queries never mix incomparable scores. v2 = adds batched timing
 *  + high-performance adapter (v1 rows predate both). */
export const BENCH_PROTOCOL_VERSION = 2;

const BATCH_SIZE = 8;
const BATCH_ITERATIONS = 5;

export class BenchmarkRunner {
  private device: GPUDevice | null = null;

  async init(): Promise<void> {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    // Without powerPreference, dual-GPU laptops may hand us the integrated
    // GPU while the leaderboard row gets labeled with whatever adapter info
    // says — always ask for the fast one, matching the transformer bench.
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No GPU adapter found");
    this.device = await adapter.requestDevice();
  }

  async run(
    config: BenchmarkConfig,
    onProgress: (pct: number) => void,
    onPhase?: (phase: "warmup" | "running") => void
  ): Promise<BenchmarkResult> {
    const device = this.device;
    if (!device) throw new Error("Runner not initialized");

    const { shader, populationSize, dimensions, params, input, warmupIterations, benchmarkIterations } = config;

    const shaderModule = device.createShaderModule({ code: shader });

    const buffersToDestroy: GPUBuffer[] = [];
    try {
      // --- Input buffer ---
      const inputElementCount = populationSize * Math.max(dimensions, 1);
      const inputData =
        input === "random-seeds"
          ? new Uint32Array(populationSize)
          : new Float32Array(inputElementCount);

      const chunkSize = 16384;
      if (input === "random-seeds") {
        const rng = inputData as Uint32Array;
        for (let offset = 0; offset < rng.length; offset += chunkSize) {
          const len = Math.min(chunkSize, rng.length - offset);
          crypto.getRandomValues(rng.subarray(offset, offset + len));
        }
      } else {
        const floatData = inputData as Float32Array;
        const rng = new Uint32Array(Math.min(chunkSize, floatData.length));
        for (let offset = 0; offset < floatData.length; offset += rng.length) {
          const len = Math.min(rng.length, floatData.length - offset);
          crypto.getRandomValues(rng.subarray(0, len));
          for (let i = 0; i < len; i++) {
            floatData[offset + i] = ((rng[i] ?? 0) / 4294967295) * 2 - 1;
          }
        }
      }

      const inputBuffer = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(inputBuffer, 0, inputData);
      buffersToDestroy.push(inputBuffer);

      // --- Output buffer ---
      const outputBuffer = device.createBuffer({
        size: populationSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      buffersToDestroy.push(outputBuffer);

      // --- Params buffer (vec2<u32> = 8 bytes, vec3<u32> padded to 16) ---
      const paramsData = new Uint32Array(params);
      const paramsBuffer = device.createBuffer({
        size: params.length <= 2 ? 8 : 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      buffersToDestroy.push(paramsBuffer);

      // --- Pipeline + bind group ---
      const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
      });

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const workgroupCount = Math.ceil(populationSize / 64);

      const encodeDispatches = (count: number) => {
        const encoder = device.createCommandEncoder();
        for (let i = 0; i < count; i++) {
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(workgroupCount);
          pass.end();
        }
        device.queue.submit([encoder.finish()]);
      };

      const totalUnits = benchmarkIterations + BATCH_ITERATIONS;
      let doneUnits = 0;
      const tick = () => {
        doneUnits++;
        onProgress(Math.round((doneUnits / totalUnits) * 100));
      };

      // --- Warmup ---
      onPhase?.("warmup");
      for (let i = 0; i < warmupIterations; i++) {
        encodeDispatches(1);
        await device.queue.onSubmittedWorkDone();
      }
      onPhase?.("running");

      // --- Benchmark: one dispatch per submit (v1 protocol) ---
      const times: number[] = [];
      for (let i = 0; i < benchmarkIterations; i++) {
        const start = performance.now();
        encodeDispatches(1);
        await device.queue.onSubmittedWorkDone();
        times.push(performance.now() - start);
        tick();
      }

      // --- Benchmark: BATCH_SIZE dispatches per submit ---
      const batchedTimes: number[] = [];
      for (let i = 0; i < BATCH_ITERATIONS; i++) {
        const start = performance.now();
        encodeDispatches(BATCH_SIZE);
        await device.queue.onSubmittedWorkDone();
        batchedTimes.push((performance.now() - start) / BATCH_SIZE);
        tick();
      }

      // --- Stats ---
      const mean = times.reduce((a, b) => a + b, 0) / times.length;
      const min = Math.min(...times);
      const max = Math.max(...times);
      const variance = times.reduce((acc, t) => acc + (t - mean) ** 2, 0) / times.length;
      const stdDev = Math.sqrt(variance);
      const batchedMean = batchedTimes.reduce((a, b) => a + b, 0) / batchedTimes.length;

      return {
        name: config.name,
        throughput: Math.round(1000 / mean),
        meanTime: mean,
        minTime: min,
        maxTime: max,
        stdDev,
        batchedMeanTime: batchedMean,
        batchedThroughput: Math.round(1000 / batchedMean),
        iterations: benchmarkIterations,
        populationSize,
        dimensions,
      };
    } finally {
      for (const buf of buffersToDestroy) {
        buf.destroy();
      }
    }
  }

  destroy(): void {
    this.device?.destroy();
    this.device = null;
  }
}
