export interface BenchmarkConfig {
  name: string;
  shader: string;
  populationSize: number;
  dimensions: number;
  warmupIterations: number;
  benchmarkIterations: number;
}

export interface BenchmarkResult {
  name: string;
  throughput: number;
  meanTime: number;
  minTime: number;
  maxTime: number;
  stdDev: number;
  iterations: number;
  populationSize: number;
  dimensions: number;
}

export class BenchmarkRunner {
  private device: GPUDevice | null = null;

  async init(): Promise<void> {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter found");
    this.device = await adapter.requestDevice();
  }

  async run(
    config: BenchmarkConfig,
    onProgress: (pct: number) => void
  ): Promise<BenchmarkResult> {
    const device = this.device;
    if (!device) throw new Error("Runner not initialized");

    const { shader, populationSize, dimensions, warmupIterations, benchmarkIterations } = config;

    const shaderModule = device.createShaderModule({ code: shader });

    // Detect shader type for buffer setup
    const isMonteCarlo = shader.includes("seeds");
    const isNbody = shader.includes("initial_pos");
    const hasVec3Params = shader.includes("vec3<u32>");

    // --- Create buffers ---
    const buffersToDestroy: GPUBuffer[] = [];

    // Input buffer
    const inputElementCount = populationSize * Math.max(dimensions, 1);
    const inputData = isMonteCarlo
      ? new Uint32Array(populationSize) // seeds for Monte Carlo
      : new Float32Array(isNbody ? populationSize * 4 : inputElementCount);

    // Fill input with random data
    const chunkSize = 16384;
    if (isMonteCarlo) {
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

    // Output buffer
    const outputBuffer = device.createBuffer({
      size: populationSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    buffersToDestroy.push(outputBuffer);

    // Params buffer
    const paramsSize = hasVec3Params ? 16 : 8; // vec3 padded to 16 bytes
    const paramsData = hasVec3Params
      ? new Uint32Array([populationSize, dimensions, isMonteCarlo ? 100000 : isNbody ? 200 : 500])
      : new Uint32Array([populationSize, isMonteCarlo ? 100000 : dimensions]);

    const paramsBuffer = device.createBuffer({
      size: paramsSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    buffersToDestroy.push(paramsBuffer);

    // --- Create pipeline + bind group ---
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

    // --- Warmup ---
    for (let i = 0; i < warmupIterations; i++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    // --- Benchmark ---
    const times: number[] = [];

    for (let i = 0; i < benchmarkIterations; i++) {
      const start = performance.now();

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      times.push(performance.now() - start);
      onProgress(Math.round(((i + 1) / benchmarkIterations) * 100));
    }

    // --- Cleanup ---
    for (const buf of buffersToDestroy) {
      buf.destroy();
    }

    // --- Stats ---
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);
    const variance = times.reduce((acc, t) => acc + (t - mean) ** 2, 0) / times.length;
    const stdDev = Math.sqrt(variance);

    return {
      name: config.name,
      throughput: Math.round(1000 / mean),
      meanTime: mean,
      minTime: min,
      maxTime: max,
      stdDev,
      iterations: benchmarkIterations,
      populationSize,
      dimensions,
    };
  }

  destroy(): void {
    this.device?.destroy();
    this.device = null;
  }
}
