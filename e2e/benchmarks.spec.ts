import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// Playwright runs from the project root; no need for import.meta.url gymnastics.
const SHADERS_SRC = readFileSync(
  resolve(process.cwd(), "src", "lib", "shaders.ts"),
  "utf8",
);

// Pull each `export const FOO_SHADER = /* wgsl */ \`…\`;` body out of shaders.ts.
function extractShader(name: string): string {
  const re = new RegExp(
    `export\\s+const\\s+${name}\\s*=\\s*\\/\\*\\s*wgsl\\s*\\*\\/\\s*\`([\\s\\S]*?)\`;`,
  );
  const m = SHADERS_SRC.match(re);
  if (!m?.[1]) throw new Error(`shader export not found: ${name}`);
  return m[1];
}

const SHADERS: Record<string, string> = {
  RASTRIGIN_SHADER: extractShader("RASTRIGIN_SHADER"),
  NBODY_SHADER: extractShader("NBODY_SHADER"),
  ACROBOT_SHADER: extractShader("ACROBOT_SHADER"),
  MOUNTAINCAR_SHADER: extractShader("MOUNTAINCAR_SHADER"),
  CARTPOLE_SHADER: extractShader("CARTPOLE_SHADER"),
  MONTECARLO_SHADER: extractShader("MONTECARLO_SHADER"),
};

test("WebGPU is available in this Chromium build", async ({ page }) => {
  await page.goto("/");
  const hasWebGPU = await page.evaluate(() => "gpu" in navigator);
  expect(hasWebGPU, "navigator.gpu must be defined — check launch flags").toBe(true);

  const adapter = await page.evaluate(async () => {
    const a = await (
      navigator as Navigator & { gpu: { requestAdapter: () => Promise<unknown> } }
    ).gpu.requestAdapter();
    return a ? "ok" : "null";
  });
  expect(adapter, "requestAdapter() returned null — WebGPU not actually working").toBe("ok");
});

test("page renders all 6 benchmark cards", async ({ page }) => {
  await page.goto("/");
  for (const name of [
    "Rastrigin",
    "N-Body Simulation",
    "Acrobot-v1",
    "MountainCar-v0",
    "CartPole-v1",
    "Monte Carlo Pi",
  ]) {
    await expect(page.getByText(name).first()).toBeVisible({ timeout: 15_000 });
  }
});

test("all 6 benchmark WGSL shaders compile + create a pipeline without errors", async ({
  page,
}) => {
  await page.goto("/");

  // Compile each shader on a real GPUDevice. A WGSL syntax error or missing
  // binding raises a GPUValidationError on createComputePipeline; this is the
  // single most common regression a contributor can introduce.
  const result = await page.evaluate(async (shaders: Record<string, string>) => {
    type Per = { key: string; ok: boolean; error?: string };
    if (!navigator.gpu) return { fatal: "no navigator.gpu" as string, per: [] as Per[] };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { fatal: "no adapter" as string, per: [] as Per[] };
    const device = await adapter.requestDevice();

    const out: Per[] = [];
    for (const [key, code] of Object.entries(shaders)) {
      if (typeof code !== "string" || code.length === 0) {
        out.push({ key, ok: false, error: "empty shader source" });
        continue;
      }
      device.pushErrorScope("validation");
      const shaderModule = device.createShaderModule({ code });
      device.createComputePipeline({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
      });
      const err = await device.popErrorScope();
      out.push({ key, ok: err === null, error: err?.message });
    }

    device.destroy();
    return { fatal: null as string | null, per: out };
  }, SHADERS);

  expect(result.fatal, `WebGPU not usable in this run: ${result.fatal}`).toBeNull();
  for (const r of result.per) {
    expect(r.ok, `${r.key} failed to compile: ${r.error}`).toBe(true);
  }
});

test("CARTPOLE_SHADER produces finite reward on a small dispatch", async ({ page }) => {
  await page.goto("/");

  // Tiny end-to-end smoke run for the new benchmark — pop=64, steps=50,
  // random genomes. Verifies the shader actually runs and writes finite
  // output without going through the heavyweight benchmark UI loop.
  const reward = await page.evaluate(async (code: string) => {
    if (!navigator.gpu) return { fatal: "no navigator.gpu" } as const;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { fatal: "no adapter" } as const;
    const device = await adapter.requestDevice();

    const POP = 64;
    const GSIZE = 58;
    const STEPS = 50;

    const genomes = new Float32Array(POP * GSIZE);
    for (let i = 0; i < genomes.length; i++) genomes[i] = (Math.random() - 0.5) * 0.2;

    const inBuf = device.createBuffer({
      size: genomes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inBuf, 0, genomes);

    const outBuf = device.createBuffer({
      size: POP * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const params = new Uint32Array([POP, GSIZE, STEPS]);
    const paramsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, params);

    const module = device.createShaderModule({ code });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(POP / 64));
    pass.end();

    const readback = device.createBuffer({
      size: POP * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    enc.copyBufferToBuffer(outBuf, 0, readback, 0, POP * 4);
    device.queue.submit([enc.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    device.destroy();

    let allFinite = true;
    let anyNonzero = false;
    for (const v of data) {
      if (!Number.isFinite(v)) allFinite = false;
      if (v !== 0) anyNonzero = true;
    }
    return { fatal: null, allFinite, anyNonzero, sample: Array.from(data.slice(0, 4)) };
  }, SHADERS["CARTPOLE_SHADER"]!);

  expect(reward.fatal, `WebGPU not usable: ${reward.fatal}`).toBeNull();
  expect(reward.allFinite, `CartPole produced NaN/Inf in output: ${JSON.stringify(reward.sample)}`).toBe(true);
  expect(
    reward.anyNonzero,
    `CartPole output all zeros — physics or NN forward likely broken: ${JSON.stringify(reward.sample)}`,
  ).toBe(true);
});
