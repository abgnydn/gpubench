/**
 * Transformer benchmark config + shader-gen invariants.
 * Runs under vitest (`npm run test`) — no GPU required.
 */
import { describe, it, expect } from "vitest";
import { CONFIGS, SEQ_CONFIGS, runSweepWithConfigs } from "@/lib/transformer-bench.js";
import {
  generateConfig, layerOffsets,
  generateFusedShader, generateParallelFusedShader,
  generateUnfusedAttn, generateUnfusedFFN, generateUnfusedLN,
} from "@/lib/shader-gen.js";

// Mirrors ALL_CONFIGS in src/app/transformer/page.tsx — the page filters
// module CONFIGS by label, so the labels must stay in sync.
const PAGE_CONFIGS = [
  { key: "d32l1", label: "D=32, L=1", default: true },
  { key: "d32l4", label: "D=32, L=4", default: true },
  { key: "d64l1", label: "D=64, L=1", default: true },
  { key: "d64l4", label: "D=64, L=4", default: true },
  { key: "d128l1", label: "D=128, L=1", default: false },
  { key: "d128l4", label: "D=128, L=4", default: false },
];

function filterByKeys(keys) {
  const labels = new Set(PAGE_CONFIGS.filter((c) => keys.has(c.key)).map((c) => c.label));
  return CONFIGS.filter((c) => labels.has(c.label));
}

describe("config selection", () => {
  it("defaults to the 4 fast configs", () => {
    const selected = new Set(PAGE_CONFIGS.filter((c) => c.default).map((c) => c.key));
    expect(selected.size).toBe(4);
    expect(selected.has("d32l1")).toBe(true);
    expect(selected.has("d64l4")).toBe(true);
    expect(selected.has("d128l1")).toBe(false);
  });

  it("filters CONFIGS by page selection without mutating", () => {
    const before = CONFIGS.length;
    expect(filterByKeys(new Set(["d32l1"])).map((c) => c.label)).toEqual(["D=32, L=1"]);
    expect(filterByKeys(new Set(["d32l1", "d32l4"]))).toHaveLength(2);
    expect(filterByKeys(new Set(PAGE_CONFIGS.map((c) => c.key)))).toHaveLength(6);
    expect(filterByKeys(new Set())).toHaveLength(0);
    expect(CONFIGS.length).toBe(before);
  });

  it("page labels match module labels exactly", () => {
    expect(PAGE_CONFIGS.map((c) => c.label)).toEqual(CONFIGS.map((c) => c.label));
  });

  it("runSweepWithConfigs takes (configs, log, onResult)", () => {
    expect(runSweepWithConfigs.length).toBe(3);
  });

  it("SEQ_CONFIGS keeps D=32, L=1 across sequence lengths", () => {
    for (const c of SEQ_CONFIGS) {
      expect(c.D).toBe(32);
      expect(c.layers).toBe(1);
    }
  });
});

describe("shader-gen invariants", () => {
  const cfg = generateConfig(64, 2, 4, 64, 4);

  it("layerOffsets covers exactly one layer of packed weights", () => {
    const o = layerOffsets(cfg);
    const { D, DF } = cfg;
    // Last tensor is b2 (D floats); its end must equal perLayer.
    expect(o.B2 + D).toBe(cfg.perLayer);
    expect(o.W2).toBe(o.B1 + DF);
    expect(cfg.totalWeights).toBe(cfg.perLayer * cfg.NL);
  });

  it("unfused attn/ffn expose a residual binding so the chain matches the fused dataflow", () => {
    expect(generateUnfusedAttn(cfg)).toContain("binding(7)");
    expect(generateUnfusedFFN(cfg)).toContain("binding(7)");
    expect(generateUnfusedLN(cfg)).not.toContain("binding(7)");
  });

  it("parallel-fused FFN scratch writes past the SL*D output region", () => {
    const src = generateParallelFusedShader(cfg, 64);
    const scratchBase = cfg.SL * cfg.D;
    expect(src).toContain(`out[${scratchBase}u + i] = gelu(a)`);
  });

  it("fused shader bakes the config constants", () => {
    const src = generateFusedShader(cfg);
    expect(src).toContain(`const D: u32 = ${cfg.D}u`);
    expect(src).toContain(`const NL: u32 = ${cfg.NL}u`);
  });
});
