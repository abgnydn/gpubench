/**
 * Parameterized WGSL Shader Generator
 *
 * Generates fused and unfused transformer shaders for any configuration.
 * Mirrors the pattern from webgpu-kernel-fusion: constants baked at generation time.
 */

export function generateConfig(D, nHeads, ffnMul, seqLen, nLayers) {
  const HD = D / nHeads;
  const DF = D * ffnMul;

  // Weight layout per layer
  const perLayer =
    D +           // ln1_gamma
    D +           // ln1_beta
    D * D +       // Wq
    D * D +       // Wk
    D * D +       // Wv
    D * D +       // Wo
    D +           // ln2_gamma
    D +           // ln2_beta
    D * DF +      // W1
    DF +          // b1
    DF * D +      // W2
    D;            // b2

  return { D, NH: nHeads, HD, DF, SL: seqLen, NL: nLayers, perLayer, totalWeights: perLayer * nLayers };
}

export function generateFusedShader(cfg) {
  const { D, NH, HD, DF, SL, NL, perLayer } = cfg;

  // Compute offsets per layer (relative to layer base)
  const o = {};
  let pos = 0;
  o.LG = pos; pos += D;
  o.LB = pos; pos += D;
  o.WQ = pos; pos += D * D;
  o.WK = pos; pos += D * D;
  o.WV = pos; pos += D * D;
  o.WO = pos; pos += D * D;
  o.LG2 = pos; pos += D;
  o.LB2 = pos; pos += D;
  o.W1 = pos; pos += D * DF;
  o.B1 = pos; pos += DF;
  o.W2 = pos; pos += DF * D;
  o.B2 = pos; pos += D;

  return /* wgsl */ `
// FUSED TRANSFORMER — ${NL} layer(s), D=${D}, H=${NH}, FFN=${DF}, SEQ=${SL}
// Single dispatch: all layers × all tokens in one kernel.

const D: u32 = ${D}u;
const NH: u32 = ${NH}u;
const HD: u32 = ${HD}u;
const DF: u32 = ${DF}u;
const SL: u32 = ${SL}u;
const NL: u32 = ${NL}u;
const EPS: f32 = 1e-5;
const PL: u32 = ${perLayer}u;  // weights per layer

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> emb: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

fn gelu(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 1u) { return; }
    let nt = u.x;

    for (var t = 0u; t < nt; t++) {
        let tb = t * D;

        // Load embedding into x
        var x: array<f32, ${D}>;
        for (var i = 0u; i < D; i++) { x[i] = emb[tb + i]; }

        // Layer loop
        for (var layer = 0u; layer < NL; layer++) {
            let LB = layer * PL;  // layer base offset into weights
            let kvBase = layer * SL * D;  // per-layer KV cache offset

            // === LN1 ===
            var ln: array<f32, ${D}>;
            var mu: f32 = 0.0;
            for (var i = 0u; i < D; i++) { mu += x[i]; }
            mu /= f32(D);
            var v: f32 = 0.0;
            for (var i = 0u; i < D; i++) { let d = x[i] - mu; v += d * d; }
            v /= f32(D);
            let is1 = 1.0 / sqrt(v + EPS);
            for (var i = 0u; i < D; i++) {
                ln[i] = (x[i] - mu) * is1 * W[LB + ${o.LG}u + i] + W[LB + ${o.LB}u + i];
            }

            // === Q projection ===
            var Q: array<f32, ${D}>;
            for (var i = 0u; i < D; i++) {
                var a: f32 = 0.0;
                for (var j = 0u; j < D; j++) { a += ln[j] * W[LB + ${o.WQ}u + j * D + i]; }
                Q[i] = a;
            }

            // === K, V → cache ===
            for (var i = 0u; i < D; i++) {
                var ka: f32 = 0.0; var va: f32 = 0.0;
                for (var j = 0u; j < D; j++) {
                    ka += ln[j] * W[LB + ${o.WK}u + j * D + i];
                    va += ln[j] * W[LB + ${o.WV}u + j * D + i];
                }
                kv[kvBase + t * D + i] = ka;
                kv[NL * SL * D + kvBase + t * D + i] = va;
            }

            // === Multi-head attention ===
            var ao: array<f32, ${D}>;
            for (var i = 0u; i < D; i++) { ao[i] = 0.0; }

            for (var h = 0u; h < NH; h++) {
                let ho = h * HD;
                var mx: f32 = -1e9;
                for (var s = 0u; s <= t; s++) {
                    var sc: f32 = 0.0;
                    for (var d = 0u; d < HD; d++) {
                        sc += Q[ho + d] * kv[kvBase + s * D + ho + d];
                    }
                    sc /= sqrt(f32(HD));
                    if (sc > mx) { mx = sc; }
                }
                var es: f32 = 0.0;
                var hout: array<f32, ${HD}>;
                for (var d = 0u; d < HD; d++) { hout[d] = 0.0; }
                for (var s = 0u; s <= t; s++) {
                    var sc: f32 = 0.0;
                    for (var d = 0u; d < HD; d++) {
                        sc += Q[ho + d] * kv[kvBase + s * D + ho + d];
                    }
                    let w = exp(sc / sqrt(f32(HD)) - mx);
                    es += w;
                    for (var d = 0u; d < HD; d++) {
                        hout[d] += w * kv[NL * SL * D + kvBase + s * D + ho + d];
                    }
                }
                if (es > 0.0) {
                    for (var d = 0u; d < HD; d++) { ao[ho + d] = hout[d] / es; }
                }
            }

            // Output projection + residual
            for (var i = 0u; i < D; i++) {
                var a: f32 = 0.0;
                for (var j = 0u; j < D; j++) { a += ao[j] * W[LB + ${o.WO}u + j * D + i]; }
                x[i] = x[i] + a;
            }

            // === LN2 ===
            var ln2: array<f32, ${D}>;
            mu = 0.0;
            for (var i = 0u; i < D; i++) { mu += x[i]; }
            mu /= f32(D);
            v = 0.0;
            for (var i = 0u; i < D; i++) { let d = x[i] - mu; v += d * d; }
            v /= f32(D);
            let is2 = 1.0 / sqrt(v + EPS);
            for (var i = 0u; i < D; i++) {
                ln2[i] = (x[i] - mu) * is2 * W[LB + ${o.LG2}u + i] + W[LB + ${o.LB2}u + i];
            }

            // === FFN ===
            var hid: array<f32, ${DF}>;
            for (var i = 0u; i < DF; i++) {
                var a = W[LB + ${o.B1}u + i];
                for (var j = 0u; j < D; j++) { a += ln2[j] * W[LB + ${o.W1}u + j * DF + i]; }
                hid[i] = gelu(a);
            }
            for (var i = 0u; i < D; i++) {
                var a = W[LB + ${o.B2}u + i];
                for (var j = 0u; j < DF; j++) { a += hid[j] * W[LB + ${o.W2}u + j * D + i]; }
                x[i] = x[i] + a;
            }
        } // end layer loop

        for (var i = 0u; i < D; i++) { out[tb + i] = x[i]; }
    }
}
`;
}

export function generateUnfusedLN(cfg) {
  const { D } = cfg;
  return /* wgsl */ `
const D: u32 = ${D}u;
const EPS: f32 = 1e-5;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    if (token >= u.x) { return; }
    let base = token * D;
    var mean: f32 = 0.0;
    for (var i = 0u; i < D; i++) { mean += input[base + i]; }
    mean /= f32(D);
    var variance: f32 = 0.0;
    for (var i = 0u; i < D; i++) { let d = input[base + i] - mean; variance += d * d; }
    variance /= f32(D);
    let inv_std = 1.0 / sqrt(variance + EPS);
    for (var i = 0u; i < D; i++) {
        output[base + i] = (input[base + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}`;
}

export function generateUnfusedAttn(cfg) {
  const { D, NH, HD } = cfg;
  return /* wgsl */ `
const D: u32 = ${D}u;
const NH: u32 = ${NH}u;
const HD: u32 = ${HD}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> Wq: array<f32>;
@group(0) @binding(3) var<storage, read> Wk: array<f32>;
@group(0) @binding(4) var<storage, read> Wv: array<f32>;
@group(0) @binding(5) var<storage, read> Wo: array<f32>;
@group(0) @binding(6) var<uniform> u: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    if (token >= u.x) { return; }
    let base = token * D;

    var Q: array<f32, ${D}>;
    for (var i = 0u; i < D; i++) {
        var acc: f32 = 0.0;
        for (var j = 0u; j < D; j++) { acc += input[base + j] * Wq[j * D + i]; }
        Q[i] = acc;
    }

    for (var h = 0u; h < NH; h++) {
        let ho = h * HD;
        var max_score: f32 = -1e9;
        for (var s = 0u; s <= token; s++) {
            let sb = s * D;
            var score: f32 = 0.0;
            for (var d = 0u; d < HD; d++) {
                var k_val: f32 = 0.0;
                for (var j = 0u; j < D; j++) { k_val += input[sb + j] * Wk[j * D + ho + d]; }
                score += Q[ho + d] * k_val;
            }
            score /= sqrt(f32(HD));
            if (score > max_score) { max_score = score; }
        }
        var attn_sum: f32 = 0.0;
        var head_out: array<f32, ${HD}>;
        for (var d = 0u; d < HD; d++) { head_out[d] = 0.0; }
        for (var s = 0u; s <= token; s++) {
            let sb = s * D;
            var score: f32 = 0.0;
            for (var d = 0u; d < HD; d++) {
                var k_val: f32 = 0.0;
                for (var j = 0u; j < D; j++) { k_val += input[sb + j] * Wk[j * D + ho + d]; }
                score += Q[ho + d] * k_val;
            }
            let w = exp(score / sqrt(f32(HD)) - max_score);
            attn_sum += w;
            for (var d = 0u; d < HD; d++) {
                var v_val: f32 = 0.0;
                for (var j = 0u; j < D; j++) { v_val += input[sb + j] * Wv[j * D + ho + d]; }
                head_out[d] += w * v_val;
            }
        }
        if (attn_sum > 0.0) {
            for (var d = 0u; d < HD; d++) { output[base + ho + d] = head_out[d] / attn_sum; }
        }
    }

    var final_out: array<f32, ${D}>;
    for (var i = 0u; i < D; i++) {
        var acc: f32 = 0.0;
        for (var j = 0u; j < D; j++) { acc += output[base + j] * Wo[j * D + i]; }
        final_out[i] = acc;
    }
    for (var i = 0u; i < D; i++) { output[base + i] = input[base + i] + final_out[i]; }
}`;
}

export function generateUnfusedFFN(cfg) {
  const { D, DF } = cfg;
  return /* wgsl */ `
const D: u32 = ${D}u;
const DF: u32 = ${DF}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> W1: array<f32>;
@group(0) @binding(3) var<storage, read> b1: array<f32>;
@group(0) @binding(4) var<storage, read> W2: array<f32>;
@group(0) @binding(5) var<storage, read> b2: array<f32>;
@group(0) @binding(6) var<uniform> u: vec4<u32>;

fn gelu(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    if (token >= u.x) { return; }
    let base = token * D;

    var hidden: array<f32, ${DF}>;
    for (var i = 0u; i < DF; i++) {
        var acc = b1[i];
        for (var j = 0u; j < D; j++) { acc += input[base + j] * W1[j * DF + i]; }
        hidden[i] = gelu(acc);
    }
    for (var i = 0u; i < D; i++) {
        var acc = b2[i];
        for (var j = 0u; j < DF; j++) { acc += hidden[j] * W2[j * D + i]; }
        output[base + i] = input[base + i] + acc;
    }
}`;
}

// ═══════════════════════════════════════════════════════════════
// PARALLEL FUSED KERNEL
// workgroup_size(WG) — threads split the hidden dimension
// Shared memory for LN reductions and attention softmax
// Token loop stays fused (no dispatch per token)
// ═══════════════════════════════════════════════════════════════
export function generateParallelFusedShader(cfg, WG = 64) {
  const { D, NH, HD, DF, SL, NL, perLayer } = cfg;

  const o = {};
  let pos = 0;
  o.LG = pos; pos += D;
  o.LB = pos; pos += D;
  o.WQ = pos; pos += D * D;
  o.WK = pos; pos += D * D;
  o.WV = pos; pos += D * D;
  o.WO = pos; pos += D * D;
  o.LG2 = pos; pos += D;
  o.LB2 = pos; pos += D;
  o.W1 = pos; pos += D * DF;
  o.B1 = pos; pos += DF;
  o.W2 = pos; pos += DF * D;
  o.B2 = pos; pos += D;

  // Each thread handles D/WG elements of the hidden dim (for LN, projections)
  // For FFN up-projection: each thread handles DF/WG elements
  const elemsPerThread = Math.ceil(D / WG);
  const ffnElemsPerThread = Math.ceil(DF / WG);

  return /* wgsl */ `
// PARALLEL FUSED TRANSFORMER — ${NL}L, D=${D}, H=${NH}, FFN=${DF}, SEQ=${SL}, WG=${WG}
// Token loop fused. Workgroup threads parallelize matmuls + reductions.

const D: u32 = ${D}u;
const NH: u32 = ${NH}u;
const HD: u32 = ${HD}u;
const DF: u32 = ${DF}u;
const SL: u32 = ${SL}u;
const NL: u32 = ${NL}u;
const EPS: f32 = 1e-5;
const PL: u32 = ${perLayer}u;
const WG: u32 = ${WG}u;

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> emb: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

// Shared memory for inter-thread communication
var<workgroup> shared_x: array<f32, ${D}>;
var<workgroup> shared_buf: array<f32, ${Math.max(D, DF)}>;
var<workgroup> shared_reduce: array<f32, ${WG}>;

fn gelu(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(${WG})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch = wid.x;
    let nt = u.x;
    if (batch >= 1u) { return; }

    for (var t = 0u; t < nt; t++) {
        let tb = t * D;

        // Load embedding into shared_x (parallel)
        for (var i = tid; i < D; i += WG) { shared_x[i] = emb[tb + i]; }
        workgroupBarrier();

        for (var layer = 0u; layer < NL; layer++) {
            let LB = layer * PL;
            let kvBase = layer * SL * D;

            // ══════ LN1 (parallel reduction) ══════
            // Step 1: partial sum for mean
            var partial_sum: f32 = 0.0;
            for (var i = tid; i < D; i += WG) { partial_sum += shared_x[i]; }
            shared_reduce[tid] = partial_sum;
            workgroupBarrier();
            // Tree reduction
            for (var stride = WG / 2u; stride > 0u; stride /= 2u) {
                if (tid < stride) { shared_reduce[tid] += shared_reduce[tid + stride]; }
                workgroupBarrier();
            }
            let mu = shared_reduce[0] / f32(D);
            workgroupBarrier();

            // Step 2: partial sum for variance
            var partial_var: f32 = 0.0;
            for (var i = tid; i < D; i += WG) { let d = shared_x[i] - mu; partial_var += d * d; }
            shared_reduce[tid] = partial_var;
            workgroupBarrier();
            for (var stride = WG / 2u; stride > 0u; stride /= 2u) {
                if (tid < stride) { shared_reduce[tid] += shared_reduce[tid + stride]; }
                workgroupBarrier();
            }
            let inv_std = 1.0 / sqrt(shared_reduce[0] / f32(D) + EPS);
            workgroupBarrier();

            // Step 3: normalize (parallel)
            for (var i = tid; i < D; i += WG) {
                shared_buf[i] = (shared_x[i] - mu) * inv_std * W[LB + ${o.LG}u + i] + W[LB + ${o.LB}u + i];
            }
            workgroupBarrier();
            // shared_buf = LN1 output

            // ══════ Q/K/V Projections (parallel over output dim) ══════
            // Each thread computes its slice of Q, K, V and writes K/V to cache
            for (var i = tid; i < D; i += WG) {
                var qa: f32 = 0.0; var ka: f32 = 0.0; var va: f32 = 0.0;
                for (var j = 0u; j < D; j++) {
                    let ln_j = shared_buf[j];
                    qa += ln_j * W[LB + ${o.WQ}u + j * D + i];
                    ka += ln_j * W[LB + ${o.WK}u + j * D + i];
                    va += ln_j * W[LB + ${o.WV}u + j * D + i];
                }
                out[tb + i] = qa;  // temp: store Q in output buffer
                kv[kvBase + t * D + i] = ka;
                kv[NL * SL * D + kvBase + t * D + i] = va;
            }
            workgroupBarrier();

            // ══════ Attention (parallel over heads) ══════
            // Each thread handles one or more heads
            for (var i = tid; i < D; i += WG) { shared_buf[i] = 0.0; }
            workgroupBarrier();

            for (var h = tid; h < NH; h += WG) {
                let ho = h * HD;
                // Find max score
                var mx: f32 = -1e9;
                for (var s = 0u; s <= t; s++) {
                    var sc: f32 = 0.0;
                    for (var d = 0u; d < HD; d++) {
                        sc += out[tb + ho + d] * kv[kvBase + s * D + ho + d];
                    }
                    sc /= sqrt(f32(HD));
                    if (sc > mx) { mx = sc; }
                }
                // Softmax + V accumulation
                var es: f32 = 0.0;
                var hout: array<f32, ${HD}>;
                for (var d = 0u; d < HD; d++) { hout[d] = 0.0; }
                for (var s = 0u; s <= t; s++) {
                    var sc: f32 = 0.0;
                    for (var d = 0u; d < HD; d++) {
                        sc += out[tb + ho + d] * kv[kvBase + s * D + ho + d];
                    }
                    let w = exp(sc / sqrt(f32(HD)) - mx);
                    es += w;
                    for (var d = 0u; d < HD; d++) {
                        hout[d] += w * kv[NL * SL * D + kvBase + s * D + ho + d];
                    }
                }
                if (es > 0.0) {
                    for (var d = 0u; d < HD; d++) { shared_buf[ho + d] = hout[d] / es; }
                }
            }
            workgroupBarrier();
            // shared_buf = attention output

            // ══════ Output projection + residual (parallel) ══════
            for (var i = tid; i < D; i += WG) {
                var a: f32 = 0.0;
                for (var j = 0u; j < D; j++) { a += shared_buf[j] * W[LB + ${o.WO}u + j * D + i]; }
                shared_x[i] = shared_x[i] + a;  // residual 1
            }
            workgroupBarrier();

            // ══════ LN2 (parallel reduction) ══════
            partial_sum = 0.0;
            for (var i = tid; i < D; i += WG) { partial_sum += shared_x[i]; }
            shared_reduce[tid] = partial_sum;
            workgroupBarrier();
            for (var stride = WG / 2u; stride > 0u; stride /= 2u) {
                if (tid < stride) { shared_reduce[tid] += shared_reduce[tid + stride]; }
                workgroupBarrier();
            }
            let mu2 = shared_reduce[0] / f32(D);
            workgroupBarrier();

            partial_var = 0.0;
            for (var i = tid; i < D; i += WG) { let d = shared_x[i] - mu2; partial_var += d * d; }
            shared_reduce[tid] = partial_var;
            workgroupBarrier();
            for (var stride = WG / 2u; stride > 0u; stride /= 2u) {
                if (tid < stride) { shared_reduce[tid] += shared_reduce[tid + stride]; }
                workgroupBarrier();
            }
            let inv_std2 = 1.0 / sqrt(shared_reduce[0] / f32(D) + EPS);
            workgroupBarrier();

            for (var i = tid; i < D; i += WG) {
                shared_buf[i] = (shared_x[i] - mu2) * inv_std2 * W[LB + ${o.LG2}u + i] + W[LB + ${o.LB2}u + i];
            }
            workgroupBarrier();
            // shared_buf = LN2 output

            // ══════ FFN up-projection (parallel over DF) ══════
            // Each thread computes its slice of the hidden dim
            for (var i = tid; i < DF; i += WG) {
                var a = W[LB + ${o.B1}u + i];
                for (var j = 0u; j < D; j++) { a += shared_buf[j] * W[LB + ${o.W1}u + j * DF + i]; }
                out[i] = gelu(a);  // temp: store FFN hidden in output buffer
            }
            workgroupBarrier();

            // ══════ FFN down-projection + residual (parallel over D) ══════
            for (var i = tid; i < D; i += WG) {
                var a = W[LB + ${o.B2}u + i];
                for (var j = 0u; j < DF; j++) { a += out[j] * W[LB + ${o.W2}u + j * D + i]; }
                shared_x[i] = shared_x[i] + a;  // residual 2
            }
            workgroupBarrier();
        } // end layer loop

        // Write output (parallel)
        for (var i = tid; i < D; i += WG) { out[tb + i] = shared_x[i]; }
        workgroupBarrier();
    } // end token loop
}
`;
}

// ═══════════════════════════════════════════════════════════════
// F16 FUSED KERNEL (single-thread, half precision)
// Uses f16 for weights and activations where possible
// ═══════════════════════════════════════════════════════════════
export function generateF16FusedShader(cfg) {
  const { D, NH, HD, DF, SL, NL, perLayer } = cfg;

  const o = {};
  let pos = 0;
  o.LG = pos; pos += D;
  o.LB = pos; pos += D;
  o.WQ = pos; pos += D * D;
  o.WK = pos; pos += D * D;
  o.WV = pos; pos += D * D;
  o.WO = pos; pos += D * D;
  o.LG2 = pos; pos += D;
  o.LB2 = pos; pos += D;
  o.W1 = pos; pos += D * DF;
  o.B1 = pos; pos += DF;
  o.W2 = pos; pos += DF * D;
  o.B2 = pos; pos += D;

  // Same structure as fused but accumulate in f32, load weights as f16
  // Note: WGSL f16 requires 'enable f16;' and shader-f16 feature
  return /* wgsl */ `
enable f16;

// F16 FUSED TRANSFORMER — ${NL}L, D=${D}, H=${NH}, FFN=${DF}, SEQ=${SL}
// Weights stored as f32 but cast to f16 for computation. Accumulation in f32.

const D: u32 = ${D}u;
const NH: u32 = ${NH}u;
const HD: u32 = ${HD}u;
const DF: u32 = ${DF}u;
const SL: u32 = ${SL}u;
const NL: u32 = ${NL}u;
const EPS: f32 = 1e-5;
const PL: u32 = ${perLayer}u;

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> emb: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;

fn gelu16(x: f16) -> f16 {
    let xf = f32(x);
    let inner = 0.7978845608 * (xf + 0.044715 * xf * xf * xf);
    return f16(0.5 * xf * (1.0 + tanh(inner)));
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 1u) { return; }
    let nt = u.x;

    for (var t = 0u; t < nt; t++) {
        let tb = t * D;

        var x: array<f16, ${D}>;
        for (var i = 0u; i < D; i++) { x[i] = f16(emb[tb + i]); }

        for (var layer = 0u; layer < NL; layer++) {
            let LB = layer * PL;
            let kvBase = layer * SL * D;

            // === LN1 (accumulate in f32) ===
            var ln: array<f16, ${D}>;
            var mu: f32 = 0.0;
            for (var i = 0u; i < D; i++) { mu += f32(x[i]); }
            mu /= f32(D);
            var v: f32 = 0.0;
            for (var i = 0u; i < D; i++) { let d = f32(x[i]) - mu; v += d * d; }
            v /= f32(D);
            let is1 = 1.0 / sqrt(v + EPS);
            for (var i = 0u; i < D; i++) {
                ln[i] = f16((f32(x[i]) - mu) * is1 * W[LB + ${o.LG}u + i] + W[LB + ${o.LB}u + i]);
            }

            // === Q (f16 matmul with f32 accumulation) ===
            var Q: array<f16, ${D}>;
            for (var i = 0u; i < D; i++) {
                var a: f32 = 0.0;
                for (var j = 0u; j < D; j++) { a += f32(ln[j]) * W[LB + ${o.WQ}u + j * D + i]; }
                Q[i] = f16(a);
            }

            // === K, V → cache (store as f32 for precision) ===
            for (var i = 0u; i < D; i++) {
                var ka: f32 = 0.0; var va: f32 = 0.0;
                for (var j = 0u; j < D; j++) {
                    let lnj = f32(ln[j]);
                    ka += lnj * W[LB + ${o.WK}u + j * D + i];
                    va += lnj * W[LB + ${o.WV}u + j * D + i];
                }
                kv[kvBase + t * D + i] = ka;
                kv[NL * SL * D + kvBase + t * D + i] = va;
            }

            // === Attention ===
            var ao: array<f16, ${D}>;
            for (var i = 0u; i < D; i++) { ao[i] = f16(0.0); }

            for (var h = 0u; h < NH; h++) {
                let ho = h * HD;
                var mx: f32 = -1e9;
                for (var s = 0u; s <= t; s++) {
                    var sc: f32 = 0.0;
                    for (var d = 0u; d < HD; d++) { sc += f32(Q[ho + d]) * kv[kvBase + s * D + ho + d]; }
                    sc /= sqrt(f32(HD));
                    if (sc > mx) { mx = sc; }
                }
                var es: f32 = 0.0;
                var hout: array<f32, ${HD}>;
                for (var d = 0u; d < HD; d++) { hout[d] = 0.0; }
                for (var s = 0u; s <= t; s++) {
                    var sc: f32 = 0.0;
                    for (var d = 0u; d < HD; d++) { sc += f32(Q[ho + d]) * kv[kvBase + s * D + ho + d]; }
                    let w = exp(sc / sqrt(f32(HD)) - mx);
                    es += w;
                    for (var d = 0u; d < HD; d++) { hout[d] += w * kv[NL * SL * D + kvBase + s * D + ho + d]; }
                }
                if (es > 0.0) {
                    for (var d = 0u; d < HD; d++) { ao[ho + d] = f16(hout[d] / es); }
                }
            }

            // Output projection + residual
            for (var i = 0u; i < D; i++) {
                var a: f32 = 0.0;
                for (var j = 0u; j < D; j++) { a += f32(ao[j]) * W[LB + ${o.WO}u + j * D + i]; }
                x[i] = f16(f32(x[i]) + a);
            }

            // === LN2 ===
            var ln2: array<f16, ${D}>;
            mu = 0.0;
            for (var i = 0u; i < D; i++) { mu += f32(x[i]); }
            mu /= f32(D);
            v = 0.0;
            for (var i = 0u; i < D; i++) { let d = f32(x[i]) - mu; v += d * d; }
            v /= f32(D);
            let is2 = 1.0 / sqrt(v + EPS);
            for (var i = 0u; i < D; i++) {
                ln2[i] = f16((f32(x[i]) - mu) * is2 * W[LB + ${o.LG2}u + i] + W[LB + ${o.LB2}u + i]);
            }

            // === FFN (f16 activations, f32 accumulation) ===
            var hid: array<f16, ${DF}>;
            for (var i = 0u; i < DF; i++) {
                var a: f32 = W[LB + ${o.B1}u + i];
                for (var j = 0u; j < D; j++) { a += f32(ln2[j]) * W[LB + ${o.W1}u + j * DF + i]; }
                hid[i] = gelu16(f16(a));
            }
            for (var i = 0u; i < D; i++) {
                var a: f32 = W[LB + ${o.B2}u + i];
                for (var j = 0u; j < DF; j++) { a += f32(hid[j]) * W[LB + ${o.W2}u + j * D + i]; }
                x[i] = f16(f32(x[i]) + a);
            }
        } // end layer

        for (var i = 0u; i < D; i++) { out[tb + i] = f32(x[i]); }
    }
}
`;
}
