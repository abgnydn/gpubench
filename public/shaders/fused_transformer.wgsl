// FUSED TRANSFORMER KERNEL
// Single dispatch: LayerNorm → Attention → LayerNorm → FFN
// All operations run inside ONE compute shader, looping over tokens.
// No intermediate GPU dispatch overhead. State stays in thread-local registers.

const D: u32 = 32u;        // D_MODEL
const NH: u32 = 2u;        // N_HEADS
const HD: u32 = 16u;       // HEAD_DIM = D / NH
const DF: u32 = 128u;      // D_FFN = 4 * D
const SL: u32 = 64u;       // SEQ_LEN
const EPS: f32 = 1e-5;

// Weight layout (all packed into one buffer):
//   ln1_gamma  [0      .. 32)         = 32
//   ln1_beta   [32     .. 64)         = 32
//   Wq         [64     .. 1088)       = 32*32 = 1024
//   Wk         [1088   .. 2112)       = 1024
//   Wv         [2112   .. 3136)       = 1024
//   Wo         [3136   .. 4160)       = 1024
//   ln2_gamma  [4160   .. 4192)       = 32
//   ln2_beta   [4192   .. 4224)       = 32
//   W1         [4224   .. 8320)       = 32*128 = 4096
//   b1         [8320   .. 8448)       = 128
//   W2         [8448   .. 12544)      = 128*32 = 4096
//   b2         [12544  .. 12576)      = 32
// Total: 12576 f32s

const O_LG: u32 = 0u;
const O_LB: u32 = 32u;
const O_WQ: u32 = 64u;
const O_WK: u32 = 1088u;
const O_WV: u32 = 2112u;
const O_WO: u32 = 3136u;
const O_LG2: u32 = 4160u;
const O_LB2: u32 = 4192u;
const O_W1: u32 = 4224u;
const O_B1: u32 = 8320u;
const O_W2: u32 = 8448u;
const O_B2: u32 = 12544u;

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> emb: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv: array<f32>;    // [SL*D*2]
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;                  // x = num_tokens

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

        // Load embedding
        var x: array<f32, 32>;
        for (var i = 0u; i < D; i++) { x[i] = emb[tb + i]; }

        // === LN1 ===
        var ln: array<f32, 32>;
        var mu: f32 = 0.0;
        for (var i = 0u; i < D; i++) { mu += x[i]; }
        mu /= f32(D);
        var v: f32 = 0.0;
        for (var i = 0u; i < D; i++) { let d = x[i] - mu; v += d * d; }
        v /= f32(D);
        let is1 = 1.0 / sqrt(v + EPS);
        for (var i = 0u; i < D; i++) {
            ln[i] = (x[i] - mu) * is1 * W[O_LG + i] + W[O_LB + i];
        }

        // === Q projection ===
        var Q: array<f32, 32>;
        for (var i = 0u; i < D; i++) {
            var a: f32 = 0.0;
            for (var j = 0u; j < D; j++) { a += ln[j] * W[O_WQ + j * D + i]; }
            Q[i] = a;
        }

        // === K, V projection → cache ===
        for (var i = 0u; i < D; i++) {
            var ka: f32 = 0.0; var va: f32 = 0.0;
            for (var j = 0u; j < D; j++) {
                ka += ln[j] * W[O_WK + j * D + i];
                va += ln[j] * W[O_WV + j * D + i];
            }
            kv[t * D + i] = ka;
            kv[SL * D + t * D + i] = va;
        }

        // === Multi-head attention ===
        var ao: array<f32, 32>;
        for (var i = 0u; i < D; i++) { ao[i] = 0.0; }

        for (var h = 0u; h < NH; h++) {
            let ho = h * HD;

            // Max score for stability
            var mx: f32 = -1e9;
            for (var s = 0u; s <= t; s++) {
                var sc: f32 = 0.0;
                for (var d = 0u; d < HD; d++) {
                    sc += Q[ho + d] * kv[s * D + ho + d];
                }
                sc /= sqrt(f32(HD));
                if (sc > mx) { mx = sc; }
            }

            // Softmax + V accumulation
            var es: f32 = 0.0;
            var hout: array<f32, 16>;
            for (var d = 0u; d < HD; d++) { hout[d] = 0.0; }

            for (var s = 0u; s <= t; s++) {
                var sc: f32 = 0.0;
                for (var d = 0u; d < HD; d++) {
                    sc += Q[ho + d] * kv[s * D + ho + d];
                }
                let w = exp(sc / sqrt(f32(HD)) - mx);
                es += w;
                for (var d = 0u; d < HD; d++) {
                    hout[d] += w * kv[SL * D + s * D + ho + d];
                }
            }

            if (es > 0.0) {
                for (var d = 0u; d < HD; d++) { ao[ho + d] = hout[d] / es; }
            }
        }

        // Output projection + residual
        for (var i = 0u; i < D; i++) {
            var a: f32 = 0.0;
            for (var j = 0u; j < D; j++) { a += ao[j] * W[O_WO + j * D + i]; }
            x[i] = x[i] + a;
        }

        // === LN2 ===
        var ln2: array<f32, 32>;
        mu = 0.0;
        for (var i = 0u; i < D; i++) { mu += x[i]; }
        mu /= f32(D);
        v = 0.0;
        for (var i = 0u; i < D; i++) { let d = x[i] - mu; v += d * d; }
        v /= f32(D);
        let is2 = 1.0 / sqrt(v + EPS);
        for (var i = 0u; i < D; i++) {
            ln2[i] = (x[i] - mu) * is2 * W[O_LG2 + i] + W[O_LB2 + i];
        }

        // === FFN ===
        var hid: array<f32, 128>;
        for (var i = 0u; i < DF; i++) {
            var a = W[O_B1 + i];
            for (var j = 0u; j < D; j++) { a += ln2[j] * W[O_W1 + j * DF + i]; }
            hid[i] = gelu(a);
        }
        for (var i = 0u; i < D; i++) {
            var a = W[O_B2 + i];
            for (var j = 0u; j < DF; j++) { a += hid[j] * W[O_W2 + j * D + i]; }
            x[i] = x[i] + a;
        }

        // Write output
        for (var i = 0u; i < D; i++) { out[tb + i] = x[i]; }
    }
}
