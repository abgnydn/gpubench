// UNFUSED KERNEL 2: Causal Self-Attention (separate dispatch)

const D: u32 = 32u;
const NH: u32 = 2u;
const HD: u32 = 16u;
const SL: u32 = 64u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> Wq: array<f32>;
@group(0) @binding(3) var<storage, read> Wk: array<f32>;
@group(0) @binding(4) var<storage, read> Wv: array<f32>;
@group(0) @binding(5) var<storage, read> Wo: array<f32>;
@group(0) @binding(6) var<uniform> u: vec4<u32>;  // x = current_seq_len

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    let cur_len = u.x;
    if (token >= cur_len) { return; }

    let base = token * D;

    // Q projection
    var Q: array<f32, 32>;
    for (var i = 0u; i < D; i++) {
        var acc: f32 = 0.0;
        for (var j = 0u; j < D; j++) { acc += input[base + j] * Wq[j * D + i]; }
        Q[i] = acc;
    }

    // Per-head attention
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
        var head_out: array<f32, 16>;
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
            for (var d = 0u; d < HD; d++) {
                output[base + ho + d] = head_out[d] / attn_sum;
            }
        }
    }

    // Output projection + residual
    var final_out: array<f32, 32>;
    for (var i = 0u; i < D; i++) {
        var acc: f32 = 0.0;
        for (var j = 0u; j < D; j++) { acc += output[base + j] * Wo[j * D + i]; }
        final_out[i] = acc;
    }
    for (var i = 0u; i < D; i++) {
        output[base + i] = input[base + i] + final_out[i];
    }
}
