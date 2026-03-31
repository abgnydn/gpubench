// UNFUSED KERNEL 1: Layer Normalization (separate dispatch)

const D: u32 = 32u;
const SL: u32 = 64u;
const EPS: f32 = 1e-5;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>;  // x = current_seq_len

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    if (token >= u.x) { return; }

    let base = token * D;
    var mean: f32 = 0.0;
    for (var i = 0u; i < D; i++) { mean += input[base + i]; }
    mean /= f32(D);

    var variance: f32 = 0.0;
    for (var i = 0u; i < D; i++) {
        let diff = input[base + i] - mean;
        variance += diff * diff;
    }
    variance /= f32(D);
    let inv_std = 1.0 / sqrt(variance + EPS);

    for (var i = 0u; i < D; i++) {
        output[base + i] = (input[base + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
