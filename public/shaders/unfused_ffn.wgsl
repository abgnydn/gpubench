// UNFUSED KERNEL 3: Feed-Forward Network (separate dispatch)

const D: u32 = 32u;
const DF: u32 = 128u;
const SL: u32 = 64u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> W1: array<f32>;
@group(0) @binding(3) var<storage, read> b1: array<f32>;
@group(0) @binding(4) var<storage, read> W2: array<f32>;
@group(0) @binding(5) var<storage, read> b2: array<f32>;
@group(0) @binding(6) var<uniform> u: vec4<u32>;  // x = current_seq_len

fn gelu(x: f32) -> f32 {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token = gid.x;
    if (token >= u.x) { return; }

    let base = token * D;

    var hidden: array<f32, 128>;
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
}
