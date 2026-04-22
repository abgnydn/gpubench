const e=`// GENERIC INT4 DEQUANT MATMUL — replaces 31 TVM shader variants.
//
// Computes: output[row] = dot(input[0..K-1], dequant(weights[row, 0..K-1]))
//   where dequant(packed) = (nibble - 7) * scale
//
// One workgroup = one output element (row).
// 64 threads cooperatively compute the dot product, then tree-reduce.
//
// Handles ANY dimension via uniforms:
//   K_PACKED  = K / 8  (number of u32 words per weight row)
//   SCALES_PER_ROW = K / 32 (number of scale values per weight row)
//   CHUNKS = K_PACKED / 64 (iterations per thread)
//
// Accumulates in f32 to avoid TVM's f16 precision loss.
//
// Bindings (match TVM convention):
//   @binding(0): output     array<f16>  (read_write) — matmul result
//   @binding(1): input      array<f16>  (read)       — input vector
//   @binding(2): scales     array<f16>  (read)       — per-group scales
//   @binding(3): weights    array<u32>  (read)       — int4 packed weights
//   @binding(4): podArgs    uniform     — dimensions

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,        // K / 8 (e.g. 384 for K=3072, 1024 for K=8192)
  SCALES_PER_ROW: u32,  // K / 32 (e.g. 96 for K=3072, 256 for K=8192)
  packGridDimX: u32     // number of output elements
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn int4_matmul(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(row) >= podArgs.packGridDimX) { return; }

  let K_PACKED : i32 = i32(podArgs.K_PACKED);
  let SCALES_PER_ROW : i32 = i32(podArgs.SCALES_PER_ROW);
  let tid : i32 = i32(threadIdx.x);

  // Each thread accumulates its portion of the dot product in f32
  var acc : f32 = 0.0;

  // Process K_PACKED / 64 chunks per thread
  for (var chunk : i32 = 0; chunk < K_PACKED / 64; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 64;
    let packed : u32 = weights[row * K_PACKED + w_offset];
    let scale : f32 = f32(scales[row * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    // Unpack 8 int4 values and accumulate
    acc = acc + f32(input_buf[base])     * (f32(((packed >>  0u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 1]) * (f32(((packed >>  4u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 2]) * (f32(((packed >>  8u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 3]) * (f32(((packed >> 12u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 4]) * (f32(((packed >> 16u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 5]) * (f32(((packed >> 20u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 6]) * (f32(((packed >> 24u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 7]) * (f32(((packed >> 28u) & 15u)) - 7.0) * scale;
  }

  // Tree reduction in f32 (fixes TVM's f16 precision loss)
  red_buf[tid] = acc;
  workgroupBarrier();

  if (tid < 32) { red_buf[tid] = red_buf[tid] + red_buf[tid + 32]; }
  workgroupBarrier();
  if (tid < 16) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16]; }
  workgroupBarrier();
  if (tid < 8) { red_buf[tid] = red_buf[tid] + red_buf[tid + 8]; }
  workgroupBarrier();
  if (tid < 4) { red_buf[tid] = red_buf[tid] + red_buf[tid + 4]; }
  workgroupBarrier();
  if (tid < 2) { red_buf[tid] = red_buf[tid] + red_buf[tid + 2]; }
  workgroupBarrier();
  if (tid < 1) { red_buf[tid] = red_buf[tid] + red_buf[tid + 1]; }
  workgroupBarrier();

  if (tid == 0) {
    output_buf[row] = f16(red_buf[0]);
  }
}
`,n=`// RMSNORM — normalize hidden state.
// output[i] = (input[i] / rms) * gamma[i]
// where rms = sqrt(mean(input^2) + eps)
//
// All accumulation in f32 (matches TVM's rms_norm2_kernel).
// D=3072, 64 threads, each handles 48 elements.

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> gamma : array<f16>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn rms_norm(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(batch) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);
  let base : i32 = batch * 3072;

  // Phase 1: compute sum of squares in f32
  var sum_sq : f32 = 0.0;
  for (var i : i32 = 0; i < 48; i = i + 1) {
    let idx : i32 = tid * 48 + i;
    let val : f32 = f32(input_buf[base + idx]);
    sum_sq = sum_sq + val * val;
  }

  // Tree reduce
  red_buf[tid] = sum_sq;
  workgroupBarrier();
  if (tid < 32) { red_buf[tid] = red_buf[tid] + red_buf[tid + 32]; }
  workgroupBarrier();
  if (tid < 16) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16]; }
  workgroupBarrier();
  if (tid < 8) { red_buf[tid] = red_buf[tid] + red_buf[tid + 8]; }
  workgroupBarrier();
  if (tid < 4) { red_buf[tid] = red_buf[tid] + red_buf[tid + 4]; }
  workgroupBarrier();
  if (tid < 2) { red_buf[tid] = red_buf[tid] + red_buf[tid + 2]; }
  workgroupBarrier();
  if (tid < 1) { red_buf[tid] = red_buf[tid] + red_buf[tid + 1]; }
  workgroupBarrier();

  let rms_inv : f32 = 1.0 / sqrt(red_buf[0] / 3072.0 + 1e-5);

  // Phase 2: normalize and scale
  for (var i : i32 = 0; i < 48; i = i + 1) {
    let idx : i32 = tid * 48 + i;
    output_buf[base + idx] = f16(f32(input_buf[base + idx]) * rms_inv * f32(gamma[idx]));
  }
}
`,r=`// ADD + RMSNORM — residual connection + normalize.
// residual[i] = A[i] + B[i]          (store to residual buffer)
// output[i] = rmsnorm(residual) * gamma[i]
//
// Fuses TVM's fuse_add_norm_decode_kernel.
// 256 threads, each handles 12 elements of D=3072.

enable f16;

@group(0) @binding(0) var<storage, read> A : array<f16>;
@group(0) @binding(1) var<storage, read> B : array<f16>;
@group(0) @binding(2) var<storage, read> gamma : array<f16>;
@group(0) @binding(3) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(4) var<storage, read_write> residual : array<f16>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn add_norm(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(batch) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);
  let base : i32 = batch * 3072;

  // Phase 1: add residual and compute local sum of squares
  var local_vals : array<f16, 12>;
  var sum_sq : f32 = 0.0;

  for (var i : i32 = 0; i < 12; i = i + 1) {
    let idx : i32 = tid + i * 256;
    let val : f16 = A[base + idx] + B[base + idx];
    local_vals[i] = val;
    residual[base + idx] = val;
    sum_sq = sum_sq + f32(val) * f32(val);
  }

  // Tree reduce
  red_buf[tid] = sum_sq;
  workgroupBarrier();
  if (tid < 128) { red_buf[tid] = red_buf[tid] + red_buf[tid + 128]; } workgroupBarrier();
  if (tid < 64) { red_buf[tid] = red_buf[tid] + red_buf[tid + 64]; } workgroupBarrier();
  if (tid < 32) { red_buf[tid] = red_buf[tid] + red_buf[tid + 32]; } workgroupBarrier();
  if (tid < 16) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16]; } workgroupBarrier();
  if (tid < 8) { red_buf[tid] = red_buf[tid] + red_buf[tid + 8]; } workgroupBarrier();
  if (tid < 4) { red_buf[tid] = red_buf[tid] + red_buf[tid + 4]; } workgroupBarrier();
  if (tid < 2) { red_buf[tid] = red_buf[tid] + red_buf[tid + 2]; } workgroupBarrier();
  if (tid < 1) { red_buf[tid] = red_buf[tid] + red_buf[tid + 1]; } workgroupBarrier();

  let rms_inv : f32 = 1.0 / sqrt(red_buf[0] / 3072.0 + 1e-5);

  // Phase 2: normalize
  for (var i : i32 = 0; i < 12; i = i + 1) {
    let idx : i32 = tid + i * 256;
    output_buf[base + idx] = f16(f32(local_vals[i]) * rms_inv * f32(gamma[idx]));
  }
}
`,a=`// ROPE — Rotary Position Embedding + QKV split.
//
// Input: QKV projection output [9216 f16] = Q[3072] + K[3072] + V[3072]
// Output: Q buffer [3072], K buffer [3072], V buffer [3072]
//
// Q and K get RoPE applied (sin/cos rotation of pairs).
// V is just copied.
//
// head_dim=96, so rotation pairs are at distance 48.
// theta = position / (10000 ^ (2i/96))

enable f16;

@group(0) @binding(0) var<storage, read_write> q_out : array<f16>;
@group(0) @binding(1) var<storage, read_write> k_out : array<f16>;
@group(0) @binding(2) var<storage, read_write> v_out : array<f16>;
@group(0) @binding(3) var<storage, read> qkv : array<f16>;
@group(0) @binding(4) var<storage, read> position_map : array<i32>;

struct PODArgs {
  apply_rope: i32,
  position_map_elem_offset: i32,
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn rope_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let global_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(global_id) >= podArgs.packGridDimX) { return; }

  // Decompose: global_id covers seq_len * 36 workgroups
  // 36 = 32 Q heads * 96/256 + ... = (32+32+32) heads * 96 / 256
  // Simpler: each element is one f16 in the QKV tensor
  let tid : i32 = i32(threadIdx.x);
  let flat : i32 = global_id * 256 + tid;

  let seq_idx : i32 = flat / 9216;
  let within : i32 = flat % 9216;

  let head_idx : i32 = within / 96;
  let dim_idx : i32 = within % 96;

  let qkv_val : f16 = qkv[flat];

  if (head_idx < 32) {
    // Q head — apply RoPE
    var out_val : f16 = qkv_val;
    if (podArgs.apply_rope != 0) {
      let pos : f32 = f32(position_map[seq_idx + podArgs.position_map_elem_offset]);
      let freq : f32 = pos / pow(10000.0, f32(((dim_idx % 48) * 2)) / 96.0);
      let cos_f : f32 = cos(freq);
      let sin_f : f32 = sin(freq);

      var pair_val : f16;
      if (dim_idx < 48) {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx + 48] * -1.0h;
      } else {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx - 48];
      }
      out_val = f16(cos_f * f32(qkv_val) + sin_f * f32(pair_val));
    }
    q_out[seq_idx * 3072 + head_idx * 96 + dim_idx] = out_val;
  } else if (head_idx < 64) {
    // K head — apply RoPE
    let k_head : i32 = head_idx - 32;
    var out_val : f16 = qkv_val;
    if (podArgs.apply_rope != 0) {
      let pos : f32 = f32(position_map[seq_idx + podArgs.position_map_elem_offset]);
      let freq : f32 = pos / pow(10000.0, f32(((dim_idx % 48) * 2)) / 96.0);
      let cos_f : f32 = cos(freq);
      let sin_f : f32 = sin(freq);

      var pair_val : f16;
      if (dim_idx < 48) {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx + 48] * -1.0h;
      } else {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx - 48];
      }
      out_val = f16(cos_f * f32(qkv_val) + sin_f * f32(pair_val));
    }
    k_out[seq_idx * 3072 + k_head * 96 + dim_idx] = out_val;
  } else {
    // V head — just copy
    let v_head : i32 = head_idx - 64;
    v_out[seq_idx * 3072 + v_head * 96 + dim_idx] = qkv_val;
  }
}
`,i=`// KV CACHE APPEND — write K,V vectors into paged cache.
//
// K and V are written to page[position] in the KV cache.
// Layout: pages[page_no * 98304 + head * 1536 + slot * 96 + dim]
//   where: 98304 = 32 heads * 16 slots * 96 dims * 2 (K+V)
//          1536 = 16 slots * 96 dims
//          K at offset 0, V at offset 49152 (= 32 * 1536)
//
// Matches TVM's tir_kv_cache_transpose_append_kernel.

enable f16;

@group(0) @binding(0) var<storage, read> k_data : array<f16>;
@group(0) @binding(1) var<storage, read> v_data : array<f16>;
@group(0) @binding(2) var<storage, read_write> pages : array<f16>;
@group(0) @binding(3) var<storage, read> position_map : array<i32>;

struct PODArgs {
  ntoken: i32,
  num_pages: i32,
  pages_elem_offset: i32,
  position_map_elem_offset: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn kv_append(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let global_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(global_id) >= podArgs.packGridDimX) { return; }

  let flat : i32 = global_id * 256 + i32(threadIdx.x);
  // flat covers ntoken * 32 heads * 96 dims = ntoken * 3072
  let token_idx : i32 = flat / 3072;
  let within : i32 = flat % 3072;
  let head : i32 = within / 96;
  let dim : i32 = within % 96;

  if (token_idx >= podArgs.ntoken) { return; }

  let position : i32 = position_map[token_idx + podArgs.position_map_elem_offset];
  if (position == -1) { return; }

  let page_no : i32 = position / 16;
  let slot : i32 = position % 16;

  // Write K
  let k_offset : i32 = page_no * 98304 + head * 1536 + slot * 96 + dim + podArgs.pages_elem_offset;
  pages[k_offset] = k_data[token_idx * 3072 + within];

  // Write V (offset by 49152 = 32 * 1536)
  let v_offset : i32 = k_offset + 49152;
  pages[v_offset] = v_data[token_idx * 3072 + within];
}
`,t=`// PAGED KV ATTENTION (DECODE) — FlashDecoding style.
//
// For single-token decode: Q has 1 token, K/V are in paged cache.
// One workgroup per (batch, head) pair.
// 32 threads, each owns 3 elements of head_dim=96.
//
// Algorithm: online softmax (FlashAttention)
//   for each page of KV cache:
//     for each slot in page:
//       score = dot(Q, K[slot]) * sm_scale
//       update running (max, sum, output) with online softmax
//   normalize output
//
// Pages layout: pages[page * 98304 + head * 1536 + slot * 96 + dim]
//   K at offset 0, V at offset 49152

enable f16;

@group(0) @binding(0) var<storage, read> Q : array<f16>;
@group(0) @binding(1) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(2) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(3) var<storage, read> pages : array<f16>;
@group(0) @binding(4) var<storage, read> length_info : array<i32>;
@group(0) @binding(5) var<storage, read_write> output_buf : array<f16>;

struct PODArgs {
  B: i32,
  max_num_pages: i32,
  nnz_pages: i32,
  pages_elem_offset: i32,
  page_indptr_elem_offset: i32,
  page_values_elem_offset: i32,
  length_info_elem_offset: i32,
  sm_scale: f32,
  packGridDimX: u32
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

var<workgroup> score_reduce : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn attention(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.B) { return; }

  // Each thread owns 3 elements of Q (32 threads × 3 = 96 = head_dim)
  var q0 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3]);
  var q1 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 1]);
  var q2 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 2]);

  // Page range for this batch
  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  // Online softmax state
  var m : f32 = -50000.0;
  var d : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;

  // Iterate over all KV positions
  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * 16;
    let slots_in_page : i32 = min(16, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + podArgs.pages_elem_offset;

      // Partial dot product: each thread computes 3 multiplies
      let partial : f32 = q0 * f32(pages[k_base + tid * 3])
                        + q1 * f32(pages[k_base + tid * 3 + 1])
                        + q2 * f32(pages[k_base + tid * 3 + 2]);

      // Tree reduction across 32 threads to get full dot product
      score_reduce[tid] = partial;
      workgroupBarrier();
      if (tid < 16) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 16]; }
      workgroupBarrier();
      if (tid < 8) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 8]; }
      workgroupBarrier();
      if (tid < 4) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 4]; }
      workgroupBarrier();
      if (tid < 2) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 2]; }
      workgroupBarrier();
      if (tid < 1) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 1]; }
      workgroupBarrier();

      let s : f32 = score_reduce[0] * podArgs.sm_scale;

      // Online softmax update
      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      // Load V and accumulate
      let v_base : i32 = k_base + 49152;
      o0 = o0 * scale_prev + scale_new * f32(pages[v_base + tid * 3]);
      o1 = o1 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 1]);
      o2 = o2 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 2]);
    }
  }

  // Normalize and write output
  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    output_buf[batch * 3072 + head * 96 + tid * 3] = f16(o0 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 1] = f16(o1 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 2] = f16(o2 * inv_d);
  }
}
`,d=`// FUSED FFN: gate+up int4 matmul + SiLU in ONE dispatch.
//
// Replaces 2 TVM dispatches:
//   fused_dequantize3_NT_matmul12_kernel (16384 workgroups, 64 threads)
//   fused_split2_silu2_multiply2_kernel  (32 workgroups, 256 threads)
// With: 8192 workgroups, 64 threads each
//
// Weight layout (confirmed from TVM source):
//   Rows 0..8191    = gate weights
//   Rows 8192..16383 = up weights
//   output[i] = SiLU(gate[i]) * up[i]
//
// CRITICAL: input and output are the SAME buffer (BUF#730 in decode).
// We cache the 3072 f16 input in shared memory to avoid the race condition.
//
// Bindings (match TVM's matmul pattern):
//   @binding(0): output  array<f16>  (read_write) — SiLU result, 8192 elements
//   @binding(1): input   array<f16>  (read)       — normed hidden state, 3072 elements
//   @binding(2): scales  array<f16>  (read)       — weight scales
//   @binding(3): weights array<u32>  (read)       — int4 packed weights
//   @binding(4): podArgs uniform     — {packGridDimX}

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

// Cache input (3072 f16 = 6KB) + reduction buffers for gate and up
var<workgroup> shared_input : array<f16, 3072>;
var<workgroup> red_gate : array<f16, 64>;
var<workgroup> red_up : array<f16, 64>;

@compute @workgroup_size(64, 1, 1)
fn fused_ffn_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let output_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);

  // Phase 1: Cooperatively load input into shared memory (48 elements per thread)
  for (var i : u32 = 0u; i < 48u; i = i + 1u) {
    let idx : u32 = threadIdx.x * 48u + i;
    if (idx < 3072u) {
      shared_input[idx] = input_buf[idx];
    }
  }
  workgroupBarrier();

  // Phase 2: Dual dot product — gate (row i) and up (row i+8192)
  let gate_row : i32 = output_idx;
  let up_row : i32 = output_idx + 8192i;
  let D_PACKED : i32 = 384i;       // 3072 / 8
  let SCALES_PER_ROW : i32 = 96i;  // 3072 / 32

  var gate_acc : f16 = 0.000000e+00h;
  var up_acc : f16 = 0.000000e+00h;

  // 6 chunks × 64 threads × 8 nibbles = 3072 elements (matches TVM's unrolled structure)
  for (var chunk : i32 = 0i; chunk < 6i; chunk = chunk + 1i) {
    let w_offset : i32 = i32(threadIdx.x) + chunk * 64i;

    let gate_packed : u32 = weights[gate_row * D_PACKED + w_offset];
    let gate_scale : f16 = scales[gate_row * SCALES_PER_ROW + (w_offset >> 2u)];

    let up_packed : u32 = weights[up_row * D_PACKED + w_offset];
    let up_scale : f16 = scales[up_row * SCALES_PER_ROW + (w_offset >> 2u)];

    let base : i32 = w_offset * 8i;

    // Unpack 8 int4 values, accumulate both gate and up using shared_input
    gate_acc = fma(shared_input[base],     (f16(((gate_packed >>  0u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 1], (f16(((gate_packed >>  4u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 2], (f16(((gate_packed >>  8u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 3], (f16(((gate_packed >> 12u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 4], (f16(((gate_packed >> 16u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 5], (f16(((gate_packed >> 20u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 6], (f16(((gate_packed >> 24u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 7], (f16(((gate_packed >> 28u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);

    up_acc = fma(shared_input[base],     (f16(((up_packed >>  0u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 1], (f16(((up_packed >>  4u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 2], (f16(((up_packed >>  8u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 3], (f16(((up_packed >> 12u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 4], (f16(((up_packed >> 16u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 5], (f16(((up_packed >> 20u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 6], (f16(((up_packed >> 24u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 7], (f16(((up_packed >> 28u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
  }

  // Phase 3: Tree reduction (64 → 1) for both gate and up
  red_gate[threadIdx.x] = gate_acc;
  red_up[threadIdx.x] = up_acc;
  workgroupBarrier();

  if (threadIdx.x < 32u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 32u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 32u]; }
  workgroupBarrier();
  if (threadIdx.x < 16u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 16u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 16u]; }
  workgroupBarrier();
  if (threadIdx.x < 8u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 8u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 8u]; }
  workgroupBarrier();
  if (threadIdx.x < 4u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 4u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 4u]; }
  workgroupBarrier();
  if (threadIdx.x < 2u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 2u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 2u]; }
  workgroupBarrier();
  if (threadIdx.x < 1u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 1u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 1u]; }
  workgroupBarrier();

  // Phase 4: SiLU(gate) * up — matches TVM's formula exactly
  if (threadIdx.x == 0u) {
    let gate_val : f32 = f32(red_gate[0]);
    let up_val : f16 = red_up[0];
    let silu_gate : f16 = f16(gate_val * (1.0 / (1.0 + exp(-gate_val))));
    output_buf[output_idx] = up_val * silu_gate;
  }
}
`,u=`// EMBEDDING — int4 dequant + token lookup.
// output[seq * 3072 + i] = dequant(embd_weight[token_id, i])
// Same int4 format: (nibble - 7) * scale, group_size=32
//
// Matches TVM's fused_dequantize_take1_kernel.

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_ids : array<i32>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn embedding(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let global_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(global_id) >= podArgs.packGridDimX) { return; }

  let flat : i32 = global_id * 256 + i32(threadIdx.x);
  // flat covers seq_len * 3072 elements
  // Each workgroup block covers 256 contiguous output elements
  // 3072 / 256 = 12 blocks per token

  let token_idx : i32 = flat / 3072;
  if (token_idx >= podArgs.seq_len) { return; }

  let dim : i32 = flat % 3072;
  let token_id : i32 = input_ids[token_idx];

  // Dequantize: 8 nibbles per u32, group_size=32
  let packed_idx : i32 = token_id * 384 + (dim / 8);  // 384 = 3072/8
  let nibble_idx : u32 = u32(dim % 8);
  let packed : u32 = weights[packed_idx];
  let scale : f16 = scales[token_id * 96 + (dim / 32)]; // 96 = 3072/32

  let nibble : u32 = (packed >> (nibble_idx * 4u)) & 15u;
  output_buf[flat] = (f16(nibble) - 7.0h) * scale;
}
`,o=`// ARGMAX — Find the index of the maximum value in a f32 array.
// Replaces TVM's 20-dispatch hierarchical argsort + sampling pipeline.
//
// Input:  logits array (f32, length = vocab_size)
// Output: single i32 token ID
//
// Strategy: parallel reduction in shared memory.
// 256 threads, each scans vocab_size/256 elements, then tree reduce.

@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> result : array<i32>;

struct Params {
  vocab_size: u32,
}
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> shared_val : array<f32, 256>;
var<workgroup> shared_idx : array<i32, 256>;

@compute @workgroup_size(256, 1, 1)
fn argmax_kernel(@builtin(local_invocation_id) tid : vec3<u32>) {
  let thread_id = tid.x;
  let vocab = params.vocab_size;
  let chunk = (vocab + 255u) / 256u;
  let start = thread_id * chunk;
  let end = min(start + chunk, vocab);

  // Phase 1: each thread finds max in its chunk
  var best_val : f32 = -1e30;
  var best_idx : i32 = 0;

  for (var i = start; i < end; i = i + 1u) {
    let v = logits[i];
    if (v > best_val) {
      best_val = v;
      best_idx = i32(i);
    }
  }

  shared_val[thread_id] = best_val;
  shared_idx[thread_id] = best_idx;
  workgroupBarrier();

  // Phase 2: tree reduction (256 → 128 → 64 → ... → 1)
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (thread_id < stride) {
      if (shared_val[thread_id + stride] > shared_val[thread_id]) {
        shared_val[thread_id] = shared_val[thread_id + stride];
        shared_idx[thread_id] = shared_idx[thread_id + stride];
      }
    }
    workgroupBarrier();
  }

  // Thread 0 writes result
  if (thread_id == 0u) {
    result[0] = shared_idx[0];
  }
}
`;export{t as a,o as b,r as c,n as d,u as e,d as f,e as i,i as k,a as r};
