import{b as N,i as R,f as U,c as F,a as $,k as X,r as j,d as Q,e as V}from"./argmax-B2Isa9Fr.js";import{i as H}from"./int4_matmul_f32-BY4OHOh5.js";import{q as Y}from"./qkv_fused-_Y_hop8R.js";const G="https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC/resolve/main/";async function Z(){var n;try{return typeof navigator>"u"||!((n=navigator.storage)!=null&&n.getDirectory)?null:await(await navigator.storage.getDirectory()).getDirectoryHandle("zero-tvm-weights",{create:!0})}catch{return null}}function T(n){return n.replace(/[^A-Za-z0-9._-]/g,"_")}async function J(n,a){if(!n)return null;try{return await(await(await n.getFileHandle(T(a))).getFile()).arrayBuffer()}catch{return null}}async function q(n,a,e){if(n)try{const o=await(await n.getFileHandle(T(a),{create:!0})).createWritable();await o.write(e),await o.close()}catch{}}async function L(n,a,e,t){const o=n.split("/").at(-1),d=await J(e,a);if(d)return t==null||t(`[opfs] ${o}`),d;try{const x=await caches.keys();for(const S of x){const k=await(await caches.open(S)).match(n);if(k){t==null||t(`[cache] ${o}`);const E=await k.arrayBuffer();return q(e,a,E),E}}}catch{}t==null||t(`[fetch] ${o}`);const h=await fetch(n);if(!h.ok)throw new Error(`HTTP ${h.status} fetching ${n}`);const m=await h.arrayBuffer();return q(e,a,m),m}function nn(n){const a=[];for(const e of n.records)if("records"in e&&Array.isArray(e.records))for(const t of e.records)a.push({...t,dataPath:t.dataPath??e.dataPath});else a.push(e);return a}const en=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;function tn(n,a,e){const t=n.createBuffer({size:Math.max(e.nbytes,4),usage:en,label:e.name});return n.queue.writeBuffer(t,0,new Uint8Array(a,e.byteOffset,e.nbytes)),t}async function Kn(n,a){const e=G;a==null||a("Loading ndarray-cache.json...");const t=await Z(),o=await L(e+"ndarray-cache.json","ndarray-cache.json",t,a),d=JSON.parse(new TextDecoder().decode(o)),h=nn(d);a==null||a(`Manifest: ${h.length} parameters`);const m=new Map;for(const i of h){const _=m.get(i.dataPath);_?_.push(i):m.set(i.dataPath,[i])}const x=new Map;let S=0;const K=m.size,k=h.reduce((i,_)=>i+_.nbytes,0);let E=0;await Promise.all([...m.entries()].map(async([i,_])=>{const c=await L(e+i,i,t,a);for(const P of _)x.set(P.name,tn(n,c,P));S++,E+=c.byteLength;const C=(E/1e6).toFixed(0),W=(k/1e6).toFixed(0);a==null||a(`[${S}/${K}] ${i} · ${C}/${W} MB`)}));function b(...i){for(const _ of i){const c=x.get(_);if(c)return c}throw new Error(`Weight not found. Tried: ${i.join(", ")}
Available: ${[...x.keys()].slice(0,20).join(", ")}`)}const D=b("transformer.embd.q_weight","embed_tokens.q_weight","model.embed_tokens.q_weight"),y=b("transformer.embd.q_scale","embed_tokens.q_scale","model.embed_tokens.q_scale"),u=b("transformer.h.0.ln.weight","model.layers.0.input_layernorm.weight"),r=b("lm_head.q_weight","model.lm_head.q_weight"),g=b("lm_head.q_scale","model.lm_head.q_scale"),v=b("transformer.norm.weight","model.norm.weight","norm.weight"),p=32,w=[];for(let i=0;i<p;i++){const _=`transformer.h.${i}`,c=`model.layers.${i}`;w.push({qkvWeights:b(`${_}.mixer.qkv_proj.q_weight`,`${c}.self_attn.qkv_proj.q_weight`),qkvScales:b(`${_}.mixer.qkv_proj.q_scale`,`${c}.self_attn.qkv_proj.q_scale`),oProjWeights:b(`${_}.mixer.out_proj.q_weight`,`${c}.self_attn.o_proj.q_weight`),oProjScales:b(`${_}.mixer.out_proj.q_scale`,`${c}.self_attn.o_proj.q_scale`),normGamma1:b(`${_}.ln.weight`,`${c}.input_layernorm.weight`),normGamma2:b(`${_}.post_attention_layernorm.weight`,`${c}.post_attention_layernorm.weight`),ffnWeights:b(`${_}.mlp.gate_up_proj.q_weight`,`${c}.mlp.gate_up_proj.q_weight`),ffnScales:b(`${_}.mlp.gate_up_proj.q_scale`,`${c}.mlp.gate_up_proj.q_scale`),ffnDownWeights:b(`${_}.mlp.down_proj.q_weight`,`${c}.mlp.down_proj.q_weight`),ffnDownScales:b(`${_}.mlp.down_proj.q_scale`,`${c}.mlp.down_proj.q_scale`)})}return a==null||a(`All weights loaded · ${(k/1e6).toFixed(0)} MB · ${t?"OPFS active":"OPFS unavailable"}`),{device:n,embdWeights:D,embdScales:y,lmHeadWeights:r,lmHeadScales:g,initNormGamma:u,finalNormGamma:v,layers:w}}async function an(n){try{const e=await caches.keys();for(const t of e){const d=await(await caches.open(t)).match(n);if(d)return d.text()}}catch{}const a=await fetch(n);if(!a.ok)throw new Error(`HTTP ${a.status} fetching tokenizer.json`);return a.text()}const M="▁",un=new RegExp(M,"g"),sn=/^<0x([0-9A-Fa-f]{2})>$/;async function Pn(n){n==null||n("Loading tokenizer.json...");const a=G+"tokenizer.json",e=JSON.parse(await an(a)),t=e.model.vocab,o=e.model.merges,d=new Array(Math.max(...Object.values(t))+1);for(const[u,r]of Object.entries(t))d[r]=u;for(const u of e.added_tokens??[])d[u.id]=u.content;const h=new Map;for(let u=0;u<o.length;u++)h.set(o[u],u);const m=new Map;for(const u of e.added_tokens??[])m.set(u.content,u.id);const x=t["<s>"]??1,S=t["</s>"]??2,K=new Map,k=new Set;for(let u=0;u<d.length;u++){const r=d[u];if(!r)continue;const g=sn.exec(r);if(g){K.set(u,parseInt(g[1],16));continue}if(r==="<s>"||r==="</s>"||r==="<pad>"){k.add(u);continue}r.startsWith("<|")&&r.endsWith("|>")&&k.add(u)}function E(u){if(u.length<=1)return u;for(;;){let r=1/0,g=-1;for(let p=0;p<u.length-1;p++){const w=u[p]+" "+u[p+1],i=h.get(w);i!==void 0&&i<r&&(r=i,g=p)}if(g===-1)break;const v=u[g]+u[g+1];u=[...u.slice(0,g),v,...u.slice(g+2)]}return u}function b(u){const r=[],g=[...m.keys()].sort((p,w)=>w.length-p.length).map(p=>p.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")).join("|"),v=g?u.split(new RegExp(`(${g})`)):[u];for(const p of v){if(!p)continue;const w=m.get(p);if(w!==void 0){r.push(w);continue}const i=p.split(/(\s+)/);let _=!0;for(const c of i){if(!c)continue;if(/^\s+$/.test(c)){_=!1;continue}const C=(_&&r.length===0?"":M)+c;_=!1;const W=[...C],P=E(W);for(const O of P){const I=t[O];if(I!==void 0)r.push(I);else for(const B of[...O]){const z=t[B]??t["<unk>"]??0;r.push(z)}}}}return r}const D=new TextDecoder("utf-8",{fatal:!1});function y(u){let r="",g=[];const v=()=>{g.length!==0&&(r+=D.decode(Uint8Array.from(g)),g=[])};for(const p of u){if(p<0||k.has(p)){v();continue}const w=K.get(p);if(w!==void 0){g.push(w);continue}const i=d[p];i&&(v(),r+=i)}return v(),r.replace(un," ").trimStart()}return n==null||n("Tokenizer ready"),{encode:b,decode:y,bosId:x,eosId:S}}function Rn(n,a){let e="";for(const t of n)t.role==="system"?e+=`<|system|>
${t.content}<|end|>
`:t.role==="user"?e+=`<|user|>
${t.content}<|end|>
`:e+=`<|assistant|>
${t.content}<|end|>
`;return e+=`<|assistant|>
`,a.encode(e)}const rn=`// INT4 DEQUANT MATMUL (subgroup variant) — same math as int4_matmul.wgsl
// but replaces the 64-thread / 6-barrier tree reduction with a single
// subgroupAdd over a 32-thread workgroup = one subgroup.
//
// Assumes subgroup size == 32 (gated in chat.ts via a shader probe; we fall
// back to the scalar shader otherwise).
//
// Memory traffic is unchanged (weight reads dominate the runtime budget);
// the win is eliminating barriers and the shared-memory reduction buffer,
// plus more FMAs per thread before the cross-thread combine.
//
// Bindings match int4_matmul.wgsl 1:1 so bind groups can be reused.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(row) >= podArgs.packGridDimX) { return; }

  let K_PACKED : i32 = i32(podArgs.K_PACKED);
  let SCALES_PER_ROW : i32 = i32(podArgs.SCALES_PER_ROW);
  let tid : i32 = i32(threadIdx.x);

  var acc : f32 = 0.0;

  // Each thread processes K_PACKED / 32 chunks (twice the chunks vs the
  // 64-thread scalar shader — offsets the single subgroup sum at the end).
  for (var chunk : i32 = 0; chunk < K_PACKED / 32; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 32;
    let packed : u32 = weights[row * K_PACKED + w_offset];
    let scale : f32 = f32(scales[row * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    acc = acc + f32(input_buf[base])     * (f32(((packed >>  0u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 1]) * (f32(((packed >>  4u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 2]) * (f32(((packed >>  8u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 3]) * (f32(((packed >> 12u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 4]) * (f32(((packed >> 16u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 5]) * (f32(((packed >> 20u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 6]) * (f32(((packed >> 24u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 7]) * (f32(((packed >> 28u) & 15u)) - 7.0) * scale;
  }

  // Single subgroup sum replaces the 6-barrier tree reduction.
  let sum : f32 = subgroupAdd(acc);

  if (tid == 0) {
    output_buf[row] = f16(sum);
  }
}
`,on=`// INT4 DEQUANT MATMUL (tiled subgroup variant) — produces ROWS_PER_WG=4
// output rows per workgroup. Each thread loads 8 f16 inputs per chunk once
// and re-uses them across the 4 weight rows, so input bandwidth drops 4x vs
// the scalar shader (which does one row per workgroup).
//
// Weight bandwidth is unchanged (each int4 byte is used exactly once), but
// input-vector bandwidth was the secondary hotspot: for QKV (K=3072, N=9216)
// the scalar shader re-reads the 6 KB input 9216 times → 54 MB of redundant
// traffic per matmul. Tiling across 4 rows cuts that to 13.5 MB.
//
// Subgroup size must be 32; gated in chat.ts.
//
// Bindings match int4_matmul.wgsl so bind groups are reused verbatim.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 4u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_tiled(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let tid : u32 = threadIdx.x;

  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;
  var acc2 : f32 = 0.0;
  var acc3 : f32 = 0.0;

  let r0 = row_base;
  let r1 = row_base + 1u;
  let r2 = row_base + 2u;
  let r3 = row_base + 3u;

  // 32 threads × 8 elements per chunk = 256 input elements per chunk step.
  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base : u32 = w_offset * 8u;

    // Load 8 inputs once, reuse across 4 rows.
    let i0 = f32(input_buf[base]);
    let i1 = f32(input_buf[base + 1u]);
    let i2 = f32(input_buf[base + 2u]);
    let i3 = f32(input_buf[base + 3u]);
    let i4 = f32(input_buf[base + 4u]);
    let i5 = f32(input_buf[base + 5u]);
    let i6 = f32(input_buf[base + 6u]);
    let i7 = f32(input_buf[base + 7u]);

    let scale_idx : u32 = w_offset >> 2u;

    // Row 0
    {
      let packed = weights[r0 * K_PACKED + w_offset];
      let s = f32(scales[r0 * SCALES_PER_ROW + scale_idx]);
      acc0 = acc0
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    // Row 1
    {
      let packed = weights[r1 * K_PACKED + w_offset];
      let s = f32(scales[r1 * SCALES_PER_ROW + scale_idx]);
      acc1 = acc1
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    // Row 2
    {
      let packed = weights[r2 * K_PACKED + w_offset];
      let s = f32(scales[r2 * SCALES_PER_ROW + scale_idx]);
      acc2 = acc2
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    // Row 3
    {
      let packed = weights[r3 * K_PACKED + w_offset];
      let s = f32(scales[r3 * SCALES_PER_ROW + scale_idx]);
      acc3 = acc3
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
  }

  // Four uniform subgroup sums (all 32 lanes always present and active).
  let sum0 = subgroupAdd(acc0);
  let sum1 = subgroupAdd(acc1);
  let sum2 = subgroupAdd(acc2);
  let sum3 = subgroupAdd(acc3);

  if (tid == 0u) {
    output_buf[r0] = f16(sum0);
    output_buf[r1] = f16(sum1);
    output_buf[r2] = f16(sum2);
    output_buf[r3] = f16(sum3);
  }
}
`,_n=`// INT4 DEQUANT MATMUL (tiled-8 subgroup variant) — 8 output rows per
// workgroup. Same binding layout as int4_matmul.wgsl / int4_matmul_tiled.wgsl.
//
// Why 8 instead of 4: for ffnDown (K=8192, N=3072) the 4-row tile still reads
// ~12.5 MB of input vector per call (one read per WG × 768 WGs × 16 KB). With
// 8 rows per WG, only 384 WGs fire and input DRAM traffic halves to ~6.3 MB —
// for this matmul that's ~24% of total weight+input bytes.
//
// For LM head (K=3072, N=32064), 8-row tiling saves ~25 MB of input reads out
// of 105 MB total bytes per dispatch (~24%).
//
// Each thread holds 8 f32 accumulators; with 32-thread subgroup that's well
// inside the M-series GPU register budget.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read>       input_buf  : array<f16>;
@group(0) @binding(2) var<storage, read>       scales     : array<f16>;
@group(0) @binding(3) var<storage, read>       weights    : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 8u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_tiled8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED       : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let tid : u32 = threadIdx.x;

  var a0 : f32 = 0.0; var a1 : f32 = 0.0; var a2 : f32 = 0.0; var a3 : f32 = 0.0;
  var a4 : f32 = 0.0; var a5 : f32 = 0.0; var a6 : f32 = 0.0; var a7 : f32 = 0.0;

  let r0 = row_base;      let r1 = row_base + 1u; let r2 = row_base + 2u; let r3 = row_base + 3u;
  let r4 = row_base + 4u; let r5 = row_base + 5u; let r6 = row_base + 6u; let r7 = row_base + 7u;

  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    let i0 = f32(input_buf[base]);
    let i1 = f32(input_buf[base + 1u]);
    let i2 = f32(input_buf[base + 2u]);
    let i3 = f32(input_buf[base + 3u]);
    let i4 = f32(input_buf[base + 4u]);
    let i5 = f32(input_buf[base + 5u]);
    let i6 = f32(input_buf[base + 6u]);
    let i7 = f32(input_buf[base + 7u]);

    // 8 rows unrolled. Factor the dequantise+multiply into a per-row lambda-ish
    // block; the compiler inlines it.
    { let p = weights[r0 * K_PACKED + w_offset]; let s = f32(scales[r0 * SCALES_PER_ROW + sc_idx]);
      a0 = a0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r1 * K_PACKED + w_offset]; let s = f32(scales[r1 * SCALES_PER_ROW + sc_idx]);
      a1 = a1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r2 * K_PACKED + w_offset]; let s = f32(scales[r2 * SCALES_PER_ROW + sc_idx]);
      a2 = a2
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r3 * K_PACKED + w_offset]; let s = f32(scales[r3 * SCALES_PER_ROW + sc_idx]);
      a3 = a3
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r4 * K_PACKED + w_offset]; let s = f32(scales[r4 * SCALES_PER_ROW + sc_idx]);
      a4 = a4
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r5 * K_PACKED + w_offset]; let s = f32(scales[r5 * SCALES_PER_ROW + sc_idx]);
      a5 = a5
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r6 * K_PACKED + w_offset]; let s = f32(scales[r6 * SCALES_PER_ROW + sc_idx]);
      a6 = a6
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r7 * K_PACKED + w_offset]; let s = f32(scales[r7 * SCALES_PER_ROW + sc_idx]);
      a7 = a7
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
  }

  let s0 = subgroupAdd(a0); let s1 = subgroupAdd(a1); let s2 = subgroupAdd(a2); let s3 = subgroupAdd(a3);
  let s4 = subgroupAdd(a4); let s5 = subgroupAdd(a5); let s6 = subgroupAdd(a6); let s7 = subgroupAdd(a7);

  if (tid == 0u) {
    output_buf[r0] = f16(s0); output_buf[r1] = f16(s1);
    output_buf[r2] = f16(s2); output_buf[r3] = f16(s3);
    output_buf[r4] = f16(s4); output_buf[r5] = f16(s5);
    output_buf[r6] = f16(s6); output_buf[r7] = f16(s7);
  }
}
`,dn=`// INT4 DEQUANT MATMUL WITH F32 OUTPUT (subgroup variant) — for the LM head.
// Same structure as int4_matmul_sg.wgsl but writes f32 logits.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f32>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_f32_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(row) >= podArgs.packGridDimX) { return; }

  let K_PACKED : i32 = i32(podArgs.K_PACKED);
  let SCALES_PER_ROW : i32 = i32(podArgs.SCALES_PER_ROW);
  let tid : i32 = i32(threadIdx.x);

  var acc : f32 = 0.0;

  for (var chunk : i32 = 0; chunk < K_PACKED / 32; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 32;
    let packed : u32 = weights[row * K_PACKED + w_offset];
    let scale : f32 = f32(scales[row * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    acc = acc + f32(input_buf[base])     * (f32(((packed >>  0u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 1]) * (f32(((packed >>  4u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 2]) * (f32(((packed >>  8u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 3]) * (f32(((packed >> 12u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 4]) * (f32(((packed >> 16u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 5]) * (f32(((packed >> 20u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 6]) * (f32(((packed >> 24u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 7]) * (f32(((packed >> 28u) & 15u)) - 7.0) * scale;
  }

  let sum : f32 = subgroupAdd(acc);

  if (tid == 0) {
    output_buf[row] = sum;
  }
}
`,fn=`// INT4 DEQUANT MATMUL WITH F32 OUTPUT (tiled subgroup variant).
// Same design as int4_matmul_tiled.wgsl but writes f32 logits for the LM head.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f32>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 4u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_f32_tiled(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let tid : u32 = threadIdx.x;

  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;
  var acc2 : f32 = 0.0;
  var acc3 : f32 = 0.0;

  let r0 = row_base;
  let r1 = row_base + 1u;
  let r2 = row_base + 2u;
  let r3 = row_base + 3u;

  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base : u32 = w_offset * 8u;

    let i0 = f32(input_buf[base]);
    let i1 = f32(input_buf[base + 1u]);
    let i2 = f32(input_buf[base + 2u]);
    let i3 = f32(input_buf[base + 3u]);
    let i4 = f32(input_buf[base + 4u]);
    let i5 = f32(input_buf[base + 5u]);
    let i6 = f32(input_buf[base + 6u]);
    let i7 = f32(input_buf[base + 7u]);

    let scale_idx : u32 = w_offset >> 2u;

    {
      let packed = weights[r0 * K_PACKED + w_offset];
      let s = f32(scales[r0 * SCALES_PER_ROW + scale_idx]);
      acc0 = acc0
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    {
      let packed = weights[r1 * K_PACKED + w_offset];
      let s = f32(scales[r1 * SCALES_PER_ROW + scale_idx]);
      acc1 = acc1
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    {
      let packed = weights[r2 * K_PACKED + w_offset];
      let s = f32(scales[r2 * SCALES_PER_ROW + scale_idx]);
      acc2 = acc2
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    {
      let packed = weights[r3 * K_PACKED + w_offset];
      let s = f32(scales[r3 * SCALES_PER_ROW + scale_idx]);
      acc3 = acc3
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
  }

  let sum0 = subgroupAdd(acc0);
  let sum1 = subgroupAdd(acc1);
  let sum2 = subgroupAdd(acc2);
  let sum3 = subgroupAdd(acc3);

  if (tid == 0u) {
    output_buf[r0] = sum0;
    output_buf[r1] = sum1;
    output_buf[r2] = sum2;
    output_buf[r3] = sum3;
  }
}
`,ln=`// INT4 DEQUANT MATMUL WITH F32 OUTPUT (tiled-8 subgroup variant).
// Same design as int4_matmul_tiled8.wgsl but writes f32 logits for the LM head.
// 8 rows per WG × 32 threads per WG.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f32>;
@group(0) @binding(1) var<storage, read>       input_buf  : array<f16>;
@group(0) @binding(2) var<storage, read>       scales     : array<f16>;
@group(0) @binding(3) var<storage, read>       weights    : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 8u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_f32_tiled8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED       : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let tid : u32 = threadIdx.x;

  var a0 : f32 = 0.0; var a1 : f32 = 0.0; var a2 : f32 = 0.0; var a3 : f32 = 0.0;
  var a4 : f32 = 0.0; var a5 : f32 = 0.0; var a6 : f32 = 0.0; var a7 : f32 = 0.0;

  let r0 = row_base;      let r1 = row_base + 1u; let r2 = row_base + 2u; let r3 = row_base + 3u;
  let r4 = row_base + 4u; let r5 = row_base + 5u; let r6 = row_base + 6u; let r7 = row_base + 7u;

  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    let i0 = f32(input_buf[base]);
    let i1 = f32(input_buf[base + 1u]);
    let i2 = f32(input_buf[base + 2u]);
    let i3 = f32(input_buf[base + 3u]);
    let i4 = f32(input_buf[base + 4u]);
    let i5 = f32(input_buf[base + 5u]);
    let i6 = f32(input_buf[base + 6u]);
    let i7 = f32(input_buf[base + 7u]);

    { let p = weights[r0 * K_PACKED + w_offset]; let s = f32(scales[r0 * SCALES_PER_ROW + sc_idx]);
      a0 = a0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r1 * K_PACKED + w_offset]; let s = f32(scales[r1 * SCALES_PER_ROW + sc_idx]);
      a1 = a1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r2 * K_PACKED + w_offset]; let s = f32(scales[r2 * SCALES_PER_ROW + sc_idx]);
      a2 = a2
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r3 * K_PACKED + w_offset]; let s = f32(scales[r3 * SCALES_PER_ROW + sc_idx]);
      a3 = a3
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r4 * K_PACKED + w_offset]; let s = f32(scales[r4 * SCALES_PER_ROW + sc_idx]);
      a4 = a4
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r5 * K_PACKED + w_offset]; let s = f32(scales[r5 * SCALES_PER_ROW + sc_idx]);
      a5 = a5
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r6 * K_PACKED + w_offset]; let s = f32(scales[r6 * SCALES_PER_ROW + sc_idx]);
      a6 = a6
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r7 * K_PACKED + w_offset]; let s = f32(scales[r7 * SCALES_PER_ROW + sc_idx]);
      a7 = a7
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
  }

  let s0 = subgroupAdd(a0); let s1 = subgroupAdd(a1); let s2 = subgroupAdd(a2); let s3 = subgroupAdd(a3);
  let s4 = subgroupAdd(a4); let s5 = subgroupAdd(a5); let s6 = subgroupAdd(a6); let s7 = subgroupAdd(a7);

  if (tid == 0u) {
    output_buf[r0] = s0; output_buf[r1] = s1; output_buf[r2] = s2; output_buf[r3] = s3;
    output_buf[r4] = s4; output_buf[r5] = s5; output_buf[r6] = s6; output_buf[r7] = s7;
  }
}
`,pn=`// INT4 DEQUANT MATMUL — batched (M=4) variant of int4_matmul_tiled.wgsl.
//
// Shape: input [M, K] × weights [N, K] → output [M, N]    with M fixed at 4.
// Each WG computes a TILE_M × ROWS_PER_WG = 4 × 4 = 16 output cells:
//   - Loads ROWS_PER_WG weight rows (and their scales) ONCE per K-chunk
//     and reuses them across all TILE_M input rows — this is the point.
//     Running total-weight-bytes for M=4 forwards becomes 1×W instead of
//     4×W, which is the enabler for prompt-lookup spec decoding on
//     memory-bandwidth-bound hardware.
//   - Loads TILE_M × 8 f32 inputs per chunk and reuses across all 4 weight
//     rows — same input-amortization argument as the non-batched shader.
//
// Output layout: output_buf[m * N + n] for batch row m, output column n.
//                Column N is podArgs.packGridDimX (matches existing kernels).
// Input layout:  input_buf[m * K + k] — the engine strides M copies of
//                the activation vector contiguously.
//
// Bindings match int4_matmul_tiled.wgsl exactly; sizes of input_buf and
// output_buf are M× larger (the engine owns buffer sizing).
//
// Requires sg_size = 32; gated in chat.ts.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read>       input_buf  : array<f16>;
@group(0) @binding(2) var<storage, read>       scales     : array<f16>;
@group(0) @binding(3) var<storage, read>       weights    : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32,   // N (number of output cols)
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 4u;
const TILE_M      : u32 = 4u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_batched_m4(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED       : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let K              : u32 = K_PACKED * 8u;          // elements per input row
  let tid : u32 = threadIdx.x;

  // 16 f32 accumulators per thread: 4 batch rows × 4 output rows.
  var a00 : f32 = 0.0; var a01 : f32 = 0.0; var a02 : f32 = 0.0; var a03 : f32 = 0.0;
  var a10 : f32 = 0.0; var a11 : f32 = 0.0; var a12 : f32 = 0.0; var a13 : f32 = 0.0;
  var a20 : f32 = 0.0; var a21 : f32 = 0.0; var a22 : f32 = 0.0; var a23 : f32 = 0.0;
  var a30 : f32 = 0.0; var a31 : f32 = 0.0; var a32 : f32 = 0.0; var a33 : f32 = 0.0;

  let r0 = row_base;      let r1 = row_base + 1u;
  let r2 = row_base + 2u; let r3 = row_base + 3u;

  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    // Load 8 f32 inputs for EACH of the 4 batch rows (32 inputs per thread).
    let b0_0 = f32(input_buf[0u * K + base     ]);
    let b0_1 = f32(input_buf[0u * K + base + 1u]);
    let b0_2 = f32(input_buf[0u * K + base + 2u]);
    let b0_3 = f32(input_buf[0u * K + base + 3u]);
    let b0_4 = f32(input_buf[0u * K + base + 4u]);
    let b0_5 = f32(input_buf[0u * K + base + 5u]);
    let b0_6 = f32(input_buf[0u * K + base + 6u]);
    let b0_7 = f32(input_buf[0u * K + base + 7u]);
    let b1_0 = f32(input_buf[1u * K + base     ]);
    let b1_1 = f32(input_buf[1u * K + base + 1u]);
    let b1_2 = f32(input_buf[1u * K + base + 2u]);
    let b1_3 = f32(input_buf[1u * K + base + 3u]);
    let b1_4 = f32(input_buf[1u * K + base + 4u]);
    let b1_5 = f32(input_buf[1u * K + base + 5u]);
    let b1_6 = f32(input_buf[1u * K + base + 6u]);
    let b1_7 = f32(input_buf[1u * K + base + 7u]);
    let b2_0 = f32(input_buf[2u * K + base     ]);
    let b2_1 = f32(input_buf[2u * K + base + 1u]);
    let b2_2 = f32(input_buf[2u * K + base + 2u]);
    let b2_3 = f32(input_buf[2u * K + base + 3u]);
    let b2_4 = f32(input_buf[2u * K + base + 4u]);
    let b2_5 = f32(input_buf[2u * K + base + 5u]);
    let b2_6 = f32(input_buf[2u * K + base + 6u]);
    let b2_7 = f32(input_buf[2u * K + base + 7u]);
    let b3_0 = f32(input_buf[3u * K + base     ]);
    let b3_1 = f32(input_buf[3u * K + base + 1u]);
    let b3_2 = f32(input_buf[3u * K + base + 2u]);
    let b3_3 = f32(input_buf[3u * K + base + 3u]);
    let b3_4 = f32(input_buf[3u * K + base + 4u]);
    let b3_5 = f32(input_buf[3u * K + base + 5u]);
    let b3_6 = f32(input_buf[3u * K + base + 6u]);
    let b3_7 = f32(input_buf[3u * K + base + 7u]);

    // For each of the 4 output rows: load (packed, scale) ONCE, dequantize
    // 8 weights, and multiply-accumulate into 4 batch rows. 4 × 4 = 16 FMAs
    // per row per chunk iteration.
    {
      let p = weights[r0 * K_PACKED + w_offset];
      let s = f32(scales[r0 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a00 = a00 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a10 = a10 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a20 = a20 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a30 = a30 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
    {
      let p = weights[r1 * K_PACKED + w_offset];
      let s = f32(scales[r1 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a01 = a01 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a11 = a11 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a21 = a21 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a31 = a31 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
    {
      let p = weights[r2 * K_PACKED + w_offset];
      let s = f32(scales[r2 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a02 = a02 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a12 = a12 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a22 = a22 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a32 = a32 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
    {
      let p = weights[r3 * K_PACKED + w_offset];
      let s = f32(scales[r3 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a03 = a03 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a13 = a13 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a23 = a23 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a33 = a33 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
  }

  let s00 = subgroupAdd(a00); let s01 = subgroupAdd(a01);
  let s02 = subgroupAdd(a02); let s03 = subgroupAdd(a03);
  let s10 = subgroupAdd(a10); let s11 = subgroupAdd(a11);
  let s12 = subgroupAdd(a12); let s13 = subgroupAdd(a13);
  let s20 = subgroupAdd(a20); let s21 = subgroupAdd(a21);
  let s22 = subgroupAdd(a22); let s23 = subgroupAdd(a23);
  let s30 = subgroupAdd(a30); let s31 = subgroupAdd(a31);
  let s32 = subgroupAdd(a32); let s33 = subgroupAdd(a33);

  if (tid == 0u) {
    let N = podArgs.packGridDimX;
    output_buf[0u * N + r0] = f16(s00); output_buf[0u * N + r1] = f16(s01);
    output_buf[0u * N + r2] = f16(s02); output_buf[0u * N + r3] = f16(s03);
    output_buf[1u * N + r0] = f16(s10); output_buf[1u * N + r1] = f16(s11);
    output_buf[1u * N + r2] = f16(s12); output_buf[1u * N + r3] = f16(s13);
    output_buf[2u * N + r0] = f16(s20); output_buf[2u * N + r1] = f16(s21);
    output_buf[2u * N + r2] = f16(s22); output_buf[2u * N + r3] = f16(s23);
    output_buf[3u * N + r0] = f16(s30); output_buf[3u * N + r1] = f16(s31);
    output_buf[3u * N + r2] = f16(s32); output_buf[3u * N + r3] = f16(s33);
  }
}
`,cn=`// QKV_FUSED_SG — subgroup-reduction variant of qkv_fused.wgsl.
//
// Same fusion (QKV matmul + RoPE + KV append in one dispatch) and same bind
// group layout. The only difference is the tail reduction: the scalar variant
// does a 64→1 tree reduction with 5 workgroupBarriers; this variant does one
// subgroupAdd per subgroup (two subgroups of 32 lanes on Apple Metal-3), then a
// single workgroupBarrier + lane-0 finalisation.
//
// Requires sg_size = 32 (gated in chat.ts).

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;
@group(0) @binding(1) var<storage, read_write> kv_pages     : array<f16>;
@group(0) @binding(2) var<storage, read>       hidden       : array<f16>;
@group(0) @binding(3) var<storage, read>       scales       : array<f16>;
@group(0) @binding(4) var<storage, read>       weights      : array<u32>;
@group(0) @binding(5) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,
  packGridDimX             : u32,
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const K_PACKED       : i32 = 384;
const SCALES_PER_ROW : i32 = 96;

// Two subgroups per workgroup (64 threads / 32 lanes). Store per-subgroup
// partial sums so lane 0 can finalise.
var<workgroup> sg_sum0 : array<f32, 2>;
var<workgroup> sg_sum1 : array<f32, 2>;

@compute @workgroup_size(64, 1, 1)
fn qkv_fused_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lane : u32,
  @builtin(subgroup_size) sg_size : u32,
) {
  let pair_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(pair_idx) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  let group         : i32 = pair_idx / 1536;
  let pair_in_group : i32 = pair_idx - group * 1536;
  let head          : i32 = pair_in_group / 48;
  let dim_lo        : i32 = pair_in_group - head * 48;

  let row_lo : i32 = group * 3072 + head * 96 + dim_lo;
  let row_hi : i32 = row_lo + 48;

  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;

  for (var chunk : i32 = 0; chunk < K_PACKED / 64; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 64;
    let packed_lo : u32 = weights[row_lo * K_PACKED + w_offset];
    let packed_hi : u32 = weights[row_hi * K_PACKED + w_offset];
    let scale_lo : f32 = f32(scales[row_lo * SCALES_PER_ROW + (w_offset >> 2)]);
    let scale_hi : f32 = f32(scales[row_hi * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    let x0 = f32(hidden[base    ]);
    let x1 = f32(hidden[base + 1]);
    let x2 = f32(hidden[base + 2]);
    let x3 = f32(hidden[base + 3]);
    let x4 = f32(hidden[base + 4]);
    let x5 = f32(hidden[base + 5]);
    let x6 = f32(hidden[base + 6]);
    let x7 = f32(hidden[base + 7]);

    acc0 = acc0 + x0 * (f32((packed_lo >>  0u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x1 * (f32((packed_lo >>  4u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x2 * (f32((packed_lo >>  8u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x3 * (f32((packed_lo >> 12u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x4 * (f32((packed_lo >> 16u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x5 * (f32((packed_lo >> 20u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x6 * (f32((packed_lo >> 24u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x7 * (f32((packed_lo >> 28u) & 15u) - 7.0) * scale_lo;

    acc1 = acc1 + x0 * (f32((packed_hi >>  0u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x1 * (f32((packed_hi >>  4u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x2 * (f32((packed_hi >>  8u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x3 * (f32((packed_hi >> 12u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x4 * (f32((packed_hi >> 16u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x5 * (f32((packed_hi >> 20u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x6 * (f32((packed_hi >> 24u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x7 * (f32((packed_hi >> 28u) & 15u) - 7.0) * scale_hi;
  }

  // One subgroup-wide add per accumulator (all 32 lanes hold the same sum).
  let partial0 : f32 = subgroupAdd(acc0);
  let partial1 : f32 = subgroupAdd(acc1);

  let sg_id : u32 = u32(tid) / sg_size;  // 0 or 1 with sg_size=32, 64 threads
  if (sg_lane == 0u) {
    sg_sum0[sg_id] = partial0;
    sg_sum1[sg_id] = partial1;
  }
  workgroupBarrier();

  if (tid != 0) { return; }

  let v_lo : f32 = sg_sum0[0] + sg_sum0[1];
  let v_hi : f32 = sg_sum1[0] + sg_sum1[1];

  if (group == 2) {
    let position : i32 = position_map[podArgs.position_map_elem_offset];
    let page_no : i32 = position / 16;
    let slot : i32 = position - page_no * 16;
    let v_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + 49152
                       + podArgs.pages_elem_offset;
    kv_pages[v_base + dim_lo]      = f16(v_lo);
    kv_pages[v_base + dim_lo + 48] = f16(v_hi);
    return;
  }

  let pos  : f32 = f32(position_map[podArgs.position_map_elem_offset]);
  let freq : f32 = pos / pow(10000.0, f32(dim_lo * 2) / 96.0);
  let c    : f32 = cos(freq);
  let s    : f32 = sin(freq);

  let rot_lo : f32 = c * v_lo + s * (-v_hi);
  let rot_hi : f32 = c * v_hi + s * ( v_lo);

  if (group == 0) {
    let base = head * 96 + dim_lo;
    q_out[base     ] = f16(rot_lo);
    q_out[base + 48] = f16(rot_hi);
  } else {
    let position : i32 = position_map[podArgs.position_map_elem_offset];
    let page_no : i32 = position / 16;
    let slot : i32 = position - page_no * 16;
    let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96
                       + podArgs.pages_elem_offset;
    kv_pages[k_base + dim_lo]      = f16(rot_lo);
    kv_pages[k_base + dim_lo + 48] = f16(rot_hi);
  }
}
`,gn=`// QKV_FUSED_TILED_SG — tiled + subgroup variant of qkv_fused_sg.wgsl.
//
// NEGATIVE RESULT on Apple M-series: both 4-pair (57% slower) and 2-pair
// (78% slower on the qkv kernel) versions regressed vs qkv_fused_sg. See
// BENCH.md "Negative result: tiled qkv_fused" for numbers. Kept compiled
// behind \`?qkvtile=1\` for re-testing on other GPUs; default stays \`_sg\`.
//
// The shader below is the 2-pair variant, kept as the cleanest reference
// implementation of the tiled pattern (4 output rows, 4 accumulators/thread,
// 2304 WGs, 1 subgroup of 32 threads, shared-mem input cache for the 3072
// f16 hidden vector). Bind-group layout matches qkv_fused_sg.wgsl so chat.ts
// can swap the pipeline reference.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;
@group(0) @binding(1) var<storage, read_write> kv_pages     : array<f16>;
@group(0) @binding(2) var<storage, read>       hidden       : array<f16>;
@group(0) @binding(3) var<storage, read>       scales       : array<f16>;
@group(0) @binding(4) var<storage, read>       weights      : array<u32>;
@group(0) @binding(5) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,
  packGridDimX             : u32,
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const K_PACKED       : u32 = 384u;
const SCALES_PER_ROW : u32 = 96u;
const PAIRS_PER_WG   : u32 = 2u;

var<workgroup> shared_input : array<f16, 3072>;

@compute @workgroup_size(32, 1, 1)
fn qkv_fused_tiled_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let wg_idx : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let pair_base : u32 = wg_idx * PAIRS_PER_WG;
  if (pair_base >= podArgs.packGridDimX) { return; }

  let tid : u32 = threadIdx.x;

  // Snapshot-load input → shared mem. 32 threads × 96 elements = 3072.
  for (var i : u32 = 0u; i < 96u; i = i + 1u) {
    let idx : u32 = tid + i * 32u;
    shared_input[idx] = hidden[idx];
  }
  workgroupBarrier();

  // All 4 pairs in this WG share group and head (PAIRS_PER_WG=4 divides
  // 48 = pairs per head, so consecutive pair_idx stay in one head).
  let group_id : u32 = pair_base / 1536u;
  let pair_in_group : u32 = pair_base - group_id * 1536u;
  let head : u32 = pair_in_group / 48u;
  let dim_lo_0 : u32 = pair_in_group - head * 48u;

  let row_base : u32 = group_id * 3072u + head * 96u + dim_lo_0;
  // 4 rows: 2 "lo" + 2 "hi" (hi = lo + 48).
  let rl0 = row_base;  let rl1 = row_base + 1u;
  let rh0 = rl0 + 48u; let rh1 = rl1 + 48u;

  var acc_l0 : f32 = 0.0; var acc_l1 : f32 = 0.0;
  var acc_h0 : f32 = 0.0; var acc_h1 : f32 = 0.0;

  // K_PACKED / 32 = 12 chunks.
  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    let i0 = f32(shared_input[base     ]);
    let i1 = f32(shared_input[base + 1u]);
    let i2 = f32(shared_input[base + 2u]);
    let i3 = f32(shared_input[base + 3u]);
    let i4 = f32(shared_input[base + 4u]);
    let i5 = f32(shared_input[base + 5u]);
    let i6 = f32(shared_input[base + 6u]);
    let i7 = f32(shared_input[base + 7u]);

    { let p = weights[rl0 * K_PACKED + w_offset]; let s = f32(scales[rl0 * SCALES_PER_ROW + sc_idx]);
      acc_l0 = acc_l0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[rl1 * K_PACKED + w_offset]; let s = f32(scales[rl1 * SCALES_PER_ROW + sc_idx]);
      acc_l1 = acc_l1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[rh0 * K_PACKED + w_offset]; let s = f32(scales[rh0 * SCALES_PER_ROW + sc_idx]);
      acc_h0 = acc_h0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[rh1 * K_PACKED + w_offset]; let s = f32(scales[rh1 * SCALES_PER_ROW + sc_idx]);
      acc_h1 = acc_h1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
  }

  let vl0 = subgroupAdd(acc_l0); let vl1 = subgroupAdd(acc_l1);
  let vh0 = subgroupAdd(acc_h0); let vh1 = subgroupAdd(acc_h1);

  if (tid != 0u) { return; }

  // Per-pair write-back (2 pairs, each with its own RoPE / KV-append path).
  // Both pairs share group_id / head; only dim_lo differs.
  let vl = array<f32, 2>(vl0, vl1);
  let vh = array<f32, 2>(vh0, vh1);

  let position : i32 = position_map[podArgs.position_map_elem_offset];
  let pos_f : f32 = f32(position);
  let page_no : i32 = position / 16;
  let slot    : i32 = position - page_no * 16;

  for (var k : u32 = 0u; k < PAIRS_PER_WG; k = k + 1u) {
    let dim_lo : u32 = dim_lo_0 + k;
    let v_lo : f32 = vl[k];
    let v_hi : f32 = vh[k];

    if (group_id == 2u) {
      // V: no RoPE; store raw.
      let v_base : i32 = page_no * 98304 + i32(head) * 1536 + slot * 96 + 49152
                         + podArgs.pages_elem_offset;
      kv_pages[v_base + i32(dim_lo)     ] = f16(v_lo);
      kv_pages[v_base + i32(dim_lo) + 48] = f16(v_hi);
      continue;
    }

    let freq : f32 = pos_f / pow(10000.0, f32(dim_lo * 2u) / 96.0);
    let c    : f32 = cos(freq);
    let s    : f32 = sin(freq);
    let rot_lo : f32 = c * v_lo + s * (-v_hi);
    let rot_hi : f32 = c * v_hi + s * ( v_lo);

    if (group_id == 0u) {
      let b = i32(head) * 96 + i32(dim_lo);
      q_out[b     ] = f16(rot_lo);
      q_out[b + 48] = f16(rot_hi);
    } else {
      // K
      let k_base : i32 = page_no * 98304 + i32(head) * 1536 + slot * 96
                         + podArgs.pages_elem_offset;
      kv_pages[k_base + i32(dim_lo)     ] = f16(rot_lo);
      kv_pages[k_base + i32(dim_lo) + 48] = f16(rot_hi);
    }
  }
}
`,bn=`// QKV_FUSED_TILED_2SG — 2-subgroup, 2-pair tile variant.
//
// Motivation: previous qkv_fused_tiled_sg.wgsl tried 2 pairs / 32 threads /
// 2304 WGs; threads-in-flight dropped 4× vs the \`_sg\` baseline (9216 → 2304
// active warps) and we regressed 20%+. This variant keeps 64 threads per WG
// (2 subgroups) so threads-in-flight only halve (9216 → 4608).
//
// Layout:
//   - 64 threads per WG, 2 subgroups of 32
//   - 2 pairs per WG = 4 output rows
//   - Subgroup 0 computes pair 0 (rows rl0, rh0)
//   - Subgroup 1 computes pair 1 (rows rl1, rh1)
//   - Input cached into workgroup shared memory (3072 f16 = 6 KB, fits)
//
// If this still regresses, the 22% gap vs WebLLM is definitively not
// reachable via hand-tuned QKV tiling on Apple.
//
// Bind-group layout matches qkv_fused_sg.wgsl so chat.ts can swap pipelines.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;
@group(0) @binding(1) var<storage, read_write> kv_pages     : array<f16>;
@group(0) @binding(2) var<storage, read>       hidden       : array<f16>;
@group(0) @binding(3) var<storage, read>       scales       : array<f16>;
@group(0) @binding(4) var<storage, read>       weights      : array<u32>;
@group(0) @binding(5) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,
  packGridDimX             : u32,
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const K_PACKED       : u32 = 384u;
const SCALES_PER_ROW : u32 = 96u;
const PAIRS_PER_WG   : u32 = 2u;

var<workgroup> shared_input : array<f16, 3072>;

@compute @workgroup_size(64, 1, 1)
fn qkv_fused_tiled2sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lane : u32,
  @builtin(subgroup_size) sg_size : u32,
) {
  let wg_idx : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let pair_base : u32 = wg_idx * PAIRS_PER_WG;
  if (pair_base >= podArgs.packGridDimX) { return; }

  let tid : u32 = threadIdx.x;
  let sg_id : u32 = tid / sg_size;  // 0 or 1 (assumes sg_size == 32)

  // Cooperative load of hidden[0..3072] into shared mem.
  // 64 threads × 48 elements = 3072.
  for (var i : u32 = 0u; i < 48u; i = i + 1u) {
    let idx : u32 = tid + i * 64u;
    shared_input[idx] = hidden[idx];
  }
  workgroupBarrier();

  // Both pairs share group and head (PAIRS_PER_WG=2 divides 48 pairs/head).
  let group_id : u32 = pair_base / 1536u;
  let pair_in_group : u32 = pair_base - group_id * 1536u;
  let head : u32 = pair_in_group / 48u;
  let dim_lo_0 : u32 = pair_in_group - head * 48u;

  let row_base : u32 = group_id * 3072u + head * 96u + dim_lo_0;
  // Subgroup 0 owns pair 0 (rows row_base+0, row_base+48)
  // Subgroup 1 owns pair 1 (rows row_base+1, row_base+49)
  let row_offset : u32 = sg_id;  // 0 or 1
  let my_rl : u32 = row_base + row_offset;
  let my_rh : u32 = my_rl + 48u;

  var acc_lo : f32 = 0.0;
  var acc_hi : f32 = 0.0;

  // Each subgroup covers all K_PACKED / 32 = 12 chunks with its 32 lanes.
  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = sg_lane + chunk * 32u;
    let base : u32 = w_offset * 8u;
    let sc_idx : u32 = w_offset >> 2u;

    let i0 = f32(shared_input[base     ]);
    let i1 = f32(shared_input[base + 1u]);
    let i2 = f32(shared_input[base + 2u]);
    let i3 = f32(shared_input[base + 3u]);
    let i4 = f32(shared_input[base + 4u]);
    let i5 = f32(shared_input[base + 5u]);
    let i6 = f32(shared_input[base + 6u]);
    let i7 = f32(shared_input[base + 7u]);

    {
      let p = weights[my_rl * K_PACKED + w_offset];
      let s = f32(scales[my_rl * SCALES_PER_ROW + sc_idx]);
      acc_lo = acc_lo
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    {
      let p = weights[my_rh * K_PACKED + w_offset];
      let s = f32(scales[my_rh * SCALES_PER_ROW + sc_idx]);
      acc_hi = acc_hi
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
  }

  let v_lo : f32 = subgroupAdd(acc_lo);
  let v_hi : f32 = subgroupAdd(acc_hi);

  // Lane 0 of each subgroup writes its pair's output.
  if (sg_lane != 0u) { return; }

  let dim_lo : u32 = dim_lo_0 + row_offset;

  let position : i32 = position_map[podArgs.position_map_elem_offset];
  let page_no : i32 = position / 16;
  let slot    : i32 = position - page_no * 16;

  if (group_id == 2u) {
    let v_base : i32 = page_no * 98304 + i32(head) * 1536 + slot * 96 + 49152
                       + podArgs.pages_elem_offset;
    kv_pages[v_base + i32(dim_lo)     ] = f16(v_lo);
    kv_pages[v_base + i32(dim_lo) + 48] = f16(v_hi);
    return;
  }

  let pos_f : f32 = f32(position);
  let freq : f32 = pos_f / pow(10000.0, f32(dim_lo * 2u) / 96.0);
  let c    : f32 = cos(freq);
  let s    : f32 = sin(freq);
  let rot_lo : f32 = c * v_lo + s * (-v_hi);
  let rot_hi : f32 = c * v_hi + s * ( v_lo);

  if (group_id == 0u) {
    let b = i32(head) * 96 + i32(dim_lo);
    q_out[b     ] = f16(rot_lo);
    q_out[b + 48] = f16(rot_hi);
  } else {
    let k_base : i32 = page_no * 98304 + i32(head) * 1536 + slot * 96
                       + podArgs.pages_elem_offset;
    kv_pages[k_base + i32(dim_lo)     ] = f16(rot_lo);
    kv_pages[k_base + i32(dim_lo) + 48] = f16(rot_hi);
  }
}
`,hn=`// QKV_FUSED_SCRATCH — int8-KV-mode variant of qkv_fused.wgsl.
//
// Same math as qkv_fused: one workgroup per (group, head, dim-pair), computes
// two output rows, applies RoPE to Q/K in registers. The only change is
// WHERE K and V get written: into per-layer f16 scratch buffers (k_slot,
// v_slot) instead of the paged int8 cache. A subsequent kv_quantize_int8
// dispatch reads those scratches and writes the int8 + scale cache.
//
// Decode-only (ntoken=1). Prefill still uses the legacy 3-dispatch path.

enable f16;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;  // 3072 Q
@group(0) @binding(1) var<storage, read_write> k_slot       : array<f16>;  // 3072 K (scratch)
@group(0) @binding(2) var<storage, read_write> v_slot       : array<f16>;  // 3072 V (scratch)
@group(0) @binding(3) var<storage, read>       hidden       : array<f16>;  // 3072 hidden
@group(0) @binding(4) var<storage, read>       scales       : array<f16>;  // 9216 × 96 f16
@group(0) @binding(5) var<storage, read>       weights      : array<u32>;  // 9216 × 384 u32
@group(0) @binding(6) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  packGridDimX             : u32,
}
@group(0) @binding(7) var<uniform> podArgs : PODArgs;

const K_PACKED       : i32 = 384;
const SCALES_PER_ROW : i32 = 96;

var<workgroup> red0 : array<f32, 64>;
var<workgroup> red1 : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn qkv_fused_scratch(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let pair_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(pair_idx) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  let group         : i32 = pair_idx / 1536;
  let pair_in_group : i32 = pair_idx - group * 1536;
  let head          : i32 = pair_in_group / 48;
  let dim_lo        : i32 = pair_in_group - head * 48;

  let row_lo : i32 = group * 3072 + head * 96 + dim_lo;
  let row_hi : i32 = row_lo + 48;

  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;

  for (var chunk : i32 = 0; chunk < K_PACKED / 64; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 64;
    let packed_lo : u32 = weights[row_lo * K_PACKED + w_offset];
    let packed_hi : u32 = weights[row_hi * K_PACKED + w_offset];
    let scale_lo : f32 = f32(scales[row_lo * SCALES_PER_ROW + (w_offset >> 2)]);
    let scale_hi : f32 = f32(scales[row_hi * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    let x0 = f32(hidden[base    ]);
    let x1 = f32(hidden[base + 1]);
    let x2 = f32(hidden[base + 2]);
    let x3 = f32(hidden[base + 3]);
    let x4 = f32(hidden[base + 4]);
    let x5 = f32(hidden[base + 5]);
    let x6 = f32(hidden[base + 6]);
    let x7 = f32(hidden[base + 7]);

    acc0 = acc0 + x0 * (f32((packed_lo >>  0u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x1 * (f32((packed_lo >>  4u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x2 * (f32((packed_lo >>  8u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x3 * (f32((packed_lo >> 12u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x4 * (f32((packed_lo >> 16u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x5 * (f32((packed_lo >> 20u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x6 * (f32((packed_lo >> 24u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x7 * (f32((packed_lo >> 28u) & 15u) - 7.0) * scale_lo;

    acc1 = acc1 + x0 * (f32((packed_hi >>  0u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x1 * (f32((packed_hi >>  4u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x2 * (f32((packed_hi >>  8u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x3 * (f32((packed_hi >> 12u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x4 * (f32((packed_hi >> 16u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x5 * (f32((packed_hi >> 20u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x6 * (f32((packed_hi >> 24u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x7 * (f32((packed_hi >> 28u) & 15u) - 7.0) * scale_hi;
  }

  red0[tid] = acc0;
  red1[tid] = acc1;
  workgroupBarrier();
  if (tid < 32) { red0[tid] = red0[tid] + red0[tid + 32]; red1[tid] = red1[tid] + red1[tid + 32]; }
  workgroupBarrier();
  if (tid < 16) { red0[tid] = red0[tid] + red0[tid + 16]; red1[tid] = red1[tid] + red1[tid + 16]; }
  workgroupBarrier();
  if (tid < 8)  { red0[tid] = red0[tid] + red0[tid + 8];  red1[tid] = red1[tid] + red1[tid + 8]; }
  workgroupBarrier();
  if (tid < 4)  { red0[tid] = red0[tid] + red0[tid + 4];  red1[tid] = red1[tid] + red1[tid + 4]; }
  workgroupBarrier();
  if (tid < 2)  { red0[tid] = red0[tid] + red0[tid + 2];  red1[tid] = red1[tid] + red1[tid + 2]; }
  workgroupBarrier();
  if (tid != 0) { return; }

  let v_lo : f32 = red0[0] + red0[1];
  let v_hi : f32 = red1[0] + red1[1];

  // V: no RoPE, write to v_slot scratch.
  if (group == 2) {
    let base = head * 96 + dim_lo;
    v_slot[base     ] = f16(v_lo);
    v_slot[base + 48] = f16(v_hi);
    return;
  }

  // Q / K: apply RoPE.
  let pos  : f32 = f32(position_map[podArgs.position_map_elem_offset]);
  let freq : f32 = pos / pow(10000.0, f32(dim_lo * 2) / 96.0);
  let c    : f32 = cos(freq);
  let s    : f32 = sin(freq);

  let rot_lo : f32 = c * v_lo + s * (-v_hi);
  let rot_hi : f32 = c * v_hi + s * ( v_lo);

  let base = head * 96 + dim_lo;
  if (group == 0) {
    q_out[base     ] = f16(rot_lo);
    q_out[base + 48] = f16(rot_hi);
  } else {
    k_slot[base     ] = f16(rot_lo);
    k_slot[base + 48] = f16(rot_hi);
  }
}
`,wn=`// KV_QUANTIZE_INT8 — int8 quantize of one (K,V) slot per layer, per decode step.
//
// Input:  k_slot [3072 f16], v_slot [3072 f16]  (from qkv_fused_scratch)
// Output: kv_pages_i8 (packed int8 in u32), kv_scales (per-(head, side) f16)
//
// Granularity: one scale per (page, slot, head, side) — 96 f16 values share
// one scale. Per-row (head-slot-side) scale keeps accuracy close to f16.
//
// Layout of kv_pages_i8 (u32 words; 4 int8 per u32):
//   page[page_no] · head[h] · slot[s] · side(K|V) · dim_quad[q]
//     word_idx = page_no * (32*16*2*24)
//              + h * (16*2*24)
//              + s * (2*24)
//              + side * 24
//              + q
//     (side = 0 for K, 1 for V; q = dim/4)
//
// Layout of kv_scales (f16):
//   scale_idx = page_no * (32*16*2)  +  h * (16*2)  +  s * 2  +  side
//
// One workgroup = one (head, side). 32 threads cooperate to compute max over
// 96 dims, then each thread quantizes 3 consecutive dims (96 = 32 × 3).

enable f16;

@group(0) @binding(0) var<storage, read>       k_slot    : array<f16>;  // 3072
@group(0) @binding(1) var<storage, read>       v_slot    : array<f16>;  // 3072
@group(0) @binding(2) var<storage, read_write> pages_i8  : array<u32>;  // packed int8
@group(0) @binding(3) var<storage, read_write> scales    : array<f16>;
@group(0) @binding(4) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,  // u32-word offset
  scales_elem_offset       : i32,
  packGridDimX             : u32,  // 64 = 32 heads × 2 sides
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> max_reduce : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn kv_quantize_int8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let wg_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg_id) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  // Decompose wg_id → (head, side). side=0 for K, 1 for V.
  let head : i32 = wg_id / 2;
  let side : i32 = wg_id - head * 2;

  // Each thread reads 3 of the 96 dims for this (head, side).
  let slot_base : i32 = head * 96 + tid * 3;
  var v0 : f32;
  var v1 : f32;
  var v2 : f32;
  if (side == 0) {
    v0 = f32(k_slot[slot_base]);
    v1 = f32(k_slot[slot_base + 1]);
    v2 = f32(k_slot[slot_base + 2]);
  } else {
    v0 = f32(v_slot[slot_base]);
    v1 = f32(v_slot[slot_base + 1]);
    v2 = f32(v_slot[slot_base + 2]);
  }

  // Tree-reduce max(|x|) over 32 threads.
  var m : f32 = max(abs(v0), max(abs(v1), abs(v2)));
  max_reduce[tid] = m;
  workgroupBarrier();
  if (tid < 16) { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 16]); }
  workgroupBarrier();
  if (tid < 8)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 8]); }
  workgroupBarrier();
  if (tid < 4)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 4]); }
  workgroupBarrier();
  if (tid < 2)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 2]); }
  workgroupBarrier();
  if (tid < 1)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 1]); }
  workgroupBarrier();

  // scale = max / 127; guard against zero (empty row).
  var scale : f32 = max_reduce[0] / 127.0;
  if (scale < 1e-8) { scale = 1e-8; }
  let inv_scale : f32 = 1.0 / scale;

  // Compute destination page/slot.
  let position : i32 = position_map[podArgs.position_map_elem_offset];
  let page_no : i32 = position / 16;
  let slot : i32 = position - page_no * 16;

  // Thread 0 writes the scale.
  if (tid == 0) {
    let scale_idx : i32 = page_no * (32 * 16 * 2) + head * (16 * 2) + slot * 2 + side
                          + podArgs.scales_elem_offset;
    scales[scale_idx] = f16(scale);
  }

  // Quantize 3 dims and pack into int8. Because 3 is not a multiple of 4,
  // we use atomic-free write-via-packing: each thread writes its own
  // contribution to one u32 word using read-modify-write (safe because
  // different threads touch non-overlapping 8-bit lanes within each word).
  //
  // Simpler: redistribute so each thread owns ONE u32 word = 4 dims.
  // 24 words per row, 32 threads → threads 0..23 each own one word.
  // We already have v0..v2 (dims tid*3, tid*3+1, tid*3+2). That does NOT
  // align to 4-dim groups. So we need to re-read.
  //
  // Switch to per-word indexing: thread \`t\` in [0, 24) writes one word
  // containing dims [t*4, t*4+4). Re-read those dims.
  if (tid >= 24) { return; }

  let d0 : i32 = tid * 4;
  var b0 : f32;
  var b1 : f32;
  var b2 : f32;
  var b3 : f32;
  if (side == 0) {
    b0 = f32(k_slot[head * 96 + d0]);
    b1 = f32(k_slot[head * 96 + d0 + 1]);
    b2 = f32(k_slot[head * 96 + d0 + 2]);
    b3 = f32(k_slot[head * 96 + d0 + 3]);
  } else {
    b0 = f32(v_slot[head * 96 + d0]);
    b1 = f32(v_slot[head * 96 + d0 + 1]);
    b2 = f32(v_slot[head * 96 + d0 + 2]);
    b3 = f32(v_slot[head * 96 + d0 + 3]);
  }

  let q0 : i32 = clamp(i32(round(b0 * inv_scale)), -127, 127);
  let q1 : i32 = clamp(i32(round(b1 * inv_scale)), -127, 127);
  let q2 : i32 = clamp(i32(round(b2 * inv_scale)), -127, 127);
  let q3 : i32 = clamp(i32(round(b3 * inv_scale)), -127, 127);

  let packed : u32 =
      (u32(q0) & 0xffu)
    | ((u32(q1) & 0xffu) << 8u)
    | ((u32(q2) & 0xffu) << 16u)
    | ((u32(q3) & 0xffu) << 24u);

  // word_idx: page[page_no] · head[h] · slot[s] · side · dim_quad[tid]
  let word_idx : i32 = page_no * (32 * 16 * 2 * 24)
                     + head * (16 * 2 * 24)
                     + slot * (2 * 24)
                     + side * 24
                     + tid
                     + podArgs.pages_elem_offset;
  pages_i8[word_idx] = packed;
}
`,mn=`// PAGED KV ATTENTION — int8 KV cache variant of attention.wgsl.
//
// Identical online-softmax math; the only change is reading K,V as int8 + f16
// scale instead of f16. Per-row scale (one scale per head-slot-side) keeps
// accuracy close to f16.
//
// Layout of pages_i8 (u32 words; 4 int8 per u32):
//   word_idx = page_no * (32*16*2*24) + head * (16*2*24)
//            + slot * (2*24) + side * 24 + (dim/4)
//
// Layout of scales (f16):
//   scale_idx = page_no * (32*16*2) + head * (16*2) + slot * 2 + side

enable f16;

@group(0) @binding(0) var<storage, read> Q                 : array<f16>;
@group(0) @binding(1) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(2) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(3) var<storage, read> pages_i8          : array<u32>;
@group(0) @binding(4) var<storage, read> scales            : array<f16>;
@group(0) @binding(5) var<storage, read> length_info       : array<i32>;
@group(0) @binding(6) var<storage, read_write> output_buf  : array<f16>;

struct PODArgs {
  B: i32,
  max_num_pages: i32,
  nnz_pages: i32,
  pages_elem_offset: i32,
  page_indptr_elem_offset: i32,
  page_values_elem_offset: i32,
  length_info_elem_offset: i32,
  scales_elem_offset: i32,
  sm_scale: f32,
  packGridDimX: u32,
}
@group(0) @binding(7) var<uniform> podArgs : PODArgs;

var<workgroup> score_reduce : array<f32, 32>;

// Unpack signed int8 from a packed u32 word at byte index (0..3). Sign-extends.
fn unpack_i8(word : u32, byte_idx : u32) -> i32 {
  let raw : u32 = (word >> (byte_idx * 8u)) & 0xffu;
  // Sign-extend via arithmetic shift.
  return i32(raw << 24u) >> 24u;
}

@compute @workgroup_size(32, 1, 1)
fn attention_int8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let batch : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.B) { return; }

  // Each thread owns 3 elements of head_dim=96.
  var q0 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3]);
  var q1 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 1]);
  var q2 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 2]);

  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end   : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  var m  : f32 = -50000.0;
  var d  : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;

  // tid * 3 in [0, 96). The 3 dims span either 1 u32 word (aligned) or
  // straddle 2 words. Precompute byte layout:
  let byte_0 : i32 = tid * 3;                 // 0..95
  let word_0 : i32 = byte_0 / 4;              // 0..23
  let lane_0 : u32 = u32(byte_0 - word_0 * 4);
  let word_1 : i32 = (byte_0 + 1) / 4;
  let lane_1 : u32 = u32((byte_0 + 1) - word_1 * 4);
  let word_2 : i32 = (byte_0 + 2) / 4;
  let lane_2 : u32 = u32((byte_0 + 2) - word_2 * 4);

  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * 16;
    let slots_in_page : i32 = min(16, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      // Base word indices for K and V within this (page, head, slot).
      let kv_word_base : i32 = page_no * (32 * 16 * 2 * 24)
                             + head * (16 * 2 * 24)
                             + slot * (2 * 24)
                             + podArgs.pages_elem_offset;
      let k_word_base : i32 = kv_word_base;          // side=0
      let v_word_base : i32 = kv_word_base + 24;     // side=1

      // Per-(head, slot, side) scales.
      let scale_base : i32 = page_no * (32 * 16 * 2)
                           + head * (16 * 2)
                           + slot * 2
                           + podArgs.scales_elem_offset;
      let k_scale : f32 = f32(scales[scale_base]);
      let v_scale : f32 = f32(scales[scale_base + 1]);

      // K dequant: read 3 int8 values, multiply by k_scale.
      let k0 : f32 = f32(unpack_i8(pages_i8[k_word_base + word_0], lane_0)) * k_scale;
      let k1 : f32 = f32(unpack_i8(pages_i8[k_word_base + word_1], lane_1)) * k_scale;
      let k2 : f32 = f32(unpack_i8(pages_i8[k_word_base + word_2], lane_2)) * k_scale;

      let partial : f32 = q0 * k0 + q1 * k1 + q2 * k2;

      score_reduce[tid] = partial;
      workgroupBarrier();
      if (tid < 16) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 16]; }
      workgroupBarrier();
      if (tid < 8)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 8]; }
      workgroupBarrier();
      if (tid < 4)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 4]; }
      workgroupBarrier();
      if (tid < 2)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 2]; }
      workgroupBarrier();
      if (tid < 1)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 1]; }
      workgroupBarrier();

      let s : f32 = score_reduce[0] * podArgs.sm_scale;

      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new  : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      // V dequant + online softmax accumulate.
      let vv0 : f32 = f32(unpack_i8(pages_i8[v_word_base + word_0], lane_0)) * v_scale;
      let vv1 : f32 = f32(unpack_i8(pages_i8[v_word_base + word_1], lane_1)) * v_scale;
      let vv2 : f32 = f32(unpack_i8(pages_i8[v_word_base + word_2], lane_2)) * v_scale;

      o0 = o0 * scale_prev + scale_new * vv0;
      o1 = o1 * scale_prev + scale_new * vv1;
      o2 = o2 * scale_prev + scale_new * vv2;
    }
  }

  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    output_buf[batch * 3072 + head * 96 + tid * 3]     = f16(o0 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 1] = f16(o1 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 2] = f16(o2 * inv_d);
  }
}
`,kn=`// PAGED KV ATTENTION (subgroup variant) — same math as attention.wgsl,
// but replaces the 6-barrier tree reduction with a single subgroupAdd.
//
// Assumes subgroup size >= 32 (true on every shipping WebGPU backend that
// exposes the \`subgroups\` feature). The full 32-lane workgroup sits inside
// one subgroup, so subgroupAdd(partial) returns the full dot product with
// zero workgroup barriers per slot.
//
// Expected win: 6 workgroupBarrier() per KV slot → 0. Attention dominates
// tail latency on short-context decode; this is pure barrier elimination.

enable f16;
enable subgroups;

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

@compute @workgroup_size(32, 1, 1)
fn attention_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let batch : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.B) { return; }

  // Each thread owns 3 elements of head_dim=96.
  var q0 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3]);
  var q1 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 1]);
  var q2 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 2]);

  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  var m  : f32 = -50000.0;
  var d  : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;

  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * 16;
    let slots_in_page : i32 = min(16, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + podArgs.pages_elem_offset;

      let partial : f32 = q0 * f32(pages[k_base + tid * 3])
                        + q1 * f32(pages[k_base + tid * 3 + 1])
                        + q2 * f32(pages[k_base + tid * 3 + 2]);

      // Single subgroupAdd replaces 6 tree-reduction barriers.
      let s : f32 = subgroupAdd(partial) * podArgs.sm_scale;

      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new  : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      let v_base : i32 = k_base + 49152;
      o0 = o0 * scale_prev + scale_new * f32(pages[v_base + tid * 3]);
      o1 = o1 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 1]);
      o2 = o2 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 2]);
    }
  }

  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    output_buf[batch * 3072 + head * 96 + tid * 3]     = f16(o0 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 1] = f16(o1 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 2] = f16(o2 * inv_d);
  }
}
`,vn=`// FUSED_FFN_TILED_SG — tiled + subgroup variant of fused_ffn.wgsl.
//
// Same fusion (gate+up int4 matmul + SiLU + elementwise multiply in one
// dispatch) and same bind-group layout as fused_ffn.wgsl, so chat.ts can swap
// the pipeline reference without rebuilding bind groups.
//
// Differences vs the scalar kernel:
//   • Workgroup computes ROWS_PER_WG=4 output elements (gate+up pair ×4).
//     Input vector (3072 f16 = 6 KB) is loaded into shared once and reused
//     across all 8 weight rows — input-vector DRAM traffic drops 4x vs the
//     scalar shader's 1-row-per-WG layout.
//   • 32 threads per WG → one subgroup. Final reduction is 8× subgroupAdd
//     (one per accumulator), no workgroupBarriers in the hot loop.
//
// Weight layout (unchanged):
//   Rows 0..8191     = gate weights
//   Rows 8192..16383 = up weights
//   output[i] = SiLU(gate[i]) * up[i]
//
// Input/output share a buffer. shared_input snapshot-copies at WG start;
// output writes happen at kernel end, so by the time any WG writes an output
// every WG's input load has long since finished (same timing argument as
// fused_ffn.wgsl — the dot-product phase dwarfs the load phase).
//
// Requires sg_size = 32; gated in chat.ts.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read>       input_buf  : array<f16>;
@group(0) @binding(2) var<storage, read>       scales     : array<f16>;
@group(0) @binding(3) var<storage, read>       weights    : array<u32>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG    : u32 = 4u;
const D              : u32 = 3072u;
const D_PACKED       : u32 = 384u;   // 3072 / 8
const SCALES_PER_ROW : u32 = 96u;    // 3072 / 32
const GATE_UP_STRIDE : u32 = 8192u;

var<workgroup> shared_input : array<f16, 3072>;

@compute @workgroup_size(32, 1, 1)
fn fused_ffn_tiled_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let wg_idx : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let row_base : u32 = wg_idx * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let tid : u32 = threadIdx.x;

  // Snapshot-load input into shared mem. 32 threads × 96 elements = 3072.
  // Strided pattern (tid + i*32) coalesces adjacent-thread accesses.
  for (var i : u32 = 0u; i < 96u; i = i + 1u) {
    let idx : u32 = tid + i * 32u;
    shared_input[idx] = input_buf[idx];
  }
  workgroupBarrier();

  var gacc0 : f32 = 0.0; var gacc1 : f32 = 0.0; var gacc2 : f32 = 0.0; var gacc3 : f32 = 0.0;
  var uacc0 : f32 = 0.0; var uacc1 : f32 = 0.0; var uacc2 : f32 = 0.0; var uacc3 : f32 = 0.0;

  let gr0 = row_base;           let gr1 = row_base + 1u;
  let gr2 = row_base + 2u;      let gr3 = row_base + 3u;
  let ur0 = gr0 + GATE_UP_STRIDE; let ur1 = gr1 + GATE_UP_STRIDE;
  let ur2 = gr2 + GATE_UP_STRIDE; let ur3 = gr3 + GATE_UP_STRIDE;

  // K_PACKED / 32 = 12 chunks.
  for (var chunk : u32 = 0u; chunk < D_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    let i0 = f32(shared_input[base     ]);
    let i1 = f32(shared_input[base + 1u]);
    let i2 = f32(shared_input[base + 2u]);
    let i3 = f32(shared_input[base + 3u]);
    let i4 = f32(shared_input[base + 4u]);
    let i5 = f32(shared_input[base + 5u]);
    let i6 = f32(shared_input[base + 6u]);
    let i7 = f32(shared_input[base + 7u]);

    // Gate row 0
    {
      let p = weights[gr0 * D_PACKED + w_offset];
      let s = f32(scales[gr0 * SCALES_PER_ROW + sc_idx]);
      gacc0 = gacc0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    // Gate row 1
    {
      let p = weights[gr1 * D_PACKED + w_offset];
      let s = f32(scales[gr1 * SCALES_PER_ROW + sc_idx]);
      gacc1 = gacc1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    // Gate row 2
    {
      let p = weights[gr2 * D_PACKED + w_offset];
      let s = f32(scales[gr2 * SCALES_PER_ROW + sc_idx]);
      gacc2 = gacc2
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    // Gate row 3
    {
      let p = weights[gr3 * D_PACKED + w_offset];
      let s = f32(scales[gr3 * SCALES_PER_ROW + sc_idx]);
      gacc3 = gacc3
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    // Up row 0
    {
      let p = weights[ur0 * D_PACKED + w_offset];
      let s = f32(scales[ur0 * SCALES_PER_ROW + sc_idx]);
      uacc0 = uacc0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    // Up row 1
    {
      let p = weights[ur1 * D_PACKED + w_offset];
      let s = f32(scales[ur1 * SCALES_PER_ROW + sc_idx]);
      uacc1 = uacc1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    // Up row 2
    {
      let p = weights[ur2 * D_PACKED + w_offset];
      let s = f32(scales[ur2 * SCALES_PER_ROW + sc_idx]);
      uacc2 = uacc2
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
    // Up row 3
    {
      let p = weights[ur3 * D_PACKED + w_offset];
      let s = f32(scales[ur3 * SCALES_PER_ROW + sc_idx]);
      uacc3 = uacc3
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((p >> 28u) & 15u) - 7.0) * s;
    }
  }

  let g0 = subgroupAdd(gacc0); let g1 = subgroupAdd(gacc1);
  let g2 = subgroupAdd(gacc2); let g3 = subgroupAdd(gacc3);
  let u0 = subgroupAdd(uacc0); let u1 = subgroupAdd(uacc1);
  let u2 = subgroupAdd(uacc2); let u3 = subgroupAdd(uacc3);

  if (tid == 0u) {
    // SiLU(x) * up = x * sigmoid(x) * up — done in f32 for parity with
    // fused_ffn.wgsl (which upcasts gate to f32 before the sigmoid).
    let silu0 = g0 * (1.0 / (1.0 + exp(-g0)));
    let silu1 = g1 * (1.0 / (1.0 + exp(-g1)));
    let silu2 = g2 * (1.0 / (1.0 + exp(-g2)));
    let silu3 = g3 * (1.0 / (1.0 + exp(-g3)));
    output_buf[gr0] = f16(u0 * silu0);
    output_buf[gr1] = f16(u1 * silu1);
    output_buf[gr2] = f16(u2 * silu2);
    output_buf[gr3] = f16(u3 * silu3);
  }
}
`,An=`// ARGMAX (subgroup variant) — same result as argmax.wgsl.
// Two-level: (1) per-subgroup butterfly reduction using subgroupShuffleXor,
// (2) thread 0 scans the 8 subgroup winners serially. Replaces the 8 barriers
// of the scalar tree reduction with 1.
//
// Branchless compare-and-swap via select() — subgroup shuffles want uniform
// control flow for portability, so we avoid \`if\` inside the butterfly.
//
// 256 threads per workgroup. With subgroup size 32 that's 8 subgroups.

enable subgroups;

@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> result : array<i32>;

struct Params {
  vocab_size: u32,
}
@group(0) @binding(2) var<uniform> params : Params;

// One slot per subgroup. 256 / 32 = 8; overprovisioned to 16 for safety.
var<workgroup> sg_val : array<f32, 16>;
var<workgroup> sg_idx : array<i32, 16>;

@compute @workgroup_size(256, 1, 1)
fn argmax_sg(
  @builtin(local_invocation_id) tid : vec3<u32>,
  @builtin(subgroup_invocation_id) lane : u32,
  @builtin(subgroup_size) sg_size : u32,
) {
  let thread_id = tid.x;
  let vocab = params.vocab_size;
  let chunk = (vocab + 255u) / 256u;
  let start = thread_id * chunk;
  let end = min(start + chunk, vocab);

  var best_val : f32 = -1e30;
  var best_idx : i32 = 0;
  for (var i = start; i < end; i = i + 1u) {
    let v = logits[i];
    if (v > best_val) { best_val = v; best_idx = i32(i); }
  }

  // Butterfly reduction within the subgroup. select() keeps control flow
  // uniform across the shuffles.
  for (var stride : u32 = 1u; stride < sg_size; stride = stride << 1u) {
    let other_val = subgroupShuffleXor(best_val, stride);
    let other_idx = subgroupShuffleXor(best_idx, stride);
    let take = other_val > best_val;
    best_val = select(best_val, other_val, take);
    best_idx = select(best_idx, other_idx, take);
  }

  // Lane 0 of each subgroup writes its winner to shared memory.
  let sg_id : u32 = thread_id / sg_size;
  if (lane == 0u) {
    sg_val[sg_id] = best_val;
    sg_idx[sg_id] = best_idx;
  }
  workgroupBarrier();

  // Cross-subgroup: thread 0 scans the 8 subgroup winners serially. Simpler
  // than a second butterfly and avoids subgroup ops in divergent control.
  if (thread_id == 0u) {
    let n_subgroups : u32 = 256u / sg_size;
    var v : f32 = sg_val[0];
    var idx : i32 = sg_idx[0];
    for (var i : u32 = 1u; i < n_subgroups; i = i + 1u) {
      if (sg_val[i] > v) { v = sg_val[i]; idx = sg_idx[i]; }
    }
    result[0] = idx;
  }
}
`,A={D:3072,HEADS:32,HEAD_DIM:96,LAYERS:32,FFN:8192,VOCAB:32064,QKV_DIM:9216,PAGE_SIZE:16,MAX_PAGES:257,MAX_SEQ:4096};function s(n,a,e){const t=n.createShaderModule({code:a});return n.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:e}})}function f(n,a,e,t){return n.createBuffer({size:Math.max(a,4),usage:e|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:t})}const l=GPUBufferUsage.STORAGE;function Dn(n,a={}){console.log("[compiler] Creating pipelines...");const e=!!a.subgroups;e&&console.log("[compiler] subgroups feature enabled — compiling _sg variants");const t={embedding:s(n,V,"embedding"),rmsNorm:s(n,Q,"rms_norm"),qkvMatmul:s(n,R,"int4_matmul"),int4Matmul:s(n,R,"int4_matmul"),int4MatmulSg:e?s(n,rn,"int4_matmul_sg"):null,int4MatmulTiled:e?s(n,on,"int4_matmul_tiled"):null,int4MatmulTiled8:e?s(n,_n,"int4_matmul_tiled8"):null,int4MatmulF32Sg:e?s(n,dn,"int4_matmul_f32_sg"):null,int4MatmulF32Tiled:e?s(n,fn,"int4_matmul_f32_tiled"):null,int4MatmulF32Tiled8:e?s(n,ln,"int4_matmul_f32_tiled8"):null,int4MatmulBatchedM4:e?s(n,pn,"int4_matmul_batched_m4"):null,rope:s(n,j,"rope_kernel"),kvAppend:s(n,X,"kv_append"),qkvFused:s(n,Y,"qkv_fused"),qkvFusedSg:e?s(n,cn,"qkv_fused_sg"):null,qkvFusedTiledSg:e?s(n,gn,"qkv_fused_tiled_sg"):null,qkvFusedTiled2Sg:e?s(n,bn,"qkv_fused_tiled2sg"):null,qkvFusedScratch:s(n,hn,"qkv_fused_scratch"),kvQuantizeInt8:s(n,wn,"kv_quantize_int8"),attentionInt8:s(n,mn,"attention_int8"),attention:s(n,$,"attention"),attentionSg:e?s(n,kn,"attention_sg"):null,oProjMatmul:s(n,R,"int4_matmul"),addNorm:s(n,F,"add_norm"),fusedFfn:s(n,U,"fused_ffn_kernel"),fusedFfnTiledSg:e?s(n,vn,"fused_ffn_tiled_sg"):null,ffnDownMatmul:s(n,R,"int4_matmul"),lmHead:s(n,H,"int4_matmul_f32"),argmax:s(n,N,"argmax_kernel"),argmaxSg:e?s(n,An,"argmax_sg"):null};console.log("[compiler] Allocating buffers...");const o=2,d=A.D,h={hidden1:f(n,d*o,l,"hidden1"),hidden2:f(n,d*o,l,"hidden2"),residual:f(n,d*o,l,"residual"),qkvOut:f(n,A.QKV_DIM*o,l,"qkvOut"),qOut:f(n,d*o,l,"qOut"),kOut:f(n,d*o,l,"kOut"),vOut:f(n,d*o,l,"vOut"),attnOut:f(n,d*o,l,"attnOut"),ffnOut:f(n,A.FFN*o,l,"ffnOut"),logits:f(n,A.VOCAB*4,l,"logits"),tokenResult:f(n,4,l,"tokenResult"),inputIds:f(n,A.MAX_SEQ*4,l,"inputIds"),positionMap:f(n,A.MAX_SEQ*4,l,"positionMap"),pageTable:f(n,A.MAX_PAGES*4,l,"pageTable"),pageIndptr:f(n,8,l,"pageIndptr"),pageValues:f(n,A.MAX_PAGES*4,l,"pageValues"),lengthInfo:f(n,12,l,"lengthInfo"),embdWeights:f(n,4,l,"embdWeights_placeholder"),embdScales:f(n,4,l,"embdScales_placeholder"),lmHeadWeights:f(n,4,l,"lmHeadWeights_placeholder"),lmHeadScales:f(n,4,l,"lmHeadScales_placeholder"),initNormGamma:f(n,d*o,l,"initNormGamma")};return console.log(`[compiler] Done: ${Object.keys(t).length} pipelines, ${Object.keys(h).length} buffers`),{pipelines:t,buffers:h}}export{A as P,Kn as a,Rn as b,Dn as c,Pn as l};
