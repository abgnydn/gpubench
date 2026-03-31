// ═══════════════════════════════════════════
// 1. RASTRIGIN — Standard optimization benchmark (PARALLEL)
// ═══════════════════════════════════════════
export const RASTRIGIN_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>; // [populationSize, dimensions]

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let pop = params.x;
  let dim = params.y;
  if (idx >= pop) { return; }

  var fitness: f32 = 10.0 * f32(dim);
  let base = idx * dim;
  for (var i: u32 = 0u; i < dim; i = i + 1u) {
    let x = input[base + i];
    fitness = fitness + (x * x - 10.0 * cos(2.0 * PI * x));
  }
  output[idx] = fitness;
}
`;

// ═══════════════════════════════════════════
// 2. N-BODY — Classic GPU sequential simulation (SEQUENTIAL)
// Each thread simulates one body through all timesteps
// ═══════════════════════════════════════════
export const NBODY_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> initial_pos: array<f32>;   // N * 4 (x, y, vx, vy per body)
@group(0) @binding(1) var<storage, read_write> output: array<f32>;  // N energy values
@group(0) @binding(2) var<uniform> params: vec3<u32>;               // [numBodies, timesteps, _pad]

const G: f32 = 6.674e-3;  // scaled gravitational constant
const DT: f32 = 0.01;
const SOFTENING: f32 = 0.01;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let N = params.x;
  let steps = params.y;
  if (idx >= N) { return; }

  let base = idx * 4u;
  var px: f32 = initial_pos[base + 0u];
  var py: f32 = initial_pos[base + 1u];
  var vx: f32 = initial_pos[base + 2u];
  var vy: f32 = initial_pos[base + 3u];

  // Fused: all timesteps in one dispatch
  for (var t: u32 = 0u; t < steps; t = t + 1u) {
    var ax: f32 = 0.0;
    var ay: f32 = 0.0;

    // Compute gravitational force from all other bodies
    for (var j: u32 = 0u; j < N; j = j + 1u) {
      if (j == idx) { continue; }
      let jbase = j * 4u;
      let dx = initial_pos[jbase + 0u] - px;
      let dy = initial_pos[jbase + 1u] - py;
      let dist2 = dx * dx + dy * dy + SOFTENING;
      let inv_dist = 1.0 / sqrt(dist2);
      let inv_dist3 = inv_dist * inv_dist * inv_dist;
      ax = ax + G * dx * inv_dist3;
      ay = ay + G * dy * inv_dist3;
    }

    vx = vx + ax * DT;
    vy = vy + ay * DT;
    px = px + vx * DT;
    py = py + vy * DT;
  }

  // Output: kinetic energy
  output[idx] = 0.5 * (vx * vx + vy * vy);
}
`;

// ═══════════════════════════════════════════
// 3. ACROBOT — Standard Gym RL environment (SEQUENTIAL)
// Double pendulum swing-up with RK4 physics
// ═══════════════════════════════════════════
export const ACROBOT_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> genomes: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec3<u32>; // [pop, genomeSize, timesteps]

const PI: f32 = 3.14159265358979323846;
const DT: f32 = 0.05;
const LINK1: f32 = 1.0;
const LINK2: f32 = 1.0;
const M1: f32 = 1.0;
const M2: f32 = 1.0;
const LC1: f32 = 0.5;
const LC2: f32 = 0.5;
const I1: f32 = 1.0;
const I2: f32 = 1.0;
const GRAVITY: f32 = 9.8;

fn nn_forward(genome_base: u32, s0: f32, s1: f32, s2: f32, s3: f32, s4: f32, s5: f32) -> f32 {
  // 6 inputs -> 16 hidden (tanh) -> 3 outputs (torque selection)
  var hidden: array<f32, 16>;
  let inputs = array<f32, 6>(s0, s1, s2, s3, s4, s5);
  for (var h: u32 = 0u; h < 16u; h = h + 1u) {
    var sum: f32 = genomes[genome_base + 96u + h]; // bias
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
      sum = sum + inputs[i] * genomes[genome_base + h * 6u + i];
    }
    hidden[h] = tanh(sum);
  }
  var best_val: f32 = -1e10;
  var best_act: f32 = -1.0;
  for (var o: u32 = 0u; o < 3u; o = o + 1u) {
    var sum: f32 = genomes[genome_base + 112u + 48u + o]; // output bias
    for (var h: u32 = 0u; h < 16u; h = h + 1u) {
      sum = sum + hidden[h] * genomes[genome_base + 112u + o * 16u + h];
    }
    if (sum > best_val) { best_val = sum; best_act = f32(o) - 1.0; }
  }
  return best_act;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let pop = params.x;
  let gSize = params.y;
  let steps = params.z;
  if (idx >= pop) { return; }

  let gBase = idx * gSize;
  var theta1: f32 = 0.0;
  var theta2: f32 = 0.0;
  var dtheta1: f32 = 0.0;
  var dtheta2: f32 = 0.0;
  var reward: f32 = 0.0;

  for (var t: u32 = 0u; t < steps; t = t + 1u) {
    let s0 = cos(theta1);
    let s1 = sin(theta1);
    let s2 = cos(theta2);
    let s3 = sin(theta2);
    let s4 = dtheta1;
    let s5 = dtheta2;

    let torque = nn_forward(gBase, s0, s1, s2, s3, s4, s5);

    // RK4 integration (simplified Acrobot dynamics)
    let d2 = M2 * (LC2 * LC2 + LINK1 * LC2 * cos(theta2) + I2);
    let phi2 = M2 * LC2 * GRAVITY * cos(theta1 + theta2 - PI / 2.0);
    let phi1 = -M2 * LINK1 * LC2 * dtheta2 * dtheta2 * sin(theta2)
               + (M1 * LC1 + M2 * LINK1) * GRAVITY * cos(theta1 - PI / 2.0) + phi2;
    let ddtheta2 = (torque + d2 / d2 * phi1 - phi2) / (M2 * LC2 * LC2 + I2);
    let ddtheta1 = -(d2 * ddtheta2 + phi1) / (M1 * LC1 * LC1 + M2 * (LINK1 * LINK1 + LC2 * LC2) + I1 + I2);

    dtheta1 = dtheta1 + ddtheta1 * DT;
    dtheta2 = dtheta2 + ddtheta2 * DT;
    theta1 = theta1 + dtheta1 * DT;
    theta2 = theta2 + dtheta2 * DT;

    dtheta1 = clamp(dtheta1, -4.0 * PI, 4.0 * PI);
    dtheta2 = clamp(dtheta2, -9.0 * PI, 9.0 * PI);

    let tip_y = -LINK1 * cos(theta1) - LINK2 * cos(theta1 + theta2);
    if (tip_y > LINK1) { reward = reward + 1.0; }
    reward = reward - 0.01;
  }

  output[idx] = reward;
}
`;

// ═══════════════════════════════════════════
// 4. MOUNTAINCAR — Standard Gym RL environment (SEQUENTIAL)
// ═══════════════════════════════════════════
export const MOUNTAINCAR_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> genomes: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec3<u32>; // [pop, genomeSize, timesteps]

const GOAL_POS: f32 = 0.5;
const MIN_POS: f32 = -1.2;
const MAX_SPEED: f32 = 0.07;
const GRAVITY: f32 = 0.0025;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let pop = params.x;
  let gSize = params.y;
  let steps = params.z;
  if (idx >= pop) { return; }

  let gBase = idx * gSize;
  var position: f32 = -0.5;
  var velocity: f32 = 0.0;
  var reward: f32 = 0.0;
  var solved: bool = false;

  for (var t: u32 = 0u; t < steps; t = t + 1u) {
    if (solved) { continue; }

    // Simple linear policy: 2 inputs (pos, vel) -> 3 weights -> action
    var best_val: f32 = -1e10;
    var action: f32 = 0.0;
    for (var a: u32 = 0u; a < 3u; a = a + 1u) {
      let val = genomes[gBase + a * 3u] * position
              + genomes[gBase + a * 3u + 1u] * velocity
              + genomes[gBase + a * 3u + 2u];
      if (val > best_val) { best_val = val; action = f32(a) - 1.0; }
    }

    // MountainCar physics
    velocity = velocity + action * 0.001 + cos(3.0 * position) * (-GRAVITY);
    velocity = clamp(velocity, -MAX_SPEED, MAX_SPEED);
    position = position + velocity;
    position = clamp(position, MIN_POS, 0.6);

    if (position <= MIN_POS && velocity < 0.0) { velocity = 0.0; }
    if (position >= GOAL_POS) { solved = true; reward = f32(steps - t); }

    reward = reward - 1.0;
  }

  output[idx] = reward;
}
`;

// ═══════════════════════════════════════════
// 5. MONTE CARLO PI — Classic parallel estimation (PARALLEL)
// ═══════════════════════════════════════════
export const MONTECARLO_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> seeds: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>; // [numWorkers, samplesPerWorker]

fn pcg_hash(input: u32) -> u32 {
  var state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
  *seed = pcg_hash(*seed);
  return f32(*seed) / 4294967295.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let N = params.x;
  let samples = params.y;
  if (idx >= N) { return; }

  var seed: u32 = seeds[idx];
  var inside: u32 = 0u;

  for (var i: u32 = 0u; i < samples; i = i + 1u) {
    let x = rand_f32(&seed);
    let y = rand_f32(&seed);
    if (x * x + y * y <= 1.0) { inside = inside + 1u; }
  }

  output[idx] = 4.0 * f32(inside) / f32(samples);
}
`;
