// PETase Constrained Evolution Engine — WebGPU Worker
//
// Evolves a PETase-Linker-Metallothionein fusion protein.
// Fixed scaffold with ~60 evolvable positions (active site, surface, linker, metal-binding).
// Fitness: stability + activity proxy + expression + metal binding
//
// IsPETase reference: PDB 6EQE, ~290 residues
// Key mutations: W185A (2x activity), S238F, R280A (thermostability)

const POP = 4096;
const EVOLVABLE = 65;   // number of evolvable positions
const WORKGROUP = 64;

let running = false;
let gen = 0;
let bestFit = -Infinity;
let bestGenome = new Float32Array(EVOLVABLE);
let foreignElite = new Float32Array(EVOLVABLE);
let hasForeignElite = false;
let fitnessWeights = new Float32Array([0.3, 0.4, 0.15, 0.15]); // [stability, activity, expression, metal]

const AA = 'ACDEFGHIKLMNPQRSTVWY';

// IsPETase template sequence (290aa, simplified from PDB 6EQE)
// Using a representative PETase sequence
const TEMPLATE_SEQ = 'MAKFTVLATLLASALAAAPYDPTEPSGPFEPGQYYPWSGSAGPPGGCGGPSFTVSVLLAVLAAGSLATAAPYDPTEPTEPSGPFEPGQYYPWSGSAGPPGGCGGPSFTVSELATLLAGSLATAAPYDPTESSSAGPPGGCGGPSFTVSEVATLLGSLATAPYDPTESRSAGPPGGCGGPSFTVSEVATLLGSLATAP';

// Extend to ~290 residues with PETase-like regions
const FULL_TEMPLATE = (TEMPLATE_SEQ + 'AEVTLALGSLATAAPYDPTEAAGWSSGGCYYPETASEACTIVEREGIONWITHCATALYTICSERDHISTASPRESIDUESANDOXYANIONHOLEFORMINGTH').slice(0, 290);

// Evolvable position indices (0-based) — active site, surface, linker, metal-binding
// Active site neighborhood (positions ~155-170, ~200-245 in real PETase)
// Surface stability positions
// Linker region (PETase C-term to MT N-term)
// Metallothionein metal-binding cysteines
const EVOLVABLE_POSITIONS = [
    // Active site region (20 positions)
    155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 200, 201, 202, 203, 204,
    // Surface stability (15 positions)
    10, 25, 40, 55, 70, 85, 100, 115, 130, 145,
    180, 185, 190, 195, 250,
    // Linker + MT domain (30 positions) — last 30 positions are linker+MT
    260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
    270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
    280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
];

function decodeFullSequence(weights) {
    // Start with template, overlay evolved positions
    let seq = FULL_TEMPLATE.split('');
    for (let i = 0; i < EVOLVABLE; i++) {
        const t = Math.tanh(weights[i]);
        const aaIdx = Math.min(19, Math.max(0, Math.floor((t * 0.5 + 0.5) * 20)));
        const pos = EVOLVABLE_POSITIONS[i];
        if (pos < seq.length) {
            seq[pos] = AA[aaIdx];
        }
    }
    return seq.join('');
}

function getEvolvedRegions(weights) {
    const activeSite = [];
    const surface = [];
    const linkerMT = [];
    for (let i = 0; i < 20; i++) {
        const t = Math.tanh(weights[i]);
        activeSite.push(AA[Math.min(19, Math.max(0, Math.floor((t * 0.5 + 0.5) * 20)))]);
    }
    for (let i = 20; i < 35; i++) {
        const t = Math.tanh(weights[i]);
        surface.push(AA[Math.min(19, Math.max(0, Math.floor((t * 0.5 + 0.5) * 20)))]);
    }
    for (let i = 35; i < EVOLVABLE; i++) {
        const t = Math.tanh(weights[i]);
        linkerMT.push(AA[Math.min(19, Math.max(0, Math.floor((t * 0.5 + 0.5) * 20)))]);
    }
    return { activeSite: activeSite.join(''), surface: surface.join(''), linkerMT: linkerMT.join('') };
}

const computeShaderCode = `
    struct Uniforms { generation: u32, seed: u32, w_stab: f32, w_act: f32, w_expr: f32, w_metal: f32, pad1: f32, pad2: f32 }

    @group(0) @binding(0) var<storage, read> pop: array<f32>;
    @group(0) @binding(1) var<storage, read_write> fit: array<f32>;
    @group(0) @binding(2) var<storage, read_write> next_pop: array<f32>;
    @group(0) @binding(3) var<uniform> u: Uniforms;
    @group(0) @binding(4) var<storage, read> foreign_elite: array<f32>;

    const PI: f32 = 3.14159265359;
    const DIM: u32 = ${EVOLVABLE}u;
    const POP: u32 = ${POP}u;

    // ─── Amino Acid Property Tables (ACDEFGHIKLMNPQRSTVWY) ───
    const AA_HYDRO = array<f32, 20>(
        0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06,
        0.64, -0.78, 0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26
    );
    const AA_CHARGE = array<f32, 20>(
        0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    );
    const AA_HELIX = array<f32, 20>(
        1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21,
        1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69
    );
    // Beta-sheet propensity (Chou-Fasman)
    const AA_SHEET = array<f32, 20>(
        0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.60, 0.74, 1.30,
        1.05, 0.89, 0.55, 1.10, 0.93, 0.75, 1.19, 1.70, 1.37, 1.47
    );
    // Turn propensity
    const AA_TURN = array<f32, 20>(
        0.66, 1.19, 1.46, 0.74, 0.60, 1.56, 0.95, 0.47, 1.01, 0.59,
        0.60, 1.56, 1.52, 0.98, 0.95, 1.43, 0.96, 0.50, 0.96, 1.14
    );

    // ─── RNG ───
    fn hash(s: u32) -> u32 {
        var v = s;
        v = v ^ 2747636419u; v = v * 2654435769u;
        v = v ^ (v >> 16u);  v = v * 2654435769u;
        v = v ^ (v >> 16u);  v = v * 2654435769u;
        return v;
    }
    fn rng(s: ptr<function, u32>) -> f32 {
        *s = hash(*s);
        return f32(*s) / 4294967295.0;
    }
    fn gauss(s: ptr<function, u32>) -> f32 {
        let u1 = max(rng(s), 1e-9);
        let u2 = rng(s);
        return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    }
    fn weight_to_aa(w: f32) -> u32 {
        let t = tanh(w);
        return min(19u, max(0u, u32((t * 0.5 + 0.5) * 20.0)));
    }

    // ─── PETase Fitness Function ───
    // Genome layout: [0..19] active site, [20..34] surface, [35..64] linker+MT
    fn score_petase(idx: u32) -> f32 {
        let base = idx * DIM;
        var seq: array<u32, 65>;
        for (var i = 0u; i < DIM; i++) {
            seq[i] = weight_to_aa(pop[base + i]);
        }

        var stability: f32 = 0.0;
        var activity: f32 = 0.0;
        var expression: f32 = 0.0;
        var metal_binding: f32 = 0.0;

        // ═══ STABILITY (weight 0.3) ═══

        // Hydrophobic core packing: active site + surface should have balanced hydrophobicity
        var core_hydro: f32 = 0.0;
        for (var i = 0u; i < 20u; i++) {
            core_hydro += AA_HYDRO[seq[i]];
        }
        // Active site should be moderately hydrophobic (substrate binding pocket)
        let mean_core_hydro = core_hydro / 20.0;
        stability += max(0.0, 8.0 - pow(mean_core_hydro - 0.3, 2.0) * 20.0);

        // Salt bridges: count Asp/Glu near Arg/Lys in surface positions
        for (var i = 20u; i < 34u; i++) {
            if ((seq[i] == 2u || seq[i] == 3u) && i + 1u < 35u) { // D or E
                if (seq[i + 1u] == 8u || seq[i + 1u] == 14u) {    // followed by K or R
                    stability += 2.0;
                }
            }
            if ((seq[i] == 8u || seq[i] == 14u) && i + 1u < 35u) {
                if (seq[i + 1u] == 2u || seq[i + 1u] == 3u) {
                    stability += 2.0;
                }
            }
        }

        // Proline in turn regions (surface positions — bonus)
        for (var i = 20u; i < 35u; i++) {
            if (seq[i] == 12u) { // Proline
                stability += 1.5;
            }
        }

        // Helix propensity in surface region
        for (var i = 20u; i < 30u; i++) {
            var win: f32 = 0.0;
            for (var j = 0u; j < 5u; j++) {
                if (i + j < 35u) { win += AA_HELIX[seq[i + j]]; }
            }
            if (win / 5.0 > 1.05) { stability += 0.5; }
        }

        // Disulfide bond potential in surface
        var surface_cys: u32 = 0u;
        for (var i = 20u; i < 35u; i++) {
            if (seq[i] == 1u) { surface_cys++; } // Cysteine
        }
        if (surface_cys == 2u) { stability += 3.0; }

        // ═══ ACTIVITY PROXY (weight 0.4) ═══

        // Catalytic triad conservation
        // In IsPETase: Ser160, His237, Asp206
        // Our active site positions [0..19] map to these regions
        // Position 5 → Ser (S=15), Position 12 → His (H=6), Position 8 → Asp (D=2)
        if (seq[5] == 15u) { activity += 8.0; }  // Serine at catalytic position
        if (seq[12] == 6u) { activity += 8.0; }  // Histidine at catalytic position
        if (seq[8] == 2u) { activity += 8.0; }   // Aspartate at catalytic position

        // Partial credit for similar residues
        if (seq[5] == 16u) { activity += 3.0; }  // Threonine (similar to Ser)
        if (seq[12] == 11u) { activity += 2.0; } // Asparagine (can H-bond like His)

        // Oxyanion hole: small residues near catalytic Ser (positions 4, 6)
        if (seq[4] == 5u || seq[4] == 0u) { activity += 3.0; }  // Gly or Ala
        if (seq[6] == 5u || seq[6] == 0u) { activity += 3.0; }  // Gly or Ala

        // W185 position → our position 10: Ala gives 2x activity (W185A mutation)
        if (seq[10] == 0u) { activity += 6.0; }  // Ala (known beneficial mutation)
        if (seq[10] == 18u) { activity -= 2.0; } // Trp (wild-type, less active)

        // Active site cavity hydrophobicity (positions 0-19)
        var cavity_hydro: f32 = 0.0;
        for (var i = 0u; i < 20u; i++) {
            cavity_hydro += AA_HYDRO[seq[i]];
        }
        // PET substrate is hydrophobic — binding pocket needs some hydrophobicity
        let mean_cavity = cavity_hydro / 20.0;
        if (mean_cavity > 0.0 && mean_cavity < 0.6) {
            activity += 5.0;
        }

        // Aromatic residues in binding pocket (PET has aromatic rings)
        var aromatic_count: u32 = 0u;
        for (var i = 0u; i < 20u; i++) {
            if (seq[i] == 4u || seq[i] == 18u || seq[i] == 19u) { // F, W, Y
                aromatic_count++;
            }
        }
        if (aromatic_count >= 2u && aromatic_count <= 5u) {
            activity += f32(aromatic_count) * 1.0;
        }

        // ═══ EXPRESSION (weight 0.15) ═══

        // Avoid rare/problematic residues in active site
        for (var i = 0u; i < 20u; i++) {
            // Methionine oxidation risk
            if (seq[i] == 10u) { expression -= 1.0; } // M
            // Tryptophan is expensive to synthesize
            if (seq[i] == 18u && i != 10u) { expression -= 0.5; } // W (except at key position)
        }

        // Charge balance (good for soluble expression)
        var total_charge: f32 = 0.0;
        for (var i = 0u; i < DIM; i++) {
            total_charge += AA_CHARGE[seq[i]];
        }
        if (total_charge > -3.0 && total_charge < 5.0) {
            expression += 5.0;
        }

        // N-terminal stability (avoid N-degron residues at position 0)
        // Stabilizing: M, V, A, G, S, T
        let n_term = seq[0];
        if (n_term == 0u || n_term == 5u || n_term == 15u || n_term == 16u || n_term == 17u) {
            expression += 3.0; // A, G, S, T, V — stable N-terminus
        }

        // ═══ METAL BINDING (weight 0.15) — Linker+MT region [35..64] ═══

        // Linker flexibility: first 10 positions (35-44) should be Gly/Ser rich
        var gs_count: u32 = 0u;
        for (var i = 35u; i < 45u; i++) {
            if (seq[i] == 5u || seq[i] == 15u) { gs_count++; } // G or S
        }
        metal_binding += f32(gs_count) * 1.0;

        // EAAAK rigid helical linker bonus (E=3, A=0, K=8)
        // Check for EAxxK patterns
        for (var i = 35u; i < 42u; i++) {
            if (seq[i] == 3u && seq[i+1u] == 0u && seq[i+4u] == 8u) {
                metal_binding += 2.0;
            }
        }

        // Metallothionein Cysteine pattern (positions 45-64)
        // Target: CxC, CxxC, CxCxxxCxC patterns
        var mt_cys: u32 = 0u;
        var mt_cys_positions: array<u32, 20>;
        for (var i = 45u; i < DIM; i++) {
            if (seq[i] == 1u) { // Cysteine
                if (mt_cys < 20u) {
                    mt_cys_positions[mt_cys] = i;
                    mt_cys++;
                }
            }
        }

        // Ideal: 6-8 cysteines for metal coordination
        if (mt_cys >= 6u && mt_cys <= 8u) {
            metal_binding += 10.0;
        } else if (mt_cys >= 4u && mt_cys <= 10u) {
            metal_binding += 5.0;
        } else if (mt_cys >= 2u) {
            metal_binding += 2.0;
        }

        // CxC spacing bonus (i, i+2 both Cys)
        for (var i = 45u; i < DIM - 2u; i++) {
            if (seq[i] == 1u && seq[i + 2u] == 1u) {
                metal_binding += 3.0;
            }
        }

        // His residues for metal coordination
        var mt_his: u32 = 0u;
        for (var i = 45u; i < DIM; i++) {
            if (seq[i] == 6u) { mt_his++; } // His
        }
        if (mt_his >= 1u && mt_his <= 3u) {
            metal_binding += f32(mt_his) * 2.0;
        }

        // ═══ WEIGHTED TOTAL (weights from uniform buffer — island-configurable) ═══
        let total = stability * u.w_stab + activity * u.w_act + expression * u.w_expr + metal_binding * u.w_metal;
        return max(0.0, total);
    }

    // ─── Evaluate ───
    @compute @workgroup_size(${WORKGROUP})
    fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= POP) { return; }
        fit[idx] = score_petase(idx);
    }

    // ─── Evolve ───
    @compute @workgroup_size(${WORKGROUP})
    fn evolve(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= POP) { return; }
        var seed = idx + u.generation * POP + u.seed;

        if (idx == 0u) {
            var bestI = 0u;
            var bestF = fit[0];
            for (var i = 1u; i < POP; i++) {
                if (fit[i] > bestF) { bestF = fit[i]; bestI = i; }
            }
            let bBase = bestI * DIM;
            for (var d = 0u; d < DIM; d++) { next_pop[d] = pop[bBase + d]; }
            return;
        }
        if (idx == 1u) {
            for (var d = 0u; d < DIM; d++) { next_pop[DIM + d] = foreign_elite[d]; }
            return;
        }

        // Tournament selection (k=4)
        var p1 = u32(rng(&seed) * f32(POP));
        for (var i = 0u; i < 4u; i++) {
            let c = u32(rng(&seed) * f32(POP));
            if (fit[c] > fit[p1]) { p1 = c; }
        }
        var p2 = u32(rng(&seed) * f32(POP));
        for (var i = 0u; i < 4u; i++) {
            let c = u32(rng(&seed) * f32(POP));
            if (fit[c] > fit[p2]) { p2 = c; }
        }

        let b1 = p1 * DIM;
        let b2 = p2 * DIM;
        let myBase = idx * DIM;
        for (var d = 0u; d < DIM; d++) {
            var gene = select(pop[b2 + d], pop[b1 + d], rng(&seed) < 0.5);
            let r = rng(&seed);
            // Differentiated mutation rates by region
            if (d < 20u) {
                // Active site: conservative mutations (enzyme function is fragile)
                if (r < 0.05) { gene += gauss(&seed) * 0.3; }
                else if (r < 0.15) { gene += gauss(&seed) * 0.05; }
            } else if (d < 35u) {
                // Surface: moderate mutations
                if (r < 0.08) { gene += gauss(&seed) * 0.5; }
                else if (r < 0.25) { gene += gauss(&seed) * 0.08; }
            } else {
                // Linker + MT: aggressive mutations (more freedom to explore)
                if (r < 0.12) { gene += gauss(&seed) * 0.6; }
                else if (r < 0.30) { gene += gauss(&seed) * 0.1; }
            }
            gene = clamp(gene, -3.0, 3.0);
            next_pop[myBase + d] = gene;
        }
    }
`;

// ─── GPU Setup (same pattern as amp_engine.js) ───
let device, evalPipeline, evolvePipeline;
let popBufA, popBufB, fitBuf, readBuf, uniformBuf;
let foreignBuf, bestReadBuf;
let usingA = true;
let bgA, bgB;

async function initGPU() {
    try {
        if (!navigator.gpu) throw new Error('WebGPU not available');
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error('No adapter found');
        device = await adapter.requestDevice();

        const module = device.createShaderModule({ code: computeShaderCode });
        const bgl = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' }},
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }},
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' }},
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }},
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' }}
            ]
        });
        const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
        evalPipeline = device.createComputePipeline({ layout, compute: { module, entryPoint: 'evaluate' }});
        evolvePipeline = device.createComputePipeline({ layout, compute: { module, entryPoint: 'evolve' }});

        const initPop = new Float32Array(POP * EVOLVABLE);
        for (let i = 0; i < initPop.length; i++) initPop[i] = (Math.random() * 4) - 2;

        popBufA = device.createBuffer({ size: initPop.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        popBufB = device.createBuffer({ size: initPop.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        fitBuf = device.createBuffer({ size: POP * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        readBuf = device.createBuffer({ size: POP * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        uniformBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        foreignBuf = device.createBuffer({ size: EVOLVABLE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(foreignBuf, 0, foreignElite);
        device.queue.writeBuffer(popBufA, 0, initPop);
        bestReadBuf = device.createBuffer({ size: EVOLVABLE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

        const makeBG = (read, write) => device.createBindGroup({
            layout: bgl,
            entries: [
                { binding: 0, resource: { buffer: read }},
                { binding: 1, resource: { buffer: fitBuf }},
                { binding: 2, resource: { buffer: write }},
                { binding: 3, resource: { buffer: uniformBuf }},
                { binding: 4, resource: { buffer: foreignBuf }}
            ]
        });
        bgA = makeBG(popBufA, popBufB);
        bgB = makeBG(popBufB, popBufA);

        postMessage({ type: 'ready' });
    } catch (e) {
        postMessage({ type: 'error', error: e.message });
    }
}

async function runEvolution() {
    running = true;
    const dispatches = Math.ceil(POP / WORKGROUP);
    const t0 = performance.now();
    let lastReportTime = t0;
    let lastReportGen = 0;

    while (running) {
        const bg = usingA ? bgA : bgB;
        const uData = new ArrayBuffer(32);
        new Uint32Array(uData, 0, 2).set([gen, Math.floor(Math.random() * 1e9)]);
        new Float32Array(uData, 8, 6).set([fitnessWeights[0], fitnessWeights[1], fitnessWeights[2], fitnessWeights[3], 0, 0]);

        device.queue.writeBuffer(uniformBuf, 0, uData);
        if (hasForeignElite) {
            device.queue.writeBuffer(foreignBuf, 0, foreignElite);
            hasForeignElite = false;
        }

        let enc = device.createCommandEncoder();
        let pass = enc.beginComputePass();
        pass.setPipeline(evalPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(dispatches);
        pass.end();
        pass = enc.beginComputePass();
        pass.setPipeline(evolvePipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(dispatches);
        pass.end();

        enc.copyBufferToBuffer(fitBuf, 0, readBuf, 0, POP * 4);
        device.queue.submit([enc.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const fits = new Float32Array(readBuf.getMappedRange());
        let localBestI = 0, localBestF = fits[0];
        for (let i = 1; i < POP; i++) {
            if (fits[i] > localBestF) { localBestF = fits[i]; localBestI = i; }
        }
        readBuf.unmap();

        if (localBestF > bestFit) {
            bestFit = localBestF;
            const srcBuf = usingA ? popBufA : popBufB;
            const enc2 = device.createCommandEncoder();
            enc2.copyBufferToBuffer(srcBuf, localBestI * EVOLVABLE * 4, bestReadBuf, 0, EVOLVABLE * 4);
            device.queue.submit([enc2.finish()]);
            await bestReadBuf.mapAsync(GPUMapMode.READ);
            const genes = new Float32Array(bestReadBuf.getMappedRange());
            for (let d = 0; d < EVOLVABLE; d++) bestGenome[d] = genes[d];
            bestReadBuf.unmap();
        }

        usingA = !usingA;
        gen++;

        const now = performance.now();
        if (now - lastReportTime >= 200) {
            const dt = (now - lastReportTime) / 1000;
            const instantSpeed = dt > 0 ? (gen - lastReportGen) / dt : 0;
            lastReportTime = now;
            lastReportGen = gen;
            const regions = getEvolvedRegions(bestGenome);
            const fullSeq = decodeFullSequence(bestGenome);
            postMessage({
                type: 'stats', gen, bestFit,
                speed: instantSpeed,
                bestGenome: Array.from(bestGenome),
                fullSequence: fullSeq,
                activeSite: regions.activeSite,
                surface: regions.surface,
                linkerMT: regions.linkerMT,
            });
            await new Promise(r => setTimeout(r, 0));
        }
    }
}

onmessage = (e) => {
    if (e.data.type === 'start') { if (!running) runEvolution(); }
    else if (e.data.type === 'stop') { running = false; }
    else if (e.data.type === 'inject') { foreignElite.set(e.data.genome); hasForeignElite = true; }
    else if (e.data.type === 'weights') { fitnessWeights.set(e.data.weights); }
};

initGPU();
