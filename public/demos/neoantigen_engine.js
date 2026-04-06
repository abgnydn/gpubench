// Neoantigen Evolution Engine — WebGPU Worker
//
// Evolves 9-mer peptides for MHC class I binding (HLA-A*02:01).
// Fitness: anchor residue matching + position-specific preferences +
//          proteasomal cleavage + TAP transport + immunogenicity
//
// Population: 4096 candidates, 9 positions each
// Based on PSSM data from IEDB/NetMHCpan literature

const POP = 4096;
const DIM = 9;       // 9-mer peptides for MHC-I
const WORKGROUP = 64;

let running = false;
let gen = 0;
let bestFit = -Infinity;
let bestGenome = new Float32Array(DIM);
let foreignElite = new Float32Array(DIM);
let hasForeignElite = false;

// Allele weights — can switch between HLA alleles via message
// Default: HLA-A*02:01 (most common, ~40% of Caucasians)
let alleleIndex = 0;

const AA = 'ACDEFGHIKLMNPQRSTVWY';

function decodeSequence(weights) {
    let seq = '';
    for (let i = 0; i < DIM; i++) {
        const t = Math.tanh(weights[i]);
        const aaIdx = Math.min(19, Math.max(0, Math.floor((t * 0.5 + 0.5) * 20)));
        seq += AA[aaIdx];
    }
    return seq;
}

const computeShaderCode = `
    struct Uniforms {
        generation: u32,
        seed: u32,
        allele: u32,    // 0=HLA-A*02:01, 1=HLA-A*03:01, 2=HLA-B*07:02, 3=HLA-B*08:01
        pad: u32
    }

    @group(0) @binding(0) var<storage, read> pop: array<f32>;
    @group(0) @binding(1) var<storage, read_write> fit: array<f32>;
    @group(0) @binding(2) var<storage, read_write> next_pop: array<f32>;
    @group(0) @binding(3) var<uniform> u: Uniforms;
    @group(0) @binding(4) var<storage, read> foreign_elite: array<f32>;

    const PI: f32 = 3.14159265359;
    const DIM: u32 = ${DIM}u;
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
    // Molecular weight (Da, scaled /100)
    const AA_MW = array<f32, 20>(
        0.89, 1.21, 1.33, 1.47, 1.65, 0.75, 1.55, 1.31, 1.46, 1.31,
        1.49, 1.32, 1.15, 1.46, 1.74, 1.05, 1.19, 1.17, 2.04, 1.81
    );
    // Proteasomal cleavage preference at C-terminus (higher = more likely to be cleaved)
    const AA_PROTEASOME = array<f32, 20>(
        0.4, 0.2, 0.3, 0.3, 0.7, 0.2, 0.3, 0.5, 0.6, 0.8,
        0.5, 0.3, 0.2, 0.3, 0.6, 0.3, 0.3, 0.5, 0.6, 0.7
    );
    // TAP transport preference (higher = better transport into ER)
    const AA_TAP = array<f32, 20>(
        0.3, 0.2, 0.1, 0.1, 0.8, 0.2, 0.3, 0.7, 0.4, 0.9,
        0.5, 0.2, 0.2, 0.2, 0.5, 0.2, 0.3, 0.6, 0.7, 0.8
    );

    // ─── HLA-A*02:01 Position-Specific Scoring Matrix ───
    // 9 positions x 20 amino acids = 180 values
    // Based on IEDB binding data: strong binders have L/M at P2, V/L at P9
    // P1: prefers hydrophobic/small
    const A0201_P1 = array<f32, 20>(
        0.30, 0.05, 0.05, 0.10, 0.20, 0.40, 0.10, 0.20, 0.10, 0.25,
        0.20, 0.10, 0.05, 0.10, 0.05, 0.25, 0.20, 0.20, 0.10, 0.15
    );
    // P2: PRIMARY ANCHOR — L, M strongly preferred; I, V, A accepted
    const A0201_P2 = array<f32, 20>(
        0.30, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.30, 0.00, 0.95,
        0.80, 0.00, 0.00, 0.05, 0.00, 0.05, 0.15, 0.40, 0.00, 0.00
    );
    // P3: broad tolerance, slight preference for hydrophobic
    const A0201_P3 = array<f32, 20>(
        0.20, 0.05, 0.10, 0.15, 0.25, 0.15, 0.15, 0.20, 0.15, 0.25,
        0.15, 0.10, 0.10, 0.15, 0.10, 0.15, 0.15, 0.25, 0.20, 0.20
    );
    // P4: broad tolerance
    const A0201_P4 = array<f32, 20>(
        0.15, 0.05, 0.10, 0.20, 0.20, 0.10, 0.15, 0.20, 0.20, 0.20,
        0.15, 0.15, 0.10, 0.15, 0.15, 0.15, 0.15, 0.20, 0.15, 0.15
    );
    // P5: broad tolerance, slight aromatic preference
    const A0201_P5 = array<f32, 20>(
        0.15, 0.05, 0.10, 0.15, 0.25, 0.10, 0.15, 0.20, 0.15, 0.20,
        0.15, 0.10, 0.10, 0.15, 0.10, 0.15, 0.15, 0.20, 0.25, 0.25
    );
    // P6: broad tolerance
    const A0201_P6 = array<f32, 20>(
        0.20, 0.05, 0.10, 0.15, 0.20, 0.15, 0.15, 0.20, 0.15, 0.20,
        0.15, 0.10, 0.10, 0.15, 0.10, 0.15, 0.15, 0.25, 0.15, 0.15
    );
    // P7: preference for hydrophobic, especially I, L, V
    const A0201_P7 = array<f32, 20>(
        0.20, 0.05, 0.05, 0.10, 0.25, 0.10, 0.10, 0.30, 0.10, 0.30,
        0.15, 0.05, 0.05, 0.10, 0.05, 0.10, 0.15, 0.25, 0.20, 0.20
    );
    // P8: preference for small/hydrophobic
    const A0201_P8 = array<f32, 20>(
        0.25, 0.05, 0.05, 0.10, 0.20, 0.20, 0.10, 0.20, 0.15, 0.25,
        0.15, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.25, 0.10, 0.15
    );
    // P9: PRIMARY ANCHOR — V, L strongly preferred; I, A accepted
    const A0201_P9 = array<f32, 20>(
        0.25, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.30, 0.00, 0.80,
        0.15, 0.00, 0.00, 0.05, 0.00, 0.05, 0.15, 0.95, 0.00, 0.00
    );

    // ─── HLA-A*03:01 PSSM (prefers K/R at P9 — basic C-terminal anchor) ───
    const A0301_P2 = array<f32, 20>(
        0.30, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.30, 0.05, 0.80,
        0.60, 0.00, 0.00, 0.05, 0.05, 0.10, 0.20, 0.40, 0.00, 0.05
    );
    const A0301_P9 = array<f32, 20>(
        0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.05, 0.95, 0.10,
        0.05, 0.00, 0.00, 0.00, 0.80, 0.00, 0.05, 0.05, 0.00, 0.10
    );

    // ─── HLA-B*07:02 PSSM (prefers P at P2) ───
    const B0702_P2 = array<f32, 20>(
        0.15, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.10, 0.00, 0.20,
        0.10, 0.00, 0.90, 0.00, 0.00, 0.10, 0.05, 0.10, 0.00, 0.00
    );
    const B0702_P9 = array<f32, 20>(
        0.15, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.15, 0.00, 0.80,
        0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.30, 0.00, 0.15
    );

    // ─── HLA-B*08:01 PSSM ───
    const B0801_P2 = array<f32, 20>(
        0.10, 0.00, 0.10, 0.20, 0.05, 0.00, 0.00, 0.05, 0.10, 0.10,
        0.05, 0.05, 0.05, 0.10, 0.70, 0.05, 0.05, 0.05, 0.00, 0.00
    );
    const B0801_P9 = array<f32, 20>(
        0.10, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.10, 0.10, 0.80,
        0.15, 0.00, 0.00, 0.00, 0.10, 0.00, 0.05, 0.10, 0.10, 0.10
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

    // ─── MHC-I Binding Fitness Function ───
    fn score_neoantigen(idx: u32) -> f32 {
        let base = idx * DIM;
        var seq: array<u32, 9>;
        for (var i = 0u; i < DIM; i++) {
            seq[i] = weight_to_aa(pop[base + i]);
        }

        var binding: f32 = 0.0;
        var immunogenicity: f32 = 0.0;
        var processing: f32 = 0.0;

        // ═══ MHC BINDING (weight ~0.5) ═══

        // Position-specific scoring — allele-dependent
        // All alleles share P1, P3-P8 from A*02:01 (similar preferences)
        // P2 and P9 are the key anchor positions that differ by allele
        binding += A0201_P1[seq[0]] * 3.0;
        binding += A0201_P3[seq[2]] * 2.0;
        binding += A0201_P4[seq[3]] * 1.5;
        binding += A0201_P5[seq[4]] * 2.0;
        binding += A0201_P6[seq[5]] * 1.5;
        binding += A0201_P7[seq[6]] * 2.5;
        binding += A0201_P8[seq[7]] * 2.0;

        // Allele-specific anchor scoring (P2 and P9 — the critical positions)
        if (u.allele == 0u) {
            // HLA-A*02:01: L/M at P2, V/L at P9
            binding += A0201_P2[seq[1]] * 12.0;
            binding += A0201_P9[seq[8]] * 12.0;
        } else if (u.allele == 1u) {
            // HLA-A*03:01: L/M at P2, K/R at P9
            binding += A0301_P2[seq[1]] * 12.0;
            binding += A0301_P9[seq[8]] * 12.0;
        } else if (u.allele == 2u) {
            // HLA-B*07:02: P at P2, L at P9
            binding += B0702_P2[seq[1]] * 12.0;
            binding += B0702_P9[seq[8]] * 12.0;
        } else {
            // HLA-B*08:01: R at P2, L at P9
            binding += B0801_P2[seq[1]] * 12.0;
            binding += B0801_P9[seq[8]] * 12.0;
        }

        // ═══ IMMUNOGENICITY (weight ~0.3) ═══
        // T-cell recognition prefers aromatic and large residues at TCR-contact positions (P4, P5, P6)

        // Aromatic residues at TCR-contact positions
        for (var i = 3u; i < 7u; i++) {
            if (seq[i] == 4u || seq[i] == 18u || seq[i] == 19u) { // F, W, Y
                immunogenicity += 3.0;
            }
        }

        // Large hydrophobic residues at solvent-exposed positions
        for (var i = 3u; i < 7u; i++) {
            if (AA_HYDRO[seq[i]] > 0.5) {
                immunogenicity += 1.5;
            }
        }

        // Avoid self-similarity: penalize common human proteome motifs
        // Consecutive identical residues (rare in immunogenic peptides)
        for (var i = 0u; i < 8u; i++) {
            if (seq[i] == seq[i + 1u]) {
                immunogenicity -= 2.0;
            }
        }

        // Charge at TCR-contact positions (slight positive charge is immunogenic)
        var tcr_charge: f32 = 0.0;
        for (var i = 3u; i < 7u; i++) {
            tcr_charge += AA_CHARGE[seq[i]];
        }
        if (tcr_charge > 0.0 && tcr_charge < 2.5) {
            immunogenicity += 2.0;
        }

        // Sequence complexity (avoid low-complexity peptides)
        var unique_count: u32 = 0u;
        var seen: array<u32, 20>;
        for (var i = 0u; i < 20u; i++) { seen[i] = 0u; }
        for (var i = 0u; i < DIM; i++) {
            if (seen[seq[i]] == 0u) { unique_count++; seen[seq[i]] = 1u; }
        }
        if (unique_count >= 5u) { immunogenicity += 3.0; }
        else if (unique_count >= 3u) { immunogenicity += 1.0; }

        // ═══ ANTIGEN PROCESSING (weight ~0.2) ═══

        // Proteasomal cleavage at C-terminus (position 9)
        processing += AA_PROTEASOME[seq[8]] * 5.0;

        // TAP transport preference (peptides must be transported to ER)
        var tap_score: f32 = 0.0;
        for (var i = 0u; i < DIM; i++) {
            tap_score += AA_TAP[seq[i]];
        }
        processing += tap_score * 0.8;

        // N-terminal residue preference for ER aminopeptidase trimming
        // Small residues at N-term are trimmed efficiently
        if (seq[0] == 0u || seq[0] == 5u || seq[0] == 15u) { // A, G, S
            processing += 2.0;
        }

        // ═══ WEIGHTED TOTAL ═══
        let total = binding * 0.50 + immunogenicity * 0.30 + processing * 0.20;
        return max(0.0, total);
    }

    // ─── Evaluate ───
    @compute @workgroup_size(${WORKGROUP})
    fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= POP) { return; }
        fit[idx] = score_neoantigen(idx);
    }

    // ─── Evolve ───
    @compute @workgroup_size(${WORKGROUP})
    fn evolve(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= POP) { return; }
        var seed = idx + u.generation * POP + u.seed;

        // Elitism: slot 0 = best, slot 1 = foreign elite
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
            // Anchor positions (1, 8) — conservative mutation (critical for binding)
            if (d == 1u || d == 8u) {
                if (r < 0.05) { gene += gauss(&seed) * 0.4; }
                else if (r < 0.12) { gene += gauss(&seed) * 0.06; }
            } else {
                // Non-anchor positions — more aggressive exploration
                if (r < 0.10) { gene += gauss(&seed) * 0.6; }
                else if (r < 0.25) { gene += gauss(&seed) * 0.1; }
            }
            gene = clamp(gene, -3.0, 3.0);
            next_pop[myBase + d] = gene;
        }
    }
`;

// ─── GPU Setup ───
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

        const initPop = new Float32Array(POP * DIM);
        for (let i = 0; i < initPop.length; i++) initPop[i] = (Math.random() * 4) - 2;

        popBufA = device.createBuffer({ size: initPop.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        popBufB = device.createBuffer({ size: initPop.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        fitBuf = device.createBuffer({ size: POP * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        readBuf = device.createBuffer({ size: POP * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        uniformBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        foreignBuf = device.createBuffer({ size: DIM * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(foreignBuf, 0, foreignElite);
        device.queue.writeBuffer(popBufA, 0, initPop);
        bestReadBuf = device.createBuffer({ size: DIM * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

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
    let topCandidates = [];

    while (running) {
        const bg = usingA ? bgA : bgB;
        const uData = new Uint32Array([gen, Math.floor(Math.random() * 1e9), alleleIndex, 0]);

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
            enc2.copyBufferToBuffer(srcBuf, localBestI * DIM * 4, bestReadBuf, 0, DIM * 4);
            device.queue.submit([enc2.finish()]);
            await bestReadBuf.mapAsync(GPUMapMode.READ);
            const genes = new Float32Array(bestReadBuf.getMappedRange());
            for (let d = 0; d < DIM; d++) bestGenome[d] = genes[d];
            bestReadBuf.unmap();

            // Track top unique candidates
            const seq = decodeSequence(bestGenome);
            if (!topCandidates.find(c => c.sequence === seq)) {
                topCandidates.push({ sequence: seq, fitness: bestFit, gen });
                topCandidates.sort((a, b) => b.fitness - a.fitness);
                if (topCandidates.length > 20) topCandidates.pop();
            }
        }

        usingA = !usingA;
        gen++;

        const now = performance.now();
        if (now - lastReportTime >= 200) {
            lastReportTime = now;
            const elapsed = (now - t0) / 1000;
            const seq = decodeSequence(bestGenome);
            postMessage({
                type: 'stats', gen, bestFit,
                speed: gen / elapsed,
                bestGenome: Array.from(bestGenome),
                sequence: seq,
                topCandidates: topCandidates.slice(0, 10),
                allele: alleleIndex,
            });
            await new Promise(r => setTimeout(r, 0));
        }
    }
}

onmessage = (e) => {
    if (e.data.type === 'start') { if (!running) runEvolution(); }
    else if (e.data.type === 'stop') { running = false; }
    else if (e.data.type === 'inject') { foreignElite.set(e.data.genome); hasForeignElite = true; }
    else if (e.data.type === 'allele') { alleleIndex = e.data.allele; }
};

initGPU();
