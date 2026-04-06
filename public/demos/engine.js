const POP = 4096;
const DIM = 2000;
const WORKGROUP = 64;

let running = false;
let isStopped = false;
let gen = 0;
let bestFit = -Infinity;
let bestGenome = new Float32Array(DIM);
let foreignElite = new Float32Array(DIM);
for (let i = 0; i < DIM; i++) foreignElite[i] = (Math.random() * 10.24) - 5.12;
let hasForeignElite = false;

const computeShaderCode = `
    struct Uniforms { generation: u32, seed: u32, foreignFit: f32, pad: f32 }

    @group(0) @binding(0) var<storage, read> pop: array<f32>;
    @group(0) @binding(1) var<storage, read_write> fit: array<f32>;
    @group(0) @binding(2) var<storage, read_write> next_pop: array<f32>;
    @group(0) @binding(3) var<uniform> u: Uniforms;
    @group(0) @binding(4) var<storage, read> foreign_elite: array<f32>;

    const PI: f32 = 3.14159265359;
    const DIM: u32 = ${DIM}u;
    const POP: u32 = ${POP}u;

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

    @compute @workgroup_size(${WORKGROUP})
    fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= POP) { return; }
        let base = idx * DIM;
        var sum = 0.0;
        for (var d = 0u; d < DIM; d++) {
            let x = pop[base + d];
            sum += x * x - 10.0 * cos(2.0 * PI * x);
        }
        fit[idx] = -(10.0 * f32(DIM) + sum);
    }

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
        
        // Elitism injection from MCP/P2P
        if (idx == 1u) {
            for (var d = 0u; d < DIM; d++) { next_pop[DIM + d] = foreign_elite[d]; }
            return;
        }
        
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
            if (r < 0.1) { gene += gauss(&seed) * 0.5; }
            else if (r < 0.3) { gene += gauss(&seed) * 0.05; }
            gene = clamp(gene, -5.12, 5.12);
            next_pop[myBase + d] = gene;
        }
    }
`;

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
        for (let i = 0; i < initPop.length; i++) initPop[i] = (Math.random() * 10.24) - 5.12;

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
    postMessage({ type: 'benchmark_start' });
    const t0 = performance.now();
    let lastReportTime = t0;

    while (running) {
        const bg = usingA ? bgA : bgB;
        const uData = new ArrayBuffer(16);
        new Uint32Array(uData, 0, 2).set([gen, Math.floor(Math.random() * 1e9)]);
        new Float32Array(uData, 8, 1).set([0]);
        
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
        }

        usingA = !usingA;
        gen++;

        const now = performance.now();
        if (now - lastReportTime >= 200) {
            lastReportTime = now;
            const elapsed = (now - t0) / 1000;
            postMessage({ type: 'stats', gen, bestFit, speed: gen / elapsed, bestGenome: Array.from(bestGenome) });
            await new Promise(r => setTimeout(r, 0));
        }
    }
}

onmessage = (e) => {
    if (e.data.type === 'start') {
        if (!running) runEvolution();
    } else if (e.data.type === 'stop') {
        running = false;
    } else if (e.data.type === 'inject') {
        foreignElite.set(e.data.genome);
        hasForeignElite = true;
    }
};

initGPU();
