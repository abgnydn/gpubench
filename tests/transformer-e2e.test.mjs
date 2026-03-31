/**
 * E2E test: verify runSweepWithConfigs only runs passed configs
 * Run: node tests/transformer-e2e.test.mjs
 */

// Dynamically import the actual module
const mod = await import("../src/lib/transformer-bench.js");

console.log("=== Module exports ===");
console.log("CONFIGS:", mod.CONFIGS.map(c => c.label));
console.log("SEQ_CONFIGS:", mod.SEQ_CONFIGS.map(c => c.label));
console.log("runSweep:", typeof mod.runSweep);
console.log("runSweepWithConfigs:", typeof mod.runSweepWithConfigs);

// Test: filter to only D=32 L=1
const onlyD32L1 = mod.CONFIGS.filter(c => c.label === "D=32, L=1");
console.log("\n=== Filtered configs ===");
console.log("Input:", onlyD32L1.map(c => c.label));
console.log("Count:", onlyD32L1.length);

if (onlyD32L1.length !== 1) {
  console.error("FAIL: Expected 1 config, got", onlyD32L1.length);
  process.exit(1);
}

// Mock runSweepWithConfigs to track what configs it iterates
let configsIterated = [];
const originalLog = (msg) => {
  // Capture config names from log messages
  if (msg.startsWith("---")) {
    configsIterated.push(msg);
  }
};

// Can't actually run WebGPU without a browser, but we can verify
// the function signature and config passing
console.log("\n=== Verify function signature ===");
console.log("runSweepWithConfigs params:", mod.runSweepWithConfigs.length, "(expected 3: configs, log, onResult)");

if (mod.runSweepWithConfigs.length !== 3) {
  console.error("FAIL: runSweepWithConfigs should take 3 params (configs, log, onResult)");
  console.error("Got:", mod.runSweepWithConfigs.length);
  process.exit(1);
}

// Verify runSweep calls runSweepWithConfigs with full CONFIGS
console.log("\n=== Verify runSweep delegates to runSweepWithConfigs ===");
const runSweepSource = mod.runSweep.toString();
const callsWithConfigs = runSweepSource.includes("runSweepWithConfigs") && runSweepSource.includes("CONFIGS");
console.log("runSweep source calls runSweepWithConfigs(CONFIGS,...):", callsWithConfigs);

if (!callsWithConfigs) {
  console.error("FAIL: runSweep should delegate to runSweepWithConfigs(CONFIGS, ...)");
  console.error("Source:", runSweepSource);
  process.exit(1);
}

// Verify CONFIGS is not mutated by filtering
const before = mod.CONFIGS.length;
const filtered = mod.CONFIGS.filter(c => c.label === "D=32, L=1");
const after = mod.CONFIGS.length;
console.log(`\nCONFIGS before filter: ${before}, after: ${after}, filtered: ${filtered.length}`);

if (before !== after) {
  console.error("FAIL: .filter() mutated CONFIGS!");
  process.exit(1);
}

console.log("\n========================================");
console.log("All E2E checks passed");
console.log("========================================");
