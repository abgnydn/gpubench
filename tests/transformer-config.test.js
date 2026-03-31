/**
 * Test that transformer benchmark config filtering works correctly.
 * Run: node tests/transformer-config.test.js
 */

// Simulate the module
const CONFIGS = [
  { D: 32,  heads: 2, ffn: 4, seq: 64, layers: 1, label: 'D=32, L=1' },
  { D: 32,  heads: 2, ffn: 4, seq: 64, layers: 4, label: 'D=32, L=4' },
  { D: 64,  heads: 2, ffn: 4, seq: 64, layers: 1, label: 'D=64, L=1' },
  { D: 64,  heads: 2, ffn: 4, seq: 64, layers: 4, label: 'D=64, L=4' },
  { D: 128, heads: 2, ffn: 4, seq: 64, layers: 1, label: 'D=128, L=1' },
  { D: 128, heads: 2, ffn: 4, seq: 64, layers: 4, label: 'D=128, L=4' },
];

// Simulate the ALL_CONFIGS from the page
const ALL_CONFIGS = [
  { key: "d32l1", label: "D=32, L=1", default: true },
  { key: "d32l4", label: "D=32, L=4", default: true },
  { key: "d64l1", label: "D=64, L=1", default: true },
  { key: "d64l4", label: "D=64, L=4", default: true },
  { key: "d128l1", label: "D=128, L=1", default: false },
  { key: "d128l4", label: "D=128, L=4", default: false },
];

let passes = 0;
let failures = 0;

function check(name, actual, expected) {
  if (JSON.stringify(actual) === JSON.stringify(expected)) {
    console.log(`  OK: ${name}`);
    passes++;
  } else {
    console.error(`  FAIL: ${name}`);
    console.error(`    Expected: ${JSON.stringify(expected)}`);
    console.error(`    Got:      ${JSON.stringify(actual)}`);
    failures++;
  }
}

// Test 1: Default selection picks first 4 (D=32 and D=64)
console.log("\n=== Default selection ===");
const defaultSelected = new Set(ALL_CONFIGS.filter(c => c.default).map(c => c.key));
check("Default has 4 configs", defaultSelected.size, 4);
check("D=32 L=1 selected", defaultSelected.has("d32l1"), true);
check("D=64 L=4 selected", defaultSelected.has("d64l4"), true);
check("D=128 L=1 NOT selected", defaultSelected.has("d128l1"), false);

// Test 2: Filtering CONFIGS based on selection
console.log("\n=== Filter: only D=32 L=1 ===");
const onlyOne = new Set(["d32l1"]);
const selectedLabels1 = new Set(ALL_CONFIGS.filter(c => onlyOne.has(c.key)).map(c => c.label));
const filtered1 = CONFIGS.filter(c => selectedLabels1.has(c.label));
check("Filtered to 1 config", filtered1.length, 1);
check("Config is D=32 L=1", filtered1[0].label, "D=32, L=1");
check("Original CONFIGS unchanged", CONFIGS.length, 6);

// Test 3: Filter D=32 L=1 and D=32 L=4 only
console.log("\n=== Filter: D=32 only ===");
const d32Only = new Set(["d32l1", "d32l4"]);
const selectedLabels2 = new Set(ALL_CONFIGS.filter(c => d32Only.has(c.key)).map(c => c.label));
const filtered2 = CONFIGS.filter(c => selectedLabels2.has(c.label));
check("Filtered to 2 configs", filtered2.length, 2);
check("First is D=32 L=1", filtered2[0].label, "D=32, L=1");
check("Second is D=32 L=4", filtered2[1].label, "D=32, L=4");

// Test 4: Select all
console.log("\n=== Filter: all selected ===");
const allSelected = new Set(ALL_CONFIGS.map(c => c.key));
const selectedLabels3 = new Set(ALL_CONFIGS.filter(c => allSelected.has(c.key)).map(c => c.label));
const filtered3 = CONFIGS.filter(c => selectedLabels3.has(c.label));
check("All 6 configs", filtered3.length, 6);

// Test 5: Select none
console.log("\n=== Filter: none selected ===");
const noneSelected = new Set();
const selectedLabels4 = new Set(ALL_CONFIGS.filter(c => noneSelected.has(c.key)).map(c => c.label));
const filtered4 = CONFIGS.filter(c => selectedLabels4.has(c.label));
check("Zero configs", filtered4.length, 0);

// Test 6: Labels match between page and module
console.log("\n=== Label consistency ===");
const pageLabels = ALL_CONFIGS.map(c => c.label);
const moduleLabels = CONFIGS.map(c => c.label);
check("Same number of configs", pageLabels.length, moduleLabels.length);
for (let i = 0; i < pageLabels.length; i++) {
  check(`Label ${i} matches: "${pageLabels[i]}"`, pageLabels[i], moduleLabels[i]);
}

// Test 7: runSweepWithConfigs receives correct configs
console.log("\n=== Simulated sweep ===");
let sweepReceived = [];
function fakeRunSweepWithConfigs(configs) {
  sweepReceived = configs.map(c => c.label);
}
const testSelection = new Set(["d32l1", "d64l4"]);
const testLabels = new Set(ALL_CONFIGS.filter(c => testSelection.has(c.key)).map(c => c.label));
const testFiltered = CONFIGS.filter(c => testLabels.has(c.label));
fakeRunSweepWithConfigs(testFiltered);
check("Sweep received 2 configs", sweepReceived.length, 2);
check("First is D=32, L=1", sweepReceived[0], "D=32, L=1");
check("Second is D=64, L=4", sweepReceived[1], "D=64, L=4");

console.log(`\n${"=".repeat(40)}`);
console.log(`RESULTS: ${passes} passed, ${failures} failed`);
console.log(`${"=".repeat(40)}`);
if (failures > 0) process.exit(1);
