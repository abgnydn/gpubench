import { defineConfig } from "@playwright/test";

// WebGPU is gated behind a flag in Chromium's headless mode.
// On macOS we use ANGLE+Metal; on Linux/Windows we use Vulkan.
// `--enable-unsafe-webgpu` is required because adapter requests are
// otherwise denied in non-secure / headless contexts.
const WEBGPU_ARGS = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan",
  "--use-angle=metal",
  "--no-sandbox",
];

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000, // 3 min — benchmarks can be slow on cold start
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: "list",

  use: {
    // Port chosen to avoid colliding with other local dev servers (3000/3001).
    baseURL: "http://127.0.0.1:3017",
    headless: true,
    launchOptions: {
      args: WEBGPU_ARGS,
    },
    trace: "retain-on-failure",
  },

  webServer: {
    // Production build is more deterministic than dev mode (no HMR, no
    // turbopack dynamic-import quirks). The build runs once per `test:e2e`.
    command: "npm run build && npx next start --port 3017",
    url: "http://127.0.0.1:3017",
    reuseExistingServer: false,
    timeout: 300_000,
  },
});
