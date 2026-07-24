import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": resolve(__dirname, "src") } },
  test: {
    environment: "jsdom",
    // e2e/ is Playwright's — it must not be collected by vitest.
    include: ["tests/**/*.test.{js,ts,mjs}"],
  },
});
