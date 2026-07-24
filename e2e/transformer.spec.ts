import { test, expect } from "@playwright/test";

// Runs the real transformer sweep on the smallest config and asserts the
// numerical equivalence check passes: fused, unfused, and parallel variants
// must compute the same function or every published speedup is meaningless.
test("transformer sweep: variants verified equivalent, speedups reported", async ({ page }) => {
  await page.goto("/transformer");

  // Keep only D=32, L=1 (the default selection is 4 configs — too slow for CI).
  for (const label of ["D=32, L=4", "D=64, L=1", "D=64, L=4"]) {
    await page.locator("label", { hasText: label }).click();
  }
  await expect(page.locator('input[type="checkbox"]:checked')).toHaveCount(1);

  await page.getByRole("button", { name: /Run Full Sweep/ }).click();

  // Equivalence line appears before any timing runs.
  await expect(page.getByText("max|Δ| vs fused")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("MISMATCH")).toHaveCount(0);
  await expect(page.getByText(/max\|Δ\| vs fused.*OK/)).toBeVisible();

  // Full sweep completes and the summary table renders with both baselines.
  await expect(page.getByText("FULL RESULTS")).toBeVisible({ timeout: 150_000 });
  await expect(page.getByRole("cell", { name: "D=32, L=1" })).toBeVisible();
  await expect(page.getByText("output mismatch")).toHaveCount(0);
});
