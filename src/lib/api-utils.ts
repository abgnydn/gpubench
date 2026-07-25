// Shared helpers for the API routes (previously copy-pasted into each one).

export function parseUA(ua: string): { browser: string; os: string } {
  let browser = "Unknown";
  let os = "Unknown";
  if (ua.includes("Edg/")) {
    browser = "Edge";
  } else if (ua.includes("Chrome/")) {
    const match = /Chrome\/([\d.]+)/.exec(ua);
    browser = match ? `Chrome ${match[1]}` : "Chrome";
  } else if (ua.includes("Firefox/")) {
    browser = "Firefox";
  } else if (ua.includes("Safari/") && !ua.includes("Chrome")) {
    browser = "Safari";
  }
  if (ua.includes("Mac OS X")) os = "macOS";
  else if (ua.includes("Windows")) os = "Windows";
  else if (ua.includes("CrOS")) os = "ChromeOS";
  else if (ua.includes("Android")) os = "Android";
  else if (ua.includes("iPhone") || ua.includes("iPad")) os = "iOS";
  else if (ua.includes("Linux")) os = "Linux";
  return { browser, os };
}

export function num(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return v;
}

export function str(v: unknown, max = 500): string {
  if (typeof v !== "string") return "";
  return v.slice(0, max);
}

export function bool(v: unknown): boolean {
  return v === true;
}

/** Median of the finite numbers in `values`; null when there are none.
 *  D1/SQLite has no percentile_cont, so median aggregations live in JS. */
export function median(values: unknown[]): number | null {
  const nums = values
    .map((v) => (typeof v === "number" ? v : Number(v)))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  if (nums.length === 0) return null;
  const mid = nums.length >> 1;
  return nums.length % 2 === 0 ? (nums[mid - 1]! + nums[mid]!) / 2 : nums[mid]!;
}

export function roundTo(v: number | null, digits: number): number | null {
  if (v === null) return null;
  const f = 10 ** digits;
  return Math.round(v * f) / f;
}

/** D1 stores booleans as 0/1; older rows may hold true/false. */
export function truthyFlag(v: unknown): boolean {
  return v === true || v === 1 || v === "1";
}

export function getClientIp(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  return forwarded?.split(",")[0]?.trim() ?? "unknown";
}

// In-memory rate limiter. Serverless caveat: each instance has its own map,
// so this only throttles bursts hitting one warm instance — it is a speed
// bump, not a security boundary. Expired entries are purged so the map
// cannot grow without bound on a long-lived instance.
export function createRateLimiter(maxPerMinute: number): (ip: string) => boolean {
  const hits = new Map<string, { count: number; resetAt: number }>();
  return (ip: string): boolean => {
    const now = Date.now();
    if (hits.size > 5_000) {
      for (const [key, entry] of hits) {
        if (now > entry.resetAt) hits.delete(key);
      }
    }
    const entry = hits.get(ip);
    if (!entry || now > entry.resetAt) {
      hits.set(ip, { count: 1, resetAt: now + 60_000 });
      return false;
    }
    entry.count++;
    return entry.count > maxPerMinute;
  };
}

/**
 * Sanity-check one benchmark's reported stats. The submission endpoint is
 * unauthenticated, so anything structurally inconsistent (min > mean,
 * negative std, throughput that doesn't match the reported mean time) is
 * rejected rather than poisoning the public dataset. All fields are optional
 * — rules only apply to values that were actually sent.
 */
export function timingStatsPlausible(stats: {
  gps: number | null;
  mean: number | null;
  min: number | null;
  max: number | null;
  std: number | null;
}): boolean {
  const { gps, mean, min, max, std } = stats;
  if (gps !== null && (gps <= 0 || gps > 1_000_000)) return false;
  if (mean !== null && (mean <= 0 || mean > 600_000)) return false;
  if (min !== null && min <= 0) return false;
  if (std !== null && std < 0) return false;
  if (min !== null && mean !== null && min > mean + 1e-6) return false;
  if (mean !== null && max !== null && mean > max + 1e-6) return false;
  if (gps !== null && mean !== null) {
    const implied = 1000 / mean;
    if (Math.abs(gps - implied) > Math.max(2, implied * 0.25)) return false;
  }
  return true;
}
