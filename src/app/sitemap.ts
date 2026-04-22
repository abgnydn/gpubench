import type { MetadataRoute } from "next";

const BASE = "https://gpubench.dev";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  return [
    { url: `${BASE}/`, lastModified: now, changeFrequency: "daily", priority: 1.0 },
    { url: `${BASE}/results`, lastModified: now, changeFrequency: "daily", priority: 0.9 },
    { url: `${BASE}/why`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE}/transformer`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE}/swarm`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE}/zerotvm`, lastModified: now, changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE}/privacy`, lastModified: now, changeFrequency: "yearly", priority: 0.3 },
  ];
}
