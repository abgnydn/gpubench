import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://gpubench.dev"),
  alternates: { canonical: "/" },
  title: "WebGPU Bench — How fast is your GPU in the browser?",
  description:
    "Run real WebGPU compute benchmarks. 92 devices tested across 7 GPU vendors — kernel fusion median 71× on Apple Silicon, 56× on NVIDIA, 20× on phones. No install — just click Run.",
  keywords: [
    "WebGPU",
    "benchmark",
    "GPU",
    "compute shader",
    "browser",
    "performance",
    "WGSL",
  ],
  openGraph: {
    title: "WebGPU Bench — How fast is your GPU in the browser?",
    description:
      "92 devices tested across 7 GPU vendors. Kernel fusion median: 71× on Apple Silicon, 56× on NVIDIA, 20× on phones. Test yours — no install required.",
    type: "website",
    url: "https://gpubench.dev",
    siteName: "WebGPU Bench",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "WebGPU Bench — 71× median Apple Silicon, 56× NVIDIA, 20× phones, 92 devices" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "WebGPU Bench — How fast is your GPU in the browser?",
    description: "92 devices tested across 7 GPU vendors. Kernel fusion median: 71× Apple, 56× NVIDIA, 20× phones. No install required.",
    images: ["/og.png"],
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "SoftwareApplication",
      "@id": "https://gpubench.dev#app",
      "name": "WebGPU Bench",
      "url": "https://gpubench.dev",
      "applicationCategory": "DeveloperApplication",
      "operatingSystem": "Any (WebGPU browser)",
      "description": "Open WebGPU compute benchmarks. 92 devices tested across 7 GPU vendors — Rastrigin, N-body, Monte Carlo Pi, RL environments.",
      "author": { "@id": "https://gpubench.dev#author" },
      "isPartOf": {
        "@type": "CreativeWork",
        "name": "Kernel-fusion research line",
        "url": "https://kernelfusion.dev"
      }
    },
    {
      "@type": "Person",
      "@id": "https://gpubench.dev#author",
      "name": "Ahmet Baris Gunaydin",
      "url": "https://gpubench.dev",
      "sameAs": [
        "https://barisgunaydin.com",
        "https://kernelfusion.dev",
        "https://gpubench.dev",
        "https://zerotvm.com",
        "https://webgpudna.com",
        "https://neuropulse.live",
        "https://markview.ai",
        "https://safenpm.dev",
        "https://github.com/abgnydn",
        "https://www.linkedin.com/in/abgnydn/"
      ]
    },
    {
      "@type": "WebSite",
      "@id": "https://gpubench.dev#site",
      "url": "https://gpubench.dev",
      "name": "WebGPU Bench",
      "publisher": { "@id": "https://gpubench.dev#author" }
    }
  ]
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className={`${inter.className} min-h-screen`}>{children}<Analytics /></body>
    </html>
  );
}
