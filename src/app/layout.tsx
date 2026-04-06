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
  title: "WebGPU Bench — How fast is your GPU in the browser?",
  description:
    "Run real WebGPU compute benchmarks. 487 devices tested — kernel fusion averages 4,081x on Apple Silicon, 826x on phones. No install — just click Run.",
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
      "487 devices tested. Kernel fusion: 4,081x on Apple Silicon, 826x on phones. Test yours — no install required.",
    type: "website",
    url: "https://gpubench.dev",
    siteName: "WebGPU Bench",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "WebGPU Bench — 4,081x Apple Silicon, 826x phones, 487 devices" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "WebGPU Bench — How fast is your GPU in the browser?",
    description: "487 devices tested. Kernel fusion: 4,081x on Apple Silicon, 826x on phones. No install required.",
    images: ["/og.png"],
  },
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
      </head>
      <body className={`${inter.className} min-h-screen`}>{children}<Analytics /></body>
    </html>
  );
}
