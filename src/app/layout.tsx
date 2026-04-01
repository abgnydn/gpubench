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
    "Run real WebGPU compute benchmarks. Kernel fusion achieves 159-720x over PyTorch on the same GPU. No install — just click Run.",
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
      "Kernel fusion: 159x (WebGPU) to 720x (CUDA) over PyTorch on the same GPU. Test yours — no install required.",
    type: "website",
    url: "https://gpubench.dev",
    siteName: "WebGPU Bench",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "WebGPU Bench — 159-720x over PyTorch via kernel fusion" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "WebGPU Bench — How fast is your GPU in the browser?",
    description: "Kernel fusion: 159x (WebGPU) to 720x (CUDA) over PyTorch on the same GPU. No install required.",
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
