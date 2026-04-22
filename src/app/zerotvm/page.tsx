"use client";

import { TabSwitcher } from "@/components/tab-switcher";
import { LINKS } from "@/lib/constants";

/**
 * /zerotvm — hosts the Zero-TVM Phi-3 chat demo inside the standard
 * gpubench layout (shared logo + TabSwitcher at top).
 *
 * The chat is a pre-bundled Vite module at /demos/zerotvm-chat.html.
 * Rather than porting ~6 MB of compiled WGSL/tokenizer/weights logic
 * into React, we render it in an iframe with ?embed=1 so the HTML
 * file hides its own redundant nav/footer and fills this container.
 *
 * Above the iframe there's a compact info strip — Zero-TVM is a
 * separate project (not gpubench), so users landing on this tab
 * cold need the tagline + links (site, repo, dispatch viz) to
 * understand what they're about to run. Stats are from the
 * zerotvm.com landing page.
 *
 * Telemetry: the chat loads /demos/device-telemetry.js inside the
 * iframe itself, so window.reportDevice is defined in the iframe's
 * own window — the generate hook works identically to standalone.
 */

// Stats surfaced as chips. Source: zerotvm.com landing page
// (2026-04). The canonical "10" claim is now "10 kernel roles"
// (the file count expanded to 27 with A/B tuning variants) —
// reflect that precisely rather than the old "10 shaders."
const CHIPS: Array<{ label: string; value: string }> = [
  { value: "3.8B", label: "Phi-3 Mini params" },
  { value: "10", label: "kernel roles" },
  { value: "27", label: "WGSL files" },
  { value: "228", label: "dispatches / token" },
  { value: "33 KB", label: "gzipped JS" },
  { value: "~40 tok/s", label: "M2 Pro" },
];

// External surface. Ordered for a first-time visitor: site →
// repo → deeper write-ups (docs, architecture) → comparison /
// introspection tools. Keep the row short enough to fit one line;
// the project site itself is the canonical index for everything
// else (Validate, TVM shader source, Compiler Chat, etc.).
const EXTERNAL: Array<{ href: string; label: string; hint: string }> = [
  { href: LINKS.zerotvmSite, label: "zerotvm.com", hint: "Project site" },
  { href: LINKS.zerotvmRepo, label: "GitHub", hint: "Source (MIT)" },
  { href: LINKS.zerotvmDocs, label: "Docs", hint: "How it works" },
  { href: LINKS.zerotvmArchitecture, label: "Architecture", hint: "32-layer pipeline" },
  { href: LINKS.zerotvmDispatchViz, label: "Dispatch viz", hint: "228 per decode token" },
  { href: LINKS.zerotvmWebllmBench, label: "vs WebLLM", hint: "22% slower, 10× less code" },
  { href: "/results", label: "All runs", hint: "Telemetry across devices" },
];

export default function ZeroTvmPage() {
  return (
    // Fixed-height page: nav + info strip on top, iframe fills the
    // rest of the viewport. The chat needs a constrained height to
    // get its scrollable-messages + pinned-input layout right.
    <div className="h-screen flex flex-col">
      {/* Standard gpubench header — same as other top-level pages */}
      <header className="max-w-3xl mx-auto px-6 pt-8 pb-2 text-center w-full flex-shrink-0">
        <div className="flex items-center justify-center gap-2 mb-6">
          <svg
            className="w-5 h-5 text-bench-accent animate-spin-slow"
            viewBox="0 0 32 32"
            fill="none"
          >
            <circle
              cx="16"
              cy="16"
              r="14"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeDasharray="60 28"
              strokeLinecap="round"
            />
            <circle cx="16" cy="16" r="2.5" fill="currentColor" />
          </svg>
          <span className="font-bold text-bench-text">WebGPU Bench</span>
        </div>
        <TabSwitcher />
      </header>

      {/* Info strip: tagline + one-line description, a row of
          compact stat chips, and external links. Kept tight
          vertically so the chat iframe below still gets most of
          the viewport. */}
      <section className="max-w-5xl w-full mx-auto px-6 pt-4 pb-3 flex-shrink-0">
        <div className="card p-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <h1 className="text-lg font-extrabold tracking-tight text-bench-text">
                Zero-TVM{" "}
                <span className="font-medium text-bench-muted">
                  — Run Phi-3 in your browser with 10 hand-written shaders
                </span>
              </h1>
              <p className="text-xs text-bench-muted mt-1 max-w-3xl leading-relaxed">
                In <span className="text-bench-text">Feb 2026</span> Hugging
                Face shipped <span className="text-bench-text">Transformers.js v4</span>{" "}
                — a C++ WebGPU runtime built with Microsoft&apos;s ONNX Runtime
                team — as the production answer for browser LLM inference.
                Zero-TVM shows that for Phi-3 Mini specifically, the answer
                can instead be{" "}
                <span className="text-bench-text">10 kernel roles across 27 WGSL files</span>{" "}
                and ~2,000 lines of TypeScript. No compiler, no WASM, no
                server. Requires WebGPU with{" "}
                <code className="text-bench-accent">shader-f16</code>; first
                load downloads ~2.1 GB of Q4F16 weights, cached after.
              </p>
            </div>
          </div>

          {/* Stats row — compact tabular-num chips */}
          <div className="mt-3 flex flex-wrap gap-2">
            {CHIPS.map((c) => (
              <div
                key={c.label}
                className="inline-flex items-baseline gap-1.5 rounded-md border border-bench-border bg-bench-bg/40 px-2 py-1"
              >
                <span className="text-bench-accent text-xs font-extrabold tabular-nums">
                  {c.value}
                </span>
                <span className="text-[10px] text-bench-muted uppercase tracking-wider">
                  {c.label}
                </span>
              </div>
            ))}
          </div>

          {/* External links — open in a new tab so the chat (which
              may have a multi-GB model warming up in the iframe)
              doesn't get unloaded if the user clicks around. */}
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
            {EXTERNAL.map((l, i) => {
              const external = l.href.startsWith("http");
              return (
                <a
                  key={l.href}
                  href={l.href}
                  target={external ? "_blank" : undefined}
                  rel={external ? "noopener noreferrer" : undefined}
                  className="group inline-flex items-baseline gap-1.5 text-bench-muted hover:text-bench-accent transition-colors"
                >
                  <span className="font-semibold underline decoration-bench-border group-hover:decoration-bench-accent underline-offset-2">
                    {l.label}
                  </span>
                  <span className="text-[10px] text-bench-muted/60 group-hover:text-bench-muted">
                    {l.hint}
                  </span>
                  {i < EXTERNAL.length - 1 && (
                    <span className="text-bench-muted/30 ml-3" aria-hidden>
                      ·
                    </span>
                  )}
                </a>
              );
            })}
          </div>
        </div>
      </section>

      {/* Chat surface — iframe fills remaining viewport. Bordered and
          rounded to feel like a contained demo panel rather than a
          raw iframe. */}
      <div className="flex-1 max-w-5xl w-full mx-auto px-6 pb-6 min-h-0">
        <iframe
          src="/demos/zerotvm-chat.html?embed=1"
          title="Zero-TVM Phi-3 Chat"
          className="w-full h-full rounded-xl border border-bench-border bg-bench-surface"
          // webgpu + storage-access required for GPU inference + cached
          // weights in IndexedDB. Same-origin (both served from gpubench),
          // so the permissions policy simply forwards the top-level grants.
          allow="webgpu *; storage-access *; clipboard-read *; clipboard-write *"
        />
      </div>
    </div>
  );
}
