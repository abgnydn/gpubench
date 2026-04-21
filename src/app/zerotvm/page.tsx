"use client";

import { TabSwitcher } from "@/components/tab-switcher";

/**
 * /zerotvm — hosts the Zero-TVM Phi-3 chat demo inside the standard
 * gpubench layout (shared logo + TabSwitcher at top).
 *
 * The chat is a pre-bundled Vite module at /demos/zerotvm-chat.html.
 * Rather than porting ~6 MB of compiled WGSL/tokenizer/weights logic
 * into React, we render it in an iframe with ?embed=1 so the HTML
 * file hides its own redundant nav/footer and fills this container.
 *
 * Telemetry: the chat loads /demos/device-telemetry.js inside the
 * iframe itself, so window.reportDevice is defined in the iframe's
 * own window — the generate hook works identically to standalone.
 */
export default function ZeroTvmPage() {
  return (
    // Fixed-height page: nav on top, iframe fills the rest of the
    // viewport. The chat needs a constrained height to get its
    // scrollable-messages + pinned-input layout right.
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
