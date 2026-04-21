"use client";

import { usePathname } from "next/navigation";

/**
 * Single source of truth for the primary nav tabs.
 *
 * The active tab is derived from the route via usePathname() — pages
 * don't pass an `active` prop, so they can't drift.
 *
 */
const TABS = [
  { label: "GPU Compute", href: "/" },
  { label: "Transformer Fusion", href: "/transformer" },
  { label: "Distributed P2P", href: "/swarm" },
  { label: "Zero-TVM", href: "/zerotvm" },
  { label: "All Results", href: "/results" },
] as const;

export function TabSwitcher() {
  const pathname = usePathname();

  return (
    <div className="inline-flex rounded-lg bg-bench-surface border border-bench-border p-1 mb-8">
      {TABS.map((tab) => {
        const isActive = tab.href === pathname;
        const className =
          "px-4 py-2 rounded-md text-sm font-medium transition";

        return isActive ? (
          <span
            key={tab.href}
            className={`${className} bg-bench-accent/10 text-bench-accent`}
            aria-current="page"
          >
            {tab.label}
          </span>
        ) : (
          <a
            key={tab.href}
            href={tab.href}
            className={`${className} text-bench-muted hover:text-bench-text`}
          >
            {tab.label}
          </a>
        );
      })}
    </div>
  );
}
