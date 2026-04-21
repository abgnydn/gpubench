"use client";

import { usePathname } from "next/navigation";

/**
 * Single source of truth for the primary nav tabs.
 *
 * The active tab is derived from the route via usePathname() — pages
 * don't pass an `active` prop, so they can't drift.
 *
 * Note on the Zero-TVM entry: it links to a static HTML file served from
 * /public/demos/, not a Next route. That's fine — it will simply never
 * match usePathname() (which only sees Next pages), so it always renders
 * as a non-active link, which is correct: when a user is on the chat
 * page they aren't in the Next app anyway.
 */
const TABS = [
  { label: "GPU Compute", href: "/" },
  { label: "Transformer Fusion", href: "/transformer" },
  { label: "Distributed P2P", href: "/swarm" },
  { label: "Zero-TVM", href: "/demos/zerotvm-chat.html" },
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
