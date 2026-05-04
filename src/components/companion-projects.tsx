import { SITES, CROSSLINKS, type SiteInfo } from "@/lib/sites";

const FLAGSHIP: SiteInfo = SITES[CROSSLINKS.gpubench[0]!];
const ADJACENT: SiteInfo[] = CROSSLINKS.gpubench.slice(1).map((k) => SITES[k]);

const CATEGORY_BADGE: Record<string, string> = {
  Theory: "bg-bench-accent/10 text-bench-accent",
  Radiobiology: "bg-bench-green/10 text-bench-green",
  "LLM inference": "bg-bench-yellow/10 text-bench-yellow",
  Visualization: "bg-bench-red/10 text-bench-red",
  Quantum: "bg-bench-yellow/10 text-bench-yellow",
  Benchmarks: "bg-bench-accent/10 text-bench-accent",
  Personal: "bg-bench-muted/10 text-bench-muted",
  Utility: "bg-bench-muted/10 text-bench-muted",
  Tooling: "bg-bench-muted/10 text-bench-muted",
};

interface Props {
  /** Smaller body copy + tighter grid for embed inside dense pages. */
  compact?: boolean;
}

export function CompanionProjects({ compact = false }: Props) {
  return (
    <section className={compact ? "pt-2" : "pt-4"}>
      <div className="flex items-center gap-4 mb-4">
        <div className="flex-1 h-px bg-bench-border" />
        <span className="text-xs text-bench-muted font-medium uppercase tracking-widest">
          Companion projects
        </span>
        <div className="flex-1 h-px bg-bench-border" />
      </div>
      {!compact && (
        <p className="text-xs text-bench-muted leading-relaxed mb-4 text-center max-w-md mx-auto">
          The research line and the end-to-end projects that build on it.
        </p>
      )}

      {/* Flagship hero */}
      <a
        href={FLAGSHIP.url}
        target="_blank"
        rel="noopener"
        className="card block transition hover:border-bench-accent/50 ring-1 ring-bench-accent/20 mb-3"
      >
        <div className="flex items-start gap-3 mb-2">
          <span
            className={`text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full inline-block ${
              CATEGORY_BADGE[FLAGSHIP.category] ?? "bg-bench-accent/10 text-bench-accent"
            }`}
          >
            {FLAGSHIP.category}
          </span>
          <span className="text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full bg-bench-accent/10 text-bench-accent inline-block">
            Research line
          </span>
        </div>
        <h3 className="font-semibold text-bench-text text-base mb-1">{FLAGSHIP.domain}</h3>
        <p className="text-xs text-bench-muted leading-relaxed mb-3">{FLAGSHIP.shortDesc}</p>
        {FLAGSHIP.stats && (
          <div className="grid grid-cols-3 gap-2 mb-3">
            {FLAGSHIP.stats.map((s) => (
              <div key={s.label} className="bg-bench-bg/40 rounded px-2 py-1.5">
                <div className="text-sm font-semibold text-bench-text">{s.value}</div>
                <div className="text-[10px] text-bench-muted">{s.label}</div>
              </div>
            ))}
          </div>
        )}
        <span className="text-xs font-medium text-bench-accent">
          {FLAGSHIP.cta ?? "Visit site →"}
        </span>
      </a>

      {/* Adjacent siblings — wraps to 1/2/3 cols by viewport */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {ADJACENT.map((site) => {
          const badge =
            CATEGORY_BADGE[site.category] ?? "bg-bench-accent/10 text-bench-accent";
          return (
            <a
              key={site.key}
              href={site.url}
              target="_blank"
              rel="noopener"
              className="card block transition hover:border-bench-accent/30"
            >
              <span
                className={`text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full mb-2 inline-block ${badge}`}
              >
                {site.category}
              </span>
              <h3 className="font-semibold text-bench-text text-sm mb-1">{site.domain}</h3>
              <p className="text-xs text-bench-muted leading-relaxed mb-2">{site.shortDesc}</p>
              <span className="text-xs font-medium text-bench-accent">
                {site.cta ?? "Visit →"}
              </span>
            </a>
          );
        })}
      </div>
    </section>
  );
}
