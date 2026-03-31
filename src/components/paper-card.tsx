"use client";

interface PaperCardProps {
  title: string;
  description: string;
  doi: string;
  doiLabel: string;
}

export function PaperCard({ title, description, doi, doiLabel }: PaperCardProps) {
  return (
    <div className="pt-8">
      <div className="flex items-center gap-4 pb-6">
        <div className="flex-1 h-px bg-bench-border" />
        <span className="text-xs text-bench-muted font-medium uppercase tracking-widest">Research</span>
        <div className="flex-1 h-px bg-bench-border" />
      </div>

      <section className="card">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-bench-accent/10 flex items-center justify-center">
            <svg className="w-5 h-5 text-bench-accent" viewBox="0 0 20 20" fill="none">
              <path d="M4 2h8l4 4v12a2 2 0 01-2 2H4a2 2 0 01-2-2V4a2 2 0 012-2z" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M12 2v4h4M6 10h8M6 14h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-semibold mb-2">The science behind the benchmarks</h2>
            <p className="text-sm text-bench-muted leading-relaxed mb-4">{description}</p>
            <div className="bg-bench-bg rounded-lg p-4 font-mono text-xs text-bench-muted/80 border border-bench-border/50">
              <p className="text-bench-text/80">Gunaydin, A.B. (2026)</p>
              <p>{title}</p>
              <a href={doi} className="text-bench-accent hover:underline">{doiLabel}</a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
