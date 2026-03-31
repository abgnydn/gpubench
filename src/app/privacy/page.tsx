export default function PrivacyPage() {
  return (
    <div className="min-h-screen">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.03)_0%,_transparent_50%)]" />
      </div>

      <div className="max-w-2xl mx-auto px-6 py-20">
        {/* Back link */}
        <a href="/" className="inline-flex items-center gap-1.5 text-sm text-bench-muted hover:text-bench-accent transition mb-12 group">
          <svg className="w-4 h-4 transition group-hover:-translate-x-0.5" viewBox="0 0 16 16" fill="none">
            <path d="M10 4l-4 4 4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to benchmarks
        </a>

        {/* Header */}
        <div className="mb-12">
          <h1 className="text-3xl font-bold tracking-tight mb-3">Privacy Policy</h1>
          <p className="text-bench-muted">
            Short version: we collect anonymous GPU stats. Nothing personal. That&apos;s it.
          </p>
        </div>

        {/* Sections */}
        <div className="space-y-10">

          <section className="card">
            <div className="flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-bench-accent/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-bench-accent" viewBox="0 0 16 16" fill="none">
                  <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.5 3.5l1.4 1.4M11.1 11.1l1.4 1.4M3.5 12.5l1.4-1.4M11.1 4.9l1.4-1.4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div>
                <h2 className="text-base font-semibold text-bench-text mb-2">What we collect</h2>
                <p className="text-sm text-bench-muted leading-relaxed mb-3">
                  When you click &ldquo;Run All Benchmarks,&rdquo; we save:
                </p>
                <div className="space-y-2">
                  {[
                    "GPU adapter name, vendor, and architecture",
                    "WebGPU device limits (buffer size, workgroup size)",
                    "Benchmark throughput results (gen/s, timing stats)",
                    "Browser and OS (from user agent)",
                  ].map((item) => (
                    <div key={item} className="flex items-start gap-2 text-sm text-bench-muted">
                      <svg className="w-3.5 h-3.5 text-bench-accent mt-0.5 flex-shrink-0" viewBox="0 0 14 14" fill="none">
                        <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5"/>
                        <circle cx="7" cy="7" r="2" fill="currentColor"/>
                      </svg>
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section className="card border-bench-green/20">
            <div className="flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-bench-green/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-bench-green" viewBox="0 0 16 16" fill="none">
                  <path d="M8 1L2 4v4.5c0 3.5 2.5 6.5 6 7.5 3.5-1 6-4 6-7.5V4L8 1z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
                  <path d="M5.5 8l2 2 3.5-3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div>
                <h2 className="text-base font-semibold text-bench-text mb-2">What we do NOT collect</h2>
                <div className="space-y-2">
                  {[
                    "Names, emails, or any personal identifiers",
                    "IP addresses (not logged)",
                    "Cookies or tracking pixels",
                    "Location data",
                    "Browsing history or data from other tabs",
                  ].map((item) => (
                    <div key={item} className="flex items-start gap-2 text-sm text-bench-muted">
                      <svg className="w-3.5 h-3.5 text-bench-green mt-0.5 flex-shrink-0" viewBox="0 0 14 14" fill="none">
                        <path d="M3 7l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section className="card">
            <div className="flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-purple-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-purple-400" viewBox="0 0 16 16" fill="none">
                  <path d="M2 4h12M2 8h12M2 12h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div>
                <h2 className="text-base font-semibold text-bench-text mb-2">How we use the data</h2>
                <p className="text-sm text-bench-muted leading-relaxed">
                  Collected data builds an aggregate picture of WebGPU compute performance across
                  real-world hardware. We may publish aggregate statistics (e.g. &ldquo;median
                  throughput on M2 Macs is X gen/s&rdquo;) in research papers or public reports.
                  Individual benchmark submissions are never published or shared.
                </p>
              </div>
            </div>
          </section>

          <section className="card">
            <div className="flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-bench-accent/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-bench-accent" viewBox="0 0 16 16" fill="none">
                  <rect x="2" y="2" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                  <path d="M5 8h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div>
                <h2 className="text-base font-semibold text-bench-text mb-2">All computation is local</h2>
                <p className="text-sm text-bench-muted leading-relaxed">
                  The benchmark shaders run entirely on your GPU inside your browser. No computation
                  is offloaded to our servers. Only the final results (throughput numbers and hardware
                  info) are transmitted after the benchmarks complete.
                </p>
              </div>
            </div>
          </section>

          <section className="card">
            <div className="flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-bench-muted/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-bench-muted" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.5"/>
                  <path d="M8 5v3l2 1.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div>
                <h2 className="text-base font-semibold text-bench-text mb-2">Data retention</h2>
                <p className="text-sm text-bench-muted leading-relaxed">
                  Benchmark results are stored indefinitely as anonymous records. Because no personal
                  information is attached, individual records cannot be identified or deleted.
                </p>
              </div>
            </div>
          </section>

        </div>

        {/* Footer */}
        <div className="mt-16 pt-8 border-t border-bench-border/50 flex items-center justify-between">
          <p className="text-xs text-bench-muted/50">Last updated: March 2026</p>
          <a href="https://github.com/abgnydn/webgpu-kernel-fusion" className="text-xs text-bench-accent/60 hover:text-bench-accent transition">
            Questions? GitHub &rarr;
          </a>
        </div>
      </div>
    </div>
  );
}
