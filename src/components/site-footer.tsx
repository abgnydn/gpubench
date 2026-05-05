import { AUTHOR, LINKS, SITES } from "@/lib/constants";

export function SiteFooter({ githubRepo }: { githubRepo?: string }) {
  const repoUrl = githubRepo ?? LINKS.repo;
  return (
    <footer className="border-t border-bench-border/50 mt-24 py-10 text-sm">
      <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row md:items-start justify-between gap-8">
        <div className="md:max-w-md">
          <div className="text-bench-text font-medium">
            Built by{" "}
            <a
              href={SITES.barisgunaydin.url}
              className="text-bench-accent hover:underline"
            >
              {AUTHOR.name}
            </a>
          </div>
          <div className="text-bench-muted mt-1 leading-relaxed">
            Independent researcher · browser-native GPU computing.
            Two preprints, one open-source SDK, public benchmark database.
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <a
            href={repoUrl}
            className="inline-flex items-center gap-2 text-bench-text hover:text-bench-accent transition-colors"
          >
            <span className="text-bench-yellow">★</span> Star this repo on GitHub
          </a>
          <a
            href={AUTHOR.github}
            className="text-bench-muted hover:text-bench-accent transition-colors"
          >
            github.com/abgnydn — all my projects
          </a>
          <a
            href={AUTHOR.linkedin}
            className="text-bench-muted hover:text-bench-accent transition-colors"
          >
            Follow my research on LinkedIn
          </a>
          <a
            href={SITES.barisgunaydin.url}
            className="text-bench-muted hover:text-bench-accent transition-colors"
          >
            barisgunaydin.com
          </a>
        </div>
      </div>
      <div className="max-w-6xl mx-auto px-6 mt-8 pt-6 border-t border-bench-border/30 text-xs text-bench-muted/70 flex flex-wrap gap-x-4 gap-y-1">
        <span>© {new Date().getFullYear()} Ahmet Baris Gunaydin · MIT</span>
        <span>·</span>
        <span>
          More work:{" "}
          <a
            href={SITES.kernelfusion.url}
            className="hover:text-bench-accent"
          >
            kernelfusion.dev
          </a>{" "}·{" "}
          <a href={SITES.zerotvm.url} className="hover:text-bench-accent">
            zerotvm.com
          </a>{" "}·{" "}
          <a href={SITES.neuropulse.url} className="hover:text-bench-accent">
            neuropulse.live
          </a>{" "}·{" "}
          <a href={SITES.webgpudna.url} className="hover:text-bench-accent">
            webgpudna.com
          </a>
        </span>
      </div>
    </footer>
  );
}
