"use client";

import { useEffect, useState } from "react";

interface ShareButtonsProps {
  text: string;
  url: string;
  title?: string;
  className?: string;
}

export function ShareButtons({ text, url, title, className = "" }: ShareButtonsProps) {
  const [open, setOpen] = useState(false);
  const [editedText, setEditedText] = useState(text);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (open) setEditedText(text);
  }, [open, text]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open]);

  const encodedText = encodeURIComponent(editedText);
  const encodedUrl = encodeURIComponent(url);
  const encodedTitle = encodeURIComponent(title ?? editedText);

  const twitterUrl = `https://twitter.com/intent/tweet?text=${encodedText}&url=${encodedUrl}`;
  const linkedinUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`;
  const hnUrl = `https://news.ycombinator.com/submitlink?u=${encodedUrl}&t=${encodedTitle}`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(`${editedText}\n${url}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };

  const platformBtn =
    "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition border border-bench-border bg-bench-surface hover:bg-bench-accent/10 hover:border-bench-accent/40 hover:text-bench-accent text-bench-text";

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={`inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition border border-bench-accent/40 bg-bench-accent/10 hover:bg-bench-accent/20 text-bench-accent ${className}`}
        aria-label="Share"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="18" cy="5" r="3" />
          <circle cx="6" cy="12" r="3" />
          <circle cx="18" cy="19" r="3" />
          <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
          <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
        </svg>
        Share results
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={() => setOpen(false)}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

          {/* Modal */}
          <div
            className="relative w-full max-w-lg bg-bench-surface border border-bench-border rounded-xl shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-bench-border/50">
              <h3 className="text-sm font-semibold text-bench-text">Share these results</h3>
              <button
                onClick={() => setOpen(false)}
                className="text-bench-muted hover:text-bench-text transition"
                aria-label="Close"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* Preview */}
            <div className="px-5 py-4 space-y-3">
              <p className="text-[10px] text-bench-muted uppercase tracking-widest">Preview &middot; editable</p>
              <div className="rounded-lg border border-bench-border bg-bench-bg/50 p-4">
                <textarea
                  value={editedText}
                  onChange={(e) => setEditedText(e.target.value)}
                  rows={Math.min(12, Math.max(4, editedText.split("\n").length))}
                  className="w-full bg-transparent text-xs text-bench-text leading-relaxed resize-none focus:outline-none font-mono whitespace-pre"
                />
                <div className="mt-2 pt-2 border-t border-bench-border/30 flex items-center justify-between">
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-bench-accent hover:underline truncate max-w-[70%]"
                  >
                    {url}
                  </a>
                  <span className="text-[10px] text-bench-muted/60 tabular-nums">
                    {editedText.length} chars
                  </span>
                </div>
              </div>
            </div>

            {/* Platforms */}
            <div className="px-5 pb-5 space-y-3">
              <p className="text-[10px] text-bench-muted uppercase tracking-widest">Share to</p>
              <div className="grid grid-cols-2 gap-2">
                <a href={twitterUrl} target="_blank" rel="noopener noreferrer" className={platformBtn}>
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                  Post on X
                </a>
                <a href={linkedinUrl} target="_blank" rel="noopener noreferrer" className={platformBtn}>
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.063 2.063 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                  </svg>
                  LinkedIn
                </a>
                <a href={hnUrl} target="_blank" rel="noopener noreferrer" className={platformBtn}>
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M0 24V0h24v24H0zM6.951 5.896l4.112 7.708v5.064h1.583v-4.972l4.148-7.799h-1.749l-2.457 4.875c-.372.745-.688 1.434-.688 1.434s-.297-.708-.651-1.434L8.831 5.896h-1.88z" />
                  </svg>
                  Hacker News
                </a>
                <button onClick={handleCopy} className={platformBtn}>
                  {copied ? (
                    <>
                      <svg className="w-4 h-4 text-bench-green" viewBox="0 0 16 16" fill="none">
                        <path d="M3 8l3 3 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none">
                        <rect x="5" y="5" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
                        <path d="M3 11V3a2 2 0 012-2h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                      </svg>
                      Copy text
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
