"""Render docs/REPORT.md to docs/REPORT.html and (if Google Chrome is installed) docs/REPORT.pdf.

Usage:
    python scripts/build_report.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #14213D; line-height: 1.5;
       font-size: 10.5pt; max-width: 900px; margin: 0 auto; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #d8dde6; }
h3 { font-size: 11.5pt; margin-top: 12pt; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; page-break-inside: avoid; }
td, th { border: 1px solid #d8dde6; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef2f8; }
img { max-width: 100%; display: block; margin: 8pt auto; page-break-inside: avoid; }
pre, code { font-family: Menlo, Consolas, monospace; font-size: 8.5pt; }
pre { background: #f4f6fa; padding: 8pt; overflow-x: auto; white-space: pre; }
hr { border: none; border-top: 1px solid #d8dde6; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always;
         border-left: 6pt solid #2A78D6; padding-left: 18mm; }
.cover-kicker { font-size: 11pt; letter-spacing: 0.08em; text-transform: uppercase; color: #4A5568; margin: 0; }
.cover-title { font-size: 28pt; line-height: 1.2; margin: 10pt 0 12pt; }
.cover-sub { font-size: 12.5pt; color: #4A5568; margin: 0 0 40pt; max-width: 140mm; }
.cover-label { font-size: 10pt; letter-spacing: 0.08em; text-transform: uppercase; color: #4A5568; margin: 18pt 0 4pt; }
.cover-value { font-size: 16pt; font-weight: 600; margin: 0; }
.cover-members { font-size: 14pt; line-height: 1.8; margin: 0; padding-left: 20pt; }
"""


def main() -> int:
    md = (DOCS / "REPORT.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ExoScope — Group L3 Project Report</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = DOCS / "REPORT.html"
    out_html.write_text(html)
    print("wrote", out_html.relative_to(ROOT))
    chrome = next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        print("Google Chrome not found: open docs/REPORT.html and use Print → Save as PDF.")
        return 0
    out_pdf = DOCS / "REPORT.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print("wrote", out_pdf.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
