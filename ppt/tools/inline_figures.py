"""Inline the generated SVG fragments into ppt/index.html.

Run this after build_figures.py / build_zonemap.py. It is idempotent: each
placeholder comment is left in place next to the content it guards, so the
script can be re-run after a figure is regenerated.

    python ppt/tools/build_zonemap.py
    python ppt/tools/build_figures.py
    python ppt/tools/inline_figures.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "ppt" / "index.html"
FRAG = Path(__file__).resolve().parent / "_fragments"

SLOTS = {
    "SKYLINE": "motif-skyline.svg",
    "FIG_HOURLY": "fig-hourly.svg",
    "FIG_SKEW": "fig-skew.svg",
    "ZONES": "zones-paths.svg",
}

html = DECK.read_text(encoding="utf-8")

for token, fname in SLOTS.items():
    body = FRAG.joinpath(fname).read_text(encoding="utf-8")
    block = f"<!--{token}-->{body}<!--/{token}-->"
    # Match either the bare placeholder or an already-filled block, so re-runs
    # replace the previous content instead of stacking copies.
    pattern = re.compile(rf"<!--{token}-->(?:.*?<!--/{token}-->)?", re.S)
    html, n = pattern.subn(lambda _m: block, html)
    print(f"{token:<12} <- {fname:<20} ({len(body):>6,} bytes)  x{n}")

DECK.write_text(html, encoding="utf-8")
print(f"\nwrote {DECK.relative_to(ROOT)}  ({len(html):,} bytes)")
