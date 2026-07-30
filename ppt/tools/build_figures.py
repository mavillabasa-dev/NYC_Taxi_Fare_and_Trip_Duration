"""Build the data-driven SVG figures used in the pre-demo deck.

Every number below was measured from dataset/yellow_tripdata_2022-05.parquet and
dataset/taxi_zones/taxi_zones.shp -- see `measure.py` in this folder for the
script that produced them. Nothing here is illustrative or invented.

Usage (from the repo root, with the project venv active):

    python ppt/tools/build_figures.py

Writes standalone assets into ppt/assets/ and fragment files into
ppt/tools/_fragments/ which are inlined into ppt/index.html.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "ppt" / "assets"
FRAG = Path(__file__).resolve().parent / "_fragments"
ASSETS.mkdir(parents=True, exist_ok=True)
FRAG.mkdir(parents=True, exist_ok=True)

# --- palette (validated: see ppt/README.md) ---------------------------------
INK = "#0B0A0C"
GOLD = "#F0BE4A"
GOLD_DEEP = "#B98C2C"
GOLD_DIM = "#8A6821"
RULE = "#332F36"
TEXT2 = "#A9A294"
TEXT3 = "#7C7568"

# --- measured data ----------------------------------------------------------
# Average yellow-taxi pickups per hour-of-day, over the 31 days of May 2022.
TRIPS_BY_HOUR = [3286, 2121, 1376, 894, 637, 783, 1894, 3420, 4369, 4935, 5495,
                 6084, 6515, 6568, 7011, 7157, 7128, 7675, 8171, 7451, 6338,
                 6156, 5653, 4632]

# fare_amount histogram, 40 bins over $0-80, heights normalised to the mode.
FARE_RAW = [0.0008, 0.0908, 0.668, 1.0, 0.9387, 0.7315, 0.5281, 0.3796, 0.2747,
            0.2026, 0.1569, 0.1182, 0.0961, 0.0803, 0.0718, 0.0657, 0.0589,
            0.0504, 0.0396, 0.031, 0.0265, 0.0208, 0.0186, 0.0131, 0.0107,
            0.0106, 0.255, 0.008, 0.0066, 0.0059, 0.0063, 0.006, 0.0106,
            0.0062, 0.0067, 0.005, 0.0031, 0.0039, 0.0019, 0.0031]

# log1p(fare_amount) histogram, 40 bins over the transformed range.
FARE_LOG = [0.0008, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0589, 0.0621,
            0.1295, 0.461, 0.2999, 0.6612, 1.0, 0.9142, 0.7642, 0.5989, 0.5961,
            0.5174, 0.3531, 0.3285, 0.2206, 0.1932, 0.1739, 0.1282, 0.078,
            0.3767, 0.0373, 0.0436, 0.0186, 0.0071, 0.0049, 0.0035, 0.0021,
            0.0017, 0.001, 0.0007, 0.0004]


def col(x, y, w, h, baseline, r=4):
    """A column with a rounded data-end and a square foot on the baseline."""
    r = min(r, w / 2, h) if h > 0 else 0
    return (f"M{x:.1f},{baseline:.1f} V{y + r:.1f} "
            f"Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{baseline:.1f} Z")


# ===========================================================================
# FIGURE 1 -- Pickups by hour of day.
# One series, so no legend box: the title names it. Emphasis pattern -- the
# peak and trough columns carry the bright step, the rest are context.
# ===========================================================================
def fig_hourly():
    W, H = 1180, 380
    left, right, top, base = 92, 24, 46, 300
    plot_w = W - left - right
    band = plot_w / 24
    bar_w = 24                                  # capped per mark spec
    peak = max(range(24), key=lambda i: TRIPS_BY_HOUR[i])
    trough = min(range(24), key=lambda i: TRIPS_BY_HOUR[i])
    ymax = 9000

    def ypos(v):
        return base - (v / ymax) * (base - top)

    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'class="fig" role="img" aria-label="Average yellow-taxi pickups by '
           f'hour of day in May 2022. The peak is 8,171 pickups at 6 PM; the '
           f'low is 637 at 4 AM.">']

    # recessive gridlines + y ticks
    for v in (0, 3000, 6000, 9000):
        y = ypos(v)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W - right}" y2="{y:.1f}" '
                   f'stroke="{RULE}" stroke-width="1"/>')
        out.append(f'<text x="{left - 16}" y="{y + 7:.1f}" text-anchor="end" '
                   f'class="fig-tick">{v:,}</text>')

    # columns
    for i, v in enumerate(TRIPS_BY_HOUR):
        x = left + i * band + (band - bar_w) / 2
        y = ypos(v)
        fill = GOLD if i in (peak, trough) else GOLD_DEEP
        out.append(f'<path d="{col(x, y, bar_w, base - y, base)}" fill="{fill}"/>')

    # selective direct labels only -- never a number on every column
    px = left + peak * band + band / 2
    out.append(f'<text x="{px:.1f}" y="{ypos(TRIPS_BY_HOUR[peak]) - 26:.1f}" '
               f'text-anchor="middle" class="fig-val">8,171</text>')
    out.append(f'<text x="{px:.1f}" y="{ypos(TRIPS_BY_HOUR[peak]) - 8:.1f}" '
               f'text-anchor="middle" class="fig-note">6 PM peak</text>')
    tx = left + trough * band + band / 2
    out.append(f'<text x="{tx:.1f}" y="{ypos(TRIPS_BY_HOUR[trough]) - 26:.1f}" '
               f'text-anchor="middle" class="fig-val">637</text>')
    out.append(f'<text x="{tx:.1f}" y="{ypos(TRIPS_BY_HOUR[trough]) - 8:.1f}" '
               f'text-anchor="middle" class="fig-note">4 AM low</text>')

    # x axis
    out.append(f'<line x1="{left}" y1="{base}" x2="{W - right}" y2="{base}" '
               f'stroke="{RULE}" stroke-width="1"/>')
    for i in (0, 6, 12, 18, 23):
        x = left + i * band + band / 2
        out.append(f'<text x="{x:.1f}" y="{base + 30}" text-anchor="middle" '
                   f'class="fig-tick">{i:02d}</text>')
    out.append(f'<text x="{left + plot_w / 2:.1f}" y="{base + 62}" '
               f'text-anchor="middle" class="fig-axis">HOUR OF DAY</text>')
    out.append("</svg>")
    return "".join(out)


# ===========================================================================
# FIGURE 2 -- fare distribution before and after log1p.
# Before -> after on the same measure: one hue, two shades. Two series, so a
# legend is present; the shapes are also directly labelled.
# ===========================================================================
def fig_skew():
    W, H = 1180, 330
    top, base = 40, 232
    gap = 96
    panel_w = (W - gap) / 2

    def area(vals, x0, colour, opacity):
        n = len(vals)
        step = panel_w / (n - 1)
        pts = [(x0 + i * step, base - v * (base - top)) for i, v in enumerate(vals)]
        line = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        fill = (line + f"L{pts[-1][0]:.1f},{base} L{pts[0][0]:.1f},{base} Z")
        return (f'<path d="{fill}" fill="{colour}" fill-opacity="{opacity}"/>'
                f'<path d="{line}" fill="none" stroke="{colour}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')

    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'class="fig" role="img" aria-label="Fare distribution before and '
           f'after a log1p transform. Raw fare is a sharp spike near $10 with a '
           f'tail reaching $6,966 and a skew of 31.6. After log1p the shape is '
           f'a broad symmetric hump with a skew of 0.8.">']

    # --- left panel: raw dollars
    out.append(area(FARE_RAW, 0, GOLD_DIM, 0.18))
    out.append(f'<line x1="0" y1="{base}" x2="{panel_w}" y2="{base}" '
               f'stroke="{RULE}" stroke-width="1"/>')
    out.append(f'<text x="0" y="{base + 30}" class="fig-tick">$0</text>')
    out.append(f'<text x="{panel_w / 2:.0f}" y="{base + 30}" text-anchor="middle" '
               f'class="fig-tick">$40</text>')
    out.append(f'<text x="{panel_w:.0f}" y="{base + 30}" text-anchor="end" '
               f'class="fig-tick">$80</text>')
    # the JFK flat-fare spike is a real artefact worth naming
    jfk_x = 26 / (len(FARE_RAW) - 1) * panel_w
    out.append(f'<line x1="{jfk_x:.1f}" y1="{base - 0.255 * (base - top) - 8:.1f}" '
               f'x2="{jfk_x:.1f}" y2="{top + 6}" stroke="{TEXT3}" stroke-width="1"/>')
    out.append(f'<text x="{jfk_x + 10:.1f}" y="{top + 14}" class="fig-note">'
               f'JFK $52 flat fare</text>')
    out.append(f'<text x="{panel_w:.0f}" y="{base - 14}" text-anchor="end" '
               f'class="fig-note">tail runs to $6,966 &#8594;</text>')

    # --- right panel: log space
    x0 = panel_w + gap
    out.append(area(FARE_LOG, x0, GOLD, 0.16))
    out.append(f'<line x1="{x0}" y1="{base}" x2="{W}" y2="{base}" '
               f'stroke="{RULE}" stroke-width="1"/>')
    out.append(f'<text x="{x0}" y="{base + 30}" class="fig-tick">log1p</text>')
    out.append(f'<text x="{W}" y="{base + 30}" text-anchor="end" '
               f'class="fig-tick">8.9</text>')

    # legend -- identity never rests on colour alone
    out.append(f'<rect x="0" y="{H - 34}" width="26" height="4" rx="2" fill="{GOLD_DIM}"/>')
    out.append(f'<text x="36" y="{H - 25}" class="fig-legend">fare_amount &#183; skew 31.6</text>')
    out.append(f'<rect x="{x0}" y="{H - 34}" width="26" height="4" rx="2" fill="{GOLD}"/>')
    out.append(f'<text x="{x0 + 36}" y="{H - 25}" class="fig-legend">'
               f'log1p(fare_amount) &#183; skew 0.8</text>')
    out.append("</svg>")
    return "".join(out)


# ===========================================================================
# MOTIF -- Manhattan skyline silhouette for the title and closing slides.
# Authored, not random: an explicit roof-height profile with four spires, so
# it renders identically every build.
# ===========================================================================
# (width, roof height above street) repeated across 1920px. Spires are marked.
SKYLINE = [
    (46, 50), (46, 68), (28, 38), (48, 95), (28, 78), (42, 120), (24, 56),
    (38, 104), (18, 150, "spire"), (34, 86), (44, 128), (28, 64), (46, 142),
    (30, 100), (34, 60), (46, 114), (24, 76), (48, 160, "spire"), (34, 98),
    (38, 132), (36, 66), (40, 110), (26, 54), (48, 152), (32, 92), (42, 124),
    (32, 60), (44, 138), (28, 74), (46, 168, "spire"), (34, 102), (44, 134),
    (32, 68), (44, 116), (28, 56), (46, 146), (34, 88), (44, 126), (32, 62),
    (44, 140), (32, 94), (46, 158, "spire"), (34, 80), (44, 122), (30, 52),
    (46, 128), (32, 90), (44, 112), (32, 72), (46, 134),
]


def build_skyline():
    H, GROUND = 220, 200
    pts, x = ["M0,%d" % H, "L0,%d" % GROUND], 0
    spires = []
    for b in SKYLINE:
        w, h = b[0], b[1]
        roof = GROUND - h
        pts.append(f"L{x},{roof}L{x + w},{roof}")
        if len(b) > 2:                       # a spire rises from the roof centre
            cx = x + w / 2
            spires.append(f'<path d="M{cx - 3:.0f},{roof}L{cx:.0f},{roof - 46}'
                          f'L{cx + 3:.0f},{roof}Z"/>')
        x += w
    pts.append(f"L{x},{GROUND}L{x},{H}Z")
    total = x

    # A few lit windows, placed on a fixed lattice so they never land off-building.
    windows = []
    for i in range(0, total, 34):
        for j in range(GROUND - 26, GROUND - 96, -26):
            if (i // 34 + j // 26) % 5 == 0:
                windows.append(f'<rect x="{i + 12}" y="{j}" width="4" height="7"/>')

    return (f'<svg viewBox="0 0 {total} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'preserveAspectRatio="xMidYMax slice" class="skyline" aria-hidden="true">'
            f'<g fill="currentColor"><path d="{"".join(pts)}"/>{"".join(spires)}</g>'
            f'<g fill="#0B0A0C" opacity=".55">{"".join(windows)}</g></svg>')


def main():
    figures = {"fig-hourly.svg": fig_hourly(), "fig-skew.svg": fig_skew(),
               "motif-skyline.svg": build_skyline()}
    css = (f'<style>.fig-tick,.fig-axis,.fig-note,.fig-legend,.fig-val{{'
           f'font-family:"Supreme",Arial,sans-serif}}'
           f'.fig-tick{{font-size:19px;fill:{TEXT3}}}'
           f'.fig-axis{{font-size:16px;fill:{TEXT3};letter-spacing:.22em}}'
           f'.fig-note{{font-size:19px;fill:{TEXT2}}}'
           f'.fig-legend{{font-size:21px;fill:{TEXT2}}}'
           f'.fig-val{{font-size:30px;font-weight:700;fill:#F2ECE0}}</style>')

    for name, svg in figures.items():
        FRAG.joinpath(name).write_text(svg, encoding="utf-8")
        # The standalone copies have to stand on their own: the deck supplies the
        # surface and `currentColor`, a bare .svg file does not.
        standalone = svg.replace("<svg ", f'<svg style="background:{INK};color:{GOLD}" ', 1)
        standalone = standalone.replace(">", ">" + css, 1)
        ASSETS.joinpath(name).write_text(standalone, encoding="utf-8")
        print(f"wrote {name}  ({len(svg):,} bytes)")


if __name__ == "__main__":
    main()
