# Pre-Demo Presentation

The deck we show **before** the final demo: the problem, the dataset, and the
proposed solution phase by phase.

```
ppt/
├── index.html              the presentation — one self-contained file
├── script.md               speaker script, timed, one section per presenter
├── assets/                 standalone copies of the generated graphics
│   ├── zones-map.svg       263 TLC taxi zones, rendered from the shapefile
│   ├── fig-hourly.svg      pickups by hour of day, May 2022
│   ├── fig-skew.svg        fare distribution before / after log1p
│   └── motif-skyline.svg   skyline silhouette used on the title and closing
└── tools/                  the scripts that produced the numbers and graphics
    ├── measure.py          re-derives every dataset figure quoted on a slide
    ├── build_figures.py    builds the chart SVGs + the skyline motif
    ├── build_zonemap.py    renders taxi_zones.shp into an SVG map
    ├── inline_figures.py   inlines the fragments into index.html
    └── _fragments/         generated SVG fragments (build output)
```

## Presenting

Open `index.html` in any browser — no server, no build step, no install.

| Key | Action |
|-----|--------|
| `→` `↓` `Space` `PgDn` | next slide |
| `←` `↑` `PgUp` | previous slide |
| `Home` / `End` | first / last slide |
| `F` | fullscreen |
| `E` | inline edit mode — click any text, `Ctrl+S` to save |

Swipe and scroll also work. `index.html#14` opens straight at slide 14, which is
handy when rehearsing one section. The current slide is remembered across a
reload, so an accidental refresh mid-talk does not send you back to slide 1.

**Fonts load from a CDN.** Everything else is inline. Without internet the deck
still works and still looks intentional — it falls back to a serif/grotesk pair —
but if the venue is offline and you want it pixel-identical, open it once on the
presenting machine while online so the fonts land in the browser cache.

## The numbers are measured, not illustrative

Every dataset figure on a slide comes from the real files in `dataset/`. To check
the deck against the data:

```powershell
python ppt/tools/measure.py
```

Each block of output is labelled with the slide it belongs to. Highlights:

| Slide | Claim | Source |
|-------|-------|--------|
| 2, 3 | 115,747 rides/day; peak 8,171 at 18:00 vs 637 at 04:00 | `yellow_tripdata_2022-05.parquet` |
| 9 | 3,588,295 rows × 19 columns, 524 MB; median fare $10.50; max $6,966.50 | same |
| 10 | 259 pickup zones used; JFK is the busiest at 175,937 | same + `taxi_zone_lookup.csv` |
| 15 | fare skew 31.6 → 0.8 under `log1p`; 7,267,627 mph worst record | same |
| 15 | `corr(trip_distance, fare)` 0.01 raw → 0.95 cleaned | same |
| 16 | train 2,847,337 (79.4%) / test 740,815 (20.6%) | same |

Two figures are **external**, not from our data, and are attributed on the slide:
~13,600 licensed medallions and ~$1.6 B/year in fares (2019, pre-pandemic), plus
the TLC's upfront-pricing rule change on slide 4.

The `log1p` skew is quoted to one decimal on purpose: `measure.py` and
`notebooks/01_EDA.ipynb` use slightly different filters for non-positive fares
and agree at that precision, not beyond it.

### The map is the real shapefile

`assets/zones-map.svg` is `dataset/taxi_zones/taxi_zones.shp`, simplified and
projected to SVG — the same geometry T-116 caches centroids from and T-111 will
draw its choropleth with. Boroughs step down one gold ramp with Manhattan
brightest, because that is where the pickups are.

## Rebuilding the graphics

Only needed if the dataset is re-downloaded or a figure changes:

```powershell
python ppt/tools/build_zonemap.py     # shapefile -> SVG map
python ppt/tools/build_figures.py     # charts + skyline motif
python ppt/tools/inline_figures.py    # inline them into index.html
```

`inline_figures.py` is idempotent — it replaces the previous block rather than
appending, so it is safe to re-run. Hand edits to `index.html` outside the
`<!--FIG_*-->` markers are preserved.

## Design notes

Built on the [Frontend Slides](https://github.com/zarazhangrui/frontend-slides)
skill: a fixed **1920×1080** stage scaled by a single transform, so slides stay
16:9 on any screen and never reflow. `viewport-base.css` is included verbatim in
the `<style>` block, as that skill requires.

**Palette — black and gold.** Slide surface `#0B0A0C`, accent `#F0BE4A`, taxi
yellow `#FFC72C` for highlights. Every value was contrast-checked against the
surface: all chart marks clear 3:1, all body text clears WCAG AA (cream 16.8:1,
secondary 7.8:1, muted 4.3:1 at large sizes). Gridlines sit at 1.5:1 by design —
they are meant to recede behind the data.

**Type.** Zodiak for display, Supreme for body and every figure, JetBrains Mono
for column names and ticket IDs. Big numbers stay in the sans, never the serif.

**Charts.** All single-series, so they use one gold hue rather than a categorical
palette — colour never re-encodes what bar length already shows. The hour-of-day
chart uses the emphasis pattern: the peak and the trough carry the bright step,
the other 22 hours are context. Labels are selective, never one per bar.

**The taxi checker** is the recurring motif — under every heading, on the
dividers, and as the title separator. Phase numbers sit in a medallion, which is
what a NYC taxi licence is actually called.

## Exporting

The deck prints one slide per page (`Ctrl+P` → landscape → background graphics
on). For a PDF with the animations flattened, the Frontend Slides repo ships
`scripts/export-pdf.sh`, which needs Node and Playwright.
