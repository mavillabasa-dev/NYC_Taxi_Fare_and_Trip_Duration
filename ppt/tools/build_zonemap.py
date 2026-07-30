"""Render the real TLC taxi-zone shapefile into the SVG map used on slide 10.

This is the same shapefile T-116 has to download and T-105 derives centroids
from, so the map on the slide is the actual geometry the model will use --
not a stock illustration of New York.

Usage (from the repo root, with the project venv active):

    python ppt/tools/build_zonemap.py

Writes ppt/assets/zones-map.svg (standalone) and
ppt/tools/_fragments/zones-paths.svg (the bare <path> set inlined into the deck).
"""
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
SHP = ROOT / "dataset" / "taxi_zones" / "taxi_zones.shp"
ASSETS = ROOT / "ppt" / "assets"
FRAG = Path(__file__).resolve().parent / "_fragments"
ASSETS.mkdir(parents=True, exist_ok=True)
FRAG.mkdir(parents=True, exist_ok=True)

W, H, PAD = 1000, 1100, 12
MIN_AREA = 2.0e6          # drop islets that are sub-pixel at deck scale
SIMPLIFY_FT = 500         # the map is drawn ~520px wide, so this is free

# Borough tones step down one gold ramp: Manhattan carries the most pickups,
# so it takes the brightest step and the outer boroughs recede.
TONE = {"Manhattan": "m", "EWR": "e", "Queens": "q",
        "Brooklyn": "k", "Bronx": "x", "Staten Island": "s"}
STYLE = (".z{stroke:#0B0A0C;stroke-width:1.1;stroke-linejoin:round}"
         ".m{fill:#D9A93A}.e{fill:#B98C2C}.q{fill:#8A6821}"
         ".k{fill:#7A5C1D}.x{fill:#6B5019}.s{fill:#5A431A}")


def main():
    gdf = gpd.read_file(SHP)
    print(f"zones: {len(gdf)} | crs: {gdf.crs}")
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_FT, preserve_topology=True)

    minx, miny, maxx, maxy = gdf.total_bounds
    s = min((W - 2 * PAD) / (maxx - minx), (H - 2 * PAD) / (maxy - miny))
    ox = PAD + ((W - 2 * PAD) - (maxx - minx) * s) / 2
    oy = PAD + ((H - 2 * PAD) - (maxy - miny) * s) / 2

    def pt(x, y):
        # flip Y: the shapefile's Y grows north, SVG's grows down
        return f"{ox + (x - minx) * s:.0f},{oy + (maxy - y) * s:.0f}"

    def geom_d(geom):
        if geom is None or geom.is_empty:
            return ""
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        return "".join("M" + "L".join(pt(x, y) for x, y, *_ in p.exterior.coords) + "Z"
                       for p in polys if p.area >= MIN_AREA)

    paths = []
    for _, row in gdf.iterrows():
        d = geom_d(row.geometry)
        if d:
            paths.append(f'<path class="z {TONE.get(row.borough, "q")}" d="{d}"/>')
    print(f"polygons emitted: {len(paths)}")

    body = "".join(paths)
    FRAG.joinpath("zones-paths.svg").write_text(body, encoding="utf-8")

    ASSETS.joinpath("zones-map.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="The {len(paths)} NYC TLC taxi zones, shaded by borough">'
        f"<style>{STYLE}</style>"
        f'<rect width="{W}" height="{H}" fill="#0B0A0C"/>{body}</svg>',
        encoding="utf-8")

    # Same geometry T-116 caches as a centroid lookup keyed by LocationID:
    # take centroids in the projected CRS, then convert to lon/lat.
    cent = gdf.geometry.centroid.to_crs(4326)
    print(f"centroids derived for {int(cent.notna().sum())} zones "
          f"(LocationID 1-265, {len(gdf)} present in the shapefile)")
    print("wrote ppt/assets/zones-map.svg + ppt/tools/_fragments/zones-paths.svg")


if __name__ == "__main__":
    main()
