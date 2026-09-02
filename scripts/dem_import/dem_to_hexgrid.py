#!/usr/bin/env python3
"""Projects a real elevation survey (SRTM 30 m, via opentopodata.org) onto
the engine's hexagonal grid, for a validation load (see
`hexsim_core::terrain::apply_dem_override`).

Geometry replicated bit-for-bit from the Rust core's:
- hexagonal domain `HexGrid::from_radius` (grid.rs)
- `hex_to_world` (terrain.rs): unit circumradius, pointy-top
- real side = spacing / sqrt(3): the hexagon is regular, so side =
  circumradius. The engine's spacing is `CELL_SPACING_M` (dynamics.rs),
  set to 130 m on 2026-07-09 (commit d6be105).

WARNING if --cell-spacing-m != CELL_SPACING_M: the engine places its hexes
at ITS spacing, not the file's. A survey sampled finer than the engine has
its relief spread over a larger surface (the 2026-07-09 trap: a 130 m DEM
rendered at 1074 m, mountains diluted over 8x the intended width); coarser,
it gets compressed. Keep both values equal.

The radius must also match: a radius-120 survey loaded into a radius-80
grid loses 55% of its cells, the border ones, i.e. the peripheral relief.
The core now reports this (`DemApplyReport.skipped`), but the only correct
framing is the one where both radii coincide.

Fetches no raster file (no rasterio/gdal): queries the public
opentopodata.org API point by point. `--samples-per-cell 7` averages the
center + 6 edge midpoints (approximates an area average, useful when the
cell is large relative to SRTM's 30 m resolution); `1` queries only the
center (7x faster, relevant when the cell is already fine, since averaging
would reintroduce the very blur we're trying to avoid). By default the
choice is derived from `--cell-spacing-m`.

Output: `{"meta": {...}, "cells": [{"q":.., "r":.., "elevation":..}, ...]}`.
`meta` carries the center, radius, and spacing: without it a file doesn't
say which portion of the globe it describes, and recovering the framing of
an orphaned survey requires a correlation-based realignment (lived through
on 2026-07-10). The core still accepts the old format (raw array) but warns
about the missing provenance.
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from pyproj import Transformer

ENGINE_CELL_SPACING_M = 130.0  # simulation/crates/hexsim-core/src/dynamics.rs (CELL_SPACING_M)

# Beyond this spacing, a cell covers enough SRTM pixels (30 m) that
# averaging 7 points represents it better than its center alone.
SUBPIXEL_SPACING_M = 300.0

# hex_direction_to_world (coord.rs): 6 unit center-to-center directions.
NEIGHBOR_DIRECTIONS = [
    (1.0, 0.0),
    (0.5, -math.sqrt(3) / 2),
    (-0.5, -math.sqrt(3) / 2),
    (-1.0, 0.0),
    (-0.5, math.sqrt(3) / 2),
    (0.5, math.sqrt(3) / 2),
]

OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"
BATCH_SIZE = 100  # public API cap
REQUEST_INTERVAL_S = 1.05  # public rate limit ~1 req/s, safety margin

LAMBERT93 = "EPSG:2154"
WGS84 = "EPSG:4326"


def hex_domain(radius):
    """Reproduces HexGrid::from_radius (grid.rs) exactly: valid (q, r)."""
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            yield q, r


def hex_to_world_m(q, r, side_m):
    """terrain.rs::hex_to_world, scaled to real units (meters)."""
    x = math.sqrt(3.0) * q + (math.sqrt(3.0) / 2.0) * r
    y = 1.5 * r
    return x * side_m, y * side_m


def sample_points_for_hex(q, r, cell_spacing_m, samples_per_cell):
    """`samples_per_cell=1`: center only. `7`: center + 6 edge midpoints
    (apothem = cell_spacing_m/2), as meter offsets (x, y)."""
    side_m = cell_spacing_m / math.sqrt(3.0)
    cx, cy = hex_to_world_m(q, r, side_m)
    if samples_per_cell == 1:
        return [(cx, cy)]
    apothem_m = cell_spacing_m / 2.0
    points = [(cx, cy)]
    for dx, dy in NEIGHBOR_DIRECTIONS:
        points.append((cx + dx * apothem_m, cy + dy * apothem_m))
    return points


def fetch_elevations(locations, log):
    """locations: list of (lat, lon). Returns the list of elevations (m),
    same order (chunks processed and appended sequentially). Requests
    batched at BATCH_SIZE, throttled."""
    elevations = []
    for start in range(0, len(locations), BATCH_SIZE):
        chunk = locations[start : start + BATCH_SIZE]
        locs_param = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in chunk)
        url = f"{OPENTOPODATA_URL}?locations={urllib.parse.quote(locs_param, safe='|,.')}"
        req = urllib.request.Request(url, headers={"User-Agent": "hexsim-dem-import/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"opentopodata responded {e.code}: {body}") from e
        if data.get("status") != "OK":
            raise RuntimeError(f"opentopodata status={data.get('status')}: {data}")
        elevations.extend(result["elevation"] for result in data["results"])
        log(f"  {min(start + BATCH_SIZE, len(locations))}/{len(locations)} points fetched")
        if start + BATCH_SIZE < len(locations):
            time.sleep(REQUEST_INTERVAL_S)
    return elevations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--center-lat", type=float, required=True, help="WGS84 latitude of the center (hex 0,0)")
    parser.add_argument("--center-lon", type=float, required=True, help="WGS84 longitude of the center (hex 0,0)")
    parser.add_argument("--radius", type=int, required=True, help="hexagonal radius (must match HEXSIM_RADIUS)")
    parser.add_argument(
        "--cell-spacing-m",
        type=float,
        default=ENGINE_CELL_SPACING_M,
        help=f"center-to-center spacing in m (default {ENGINE_CELL_SPACING_M} = the engine's; "
        "lower it for a finer visualization grid, see WARNING at the top of the file)",
    )
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        choices=(1, 7),
        default=None,
        help="1 = center only, 7 = center+edge midpoints (smooths SRTM aliasing). "
        f"Default: 1 if the cell is finer than {SUBPIXEL_SPACING_M:.0f} m, else 7",
    )
    parser.add_argument("--out", required=True, help="path to the output JSON")
    args = parser.parse_args()

    if args.samples_per_cell is None:
        args.samples_per_cell = 1 if args.cell_spacing_m < SUBPIXEL_SPACING_M else 7

    if args.cell_spacing_m != ENGINE_CELL_SPACING_M:
        print(
            f"WARNING: cell-spacing-m={args.cell_spacing_m} != engine spacing "
            f"({ENGINE_CELL_SPACING_M}). The engine will place hexes at ITS spacing: the relief "
            f"will be {'stretched' if args.cell_spacing_m < ENGINE_CELL_SPACING_M else 'compressed'} by a "
            f"factor {ENGINE_CELL_SPACING_M / args.cell_spacing_m:.2f}. "
            "See WARNING at the top of the file.",
            file=sys.stderr,
        )

    def log(msg):
        print(msg, file=sys.stderr)

    coords = list(hex_domain(args.radius))
    log(
        f"Domain: radius={args.radius} spacing={args.cell_spacing_m}m -> "
        f"{len(coords)} hexagons, {len(coords) * args.samples_per_cell} points to sample"
    )

    to_lambert = Transformer.from_crs(WGS84, LAMBERT93, always_xy=True)
    to_wgs84 = Transformer.from_crs(LAMBERT93, WGS84, always_xy=True)

    center_x, center_y = to_lambert.transform(args.center_lon, args.center_lat)
    log(f"Lambert-93 center: x={center_x:.1f} y={center_y:.1f}")

    # Flattens all points from all hexes into a single list of requests,
    # with offsets (start index) to fold back afterward.
    all_locations = []
    offsets = []
    for q, r in coords:
        offsets.append(len(all_locations))
        for ox, oy in sample_points_for_hex(q, r, args.cell_spacing_m, args.samples_per_cell):
            lon, lat = to_wgs84.transform(center_x + ox, center_y + oy)
            all_locations.append((lat, lon))
    offsets.append(len(all_locations))

    log(f"Querying opentopodata.org (srtm30m), {len(all_locations)} points...")
    elevations = fetch_elevations(all_locations, log)

    cells = []
    for i, (q, r) in enumerate(coords):
        sub = elevations[offsets[i] : offsets[i + 1]]
        mean_elev = sum(sub) / len(sub)
        cells.append({"q": q, "r": r, "elevation": round(mean_elev, 1)})

    payload = {
        "meta": {
            "center_lat": args.center_lat,
            "center_lon": args.center_lon,
            "radius": args.radius,
            "cell_spacing_m": args.cell_spacing_m,
            "samples_per_cell": args.samples_per_cell,
            "dataset": "srtm30m",
            "source": OPENTOPODATA_URL,
        },
        "cells": cells,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    elevs = [c["elevation"] for c in cells]
    half_width_km = args.radius * args.cell_spacing_m / 1000.0
    log(
        f"Wrote {args.out}: {len(cells)} cells, "
        f"min={min(elevs):.0f}m max={max(elevs):.0f}m mean={sum(elevs) / len(elevs):.0f}m"
    )
    log(
        f"Center ({args.center_lat}, {args.center_lon}), radius {args.radius}, "
        f"real half-width {half_width_km:.1f} km. Load into a grid of radius "
        f"{args.radius} (HEXSIM_RADIUS), otherwise the edge relief will be truncated."
    )


if __name__ == "__main__":
    main()
