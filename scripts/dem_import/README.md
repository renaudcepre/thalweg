# dem_import — real elevation survey on the hex grid

Projects a 30 m SRTM survey (via the public `opentopodata.org` API) onto the
engine's hexagonal domain, to serve as a **validation oracle**: a world whose
true shape is known, to judge procedural terrain against.

This is not a playable world. The engine is a torus; a real survey isn't
periodic, so the border cells suffer an elevation discontinuity across the
seam.

## The reference world

`out/` is gitignored (1.7 MB per survey). The file doesn't live in the repo,
only this command rebuilds it:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python dem_to_hexgrid.py \
    --center-lat 44.620 --center-lon 5.090 \
    --radius 120 --cell-spacing-m 130 \
    --out out/drome_saou_r120.json
```

About 8 minutes (43,561 points, the public API caps at ~1 request/s in
batches of 100).

Drôme provençale, 31.2 km wide. The Saoû massif tops out at 1547 m, 8.9 km
east-northeast of the center. In frame: Saoû, Bourdeaux, Soyans, Crest,
Dieulefit, Le Poët-Laval, La Bégude-de-Mazenc, Pont-de-Barret. Bonlieu-sur-Roubion
falls outside it.

Loading:

```bash
cd ../../simulation
HEXSIM_RADIUS=120 HEXSIM_DEM_OVERRIDE=../scripts/dem_import/out/drome_saou_r120.json \
    cargo run --release --bin hexsim-cli
```

## The two pitfalls, both hit in practice

**The radius must match.** A radius-120 survey loaded into a radius-80 grid
loses 55% of its cells — the border ones, i.e. the peripheral relief. Until
2026-07-10 this was silent, and the world capped at 1059 m with nothing to
flag it. The core now emits a `warn!` and counts the cells actually written
(`DemApplyReport`).

**The spacing must match.** The engine places its hexes at `CELL_SPACING_M`
(`dynamics.rs`), not the file's own spacing. A survey sampled at 130 m and
rendered at 1074 m spreads its relief over 8× the intended width: mountains
diluted into hills. The script warns if the two values diverge.

## Provenance

Output: `{"meta": {center_lat, center_lon, radius, cell_spacing_m,
samples_per_cell, dataset}, "cells": [{q, r, elevation}, ...]}`.

`meta` exists because an orphaned survey doesn't say which portion of the
globe it describes. Recovering the framing of `drome_fine_r120.json` took a
correlation-based realignment against an independent SRTM grid. The core
still accepts the old format (raw array), warning about the missing
provenance.

`--samples-per-cell` is derived from the mesh: 1 below 300 m (the center is
enough, averaging would reintroduce the blur it's meant to avoid), 7 above
(center + edge midpoints, smooths SRTM aliasing).
