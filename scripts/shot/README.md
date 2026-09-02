# hexsim-shot

Playwright screenshots of the front end.

Screenshot tool for the HexSim 3D scene, meant so Claude (or a human) can
*see* the simulation instead of only reading JSON diagnostics.

The front end (`localhost:8355`) renders the state produced by the engine.
This script drives the camera deterministically via the `window.__hexcam`
hook (`frontend/main.js`), forces a render, then captures the PNG.

## Setup (once)

```bash
just shot-setup        # npm install + playwright install chromium
```

## Usage

The server must be running (`just run` or `just rebuild`). Drive the
simulation with the usual commands, then capture:

```bash
just play              # or: just pause / just step 30 / just reset 42
just shot              # framed map, oblique view → scripts/shot/out/shot-*.png
```

The script prints a JSON line `{ path, tick, hourTick, view, bytes }`.
Claude then reads the PNG via the Read tool (visual rendering).

### Framing examples

```bash
just shot --zoom-factor 0.5            # zoom in ×2
just shot --azimuth 90 --polar 30      # different viewing angle
just shot-top                          # top-down view, bare map
just shot --view temperature --clean   # temperature backdrop, no sidebars
just shot --target "5,-3" --zoom 12    # recentered close-up
just shot --canvas-only --clean        # canvas only, no surrounding DOM
```

The full list of options is documented in `shot.mjs`'s header
(`--help` also returns it).

## Camera hook (`window.__hexcam`)

Exposed by `frontend/main.js`. Angular convention (THREE.Spherical):

- `radius`, camera→target distance in world units (1 hex ≈ 1.0)
- `polarDeg`, tilt: `0` = top-down view, `90` = grazing
- `azimuthDeg`, rotation around the vertical axis: `0` = view from the south

Methods: `ready()`, `state()`, `view()`, `setView(opts)`, `zoom(factor)`,
`fitAll(margin)`, `render()`, `setChromeVisible(bool)`, `cover()`, `mix()`.

## Cover composition (`just cover`)

`cover.mjs` reads `__hexcam.cover()` and prints the cover composition of the
current state (no recomputation: the core supplies `dominant_species` +
`species_mix`):

- `count` / `pct`, cells per dominant species (+ `water`, `bare`)
- `meanVeg`, mean total vegetation cover [0,1]
- `mix`, mixing *within* hexes: `meanDominantShare` (share of the dominant
  species), `meanEffSpecies` (effective number of species/hex, inverse
  Simpson), `monoPct` (> 90% one species) / `mixedPct` (< 70% dominant),
  `communityPct` (average composition).

Flow: drive the sim (`just step|play|reset`) then `just cover`.

## Output

PNGs go to `scripts/shot/out/` (gitignored). `--out <path>` to choose
another location.
