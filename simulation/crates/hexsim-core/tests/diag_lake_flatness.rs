//! Diagnostic #106/#107: is the surface of a multi-hex lake flat?
//!
//! Breaks down the "staircase lake" observed in rendering into two contributions:
//!   1. **Physics**: the `effective_elevation` gap (= elevation + surplus/1000,
//!      in SI meters) between cells of the same connected lake. Zero = flat
//!      hydrostatic surface. Non-zero = the MFD hasn't leveled it out (equilibrium not reached).
//!   2. **Rendering**: the front draws the roof at `elevation + surplus/200`
//!      (`STOCK_VISUAL = 1/200`), i.e. ×5 the physical sheet (mm→m = /1000). At a
//!      physically flat surface, this roof equals `5C − 4·elevation`, a step of
//!      **4× the terrain relief under the lake**, inverted.
//!
//! If the physical gap is small but the rendered gap is large, it's mostly
//! rendering (the ×5 factor). If the physical gap is already large, the MFD
//! isn't leveling (real bug #106).
//!
//! Run: `cargo test -p hexsim-core --release --test diag_lake_flatness -- --ignored --nocapture`

mod common;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

/// Reproduces the author's world: water ×40, water table ×40 (seed 7, r20).
fn build(seed: u32, radius: i32) -> Simulation {
    let d = TerrainParams::default();
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed,
            initial_water: d.initial_water * 40.0,
            initial_groundwater: d.initial_groundwater * 40.0,
            ..d
        },
    );
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed,
            ..WindParams::default()
        },
    )
}

/// "Open water" threshold (mm of surplus), same spirit as `is_open_water`.
const LAKE_MM: f32 = 5.0;
/// mm-to-m conversion for front-end rendering (`STOCK_VISUAL`).
const STOCK_VISUAL: f32 = 1.0 / 200.0;
/// Physical mm-to-m conversion (`effective_elevation`).
const PHYS_MM_TO_M: f32 = 1.0 / 1000.0;

/// Largest connected component of lake cells. Returns its indices.
fn largest_lake(sim: &Simulation) -> Vec<usize> {
    let grid = sim.grid();
    let cells = grid.cells_slice();
    let n = cells.len();
    let is_lake = |i: usize| (cells[i].water_level - cells[i].water_capacity) > LAKE_MM;
    let mut seen = vec![false; n];
    let mut best: Vec<usize> = Vec::new();
    for start in 0..n {
        if seen[start] || !is_lake(start) {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            comp.push(i);
            for j in grid.neighbor_indices_toric(i) {
                if !seen[j] && is_lake(j) {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if comp.len() > best.len() {
            best = comp;
        }
    }
    best
}

fn spread(vals: &[f32]) -> (f32, f32, f32) {
    let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (min, max, max - min)
}

#[test]
#[ignore = "diagnostic #106, flatness of a multi-hex lake (seed 7, r20, water ×40)"]
fn lake_surface_flatness() {
    let (seed, radius) = (7, 20);
    eprintln!("\n=== #106 lake flatness, seed {seed} r{radius}, water/table ×40 ===");
    let mut sim = build(seed, radius);
    for _ in 0..(4 * 365) {
        sim.step();
    }

    let lake = largest_lake(&sim);
    if lake.len() < 3 {
        eprintln!("  no multi-hex lake (size {}), increase water", lake.len());
        return;
    }
    let cells = sim.grid().cells_slice();

    let elevation: Vec<f32> = lake.iter().map(|&i| cells[i].elevation).collect();
    let surplus: Vec<f32> = lake
        .iter()
        .map(|&i| (cells[i].water_level - cells[i].water_capacity).max(0.0))
        .collect();
    // Physical surface = effective_elevation (SI meters).
    let phys_top: Vec<f32> = lake
        .iter()
        .map(|&i| cells[i].elevation + surplus_at(cells, i) * PHYS_MM_TO_M)
        .collect();
    // Roof rendered by the front end (STOCK_VISUAL = 1/200).
    let rendered_top: Vec<f32> = lake
        .iter()
        .map(|&i| cells[i].elevation + surplus_at(cells, i) * STOCK_VISUAL)
        .collect();

    let (e0, e1, de) = spread(&elevation);
    let (s0, s1, ds) = spread(&surplus);
    let (_, _, dphys) = spread(&phys_top);
    let (_, _, drend) = spread(&rendered_top);

    eprintln!("  lake size            : {} cells", lake.len());
    eprintln!("  terrain elevation    : {e0:.2} → {e1:.2} m  (relief {de:.2} m)");
    eprintln!("  surplus (water sheet): {s0:.1} → {s1:.1} mm  (gap {ds:.1} mm)");
    eprintln!("  PHYSICAL surface     : gap {dphys:.3} m  (hydrostatic flat = ~0 after #106)");
    eprintln!("  RENDERED roof if ×5 (old front) : gap {drend:.3} m  ← the step before the fix");
    eprintln!("  → the front now renders at physical height (÷1000): flat = flat");
}

fn surplus_at(cells: &[hexsim_core::cell::CellProperties], i: usize) -> f32 {
    (cells[i].water_level - cells[i].water_capacity).max(0.0)
}
