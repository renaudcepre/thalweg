//! Diagnostic #105: the author's "simple" model: carve ∝ discharge
//! (LINEAR), wherever water flows, period. Does it CAPTURE parallel
//! rills (the strongest carves more, so neighbors re-route into it)?
//!
//! Author's hypothesis: stream-power (√discharge + a grading slope
//! term) doesn't capture because it compresses the strong/weak gap and
//! flattens the profile. A LINEAR incision in discharge, without
//! grading, amplifies the dominant channel fast enough for parallel
//! rills to merge into it.
//!
//! Test: raw terrain -> simple erosion (N passes of: flow accumulation
//! -> carve ∝ discharge, CFL-capped to avoid reversing the slope ->
//! re-route) -> 2-year live simulation -> river regime. Compared to
//! the raw terrain and at several (k, iterations). Successful CAPTURE =
//! `river_cells` down, `max` up.
//!
//! Lancer : `just diag-tool capture_erosion`

mod common;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::cell::CellProperties;
use hexsim_core::erosion::{WORLDGEN_FLOW_CONCENTRATION, accumulate_flow, discharge_gini};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const SEED: u32 = 7;
const RIVER_THRESHOLD: f32 = 0.5;

/// The "simple" model: at each pass, we route the flow (drained area),
/// carve EACH cell proportionally to ITS discharge (linear, normalized
/// by the max so it's independent of map size), capped at `cfl × drop`
/// to avoid reversing the slope. Then we re-route at the next pass:
/// this is where capture emerges (a neighbor of a carved gully now
/// sees it as a low point and drains into it).
fn erode_simple(grid: &mut HexGrid, k_meters: f32, iterations: u32, cfl: f32) {
    let n = grid.len();
    let mut carve = vec![0.0_f32; n];
    for _ in 0..iterations {
        let (discharge, _) = accumulate_flow(grid, WORLDGEN_FLOW_CONCENTRATION);
        let maxq = discharge.iter().copied().fold(1.0_f32, f32::max);
        let cells = grid.cells_slice();
        for i in 0..n {
            let eff_i = cells[i].effective_elevation();
            let min_nb = grid
                .neighbor_indices_toric(i)
                .iter()
                .map(|&j| cells[j].effective_elevation())
                .fold(f32::INFINITY, f32::min);
            let drop = (eff_i - min_nb).max(0.0);
            carve[i] = (k_meters * discharge[i] / maxq).min(cfl * drop);
        }
        let cells = grid.cells_slice_mut();
        for (i, c) in cells.iter_mut().enumerate() {
            c.elevation -= carve[i];
        }
    }
}

fn build_sim_from(grid: HexGrid) -> Simulation {
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed: SEED,
            ..WindParams::default()
        },
    )
}

fn raw_terrain() -> HexGrid {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: SEED,
            erosion_iterations: 0, // raw: isolates the effect of simple erosion
            ..TerrainParams::default()
        },
    );
    grid
}

fn live_river_stats(sim: &Simulation) -> (usize, f64, f32) {
    let d = sim.discharge_map();
    let rivers = d.iter().filter(|&&x| x > RIVER_THRESHOLD).count();
    let gini = discharge_gini(d);
    let max = d.iter().copied().fold(0.0_f32, f32::max);
    (rivers, gini, max)
}

/// Compares two regimes: how many cells have a "steep" elevation drop
/// to a neighbor? A proxy for clearly incised valleys (sharp channels).
fn incised_channels(grid: &HexGrid) -> usize {
    let cells = grid.cells_slice();
    (0..cells.len())
        .filter(|&i| {
            let zi = cells[i].elevation;
            grid.neighbor_indices_toric(i)
                .iter()
                .any(|&j| zi - cells[j].elevation > 30.0)
        })
        .count()
}

fn run_case(label: &str, grid: HexGrid) {
    let incised = incised_channels(&grid);
    let mut sim = build_sim_from(grid);
    for _ in 0..(2 * 365) {
        sim.step();
    }
    let (rivers, gini, max) = live_river_stats(&sim);
    eprintln!(
        "  {label:<28} incised_channels {incised:>4} | live: river_cells {rivers:>4} ; gini {gini:.4} ; max {max:>6.1}"
    );
}

#[test]
#[ignore = "diagnostic #105, simple erosion (carve ∝ discharge) vs capture (seed 7, r30, 2 years)"]
fn simple_capture_erosion() {
    eprintln!("=== #105 / SIMPLE erosion (∝ discharge) → capture / seed {SEED} (r{RADIUS}) ===");
    eprintln!("  successful CAPTURE = river_cells ↓ and max ↑ vs raw");

    // Sanity: the raw terrain must be identical for everyone (determinism).
    let _ = CellProperties::default();

    run_case("raw (erosion off)", raw_terrain());
    for (k, iter) in [(10.0_f32, 30_u32), (30.0, 30), (30.0, 60), (60.0, 60)] {
        let mut g = raw_terrain();
        erode_simple(&mut g, k, iter, 0.5);
        run_case(&format!("simple k={k:.0} iter={iter}"), g);
    }
}
