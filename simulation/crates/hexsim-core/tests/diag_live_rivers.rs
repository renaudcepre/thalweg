//! Diagnostic #105: do REAL live rivers concentrate when the terrain is
//! pre-eroded? (author feedback: "it doesn't do that").
//!
//! So far we've measured the concentration of THEORETICAL flow accumulation
//! (`accumulate_flow`) on eroded terrain. But what the user SEES is the live
//! rivers: `discharge_map` produced by the actual climate→rain→MFD runoff
//! cycle. This instrument compares, at increasing worldgen erosion
//! iterations (0 = raw terrain), the live river regime after 2 years of
//! simulation (seed 7, r30). `river_cells` (cells with `discharge > 0.5`,
//! the diag's/front's definition) MUST DROP if the rivulets merge; the
//! `gini` of live discharge MUST RISE (concentrated flow); `max_discharge`
//! MUST RISE (bigger waterways). If these three don't move with erosion,
//! the promised dendritic convergence isn't reaching the live rivers: the
//! heart of the complaint.
//!
//! Run: `just diag-tool live_rivers`

mod common;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::erosion::discharge_gini;
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

fn build(iterations: u32, accel: f32) -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: SEED,
            erosion_iterations: iterations,
            erosion_accel_years: accel,
            ..TerrainParams::default()
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
            seed: SEED,
            ..WindParams::default()
        },
    )
}

fn river_stats(sim: &Simulation) -> (usize, f64, f32) {
    let d = sim.discharge_map();
    let rivers = d.iter().filter(|&&x| x > RIVER_THRESHOLD).count();
    let gini = discharge_gini(d);
    let max = d.iter().copied().fold(0.0_f32, f32::max);
    (rivers, gini, max)
}

fn run_case(iterations: u32, accel: f32) {
    let mut sim = build(iterations, accel);
    for _ in 0..(2 * 365) {
        sim.step();
    }
    let (rivers, gini, max) = river_stats(&sim);
    eprintln!(
        "  iter {iterations:>3} (accel {accel:>5.0}) : river_cells {rivers:>4} ; gini {gini:.4} ; max_discharge {max:>6.1}"
    );
}

#[test]
#[ignore = "diagnostic #105, LIVE rivers vs erosion worldgen (seed 7, r30, 2 years)"]
fn live_rivers_vs_erosion() {
    eprintln!("=== #105 / LIVE rivers after 2 years / seed {SEED} (r{RADIUS}) ===");
    eprintln!("  Expected if erosion converges: river_cells down, gini up, max_discharge up");
    let accel = hexsim_core::erosion::WORLDGEN_ACCEL_YEARS;
    run_case(0, accel);
    run_case(25, accel);
    run_case(80, accel);
    run_case(200, accel);
    // One pass with stronger accel to see if digging HARD changes anything.
    run_case(80, 8000.0);
}
