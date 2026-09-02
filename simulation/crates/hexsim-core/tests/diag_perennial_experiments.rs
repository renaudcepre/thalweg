//! Experiments #107: does adding a base flow (water table→surface) increase
//! the persistence of bodies of water? A/B on the baseline diag metric.
//!
//! Each config is measured exactly like `diag_perennial_water` (same
//! lake/river thresholds, same monthly sampling), plus a classification
//! of the failure mode of cells that do NOT hold water all year:
//!   - "freeze": the cell loses its water in winter (accumulated snow, T < 0).
//!   - "dry": the cell dries up without freezing (flow depleted, no snow).
//!
//! This tells us which of the two blockers (#107) dominates the loss of
//! persistence.
//!
//! Run: `cargo test -p hexsim-core --release --test diag_perennial_experiments -- --ignored --nocapture`

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

const RIVER_THRESHOLD: f32 = 0.5;
const LAKE_SURPLUS_MM: f32 = 5.0;

fn build(seed: u32, radius: i32, gw: GroundwaterParams) -> Simulation {
    build_eroded(seed, radius, gw, 0)
}

fn build_eroded(seed: u32, radius: i32, gw: GroundwaterParams, erosion_iters: u32) -> Simulation {
    let d = TerrainParams::default();
    build_terrain(
        seed,
        radius,
        gw,
        &TerrainParams {
            seed,
            erosion_iterations: erosion_iters,
            ..d
        },
    )
}

/// Multiplies the water seeded at worldgen (surface + water table): the
/// terrarium being closed, this amounts to increasing the total water
/// budget, conserved forever.
fn build_watered(seed: u32, radius: i32, water_mult: f32) -> Simulation {
    let d = TerrainParams::default();
    build_terrain(
        seed,
        radius,
        GroundwaterParams::default(),
        &TerrainParams {
            seed,
            initial_water: d.initial_water * water_mult,
            initial_groundwater: d.initial_groundwater * water_mult,
            ..d
        },
    )
}

fn build_terrain(
    seed: u32,
    radius: i32,
    gw: GroundwaterParams,
    terrain: &TerrainParams,
) -> Simulation {
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(&mut grid, terrain);
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        gw,
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed,
            ..WindParams::default()
        },
    )
}

/// For each cell: is it in water (lake|river) at this instant, and if
/// not, what state is it in (frozen / dry)?
struct Sample {
    water: Vec<bool>,
    frozen: Vec<bool>, // snow > 20 mm and no free water
    budget: f32,       // total water (surface+water table+snowpack+moisture) at this instant
}

fn sample(sim: &Simulation) -> Sample {
    let grid = sim.grid();
    let d = sim.discharge_map();
    let n = grid.len();
    let mut water = Vec::with_capacity(n);
    let mut frozen = Vec::with_capacity(n);
    let mut budget = 0.0;
    for (i, (_, c)) in grid.iter().enumerate() {
        let is_lake = (c.water_level - c.water_capacity) > LAKE_SURPLUS_MM;
        let is_river = d.get(i).copied().unwrap_or(0.0) > RIVER_THRESHOLD;
        water.push(is_lake || is_river);
        frozen.push(c.snow_level > 20.0 && !(is_lake || is_river));
        budget += c.water_level + c.groundwater + c.snow_level + c.humidity_total();
    }
    Sample {
        water,
        frozen,
        budget,
    }
}

fn run(label: &str, seed: u32, radius: i32, gw: GroundwaterParams) {
    run_sim(label, build(seed, radius, gw));
}

fn run_sim(label: &str, mut sim: Simulation) {
    for _ in 0..(3 * 365) {
        sim.step();
    }
    let mut samples: Vec<Sample> = Vec::new();
    for _ in 0..(5 * 12) {
        for _ in 0..30 {
            sim.step();
        }
        samples.push(sample(&sim));
    }
    let n = samples[0].water.len();
    let perennial = (0..n)
        .filter(|&i| samples.iter().all(|s| s.water[i]))
        .count();
    let peak = samples
        .iter()
        .map(|s| s.water.iter().filter(|&&w| w).count())
        .max()
        .unwrap_or(0);

    // Failure classification: cells in water at least once, never
    // perennial. Mode = frozen at a sample where it lost its water?
    let mut fail_frozen = 0;
    let mut fail_dry = 0;
    for i in 0..n {
        let ever = samples.iter().any(|s| s.water[i]);
        let always = samples.iter().all(|s| s.water[i]);
        if ever && !always {
            let froze = samples.iter().any(|s| !s.water[i] && s.frozen[i]);
            if froze {
                fail_frozen += 1;
            } else {
                fail_dry += 1;
            }
        }
    }

    let budget = samples.iter().map(|s| s.budget).sum::<f32>()
        / f32::from(u16::try_from(samples.len()).expect("sample count fits u16"));
    eprintln!(
        "  {label:<26} budget {budget:>9.0}  perennial {perennial:>4}  peak {peak:>4}  failures: freeze {fail_frozen:>4} / dry {fail_dry:>4}"
    );
}

fn gw_with(baseflow: f32, max_cap: f32) -> GroundwaterParams {
    GroundwaterParams {
        baseflow_coef: baseflow,
        max_capacity: max_cap,
        ..GroundwaterParams::default()
    }
}

#[test]
#[ignore = "experiment #107, baseflow_coef sweep vs persistence (seed 7, r20)"]
fn baseflow_sweep_seed7() {
    let (seed, radius) = (7, 20);
    eprintln!("\n=== #107 base flow sweep / seed {seed} r{radius} (3 year warmup + 5 years) ===");
    let cap = GroundwaterParams::default().max_capacity;
    run("baseline (coef 0)", seed, radius, gw_with(0.0, cap));
    run("baseflow 0.02", seed, radius, gw_with(0.02, cap));
    run("baseflow 0.05", seed, radius, gw_with(0.05, cap));
    run("baseflow 0.10", seed, radius, gw_with(0.10, cap));
    run(
        "baseflow 0.05 + cap 300",
        seed,
        radius,
        gw_with(0.05, 300.0),
    );
}

/// Actually properly sized aquifer: capacity of several meters, strong
/// infiltration, slow lateral drainage (so it fills up and stays distributed),
/// base flow for restitution. Does THIS sustain the rivers?
#[test]
#[ignore = "experiment #107, sized aquifer vs persistence (seed 7, r20)"]
fn aquifer_sweep_seed7() {
    let (seed, radius) = (7, 20);
    eprintln!("\n=== #107 sized aquifer / seed {seed} r{radius} (3 year warmup + 5 years) ===");
    let d = GroundwaterParams::default();
    run("baseline", seed, radius, d.clone());
    run(
        "cap1000 infil0.3 diff/10 bf0.05",
        seed,
        radius,
        GroundwaterParams {
            max_capacity: 1000.0,
            infiltration_rate: 0.3,
            diffusion_rate: 0.003,
            baseflow_coef: 0.05,
        },
    );
    run(
        "cap3000 infil0.5 diff/10 bf0.03",
        seed,
        radius,
        GroundwaterParams {
            max_capacity: 3000.0,
            infiltration_rate: 0.5,
            diffusion_rate: 0.003,
            baseflow_coef: 0.03,
        },
    );
    run(
        "cap3000 infil0.5 diff0 bf0.02",
        seed,
        radius,
        GroundwaterParams {
            max_capacity: 3000.0,
            infiltration_rate: 0.5,
            diffusion_rate: 0.0,
            baseflow_coef: 0.02,
        },
    );
}

/// Erosion #105 (opt-in) digs deep basins → lakes that don't
/// freeze to the bottom nor evaporate in a single summer. Issue #107 gives it
/// as a lead for PERENNIAL lakes. We test it at worldgen (20 iterations, the
/// setting validated by the author in #105), alone then coupled with base flow.
#[test]
#[ignore = "experiment #107, erosion basins vs persistence (seed 7, r20)"]
fn erosion_basins_seed7() {
    let (seed, radius) = (7, 20);
    eprintln!("\n=== #107 erosion basins / seed {seed} r{radius} (3 year warmup + 5 years) ===");
    run_sim(
        "baseline (erosion off)",
        build(seed, radius, GroundwaterParams::default()),
    );
    run_sim(
        "erosion 20",
        build_eroded(seed, radius, GroundwaterParams::default(), 20),
    );
    run_sim(
        "erosion 20 + bf0.05",
        build_eroded(
            seed,
            radius,
            GroundwaterParams {
                baseflow_coef: 0.05,
                ..GroundwaterParams::default()
            },
            20,
        ),
    );
    run_sim(
        "erosion 60 + bf0.05 cap500",
        build_eroded(
            seed,
            radius,
            GroundwaterParams {
                baseflow_coef: 0.05,
                max_capacity: 500.0,
                ..GroundwaterParams::default()
            },
            60,
        ),
    );
}

/// THE question: is there simply enough water? The terrarium is closed, the
/// total budget is fixed at worldgen (surface water + seeded water table). We
/// scale it ×1/×10/×40/×100 and see if persistence follows. If so →
/// it's a QUANTITY problem, not a cycle mechanics one.
#[test]
#[ignore = "experiment #107, total water budget vs persistence (seed 7, r20)"]
fn water_budget_sweep_seed7() {
    let (seed, radius) = (7, 20);
    eprintln!(
        "\n=== #107 total water budget / seed {seed} r{radius} (3 year warmup + 5 years) ==="
    );
    run_sim("water ×1 (default)", build_watered(seed, radius, 1.0));
    run_sim("water ×10", build_watered(seed, radius, 10.0));
    run_sim("water ×40", build_watered(seed, radius, 40.0));
    run_sim("water ×100", build_watered(seed, radius, 100.0));
}
