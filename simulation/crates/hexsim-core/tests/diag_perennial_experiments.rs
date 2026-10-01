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
            initial_groundwater_frac: (d.initial_groundwater_frac * water_mult).min(1.0),
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
    build_full(seed, radius, gw, terrain, AtmosphereParams::default())
}

fn build_full(
    seed: u32,
    radius: i32,
    gw: GroundwaterParams,
    terrain: &TerrainParams,
    atmo: AtmosphereParams,
) -> Simulation {
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(&mut grid, terrain);
    Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        gw,
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed,
            ..WindParams::default()
        },
    )
}

/// For each cell: is it a lake, a river reach, and if neither, is it
/// under snow (frozen) at this instant? The lake mask is the core's
/// water-body predicate (liquid surplus + ice, #157): a frozen lake is
/// still a lake.
struct Sample {
    lake: Vec<bool>,
    river: Vec<bool>,
    frozen: Vec<bool>, // snow > 20 mm and no free water
    budget: f32,       // total water (surface+water table+snowpack+moisture) at this instant
    groundwater: f32,
    aquifer: f32,
}

fn sample(sim: &Simulation) -> Sample {
    let grid = sim.grid();
    let d = sim.discharge_map();
    let n = grid.len();
    let mut lake = Vec::with_capacity(n);
    let mut river = Vec::with_capacity(n);
    let mut frozen = Vec::with_capacity(n);
    let mut budget = 0.0;
    let mut groundwater = 0.0;
    let mut aquifer = 0.0;
    for (i, (_, c)) in grid.iter().enumerate() {
        let is_lake = c.is_open_water_at(LAKE_SURPLUS_MM);
        let is_river = d.get(i).copied().unwrap_or(0.0) > RIVER_THRESHOLD;
        lake.push(is_lake);
        river.push(is_river);
        frozen.push(c.snow_level > 20.0 && !(is_lake || is_river));
        // Surface stocks only: the moist upper layer is counted once,
        // outside the loop, on its coarse reference stock (coarse upper
        // layer, step 2 — the fine `humidity_upper`/`cloud_water` are
        // views, summing them is not the mass).
        budget +=
            c.water_level + c.groundwater + c.aquifer + c.frozen_surface() + c.humidity_surface;
        groundwater += c.groundwater;
        aquifer += c.aquifer;
    }
    let cells = f32::from(u16::try_from(n).expect("cell count fits u16"));
    Sample {
        lake,
        river,
        frozen,
        budget: (budget + sim.upper_water_total() + sim.sky_water_total()) / cells,
        groundwater: groundwater / cells,
        aquifer: aquifer / cells,
    }
}

/// Persistence of one kind of water body over the samples: cells wet at
/// every sample, peak of wet cells at one instant, and the failure mode
/// of the cells wet at least once but not always.
struct Persistence {
    perennial: usize,
    peak: usize,
    fail_frozen: usize,
    fail_dry: usize,
}

fn persistence(samples: &[Sample], wet: impl Fn(&Sample, usize) -> bool) -> Persistence {
    let n = samples.first().map_or(0, |s| s.lake.len());
    let mut result = Persistence {
        perennial: 0,
        peak: samples
            .iter()
            .map(|s| (0..n).filter(|&i| wet(s, i)).count())
            .max()
            .unwrap_or(0),
        fail_frozen: 0,
        fail_dry: 0,
    };
    for i in 0..n {
        let ever = samples.iter().any(|s| wet(s, i));
        let always = samples.iter().all(|s| wet(s, i));
        if always {
            result.perennial += 1;
        } else if ever {
            if samples.iter().any(|s| !wet(s, i) && s.frozen[i]) {
                result.fail_frozen += 1;
            } else {
                result.fail_dry += 1;
            }
        }
    }
    result
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
    let water = persistence(&samples, |s, i| s.lake[i] || s.river[i]);
    let lake = persistence(&samples, |s, i| s.lake[i]);
    let river = persistence(&samples, |s, i| s.river[i]);
    let river_off_lake = (0..samples[0].lake.len())
        .filter(|&i| samples.iter().all(|s| s.river[i] && !s.lake[i]))
        .count();

    let count = f32::from(u16::try_from(samples.len()).expect("sample count fits u16"));
    let budget = samples.iter().map(|s| s.budget).sum::<f32>() / count;
    let groundwater = samples.iter().map(|s| s.groundwater).sum::<f32>() / count;
    let aquifer = samples.iter().map(|s| s.aquifer).sum::<f32>() / count;
    eprintln!(
        "  {label:<28} budget {budget:>6.1} mm  gw {groundwater:>5.1} mm  aq {aquifer:>5.1} mm  perennial {:>4} (lake {:>3}, river {:>3}, off lake {:>3})  peak {:>4}  river failures: freeze {:>4} / dry {:>4}  lake failures: freeze {:>3} / dry {:>3}",
        water.perennial,
        lake.perennial,
        river.perennial,
        river_off_lake,
        water.peak,
        river.fail_frozen,
        river.fail_dry,
        lake.fail_frozen,
        lake.fail_dry,
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
            diffusion_rate: 0.003,
            baseflow_coef: 0.05,
            ..GroundwaterParams::default()
        },
    );
    run(
        "cap3000 infil0.5 diff/10 bf0.03",
        seed,
        radius,
        GroundwaterParams {
            max_capacity: 3000.0,
            diffusion_rate: 0.003,
            baseflow_coef: 0.03,
            ..GroundwaterParams::default()
        },
    );
    run(
        "cap3000 infil0.5 diff0 bf0.02",
        seed,
        radius,
        GroundwaterParams {
            max_capacity: 3000.0,
            diffusion_rate: 0.0,
            baseflow_coef: 0.02,
            ..GroundwaterParams::default()
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

/// Resumption of 2026-10-01, after the climatological t0 (#152): the
/// baseline has no perennial river at all, lakes are the only perennial
/// water. Knock-outs on the three seeds of #152 (r30, 3 years of warmup
/// + 5 measured years), each one a diagnosis, not a fix:
/// - transpiration off: the sky leak of #151 (plants draw the water
///   table down to ~1 % of its capacity every summer). Does a full water
///   table feed rivers through resurgence alone?
/// - baseflow 0.05: Maillet's recession, refuted in July as a lever
///   (perennial 4 → 5). Context changed since: no lateral drainage
///   under field capacity (#151), weather regime, deep lakes (#152).
/// - water ×3: is it a quantity problem? The box holds ~42 mm/cell.
#[test]
#[ignore = "experiment #107, knock-outs vs persistence (three seeds, r30)"]
fn knockouts_three_seeds_r30() {
    let radius = 30;
    for seed in [42, 7, 123] {
        eprintln!("\n=== #107 knock-outs / seed {seed} r{radius} (3 year warmup + 5 years) ===");
        let terrain = TerrainParams {
            seed,
            ..TerrainParams::default()
        };
        run_sim(
            "baseline",
            build(seed, radius, GroundwaterParams::default()),
        );
        run_sim(
            "transpiration off",
            build_full(
                seed,
                radius,
                GroundwaterParams::default(),
                &terrain,
                AtmosphereParams {
                    transpiration_coef: 0.0,
                    ..AtmosphereParams::default()
                },
            ),
        );
        run_sim(
            "baseflow 0.05",
            build(
                seed,
                radius,
                gw_with(0.05, GroundwaterParams::default().max_capacity),
            ),
        );
        run_sim(
            "surface water ×3",
            build_terrain(
                seed,
                radius,
                GroundwaterParams::default(),
                &TerrainParams {
                    initial_water: terrain.initial_water * 3.0,
                    ..terrain
                },
            ),
        );
    }
}

/// Deep aquifer (#107): percolation of the root zone's water above field
/// capacity into an aquifer the roots can't reach, Darcy-Dupuit lateral
/// flow, springs where it overflows, Maillet baseflow. `percolation 0` is
/// the engine before the aquifer. `baseflow 0` leaves the lateral flow
/// and the springs alone; `K 50` is the top of the sand range (Freeze &
/// Cherry 1979).
#[test]
#[ignore = "experiment #107, deep aquifer sweep (three seeds, r30)"]
fn aquifer_three_seeds_r30() {
    let radius = 30;
    for seed in [42, 7, 123] {
        eprintln!("\n=== #107 deep aquifer / seed {seed} r{radius} (3 year warmup + 5 years) ===");
        let configs = [
            (
                "percolation 0 (pre-aquifer)",
                GroundwaterParams {
                    percolation_rate: 0.0,
                    ..GroundwaterParams::default()
                },
            ),
            ("aquifer, defaults", GroundwaterParams::default()),
            (
                "aquifer, baseflow 0",
                GroundwaterParams {
                    baseflow_coef: 0.0,
                    ..GroundwaterParams::default()
                },
            ),
            (
                "aquifer, K 50 m/day",
                GroundwaterParams {
                    aquifer_conductivity_m_per_day: 50.0,
                    ..GroundwaterParams::default()
                },
            ),
        ];
        for (label, params) in configs {
            run(label, seed, radius, params);
        }
    }
}

/// Where does the aquifer recharge? Share of the aquifer stock under
/// cells that are a lake, a river reach, or neither, at monthly samples
/// of one year after a 3-year warmup, default engine. If percolation
/// only happens under standing water, the aquifer is a loop under the
/// lakes and cannot feed the rivers between them.
#[test]
#[ignore = "experiment #107, aquifer location (seed 42, r30)"]
fn aquifer_location_seed42() {
    let (seed, radius) = (42, 30);
    let mut sim = build(seed, radius, GroundwaterParams::default());
    for _ in 0..(3 * 365) {
        sim.step();
    }
    eprintln!("\n=== #107 aquifer location / seed {seed} r{radius} ===");
    eprintln!(
        "  {:>5} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "day", "aq total", "under lk", "under rv", "elsewhere", "aq>1mm", "gw>FC"
    );
    let field_capacity_frac = GroundwaterParams::default().field_capacity_frac;
    let max_capacity = GroundwaterParams::default().max_capacity;
    for _ in 0..12 {
        for _ in 0..30 {
            sim.step();
        }
        let d = sim.discharge_map();
        let (mut lake, mut river, mut other) = (0.0_f32, 0.0_f32, 0.0_f32);
        let (mut wet_aquifer, mut above_fc) = (0, 0);
        for (i, (_, c)) in sim.grid().iter().enumerate() {
            if c.is_open_water_at(LAKE_SURPLUS_MM) {
                lake += c.aquifer;
            } else if d.get(i).copied().unwrap_or(0.0) > RIVER_THRESHOLD {
                river += c.aquifer;
            } else {
                other += c.aquifer;
            }
            if c.aquifer > 1.0 {
                wet_aquifer += 1;
            }
            if c.groundwater > c.permeability * max_capacity * field_capacity_frac {
                above_fc += 1;
            }
        }
        eprintln!(
            "  {:>5} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>8} {:>8}",
            sim.tick(),
            lake + river + other,
            lake,
            river,
            other,
            wet_aquifer,
            above_fc
        );
    }
}

/// Infiltration (2026-10-01): the SI default (loam, Ks 317 mm/day)
/// against a clay surface (Ks 14 mm/day, Rawls et al. 1982), three
/// seeds. Before the SI rewrite the fraction-of-ponded-water rate let
/// ~0.25 mm/day in; its ×20 ablation took rivers off lake 0/0/4 → 3/4/5.
#[test]
#[ignore = "experiment #107, infiltration capacity (three seeds, r30)"]
fn infiltration_three_seeds_r30() {
    for seed in [42, 7, 123] {
        eprintln!("\n=== #107 infiltration / seed {seed} r30 (3 year warmup + 5 years) ===");
        run(
            "Ks 317 (loam, default)",
            seed,
            30,
            GroundwaterParams::default(),
        );
        run(
            "Ks 14 (clay)",
            seed,
            30,
            GroundwaterParams {
                saturated_conductivity_mm_per_day: 14.0,
                ..GroundwaterParams::default()
            },
        );
    }
}

/// SI infiltration with more water in the box (×3, ×10 of the worldgen
/// surface water): does a wetter equilibrium carry perennial rivers?
#[test]
#[ignore = "experiment #107, infiltration and water (three seeds, r30)"]
fn infiltration_and_water_three_seeds_r30() {
    for seed in [42, 7, 123] {
        eprintln!(
            "\n=== #107 infiltration + water / seed {seed} r30 (3 year warmup + 5 years) ==="
        );
        for water_mult in [3.0, 10.0] {
            let d = TerrainParams::default();
            run_sim(
                &format!("Ks 317, water x{water_mult}"),
                build_terrain(
                    seed,
                    30,
                    GroundwaterParams::default(),
                    &TerrainParams {
                        seed,
                        initial_water: d.initial_water * water_mult,
                        ..d
                    },
                ),
            );
        }
    }
}

/// Budget curve (2026-10-01): is the 80 mm runoff sheet of the t0 on a
/// plateau of the perennial metric or on a slope? Absolute
/// `initial_water` values, SI infiltration, three seeds.
#[test]
#[ignore = "experiment #107, perennial water against the box's budget (three seeds, r30)"]
fn budget_curve_three_seeds_r30() {
    for seed in [42, 7, 123] {
        eprintln!("\n=== #107 budget curve / seed {seed} r30 (3 year warmup + 5 years) ===");
        for initial_water in [24.0, 40.0, 60.0, 80.0, 120.0, 160.0] {
            run_sim(
                &format!("initial_water {initial_water} mm"),
                build_terrain(
                    seed,
                    30,
                    GroundwaterParams::default(),
                    &TerrainParams {
                        seed,
                        initial_water,
                        ..TerrainParams::default()
                    },
                ),
            );
        }
    }
}
