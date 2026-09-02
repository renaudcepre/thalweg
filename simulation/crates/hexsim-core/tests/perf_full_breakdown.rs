//! Per-phase breakdown of the REAL tick (`Simulation::step_hour`), measured
//! by the embedded instrumentation (`phase_timing`), not a mirror of the
//! tick like `perf_phase_breakdown`: all Tier 1 AND Tier 3 phases
//! (vegetation, fire, lakes, EMA, normals, history) are covered, at
//! production cadences.
//!
//! Prints a table by world age (30-day window at the start of each
//! simulated year) to see which phases get more expensive with age.
//!
//! Run with:
//! `cargo test --release --test perf_full_breakdown -- --ignored --nocapture`
//!
//! Env: `HEXSIM_PERF_RADIUS` (default 45), `HEXSIM_PERF_YEARS` (default 2,
//! number of annual windows AFTER the young window), `HEXSIM_PERF_SEED`
//! (default 42), `HEXSIM_PERF_SAVE_DIR` (optional: saves a checkpoint
//! `world_y<N>.ckpt` at each year boundary, to replay ablations on
//! an old world without re-simulating).

use std::time::Instant;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

const MEASURE_DAYS: u64 = 30;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// World identical to the live server (`World::from_grid`): emergent fire
/// active, fire seed = world seed, erosion at default (off).
fn live_like_sim(radius: i32, seed: u32) -> Simulation {
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed,
            ..TerrainParams::default()
        },
    );
    let mut sim = Simulation::new(
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
    );
    sim.set_seed(seed);
    sim.update_param("fire.enabled", 1.0);
    sim
}

fn print_window(label: &str, sim: &Simulation, wall_s: f64) {
    let t = sim.phase_timings();
    let hours = t.hours.max(1);
    #[allow(clippy::cast_precision_loss)]
    let hours_f = hours as f64;
    let total = t.total();
    eprintln!("\n=== {label}, {MEASURE_DAYS}-day window ===");
    for (name, s) in t.rows() {
        let pct = if total > 0.0 { 100.0 * s / total } else { 0.0 };
        eprintln!(
            "  {name:<13} {:>8.3} ms/h-tick  ({pct:>5.1} %)",
            1000.0 * s / hours_f
        );
    }
    eprintln!(
        "  {:<13} {:>8.3} ms/h-tick  -> {:.1} ms/day",
        "TOTAL",
        1000.0 * total / hours_f,
        1000.0 * total / (hours_f / 24.0)
    );
    let days = hours_f / 24.0;
    eprintln!(
        "  wall {wall_s:.2} s for {days:.0} d -> {:.1} simulated days/s (glue {:.1} %)",
        days / wall_s,
        100.0 * (wall_s - total) / wall_s
    );
}

fn run_days(sim: &mut Simulation, days: u64) -> f64 {
    let t = Instant::now();
    for _ in 0..days * 24 {
        sim.step_hour();
    }
    t.elapsed().as_secs_f64()
}

#[test]
#[ignore = "benchmark — cargo test --release --test perf_full_breakdown -- --ignored --nocapture"]
fn perf_full_breakdown() {
    let radius = i32::try_from(env_u64("HEXSIM_PERF_RADIUS", 45)).expect("radius i32");
    let years = env_u64("HEXSIM_PERF_YEARS", 2);
    let seed = u32::try_from(env_u64("HEXSIM_PERF_SEED", 42)).expect("seed u32");
    let save_dir = std::env::var("HEXSIM_PERF_SAVE_DIR").ok();

    let mut sim = live_like_sim(radius, seed);
    eprintln!(
        "world r{radius} seed {seed} ({} cells), {years} annual window(s)",
        sim.grid().len()
    );

    // "Young" window: 5 days of spin-up (humidity/cloud priming) then 30 days.
    run_days(&mut sim, 5);
    sim.reset_phase_timings();
    let wall = run_days(&mut sim, MEASURE_DAYS);
    print_window("young world (d5-d35)", &sim, wall);

    for y in 1..=years {
        // Advance to the start of year y (measurement included in the age).
        let target_days = y * 365;
        let done_days = sim.hour_tick() / 24;
        if target_days > done_days {
            run_days(&mut sim, target_days - done_days);
        }
        if let Some(dir) = &save_dir {
            let bytes = sim.save_state().expect("save_state");
            let path = format!("{dir}/world_y{y}.ckpt");
            std::fs::write(&path, bytes).expect("write checkpoint");
            eprintln!("checkpoint -> {path}");
        }
        sim.reset_phase_timings();
        let wall = run_days(&mut sim, MEASURE_DAYS);
        print_window(&format!("year {y}"), &sim, wall);
    }
}
