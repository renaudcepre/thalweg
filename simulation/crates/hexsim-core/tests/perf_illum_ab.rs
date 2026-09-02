//! Paired A/B, reference vs cache, for `compute_illumination` (#65).
//!
//! Both paths run in the SAME process, on the SAME grid state, in
//! alternating order on each repetition, so machine noise (another
//! process pegging a core at 100%) hits both measurements equally
//! instead of biasing a sequential comparison. Since the outputs are
//! bit-identical (`phys_illum_cache_equiv`), the simulated state is the
//! same regardless of which path is measured.
//!
//! `cargo test --release --test perf_illum_ab -- --ignored --nocapture`
//! Env: `HEXSIM_PERF_RADIUS` (45), `HEXSIM_PERF_SEED` (42), `HEXSIM_PERF_REPS` (5).

use std::time::Instant;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::{
    IllumCache, TemperatureParams, compute_illumination, compute_illumination_cached,
    solar_beam_at_tick,
};
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Live world: 5 days of spin-up for a realistic cloud field (the
/// cost of the cloud sample depends on kstar, not the content, but we
/// might as well measure on production state).
fn spun_up_sim(radius: i32, seed: u32) -> Simulation {
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
    for _ in 0..5 * 24 {
        sim.step_hour();
    }
    sim
}

#[test]
#[ignore = "benchmark — cargo test --release --test perf_illum_ab -- --ignored --nocapture"]
fn perf_illum_ab() {
    let radius = i32::try_from(env_u64("HEXSIM_PERF_RADIUS", 45)).expect("radius i32");
    let seed = u32::try_from(env_u64("HEXSIM_PERF_SEED", 42)).expect("seed u32");
    let reps = env_u64("HEXSIM_PERF_REPS", 5);
    let sim = spun_up_sim(radius, seed);

    let temp_params = TemperatureParams::default();
    let cloud_albedo = temp_params.cloud_albedo_coef;
    let cloud_alt = AtmosphereParams::default().upper_layer_altitude_m;
    let grid = sim.grid();
    eprintln!("r{radius} seed {seed} ({} cells), {reps} reps", grid.len());

    let t0 = Instant::now();
    let mut cache = IllumCache::new();
    cache.ensure(grid);
    eprintln!("cache build: {:.1} ms", 1000.0 * t0.elapsed().as_secs_f64());

    let (mut ff, mut il) = (Vec::new(), Vec::new());
    let seasons: &[(u64, &str)] = &[(354, "winter"), (80, "equinox"), (172, "summer")];
    let mut total_ref = 0.0_f64;
    let mut total_new = 0.0_f64;
    eprintln!(
        "{:<10} {:>12} {:>12} {:>8}",
        "season", "ref ms/day", "cache ms/day", "gain"
    );
    for &(day, label) in seasons {
        let mut ref_s = 0.0_f64;
        let mut new_s = 0.0_f64;
        for rep in 0..reps {
            for hour in 0..24_u64 {
                let beam = solar_beam_at_tick(&temp_params, day * 24 + hour);
                // Alternating order every other rep: machine drift spreads out.
                for pass in 0..2 {
                    let measure_ref = (pass == 0) == (rep % 2 == 0);
                    let t = Instant::now();
                    if measure_ref {
                        compute_illumination(
                            grid,
                            &beam,
                            cloud_albedo,
                            cloud_alt,
                            &mut ff,
                            &mut il,
                        );
                    } else {
                        compute_illumination_cached(
                            grid,
                            &beam,
                            cloud_albedo,
                            cloud_alt,
                            &cache,
                            &mut ff,
                            &mut il,
                        );
                    }
                    let dt = t.elapsed().as_secs_f64();
                    if measure_ref {
                        ref_s += dt;
                    } else {
                        new_s += dt;
                    }
                }
            }
        }
        let reps_f = f64::from(u32::try_from(reps).expect("reps < 2^32"));
        let ref_day = 1000.0 * ref_s / reps_f;
        let new_day = 1000.0 * new_s / reps_f;
        eprintln!(
            "{label:<10} {ref_day:>12.2} {new_day:>12.2} {:>7.1}x",
            ref_day / new_day
        );
        total_ref += ref_s;
        total_new += new_s;
    }
    eprintln!(
        "TOTAL      ref {:.1} ms, cache {:.1} ms -> x{:.1}",
        1000.0 * total_ref,
        1000.0 * total_new,
        total_ref / total_new
    );
}
