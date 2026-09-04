//! Structural diagnostic: the same physics on the same seed, with the
//! procedural relief stretched horizontally by a factor `k`.
//!
//! `generate_terrain` works in cell units (noise frequencies per cell,
//! `elevation_scale` in metres): when `CELL_SPACING_M` went from
//! 1074.569 m to 130 m (d6be105, 2026-07-09) every landscape got
//! 8.27x narrower for the same height, i.e. 8.27x steeper, and nothing
//! rescaled the generator. This tool measures what that costs the
//! climate: `k = 1` is today's world, `k = 8.27` is the calibration-era
//! relief on the 130 m grid (a Drôme-like hillside), the values in
//! between are candidate compromises. Same metrics as `hexsim-bench`
//! (r30, warmup 1 y, measure 2 y).
//!
//! Eval style: `#[ignore]`, `eprintln!`, no assert. Env `HEXSIM_DIAG_SEED`.
//!
//! ```text
//! cargo nextest run --release -p hexsim-core --run-ignored only \
//!     -E 'binary(diag_terrain_scale_climate)' --no-capture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::bench_metrics::MetricsAccumulator;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const WARMUP: u64 = 365;
const MEASURE: u64 = 730;

fn env_seed() -> u32 {
    std::env::var("HEXSIM_DIAG_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(42)
}

/// Terrain whose horizontal wavelengths are `k` times longer than the
/// default (same seed, same `elevation_scale`).
fn stretched_terrain(seed: u32, k: f64) -> TerrainParams {
    let d = TerrainParams::default();
    TerrainParams {
        seed,
        continent_frequency: d.continent_frequency / k,
        ridge_frequency: d.ridge_frequency / k,
        swiss_frequency: d.swiss_frequency / k,
        permeability_frequency: d.permeability_frequency / k,
        ..d
    }
}

fn run(k: f64) {
    let seed = env_seed();
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(&mut grid, &stretched_terrain(seed, k));
    let slope_mean = {
        let cells = grid.cells_slice();
        let mut sum = 0.0_f64;
        for (i, _) in cells.iter().enumerate() {
            let mut max_delta = 0.0_f32;
            for j in grid.neighbor_indices_toric(i) {
                max_delta = max_delta.max((cells[j].elevation - cells[i].elevation).abs());
            }
            sum += f64::from(
                (max_delta / hexsim_core::dynamics::CELL_SPACING_M)
                    .atan()
                    .to_degrees(),
            );
        }
        sum / f64::from(u32::try_from(cells.len()).expect("fits u32"))
    };
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
    for _ in 0..WARMUP {
        sim.step();
    }
    let mut acc = MetricsAccumulator::start(&sim, MEASURE);
    let t0 = std::time::Instant::now();
    for _ in 0..MEASURE {
        sim.step();
        acc.observe(&sim);
    }
    let (m, status) = acc.finalize(&sim, t0.elapsed());
    eprintln!(
        "terrain_scale k={k:<5} seed {seed} slope_mean={slope_mean:.1}° status={status:?} \
         rain_free={} byAlt={:?} rain_max={} dry_med={:?} plainsP={:.3} sw={:.1} cloudMtn={:.2} lapse={:.2}",
        m.fully_rain_free_days_total,
        m.rain_days_median_by_altitude,
        m.rain_days_max_per_cell,
        m.dry_streak_median_per_cell_days,
        m.plains_precip_mm_per_day,
        m.ratio_precip_summer_winter,
        m.cloud_cover_mountain_pct,
        m.effective_lapse_rate_c_per_km
    );
}

#[test]
#[ignore = "diagnostic tool, run on demand (see module doc)"]
fn terrain_scale_k1_today() {
    run(1.0);
}

#[test]
#[ignore = "diagnostic tool, run on demand (see module doc)"]
fn terrain_scale_k2() {
    run(2.0);
}

#[test]
#[ignore = "diagnostic tool, run on demand (see module doc)"]
fn terrain_scale_k4() {
    run(4.0);
}

#[test]
#[ignore = "diagnostic tool, run on demand (see module doc)"]
fn terrain_scale_k8_calibration_era() {
    run(1074.569 / 130.0);
}
