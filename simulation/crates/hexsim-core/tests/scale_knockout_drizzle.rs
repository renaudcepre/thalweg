//! Knock-out test for the planetary drizzle diagnostic (issue #63).
//!
//! History: Phase 3 of milestone #63 tested 3 orographic parameters
//! (`uplift_orographic_coef`, `orographic_lift_coef`, `orographic_boost`).
//! Result: 2 of them were mute in every config tested (sensitivity x0 and
//! x10 on 2 different seeds), confirmed by #66, removed from the engine.
//! `orographic_lift_coef` remains.
//!
//! This test measures the effect of knocking out `orographic_lift_coef`:
//! - rain/year: rainy days per cell (target: decreasing with elevation)
//! - `cloud_mean`: average `cloud_water` stock (target: not saturated at
//!   altitude)
//! - `wet_max`: longest consecutive rain streak on a cell in the band
//!
//! Expected conclusion (Phase 3): this parameter is
//! REDISTRIBUTIVE; knocking it out moves the drizzle from the plains to the
//! peaks, without creating or eliminating it. The real culprit is in the
//! condensation / `cloud_evap` chain (to investigate in Phase 4 on a
//! separate branch).
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test scale_knockout_drizzle \
//!     -- --ignored --nocapture
//! ```

use std::collections::HashMap;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
/// Shorter than `scale_knockout_humidity` (5 years) since we multiply by 5
/// scenarios. 2 years is enough for the drizzle signature (cf baseline #63).
const YEARS: u64 = 2;
const TOTAL_DAYS: u64 = YEARS * 365;
const RAIN_THRESHOLD: f32 = 1e-5;

/// Same bands as `scale_knockout_humidity` and `scale_drizzle_by_altitude`.
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

#[derive(Debug, Clone)]
struct BandStat {
    name: &'static str,
    cells: usize,
    rain_per_year: f32,
    cloud_mean: f32,
    wet_max: usize,
}

fn to_f32(n: usize) -> f32 {
    f32::from(u16::try_from(n).expect("count fits u16"))
}

fn build_sim_seeded(seed: u32, atmo: AtmosphereParams) -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    let wind = WindParams {
        seed,
        ..WindParams::default()
    };
    Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind,
    )
}

fn run_scenario(label: &str, atmo: AtmosphereParams) -> Vec<BandStat> {
    run_scenario_seeded(label, SEED, atmo)
}

fn run_scenario_seeded(label: &str, seed: u32, atmo: AtmosphereParams) -> Vec<BandStat> {
    eprintln!("\n--- scenario: {label} (seed {seed}) ---");
    let t0 = std::time::Instant::now();

    let mut sim = build_sim_seeded(seed, atmo);

    let coords_elev: Vec<(HexCoord, f32)> = sim
        .grid()
        .iter()
        .map(|(c, cell)| (*c, cell.elevation))
        .collect();

    let total_days = usize::try_from(TOTAL_DAYS).expect("fits usize");
    let mut rain_history: HashMap<HexCoord, Vec<bool>> = coords_elev
        .iter()
        .map(|(c, _)| (*c, Vec::with_capacity(total_days)))
        .collect();
    let mut cloud_sum: HashMap<HexCoord, f32> =
        coords_elev.iter().map(|(c, _)| (*c, 0.0_f32)).collect();

    for _day in 0..TOTAL_DAYS {
        sim.step();
        let grid = sim.grid();
        let coords = grid.coords_slice();

        for (coord, cell) in grid.iter() {
            *cloud_sum.get_mut(coord).expect("init above") += cell.cloud_water;
        }
        for (i, rec) in sim.last_precipitation().iter().enumerate() {
            rain_history
                .get_mut(&coords[i])
                .expect("init above")
                .push(rec.rain > RAIN_THRESHOLD);
        }
    }

    let total_days_f = to_f32(total_days);
    let mut out = Vec::new();
    for (name, lo, hi) in BANDS {
        let cells: Vec<HexCoord> = coords_elev
            .iter()
            .filter(|(_, e)| *e >= *lo && *e < *hi)
            .map(|(c, _)| *c)
            .collect();
        if cells.is_empty() {
            continue;
        }
        let n = cells.len();
        let n_f = to_f32(n);

        let mut rain_per_year_sum = 0.0_f32;
        let mut wet_max_global = 0_usize;
        for c in &cells {
            let Some(hist) = rain_history.get(c) else {
                continue;
            };
            let rainy = hist.iter().filter(|&&b| b).count();
            rain_per_year_sum += to_f32(rainy) * 365.0 / total_days_f;

            let mut cur = 0_usize;
            let mut local_max = 0_usize;
            for &b in hist {
                if b {
                    cur += 1;
                    local_max = local_max.max(cur);
                } else {
                    cur = 0;
                }
            }
            wet_max_global = wet_max_global.max(local_max);
        }
        let cloud_mean = cells
            .iter()
            .filter_map(|c| cloud_sum.get(c).copied())
            .sum::<f32>()
            / (n_f * total_days_f);

        out.push(BandStat {
            name,
            cells: n,
            rain_per_year: rain_per_year_sum / n_f,
            cloud_mean,
            wet_max: wet_max_global,
        });
    }

    eprintln!("  scenario finished in {:.1}s", t0.elapsed().as_secs_f64());
    out
}

fn print_metric_table(
    metric_name: &str,
    fmt: &str,
    scenarios: &[(String, Vec<BandStat>)],
    extract: impl Fn(&BandStat) -> f32,
) {
    eprintln!("\n=== {metric_name} ===");
    eprint!("  {:>10}", "band");
    for (label, _) in scenarios {
        eprint!("  {label:>14}");
    }
    eprintln!();
    eprintln!("  {}", "-".repeat(10 + 16 * scenarios.len()));
    let n_bands = scenarios[0].1.len();
    for band_idx in 0..n_bands {
        let name = scenarios[0].1[band_idx].name;
        let cells = scenarios[0].1[band_idx].cells;
        eprint!("  {name:>10}");
        for (_, stats) in scenarios {
            let v = stats.get(band_idx).map_or(0.0, &extract);
            // formatted print with the requested format string
            // "{:>14.X}" handled inline for each metric
            match fmt {
                "f1" => eprint!("  {v:>14.1}"),
                "f4" => eprint!("  {v:>14.4}"),
                _ => eprint!("  {v:>14}"),
            }
        }
        eprintln!("   ({cells} cells)");
    }
}

fn print_comparison(scenarios: &[(String, Vec<BandStat>)]) {
    eprintln!(
        "\n\n=== Planetary drizzle (#63), knockout orographic pumps (seed {SEED}, {YEARS} years) ==="
    );

    print_metric_table(
        "rain/year (effective rain days per cell)",
        "f1",
        scenarios,
        |s| s.rain_per_year,
    );
    print_metric_table(
        "cloud_mean (mm LWP, averaged time+space)",
        "f4",
        scenarios,
        |s| s.cloud_mean,
    );
    print_metric_table(
        "wet_max (longest consecutive rain streak on a cell, days)",
        "f1",
        scenarios,
        |s| to_f32(s.wet_max),
    );

    eprintln!(
        "\nReading (Phase 3):
  - lift_coef=0 makes cloud_mean and wet_max DROP above 800m toward plain values,
    BUT makes them EXPLODE in the plains (rain/year <0m doubles, wet_max <0m rises to TOTAL_DAYS).
  - REDISTRIBUTIVE mechanism, not a creator. The drizzle doesn't come from orography.
  - Real culprit to look for in the condensation_rate / cloud_evap_rate chain (Phase 4).
"
    );
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn drizzle_knockout_oro_pumps() {
    let baseline = AtmosphereParams::default();

    // Only one orographic pump remains after the #66 cleanup:
    // `orographic_lift_coef` (used in 2 places via the same parameter).
    // The knockout shows this mechanism is REDISTRIBUTIVE: it moves the
    // drizzle from the plains to the peaks, but doesn't create it. The real
    // culprit for the drizzle is elsewhere (condensation/cloud_evap chain).
    let scenarios = vec![
        (
            "baseline".to_string(),
            run_scenario("baseline (drizzle)", baseline.clone()),
        ),
        (
            "lift_coef=0".to_string(),
            run_scenario(
                "orographic_lift_coef=0 (coupe convection oro + lift advection)",
                AtmosphereParams {
                    orographic_lift_coef: 0.0,
                    ..baseline.clone()
                },
            ),
        ),
    ];

    print_comparison(&scenarios);
}
