//! Knock-out test to diagnose the effect of temperature advection.
//!
//! Context: thermal relaxation (`temperature::step_temperature`,
//! `thermal_coupling=1.0` (default)) forces each cell toward its
//! altitude/season equilibrium at every tick. Directional advection by the
//! wind (`advect_temperature_by_wind_into`, `temperature_advection_rate=0.07`)
//! creates lateral anomalies that relaxation tries to erase within 10
//! ticks. This test measures whether advection has a NET observable effect
//! on the average climate and on the seasonal amplitude per altitude band,
//! once the 2-layer + 3-stock atmo model is in place.
//!
//! Invoke with:
//!   `cargo test --release -p hexsim-core --test scale_knockout_temperature -- --ignored --nocapture`

mod common;

use std::collections::HashMap;

use common::PerfTimer;
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
const YEARS: u64 = 3;
const YEAR: u64 = 365;
const TOTAL_TICKS: u64 = YEARS * YEAR;

#[derive(Debug, Clone)]
struct TempBandStat {
    name: &'static str,
    cells: usize,
    /// Average temperature across all ticks x all cells.
    t_mean: f32,
    /// Average seasonal amplitude: average of (`t_max` - `t_min`) per cell.
    t_amplitude_mean: f32,
    /// Spatial heterogeneity: temporal average of the intra-band stddev.
    t_stddev_spatial: f32,
}

const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

fn to_f32(n: usize) -> f32 {
    f32::from(u16::try_from(n).expect("fits u16"))
}

fn build_sim(wind: WindParams) -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed: SEED,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind,
    )
}

fn run_scenario(label: &str, wind: WindParams) -> Vec<TempBandStat> {
    eprintln!("\n--- scenario: {label} ---");
    let t0 = std::time::Instant::now();

    let mut sim = build_sim(wind);

    let elevations: HashMap<HexCoord, f32> = sim
        .grid()
        .iter()
        .map(|(c, cell)| (*c, cell.elevation))
        .collect();

    let cells_per_band: Vec<Vec<HexCoord>> = BANDS
        .iter()
        .map(|(_, lo, hi)| {
            elevations
                .iter()
                .filter(|(_, e)| **e >= *lo && **e < *hi)
                .map(|(c, _)| *c)
                .collect()
        })
        .collect();

    let mut t_sum_per_cell: HashMap<HexCoord, f32> = HashMap::new();
    let mut t_min_per_cell: HashMap<HexCoord, f32> = HashMap::new();
    let mut t_max_per_cell: HashMap<HexCoord, f32> = HashMap::new();

    // Sum of tick-by-tick spatial stddevs, per band.
    let mut band_stddev_sum: Vec<f32> = vec![0.0; BANDS.len()];
    let mut band_stddev_count: Vec<u16> = vec![0; BANDS.len()];

    for _tick in 0..TOTAL_TICKS {
        sim.step();
        let grid = sim.grid();
        for (coord, cell) in grid.iter() {
            let t = cell.temperature;
            *t_sum_per_cell.entry(*coord).or_default() += t;
            let min_v = t_min_per_cell.entry(*coord).or_insert(f32::INFINITY);
            if t < *min_v {
                *min_v = t;
            }
            let max_v = t_max_per_cell.entry(*coord).or_insert(f32::NEG_INFINITY);
            if t > *max_v {
                *max_v = t;
            }
        }

        for (band_idx, cells) in cells_per_band.iter().enumerate() {
            if cells.len() < 2 {
                continue;
            }
            let temps: Vec<f32> = cells
                .iter()
                .filter_map(|c| grid.get(*c).map(|cell| cell.temperature))
                .collect();
            let n_f = to_f32(temps.len());
            let mean = temps.iter().sum::<f32>() / n_f;
            let var = temps.iter().map(|t| (t - mean).powi(2)).sum::<f32>() / n_f;
            band_stddev_sum[band_idx] += var.sqrt();
            band_stddev_count[band_idx] += 1;
        }
    }

    let mut out = Vec::new();
    for (band_idx, (name, _, _)) in BANDS.iter().enumerate() {
        let cells = &cells_per_band[band_idx];
        if cells.is_empty() {
            continue;
        }
        let n_f = to_f32(cells.len());
        let total_ticks_f = f32::from(u16::try_from(TOTAL_TICKS).expect("fits u16"));

        let t_mean = cells
            .iter()
            .filter_map(|c| t_sum_per_cell.get(c).copied())
            .sum::<f32>()
            / (n_f * total_ticks_f);

        let t_amplitude_mean = cells
            .iter()
            .filter_map(|c| {
                t_max_per_cell
                    .get(c)
                    .copied()
                    .zip(t_min_per_cell.get(c).copied())
                    .map(|(max, min)| max - min)
            })
            .sum::<f32>()
            / n_f;

        let t_stddev_spatial = if band_stddev_count[band_idx] > 0 {
            band_stddev_sum[band_idx] / f32::from(band_stddev_count[band_idx])
        } else {
            0.0
        };

        out.push(TempBandStat {
            name,
            cells: cells.len(),
            t_mean,
            t_amplitude_mean,
            t_stddev_spatial,
        });
    }

    eprintln!(
        "  {:.2}s for {} ticks",
        t0.elapsed().as_secs_f64(),
        TOTAL_TICKS
    );
    out
}

fn print_row<F: Fn(&TempBandStat) -> f32>(
    label: &str,
    scenarios: &[(String, Vec<TempBandStat>)],
    extract: F,
    width: usize,
    precision: usize,
) {
    eprint!("  {:>10}", "band");
    for (scen_label, _) in scenarios {
        eprint!("  {scen_label:>width$}");
    }
    eprintln!();
    for band_idx in 0..scenarios[0].1.len() {
        let name = scenarios[0].1[band_idx].name;
        let cells = scenarios[0].1[band_idx].cells;
        eprint!("  {name:>10}");
        for (_, stats) in scenarios {
            let v = stats.get(band_idx).map_or(0.0, &extract);
            eprint!("  {v:>width$.precision$}");
        }
        eprintln!("   ({cells} cells)");
    }
    eprintln!("  ({label})");
}

fn print_comparison(scenarios: &[(String, Vec<TempBandStat>)]) {
    eprintln!("\n=== Mean annual T (C) ===");
    print_row("t_mean", scenarios, |s| s.t_mean, 12, 2);
    eprintln!("\n=== Mean seasonal amplitude (C) = <t_max - t_min> per cell ===");
    print_row("t_amplitude_mean", scenarios, |s| s.t_amplitude_mean, 12, 2);
    eprintln!("\n=== Spatial heterogeneity (C) = <stddev(temp)> over the band per tick ===");
    print_row("t_stddev_spatial", scenarios, |s| s.t_stddev_spatial, 12, 3);
}

#[test]
#[ignore = "run explicitly: cargo test ... -- --ignored --nocapture"]
fn scale_knockout_temperature() {
    let mut timer = PerfTimer::start("scale_knockout_temperature");

    let baseline_wind = WindParams::default();

    let scenarios = vec![
        (
            "defaults".to_string(),
            run_scenario("defaults (temp_adv=0.07)", baseline_wind.clone()),
        ),
        (
            "adv=0".to_string(),
            run_scenario(
                "temperature_advection=0 (knockout)",
                WindParams {
                    temperature_advection_rate: 0.0,
                    ..baseline_wind.clone()
                },
            ),
        ),
        (
            "adv=0.21".to_string(),
            run_scenario(
                "temperature_advection=0.21 (3x)",
                WindParams {
                    temperature_advection_rate: 0.21,
                    ..baseline_wind.clone()
                },
            ),
        ),
    ];

    timer.lap("simulations");
    timer.ticks(TOTAL_TICKS * u64::try_from(scenarios.len()).expect("fits u64"));

    print_comparison(&scenarios);
    timer.report();
}
