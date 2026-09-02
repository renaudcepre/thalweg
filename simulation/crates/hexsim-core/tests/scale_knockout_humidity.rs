//! Knockout test to diagnose vertical humidity transport.
//!
//! Runs several configs (baseline + 4 single-parameter knockouts),
//! measures climate by altitude band, and prints a comparison table.
//! No assertions: purely exploratory test to identify the mechanism
//! responsible for the inverted precipitation gradient (rain in the
//! plains, dry at altitude).
//!
//! Invoke with: `cargo test --release -p hexsim-core --test scale_knockout_humidity -- --ignored --nocapture`

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
// 5 years is enough to establish the precipitation gradient (shorter
// than scale_ten_year_climatology to keep a reasonable total with 5 runs).
const YEARS: u64 = 5;
const YEAR: u64 = 365;
const TOTAL_TICKS: u64 = YEARS * YEAR;

/// Stats per altitude band after a full simulation run.
#[derive(Debug, Clone)]
struct BandStat {
    name: &'static str,
    cells: usize,
    humidity_mean: f32,
    precip_per_year: f32,
    snow_max: f32,
}

fn build_sim(
    atmo: AtmosphereParams,
    wind: WindParams,
    snow: SnowParams,
    temp: TemperatureParams,
) -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed: SEED,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        snow,
        temp,
        wind,
    )
}

/// Altitude bands: same breakdown as `scale_ten_year_climatology`.
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

fn run_scenario(
    label: &str,
    atmo: AtmosphereParams,
    wind: WindParams,
    snow: SnowParams,
    temp: TemperatureParams,
) -> Vec<BandStat> {
    eprintln!("\n--- scenario: {label} ---");
    let t0 = std::time::Instant::now();

    let mut sim = build_sim(atmo, wind, snow, temp);

    let elevations: HashMap<HexCoord, f32> = sim
        .grid()
        .iter()
        .map(|(c, cell)| (*c, cell.elevation))
        .collect();

    let mut humidity_sum_per_cell: HashMap<HexCoord, f32> = HashMap::new();
    let mut precip_sum_per_cell: HashMap<HexCoord, f32> = HashMap::new();
    let mut snow_max_per_cell: HashMap<HexCoord, f32> = HashMap::new();

    for _tick in 0..TOTAL_TICKS {
        sim.step();
        let grid = sim.grid();
        for (coord, cell) in grid.iter() {
            *humidity_sum_per_cell.entry(*coord).or_default() += cell.humidity_total();
            let e = snow_max_per_cell.entry(*coord).or_default();
            if cell.snow_level > *e {
                *e = cell.snow_level;
            }
        }
        let coords = sim.grid().coords_slice();
        for (i, rec) in sim.last_precipitation().iter().enumerate() {
            *precip_sum_per_cell.entry(coords[i]).or_default() += rec.rain + rec.snow;
        }
    }

    let mut out = Vec::new();
    for (name, lo, hi) in BANDS {
        let cells: Vec<HexCoord> = elevations
            .iter()
            .filter(|(_, e)| **e >= *lo && **e < *hi)
            .map(|(c, _)| *c)
            .collect();
        if cells.is_empty() {
            continue;
        }
        let n = cells.len();
        let n_f = f32::from(u16::try_from(n).expect("fits u16"));
        let total_ticks_f = f32::from(u16::try_from(TOTAL_TICKS).expect("fits u16"));
        let years_f = f32::from(u16::try_from(YEARS).expect("fits u16"));
        let hum_mean = cells
            .iter()
            .filter_map(|c| humidity_sum_per_cell.get(c).copied())
            .sum::<f32>()
            / (n_f * total_ticks_f);
        let precip_annual = cells
            .iter()
            .filter_map(|c| precip_sum_per_cell.get(c).copied())
            .sum::<f32>()
            / (n_f * years_f);
        let snow_max = cells
            .iter()
            .filter_map(|c| snow_max_per_cell.get(c).copied())
            .fold(0.0_f32, f32::max);
        out.push(BandStat {
            name,
            cells: n,
            humidity_mean: hum_mean,
            precip_per_year: precip_annual,
            snow_max,
        });
    }

    eprintln!(
        "  {:.2}s for {} ticks",
        t0.elapsed().as_secs_f64(),
        TOTAL_TICKS
    );
    out
}

fn print_comparison(scenarios: &[(String, Vec<BandStat>)]) {
    eprintln!("\n=== Precipitation comparison table (units/cell/year) ===");
    eprint!("  {:>10}", "band");
    for (label, _) in scenarios {
        eprint!("  {label:>12}");
    }
    eprintln!();
    for band_idx in 0..scenarios[0].1.len() {
        let name = scenarios[0].1[band_idx].name;
        let cells = scenarios[0].1[band_idx].cells;
        eprint!("  {name:>10}");
        for (_, stats) in scenarios {
            let v = stats.get(band_idx).map_or(0.0, |s| s.precip_per_year);
            eprint!("  {v:>12.3}");
        }
        eprintln!("   ({cells} cells)");
    }

    eprintln!("\n=== Humidity mean comparison table ===");
    eprint!("  {:>10}", "band");
    for (label, _) in scenarios {
        eprint!("  {label:>12}");
    }
    eprintln!();
    for band_idx in 0..scenarios[0].1.len() {
        let name = scenarios[0].1[band_idx].name;
        eprint!("  {name:>10}");
        for (_, stats) in scenarios {
            let v = stats.get(band_idx).map_or(0.0, |s| s.humidity_mean);
            eprint!("  {v:>12.4}");
        }
        eprintln!();
    }

    eprintln!("\n=== snow_max comparison table ===");
    eprint!("  {:>10}", "band");
    for (label, _) in scenarios {
        eprint!("  {label:>12}");
    }
    eprintln!();
    for band_idx in 0..scenarios[0].1.len() {
        let name = scenarios[0].1[band_idx].name;
        eprint!("  {name:>10}");
        for (_, stats) in scenarios {
            let v = stats.get(band_idx).map_or(0.0, |s| s.snow_max);
            eprint!("  {v:>12.2}");
        }
        eprintln!();
    }
}

#[test]
#[ignore = "run explicitly: cargo test ... -- --ignored --nocapture"]
fn scale_knockout_humidity() {
    let mut timer = PerfTimer::start("scale_knockout_humidity");

    let baseline_atmo = AtmosphereParams::default();
    let baseline_wind = WindParams::default();
    let baseline_snow = SnowParams::default();
    let baseline_temp = TemperatureParams::default();

    // Default sweep: climate + advection. This test is reusable for
    // each new task by editing this list.
    // Note (v0.2.1 #37): `season_amplitude` removed, seasonality now
    // derives from latitude. To reproduce "low seasonality", lower the
    // latitude (equator = no seasons). 60° = strong boreal seasons.
    let cold_temp = |base: f32, lat: f32| TemperatureParams {
        base_temp: base,
        latitude_deg: lat,
        ..baseline_temp.clone()
    };
    let reduced_advec = |rate: f32| WindParams {
        humidity_advection_rate: rate,
        ..baseline_wind.clone()
    };
    let scenarios = vec![
        (
            "defaults".to_string(),
            run_scenario(
                "defaults (current)",
                baseline_atmo.clone(),
                baseline_wind.clone(),
                baseline_snow.clone(),
                baseline_temp.clone(),
            ),
        ),
        (
            "b=10,lat=50".to_string(),
            run_scenario(
                "base=10, lat=50°N (cool climate, strong seasons)",
                baseline_atmo.clone(),
                baseline_wind.clone(),
                baseline_snow.clone(),
                cold_temp(10.0, 50.0),
            ),
        ),
        (
            "adv=.10".to_string(),
            run_scenario(
                "advec=0.10 (2x less)",
                baseline_atmo.clone(),
                reduced_advec(0.10),
                baseline_snow.clone(),
                baseline_temp.clone(),
            ),
        ),
        (
            "adv=.05".to_string(),
            run_scenario(
                "advec=0.05 (4x less)",
                baseline_atmo.clone(),
                reduced_advec(0.05),
                baseline_snow.clone(),
                baseline_temp.clone(),
            ),
        ),
    ];

    timer.lap("simulations");
    timer.ticks(TOTAL_TICKS * scenarios.len() as u64);

    print_comparison(&scenarios);
    timer.report();
}
