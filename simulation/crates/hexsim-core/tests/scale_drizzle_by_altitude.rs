//! Exploratory diagnostic for "planetary drizzle" (issue #63).
//!
//! Crosses two dimensions already measured separately elsewhere:
//! - elevation bands (cf `scale_knockout_humidity`)
//! - rain/drought streaks per cell (cf `scale_dry_periods`)
//!
//! Adds a 3rd dimension useful for the investigation: average and peak
//! `cloud_water` per band, to spot whether clouds are stagnating (the
//! visual drizzle signature: "the next cloud does the same thing").
//!
//! **Eval style**: no assert, just a printed table. Read, form a
//! hypothesis, then attack it via a targeted `scale_knockout_*`.
//!
//! **Vocabulary convention** (cf project memory `tick_jours_vs_heures`):
//! this test talks in DAYS for the calendar dimension. `sim.step()` advances
//! 1 day (24 internal calls to [`Simulation::step_hour`]). The word "tick"
//! stays an implementation term and doesn't appear in the exposed metrics.
//!
//! **`cloud_water` sampling**: measured once a day at midnight (after the
//! 24 [`Simulation::step_hour`] calls). Sub-sampled by construction; can
//! miss an intra-day peak. If hourly resolution is needed, switch the
//! inner loop to [`Simulation::step_hour`] at the cost of a ×24 on runtime.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test scale_drizzle_by_altitude \
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
const WARMUP_DAYS: u64 = 90;
const YEARS: u64 = 2;
/// Measurement duration in DAYS (post-warmup). `sim.step()` advances 1 day.
const TOTAL_DAYS: u64 = YEARS * 365;

/// Same bands as `scale_knockout_humidity` for direct comparability.
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

/// Effective precipitation threshold (consistent with `scale_dry_periods`).
const RAIN_THRESHOLD: f32 = 1e-5;

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed: SEED,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    let wind = WindParams {
        seed: SEED,
        ..WindParams::default()
    };
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

/// Aggregated stats for an elevation band.
struct BandReport {
    name: &'static str,
    cells: usize,
    elev_mean: f32,
    /// Days/year with effective rain on the average cell (0..365).
    rain_freq_per_year: f32,
    /// Average `cloud_water` stock (mm LWP, averaged over time and space).
    cloud_water_mean: f32,
    /// 99th spatial percentile of the time-max `cloud_water` (the "big clouds").
    cloud_water_p99: f32,
    /// Median (over the band's cells) of the longest streak WITHOUT rain.
    dry_streak_median: usize,
    dry_streak_p25: usize,
    dry_streak_p75: usize,
    /// Median of the longest CONTINUOUS rain streak.
    wet_streak_median: usize,
    wet_streak_max: usize,
}

/// Safe `usize → f32` conversion via `u16::try_from` + `f32::from`.
/// All sizes handled (band cells ≤ ~2700 for radius 30, days ≤ 730) fit
/// comfortably in `u16`.
fn to_f32(n: usize) -> f32 {
    f32::from(u16::try_from(n).expect("count fits u16"))
}

/// Percentile index via integer arithmetic; avoids any `f32 → usize` cast.
/// `p_promille` ∈ [0, 1000] (e.g. 500 = median, 990 = p99).
fn percentile_idx(len: usize, p_promille: u32) -> usize {
    if len == 0 {
        return 0;
    }
    let p = usize::try_from(p_promille).expect("u32 fits usize");
    ((len - 1) * p / 1000).min(len - 1)
}

fn compute_band_report(
    name: &'static str,
    cells: &[(HexCoord, f32)],
    rain_history: &HashMap<HexCoord, Vec<bool>>,
    cloud_history_sum: &HashMap<HexCoord, f32>,
    cloud_max_per_cell: &HashMap<HexCoord, f32>,
) -> Option<BandReport> {
    if cells.is_empty() {
        return None;
    }
    let n = cells.len();
    let n_f = to_f32(n);
    let total_days = usize::try_from(TOTAL_DAYS).expect("fits usize");
    let total_days_f = to_f32(total_days);

    let elev_mean = cells.iter().map(|(_, e)| *e).sum::<f32>() / n_f;

    let mut rain_freq_sum = 0.0_f32;
    let mut dry_streak_max_per_cell: Vec<usize> = Vec::with_capacity(n);
    let mut wet_streak_max_per_cell: Vec<usize> = Vec::with_capacity(n);

    for (coord, _) in cells {
        let Some(hist) = rain_history.get(coord) else {
            continue;
        };
        let rainy_days = hist.iter().filter(|&&b| b).count();
        rain_freq_sum += to_f32(rainy_days) * 365.0 / total_days_f;

        let mut max_dry = 0_usize;
        let mut cur_dry = 0_usize;
        let mut max_wet = 0_usize;
        let mut cur_wet = 0_usize;
        for &raining in hist {
            if raining {
                cur_dry = 0;
                cur_wet += 1;
                max_wet = max_wet.max(cur_wet);
            } else {
                cur_wet = 0;
                cur_dry += 1;
                max_dry = max_dry.max(cur_dry);
            }
        }
        dry_streak_max_per_cell.push(max_dry);
        wet_streak_max_per_cell.push(max_wet);
    }
    dry_streak_max_per_cell.sort_unstable();
    wet_streak_max_per_cell.sort_unstable();

    let cloud_water_mean = cells
        .iter()
        .filter_map(|(c, _)| cloud_history_sum.get(c).copied())
        .sum::<f32>()
        / (n_f * total_days_f);

    let mut cloud_max_sorted: Vec<f32> = cells
        .iter()
        .filter_map(|(c, _)| cloud_max_per_cell.get(c).copied())
        .collect();
    cloud_max_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let dry_p25 = dry_streak_max_per_cell[percentile_idx(dry_streak_max_per_cell.len(), 250)];
    let dry_med = dry_streak_max_per_cell[percentile_idx(dry_streak_max_per_cell.len(), 500)];
    let dry_p75 = dry_streak_max_per_cell[percentile_idx(dry_streak_max_per_cell.len(), 750)];
    let wet_med = wet_streak_max_per_cell[percentile_idx(wet_streak_max_per_cell.len(), 500)];
    let cloud_p99 = if cloud_max_sorted.is_empty() {
        0.0
    } else {
        cloud_max_sorted[percentile_idx(cloud_max_sorted.len(), 990)]
    };

    Some(BandReport {
        name,
        cells: n,
        elev_mean,
        rain_freq_per_year: rain_freq_sum / n_f,
        cloud_water_mean,
        cloud_water_p99: cloud_p99,
        dry_streak_median: dry_med,
        dry_streak_p25: dry_p25,
        dry_streak_p75: dry_p75,
        wet_streak_median: wet_med,
        wet_streak_max: *wet_streak_max_per_cell.last().unwrap_or(&0),
    })
}

fn print_table(reports: &[BandReport]) {
    eprintln!(
        "\n=== Planetary drizzle (#63), diagnostic by elevation band (seed {SEED}, {YEARS} years) ===\n"
    );
    eprintln!(
        "  {:>10} {:>5} {:>8} {:>8} {:>10} {:>10} {:>5}/{:>5}/{:>5} {:>5}/{:>5}",
        "band",
        "cells",
        "elev",
        "rain/an",
        "cloud_mean",
        "cloud_p99",
        "dry25",
        "dryMed",
        "dry75",
        "wetMed",
        "wetMax"
    );
    eprintln!("  {}", "-".repeat(95));
    for r in reports {
        eprintln!(
            "  {:>10} {:>5} {:>8.0} {:>8.1} {:>10.4} {:>10.3} {:>5}/{:>5}/{:>5} {:>5}/{:>5}",
            r.name,
            r.cells,
            r.elev_mean,
            r.rain_freq_per_year,
            r.cloud_water_mean,
            r.cloud_water_p99,
            r.dry_streak_p25,
            r.dry_streak_median,
            r.dry_streak_p75,
            r.wet_streak_median,
            r.wet_streak_max,
        );
    }
    eprintln!(
        "\nLegend (all durations in DAYS):
  rain/an     : days with effective rainfall (>{RAIN_THRESHOLD:.0e}) per year, averaged over the band's cells
  cloud_mean  : mean `cloud_water` stock (mm LWP). Sampled once a day at midnight
                (after the 24 step_hour calls), undersampled, may miss intra-day peaks
  cloud_p99   : 99th percentile (over cells) of the temporal max of `cloud_water` (same sampling limit)
  dryXX       : per cell, longest streak WITHOUT rain (days), p25/median/p75 over cells
  wetMed/Max  : per cell, longest streak WITH consecutive rain (days), median and max
\nExpected reading for a healthy terrarium:
  - rain/an strictly decreasing or flat with elevation (oro = modest bonus, not runaway)
  - cloud_mean increasing with elevation (uplift causes condensation) BUT not saturated
  - dryMed > 21 days (consistent with scale_dry_periods)
  - wetMax reasonable (< 30 days, no rain for 2 consecutive months)
\nExpected drizzle signature (hypothesis):
  - rain/an close to 365 in the high bands (>800m, >1500m)
  - cloud_mean high and stagnant at altitude
  - dryMed close to 0 at altitude
  - wetMax very large (= rain never stops on relief cells)
"
    );
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn drizzle_by_altitude_baseline() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

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
    let mut cloud_history_sum: HashMap<HexCoord, f32> =
        coords_elev.iter().map(|(c, _)| (*c, 0.0_f32)).collect();
    let mut cloud_max_per_cell: HashMap<HexCoord, f32> =
        coords_elev.iter().map(|(c, _)| (*c, 0.0_f32)).collect();

    let t0 = std::time::Instant::now();
    for _day in 0..TOTAL_DAYS {
        sim.step(); // advances 1 day (= 24 internal step_hour calls)
        let grid = sim.grid();
        let coords = grid.coords_slice();

        for (coord, cell) in grid.iter() {
            let entry_sum = cloud_history_sum.get_mut(coord).expect("init above");
            *entry_sum += cell.cloud_water;
            let entry_max = cloud_max_per_cell.get_mut(coord).expect("init above");
            if cell.cloud_water > *entry_max {
                *entry_max = cell.cloud_water;
            }
        }

        for (i, rec) in sim.last_precipitation().iter().enumerate() {
            let coord = coords[i];
            let raining = rec.rain > RAIN_THRESHOLD;
            rain_history
                .get_mut(&coord)
                .expect("init above")
                .push(raining);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    // `f64::from(u32)` is lossless; TOTAL_DAYS=730 and 24×=17520 fit in u32.
    let total_days_f = f64::from(u32::try_from(TOTAL_DAYS).expect("fits u32"));
    let total_hours_f = f64::from(u32::try_from(TOTAL_DAYS * 24).expect("fits u32"));
    eprintln!(
        "  simulation {} days ({} hours) in {:.2}s, {:.1} ms/day, {:.2} ms/hour",
        TOTAL_DAYS,
        TOTAL_DAYS * 24,
        elapsed,
        1000.0 * elapsed / total_days_f,
        1000.0 * elapsed / total_hours_f,
    );

    let mut reports: Vec<BandReport> = Vec::new();
    for (name, lo, hi) in BANDS {
        let cells_in_band: Vec<(HexCoord, f32)> = coords_elev
            .iter()
            .filter(|(_, e)| *e >= *lo && *e < *hi)
            .copied()
            .collect();
        if let Some(r) = compute_band_report(
            name,
            &cells_in_band,
            &rain_history,
            &cloud_history_sum,
            &cloud_max_per_cell,
        ) {
            reports.push(r);
        }
    }

    print_table(&reports);
}
