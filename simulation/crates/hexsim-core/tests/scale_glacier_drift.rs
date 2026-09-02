//! Validation instrument for issue #56: multi-seed glacier/snow drift.
//!
//! Context: JOURNAL 2026-04-28 documents `glacier_melt_rate` 0.4 → 10.0 and
//! `glacier_threshold` 100 → 1000, calibrated on **a single seed (42), 5
//! years, visual observation "I have more lakes"** - tuning without global
//! metrics. This test adjusts nothing: it
//! measures. It's the validation gate for milestone #60 (SI overhaul of
//! snow): any change to `SnowParams` or the snow model must be re-measured
//! with this test BEFORE and AFTER the change, and the delta must be
//! justified in the JOURNAL, not just "it looks better".
//!
//! Eval style: measurements and printouts, no target calibration. The
//! only assertions are hard safety nets (NaN/inf, exponential runaway),
//! never an "expected" range that would turn this test into a tuning
//! target.
//!
//! Protocol per seed: 2 years of warm-up (transient: water tables
//! saturating, snow finding its regime) then 5 years of measurement
//! (5×365 steps, 1 step = 1 day). We sample the grid state on every day
//! of measurement.
//!
//! One `#[test]` per seed (`glacier_drift_seed_*`) so nextest
//! parallelizes them; each one calls the common `run_drift` helper. File
//! prefixed `scale_`: excluded from the default suite
//! (`.config/nextest.toml`), only runs via `just test-all` or explicitly.

mod common;

use std::collections::HashSet;

use common::build_prod_sim;
use hexsim_core::coord::HexCoord;
use hexsim_core::hydro::total_water;
use hexsim_core::snow::total_snow;

const RADIUS: i32 = 30;
const YEAR: u64 = 365;
/// Year ignored at the start of the sim (saturation transient).
// History of window artifacts caught by this warm-up:
// - 1 → 2 years (#60 Phase 4): the SI seasonal regime takes ~2 years to
//   fill (year 1 = 9.7k, year 2 = 24.8k, then a +0.2%/year plateau over
//   10 years, seed 42 r30). At 1 year the instrument showed "+31%/year".
// - 2 → 3 years (#104): the SI transition of effective_elevation shifts
//   the equilibrium (snowpack ~24.7k → ~10.6k, surface water ~7k →
//   ~22.2k) and the transition takes a full 2 years. At 2 years the
//   instrument showed "-1.8 to -2.4%/year"; the 15-year probe
//   (`snow_transient_seed42_15y`, diag_units_10y_multiseed) shows a FLAT
//   plateau from year 3 to year 15 (10.6k ± 1%). Measure from year 3
//   onward.
const WARMUP_TICKS: u64 = 3 * YEAR;
/// Measurement years after the warm-up.
const MEASURE_YEARS: u64 = 5;
const MEASURE_TICKS: u64 = MEASURE_YEARS * YEAR;
const MEASURE_YEARS_F32: f32 = 5.0;

/// Threshold (mm) above which a cell is counted as "perennial snow" by
/// THIS instrument, deliberately independent of the model's internal
/// glacier threshold (`glacier_threshold`), which milestone #60 is
/// precisely going to remove. The instrument must survive that removal.
const PERENNIAL_THRESHOLD_MM: f32 = 100.0;

/// Anti-exponential-explosion safety net. Very generous (×100 + additive
/// margin): it must never trigger on a healthy model, it only catches a
/// clear-cut numerical divergence.
const RUNAWAY_FACTOR: f32 = 100.0;
const RUNAWAY_OFFSET: f32 = 1000.0;

/// Glacier/snow drift report for a given seed, over 1 year of warm-up +
/// 5 years of measurement.
struct DriftReport {
    seed: u32,
    snow_total_after_warmup: f32,
    snow_total_end: f32,
    drift_mm_per_year: f32,
    drift_pct_per_year: f32,
    perennial_snow_cells: usize,
    max_snow_mm: f32,
    surface_water_after_warmup: f32,
    surface_water_end: f32,
    water_delta: f32,
}

/// Runs 1 year of warm-up + 5 years of measurement for `seed`, measures
/// the drift of the snow stock and surface water as well as perennial
/// snow, prints a summary line, and checks the hard invariants.
fn run_drift(seed: u32) -> DriftReport {
    let mut sim = build_prod_sim(seed, RADIUS);

    for _ in 0..WARMUP_TICKS {
        sim.step();
    }
    let snow_total_after_warmup = total_snow(sim.grid());
    let surface_water_after_warmup = total_water(sim.grid());

    // Cells still above the "perennial snow" threshold on ALL daily
    // samples of the last measurement year: a rolling intersection over
    // the days.
    let mut perennial_candidates: Option<HashSet<HexCoord>> = None;
    let last_year_start = MEASURE_TICKS - YEAR;

    for day in 0..MEASURE_TICKS {
        sim.step();
        if day >= last_year_start {
            let above_threshold: HashSet<HexCoord> = sim
                .grid()
                .iter()
                .filter(|(_, cell)| cell.snow_level > PERENNIAL_THRESHOLD_MM)
                .map(|(coord, _)| *coord)
                .collect();
            perennial_candidates = Some(match perennial_candidates {
                None => above_threshold,
                Some(prev) => prev.intersection(&above_threshold).copied().collect(),
            });
        }
    }

    let snow_total_end = total_snow(sim.grid());
    let surface_water_end = total_water(sim.grid());
    let max_snow_mm = sim
        .grid()
        .iter()
        .map(|(_, cell)| cell.snow_level)
        .fold(0.0_f32, f32::max);
    let perennial_snow_cells = perennial_candidates.map_or(0, |cells| cells.len());

    let drift_mm_per_year = (snow_total_end - snow_total_after_warmup) / MEASURE_YEARS_F32;
    let drift_pct_per_year = if snow_total_after_warmup > 0.0 {
        100.0 * drift_mm_per_year / snow_total_after_warmup
    } else {
        0.0
    };
    let water_delta = surface_water_end - surface_water_after_warmup;

    let report = DriftReport {
        seed,
        snow_total_after_warmup,
        snow_total_end,
        drift_mm_per_year,
        drift_pct_per_year,
        perennial_snow_cells,
        max_snow_mm,
        surface_water_after_warmup,
        surface_water_end,
        water_delta,
    };

    eprintln!(
        "[glacier_drift seed={}] snow_pw={:.1} snow_end={:.1} drift={:.2}mm/an ({:+.2}%/an) \
         perennial={} max={:.1}mm water_delta={:+.1}",
        report.seed,
        report.snow_total_after_warmup,
        report.snow_total_end,
        report.drift_mm_per_year,
        report.drift_pct_per_year,
        report.perennial_snow_cells,
        report.max_snow_mm,
        report.water_delta,
    );

    // --- Hard assertions only: never NaN/inf, never exponential
    // runaway, never a negative stock. No "expected" range: this is not
    // a calibration test.
    assert!(
        report.snow_total_after_warmup.is_finite(),
        "seed {seed}: snow_total_after_warmup not finite ({})",
        report.snow_total_after_warmup
    );
    assert!(
        report.snow_total_end.is_finite(),
        "seed {seed}: snow_total_end not finite ({})",
        report.snow_total_end
    );
    assert!(
        report.surface_water_after_warmup.is_finite(),
        "seed {seed}: surface_water_after_warmup not finite ({})",
        report.surface_water_after_warmup
    );
    assert!(
        report.surface_water_end.is_finite(),
        "seed {seed}: surface_water_end not finite ({})",
        report.surface_water_end
    );
    assert!(
        report.snow_total_end < RUNAWAY_FACTOR * (report.snow_total_after_warmup + RUNAWAY_OFFSET),
        "seed {seed}: snow runaway, snow_total_end={} >> snow_total_after_warmup={}",
        report.snow_total_end,
        report.snow_total_after_warmup
    );
    assert!(
        report.snow_total_end >= 0.0,
        "seed {seed}: snow_total_end negative ({})",
        report.snow_total_end
    );

    report
}

#[test]
fn glacier_drift_seed_42() {
    run_drift(42);
}

#[test]
fn glacier_drift_seed_7() {
    run_drift(7);
}

#[test]
fn glacier_drift_seed_99() {
    run_drift(99);
}

#[test]
fn glacier_drift_seed_1234() {
    run_drift(1234);
}

#[test]
fn glacier_drift_seed_2026() {
    run_drift(2026);
}
