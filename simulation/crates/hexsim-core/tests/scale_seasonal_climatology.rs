//! Seasonal climate contract: plains and mid-mountain in summer and winter.
//!
//! Philosophy: these tests do NOT validate a local physical rule, they
//! characterize the global climate produced by the simulation. The world
//! must roughly resemble a temperate continental climate:
//!
//! Summer (July-August):
//! - plains (<200m): average `T_max` >= 28 C
//! - high mountain (>=1500m): average `T_max` in [19, 25] C (peaks
//!   stay cool even in a hot continental climate; upper bound set on
//!   plain - lapse*altitude with margin for a limited sample)
//!
//! Winter (January):
//! - plains (<200m): January `T_mean` in [-10, 5] C
//! - altitude (>=1000m): January `T_mean` of the band < 0 C. Until
//!   2026-09-03 this was "no cell above 1000 m ever exceeds 0 C", which
//!   the owner ruled out as physically wrong: a sunny south-facing slope
//!   at 1100 m thaws on a fine January afternoon in the Drôme, the band
//!   as a whole stays below freezing on the monthly mean.
//!
//! If a test fails after a change, two possibilities:
//! 1. The change unintentionally breaks the energy budget -> bug, to fix.
//! 2. The change intentionally shifts the climatology -> adjust the
//!    thresholds here AND document why in the commit.
//!
//! Warm-up 1 year: temperature converges in ~30 ticks
//!   (`thermal_coupling=1.0` (default)), so one year is plenty to
//!   eliminate the transient from the uniform initial T.

mod common;

use std::collections::HashMap;

use common::build_prod_sim;
use hexsim_core::coord::HexCoord;
use hexsim_core::simulation::Simulation;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const YEAR: u64 = 365;
const WARMUP_TICKS: u64 = YEAR;

const PLAIN_MAX_ELEV: f32 = 200.0;
/// High mountain: summits, cool summers. Upper bound open.
const SUMMER_HIGH_MIN: f32 = 1500.0;
/// Altitude that doesn't thaw in January (mid/high mountain).
const HIGH_MIN: f32 = 1000.0;

/// July + August: calendar tick 0 = January 1st (cf. temperature.rs).
const SUMMER_START: u64 = 181;
const SUMMER_END: u64 = 242;
/// January: 31 days.
const WINTER_START: u64 = 0;
const WINTER_END: u64 = 30;

fn run_warmup() -> Simulation {
    let mut sim = build_prod_sim(SEED, RADIUS);
    for _ in 0..WARMUP_TICKS {
        sim.step();
    }
    sim
}

#[test]
fn summer_temperature_targets_plain_and_mountain() {
    let mut sim = run_warmup();

    // Year 2, July-August window: T_max per cell.
    let mut t_max_per_cell: HashMap<HexCoord, f32> = HashMap::new();
    for tick_local in 0..YEAR {
        sim.step();
        if !(SUMMER_START..=SUMMER_END).contains(&tick_local) {
            continue;
        }
        for (coord, cell) in sim.grid().iter() {
            t_max_per_cell
                .entry(*coord)
                .and_modify(|v| *v = v.max(cell.temperature))
                .or_insert(cell.temperature);
        }
    }

    let mut plain: Vec<f32> = Vec::new();
    let mut high: Vec<f32> = Vec::new();
    for (coord, &t) in &t_max_per_cell {
        let Some(cell) = sim.grid().get(*coord) else {
            continue;
        };
        if cell.elevation < PLAIN_MAX_ELEV {
            plain.push(t);
        } else if cell.elevation >= SUMMER_HIGH_MIN {
            high.push(t);
        }
    }

    assert!(
        !plain.is_empty(),
        "plain band empty (seed {SEED}, radius {RADIUS})"
    );
    assert!(
        !high.is_empty(),
        "high mountain band (>={SUMMER_HIGH_MIN} m) empty (seed {SEED}, radius {RADIUS})"
    );

    let mean = |v: &[f32]| {
        v.iter().copied().sum::<f32>() / f32::from(u16::try_from(v.len()).expect("fits u16"))
    };
    let plain_mean = mean(&plain);
    let high_mean = mean(&high);

    eprintln!(
        "\n=== Summer ({} days July-August) ===",
        SUMMER_END - SUMMER_START + 1
    );
    eprintln!(
        "  Plain (<{:.0}m, n={}) T_max mean = {:.1} C (target >= 28.0)",
        PLAIN_MAX_ELEV,
        plain.len(),
        plain_mean
    );
    eprintln!(
        "  High mountain (>={:.0}m, n={}) T_max mean = {:.1} C (target [19, 25])",
        SUMMER_HIGH_MIN,
        high.len(),
        high_mean
    );

    assert!(
        plain_mean >= 28.0,
        "plains too cool in summer: T_max mean {plain_mean:.1} C < 28 C"
    );
    assert!(
        (19.0..=25.0).contains(&high_mean),
        "high mountain out of target: T_max mean {high_mean:.1} C outside [19, 25] C"
    );
}

#[test]
fn winter_temperature_targets_plain_and_mountain() {
    let mut sim = run_warmup();

    // Year 2, January: T_mean per cell (plain) and T_max per cell (altitude).
    let mut t_sum_per_cell: HashMap<HexCoord, f32> = HashMap::new();
    let mut t_count: u32 = 0;
    let mut t_max_per_cell: HashMap<HexCoord, f32> = HashMap::new();
    for tick_local in 0..YEAR {
        sim.step();
        if !(WINTER_START..=WINTER_END).contains(&tick_local) {
            continue;
        }
        t_count += 1;
        for (coord, cell) in sim.grid().iter() {
            *t_sum_per_cell.entry(*coord).or_default() += cell.temperature;
            t_max_per_cell
                .entry(*coord)
                .and_modify(|v| *v = v.max(cell.temperature))
                .or_insert(cell.temperature);
        }
    }
    assert!(t_count > 0, "empty winter window");

    let mut plain_mean: Vec<f32> = Vec::new();
    let mut high_mean: Vec<f32> = Vec::new();
    let mut high_max: Vec<f32> = Vec::new();
    for (coord, &sum) in &t_sum_per_cell {
        let Some(cell) = sim.grid().get(*coord) else {
            continue;
        };
        let mean_t = sum / f32::from(u16::try_from(t_count).expect("fits u16"));
        if cell.elevation < PLAIN_MAX_ELEV {
            plain_mean.push(mean_t);
        } else if cell.elevation >= HIGH_MIN {
            high_mean.push(mean_t);
            let tmax = *t_max_per_cell.get(coord).unwrap();
            high_max.push(tmax);
        }
    }

    assert!(!plain_mean.is_empty(), "plain band empty in winter");
    assert!(
        !high_max.is_empty(),
        "band >={HIGH_MIN}m empty (seed {SEED})"
    );

    let mean = |v: &[f32]| {
        v.iter().copied().sum::<f32>() / f32::from(u16::try_from(v.len()).expect("fits u16"))
    };
    let plain_winter_mean = mean(&plain_mean);
    let high_winter_mean = mean(&high_mean);
    let high_winter_max_mean = mean(&high_max);
    let high_winter_max_peak = high_max.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    eprintln!(
        "\n=== Winter ({} days January) ===",
        WINTER_END - WINTER_START + 1
    );
    eprintln!(
        "  Plain (<{:.0}m, n={}) T_mean = {:.1} C (target [-10, 5])",
        PLAIN_MAX_ELEV,
        plain_mean.len(),
        plain_winter_mean
    );
    eprintln!(
        "  Altitude (>={:.0}m, n={}) T_mean = {:.1} C (target < 0), T_max mean = {:.1} C, \
         individual peak = {:.1} C (informative: a sunny adret thaws)",
        HIGH_MIN,
        high_mean.len(),
        high_winter_mean,
        high_winter_max_mean,
        high_winter_max_peak
    );

    assert!(
        (-10.0..=5.0).contains(&plain_winter_mean),
        "plains out of target in winter: T_mean {plain_winter_mean:.1} C outside [-10, 5] C"
    );
    assert!(
        high_winter_mean < 0.0,
        "altitude >={HIGH_MIN}m: January band mean {high_winter_mean:.1} C is not below freezing"
    );
}
