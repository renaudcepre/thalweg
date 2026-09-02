//! Scale test: a fire must extinguish (issue #92: perpetual fire).
//!
//! Context: on the buggy branch, a fire ignited on a hot summer day burns
//! continuously for **months** (8+ months measured, even at −6.8 °C) without
//! ever extinguishing. Cause: multiplicative combustion (`biomass ×= 0.6/day`)
//! that tends toward 0 without reaching it, recharged by the absolute
//! colonization of vegetation (`0.01 × suit × free`) which runs right before
//! the fire on the same day → positive fixed point above the extinction
//! threshold (`extinguish_fuel_min`).
//!
//! Discriminating metric: the **consecutive burn duration per cell** (mirror
//! of `dry_streak_median_per_cell_days`). A global counter like
//! `cell_days_total` doesn't distinguish "200 short fires" from "200 cells
//! burning without end"; the per-cell streak does.
//!   - healthy fire: a cell is razed in ~1-3 days then extinguishes (short streak).
//!   - bug #92: the same cell stays `fire_intensity=1` for hundreds of days.
//!
//! Two bounds (anti-pattern #5: prove the fix AND don't overcorrect):
//!   - HIGH: no cell burns for more than `MAX_BURN_DAYS` days in a row.
//!     That's the bug: fails on the current branch.
//!   - LOW: the fire starts (`ignitions > 0`) and spreads (cells catch fire
//!     other than by lightning). Prevents a fix that "extinguishes everything".
//!
//! Run with:
//!   `cargo test --release -p hexsim-core --test scale_fire_extinguishes -- --ignored --nocapture`

mod common;

use std::collections::{HashMap, HashSet};

use common::{PerfTimer, build_prod_sim};
use hexsim_core::coord::HexCoord;
use hexsim_core::fire::FireParams;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const YEAR: u64 = 365;
/// Establishes credible vegetation (lowland forest + densified montane
/// climax belt #87) before igniting anything. Fire disabled for the
/// entire warmup (`FireParams::default().enabled == false`).
const WARMUP_YEARS: u64 = 12;
/// Fire observation window. 2 years = two dry summers → guaranteed
/// ignitions, and long enough for a perpetual fire to accumulate a streak
/// >> `MAX_BURN_DAYS`.
const MEASURE_YEARS: u64 = 2;

/// A healthy fire razes a cell in ~1-3 days. Bug #92 makes the same cell
/// burn for 240+ days. Threshold placed wide between the two: < 1 month.
const MAX_BURN_DAYS: u64 = 15;

/// `ignition_rate` used for measurement. The default (`4e-5` ≈ 1 ignition /
/// 3 years across the whole map) is too rare for a test; it's raised for
/// reliable ignitions within the window. The rest of `FireParams` stays at
/// default: it's the **default combustion** being put to the test, not a
/// combustion rigged for the test.
const MEASURE_IGNITION_RATE: f32 = 5.0e-4;

fn u64_len(n: usize) -> u64 {
    u64::try_from(n).expect("cell count fits u64")
}

#[test]
#[ignore = "run explicitement : cargo test ... -- --ignored --nocapture"]
fn scale_fire_extinguishes() {
    let mut timer = PerfTimer::start("scale_fire_extinguishes");
    let mut sim = build_prod_sim(SEED, RADIUS);

    // 1) Warmup: let the vegetation settle in, fire extinguished.
    for _ in 0..(WARMUP_YEARS * YEAR) {
        sim.step();
    }
    timer.lap("warmup");

    // 2) Ignite the fire (defaults + boosted ignition for measurement).
    sim.set_fire_params(FireParams {
        enabled: true,
        ignition_rate: MEASURE_IGNITION_RATE,
        ..FireParams::default()
    });

    // 3) Measure: streak of consecutive burning days per cell.
    let mut cur_streak: HashMap<HexCoord, u64> = HashMap::new();
    let mut max_streak: u64 = 0;
    let mut ever_burned: HashSet<HexCoord> = HashSet::new();

    for _ in 0..(MEASURE_YEARS * YEAR) {
        sim.step();
        for (coord, cell) in sim.grid().iter() {
            if cell.fire_intensity > 1e-3 {
                let s = cur_streak.entry(*coord).or_insert(0);
                *s += 1;
                max_streak = max_streak.max(*s);
                ever_burned.insert(*coord);
            } else {
                cur_streak.insert(*coord, 0);
            }
        }
    }
    timer.lap("measure");

    let fs = sim.fire_stats();
    let n_burned = u64_len(ever_burned.len());
    // Cells taken by spread (~ total burnt minus lightning ignitions).
    let spread_catches = n_burned.saturating_sub(fs.ignitions_total);

    eprintln!("\n=== Fire over {MEASURE_YEARS} year(s) (seed {SEED}, radius {RADIUS}) ===");
    eprintln!("  lightning ignitions       : {}", fs.ignitions_total);
    eprintln!("  cells that burned         : {n_burned}");
    eprintln!("  of which via spread       : {spread_catches}");
    eprintln!("  peak simultaneous fires   : {}", fs.peak_burning);
    eprintln!("  cumulative cell-days      : {}", fs.cell_days_total);
    eprintln!("  MAX consecutive days/cell : {max_streak}  (tolerated < {MAX_BURN_DAYS})");

    // Sanity: without an ignition, the test proves nothing (max_streak=0 would wrongly pass).
    assert!(
        fs.ignitions_total > 0,
        "no fire ignition over the window, increase MEASURE_IGNITION_RATE"
    );
    // LOW bound: the fire must spread, otherwise a trivial "extinguish everything" fix passes.
    assert!(
        spread_catches > 0,
        "fire does not spread ({n_burned} cells for {} ignitions), \
         combustion too fast or spread_rate too low",
        fs.ignitions_total
    );
    // HIGH bound: bug #92. A cell must not burn for months.
    assert!(
        max_streak < MAX_BURN_DAYS,
        "perpetual fire (#92): a cell burned {max_streak} consecutive days \
         (max tolerated {MAX_BURN_DAYS})"
    );

    timer.ticks((WARMUP_YEARS + MEASURE_YEARS) * YEAR);
    timer.report();
}
