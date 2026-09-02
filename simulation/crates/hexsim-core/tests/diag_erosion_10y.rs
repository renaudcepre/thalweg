//! Instrument #105 - erosion trajectory over 10 years (r30), eval style.
//!
//! No blind calibration of `k_incision` / `accel_years_per_day` without
//! global metrics at scale - this instrument prints year by year the
//! "done-when" metrics of the issue:
//!   - Gini of discharge EMA (drainage convergence: parallel rills
//!     to dendritic tree = Gini that RISES);
//!   - `river_cells` / `max_discharge` (same definitions as diag:
//!     concentration = fewer river cells, larger);
//!   - bedrock depressions: total, EMERGENT (absent from initial terrain),
//!     emergent IN WATER (surplus > capacity) - future lakes;
//!   - relief: cumulative incised/deposited, transit load, max carving and
//!     max fill per cell;
//!   - hard guards: rock budget AND water budget conserved.
//!
//! Run:
//!   `just diag-tool erosion_10y`
//! Sweep acceleration factor without recompile:
//!   `HEXSIM_EROSION_ACCEL=100 cargo test --release -p hexsim-core \
//!      --test diag_erosion_10y -- --ignored --nocapture`
//! A/B baseline: `HEXSIM_EROSION_ACCEL=0` ~ erosion off (worktree unneeded,
//! switch isolates phenomenon - proven by `phys_erosion.rs`).

mod common;

use std::collections::HashSet;

use common::{build_prod_sim, total_water_budget};
use hexsim_core::erosion::{closed_depression_indices, discharge_gini, total_rock};
use hexsim_core::simulation::Simulation;

/// Year metrics: averages over 37 samples (1 per 10 days) to smooth
/// weather, perennity = in water at EVERY sample.
struct YearMetrics {
    gini_mean: f64,
    river_ema_mean: f64,
    max_dis_ema: f32,
    /// Emergent depressions (outside `depressions0`) in water at each sample.
    perennial: HashSet<usize>,
}

/// Advance sim one year sampling every 10 days.
fn run_year(sim: &mut Simulation, depressions0: &HashSet<usize>) -> YearMetrics {
    let mut gini_sum = 0.0_f64;
    let mut river_ema_sum = 0.0_f64;
    let mut max_dis_ema = 0.0_f32;
    let mut samples = 0.0_f64;
    let mut perennial: Option<HashSet<usize>> = None;
    for day in 1..=365_u64 {
        sim.step();
        if day.is_multiple_of(10) {
            let ema = sim.discharge_ema_map();
            gini_sum += discharge_gini(ema);
            river_ema_sum += f64::from(
                u32::try_from(ema.iter().filter(|&&d| d > RIVER_THRESHOLD).count())
                    .expect("fits u32"),
            );
            max_dis_ema = max_dis_ema.max(ema.iter().copied().fold(0.0_f32, f32::max));
            samples += 1.0;
            let wet_now: HashSet<usize> = {
                let cells = sim.grid().cells_slice();
                closed_depression_indices(sim.grid())
                    .into_iter()
                    .filter(|&i| {
                        !depressions0.contains(&i) && cells[i].water_level > cells[i].water_capacity
                    })
                    .collect()
            };
            perennial = Some(match perennial.take() {
                None => wet_now,
                Some(prev) => prev.intersection(&wet_now).copied().collect(),
            });
        }
    }
    YearMetrics {
        gini_mean: gini_sum / samples,
        river_ema_mean: river_ema_sum / samples,
        max_dis_ema,
        perennial: perennial.unwrap_or_default(),
    }
}

const RADIUS: i32 = 30;
const YEARS: u64 = 10;
/// Same threshold as `river_threshold` in `diagnostics.rs`.
const RIVER_THRESHOLD: f32 = 0.5;

fn erosion_accel_override() -> Option<f32> {
    std::env::var("HEXSIM_EROSION_ACCEL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
}

fn run_seed(seed: u32) {
    let mut sim = build_prod_sim(seed, RADIUS);
    // Erosion LIVE opt-in since one-shot pivot (#105): this instrument
    // measures live mode over 10 years, so explicitly enable it.
    assert!(sim.update_param("erosion.enabled", 1.0));
    if let Some(accel) = erosion_accel_override() {
        if accel <= 0.0 {
            assert!(sim.update_param("erosion.enabled", 0.0));
        } else {
            assert!(sim.update_param("erosion.accel_years_per_day", accel));
        }
    }
    let accel = sim.erosion_params().accel_years_per_day;
    let enabled = sim.erosion_params().enabled;

    let elev0: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation)
        .collect();
    let depressions0: HashSet<usize> = closed_depression_indices(sim.grid()).into_iter().collect();
    let rock0 = total_rock(sim.grid());
    let water0 = total_water_budget(&sim);

    eprintln!(
        "=== #105 / erosion 10 years / seed {seed} (r{RADIUS}, accel {accel} yr/d, enabled {enabled}) ==="
    );
    eprintln!(
        "  initial depressions: {} ; annual averages (37 samples/yr, 1 per 10 d):",
        depressions0.len()
    );
    eprintln!(
        "  year ; gini_ema ; riv_ema ; max_dis_ema ; deprs tot/new/new_lake_peren ; \
         incised ; deposited ; transit ; carved_max ; filled_max"
    );

    for year in 1..=YEARS {
        let metrics = run_year(&mut sim, &depressions0);
        let grid = sim.grid();
        let cells = grid.cells_slice();
        let depressions = closed_depression_indices(grid);
        let emergent: Vec<usize> = depressions
            .iter()
            .copied()
            .filter(|i| !depressions0.contains(i))
            .collect();
        let (incised, deposited) = sim.erosion_totals();
        let in_transit: f64 = cells.iter().map(|c| f64::from(c.sediment_load)).sum();
        let mut carved_max = 0.0_f32;
        let mut filled_max = 0.0_f32;
        for (i, c) in cells.iter().enumerate() {
            let d = c.elevation - elev0[i];
            if -d > carved_max {
                carved_max = -d;
            }
            if d > filled_max {
                filled_max = d;
            }
        }
        eprintln!(
            "  yr {year:>2} ; {:.4} ; {:>5.1} ; {:>6.2} ; {} / {} / {} ; {incised:>8.1} ; \
             {deposited:>8.1} ; {in_transit:>7.1} ; {carved_max:>6.2} m ; {filled_max:>6.2} m",
            metrics.gini_mean,
            metrics.river_ema_mean,
            metrics.max_dis_ema,
            depressions.len(),
            emergent.len(),
            metrics.perennial.len(),
        );
        if year == YEARS {
            for &i in &emergent {
                let c = &cells[i];
                eprintln!(
                    "    emergent depr idx {i}: elev {:.1} m (t0 {:.1}), water {:.1} mm / cap \
                     {:.1}, ema {:.2}, perennial {}",
                    c.elevation,
                    elev0[i],
                    c.water_level,
                    c.water_capacity,
                    sim.discharge_ema_map()[i],
                    metrics.perennial.contains(&i),
                );
            }
        }
    }

    let rock_end = total_rock(sim.grid());
    let water_end = total_water_budget(&sim);
    let rock_drift = (rock_end - rock0) / rock0.abs().max(1.0);
    let water_drift = f64::from(water_end - water0) / f64::from(water0.max(1.0));
    eprintln!(
        "  conservation: rock {rock0:.1} -> {rock_end:.1} ({rock_drift:.2e} rel) ; \
         water {water0:.1} -> {water_end:.1} ({water_drift:.2e} rel)"
    );

    // Hard guards (eval-style: invariants, not targets).
    assert!(rock_drift.abs() < 1e-5, "rock budget not conserved");
    assert!(water_drift.abs() < 1e-3, "water budget not conserved");
}

#[test]
#[ignore = "instrument #105, erosion trajectory 10 years (seed 42, r30)"]
fn erosion_10y_seed42() {
    run_seed(42);
}

#[test]
#[ignore = "instrument #105, seed 7"]
fn erosion_10y_seed7() {
    run_seed(7);
}

#[test]
#[ignore = "instrument #105, seed 99"]
fn erosion_10y_seed99() {
    run_seed(99);
}
