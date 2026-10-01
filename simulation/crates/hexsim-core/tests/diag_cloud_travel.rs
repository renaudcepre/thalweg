//! Diagnostic #110: once the ascent trigger reads a sane `w` (estimator on
//! the ambient synoptic wind, 2026-09-30), do the clouds travel, and what
//! does the trigger do to the rain? Eval style: `#[ignore]`, no assert,
//! dense `eprintln!`; the numbers go to the JOURNAL and to the issue.
//!
//! On a production world (generated terrain, shipped defaults plus the
//! trigger overrides below), over a window of hourly ticks:
//!
//! 1. **Do the clouds travel?** Temporal Pearson correlation of the fine
//!    `cloud_water` field with itself at lags 1 / 24 / 48 h, averaged over
//!    the window. The issue's criterion: 0.99 at 1 h measured 2026-07-12
//!    (frozen), 0.96 at 48 h on the July baseline, 0.35 with the trigger
//!    under weather on the orphaned July branch.
//! 2. **Does rain cover more than its sources?** Daily rain coverage,
//!    fraction of cells with > 0.1 mm in the day (the trace threshold),
//!    mean and max, days above 15 % (`synoptic_front_coverage`'s "visible
//!    rainy passage") and above 50 %.
//! 3. **Texture per hour.** Painted fraction per hour-tick (> 0.1 mm/h,
//!    the overlay's threshold), mean.
//! 4. **Totals.** Rain + snow fallen (mm per cell per day), fully
//!    rain-free days, and the water budget drift over the window
//!    (`Simulation::water_budget_total`, the box must stay closed).
//!
//! Env: `DIAG_RADIUS` (120: at r30 the coarse synoptic torus holds 9
//! cells and no weather system fits, see JOURNAL 2026-07-16), `DIAG_SEED`
//! (42), `DIAG_WARMUP_DAYS` (540: a fresh world is a dry plateau for its
//! first year, #152 — measured here at 60 d of warmup, 0.002 mm/cell/day
//! fell, nothing to correlate; 540 d puts the window in the second
//! summer, where the 2026-09-06 `rain-pattern` probe sat at day 545),
//! `DIAG_MEASURE_DAYS` (60), `DIAG_W_REF` (0 = trigger off, the shipped
//! default), `DIAG_FLOOR` (0.1).
//!
//! ```text
//! DIAG_W_REF=0.5 DIAG_FLOOR=0 cargo test -p hexsim-core \
//!     --test diag_cloud_travel -- --ignored --nocapture
//! ```

use std::collections::VecDeque;

use hexsim_core::bench_metrics::{AtmosphereParamsOverride, BenchParams, build_bench_sim};
use hexsim_core::simulation::Simulation;

const LAGS: [usize; 3] = [1, 24, 48];
/// Trace threshold (mm), WMO: below it nothing is counted as rain, the
/// same floor the overlay uses per hour.
const TRACE_MM: f32 = 0.1;
const HOURS_PER_DAY: usize = 24;

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("count fits u32"))
}

/// Pearson correlation of two fields of equal length, f64 accumulation.
fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return f64::NAN;
    }
    let nf = count_f64(n);
    let mean_a = a.iter().map(|&x| f64::from(x)).sum::<f64>() / nf;
    let mean_b = b.iter().map(|&x| f64::from(x)).sum::<f64>() / nf;
    let (mut cov, mut var_a, mut var_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let dx = f64::from(x) - mean_a;
        let dy = f64::from(y) - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return f64::NAN;
    }
    cov / (var_a * var_b).sqrt()
}

struct LagAcc {
    sum: f64,
    count: usize,
}

struct Report {
    lag_corr: Vec<LagAcc>,
    painted_frac_sum: f64,
    hours: usize,
    daily_coverage: Vec<f64>,
    daily_mm_per_cell: Vec<f64>,
    rain_free_days: usize,
    water_start: f32,
    water_end: f32,
}

fn cloud_field(sim: &Simulation) -> Vec<f32> {
    sim.grid()
        .cells_slice()
        .iter()
        .map(|c| c.cloud_water)
        .collect()
}

fn build(radius: i32, seed: u32, w_ref: f32, floor: f32) -> Simulation {
    let overrides = BenchParams {
        atmosphere: AtmosphereParamsOverride {
            updraft_ref_ms: (w_ref > 0.0).then_some(w_ref),
            updraft_floor: (w_ref > 0.0).then_some(floor),
            ..AtmosphereParamsOverride::default()
        },
        ..BenchParams::default()
    };
    build_bench_sim(seed, radius, &overrides).0
}

fn measure(sim: &mut Simulation, measure_days: usize) -> Report {
    let n = sim.grid().len();
    let n_f = count_f64(n);
    let mut ring: VecDeque<Vec<f32>> = VecDeque::with_capacity(LAGS[2] + 1);
    let mut report = Report {
        lag_corr: LAGS.iter().map(|_| LagAcc { sum: 0.0, count: 0 }).collect(),
        painted_frac_sum: 0.0,
        hours: 0,
        daily_coverage: Vec::with_capacity(measure_days),
        daily_mm_per_cell: Vec::with_capacity(measure_days),
        rain_free_days: 0,
        water_start: sim.water_budget_total(),
        water_end: 0.0,
    };

    for _hour in 0..measure_days * HOURS_PER_DAY {
        sim.step_hour();
        let field = cloud_field(sim);
        for (k, &lag) in LAGS.iter().enumerate() {
            if ring.len() >= lag {
                let past = &ring[ring.len() - lag];
                let r = pearson(&field, past);
                if r.is_finite() {
                    report.lag_corr[k].sum += r;
                    report.lag_corr[k].count += 1;
                }
            }
        }
        ring.push_back(field);
        if ring.len() > LAGS[2] {
            ring.pop_front();
        }

        let painted = sim
            .precip_this_tick()
            .iter()
            .filter(|r| r.rain + r.snow > TRACE_MM)
            .count();
        report.painted_frac_sum += count_f64(painted) / n_f;
        report.hours += 1;

        if sim.hour_of_day() == 23 {
            let day = sim.last_precipitation();
            let covered = day.iter().filter(|r| r.rain + r.snow > TRACE_MM).count();
            let fallen: f64 = day.iter().map(|r| f64::from(r.rain + r.snow)).sum();
            report.daily_coverage.push(count_f64(covered) / n_f);
            report.daily_mm_per_cell.push(fallen / n_f);
            if !day.iter().any(|r| r.wet()) {
                report.rain_free_days += 1;
            }
        }
    }
    report.water_end = sim.water_budget_total();
    report
}

fn print(report: &Report, radius: i32, seed: u32, w_ref: f32, floor: f32, warmup: usize) {
    let trigger = if w_ref > 0.0 {
        format!("ON (w_ref={w_ref} m/s, floor={floor})")
    } else {
        "OFF (shipped default)".to_string()
    };
    let days = report.daily_coverage.len();
    let days_f = count_f64(days.max(1));
    let cov_mean = report.daily_coverage.iter().sum::<f64>() / days_f;
    let cov_max = report
        .daily_coverage
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let days_15 = report.daily_coverage.iter().filter(|&&c| c >= 0.15).count();
    let days_50 = report.daily_coverage.iter().filter(|&&c| c >= 0.50).count();
    let mm_mean = report.daily_mm_per_cell.iter().sum::<f64>() / days_f;
    let painted_mean = report.painted_frac_sum / count_f64(report.hours.max(1));
    let drift_pct = if report.water_start.abs() > 1e-6 {
        f64::from((report.water_end - report.water_start) / report.water_start) * 100.0
    } else {
        f64::NAN
    };

    eprintln!(
        "\n=== diag_cloud_travel — r{radius} seed {seed}, warmup {warmup} d, window {days} d ==="
    );
    eprintln!("  trigger: {trigger}");
    eprintln!("  cloud_water temporal correlation (mean over the window):");
    for (k, &lag) in LAGS.iter().enumerate() {
        let acc = &report.lag_corr[k];
        let mean = if acc.count > 0 {
            acc.sum / count_f64(acc.count)
        } else {
            f64::NAN
        };
        eprintln!(
            "    lag {lag:>2} h : r = {mean:.3}   ({} samples)",
            acc.count
        );
    }
    eprintln!(
        "  daily rain coverage (> {TRACE_MM} mm/d): mean {:.1} %, max {:.1} %, days >= 15 %: {days_15}/{days}, days >= 50 %: {days_50}/{days}",
        cov_mean * 100.0,
        cov_max * 100.0
    );
    eprintln!(
        "  painted fraction per hour (> {TRACE_MM} mm/h): mean {:.3} %",
        painted_mean * 100.0
    );
    eprintln!(
        "  fallen: {mm_mean:.3} mm/cell/day, fully rain-free days: {}/{days}",
        report.rain_free_days
    );
    eprintln!(
        "  water budget: {:.1} -> {:.1} mm, drift {drift_pct:+.2e} %",
        report.water_start, report.water_end
    );
}

#[test]
#[ignore = "instrument — DIAG_W_REF=… cargo test -p hexsim-core --test diag_cloud_travel -- --ignored --nocapture"]
fn diag_cloud_travel() {
    let radius: i32 = env_parse("DIAG_RADIUS", 120);
    let seed: u32 = env_parse("DIAG_SEED", 42);
    let warmup_days: usize = env_parse("DIAG_WARMUP_DAYS", 540);
    let measure_days: usize = env_parse("DIAG_MEASURE_DAYS", 60);
    let w_ref: f32 = env_parse("DIAG_W_REF", 0.0);
    let floor: f32 = env_parse("DIAG_FLOOR", 0.1);

    let mut sim = build(radius, seed, w_ref, floor);
    for _ in 0..warmup_days {
        sim.step();
    }
    let report = measure(&mut sim, measure_days);
    print(&report, radius, seed, w_ref, floor, warmup_days);
}
