//! Structural diagnostic: a profiler for temporal cycles.
//!
//! Tool 4 of the `feat/structural-diag-toolkit` toolkit. Target:
//! - `project_variabilite_inter_annuelle` memory note: "annual cycles too
//!   identical, next axis: introduce oscillations/noise to break
//!   the monotony"
//!
//! Quantifies how "alive" the simulator feels on global scalars
//! (`T_mean`, `cloud_total`, `rain_total`, `raining_count`, `h_upper_mean`):
//! - **intra-day vs inter-day variance**: relative weight of diurnal
//!   vs synoptic oscillations
//! - **seasonal amplitude**: 12 monthly averages, max-min
//! - **year-over-year correlation**: `r(year1_series, year2_series)` at the
//!   same hour-of-year. r ≈ 1 = strictly periodic simulation (monotone),
//!   r ≈ 0 = chaotic. This is the direct indicator of the user memory.
//! - **autocorrelation at key lags** (1h, 24h, 168h=week, 720h=month)
//!
//! **Style eval** (`scale_tests_eval_style`) : `#[ignore]`, aucun assert.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test diag_cycles \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

use std::fmt::Write as _;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const WARMUP_DAYS: u64 = 90;
const YEARS: u64 = 2;
const HOURS_PER_DAY: u64 = 24;
const DAYS_PER_YEAR: u64 = 365;
const HOURS_PER_YEAR: u64 = HOURS_PER_DAY * DAYS_PER_YEAR;
const TOTAL_HOURS: u64 = YEARS * HOURS_PER_YEAR;

/// Chosen autocorrelation lags: 1 h (immediate persistence), 24 h
/// (diurnal cycle), 168 h (week, captures synoptic oscillations), 720
/// h (month, captures short seasonal drift). No 8760 lag (full
/// year) since the `YoY` correlation covers it better.
const LAGS: &[u64] = &[1, 24, 168, 720];

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

/// 5 global scalar series tracked at hourly resolution, over 2 years.
struct Series {
    t_mean: Vec<f64>,
    cloud_total: Vec<f64>,
    rain_total: Vec<f64>,
    raining_count: Vec<f64>,
    h_upper_mean: Vec<f64>,
}

fn collect_series(sim: &mut Simulation) -> Series {
    let cap = usize::try_from(TOTAL_HOURS).unwrap_or(0);
    let mut s = Series {
        t_mean: Vec::with_capacity(cap),
        cloud_total: Vec::with_capacity(cap),
        rain_total: Vec::with_capacity(cap),
        raining_count: Vec::with_capacity(cap),
        h_upper_mean: Vec::with_capacity(cap),
    };
    for _ in 0..TOTAL_HOURS {
        sim.step_hour();
        let cells = sim.grid().cells_slice();
        let n = cells.len();
        let n_f = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
        let mut t_sum = 0.0;
        let mut cw_sum = 0.0;
        let mut hu_sum = 0.0;
        for c in cells {
            t_sum += f64::from(c.temperature);
            cw_sum += f64::from(c.cloud_water);
            hu_sum += f64::from(c.humidity_upper);
        }
        s.t_mean.push(t_sum / n_f);
        s.cloud_total.push(cw_sum);
        s.h_upper_mean.push(hu_sum / n_f);

        let precip = sim.last_precipitation();
        let mut rain_sum = 0.0;
        let mut rain_count = 0u32;
        for rec in precip {
            let r = f64::from(rec.rain);
            rain_sum += r;
            if rec.rain > 0.0 {
                rain_count += 1;
            }
        }
        s.rain_total.push(rain_sum);
        s.raining_count.push(f64::from(rain_count));
    }
    s
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX));
    values.iter().sum::<f64>() / n
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let n = f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX));
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n_usize = x.len().min(y.len());
    if n_usize < 2 {
        return f64::NAN;
    }
    let mx = mean(&x[..n_usize]);
    let my = mean(&y[..n_usize]);
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..n_usize {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-12 { f64::NAN } else { num / denom }
}

/// Intra-day variance: for each 24 h day, computes the variance of the
/// 24 values; returns the average of these variances.
fn intraday_variance(values: &[f64]) -> f64 {
    let chunk = usize::try_from(HOURS_PER_DAY).unwrap_or(24);
    if values.len() < chunk {
        return 0.0;
    }
    let mut accum = 0.0;
    let mut n_days = 0u32;
    for window in values.chunks_exact(chunk) {
        accum += variance(window);
        n_days += 1;
    }
    if n_days == 0 {
        0.0
    } else {
        accum / f64::from(n_days)
    }
}

/// Inter-day variance: variance of daily means (the 24 h chunks
/// collapsed into 1 value per day).
fn interday_variance(values: &[f64]) -> f64 {
    let chunk = usize::try_from(HOURS_PER_DAY).unwrap_or(24);
    let daily_means: Vec<f64> = values.chunks_exact(chunk).map(mean).collect();
    variance(&daily_means)
}

/// 12 monthly averages (month ≈ 30.4 days = 730 hours). Covers 1 year if
/// the series is `HOURS_PER_YEAR` long.
fn monthly_means(values: &[f64]) -> Vec<f64> {
    let hours_per_month = HOURS_PER_YEAR / 12;
    let chunk = usize::try_from(hours_per_month).unwrap_or(730);
    let mut out = Vec::with_capacity(12);
    for k in 0..12 {
        let start = chunk * k;
        let end = (start + chunk).min(values.len());
        if start >= end {
            break;
        }
        out.push(mean(&values[start..end]));
    }
    out
}

fn seasonal_amplitude(values: &[f64]) -> f64 {
    let m = monthly_means(values);
    if m.is_empty() {
        return 0.0;
    }
    let max = m.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = m.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}

fn autocorr_at_lag(values: &[f64], lag: u64) -> f64 {
    let lag_us = usize::try_from(lag).unwrap_or(usize::MAX);
    if values.len() <= lag_us {
        return f64::NAN;
    }
    let head = &values[..values.len() - lag_us];
    let tail = &values[lag_us..];
    pearson(head, tail)
}

/// Year-over-year correlation at the same hour-of-year. The series must
/// have at least `2 * HOURS_PER_YEAR` points.
fn year_on_year_corr(values: &[f64]) -> f64 {
    let yh = usize::try_from(HOURS_PER_YEAR).unwrap_or(usize::MAX);
    if values.len() < 2 * yh {
        return f64::NAN;
    }
    pearson(&values[..yh], &values[yh..2 * yh])
}

fn print_field_summary(label: &str, values: &[f64]) {
    let m = mean(values);
    let v = variance(values);
    let sd = v.sqrt();
    let intra = intraday_variance(values);
    let inter = interday_variance(values);
    let ratio = if intra > 1e-12 {
        inter / intra
    } else {
        f64::NAN
    };
    let ampl = seasonal_amplitude(values);
    let yoy = year_on_year_corr(values);

    eprintln!("\n  -- {label} --");
    eprintln!("    mean={m:>12.4}  std={sd:>12.4}  ampl_season={ampl:>10.4}  YoY_corr={yoy:>6.3}");
    eprintln!("    intra_var={intra:>10.4}  inter_var={inter:>10.4}  inter/intra={ratio:>6.2}");
    let mut row = String::from("    autocorr  ");
    for &lag in LAGS {
        let r = autocorr_at_lag(values, lag);
        let _ = write!(row, "  lag{lag}h={r:>5.3}");
    }
    eprintln!("{row}");

    let m_monthly = monthly_means(values);
    let mut mon_row = String::from("    monthly   ");
    for v in &m_monthly {
        let _ = write!(mon_row, " {v:>7.2}");
    }
    eprintln!("{mon_row}");
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_cycles() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    eprintln!(
        "\n=== Cycle profiler diag (variabilite_inter_annuelle), radius {RADIUS}, seed {SEED}, {YEARS} years ({TOTAL_HOURS} h) ===",
    );
    eprintln!(
        "  warmup = {WARMUP_DAYS} days, n_cells = {}",
        sim.grid().len()
    );
    eprintln!(
        "  intra/inter over fixed days (chunks 24 h), 12 months ≈ {} h each",
        HOURS_PER_YEAR / 12
    );
    eprintln!(
        "  YoY_corr = pearson(year1, year2) at the same hour-of-year. r≈1 = monotone, r≈0 = chaotic."
    );

    let s = collect_series(&mut sim);

    eprintln!("\n== 5 global scalars tracked over 2 years, hourly ==");
    print_field_summary("T_mean (°C)", &s.t_mean);
    print_field_summary("cloud_total (mm·grid)", &s.cloud_total);
    print_field_summary("rain_total (mm·grid / h)", &s.rain_total);
    print_field_summary("raining_count (cells)", &s.raining_count);
    print_field_summary("h_upper_mean (mm)", &s.h_upper_mean);
    eprintln!();
}
