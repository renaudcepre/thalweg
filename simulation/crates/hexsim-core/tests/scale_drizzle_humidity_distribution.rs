//! Phase 4 of task #63, diagnostic `HR_upper` distribution by altitude band.
//!
//! Hypothesis to test (Phase 3 invalidated the orographic-pump cause and
//! pointed here instead): planetary drizzle comes from **chronic
//! asymmetry** in `condensation_rate` / `cloud_evap_rate`
//! chain:
//! - condensation activates at `HR > 0.75`
//! - `cloud_evap` only if `HR < 0.40`
//! - `subsidence` reinjets `humidity_upper → humidity_surface` in closed loop
//!
//! If `cloud_evap` window (HR < 0.40) never reached while condensation window
//! (HR > 0.75) permanent, we have root cause: `cloud_water` sticks above
//! `CLOUD_MIN_PRECIP` and makes perpetual drizzle.
//!
//! What this test measures, by elevation band (same bands as
//! `scale_drizzle_by_altitude` and `scale_knockout_drizzle`):
//! - p25 / median / p75 / p99 of `HR_upper` observed over 2 years (1 sample/day)
//! - mean `HR_upper`
//! - `% obs` in active `cloud_evap` window (`HR < 0.40`)
//! - `% obs` in dead zone (`0.40 ≤ HR ≤ 0.75`)
//! - `% obs` in active condensation window (`HR > 0.75`)
//!
//! **Eval style** (cf project memory `scale_tests_eval_style`): no assert,
//! just `eprintln!` + `#[ignore]`. We read, draw a hypothesis, then switch
//! to a targeted knockout on the chain (Step 2 of the Phase 4 plan).
//!
//! **Vocabulary convention** (cf project memory `tick_jours_vs_heures`):
//! metrics in DAYS. `sim.step()` advances 1 day. HR sample taken once
//! per day at midnight (after the 24 internal [`Simulation::step_hour`]),
//! sub-sampled by construction, may miss intra-day oscillations but that's
//! the resolution of the other drizzle tests in this phase.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test scale_drizzle_humidity_distribution \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::{AtmosphereParams, saturation_upper};
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
/// RH threshold for condensation onset in diagnostic classification. Since #63
/// P4E3, condensation anchored to `saturation_upper(T)` (no dimensionless RH
/// threshold in engine): physical onset is RH = 1.0 (saturation). Display constant,
/// delimits `%cond` window of histogram, decoupled from any engine parameter
/// (ex-`condensation_hr_threshold` dead, removed via #67).
const COND_ONSET_HR: f32 = 1.0;

/// Same bands as `scale_drizzle_by_altitude` and `scale_knockout_drizzle`.
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

/// Percentile index in integer arithmetic, deterministic and no cast.
/// `p_promille` ∈ [0, 1000] (ex: 500 = median, 990 = p99).
fn percentile_idx(len: usize, p_promille: u32) -> usize {
    if len == 0 {
        return 0;
    }
    let p = usize::try_from(p_promille).expect("u32 fits usize");
    ((len - 1) * p / 1000).min(len - 1)
}

struct BandReport {
    name: &'static str,
    cells: usize,
    obs: usize,
    hr_mean: f64,
    hr_p25: f32,
    hr_med: f32,
    hr_p75: f32,
    hr_p99: f32,
    /// Fraction (0..1) of observations at `HR < cloud_evap_hr_threshold`.
    pct_evap_active: f64,
    /// Fraction (0..1) en `HR > COND_ONSET_HR` (saturation, onset condensation).
    pct_cond_active: f64,
    /// Fraction (0..1) in the dead zone (between the two thresholds).
    pct_dead_zone: f64,
}

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

/// Cell indices per band, computed once at startup.
fn build_band_cells(sim: &Simulation) -> Vec<Vec<usize>> {
    let mut band_cells: Vec<Vec<usize>> = vec![Vec::new(); BANDS.len()];
    for (i, (_, cell)) in sim.grid().iter().enumerate() {
        for (band_idx, (_, lo, hi)) in BANDS.iter().enumerate() {
            if cell.elevation >= *lo && cell.elevation < *hi {
                band_cells[band_idx].push(i);
                break;
            }
        }
    }
    band_cells
}

/// Per-band counters of observations in each physical window.
struct WindowCounts {
    evap: Vec<u64>,
    cond: Vec<u64>,
    dead: Vec<u64>,
}

/// Runs the simulation and collects all `HR_upper` values per band +
/// the window counters. Also returns the elapsed time.
fn collect_hr_observations(
    sim: &mut Simulation,
    band_cells: &[Vec<usize>],
    atmo: &AtmosphereParams,
    t_offset: f32,
) -> (Vec<Vec<f32>>, WindowCounts, f64) {
    let mut hr_obs: Vec<Vec<f32>> = band_cells
        .iter()
        .map(|cells| Vec::with_capacity(cells.len() * usize::try_from(TOTAL_DAYS).unwrap_or(0)))
        .collect();
    let mut counts = WindowCounts {
        evap: vec![0; BANDS.len()],
        cond: vec![0; BANDS.len()],
        dead: vec![0; BANDS.len()],
    };
    let evap_threshold = atmo.cloud_evap_hr_threshold;
    let cond_threshold = COND_ONSET_HR;

    let t0 = std::time::Instant::now();
    for _day in 0..TOTAL_DAYS {
        sim.step();
        let cells_slice = sim.grid().cells_slice();
        for (band_idx, idxs) in band_cells.iter().enumerate() {
            for &i in idxs {
                let cell = &cells_slice[i];
                let t_upper = cell.temperature - t_offset;
                let sat = saturation_upper(t_upper, atmo);
                let hr = if sat > 0.0 {
                    cell.humidity_upper / sat
                } else {
                    0.0
                };
                hr_obs[band_idx].push(hr);
                if hr < evap_threshold {
                    counts.evap[band_idx] += 1;
                } else if hr > cond_threshold {
                    counts.cond[band_idx] += 1;
                } else {
                    counts.dead[band_idx] += 1;
                }
            }
        }
    }
    (hr_obs, counts, t0.elapsed().as_secs_f64())
}

fn build_reports(
    band_cells: &[Vec<usize>],
    hr_obs: &mut [Vec<f32>],
    counts: &WindowCounts,
) -> Vec<BandReport> {
    let mut reports: Vec<BandReport> = Vec::new();
    for (band_idx, (name, _, _)) in BANDS.iter().enumerate() {
        let cells = band_cells[band_idx].len();
        let obs_vec = &mut hr_obs[band_idx];
        let obs = obs_vec.len();
        if cells == 0 || obs == 0 {
            continue;
        }
        obs_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let hr_p25 = obs_vec[percentile_idx(obs, 250)];
        let hr_med = obs_vec[percentile_idx(obs, 500)];
        let hr_p75 = obs_vec[percentile_idx(obs, 750)];
        let hr_p99 = obs_vec[percentile_idx(obs, 990)];
        let sum: f64 = obs_vec.iter().map(|&h| f64::from(h)).sum();
        // obs ≤ ~2700 cells × 730 days < 2M, fits in u32 → lossless conversion.
        let total_f = f64::from(u32::try_from(obs).expect("obs fits u32"));
        let hr_mean = sum / total_f;
        let pct_evap_active =
            f64::from(u32::try_from(counts.evap[band_idx]).unwrap_or(0)) / total_f;
        let pct_cond_active =
            f64::from(u32::try_from(counts.cond[band_idx]).unwrap_or(0)) / total_f;
        let pct_dead_zone = f64::from(u32::try_from(counts.dead[band_idx]).unwrap_or(0)) / total_f;
        reports.push(BandReport {
            name,
            cells,
            obs,
            hr_mean,
            hr_p25,
            hr_med,
            hr_p75,
            hr_p99,
            pct_evap_active,
            pct_cond_active,
            pct_dead_zone,
        });
    }
    reports
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn drizzle_humidity_distribution_baseline() {
    let atmo = AtmosphereParams::default();
    let temp = TemperatureParams::default();
    let evap_threshold = atmo.cloud_evap_hr_threshold;
    let cond_threshold = COND_ONSET_HR;
    let t_offset = temp.lapse_rate * atmo.upper_layer_altitude_m / 1000.0;

    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    let band_cells = build_band_cells(&sim);
    let (mut hr_obs, counts, elapsed) =
        collect_hr_observations(&mut sim, &band_cells, &atmo, t_offset);

    let total_days_f = f64::from(u32::try_from(TOTAL_DAYS).expect("fits u32"));
    eprintln!(
        "  simulation {} days ({} hours) in {:.2}s, {:.1} ms/day",
        TOTAL_DAYS,
        TOTAL_DAYS * 24,
        elapsed,
        1000.0 * elapsed / total_days_f,
    );

    let reports = build_reports(&band_cells, &mut hr_obs, &counts);

    eprintln!(
        "\n=== Planetary drizzle (#63 Phase 4), HR_upper distribution by band (seed {SEED}, {YEARS} years) ==="
    );
    eprintln!(
        "  thresholds: cloud_evap active if HR < {evap_threshold:.2} | condensation active if HR > {cond_threshold:.2}\n"
    );
    eprintln!(
        "  {:>10} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "band",
        "cells",
        "obs",
        "HRmean",
        "HR p25",
        "HR med",
        "HR p75",
        "HR p99",
        "%evap",
        "%dead",
        "%cond",
    );
    eprintln!("  {}", "-".repeat(110));
    for r in &reports {
        eprintln!(
            "  {:>10} {:>5} {:>8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>7.1}% {:>7.1}% {:>7.1}%",
            r.name,
            r.cells,
            r.obs,
            r.hr_mean,
            r.hr_p25,
            r.hr_med,
            r.hr_p75,
            r.hr_p99,
            r.pct_evap_active * 100.0,
            r.pct_dead_zone * 100.0,
            r.pct_cond_active * 100.0,
        );
    }
    eprintln!(
        "\nLegend:
  HRmean / pXX : `HR_upper` = humidity_upper / saturation_upper(T_upper).
                 Sampled once/day at midnight after the 24 step_hour calls.
  %evap        : fraction of obs at HR < {evap_threshold:.2} (`cloud_evap` window active)
  %cond        : fraction of obs at HR > {cond_threshold:.2} (condensation window active)
  %dead        : fraction of obs in the dead zone (hysteresis between the two thresholds)

Strong hypothesis (asymmetric chain):
  - %evap ~ 0 everywhere, %cond high at altitude -> cloud_evap window dead -> drizzle.
  - HR med > 0.75 above >800m -> condensation self-sustains.

If confirmed, Step 2 (directed knockout) will test 4 levers:
  - condensation_rate = 0      (cut the source)
  - cloud_evap_rate x 10       (force decondensation from clouds)
  - cloud_evap_hr_threshold = 0.7 (widen the cloud_evap window)
  - subsidence_rate = 0        (cut the humidity_upper -> surface loop)
"
    );
}
