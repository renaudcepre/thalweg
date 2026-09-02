//! Precipitation regime measurement harness (prep for #63 + #50).
//!
//! Context: audit #67 showed that NO phenomenological coefficient creates a
//! dry day; the planetary drizzle is structural. The atmo water stock parks
//! itself right above the `CLOUD_MIN_PRECIP = 0.05` floor everywhere, and
//! since precip is allowed as soon as `cloud_water > 0.05` (no dynamic
//! gate), it drizzles permanently without ever accumulating enough for a
//! real downpour.
//!
//! Two physical levers (design #69) address this, each refuted ALONE:
//! - `updraft_ref_ms` / `updraft_floor` (ex-design C): no rain without
//!   updraft (convergence + orographic lift), dries out subsidence zones.
//!   Alone: "convergence alone kills the mountains".
//! - `precip_crit_mm` (ex-design A): below the critical mass the cloud
//!   loads up and travels without precipitating; above it, the super-linear
//!   KK2000 purges in a burst. Alone: "triggers everywhere" (refuted #69),
//!   and at the time blocked by a `scale_dry_periods` GLACIER regression,
//!   a criterion since REMOVED (the terrain caps out at ~1789 m at R=30, no
//!   perennial glacier is legitimate).
//!
//! Update (diag `diag_updraft_field`): the `updraft_ref_ms` trigger is
//! BROKEN (the `w` field is aberrant, ±50-300 m/s) and redundant (the rain
//! already coincides with the updraft), so it's dropped. This harness is
//! therefore refocused on a **`precip_crit_mm`-only sweep** (updraft OFF),
//! to find the critical mass that replaces the magic number
//! `CLOUD_MIN_PRECIP = 0.05`. Indicators:
//! - `global_dry_days`: GLOBALLY dry days (the real #63 criterion, "it
//!   rains somewhere 365 days/year"), baseline = 0, we want > 0;
//! - `intensity_wet`: mean rain per rainy cell-day (mm), the drizzle↔downpour
//!   discriminant: low = spread-out drizzle, high = bursts;
//! - `dry_median`: median dry max-streak PER CELL;
//! - `raining_cells`: rainy cells/day (drizzle extent, should go DOWN
//!   without falling to ~0);
//! - `rain/year` per band (peaks >800m must NOT become deserts);
//! - seasonal snow `snow_min_late` > 0 (sanity check, NO glacier expected).
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test scale_precip_regime \
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

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const YEARS: u64 = 2;
const TOTAL_DAYS: u64 = YEARS * 365;
const RAIN_THRESHOLD: f32 = 1e-5;
/// Elevation threshold for "peaks" for seasonal snow (m).
const HIGH_ALT_M: f32 = 800.0;

/// Elevation bands (same bounds as the other scale tests).
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

/// A scenario = a set of overrides on the two levers.
struct Regime {
    label: &'static str,
    updraft_ref_ms: f32,
    updraft_floor: f32,
    precip_crit_mm: f32,
}

struct RegimeResult {
    label: &'static str,
    dry_median: u32,
    /// Days where NO cell rains at all (the real #63 criterion: "it rains
    /// somewhere 365 days/year", we want > 0).
    global_dry_days: u32,
    raining_cells_mean: f64,
    intensity_wet: f64,
    burst_max: f32,
    rain_by_band: Vec<f64>,
    snow_max: f64,
    snow_min_late: f64,
}

fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("count fits u32"))
}

fn build_sim(atmo: AtmosphereParams) -> Simulation {
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
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind,
    )
}

fn band_index(elev: f32) -> Option<usize> {
    BANDS
        .iter()
        .position(|(_, lo, hi)| elev >= *lo && elev < *hi)
}

fn run_regime(regime: &Regime) -> RegimeResult {
    eprintln!(
        "--- regime: {} (w_ref={}, floor={}, crit={}) ---",
        regime.label, regime.updraft_ref_ms, regime.updraft_floor, regime.precip_crit_mm
    );
    let t0 = std::time::Instant::now();

    let atmo = AtmosphereParams {
        updraft_ref_ms: regime.updraft_ref_ms,
        updraft_floor: regime.updraft_floor,
        precip_crit_mm: regime.precip_crit_mm,
        ..AtmosphereParams::default()
    };
    let mut sim = build_sim(atmo);

    let elevs: Vec<f32> = sim.grid().iter().map(|(_, c)| c.elevation).collect();
    let n = elevs.len();
    let high_cells: Vec<usize> = (0..n).filter(|&i| elevs[i] > HIGH_ALT_M).collect();

    // Accumulators.
    let mut cur_dry = vec![0_u32; n];
    let mut best_dry = vec![0_u32; n];
    let mut rainy_days = vec![0_u32; n];
    let mut raining_cell_days = 0_u64;
    let mut rain_total = 0.0_f64;
    let mut raining_cells_sum = 0.0_f64;
    let mut burst_max = 0.0_f32;
    let mut snow_max = 0.0_f64;
    let mut global_dry_days = 0_u32;
    let mut snow_series: Vec<f64> = Vec::with_capacity(usize::try_from(TOTAL_DAYS).unwrap_or(0));

    for _day in 0..TOTAL_DAYS {
        sim.step();

        let mut raining_today = 0_usize;
        for (i, rec) in sim.last_precipitation().iter().enumerate() {
            let rain = rec.rain;
            if rain > RAIN_THRESHOLD {
                raining_today += 1;
                rainy_days[i] += 1;
                raining_cell_days += 1;
                rain_total += f64::from(rain);
                burst_max = burst_max.max(rain);
                cur_dry[i] = 0;
            } else {
                cur_dry[i] += 1;
                best_dry[i] = best_dry[i].max(cur_dry[i]);
            }
        }
        raining_cells_sum += count_f64(raining_today);
        if raining_today == 0 {
            global_dry_days += 1;
        }

        let grid = sim.grid();
        let cells = grid.cells_slice();
        let snow_today: f64 = high_cells
            .iter()
            .map(|&i| f64::from(cells[i].snow_level))
            .sum();
        snow_max = snow_max.max(snow_today);
        snow_series.push(snow_today);
    }

    let total_days_f = count_f64(usize::try_from(TOTAL_DAYS).expect("fits usize"));
    let dry_median = median_u32(&mut best_dry);
    let intensity_wet = if raining_cell_days > 0 {
        rain_total / count_f64(usize::try_from(raining_cell_days).expect("fits usize"))
    } else {
        0.0
    };
    let rain_by_band = rain_per_year_by_band(&elevs, &rainy_days, total_days_f);
    // Snow "min after the 1st summer": min over the 2nd half of the run
    // (avoids the warmup transient, captures the summer dip; must stay > 0
    // but melt, NOT a glacial plateau).
    let mid = snow_series.len() / 2;
    let snow_min_late = snow_series[mid..]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    eprintln!("  done in {:.1}s", t0.elapsed().as_secs_f64());
    RegimeResult {
        label: regime.label,
        dry_median,
        global_dry_days,
        raining_cells_mean: raining_cells_sum / total_days_f,
        intensity_wet,
        burst_max,
        rain_by_band,
        snow_max,
        snow_min_late,
    }
}

fn median_u32(values: &mut [u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

fn rain_per_year_by_band(elevs: &[f32], rainy_days: &[u32], total_days_f: f64) -> Vec<f64> {
    let mut sum = vec![0.0_f64; BANDS.len()];
    let mut count = vec![0_usize; BANDS.len()];
    for (i, &elev) in elevs.iter().enumerate() {
        if let Some(b) = band_index(elev) {
            sum[b] += f64::from(rainy_days[i]) * 365.0 / total_days_f;
            count[b] += 1;
        }
    }
    sum.iter()
        .zip(&count)
        .map(|(s, &c)| if c > 0 { s / count_f64(c) } else { 0.0 })
        .collect()
}

fn print_report(results: &[RegimeResult]) {
    eprintln!("\n\n=== Precip regime, sweep precip_crit_mm (seed {SEED}, {YEARS} years) ===");
    eprintln!(
        "\n  {:>12}  {:>8}  {:>9}  {:>13}  {:>10}  {:>13}  {:>9}  {:>12}",
        "regime",
        "dry_med",
        "dry_days",
        "intensity_wet",
        "burst_max",
        "raining_cells",
        "snow_max",
        "snow_min_late"
    );
    eprintln!(
        "  {}",
        "-".repeat(12 + 2 + 8 + 2 + 9 + 2 + 13 + 2 + 10 + 2 + 13 + 2 + 9 + 2 + 12)
    );
    for r in results {
        eprintln!(
            "  {:>12}  {:>8}  {:>9}  {:>13.4}  {:>10.3}  {:>13.1}  {:>9.1}  {:>12.1}",
            r.label,
            r.dry_median,
            r.global_dry_days,
            r.intensity_wet,
            r.burst_max,
            r.raining_cells_mean,
            r.snow_max,
            r.snow_min_late,
        );
    }

    eprintln!("\n=== rain/year by band (effective rain days/cell) ===");
    eprint!("  {:>12}", "regime");
    for (name, _, _) in BANDS {
        eprint!("  {name:>10}");
    }
    eprintln!();
    eprintln!("  {}", "-".repeat(12 + 12 * BANDS.len()));
    for r in results {
        eprint!("  {:>12}", r.label);
        for v in &r.rain_by_band {
            eprint!("  {v:>10.1}");
        }
        eprintln!();
    }

    eprintln!(
        "\nReading (target = precip_crit_mm calibration window):
  - dry_days: GLOBALLY dry days (the real #63 criterion). baseline = 0. We want > 0.
  - intensity_wet: drizzle<->downpour discriminant. baseline = drizzle. We want it up.
  - raining_cells: drizzle extent. We want it down (without collapsing to ~0).
  - rain/year >800m: must NOT collapse (otherwise peaks turn to desert).
  - snow_min_late > 0: snow melts but persists (seasonal sanity,
                        NO glacier expected, terrain <1800 m).
  Good value = best compromise between dry_days>0 / intensity_wet up / peaks alive.
"
    );
}

#[test]
#[ignore = "exploratory measurement harness, run with --ignored --nocapture"]
fn precip_regime_updraft_crit_grid() {
    // `precip_crit_mm`-only sweep (updraft OFF: diag `diag_updraft_field`
    // showed the trigger is broken + redundant, it doesn't help the drizzle).
    // Goal: find the critical mass that converts drizzle into downpour
    // (intensity_wet ↑, raining_cells ↓, global_dry_days > 0) without
    // desertifying the peaks (rain/year >800m) or killing the snow
    // (snow_min_late > 0). The magic 0.05 (CLOUD_MIN_PRECIP) is the current
    // implicit crit; we're looking for its physical value.
    let crit_values = [0.0_f32, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4];
    let labels = ["crit=0", "0.05", "0.10", "0.15", "0.20", "0.30", "0.40"];
    let regimes: Vec<Regime> = crit_values
        .iter()
        .zip(labels)
        .map(|(&crit, label)| Regime {
            label,
            updraft_ref_ms: 0.0,
            updraft_floor: 0.0,
            precip_crit_mm: crit,
        })
        .collect();

    let results: Vec<RegimeResult> = regimes.iter().map(run_regime).collect();
    print_report(&results);

    // Minimal sanity check: the run executes and produces cells (no assertion
    // on the physics; this is a measurement harness, calibration comes after).
    assert!(!results.is_empty(), "at least one regime measured");
    for r in &results {
        assert_eq!(r.rain_by_band.len(), BANDS.len(), "all bands present");
    }
}
