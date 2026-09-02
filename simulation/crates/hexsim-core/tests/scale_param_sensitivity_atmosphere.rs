//! Sensitivity audit of the atmospheric coefficients (issue #67).
//!
//! For each phenomenological coefficient of the `atmosphere/` module, run a
//! **knockout** scenario (×0) and a **sensitivity** scenario (×10), and
//! compare against the baseline on the same metrics as `scale_knockout_drizzle`:
//!
//! - `rain/an` per elevation band (days of effective rain per cell)
//! - `cloud_mean` per band (mean `cloud_water` stock over time+space)
//! - `wet_max` per band (longest consecutive rain streak)
//! - `humidity_total` global (Σ vapor+cloud, averaged over time; detects a
//!   terrarium stock drift)
//! - `raining_cells_mean` global (cells raining on average per day)
//!
//! A summary auto-classifies each coefficient (DEAD / DOMINANT / SECONDARY)
//! based on the max deviation of `raining_cells_mean` and `humidity_total`
//! from the baseline (first pass, to be refined by hand in the JOURNAL).
//!
//! ## Scope (list re-established from `atmosphere/params.rs`, 2026-07-15)
//!
//! The #52 split, the #66/#115 removals (`uplift_orographic_coef`,
//! `orographic_boost`) and the #68 bump (`cloud_advection_rate` 0.37→3.0)
//! made the issue's April list obsolete. Swept here are the coefficients
//! **active by default**. Deliberately EXCLUDED:
//! - the synoptic triggers OFF by default (`updraft_ref_ms`, `updraft_floor`,
//!   `precip_crit_mm`, `global_precip_gate` = 0.0); their knockout is a
//!   no-op; enabling them is milestone #69 / Phase 4, not this audit;
//! - non-microphysical anchored physical quantities (`upper_layer_altitude_m`
//!   geometry, `initial_humidity_floor` IC, `transpiration_coef` FAO-56 #77,
//!   `sublimation_rate` snow).
//!
//! `kk2000_droplet_count` has a negative exponent (`N_c^−1.79`): ×0 diverges,
//! so it is sampled at ×0.1 / ×10 instead.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test scale_param_sensitivity_atmosphere \
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
/// 2 years: trade-off between a stable climate signature and cost (~15 scenarios).
const YEARS: u64 = 2;
const TOTAL_DAYS: u64 = YEARS * 365;
const RAIN_THRESHOLD: f32 = 1e-5;

/// Auto-classification thresholds on the max relative deviation vs baseline.
const DEAD_TOL: f32 = 0.02;
const DOMINANT_TOL: f32 = 0.50;

/// Same bands as `scale_knockout_drizzle` and `scale_drizzle_by_altitude`.
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

/// A coefficient to sweep: its name, the multiplicative factors tested, and
/// the mutation that applies `field *= factor` on a copy of the baseline.
struct Sweep {
    name: &'static str,
    factors: &'static [f32],
    apply: fn(&mut AtmosphereParams, f32),
}

#[derive(Debug, Clone)]
struct BandStat {
    name: &'static str,
    cells: usize,
    rain_per_year: f32,
    cloud_mean: f32,
    wet_max: usize,
}

#[derive(Clone)]
struct ScenarioResult {
    label: String,
    bands: Vec<BandStat>,
    humidity_total: f32,
    raining_cells_mean: f32,
}

fn to_f32(n: usize) -> f32 {
    f32::from(u16::try_from(n).expect("count fits u16"))
}

/// List of atmo coefficients active by default (see module doc).
fn sweeps() -> Vec<Sweep> {
    vec![
        // --- uplift / convection ---
        Sweep {
            name: "uplift_rate",
            factors: &[0.0, 10.0],
            apply: |p, f| p.uplift_rate *= f,
        },
        Sweep {
            name: "uplift_thermal_coef",
            factors: &[0.0, 10.0],
            apply: |p, f| p.uplift_thermal_coef *= f,
        },
        Sweep {
            name: "convective_diurnal_coef",
            factors: &[0.0, 10.0],
            apply: |p, f| p.convective_diurnal_coef *= f,
        },
        Sweep {
            name: "orographic_lift_coef",
            factors: &[0.0, 10.0],
            apply: |p, f| p.orographic_lift_coef *= f,
        },
        // --- condensation / cloud chain ---
        Sweep {
            name: "condensation_rate",
            factors: &[0.0, 10.0],
            apply: |p, f| p.condensation_rate *= f,
        },
        Sweep {
            name: "cloud_evap_rate",
            factors: &[0.0, 10.0],
            apply: |p, f| p.cloud_evap_rate *= f,
        },
        Sweep {
            name: "cloud_evap_hr_threshold",
            factors: &[0.0, 10.0],
            apply: |p, f| p.cloud_evap_hr_threshold *= f,
        },
        Sweep {
            name: "kk2000_droplet_count",
            factors: &[0.1, 10.0],
            apply: |p, f| p.kk2000_droplet_count *= f,
        },
        Sweep {
            name: "cloud_diffusion_rate",
            factors: &[0.0, 10.0],
            apply: |p, f| p.cloud_diffusion_rate *= f,
        },
        Sweep {
            name: "cloud_advection_rate",
            factors: &[0.0, 10.0],
            apply: |p, f| p.cloud_advection_rate *= f,
        },
        // --- precipitation ---
        Sweep {
            name: "precip_neighbor_share",
            factors: &[0.0, 10.0],
            apply: |p, f| p.precip_neighbor_share *= f,
        },
        Sweep {
            name: "max_precip_per_tick",
            factors: &[0.0, 10.0],
            apply: |p, f| p.max_precip_per_tick *= f,
        },
        // --- brouillard (#45) ---
        Sweep {
            name: "fog_condensation_rate",
            factors: &[0.0, 10.0],
            apply: |p, f| p.fog_condensation_rate *= f,
        },
        Sweep {
            name: "fog_condensation_threshold",
            factors: &[0.0, 10.0],
            apply: |p, f| p.fog_condensation_threshold *= f,
        },
    ]
}

fn build_sim_seeded(seed: u32, atmo: AtmosphereParams) -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    let wind = WindParams {
        seed,
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

fn run_scenario(label: &str, atmo: AtmosphereParams) -> ScenarioResult {
    eprintln!("--- scenario: {label} ---");
    let t0 = std::time::Instant::now();

    let mut sim = build_sim_seeded(SEED, atmo);

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
    let mut cloud_sum: HashMap<HexCoord, f32> =
        coords_elev.iter().map(|(c, _)| (*c, 0.0_f32)).collect();
    let mut humidity_sum = 0.0_f32;
    let mut raining_cells_sum = 0.0_f32;

    for _day in 0..TOTAL_DAYS {
        sim.step();
        let grid = sim.grid();
        let coords = grid.coords_slice();

        for (coord, cell) in grid.iter() {
            *cloud_sum.get_mut(coord).expect("init above") += cell.cloud_water;
            humidity_sum += cell.humidity_surface + cell.humidity_upper + cell.cloud_water;
        }
        let mut raining_today = 0_usize;
        for (i, rec) in sim.last_precipitation().iter().enumerate() {
            let wet = rec.rain > RAIN_THRESHOLD;
            if wet {
                raining_today += 1;
            }
            rain_history
                .get_mut(&coords[i])
                .expect("init above")
                .push(wet);
        }
        raining_cells_sum += to_f32(raining_today);
    }

    let total_days_f = to_f32(total_days);
    let bands = compute_bands(&coords_elev, &rain_history, &cloud_sum, total_days_f);

    eprintln!("  done in {:.1}s", t0.elapsed().as_secs_f64());
    ScenarioResult {
        label: label.to_string(),
        bands,
        humidity_total: humidity_sum / total_days_f,
        raining_cells_mean: raining_cells_sum / total_days_f,
    }
}

fn compute_bands(
    coords_elev: &[(HexCoord, f32)],
    rain_history: &HashMap<HexCoord, Vec<bool>>,
    cloud_sum: &HashMap<HexCoord, f32>,
    total_days_f: f32,
) -> Vec<BandStat> {
    let mut out = Vec::new();
    for (name, lo, hi) in BANDS {
        let cells: Vec<HexCoord> = coords_elev
            .iter()
            .filter(|(_, e)| *e >= *lo && *e < *hi)
            .map(|(c, _)| *c)
            .collect();
        if cells.is_empty() {
            continue;
        }
        let n_f = to_f32(cells.len());

        let mut rain_per_year_sum = 0.0_f32;
        let mut wet_max_global = 0_usize;
        for coord in &cells {
            let Some(hist) = rain_history.get(coord) else {
                continue;
            };
            let rainy = hist.iter().filter(|&&b| b).count();
            rain_per_year_sum += to_f32(rainy) * 365.0 / total_days_f;
            wet_max_global = wet_max_global.max(longest_streak(hist));
        }
        let cloud_mean = cells
            .iter()
            .filter_map(|c| cloud_sum.get(c).copied())
            .sum::<f32>()
            / (n_f * total_days_f);

        out.push(BandStat {
            name,
            cells: cells.len(),
            rain_per_year: rain_per_year_sum / n_f,
            cloud_mean,
            wet_max: wet_max_global,
        });
    }
    out
}

fn longest_streak(hist: &[bool]) -> usize {
    let mut cur = 0_usize;
    let mut best = 0_usize;
    for &b in hist {
        if b {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

fn print_band_table(
    metric_name: &str,
    fmt: &str,
    results: &[ScenarioResult],
    extract: impl Fn(&BandStat) -> f32,
) {
    // Headers derived from the bands actually present (baseline first):
    // `compute_bands` skips empty bands, so we align on them rather than on
    // the `BANDS` const (otherwise the columns shift if a band is missing).
    let Some(reference) = results.first() else {
        return;
    };
    eprintln!("\n=== {metric_name} ===");
    eprint!("  {:>28}", "scenario");
    for band in &reference.bands {
        eprint!("  {:>10}", band.name);
    }
    eprintln!();
    eprint!("  {:>28}", "(cells)");
    for band in &reference.bands {
        eprint!("  {:>10}", band.cells);
    }
    eprintln!();
    eprintln!("  {}", "-".repeat(28 + 12 * reference.bands.len()));
    for res in results {
        eprint!("  {:>28}", res.label);
        for band in &res.bands {
            let v = extract(band);
            match fmt {
                "f1" => eprint!("  {v:>10.1}"),
                _ => eprint!("  {v:>10.4}"),
            }
        }
        eprintln!();
    }
}

fn print_global_table(results: &[ScenarioResult]) {
    eprintln!("\n=== global (Σ atmo stock, rainy cells/day) ===");
    eprintln!(
        "  {:>28}  {:>14}  {:>18}",
        "scenario", "humidity_total", "raining_cells_mean"
    );
    eprintln!("  {}", "-".repeat(28 + 2 + 14 + 2 + 18));
    for res in results {
        eprintln!(
            "  {:>28}  {:>14.1}  {:>18.1}",
            res.label, res.humidity_total, res.raining_cells_mean
        );
    }
}

fn rel_dev(value: f32, base: f32) -> f32 {
    if base.abs() < f32::EPSILON {
        return if value.abs() < f32::EPSILON { 0.0 } else { 1.0 };
    }
    (value - base).abs() / base.abs()
}

fn classify(dev: f32) -> &'static str {
    if dev < DEAD_TOL {
        "DEAD?"
    } else if dev > DOMINANT_TOL {
        "DOMINANT"
    } else {
        "SECONDARY"
    }
}

/// Auto-classifying summary: for each coef, max deviation (over its factors)
/// of `raining_cells_mean` and `humidity_total` vs baseline.
fn print_summary(baseline: &ScenarioResult, sweeps: &[Sweep], variants: &[ScenarioResult]) {
    eprintln!("\n\n=== SUMMARY auto-classification (vs baseline) ===");
    eprintln!(
        "  {:>28}  {:>12}  {:>12}  {:>12}",
        "coefficient", "dev_rain", "dev_humid", "verdict"
    );
    eprintln!("  {}", "-".repeat(28 + 2 + 12 + 2 + 12 + 2 + 12));

    let mut idx = 0_usize;
    for sweep in sweeps {
        let mut dev_rain = 0.0_f32;
        let mut dev_humid = 0.0_f32;
        for _ in sweep.factors {
            let res = &variants[idx];
            idx += 1;
            dev_rain = dev_rain.max(rel_dev(res.raining_cells_mean, baseline.raining_cells_mean));
            dev_humid = dev_humid.max(rel_dev(res.humidity_total, baseline.humidity_total));
        }
        let verdict = classify(dev_rain.max(dev_humid));
        eprintln!(
            "  {:>28}  {dev_rain:>12.3}  {dev_humid:>12.3}  {verdict:>12}",
            sweep.name
        );
    }

    eprintln!(
        "\nLegend:
  - DEAD?      : neither ×0 nor ×10 moves rain OR humid by >{:.0}% → removal candidate
                 (confirm on a 2nd seed before any removal PR).
  - DOMINANT   : swing >{:.0}% → deserves an SI-anchored calibration.
  - SECONDARY  : real but moderate effect → keep, document the role.
  Indicative thresholds, not proof; cross-check against the per-band tables.",
        DEAD_TOL * 100.0,
        DOMINANT_TOL * 100.0,
    );
}

#[test]
#[ignore = "exploratory sensitivity audit, run with --ignored --nocapture"]
fn atmosphere_param_sensitivity() {
    let base_params = AtmosphereParams::default();
    let sweeps = sweeps();

    let baseline = run_scenario("baseline", base_params.clone());

    let mut variants = Vec::new();
    for sweep in &sweeps {
        for &factor in sweep.factors {
            let mut atmo = base_params.clone();
            (sweep.apply)(&mut atmo, factor);
            let label = format!("{}×{factor}", sweep.name);
            variants.push(run_scenario(&label, atmo));
        }
    }

    // Full table: baseline first, then all the variants.
    let mut all = Vec::with_capacity(variants.len() + 1);
    all.push(baseline.clone());
    all.extend(variants.iter().cloned());

    eprintln!(
        "\n\n=== Atmo sensitivity audit (#67), seed {SEED}, {YEARS} years, {} coefs ===",
        sweeps.len()
    );
    print_band_table(
        "rain/year (days of effective rain per cell)",
        "f1",
        &all,
        |b| b.rain_per_year,
    );
    print_band_table(
        "cloud_mean (mm LWP, averaged over time+space)",
        "f4",
        &all,
        |b| b.cloud_mean,
    );
    print_band_table("wet_max (longest rain streak, days)", "f1", &all, |b| {
        to_f32(b.wet_max)
    });
    print_global_table(&all);
    print_summary(&baseline, &sweeps, &variants);
}
