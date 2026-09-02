//! Phase 4 Step 2 of milestone #63, knockout of the `humidity_upper`
//! sources/transport to identify the dominant contributor to the
//! pathological supersaturation observed in Step 1.
//!
//! Step 1 (an earlier diagnostic in this line of work, baseline `HR_upper`) established:
//! - `HR_upper` p99 = 16.82 on peaks >1500m, p99 = 7.20 on 800-1500m
//! - the `cloud_evap` window (HR<0.40) is active 75-95% of the time on the
//!   plain; the "dead window" hypothesis is invalidated
//! - the true root signal is the supersaturation `humidity_upper >> saturation`
//!
//! The `step_cloud_dynamics` formula is asymptotic toward HR = 0.75 (not
//! HR = 1.0): no thermodynamic barrier prevents HR=10 if the continuous
//! input keeps feeding the surplus. At steady state, HR ≈ 0.75 + input/(rate).
//! This step measures which input dominates.
//!
//! ## Scenarios tested
//!
//! 1. `baseline`: current supersaturation (reference)
//! 2. `transpiration=0`: cut plant transpiration (#77, FAO-56).
//!    Candidate #1 for the global surplus over vegetated land.
//! 3. `uplift=0`: cut the `humidity_surface → humidity_upper` transport
//! 4. `oro_lift=0`: cut the other pathway (oro convection + lift advection)
//! 5. `subsidence=0`: directional sanity check, should WORSEN HR p99 if the
//!    causality really is upper-pump (by blocking the upper→surface return)
//! 6. `cond_rate×10`: sanity check, force the drain. HR p99 should drop
//!    toward 0.75 if the asymptote really is the limiting factor.
//!
//! Plan note: a `evap_rate × 0.1` scenario was listed in the initial plan,
//! but no global `evap_rate` exists (free-water evaporation goes through
//! Meyer/Dalton with no multiplier, land through `transpiration_coef`). The
//! `transpiration=0` scenario covers the controllable part of the lever.
//!
//! ## Metrics per elevation band
//!
//! - `HR_mean` / `HR_p99`: supersaturation `humidity_upper / saturation_upper`
//! - `cloud_mean`: mean `cloud_water` stock (downstream drizzle signature)
//! - rain/an: days of effective rain per cell
//!
//! **Eval style** (see project memory `scale_tests_eval_style`): no assert,
//! `eprintln!` + `#[ignore]`. Read the output, draw the hypothesis about the
//! dominant contributor, pick the fix afterward.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test scale_knockout_humidity_sources \
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
const RAIN_THRESHOLD: f32 = 1e-5;

/// Same bands as the other drizzle tests, for direct comparability.
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

/// Percentile index in integer arithmetic, deterministic and without cast.
/// `p_promille` ∈ [0, 1000] (ex: 990 = p99).
fn percentile_idx(len: usize, p_promille: u32) -> usize {
    if len == 0 {
        return 0;
    }
    let p = usize::try_from(p_promille).expect("u32 fits usize");
    ((len - 1) * p / 1000).min(len - 1)
}

#[derive(Debug, Clone)]
struct BandStat {
    name: &'static str,
    cells: usize,
    hr_mean: f64,
    hr_p99: f64,
    cloud_mean: f64,
    rain_per_year: f64,
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

/// Per-band accumulators filled tick by tick.
struct BandAccum {
    /// All the `HR_upper` observations (1 sample/day × cells).
    hr_obs: Vec<Vec<f32>>,
    /// Cumulative sum of `cloud_water` (to be divided by `n_cells` × `n_days`).
    cloud_sum: Vec<f64>,
    /// Count of effective rain days summed over the band's cells.
    rainy_days_sum: Vec<u64>,
}

fn run_scenario(label: &str, atmo: &AtmosphereParams, temp: &TemperatureParams) -> Vec<BandStat> {
    eprintln!("\n--- scenario: {label} ---");
    let t0 = std::time::Instant::now();

    let mut sim = build_sim(atmo.clone());
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    let band_cells = build_band_cells(&sim);
    let t_offset = temp.lapse_rate * atmo.upper_layer_altitude_m / 1000.0;

    let mut accum = BandAccum {
        hr_obs: band_cells
            .iter()
            .map(|c| Vec::with_capacity(c.len() * usize::try_from(TOTAL_DAYS).unwrap_or(0)))
            .collect(),
        cloud_sum: vec![0.0_f64; BANDS.len()],
        rainy_days_sum: vec![0_u64; BANDS.len()],
    };

    for _day in 0..TOTAL_DAYS {
        sim.step();
        let cells_slice = sim.grid().cells_slice();
        let precip = sim.last_precipitation();
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
                accum.hr_obs[band_idx].push(hr);
                accum.cloud_sum[band_idx] += f64::from(cell.cloud_water);
                if precip[i].rain > RAIN_THRESHOLD {
                    accum.rainy_days_sum[band_idx] += 1;
                }
            }
        }
    }

    let total_days_f64 = f64::from(u32::try_from(TOTAL_DAYS).expect("fits u32"));
    let mut out = Vec::new();
    for (band_idx, (name, _, _)) in BANDS.iter().enumerate() {
        let cells = band_cells[band_idx].len();
        if cells == 0 {
            continue;
        }
        let obs_vec = &mut accum.hr_obs[band_idx];
        obs_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let hr_p99 = f64::from(obs_vec[percentile_idx(obs_vec.len(), 990)]);
        let sum: f64 = obs_vec.iter().map(|&h| f64::from(h)).sum();
        let n_obs_f = f64::from(u32::try_from(obs_vec.len()).expect("obs fits u32"));
        let hr_mean = sum / n_obs_f;

        let cells_f64 = f64::from(u32::try_from(cells).expect("cells fits u32"));
        let cloud_mean = accum.cloud_sum[band_idx] / (cells_f64 * total_days_f64);

        let rainy_total = accum.rainy_days_sum[band_idx];
        let rainy_f = f64::from(u32::try_from(rainy_total).expect("rainy fits u32"));
        let rain_per_year = rainy_f * 365.0 / (cells_f64 * total_days_f64);

        out.push(BandStat {
            name,
            cells,
            hr_mean,
            hr_p99,
            cloud_mean,
            rain_per_year,
        });
    }

    eprintln!("  scenario finished in {:.1}s", t0.elapsed().as_secs_f64());
    out
}

fn print_metric_table(
    metric_name: &str,
    fmt: &str,
    scenarios: &[(String, Vec<BandStat>)],
    extract: impl Fn(&BandStat) -> f64,
) {
    eprintln!("\n=== {metric_name} ===");
    eprint!("  {:>10}", "bande");
    for (label, _) in scenarios {
        eprint!("  {label:>14}");
    }
    eprintln!();
    eprintln!("  {}", "-".repeat(10 + 16 * scenarios.len()));
    let n_bands = scenarios[0].1.len();
    for band_idx in 0..n_bands {
        let name = scenarios[0].1[band_idx].name;
        let cells = scenarios[0].1[band_idx].cells;
        eprint!("  {name:>10}");
        for (_, stats) in scenarios {
            let v = stats.get(band_idx).map_or(0.0, &extract);
            match fmt {
                "f1" => eprint!("  {v:>14.1}"),
                "f3" => eprint!("  {v:>14.3}"),
                "f4" => eprint!("  {v:>14.4}"),
                _ => eprint!("  {v:>14}"),
            }
        }
        eprintln!("   ({cells} cells)");
    }
}

fn print_comparison(scenarios: &[(String, Vec<BandStat>)]) {
    eprintln!(
        "\n\n=== Planetary drizzle (#63 Phase 4 Step 2), knockout humidity_upper sources (seed {SEED}, {YEARS} years) ==="
    );

    print_metric_table("HR_upper mean", "f3", scenarios, |s| s.hr_mean);
    print_metric_table("HR_upper p99 (sursaturation)", "f3", scenarios, |s| {
        s.hr_p99
    });
    print_metric_table("cloud_mean (mm LWP)", "f4", scenarios, |s| s.cloud_mean);
    print_metric_table("rain/year (effective rain days)", "f1", scenarios, |s| {
        s.rain_per_year
    });

    eprintln!(
        "
Expected reading:
  - If `ground_evap=0` drops HR p99, that's the source of the global surplus.
  - If `uplift=0` or `oro_lift=0` drops HR p99 over the peaks, that's the transport.
  - If `subsidence=0` WORSENS HR p99, directional causality confirmed.
  - If `cond_rate×10` brings HR p99 back toward 0.75, the model's asymptote is
    indeed the limiting factor (not some other hidden dynamic).

The likely fix isn't calibration but structural: anchor the
`step_cloud_dynamics` formula to Clausius-Clapeyron (drain proportional to
surplus above saturation, not above 0.75), a thermodynamic barrier by
construction.
Cf project memory `feedback_anchor_physics_over_tuning`.
"
    );
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn drizzle_knockout_humidity_sources() {
    let baseline = AtmosphereParams::default();
    let temp = TemperatureParams::default();

    let transpiration_off = AtmosphereParams {
        transpiration_coef: 0.0,
        ..baseline.clone()
    };
    let uplift_off = AtmosphereParams {
        uplift_rate: 0.0,
        ..baseline.clone()
    };
    let oro_off = AtmosphereParams {
        orographic_lift_coef: 0.0,
        ..baseline.clone()
    };
    let cond_x10 = AtmosphereParams {
        condensation_rate: baseline.condensation_rate * 10.0,
        ..baseline.clone()
    };

    let scenarios = vec![
        (
            "baseline".to_string(),
            run_scenario("baseline", &baseline, &temp),
        ),
        (
            "transpiration=0".to_string(),
            run_scenario(
                "transpiration_coef=0 (cuts plant transpiration)",
                &transpiration_off,
                &temp,
            ),
        ),
        (
            "uplift=0".to_string(),
            run_scenario(
                "uplift_rate=0 (cuts base surface→upper transport)",
                &uplift_off,
                &temp,
            ),
        ),
        (
            "oro_lift=0".to_string(),
            run_scenario(
                "orographic_lift_coef=0 (cuts oro convection + lift advection)",
                &oro_off,
                &temp,
            ),
        ),
        (
            "cond_rate×10".to_string(),
            run_scenario(
                "condensation_rate × 10 (sanity drain, HR p99 must→0.75)",
                &cond_x10,
                &temp,
            ),
        ),
    ];

    print_comparison(&scenarios);
}
