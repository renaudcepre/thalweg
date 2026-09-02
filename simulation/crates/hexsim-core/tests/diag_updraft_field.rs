//! Diagnostic of the updraft field `w` for the precip trigger (save updraft, #63/#69).
//!
//! Finding from the `scale_precip_regime` harness: at `updraft_ref_ms=1.0/floor=0.1`,
//! the updraft trigger is **inert on rain** (`raining_cells` 354→359,
//! bands unchanged) and only bites into high-altitude snow (−28%). Hypothesis:
//! the factor `f = clamp(floor + w/w_ref, 0, 1)` (`precipitation.rs:189`) is
//! **saturated at 1.0** on the plain because `w = H·(−∇·v) + v·∇z` is
//! strongly positive there (the `H=1500 m` thickness amplifies the estimated
//! convergence over 130 m cells).
//!
//! This diag reads the REAL `w` field from the sim (`updraft_field()`, synoptic
//! included) and measures:
//! - the distribution of `w` (min/p10/median/mean/p90/p99/max/std, fractions
//!   `w<0` and `w>1`);
//! - mean `w` per elevation band (plain strongly positive? peaks negative?);
//! - for several candidate `w_ref` values, the factor `f` **weighted by rain**
//!   `Σ(f·rain)/Σrain`; if it is ≈1 the trigger doesn't touch the rain; we
//!   look for the `w_ref` that brings it down (gives back some bite on the
//!   plain).
//!
//! Note: `updraft_field()` is read at the end of the day (last sub-tick), the
//! rain is the 24 h cumulative total; a slight offset is accepted (diagnostic,
//! not an assertion).
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test diag_updraft_field \
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
const WARMUP_DAYS: u64 = 90;
const SAMPLE_DAYS: u64 = 2 * 365;
/// Samples the field 1 day out of N (the field varies slowly at the daily
/// scale; no point storing 730 × 2791 values).
const SAMPLE_EVERY: u64 = 6;
/// Reproduces the engine default (`precipitation.rs`).
const UPDRAFT_FLOOR: f32 = 0.1;

/// `w_ref` candidates to probe (the tested default was 1.0).
const WREF_CANDIDATES: &[f32] = &[1.0, 5.0, 20.0, 50.0, 100.0];

const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("count fits u32"))
}

fn band_index(elev: f32) -> Option<usize> {
    BANDS
        .iter()
        .position(|(_, lo, hi)| elev >= *lo && elev < *hi)
}

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed: SEED,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    let atmo = AtmosphereParams {
        // The `w` field is independent of w_ref (computed before it is
        // applied); it just needs to be > 0 for the scratch to be filled.
        updraft_ref_ms: 1.0,
        updraft_floor: UPDRAFT_FLOOR,
        ..AtmosphereParams::default()
    };
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

fn factor(w: f32, w_ref: f32) -> f32 {
    (UPDRAFT_FLOOR + w / w_ref).clamp(0.0, 1.0)
}

/// Percentile via integer index (permille ∈ [0, 1000]): avoids any
/// float→int cast (clippy `cast_*`).
fn percentile(sorted: &[f32], permille: usize) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let last = sorted.len() - 1;
    let i = last * permille / 1000;
    sorted[i.min(last)]
}

struct BandAcc {
    w_sum: f64,
    count: usize,
}

fn main_diag() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    let elevs: Vec<f32> = sim.grid().iter().map(|(_, c)| c.elevation).collect();
    let n = elevs.len();
    assert_eq!(
        sim.updraft_field().len(),
        n,
        "updraft actif (scratch rempli)"
    );

    let mut w_samples: Vec<f32> = Vec::new();
    let mut bands: Vec<BandAcc> = (0..BANDS.len())
        .map(|_| BandAcc {
            w_sum: 0.0,
            count: 0,
        })
        .collect();
    // Per w_ref: (Σ f·rain, Σ rain, Σ f cell, n cells) for both the
    // rain-weighted f AND the per-cell mean f.
    let mut wref_acc: Vec<(f64, f64, f64, usize)> =
        WREF_CANDIDATES.iter().map(|_| (0.0, 0.0, 0.0, 0)).collect();

    for day in 0..SAMPLE_DAYS {
        sim.step();
        if day % SAMPLE_EVERY != 0 {
            continue;
        }
        let w_field: Vec<f32> = sim.updraft_field().to_vec();
        let precip: Vec<f32> = sim.last_precipitation().iter().map(|r| r.rain).collect();

        for i in 0..n {
            let w = w_field[i];
            w_samples.push(w);
            if let Some(b) = band_index(elevs[i]) {
                bands[b].w_sum += f64::from(w);
                bands[b].count += 1;
            }
            let rain = precip[i];
            for (k, &w_ref) in WREF_CANDIDATES.iter().enumerate() {
                let f = factor(w, w_ref);
                let acc = &mut wref_acc[k];
                acc.0 += f64::from(f) * f64::from(rain);
                acc.1 += f64::from(rain);
                acc.2 += f64::from(f);
                acc.3 += 1;
            }
        }
    }

    print_report(&mut w_samples, &bands, &wref_acc);
}

fn print_report(w_samples: &mut [f32], bands: &[BandAcc], wref_acc: &[(f64, f64, f64, usize)]) {
    w_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_f = count_f64(w_samples.len());
    let mean = w_samples.iter().map(|&w| f64::from(w)).sum::<f64>() / n_f;
    let var = w_samples
        .iter()
        .map(|&w| (f64::from(w) - mean).powi(2))
        .sum::<f64>()
        / n_f;
    let frac_neg = count_f64(w_samples.iter().filter(|&&w| w < 0.0).count()) / n_f;
    let frac_gt1 = count_f64(w_samples.iter().filter(|&&w| w > 1.0).count()) / n_f;

    eprintln!(
        "\n\n=== Updraft field w diagnostic (seed {SEED}, sample of {} values) ===",
        w_samples.len()
    );
    eprintln!("\n  w distribution (m/s):");
    eprintln!(
        "    min={:.3} p10={:.3} median={:.3} mean={:.3} p90={:.3} p99={:.3} max={:.3} std={:.3}",
        percentile(w_samples, 0),
        percentile(w_samples, 100),
        percentile(w_samples, 500),
        mean,
        percentile(w_samples, 900),
        percentile(w_samples, 990),
        percentile(w_samples, 1000),
        var.sqrt(),
    );
    eprintln!(
        "    fraction w<0 (subsidence) = {:.1}%   |   fraction w>1 (saturates f) = {:.1}%",
        frac_neg * 100.0,
        frac_gt1 * 100.0
    );

    eprintln!("\n  average w per altitude band:");
    for (b, (name, _, _)) in BANDS.iter().enumerate() {
        let acc = &bands[b];
        if acc.count == 0 {
            continue;
        }
        eprintln!(
            "    {name:>10} : w̄ = {:>8.3} m/s   ({} cells)",
            acc.w_sum / count_f64(acc.count),
            acc.count
        );
    }

    eprintln!("\n  updraft factor vs w_ref (floor={UPDRAFT_FLOOR}):");
    eprintln!(
        "    {:>8}  {:>16}  {:>14}",
        "w_ref", "f̄ rain-weighted", "f̄ per cell"
    );
    eprintln!("    {}", "-".repeat(8 + 2 + 16 + 2 + 14));
    for (k, &w_ref) in WREF_CANDIDATES.iter().enumerate() {
        let (rain_weighted_sum, rain_mass, cell_factor_sum, cell_total) = wref_acc[k];
        let f_rain = if rain_mass > 0.0 {
            rain_weighted_sum / rain_mass
        } else {
            0.0
        };
        let f_cell = if cell_total > 0 {
            cell_factor_sum / count_f64(cell_total)
        } else {
            0.0
        };
        eprintln!("    {w_ref:>8.1}  {f_rain:>16.4}  {f_cell:>14.4}");
    }

    eprintln!(
        "\nReading:
  - If w>1 dominates (high fraction) and w̄ plain ≫ 1 → f saturated at 1 on the
    plain: the trigger can't bite the drizzle where it happens. That's the
    measured inertia.
  - 'f̄ rain-weighted' at w_ref=1.0 ≈ 1.0 would confirm that RAIN falls where
    f=1 (redundancy: rain already coincides with updraft).
  - The w_ref that brings 'f̄ rain-weighted' clearly below 1 is the scale at
    which the trigger regains bite, a recalibration candidate.
"
    );
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn updraft_field_distribution() {
    main_diag();
}
