//! Instrument #104: 10-year multi-seed protocol for the SI conversion of
//! `effective_elevation` (surplus mm → m).
//!
//! No physical balance change without global metrics at scale. The #103
//! ablation (180 d + 12 h, seed 42 r45) motivated
//! the fix; THIS instrument judges it over 10 years and 5 seeds (the #56
//! instrument's seed set). Eval style: measurements and observations, no
//! tuning target, only hard guard-rails (finite budget, strict
//! conservation).
//!
//! Per seed (r30, 3650 ticks), over the LAST year (37 samples,
//! 1 every 10 days):
//!   - strict conservation of the total water budget over the 10 years
//!   - hydro regime: `river_cells`, `max_discharge`, `overflow_cells`
//!     (same definitions as `diagnostics.rs`)
//!   - average discharge per elevation band (seed terrain quartiles,
//!     identical before/after since the terrain doesn't move)
//!   - "stagnant pools on slopes": cells with surplus > 100 mm, STAGNANT
//!     (`discharge < RIVER_THRESHOLD`, without this filter we'd count
//!     riverbeds, where a large surplus is water IN TRANSIT piled up at
//!     the bottleneck by the finite conveyance of `flow_rate` mm/m), whose
//!     terrain drops by at least `slope_full_mobility` (≈ 6 m, the slope at
//!     which the model makes all water mobile) toward a neighbor. This is
//!     the hybrid artifact diagnosed in #103 (water piled at ~1 mm per
//!     meter of drop), expected ≈ 0 after #104. Second counter at ≥ 20 m
//!     for extreme cases.
//!
//! Run:
//!   `cargo nextest run --no-capture --run-ignored=only -E 'binary(diag_units_10y_multiseed)'`
//! A/B baseline: copy this file as-is into a worktree at the commit
//! parent to the fix (the instrument only uses the public API, present on
//! both sides) and rerun the same command.

mod common;

use common::{build_prod_sim, total_water_budget};
use hexsim_core::simulation::Simulation;

const RADIUS: i32 = 30;
const YEARS: u64 = 10;
const TOTAL_TICKS: u64 = YEARS * 365;
/// Sampling for the last year: 1 measurement every 10 days.
const SAMPLE_EVERY: u64 = 10;
/// Same threshold as `river_threshold` in `diagnostics.rs`; the numbers
/// must be comparable to `just diag` and to the #103 ablation.
const RIVER_THRESHOLD: f32 = 0.5;
/// Surplus (mm above `water_capacity`) that qualifies as a "pool".
const NAPPE_SURPLUS_MM: f32 = 100.0;
/// Terrain drop (m, per edge) above which the MFD mobilizes 100% of the
/// water; same derivation as `HydroParams::slope_full_mobility`. A pool
/// > 100 mm that stagnates on such an edge is the #103 artifact.
const STEEP_DROP_M: f32 =
    hexsim_core::dynamics::STEEP_SLOPE_GRADE * hexsim_core::dynamics::CELL_SPACING_M;

/// Accumulator for the last year's samples.
#[derive(Default)]
struct Acc {
    samples: u32,
    river_cells_sum: f64,
    overflow_cells_sum: f64,
    max_discharge_peak: f32,
    nappes_steep_sum: f64,
    nappes_steep_peak: usize,
    nappes_20m_sum: f64,
    nappes_20m_peak: usize,
    /// Largest surplus (mm) seen on a cell on a slope ≥ `STEEP_DROP_M`,
    /// the order of magnitude of the pools from the #103 diag (100-230 mm).
    nappe_surplus_peak: f32,
    band_discharge_sum: [f64; 4],
}

fn sample(sim: &Simulation, band_edges: &[f32; 3], acc: &mut Acc) {
    let grid = sim.grid();
    let cells = grid.cells_slice();
    let discharge = sim.discharge_map();

    let mut river = 0_usize;
    let mut overflow = 0_usize;
    let mut nappes_steep = 0_usize;
    let mut nappes_20 = 0_usize;

    for (i, cell) in cells.iter().enumerate() {
        let d = discharge[i];
        if d > RIVER_THRESHOLD {
            river += 1;
        }
        if d > acc.max_discharge_peak {
            acc.max_discharge_peak = d;
        }
        let band = band_index(cell.elevation, band_edges);
        acc.band_discharge_sum[band] += f64::from(d);

        if cell.water_level > cell.water_capacity {
            overflow += 1;
        }
        let surplus = cell.water_level - cell.water_capacity;
        if surplus > NAPPE_SURPLUS_MM && d < RIVER_THRESHOLD {
            let mut drop = 0.0_f32;
            for &j in &grid.neighbor_indices_toric(i) {
                drop = drop.max(cell.elevation - cells[j].elevation);
            }
            if drop >= STEEP_DROP_M {
                nappes_steep += 1;
                if surplus > acc.nappe_surplus_peak {
                    acc.nappe_surplus_peak = surplus;
                }
            }
            if drop >= 20.0 {
                nappes_20 += 1;
            }
        }
    }

    acc.samples += 1;
    acc.river_cells_sum += approx_f64(river);
    acc.overflow_cells_sum += approx_f64(overflow);
    acc.nappes_steep_sum += approx_f64(nappes_steep);
    acc.nappes_steep_peak = acc.nappes_steep_peak.max(nappes_steep);
    acc.nappes_20m_sum += approx_f64(nappes_20);
    acc.nappes_20m_peak = acc.nappes_20m_peak.max(nappes_20);
}

fn approx_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("cell counts fit u32"))
}

fn band_index(elevation: f32, edges: &[f32; 3]) -> usize {
    edges.iter().position(|&e| elevation < e).unwrap_or(3)
}

/// Elevation quartiles of the (frozen) terrain → bounds of the 4 bands.
fn elevation_quartiles(sim: &Simulation) -> [f32; 3] {
    let mut elevs: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation)
        .collect();
    elevs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = elevs.len();
    [elevs[n / 4], elevs[n / 2], elevs[3 * n / 4]]
}

fn run_seed(seed: u32) {
    let mut sim = build_prod_sim(seed, RADIUS);
    let band_edges = elevation_quartiles(&sim);
    let band_cells = {
        let mut counts = [0_usize; 4];
        for c in sim.grid().cells_slice() {
            counts[band_index(c.elevation, &band_edges)] += 1;
        }
        counts
    };
    let initial_budget = total_water_budget(&sim);

    let mut acc = Acc::default();
    for t in 1..=TOTAL_TICKS {
        sim.step();
        if t > TOTAL_TICKS - 365 && t.is_multiple_of(SAMPLE_EVERY) {
            sample(&sim, &band_edges, &mut acc);
        }
    }

    let final_budget = total_water_budget(&sim);
    let drift = final_budget - initial_budget;
    let n_samples = f64::from(acc.samples);

    eprintln!("=== #104 / 10 years / seed {seed} (r{RADIUS}) ===");
    eprintln!(
        "  total water budget : {initial_budget:.1} -> {final_budget:.1} (drift {drift:+.3} mm, {:.5} %)",
        f64::from(drift) / f64::from(initial_budget) * 100.0
    );
    eprintln!("  last year's regime ({} samples):", acc.samples);
    eprintln!(
        "    river_cells (avg)    : {:.1}",
        acc.river_cells_sum / n_samples
    );
    eprintln!("    max_discharge (peak) : {:.1}", acc.max_discharge_peak);
    eprintln!(
        "    overflow_cells (avg) : {:.1}",
        acc.overflow_cells_sum / n_samples
    );
    eprintln!(
        "    stagnant pools on steep slope (>= {STEEP_DROP_M:.1} m) : avg {:.1} / peak {}",
        acc.nappes_steep_sum / n_samples,
        acc.nappes_steep_peak
    );
    eprintln!(
        "    stagnant pools on slope >= 20 m : avg {:.1} / peak {}",
        acc.nappes_20m_sum / n_samples,
        acc.nappes_20m_peak
    );
    eprintln!(
        "    max surplus on steep slope : {:.0} mm",
        acc.nappe_surplus_peak
    );
    eprintln!(
        "  average discharge/cell by elevation band (bounds {:.0}/{:.0}/{:.0} m) :",
        band_edges[0], band_edges[1], band_edges[2]
    );
    for (b, label) in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"].iter().enumerate() {
        eprintln!(
            "    {label:<9}: {:.2} ({} cells)",
            acc.band_discharge_sum[b] / n_samples / approx_f64(band_cells[b]),
            band_cells[b]
        );
    }

    // Hard guard-rails (eval-style: no target, only the invariant).
    assert!(
        final_budget.is_finite(),
        "water budget not finite: {final_budget}"
    );
    // Relative: the r30 budget is ~10^5 mm, the f32 accumulation over 3650
    // ticks leaves rounding noise well below 1e-3 relative (cf. tolerance
    // of `total_mass_conservation_strict`: 5e-4 relative at r3). A real
    // physical leak exceeds this threshold by several orders of magnitude.
    assert!(
        (f64::from(drift) / f64::from(initial_budget)).abs() < 1e-3,
        "strict conservation violated over 10 years: drift {drift} mm on {initial_budget} mm"
    );
}

/// Transient probe (#104): annual totals for snow / surface water /
/// budget over 15 years (seed 42, r30). The #56 drift is measured over a
/// [2-year warm-up + 5 years] window; window-dependent when the EQUILIBRIUM
/// shifts (lesson #60: two false "runaways" were window artifacts).
/// This probe shows whether the post-#104 regime converges (annual
/// drift → 0) and at what snowpack level.
#[test]
#[ignore = "instrument #104, transient probe 15 years (seed 42, r30)"]
fn snow_transient_seed42_15y() {
    let mut sim = build_prod_sim(42, RADIUS);
    eprintln!("=== #104 / snow-water transient / seed 42 (r{RADIUS}, 15 years) ===");
    eprintln!("  year ; total_snow ; surface_water ; total_budget");
    for year in 1..=15 {
        for _ in 0..365 {
            sim.step();
        }
        let snow: f32 = sim.grid().iter().map(|(_, c)| c.snow_level).sum();
        let water: f32 = sim.grid().iter().map(|(_, c)| c.water_level).sum();
        let budget = total_water_budget(&sim);
        eprintln!("  {year:>2} ; {snow:>8.0} ; {water:>8.0} ; {budget:>8.0}");
    }
}

/// Exact reproduction of the #103 ablation frame: seed 42, r45,
/// 180 days, THE frame where the "lake on a slope" diag was observed
/// (pools of 100-230 mm stable on 50 m slopes). A single sample, on day
/// 180: the accumulator's averages ARE the instantaneous values.
#[test]
#[ignore = "instrument #104, #103 ablation frame (seed 42, r45, 180 d)"]
fn nappes_ablation_frame_seed42_r45() {
    let mut sim = build_prod_sim(42, 45);
    let band_edges = elevation_quartiles(&sim);
    for _ in 0..180 {
        sim.step();
    }
    let mut acc = Acc::default();
    sample(&sim, &band_edges, &mut acc);

    eprintln!("=== #104 / #103 ablation frame / seed 42, r45, day 180 ===");
    eprintln!("  river_cells             : {:.0}", acc.river_cells_sum);
    eprintln!("  max_discharge           : {:.1}", acc.max_discharge_peak);
    eprintln!("  overflow_cells          : {:.0}", acc.overflow_cells_sum);
    eprintln!(
        "  stagnant pools on steep slope (>= {STEEP_DROP_M:.1} m) : {:.0}",
        acc.nappes_steep_sum
    );
    eprintln!(
        "  stagnant pools on slope >= 20 m : {:.0}",
        acc.nappes_20m_sum
    );
    eprintln!(
        "  max surplus on steep slope : {:.0} mm",
        acc.nappe_surplus_peak
    );

    // Named list of large surpluses on steep slopes, stagnant OR in
    // transit; the discharge column tells them apart (diagnostic
    // 2026-07-11: the 9 cells from the first run were the beds of the 2
    // main rivers, discharge 13-43/day, not pools).
    let grid = sim.grid();
    let cells = grid.cells_slice();
    let coords = grid.coords_slice();
    let discharge = sim.discharge_map();
    for (i, cell) in cells.iter().enumerate() {
        let surplus = cell.water_level - cell.water_capacity;
        if surplus <= NAPPE_SURPLUS_MM {
            continue;
        }
        let mut drop = 0.0_f32;
        for &j in &grid.neighbor_indices_toric(i) {
            drop = drop.max(cell.elevation - cells[j].elevation);
        }
        if drop >= STEEP_DROP_M {
            let nature = if discharge[i] < RIVER_THRESHOLD {
                "STAGNANT"
            } else {
                "transit (riverbed)"
            };
            eprintln!(
                "    {:?} elev {:.0} m, surplus {:.0} mm, max drop {:.1} m, discharge {:.2}, {nature}",
                coords[i], cell.elevation, surplus, drop, discharge[i]
            );
        }
    }
}

#[test]
#[ignore = "instrument #104, slow (10 years r30), just diag-tools or -E 'binary(diag_units_10y_multiseed)'"]
fn units_10y_seed_42() {
    run_seed(42);
}

#[test]
#[ignore = "instrument #104, slow (10 years r30)"]
fn units_10y_seed_7() {
    run_seed(7);
}

#[test]
#[ignore = "instrument #104, slow (10 years r30)"]
fn units_10y_seed_99() {
    run_seed(99);
}

#[test]
#[ignore = "instrument #104, slow (10 years r30)"]
fn units_10y_seed_1234() {
    run_seed(1234);
}

#[test]
#[ignore = "instrument #104, slow (10 years r30)"]
fn units_10y_seed_2026() {
    run_seed(2026);
}
