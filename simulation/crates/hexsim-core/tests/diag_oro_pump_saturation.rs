//! Structural diagnostic (#156): how often the orographic pump's rate law
//! runs at its ceiling, per elevation band.
//!
//! The pump's exported fraction (`atmosphere::uplift::oro_pump_rate`)
//! depends on exactly two things: the tick coefficient
//! (`orographic_lift_coef / TICKS_PER_DAY`, boosted by `HEXSIM_ORO_SUBSAMPLE`
//! when the cadence is coarsened) and `Σ Δz⁺`, the sum of the positive
//! elevation gaps toward the 6 toric neighbours. `Σ Δz⁺` is a property of
//! the TERRAIN, fixed at generation and never modified by the sim, so the
//! saturation profile is computable without running a single tick — which
//! is why this file generates the map and stops there. What the sim state
//! decides is only WHETHER a cell pumps at all (it needs moisture and at
//! least one higher neighbour), and above 300 m essentially every cell
//! holds some `humidity_surface` at every hour.
//!
//! Reference measurement it reproduces: JOURNAL 2026-09-05 reported, for
//! r30 / seed 42 / hourly cadence, the share of pumping cell-passes whose
//! unclamped rate exceeded the legacy 0.30 cap as 16 % / 78 % / 64 % / 29 %
//! over the four bands. If the terrain-only computation below lands on the
//! same numbers, the instrument is validated and the legacy cap really was
//! the physics on the slopes.
//!
//! Eval style: `#[ignore]`, `eprintln!`, no assert. Env `HEXSIM_DIAG_SEED`,
//! `HEXSIM_DIAG_RADIUS`, `HEXSIM_DIAG_COEF_FACTOR` (multiplies the shipped
//! coefficient, for the x3 / /3 sensitivity columns).
//!
//! ```text
//! cargo test --release -p hexsim-core --test diag_oro_pump_saturation \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::terrain::{TerrainParams, generate_terrain};

const TICKS_PER_DAY: f32 = 24.0;
/// The rate the pre-#156 law clamped to.
const LEGACY_CAP: f32 = 0.30;
/// "At the ceiling" for the shipped exponential: within 5 % of its own
/// asymptote, i.e. the coefficient has stopped being readable there.
const EXPONENTIAL_CEILING: f32 = 0.95;

const BANDS: [(&str, f32, f32); 4] = [
    ("< 300 m", f32::NEG_INFINITY, 300.0),
    ("300-800 m", 300.0, 800.0),
    ("800-1500 m", 800.0, 1500.0),
    (">= 1500 m", 1500.0, f32::INFINITY),
];

fn env_f32(key: &str, fallback: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn env_u32(key: &str, fallback: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

#[test]
#[ignore = "diagnostic, run explicitly"]
fn diag_oro_pump_saturation_by_altitude_band() {
    let seed = env_u32("HEXSIM_DIAG_SEED", 42);
    let radius = i32::try_from(env_u32("HEXSIM_DIAG_RADIUS", 30)).expect("radius fits i32");
    let coef_factor = env_f32("HEXSIM_DIAG_COEF_FACTOR", 1.0);

    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed,
            ..TerrainParams::default()
        },
    );

    let coef = AtmosphereParams::default().orographic_lift_coef * coef_factor / TICKS_PER_DAY;
    let cells = grid.cells_slice();

    eprintln!(
        "\n== Orographic pump saturation, seed {seed}, radius {radius}, \
         coef x{coef_factor} (per tick: {coef:.6} /m) ==\n"
    );
    eprintln!(
        "{:<12} {:>7} {:>9} {:>11} {:>11} {:>11} {:>11}",
        "band", "cells", "pumping", "med SumDz+", "legacy sat", "legacy rate", "new rate"
    );

    for (label, lo, hi) in BANDS {
        let mut pumping = 0_u32;
        let mut cells_in_band = 0_u32;
        let mut clamped = 0_u32;
        let mut ceilinged = 0_u32;
        let mut sums = Vec::new();
        let mut legacy_rates = Vec::new();
        let mut new_rates = Vec::new();

        for (i, cell) in cells.iter().enumerate() {
            if cell.elevation < lo || cell.elevation >= hi {
                continue;
            }
            cells_in_band += 1;
            let mut sum_positive = 0.0_f32;
            for j in grid.neighbor_indices_toric(i) {
                let gap = cells[j].elevation - cell.elevation;
                if gap > 0.0 {
                    sum_positive += gap;
                }
            }
            // Same gate as `fill_oro_outflow`: no higher neighbour, no pump.
            if sum_positive < 1e-6 {
                continue;
            }
            pumping += 1;
            let x = coef * sum_positive;
            let legacy = x.clamp(0.0, LEGACY_CAP);
            let exponential = 1.0 - (-x).exp();
            if x > LEGACY_CAP {
                clamped += 1;
            }
            if exponential > EXPONENTIAL_CEILING {
                ceilinged += 1;
            }
            sums.push(sum_positive);
            legacy_rates.push(legacy);
            new_rates.push(exponential);
        }

        let pumping_f = f64::from(pumping).max(1.0);
        eprintln!(
            "{:<12} {cells_in_band:>7} {pumping:>9} {:>11.0} {:>10.1}% {:>11.3} {:>11.3}",
            label,
            median(&mut sums),
            100.0 * f64::from(clamped) / pumping_f,
            median(&mut legacy_rates),
            median(&mut new_rates),
        );
        eprintln!(
            "{:<12} {:>7} {:>9} {:>11} {:>10.1}% {:>11} {:>11}",
            "",
            "",
            "",
            "",
            100.0 * f64::from(ceilinged) / pumping_f,
            "(new >0.95)",
            ""
        );
    }
    eprintln!(
        "\n'legacy sat' = share of pumping cells whose unclamped rate exceeds \
         {LEGACY_CAP} (the pre-#156 cap did the physics for them).\n\
         'new >0.95'  = share whose exponential rate is within 5 % of 1, the \
         only ceiling the shipped law has left.\n"
    );
}
