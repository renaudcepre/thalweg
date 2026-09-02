//! Climatological invariant: the altitude-plain thermal gradient must stay
//! close to the configured `lapse_rate` (6.5 °C/km ICAO standard).
//!
//! This test complements `effective_lapse_rate_matches_params` (in
//! `temperature.rs`) which isolates the target formula without advection,
//! seasonality, coupling. Here we run a "prod-like" world for 2 years and
//! measure the effective temp-vs-elev slope. Significant divergence signals
//! that a dynamic term (wind advection, `water_cooling`, cloud albedo)
//! dominates the configured gradient.
//!
//! Tolerance: ratio `effective / params.lapse_rate` in [0.3, 1.7].
//! - < 0: thermal inversion (plain colder than mountain)
//! - 0-0.3: gradient crushed, configured lapse ignored
//! - 0.3-1.7: acceptable dynamics (± 70 % around target)
//! - > 1.7: hyper-cooling (unlikely, altitude-cooling advection)
//!
//! The [0.3, 1.7] range is generous; the real window after convergence
//! should be closer to [0.7, 1.3] on a well-tuned system. This test flags
//! structural regressions, not fine-tuning.

mod common;

use common::build_prod_sim;
use hexsim_core::diagnostics::effective_lapse_rate_c_per_km;

const PARAMS_LAPSE_RATE: f32 = 6.5;
const TOLERANCE_MIN_RATIO: f32 = 0.3;
const TOLERANCE_MAX_RATIO: f32 = 1.7;

/// Radius 30 = ~2791 cells, consistent with other scale tests.
/// Warmup 1 year then average over 1 year (smooths seasonality + cycles).
#[test]
fn effective_lapse_rate_close_to_params_after_two_years() {
    let mut sim = build_prod_sim(42, 30);

    for _ in 0..365 {
        sim.step();
    }

    let mut samples: Vec<f32> = Vec::with_capacity(365);
    for _ in 0..365 {
        sim.step();
        samples.push(effective_lapse_rate_c_per_km(sim.grid()));
    }

    // 365 samples: well below f32 mantissa (24 bits = 16M).
    #[allow(clippy::cast_precision_loss)]
    let samples_count = samples.len() as f32;
    let mean_rate = samples.iter().copied().sum::<f32>() / samples_count;
    let ratio = mean_rate / PARAMS_LAPSE_RATE;

    let min_sample = samples.iter().copied().fold(f32::INFINITY, f32::min);
    let max_sample = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    eprintln!(
        "effective lapse rate annual mean = {mean_rate:.3} °C/km  (params = {PARAMS_LAPSE_RATE:.2} °C/km, ratio = {ratio:.2})"
    );
    eprintln!("  distribution over 365 samples: min = {min_sample:.3}, max = {max_sample:.3}");

    assert!(
        ratio > TOLERANCE_MIN_RATIO && ratio < TOLERANCE_MAX_RATIO,
        "effective lapse rate {mean_rate:.3} °C/km is outside [{TOLERANCE_MIN_RATIO}×, {TOLERANCE_MAX_RATIO}×] of params \
         {PARAMS_LAPSE_RATE} °C/km (ratio {ratio:.2}). Altitude-plain thermal gradient \
         is crushed or inverted; see wind advection, water_cooling, \
         cloud albedo.",
    );
}
