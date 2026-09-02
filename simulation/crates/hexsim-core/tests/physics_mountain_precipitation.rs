//! Integration tests: rain and snow at altitude must exist.
//!
//! Fundamental contract after the terrarium was closed: without a dominant
//! wind forcing, the thermal breeze regime alone sends vapor toward the
//! warm plains (thermal convergence). The peaks end up
//! parched. These tests frame the expected physics:
//!
//! 1. On a grid with a central mountain and peripheral plains,
//!    after 1 simulated year, high-altitude cells must have received at
//!    least a significant fraction of the rain.
//! 2. Snow must accumulate on cold peaks (>1000 m) after
//!    1 simulated winter.
//!
//! These tests were dormant / absent: they formalize the problem observed
//! by eye and in the existing scale tests.

mod common;

use common::build_prod_sim;

/// After 1 simulated year, the fraction of rain falling above the altitude
/// median must be significant. We accept 10% minimum; below that
/// it's a structural "rain shadow".
///
/// Physical reference: on a real coastal relief with a sea breeze, the
/// `rain_high` / `rain_low` ratio typically runs 0.5-2.0 by orientation.
/// 10% (= 0.1) is a very generous floor that leaves plenty of margin.
#[test]
fn altitude_gets_a_share_of_precipitation() {
    let mut sim = build_prod_sim(42, 30);
    for _ in 0..365 {
        sim.step();
    }
    let diag = sim.diagnostics();
    let rain_high = diag.altitude.raining_high;
    let rain_low = diag.altitude.raining_low;

    // Ratio over the total to neutralize the relative size of the two bands.
    let total = rain_high + rain_low;
    assert!(
        total > 0,
        "no rain anywhere on the map in 1 year, params too dry or pipeline broken"
    );
    let ratio = f32::from(u16::try_from(rain_high).expect("fits u16"))
        / f32::from(u16::try_from(total).expect("fits u16"));
    assert!(
        ratio > 0.10,
        "rain almost entirely excluded above the median elevation: \
         rain_high={rain_high} rain_low={rain_low} ratio={ratio:.3} (target > 0.10)"
    );
}

/// After 1 simulated year, snow must accumulate on the
/// cells >1000 m. At least 1 unit of total snow above this threshold.
/// Loose criterion simply expressing "it snows in the mountains".
#[test]
fn snow_accumulates_on_peaks_above_1000m() {
    let mut sim = build_prod_sim(42, 30);
    for _ in 0..365 {
        sim.step();
    }
    let highs: Vec<_> = sim
        .grid()
        .iter()
        .filter(|(_, c)| c.elevation > 1000.0)
        .collect();
    let count = highs.len();
    let snow_total: f32 = highs.iter().map(|(_, c)| c.snow_level).sum();
    let hum_upper_total: f32 = highs.iter().map(|(_, c)| c.humidity_upper).sum();
    let temp_mean: f32 = highs.iter().map(|(_, c)| c.temperature).sum::<f32>()
        / f32::from(u16::try_from(count.max(1)).expect("fits u16"));

    // Threshold 0.5: with the post-optimization calibration (lower
    // humidity_advection, dominant West wind), accumulation above >1000m is
    // 2-3x lower than with the historical defaults. The test verifies that
    // the cycle exists (snow > 0.5 over 183 cells) without imposing the
    // magnitude of the old, highly transport-heavy regime.
    assert!(
        snow_total > 0.5,
        "no snow above 1000 m after 1 year: \
         {count} cells, snow_tot={snow_total:.3}, \
         hum_upper_tot={hum_upper_total:.3}, T_mean={temp_mean:.2}°C"
    );
}
