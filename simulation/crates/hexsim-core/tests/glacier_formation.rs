//! Regressions around glacier formation and lakes.
//!
//! The "glacier" mechanism (slow melt beyond `glacier_threshold`) was added
//! so that peaks accumulate a perennial stock. But it can trigger on cells
//! that shouldn't, notably deep water bodies that freeze massively in
//! winter and trivially flip into the glacier regime. In summer, the
//! residual snow from these "ocean glaciers" stays visible despite
//! temperatures > 30°C.

use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::snow::{SnowForcing, SnowParams, step_snow};
use hexsim_core::time::{TICKS_PER_DAY, TICKS_PER_DAY_F32};

// v0.3.0 PR2 (#38): `step_snow` is now called every hour (Tier 1), with
// internal rates divided by `TICKS_PER_DAY`. To test N days of effect,
// iterate `N * TICKS_PER_DAY` times.
const HOURS_PER_DAY: u64 = TICKS_PER_DAY;

/// "Sunny noon" forcing for MELT phases (#60 Phase 1): melt is an SI
/// energy balance, not just a rate calibrated on temperature alone (cf
/// `snow.rs`, `sunny_noon` in the unit test module). Without solar input
/// a calm clear night barely melts anything, even at T > 0. These
/// integration tests want to isolate "does the snow melt enough over the
/// simulated duration", not replay a day/night cycle, so we provide a
/// constant solar forcing during phases where melt is expected.
fn sunny_forcing(ff: &[f32]) -> SnowForcing<'_> {
    SnowForcing {
        beam_w_m2: 800.0,
        ground_albedo: 0.3,
        flux_factor: ff,
        wind_mag: &[],
        rain_last_tick: &[],
        gw_max_capacity: 100.0,
    }
}

/// A sea-level cell (deep ocean) must not retain snow in summer, even if
/// it froze massively in winter.
#[test]
fn deep_water_does_not_become_permanent_glacier() {
    let mut current = HexGrid::from_radius(0);
    let c = HexCoord::new(0, 0);

    if let Some(cell) = current.get_mut(c) {
        cell.elevation = -500.0;
        // Phase 3 (#32) : rescale ×200 (50 → 10000 mm = 10 m).
        cell.water_level = 10000.0;
        cell.snow_level = 0.0;
    }

    let params = SnowParams::default();
    let mut next = current.clone();

    // Winter: 90 days at -5°C → the ocean can freeze (v0.3.0: N*24 sub-ticks).
    // FREEZE phase: solar forcing plays no role in the
    // `temperature < freeze_threshold` branch, `night_calm()` is enough.
    for _ in 0..(90 * HOURS_PER_DAY) {
        if let Some(cell) = current.get_mut(c) {
            cell.temperature = -5.0;
        }
        step_snow(&current, &mut next, &params, &SnowForcing::night_calm());
        std::mem::swap(&mut current, &mut next);
    }

    let snow_after_winter = current.get(c).unwrap().snow_level;
    let water_after_winter = current.get(c).unwrap().water_level;
    println!(
        "End of winter: snow={snow_after_winter:.2} water={water_after_winter:.2} (initial water=50)"
    );

    // Summer: 180 days at +25°C → the snow must melt. MELT phase: under
    // the new SI balance, melt requires energy (not just T > 0), cell at
    // elevation=-500 < glacier_min_elevation, so never in the glacier
    // regime; we give it sun so the expected melt happens.
    let ff = vec![1.0; current.len()];
    for _ in 0..(180 * HOURS_PER_DAY) {
        if let Some(cell) = current.get_mut(c) {
            cell.temperature = 25.0;
        }
        step_snow(&current, &mut next, &params, &sunny_forcing(&ff));
        std::mem::swap(&mut current, &mut next);
    }

    let snow_after_summer = current.get(c).unwrap().snow_level;
    let water_after_summer = current.get(c).unwrap().water_level;
    println!("End of summer: snow={snow_after_summer:.2} water={water_after_summer:.2}");

    // Phase 3 (#32): rescale x200 (0.5 -> 100 mm of "residual" snow threshold).
    assert!(
        snow_after_summer < 100.0,
        "Residual ocean glacier: {snow_after_summer:.2} mm of snow persists \
         at 25°C after a full summer (expected < 100). Massive winter freezing flipped \
         the cell into glacier regime, which blocks summer melt."
    );
}

/// At +30°C in full summer, snow must disappear completely within a few
/// weeks. No physically ungrounded "residual trace".
#[test]
fn snow_fully_melts_at_high_temperature() {
    let mut current = HexGrid::from_radius(0);
    let c = HexCoord::new(0, 0);

    if let Some(cell) = current.get_mut(c) {
        cell.elevation = 0.0;
        cell.water_level = 0.0;
        // Phase 3 (#32): rescale x200 (5 -> 1000 mm = 1 m of snow).
        cell.snow_level = 1000.0;
        cell.temperature = 30.0;
    }

    let params = SnowParams::default();
    let mut next = current.clone();

    // 60 days at +30°C: equivalent to 2 months of full summer (v0.3.0: N*24 ticks).
    // MELT phase: elevation=0 < glacier_min_elevation, never in the
    // glacier regime; solar forcing so the expected full melt has the
    // energy it needs (SI balance, cf `sunny_forcing`).
    let ff = vec![1.0; current.len()];
    for _ in 0..(60 * HOURS_PER_DAY) {
        if let Some(cell) = current.get_mut(c) {
            cell.temperature = 30.0;
        }
        step_snow(&current, &mut next, &params, &sunny_forcing(&ff));
        std::mem::swap(&mut current, &mut next);
    }

    let snow_final = current.get(c).unwrap().snow_level;
    println!("After 60 days at +30°C from 5 units: snow={snow_final:.6}");

    // Phase 3 (#32): rescale x200 (0.001 -> 0.2 mm). Display threshold.
    assert!(
        snow_final < 0.2,
        "Residual snow at +30°C: {snow_final:.6} mm after 60 days. \
         Proportional melt never reaches zero, replace with an absolute rate."
    );
}

/// A mountain peak (little liquid water, direct snow precipitation) must
/// be able to form a glacier; the fix must not break this case.
#[test]
fn high_altitude_still_forms_glacier() {
    let mut current = HexGrid::from_radius(0);
    let c = HexCoord::new(0, 0);

    if let Some(cell) = current.get_mut(c) {
        cell.elevation = 1200.0;
        cell.water_level = 0.0;
        cell.snow_level = 0.0;
    }

    let params = SnowParams::default();
    let mut next = current.clone();

    // Simulates direct snowfall (step_precipitation deposits snow
    // when T<0). Here we inject manually to isolate step_snow.
    // v0.3.0 PR2: 30 mm/day → 30/24 mm/hour for an equivalent daily
    // flux, looping over N*24 ticks.
    // FREEZE phase (T<0, manual injection): forcing has no effect, night_calm().
    let snow_injection_per_hour = 30.0_f32 / TICKS_PER_DAY_F32;
    for _ in 0..(90 * HOURS_PER_DAY) {
        if let Some(cell) = current.get_mut(c) {
            cell.temperature = -10.0;
            cell.snow_level += snow_injection_per_hour;
        }
        step_snow(&current, &mut next, &params, &SnowForcing::night_calm());
        std::mem::swap(&mut current, &mut next);
    }

    let snow_after_winter = current.get(c).unwrap().snow_level;

    // Summer at +5°C: snow_after_winter expected >> glacier_threshold
    // (1000 mm) and elevation=1200 > glacier_min_elevation → glacier
    // regime, which keeps the calibrated rate independent of forcing (cf
    // `snow.rs`, `is_glacier` branch). The stock must REMAIN, not melt:
    // night_calm() is physically correct here, not a shortcut to make the
    // test pass.
    for _ in 0..(180 * HOURS_PER_DAY) {
        if let Some(cell) = current.get_mut(c) {
            cell.temperature = 5.0;
        }
        step_snow(&current, &mut next, &params, &SnowForcing::night_calm());
        std::mem::swap(&mut current, &mut next);
    }

    let snow_after_summer = current.get(c).unwrap().snow_level;
    println!(
        "Peak: snow={snow_after_winter:.2} end of winter → {snow_after_summer:.2} end of summer (+5°C)"
    );

    // Phase 3 (#32) : rescale ×200 (3 → 600 mm survivant).
    assert!(
        snow_after_summer > 600.0,
        "A true summit glacier must survive summer (glacier regime): \
         snow={snow_after_summer:.2} mm after summer"
    );
}
