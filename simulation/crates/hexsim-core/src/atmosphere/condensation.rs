use crate::grid::HexGrid;
use crate::physics::tetens_saturation_vapor_pressure;
use crate::temperature::{SECONDS_PER_HOUR, TemperatureParams};

use super::{AtmosphereParams, R_VAP, SURFACE_LAYER_M};

/// Precipitable water (mm) at saturation for the `upper` layer at
/// temperature `t_upper`. Phase 6 (#29): replaces the phenomenological
/// curve with the physical Clausius-Clapeyron law (Tetens), converted to
/// mm of precipitable water integrated over the layer height.
///
/// Conversion chain:
/// 1. `e_s(T)` = saturation vapor pressure via Tetens (hPa).
/// 2. `rho_vap(T) = e_s / (R_vap × T_K)`, saturation vapor density
///    (kg/m³), ideal gas law applied to vapor alone.
/// 3. `PW = rho_vap × H`, simplified vertical integration (uniform
///    density over the layer), result in mm of precipitable water
///    (1 kg/m² ≡ 1 mm).
///
/// Reference values with H = 1500 m:
/// - T = 0°C: `e_s` ≈ 6.11 hPa → `PW_sat` ≈ 7.3 mm
/// - T = 15°C: `e_s` ≈ 17.0 hPa → `PW_sat` ≈ 19.3 mm
/// - T = 20°C: `e_s` ≈ 23.4 hPa → `PW_sat` ≈ 25.9 mm
#[must_use]
pub fn saturation_upper(t_upper: f32, params: &AtmosphereParams) -> f32 {
    saturation_upper_pw(t_upper, params.upper_layer_altitude_m)
}

/// Variant of `saturation_upper` taking the layer height directly. Useful
/// for callers (metrics, diagnostics) that don't need the full
/// `AtmosphereParams`.
#[must_use]
pub fn saturation_upper_pw(t_upper: f32, altitude_m: f32) -> f32 {
    let e_s_pa = tetens_saturation_vapor_pressure(t_upper).0 * 100.0;
    let t_kelvin = (t_upper + 273.15).max(1.0);
    let rho_vap = e_s_pa / (R_VAP * t_kelvin);
    rho_vap * altitude_m
}

/// Saturation precipitable water (mm) for the boundary layer at
/// `t_surface`. Issue #45: analogous to `saturation_upper` but integrated
/// over the 50 m near the ground, the layer where radiative fog forms.
///
/// Reference values:
/// - T = -5 °C: ≈ 0.16 mm
/// - T = 0 °C : ≈ 0.24 mm
/// - T = 15 °C: ≈ 0.64 mm
/// - T = 25 °C: ≈ 1.15 mm
#[must_use]
pub fn saturation_surface(t_surface: f32) -> f32 {
    saturation_upper_pw(t_surface, SURFACE_LAYER_M)
}

/// Horizontal means of the surface state that anchor the upper-air
/// temperature: `(mean surface temperature °C, mean elevation m)`.
/// Empty grid: `(0, 0)`.
#[must_use]
pub fn surface_means(grid: &HexGrid) -> (f32, f32) {
    // Running means (Welford): no integer→float cast, the count is an
    // exact f32 up to 2^24 cells.
    let mut count = 0.0_f32;
    let mut mean_t = 0.0_f32;
    let mut mean_z = 0.0_f32;
    for c in grid.cells_slice() {
        count += 1.0;
        mean_t += (c.temperature - mean_t) / count;
        mean_z += (c.elevation - mean_z) / count;
    }
    (mean_t, mean_z)
}

/// Time constant (s) of the diurnal smoothing applied to the map-mean
/// surface temperature that anchors the upper layer
/// (`Simulation::upper_air_mean_t`, stepped by [`smooth_upper_air_mean_t`]).
///
/// The diurnal cycle lives in the surface boundary layer: the daily
/// amplitude of the air temperature is ~8-10 K at screen level and
/// decays to ~1 K in the free troposphere above the boundary-layer top
/// (Stull 1988, *An Introduction to Boundary Layer Meteorology*, §1.6
/// "diurnal variation" and ch. 11; radiosonde climatologies put the
/// 850 hPa diurnal range at ~1 K). The layer this engine models sits
/// 1500 m above the mean ground (`upper_layer_altitude_m`), above the
/// daytime mixed layer most of the year. Anchored to the *instantaneous*
/// mean (2026-09-02), the whole layer followed the ~8 K nightly cooling
/// of the surface and condensed on the highest cells every night: the
/// crest of `phys_ubac_not_a_rain_attractor` was wet 365 days a year
/// and the procedural world had 0 rain-free days (#63).
///
/// τ = 24 h is the shortest first-order smoothing that removes the
/// diurnal harmonic: a 24 h sinusoid is attenuated to
/// `1/√(1+(2π)²) ≈ 0.157` of its amplitude (8 K → 1.3 K, the observed
/// order of magnitude), while the seasonal signal (period 365 d) passes
/// at 0.9999 of its amplitude with a one-day lag. A shorter τ lets the
/// night through, a longer one only buys lag on the seasons.
pub const UPPER_AIR_SMOOTHING_TAU_S: f32 = 24.0 * SECONDS_PER_HOUR;

/// One hourly step of the first-order (exponential) smoothing of the
/// map-mean surface temperature: `m += (T̄ − m)·(1 − exp(−Δt/τ))`, with
/// Δt = one tick and τ = [`UPPER_AIR_SMOOTHING_TAU_S`]. Exact
/// discretisation of `dm/dt = (T̄ − m)/τ` for a `T̄` held over the tick.
/// Called once per tick by `Simulation::step_hour` before the
/// atmosphere step; the result travels to the atmosphere as
/// `AtmoForcing::upper_air_mean_t`.
#[must_use]
pub fn smooth_upper_air_mean_t(previous: f32, instantaneous: f32) -> f32 {
    let gain = 1.0 - (-SECONDS_PER_HOUR / UPPER_AIR_SMOOTHING_TAU_S).exp();
    previous + (instantaneous - previous) * gain
}

/// Temperature (°C) of the upper layer above a cell at elevation
/// `elevation_m`: the (diurnally smoothed) map-mean surface temperature,
/// corrected by the standard lapse rate for the height of the layer
/// above the map-mean ground, `T̄ − Γ·(z − z̄ + H)/1000`.
///
/// The free atmosphere is horizontally mixed at the scale of the
/// terrarium (a few km): the air 1500 m above a north-facing slope is
/// the same air as above the south-facing slope next to it. The layer
/// therefore only follows the *elevation* of the ground (orographic
/// cooling, the mechanism that makes summits precipitate), never the
/// local surface anomaly (aspect, occlusion, lake cooling, snow albedo).
/// Before 2026-09-02 it was `T_surface − Γ·H`: any cell persistently
/// colder than its neighbours at the surface became a permanent
/// condenser aloft, raining 365 days a year while the rest of the map
/// stayed dry (JOURNAL 2026-09-02, bisected to the aspect insolation
/// e3594f9).
///
/// `mean_surface_t` is the smoothed mean kept by the simulation
/// (`Simulation::upper_air_mean_t`, see [`UPPER_AIR_SMOOTHING_TAU_S`]),
/// not the instantaneous one: the free atmosphere keeps the seasons and
/// the lapse with elevation, not the day/night swing of the surface.
#[must_use]
pub fn upper_air_temperature(
    mean_surface_t: f32,
    mean_elevation_m: f32,
    elevation_m: f32,
    params: &AtmosphereParams,
    temp_params: &TemperatureParams,
) -> f32 {
    let height_above_mean_ground = elevation_m - mean_elevation_m + params.upper_layer_altitude_m;
    mean_surface_t - temp_params.lapse_rate * height_above_mean_ground / 1000.0
}

/// Cloud dynamics: vapor ↔ droplets.
///
/// - Condensation (#63 Phase 4 Step 3): anchored to Clausius-Clapeyron via
///   Tetens. When `humidity_upper > saturation_upper(T)`, the
///   thermodynamic surplus drains into droplets at rate
///   `condensation_rate`. Natural asymptote at RH=1 (saturation), not an
///   arbitrary dimensionless RH threshold.
/// - Cloud evaporation: if RH < `cloud_evap_hr_threshold`, a fraction of
///   `cloud_water` returns to `humidity_upper`.
/// - Dead zone between saturation and `cloud_evap_hr_threshold`:
///   hysteresis that avoids pulsing.
///
/// `t_upper` = upper-air temperature per cell (`AtmoScratch::t_upper`,
/// filled by `fill_upper_air` from [`upper_air_temperature`]).
pub(crate) fn step_cloud_dynamics(next: &mut HexGrid, params: &AtmosphereParams, t_upper: &[f32]) {
    let rate = params.condensation_rate.min(1.0);
    for (nc, &t_up) in next.cells_slice_mut().iter_mut().zip(t_upper) {
        let sat = saturation_upper(t_up, params);

        let surplus_mm = nc.humidity_upper - sat;
        if surplus_mm > 0.0 {
            // CC drain: (humidity_upper - sat) × rate. At steady state
            // with input X mm/tick, hu_eq = sat + X/rate → RH = 1 +
            // X/(rate·sat). With rate=1.0/h and input bounded by the LCL
            // bound on the orographic pump (cf
            // step_orographic_convection), RH plateaus at ~1 + ε.
            let transfer = surplus_mm * rate;
            nc.humidity_upper -= transfer;
            nc.cloud_water += transfer;
        } else if sat > 0.0
            && (nc.humidity_upper / sat) < params.cloud_evap_hr_threshold
            && nc.cloud_water > 0.0
        {
            // Droplet evaporation: depends only on the cloud_water stock
            // and the rate (no amplifying "deficit").
            let transfer = nc.cloud_water * params.cloud_evap_rate;
            nc.cloud_water -= transfer;
            nc.humidity_upper += transfer;
        }
    }
}

/// Isotropic diffusion of `cloud_water` to neighbors: each cell exports a
/// `cloud_diffusion_rate` fraction of its droplets, distributed equally
/// among its topological neighbors. The flux is conservative by
/// construction. Rationale: without this smoothing, two neighboring cells
/// at `cloud_water = 0.119` and `0.121` have radically different
/// behaviors (precipitates vs. nothing), hence the checkerboard pattern.
/// With diffusion, mass is shared locally and clouds become continuous
/// regions.
pub(crate) fn step_cloud_diffusion(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    snap: &mut Vec<f32>,
    deltas: &mut Vec<f32>,
) {
    let rate = params.cloud_diffusion_rate;
    if rate <= 0.0 {
        return;
    }
    let next_cells = next.cells_slice_mut();
    let n = next_cells.len();
    snap.resize(n, 0.0);
    for (i, c) in next_cells.iter().enumerate() {
        snap[i] = c.cloud_water;
    }
    deltas.resize(n, 0.0);
    deltas.fill(0.0);
    for i in 0..n {
        let src = snap[i];
        if src <= 0.0 {
            continue;
        }
        let neighbors = current.neighbor_indices_toric(i);
        let outgoing = src * rate;
        deltas[i] -= outgoing;
        // Distribution over 6 toric neighbors (self-fallback via wrap
        // impossible → conservative: the share returns to itself,
        // equivalent to zero loss).
        let share = outgoing / 6.0;
        for &ni in &neighbors {
            deltas[ni] += share;
        }
    }
    for (i, cell) in next_cells.iter_mut().enumerate() {
        if deltas[i] != 0.0 {
            cell.cloud_water = (cell.cloud_water + deltas[i]).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::scaling::scale_atmosphere_for_hourly_tick;
    use crate::atmosphere::test_support::default_temp_params;
    use crate::coord::HexCoord;
    use crate::grid::HexGrid;

    /// Base law "warm air holds more water" (Clausius-Clapeyron via
    /// Tetens): `saturation_surface` must grow STRICTLY with temperature,
    /// and stick to the documented reference values. This is the physical
    /// foundation of "evaporation accelerates with heat"; it's pinned here
    /// so a refactor of the formula can't break it silently.

    #[test]
    fn saturation_surface_rises_strictly_with_temperature() {
        // Strictly monotonic from −20 to +40 °C.
        let mut prev = saturation_surface(-20.0);
        let mut t = -19.0;
        while t <= 40.0 {
            let s = saturation_surface(t);
            assert!(
                s > prev,
                "saturation must increase with T: {s} at {t} °C ≤ {prev} at the previous step"
            );
            prev = s;
            t += 1.0;
        }
        // Reference values from the doc-comment (±10%: these are anchors
        // rounded to 2 digits, not targets to the last decimal). They fix
        // the order of magnitude; monotonicity and concavity are the
        // strict guards.
        for (temp, expected) in [(-5.0, 0.16), (0.0, 0.24), (15.0, 0.64), (25.0, 1.15)] {
            let got = saturation_surface(temp);
            assert!(
                (got - expected).abs() / expected < 0.10,
                "saturation_surface({temp}) = {got:.3} mm, expected ≈ {expected} mm"
            );
        }
        // Clausius-Clapeyron concavity: +10 °C more than doubles between
        // 15 and 25 °C (quasi-exponential growth, not linear).
        assert!(
            saturation_surface(25.0) > 1.7 * saturation_surface(15.0),
            "the rise must accelerate with T (Clausius-Clapeyron)"
        );
    }

    #[test]
    fn phys_condensation_drain_anchored_to_clausius_clapeyron() {
        // Issue #63 Phase 4 Step 3, CC drain anchored.
        //
        // Setup: humidity_upper in extreme supersaturation + cold T. With
        // drain anchored to absolute mm and rate=1.0/h (Pruppacher & Klett
        // 1997 §13.3.1, τ_phase << 1h → complete drain per tick), the
        // thermodynamic surplus is cleared in one tick.
        //
        // Invariants checked:
        // 1. transfer ≤ surplus_mm × rate (linearity of CC drain, contrast
        //    with the old drain × RH_fraction × hu not bounded by CC)
        // 2. conservation: delta_cloud_water = transfer (no creation)
        // 3. humidity_upper ≥ sat (saturable drain, no artificial
        //    undersaturation)
        // 4. at rate=1.0/h, final RH ≈ 1.0 (complete drain in 1 tick)
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.humidity_upper = 50.0;
            cell.temperature = 0.0;
            cell.cloud_water = 0.0;
        }
        let params = AtmosphereParams::default();
        let params_hourly = scale_atmosphere_for_hourly_tick(&params);
        let temp_params = default_temp_params();

        let initial = grid.get(c0).unwrap().humidity_upper;
        // Single cell: the map means are the cell itself, so the upper
        // air is `T − Γ·H` as in the historical formula.
        let (mean_t, mean_z) = surface_means(&grid);
        let t_upper = upper_air_temperature(mean_t, mean_z, 0.0, &params, &temp_params);
        let sat = saturation_upper(t_upper, &params);
        let surplus_mm = initial - sat;
        assert!(
            surplus_mm > 0.0,
            "invalid setup: initial={initial} must be >> sat={sat}"
        );

        step_cloud_dynamics(&mut grid, &params_hourly, &[t_upper]);

        let after = grid.get(c0).unwrap();
        let transfer = initial - after.humidity_upper;
        let rate_eff = params_hourly.condensation_rate.min(1.0);
        let max_transfer_cc = surplus_mm * rate_eff;
        let tol = 1e-3;

        assert!(
            (after.cloud_water - transfer).abs() < 1e-5,
            "non-conservative transfer: delta_cloud={} != transfer={transfer}",
            after.cloud_water
        );
        assert!(
            transfer <= max_transfer_cc + tol,
            "drain not anchored to CC: transfer={transfer} > surplus_mm × rate={max_transfer_cc}"
        );
        assert!(
            after.humidity_upper >= sat - tol,
            "drain violated CC: humidity_upper={} < sat={sat}",
            after.humidity_upper
        );
        let hr_final = after.humidity_upper / sat;
        assert!(
            (hr_final - 1.0).abs() < 0.01,
            "rate=1.0/h must bring RH to saturation in 1 tick: final RH={hr_final}"
        );
    }

    /// The upper air is horizontally mixed: two cells at the same
    /// elevation share the same upper-air temperature whatever their
    /// surface anomaly (aspect, lake, snow), and a higher cell only
    /// sees the standard lapse for its extra height. A cold surface
    /// therefore cannot become a permanent condenser aloft (JOURNAL
    /// 2026-09-02, bisected to the aspect insolation e3594f9).
    #[test]
    fn upper_air_ignores_surface_anomaly_and_follows_elevation() {
        let params = AtmosphereParams::default();
        let temp_params = default_temp_params();
        let mut grid = HexGrid::from_radius(1);
        let coords: Vec<HexCoord> = grid.coords().copied().collect();
        for (k, c) in coords.iter().enumerate() {
            let cell = grid.get_mut(*c).unwrap();
            cell.elevation = if k == 0 { 1000.0 } else { 0.0 };
            // A 10 °C surface contrast between two flat cells (ubac vs adret).
            cell.temperature = if k == 1 { 5.0 } else { 15.0 };
        }
        let (mean_t, mean_z) = surface_means(&grid);
        let ubac = upper_air_temperature(mean_t, mean_z, 0.0, &params, &temp_params);
        let adret = upper_air_temperature(mean_t, mean_z, 0.0, &params, &temp_params);
        assert_eq!(
            ubac.to_bits(),
            adret.to_bits(),
            "same elevation ⇒ same upper air"
        );
        let summit = upper_air_temperature(mean_t, mean_z, 1000.0, &params, &temp_params);
        let expected_drop = temp_params.lapse_rate;
        assert!(
            ((ubac - summit) - expected_drop).abs() < 1e-4,
            "1000 m higher ⇒ {expected_drop} °C colder aloft, got {}",
            ubac - summit
        );
        // Single cell: reduces to the historical `T − Γ·H`.
        let alone = HexGrid::from_radius(0);
        let (t1, z1) = surface_means(&alone);
        let hist = 0.0 - temp_params.lapse_rate * params.upper_layer_altitude_m / 1000.0;
        assert!((upper_air_temperature(t1, z1, 0.0, &params, &temp_params) - hist).abs() < 1e-5);
    }

    #[test]
    fn cold_upper_air_precipitates_more_easily() {
        // With Clausius-Clapeyron saturation, cold air saturates for less
        // vapor: saturation(-5°C) < saturation(0°C) < saturation(20°C). So
        // a cold cell precipitates at a much lower absolute humidity than
        // a warm cell. This is the inversion of linear reasoning: the
        // sensitivity comes from the exp curve, not from an offset.
        let params = AtmosphereParams::default();
        let sat_cold = saturation_upper(0.0, &params);
        let sat_warm = saturation_upper(20.0, &params);
        assert!(
            sat_cold < sat_warm,
            "saturation 0°C ({sat_cold:.3}) must be < saturation 20°C ({sat_warm:.3})"
        );
    }

    /// Hourly samples of one day, as many as ticks in a day.
    const HOURS_PER_DAY: u8 = 24;

    /// The upper-air anchor is a first-order filter with τ = 24 h: a
    /// 24 h sinusoid (the diurnal cycle of the mean surface, ~8 K) must
    /// come out attenuated to `|H| = 1/√(1+(ωτ)²)` with ωτ = 2π, i.e.
    /// ≈ 0.157 of its amplitude, the ~1 K of the free troposphere. The
    /// discrete EMA (gain `1 − e^{−1/24}` per hour) sits within 0.5 % of
    /// the continuous filter at that frequency; measured on the last 5
    /// of 30 days, once the transient has died out.
    #[test]
    fn upper_air_smoothing_attenuates_the_diurnal_harmonic() {
        let amplitude = 8.0_f32;
        let mean = 10.0_f32;
        let days = 30;
        let mut m = mean;
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for day in 0..days {
            for hour in 0..HOURS_PER_DAY {
                let phase = 2.0 * std::f32::consts::PI * f32::from(hour) / f32::from(HOURS_PER_DAY);
                let surface_t = mean + amplitude * phase.sin();
                m = smooth_upper_air_mean_t(m, surface_t);
                if day >= days - 5 {
                    lo = lo.min(m);
                    hi = hi.max(m);
                }
            }
        }
        let measured_ratio = (hi - lo) / (2.0 * amplitude);
        let omega_tau = 2.0 * std::f32::consts::PI;
        let expected_ratio = 1.0 / (1.0 + omega_tau * omega_tau).sqrt();
        assert!(
            ((measured_ratio - expected_ratio) / expected_ratio).abs() < 0.03,
            "diurnal amplitude ratio {measured_ratio:.4}, first-order filter predicts \
             {expected_ratio:.4} (τ = 24 h)"
        );
        // The mean goes through untouched: the filter removes the
        // harmonic, not the level (the seasons must survive).
        let centre = f32::midpoint(hi, lo);
        assert!(
            (centre - mean).abs() < 0.05,
            "the diurnal mean must pass unchanged: centre {centre:.3} vs {mean}"
        );
    }

    /// Step response: after exactly τ (24 hourly steps) the residual of
    /// a step change is `e^{−1}` of the step, the e-fold time of the
    /// filter (a seasonal change reaches the upper air within days).
    #[test]
    fn upper_air_smoothing_converges_with_e_fold_time_tau() {
        let from = -5.0_f32;
        let to = 15.0_f32;
        let mut m = from;
        for _ in 0..HOURS_PER_DAY {
            m = smooth_upper_air_mean_t(m, to);
        }
        let residual = (to - m) / (to - from);
        let expected = (-1.0_f32).exp();
        assert!(
            (residual - expected).abs() < 1e-3,
            "residual after τ = {residual:.4}, expected e^-1 = {expected:.4}"
        );
        for _ in 0..(10 * HOURS_PER_DAY) {
            m = smooth_upper_air_mean_t(m, to);
        }
        assert!(
            (m - to).abs() < 1e-3,
            "after 11τ the anchor must have converged to the step: {m} vs {to}"
        );
    }
}
