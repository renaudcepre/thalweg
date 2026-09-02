use crate::grid::HexGrid;
use crate::physics::tetens_saturation_vapor_pressure;
use crate::temperature::TemperatureParams;

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
pub(crate) fn step_cloud_dynamics(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    temp_params: &TemperatureParams,
) {
    let t_offset = temp_params.lapse_rate * params.upper_layer_altitude_m / 1000.0;
    let rate = params.condensation_rate.min(1.0);
    for nc in next.cells_slice_mut() {
        let t_upper = nc.temperature - t_offset;
        let sat = saturation_upper(t_upper, params);

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
        let t_offset = temp_params.lapse_rate * params.upper_layer_altitude_m / 1000.0;
        let t_upper = grid.get(c0).unwrap().temperature - t_offset;
        let sat = saturation_upper(t_upper, &params);
        let surplus_mm = initial - sat;
        assert!(
            surplus_mm > 0.0,
            "invalid setup: initial={initial} must be >> sat={sat}"
        );

        step_cloud_dynamics(&mut grid, &params_hourly, &temp_params);

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
}
