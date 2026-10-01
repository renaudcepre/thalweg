use crate::cell::CellProperties;

use super::{AtmosphereParams, saturation_surface};

/// Radiative fog for one cell (issue #45): local supersaturation of the
/// near-ground boundary layer condenses directly into `cloud_water`.
///
/// Extracted to a per-cell function, no full-grid sweep of its own (r250
/// perf effort, chunk B2): the only caller left is
/// `atmosphere::apply_temperature_advection_then_cloud_and_condensation`,
/// which fuses it with temperature advection's apply and
/// `cloud_dynamics_for_cell` into one sweep — see that function's doc
/// for the dependency argument. Also exercised directly by this module's
/// own unit tests, on a single cell, no grid needed.
pub(crate) fn surface_condensation_for_cell(nc: &mut CellProperties, params: &AtmosphereParams) {
    if nc.humidity_surface <= 0.0 {
        return;
    }
    let sat = saturation_surface(nc.temperature);
    if sat <= 0.0 {
        return;
    }
    let hr = nc.humidity_surface / sat;
    if hr > params.fog_condensation_threshold {
        let surplus = hr - params.fog_condensation_threshold;
        let rate = (surplus * params.fog_condensation_rate).min(1.0);
        let transfer = nc.humidity_surface * rate;
        nc.humidity_surface -= transfer;
        nc.cloud_water += transfer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_condensation_above_threshold_creates_fog() {
        // Issue #45: if HR_surface > threshold, fraction of humidity_surface
        // moves to cloud_water. Test isolates `surface_condensation_for_cell`
        // without rest of atmo pipeline.
        let mut cell = CellProperties {
            temperature: 5.0,
            cloud_water: 0.0,
            // At 5°C, sat_surface = saturation_upper_pw(5, 50) ≈ 0.43 mm
            // → set humidity_surface = 0.5 → HR ≈ 1.16 (well
            // above threshold 0.95).
            humidity_surface: 0.5,
            ..CellProperties::default()
        };
        let params = AtmosphereParams::default();
        // No scaling here: call surface_condensation_for_cell directly.
        let initial_humidity = cell.humidity_surface;
        let initial_cloud = cell.cloud_water;

        surface_condensation_for_cell(&mut cell, &params);

        let transferred = initial_humidity - cell.humidity_surface;
        assert!(
            transferred > 0.0,
            "humidity_surface devait baisser : {initial_humidity} → {}",
            cell.humidity_surface
        );
        assert!(
            (cell.cloud_water - initial_cloud - transferred).abs() < 1e-6,
            "transfert non conservatif : delta_cloud {} != delta_humidity {}",
            cell.cloud_water - initial_cloud,
            transferred
        );
    }

    #[test]
    fn surface_condensation_below_threshold_is_inactive() {
        // If HR_surface ≤ fog_condensation_threshold, no condensation.
        let mut cell = CellProperties {
            temperature: 20.0,
            // At 20°C, sat_surface ≈ 0.86 mm → HR = 0.5 / 0.86 ≈ 0.58
            // (below threshold 0.95).
            humidity_surface: 0.5,
            cloud_water: 0.0,
            ..CellProperties::default()
        };
        let params = AtmosphereParams::default();
        surface_condensation_for_cell(&mut cell, &params);
        assert!(
            cell.cloud_water < 1e-9,
            "no condensation expected, got cloud_water={}",
            cell.cloud_water
        );
        assert!(
            (cell.humidity_surface - 0.5).abs() < 1e-9,
            "humidity_surface should be unchanged, got {}",
            cell.humidity_surface
        );
    }
}
