use crate::grid::HexGrid;

use super::{AtmosphereParams, saturation_surface};

pub(crate) fn step_surface_condensation(next: &mut HexGrid, params: &AtmosphereParams) {
    for nc in next.cells_slice_mut() {
        if nc.humidity_surface <= 0.0 {
            continue;
        }
        let sat = saturation_surface(nc.temperature);
        if sat <= 0.0 {
            continue;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;

    #[test]
    fn surface_condensation_above_threshold_creates_fog() {
        // Issue #45: if HR_surface > threshold, fraction of humidity_surface
        // moves to cloud_water. Test isolates `step_surface_condensation`
        // without rest of atmo pipeline.
        let mut grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        if let Some(c) = grid.get_mut(coord) {
            c.temperature = 5.0;
            c.cloud_water = 0.0;
            // At 5°C, sat_surface = saturation_upper_pw(5, 50) ≈ 0.43 mm
            // → set humidity_surface = 0.5 → HR ≈ 1.16 (well
            // above threshold 0.95).
            c.humidity_surface = 0.5;
        }
        let params = AtmosphereParams::default();
        // No scaling here: call step_surface_condensation directly.
        let initial_humidity = grid.get(coord).unwrap().humidity_surface;
        let initial_cloud = grid.get(coord).unwrap().cloud_water;

        step_surface_condensation(&mut grid, &params);

        let final_humidity = grid.get(coord).unwrap().humidity_surface;
        let final_cloud = grid.get(coord).unwrap().cloud_water;
        let transferred = initial_humidity - final_humidity;
        assert!(
            transferred > 0.0,
            "humidity_surface devait baisser : {initial_humidity} → {final_humidity}"
        );
        assert!(
            (final_cloud - initial_cloud - transferred).abs() < 1e-6,
            "transfert non conservatif : delta_cloud {} != delta_humidity {}",
            final_cloud - initial_cloud,
            transferred
        );
    }

    #[test]
    fn surface_condensation_below_threshold_is_inactive() {
        // If HR_surface ≤ fog_condensation_threshold, no condensation.
        let mut grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        if let Some(c) = grid.get_mut(coord) {
            c.temperature = 20.0;
            // At 20°C, sat_surface ≈ 0.86 mm → HR = 0.5 / 0.86 ≈ 0.58
            // (below threshold 0.95).
            c.humidity_surface = 0.5;
            c.cloud_water = 0.0;
        }
        let params = AtmosphereParams::default();
        step_surface_condensation(&mut grid, &params);
        let cell = grid.get(coord).unwrap();
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
