use crate::cell::CellProperties;
use crate::coord::{DIRECTIONS, hex_direction_to_world};
use crate::grid::HexGrid;
use crate::wind::{WindField, WindParams};

use super::{AtmoScratch, AtmosphereParams};

/// Atmospheric layer. Routes the generic advection and diffusion functions
/// toward one or the other of the two humidity pools.
#[derive(Clone, Copy, Debug)]
pub(crate) enum HumidityLayer {
    Surface,
    Upper,
}

fn read_layer(cell: &CellProperties, layer: HumidityLayer) -> f32 {
    match layer {
        HumidityLayer::Surface => cell.humidity_surface,
        HumidityLayer::Upper => cell.humidity_upper,
    }
}

fn write_layer(cell: &mut CellProperties, layer: HumidityLayer, value: f32) {
    match layer {
        HumidityLayer::Surface => cell.humidity_surface = value,
        HumidityLayer::Upper => cell.humidity_upper = value,
    }
}

/// Advection of a humidity layer by a wind field. The same logic serves
/// both `humidity_surface` (surface wind) and `humidity_upper` (upper-air
/// wind, derived by rotation + scale).
///
/// Orographic lift: when `layer == Surface` and the flux advects toward a
/// higher cell, a fraction proportional to the elevation gain is converted
/// to `humidity_upper` at the destination instead of staying in
/// `humidity_surface`. Models forced orographic condensation.
///
/// Wind selection internal to the domain (#61): `Surface` is advected by
/// `surface_wind` (surface wind, provided by the caller), `Upper` by
/// `scratch.wind_upper` (upper-air wind, already filled at the top of
/// `step_atmosphere_into` via `compute_upper_wind_field_into`), a
/// precondition the caller must guarantee for `Upper`. Additional shared
/// precondition: `scratch.sat_upper_offset` must have been filled by
/// `AtmoScratch::fill_sat_upper_offset` before the call (LCL bound of the lift).
pub(crate) fn advect_humidity_layer_into(
    current: &HexGrid,
    next: &mut HexGrid,
    surface_wind: &WindField,
    wind_params: &WindParams,
    atmo_params: &AtmosphereParams,
    layer: HumidityLayer,
    scratch: &mut AtmoScratch,
) {
    let AtmoScratch {
        wind_upper,
        sat_upper_offset,
        snap,
        deltas,
        lift_deltas_upper,
        lift_upper_snap,
        ..
    } = scratch;
    let wind_field: &WindField = match layer {
        HumidityLayer::Surface => surface_wind,
        HumidityLayer::Upper => wind_upper,
    };
    let n = current.len();
    snap.resize(n, 0.0);
    {
        let next_cells = next.cells_slice();
        for i in 0..n {
            snap[i] = read_layer(&next_cells[i], layer);
        }
    }
    deltas.resize(n, 0.0);
    deltas.fill(0.0);

    let surface_lift_active =
        matches!(layer, HumidityLayer::Surface) && atmo_params.orographic_lift_coef > 0.0;
    // Reused scratch buffers (#88/#65), filled only when the lift
    // is active, like the fresh Vecs they replace.
    lift_deltas_upper.clear();
    lift_upper_snap.clear();
    if surface_lift_active {
        lift_deltas_upper.resize(n, 0.0);
        // Snapshot of humidity_upper to compute the dynamic saturation
        // deficit at the destination (orographic lift bounded to the LCL, cf
        // step_orographic_convection #63 Phase 4 Step 3).
        let next_cells = next.cells_slice();
        lift_upper_snap.extend((0..n).map(|i| next_cells[i].humidity_upper));
    }

    let cur_cells = current.cells_slice();
    for i in 0..n {
        let wind = wind_field[i];
        let hum = snap[i];
        let src_elev = cur_cells[i].elevation;
        let neighbors = current.neighbor_indices_toric(i);

        let mut weights = [0.0_f32; 6];
        let mut total_weight = 0.0;

        for (di, w) in weights.iter_mut().enumerate() {
            let (dx, dy) = hex_direction_to_world(di);
            let dot = wind.x * dx + wind.y * dy;
            if dot > 0.0 {
                *w = dot;
                total_weight += dot;
            }
        }

        if total_weight < 1e-6 {
            continue;
        }

        let inv_total = 1.0 / total_weight;
        for w in &mut weights {
            *w *= inv_total;
        }

        // Fraction of the cell transported = rate * wind magnitude,
        // capped at 0.95 (CFL condition).
        let wind_mag = wind.magnitude();
        let fraction = (wind_params.humidity_advection_rate * wind_mag).min(0.95);
        let hum_out = fraction * hum;
        deltas[i] -= hum_out;
        for (di, &j) in neighbors.iter().enumerate() {
            if weights[di] == 0.0 {
                continue;
            }
            let flux = hum_out * weights[di];
            if surface_lift_active {
                let t_elev = cur_cells[j].elevation;
                let elev_delta = t_elev - src_elev;
                if elev_delta > 0.0 {
                    let lift = (atmo_params.orographic_lift_coef * elev_delta).clamp(0.0, 0.80);
                    let to_upper_brut = flux * lift;
                    // LCL bound: lift capped by the saturation deficit at the
                    // destination. The surplus goes back to humidity_surface dest.
                    // `sat_upper_offset[j]` = `saturation_upper(T_j - t_offset)`
                    // precomputed (#97) on the same `current` temperature, so
                    // bit-identical to the inline call it replaces.
                    let sat_j = sat_upper_offset[j];
                    let upper_running = lift_upper_snap[j] + lift_deltas_upper[j];
                    let deficit_j = (sat_j - upper_running).max(0.0);
                    let to_upper = to_upper_brut.min(deficit_j);
                    let to_surface = flux - to_upper;
                    deltas[j] += to_surface;
                    lift_deltas_upper[j] += to_upper;
                } else {
                    deltas[j] += flux;
                }
            } else {
                deltas[j] += flux;
            }
        }
    }

    for (i, cell) in next.cells_slice_mut().iter_mut().enumerate() {
        let delta = deltas[i];
        let upper_inc = if surface_lift_active {
            lift_deltas_upper[i]
        } else {
            0.0
        };
        if delta != 0.0 {
            let new_val = (read_layer(cell, layer) + delta).max(0.0);
            write_layer(cell, layer, new_val);
        }
        if upper_inc != 0.0 {
            cell.humidity_upper = (cell.humidity_upper + upper_inc).max(0.0);
        }
    }
}

/// Directional advection of `cloud_water` by the upper-air wind.
///
/// Reuses the pattern from `advect_humidity_layer_into` (push weighted by
/// dot(wind, `dir_neighbor`)), but simplified: no orographic lift
/// (the droplets are already condensed, no vapor-to-liquid transition
/// to force on the slopes). Strict conservation by construction:
/// each cell sends `fraction × cloud_water` split across the downwind
/// neighbors, and removes the same amount from itself.
///
/// Physical justification: stratiform clouds travel at the wind speed at
/// their altitude (~1500 m, hence `wind_upper`). Without this
/// advection, the droplets stay parked on the condensation cell and
/// the rain falls systematically back onto the source of the
/// vapor (cell-lake cycle, issue #24). With it, the cloud is carried
/// a few cells before precipitating, which is what lets rain
/// evaporated from a lake fall on the relief downstream.
pub fn advect_cloud_water_into(
    current: &HexGrid,
    next: &mut HexGrid,
    wind_upper: &WindField,
    atmo_params: &AtmosphereParams,
    snap: &mut Vec<f32>,
    deltas: &mut Vec<f32>,
) {
    if atmo_params.cloud_advection_rate <= 0.0 {
        return;
    }
    let n = current.len();
    snap.resize(n, 0.0);
    {
        let next_cells = next.cells_slice();
        for i in 0..n {
            snap[i] = next_cells[i].cloud_water;
        }
    }
    deltas.resize(n, 0.0);
    deltas.fill(0.0);

    for i in 0..n {
        let wind = wind_upper[i];
        let cw = snap[i];
        if cw <= 0.0 {
            continue;
        }
        let neighbors = current.neighbor_indices_toric(i);

        let mut weights = [0.0_f32; 6];
        let mut total_weight = 0.0;
        for (di, w) in weights.iter_mut().enumerate() {
            let (dx, dy) = hex_direction_to_world(di);
            let dot = wind.x * dx + wind.y * dy;
            if dot > 0.0 {
                *w = dot;
                total_weight += dot;
            }
        }
        if total_weight < 1e-6 {
            continue;
        }
        let inv_total = 1.0 / total_weight;
        for w in &mut weights {
            *w *= inv_total;
        }

        let wind_mag = wind.magnitude();
        let fraction = (atmo_params.cloud_advection_rate * wind_mag).min(0.95);
        let cw_out = fraction * cw;
        deltas[i] -= cw_out;
        for (di, &j) in neighbors.iter().enumerate() {
            if weights[di] == 0.0 {
                continue;
            }
            deltas[j] += cw_out * weights[di];
        }
    }

    for (i, cell) in next.cells_slice_mut().iter_mut().enumerate() {
        let delta = deltas[i];
        if delta != 0.0 {
            cell.cloud_water = (cell.cloud_water + delta).max(0.0);
        }
    }
}

/// Directional advection of temperature by the surface wind.
/// Symmetric transfer proportional to the temperature gradient and the
/// directional weight.
pub(crate) fn advect_temperature_by_wind_into(
    current: &HexGrid,
    next: &mut HexGrid,
    wind_field: &WindField,
    wind_params: &WindParams,
    temp_snap: &mut Vec<f32>,
    temp_deltas: &mut Vec<f32>,
) {
    if wind_params.temperature_advection_rate <= 0.0 {
        return;
    }
    let n = next.len();
    temp_snap.resize(n, 0.0);
    {
        let next_cells = next.cells_slice();
        for i in 0..n {
            temp_snap[i] = next_cells[i].temperature;
        }
    }
    temp_deltas.resize(n, 0.0);
    temp_deltas.fill(0.0);

    for i in 0..n {
        let wind = wind_field[i];
        let temp = temp_snap[i];
        let neighbors = current.neighbor_indices_toric(i);

        let mut weights = [0.0_f32; 6];
        let mut total_weight = 0.0;

        for (di, _dir) in DIRECTIONS.iter().enumerate() {
            let (dx, dy) = hex_direction_to_world(di);
            let dot = wind.x * dx + wind.y * dy;
            if dot > 0.0 {
                weights[di] = dot;
                total_weight += dot;
            }
        }

        if total_weight < 1e-6 {
            continue;
        }

        let inv_total = 1.0 / total_weight;
        for w in &mut weights {
            *w *= inv_total;
        }

        for (di, &j) in neighbors.iter().enumerate() {
            if weights[di] == 0.0 {
                continue;
            }
            let target_temp = temp_snap[j];
            let t_delta =
                wind_params.temperature_advection_rate * weights[di] * (temp - target_temp);
            temp_deltas[i] -= t_delta;
            temp_deltas[j] += t_delta;
        }
    }

    for (cell, &delta) in next.cells_slice_mut().iter_mut().zip(temp_deltas.iter()) {
        if delta != 0.0 {
            cell.temperature += delta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::test_support::{
        assert_lcl_slack, default_temp_params, default_wind_params, oro_pump_world,
    };
    use crate::atmosphere::total_humidity;
    use crate::coord::HexCoord;
    use crate::wind::WindVec;

    /// Second path of the same `orographic_lift_coef`: when the WIND pushes
    /// surface vapor toward a higher cell, the advected flux is converted
    /// to `humidity_upper` at the destination (forced orographic
    /// condensation, `lift = clamp(coef × Δz, 0, 0.80)`); on flat terrain,
    /// the same wind leaves it entirely in the surface layer. Wind is
    /// CONSTRUCTED (uniform, aligned on `DIRECTIONS[0]`): this tests the
    /// reaction to wind, not its origin, no synoptic forcing here.
    #[test]
    fn uphill_advection_lands_in_upper_layer_flat_stays_in_surface() {
        fn advect(grid: &HexGrid) -> HexGrid {
            let params = AtmosphereParams::default();
            let temp_params = default_temp_params();
            let wind_params = default_wind_params();
            // Magnitude 0.1 (WindVec unit) → advected fraction
            // = humidity_advection_rate (3.0) × 0.1 = 30% per tick.
            let (dx, dy) = hex_direction_to_world(0);
            let wind: WindField = vec![
                WindVec {
                    x: dx * 0.1,
                    y: dy * 0.1
                };
                grid.len()
            ];
            let mut next = grid.clone();
            let mut scratch = AtmoScratch::new(grid.len());
            scratch.fill_sat_upper_offset(grid, &params, &temp_params);
            advect_humidity_layer_into(
                grid,
                &mut next,
                &wind,
                &wind_params,
                &params,
                HumidityLayer::Surface,
                &mut scratch,
            );
            next
        }

        let center = HexCoord::new(0, 0);
        let downwind = center + DIRECTIONS[0];

        // World A: the downwind cell is 400 m higher.
        let (mut ridge, _coords) = oro_pump_world(10.0);
        ridge.get_mut(downwind).unwrap().elevation = 500.0;
        assert_lcl_slack(10.0 * 0.30);
        let next_ridge = advect(&ridge);
        let dest = next_ridge.get(downwind).unwrap();
        assert!(
            dest.humidity_upper > 0.0 && dest.humidity_surface > 0.0,
            "windward slope: flow must split upper/surface \
             (upper={}, surf={})",
            dest.humidity_upper,
            dest.humidity_surface
        );
        // lift = clamp(0.05 × 400 m, 0, 0.80) = 0.80 → 4× more in upper.
        let ratio = dest.humidity_upper / dest.humidity_surface;
        assert!(
            (ratio - 4.0).abs() < 1e-3,
            "upper/surface split at the peak: ratio {ratio}, expected 4.0"
        );

        // World B: flat terrain, same wind, full transport stays in surface.
        let (flat, coords_flat) = oro_pump_world(10.0);
        let next_flat = advect(&flat);
        for &c in &coords_flat {
            assert!(
                next_flat.get(c).unwrap().humidity_upper == 0.0,
                "flat terrain: nothing should rise into upper ({c:?})"
            );
        }
        assert!(
            next_flat.get(downwind).unwrap().humidity_surface > 0.0,
            "flat terrain: horizontal transport itself must still happen"
        );

        // Conservation in both worlds.
        for (label, before, after) in [("ridge", &ridge, &next_ridge), ("flat", &flat, &next_flat)]
        {
            let (t0, t1) = (total_humidity(before), total_humidity(after));
            assert!(
                (t1 - t0).abs() < 1e-4,
                "conservation ({label}): before={t0}, after={t1}"
            );
        }
    }
}
