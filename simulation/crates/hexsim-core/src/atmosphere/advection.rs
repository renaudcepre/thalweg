use crate::cell::CellProperties;
use crate::coord::{DIRECTIONS, hex_direction_to_world, opposite_direction};
use crate::grid::HexGrid;
use crate::par::{for_each_chunk_mut, for_each_chunk_mut2, for_each_chunk_mut6, sum_dir_out};
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
/// `AtmoScratch::fill_upper_air` before the call (LCL bound of the lift).
///
/// Two-phase scatter -> gather (r250 perf effort). Phase 1 (`dir_out`)
/// computes, per source cell and independent of every other source, the
/// flux advected toward each of its 6 neighbors. Phase 1b, surface layer
/// with an active lift only (`dir_out_secondary`), computes the part of
/// that same flux the destination's LCL bound converts to
/// `humidity_upper` there: bounded by the pre-tick saturation snapshot
/// `lift_upper_snap` alone, never by another source's contribution to
/// the same destination in this tick. The historical serial scatter
/// instead accumulated a *running* `lift_upper_snap[j] +
/// lift_deltas_upper[j]` budget shared across sources in source-index
/// order — an order dependency incompatible with parallel, arbitrary-
/// chunking gather; dropping the running term aligns this lift with the
/// snapshot-only LCL bound `step_orographic_convection` already uses.
/// Verified by the ablation protocol (see the perf report) rather than
/// assumed. Phase 2 gathers both `dir_out` sets back per destination via
/// `coord::opposite_direction`.
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
        dir_out,
        dir_out_secondary,
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

    for dir in dir_out.iter_mut() {
        dir.clear();
        dir.resize(n, 0.0);
    }
    fill_flux_outflow(
        wind_field,
        snap,
        wind_params.humidity_advection_rate,
        dir_out,
    );

    if surface_lift_active {
        for dir in dir_out_secondary.iter_mut() {
            dir.clear();
            dir.resize(n, 0.0);
        }
        fill_lift_outflow(
            current,
            dir_out,
            sat_upper_offset,
            lift_upper_snap,
            atmo_params.orographic_lift_coef,
            dir_out_secondary,
        );
    }

    gather_humidity_deltas(
        current,
        dir_out,
        dir_out_secondary,
        surface_lift_active,
        deltas,
        lift_deltas_upper,
    );

    // Apply pass stays serial only in appearance: it's a pure per-cell map
    // (reads `deltas[i]`/`lift_deltas_upper[i]`, writes `next[i]`), split
    // into its own function so this one stays under `too_many_lines`.
    apply_humidity_deltas(next, deltas, lift_deltas_upper, surface_lift_active, layer);
}

/// Phase 1 of [`advect_humidity_layer_into`]: per source cell, the flux
/// advected toward each downwind direction. Reads only the wind field
/// and the pre-scatter snapshot `snap`: fully independent per source,
/// parallelizable (`par::for_each_chunk_mut6`).
fn fill_flux_outflow(wind_field: &WindField, snap: &[f32], rate: f32, dir_out: &mut [Vec<f32>; 6]) {
    for_each_chunk_mut6(dir_out, |start, chunks| {
        for local in 0..chunks[0].len() {
            let i = start + local;
            let wind = wind_field[i];
            let hum = snap[i];
            let mut weights = [0.0_f32; 6];
            let mut total_weight = 0.0_f32;
            for (di, w) in weights.iter_mut().enumerate() {
                let (dx, dy) = hex_direction_to_world(di);
                let dot = wind.x * dx + wind.y * dy;
                if dot > 0.0 {
                    *w = dot;
                    total_weight += dot;
                }
            }
            if total_weight < 1e-6 {
                for chunk in chunks.iter_mut() {
                    chunk[local] = 0.0;
                }
                continue;
            }
            let inv_total = 1.0 / total_weight;
            // Fraction of the cell transported = rate * wind magnitude,
            // capped at 0.95 (CFL condition).
            let wind_mag = wind.magnitude();
            let fraction = (rate * wind_mag).min(0.95);
            let hum_out = fraction * hum;
            for (d, chunk) in chunks.iter_mut().enumerate() {
                chunk[local] = if weights[d] > 0.0 {
                    hum_out * weights[d] * inv_total
                } else {
                    0.0
                };
            }
        }
    });
}

/// Phase 1b of [`advect_humidity_layer_into`] (surface layer, active
/// lift only): per source cell, the part of each directional flux
/// (`dir_out`, already fully computed by [`fill_flux_outflow`]) the
/// destination's LCL bound converts to `humidity_upper`, bounded by the
/// pre-tick saturation snapshot `lift_upper_snap` alone (see the caller's
/// doc-comment on why this dropped the historical running term).
fn fill_lift_outflow(
    current: &HexGrid,
    dir_out: &[Vec<f32>; 6],
    sat_upper_offset: &[f32],
    lift_upper_snap: &[f32],
    coef: f32,
    dir_out_secondary: &mut [Vec<f32>; 6],
) {
    let cur_cells = current.cells_slice();
    for_each_chunk_mut6(dir_out_secondary, |start, chunks| {
        for local in 0..chunks[0].len() {
            let i = start + local;
            let src_elev = cur_cells[i].elevation;
            let neighbors = current.neighbor_indices_toric(i);
            for (d, &j) in neighbors.iter().enumerate() {
                let flux = dir_out[d][i];
                let elev_delta = cur_cells[j].elevation - src_elev;
                if flux <= 0.0 || elev_delta <= 0.0 {
                    chunks[d][local] = 0.0;
                    continue;
                }
                let lift = (coef * elev_delta).clamp(0.0, 0.80);
                let to_upper_brut = flux * lift;
                let deficit_j = (sat_upper_offset[j] - lift_upper_snap[j]).max(0.0);
                chunks[d][local] = to_upper_brut.min(deficit_j);
            }
        }
    });
}

/// Phase 2 of [`advect_humidity_layer_into`]: per destination cell, the
/// self-loss (`Σ_d dir_out[d][j]`, exactly what `j` itself advected
/// away) and the inflow gathered from neighbors via
/// `coord::opposite_direction`.
fn gather_humidity_deltas(
    current: &HexGrid,
    dir_out: &[Vec<f32>; 6],
    dir_out_secondary: &[Vec<f32>; 6],
    surface_lift_active: bool,
    deltas: &mut [f32],
    lift_deltas_upper: &mut [f32],
) {
    if surface_lift_active {
        for_each_chunk_mut2(deltas, lift_deltas_upper, |start, d_chunk, l_chunk| {
            for local in 0..d_chunk.len() {
                let j = start + local;
                let self_loss = sum_dir_out(dir_out, j);
                let neighbors = current.neighbor_indices_toric(j);
                let mut gathered_flux = 0.0_f32;
                let mut gathered_to_upper = 0.0_f32;
                for (d, &k) in neighbors.iter().enumerate() {
                    let od = opposite_direction(d);
                    gathered_flux += dir_out[od][k];
                    gathered_to_upper += dir_out_secondary[od][k];
                }
                d_chunk[local] = gathered_flux - gathered_to_upper - self_loss;
                l_chunk[local] = gathered_to_upper;
            }
        });
    } else {
        for_each_chunk_mut(deltas, |start, chunk| {
            for (local, d) in chunk.iter_mut().enumerate() {
                let j = start + local;
                let self_loss = sum_dir_out(dir_out, j);
                let neighbors = current.neighbor_indices_toric(j);
                let mut gathered_flux = 0.0_f32;
                for (dd, &k) in neighbors.iter().enumerate() {
                    gathered_flux += dir_out[opposite_direction(dd)][k];
                }
                *d = gathered_flux - self_loss;
            }
        });
    }
}

/// Apply pass of [`advect_humidity_layer_into`]: reads `deltas[i]` (and,
/// under active orographic lift, `lift_deltas_upper[i]`), writes `next[i]`
/// — a pure per-cell map, parallelizable (`par::for_each_chunk_mut`). The
/// scatter pass that fills `deltas`/`lift_deltas_upper` stays serial.
fn apply_humidity_deltas(
    next: &mut HexGrid,
    deltas: &[f32],
    lift_deltas_upper: &[f32],
    surface_lift_active: bool,
    layer: HumidityLayer,
) {
    for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            let i = start + local;
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
    });
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
///
/// Two-phase scatter -> gather (r250 perf effort), same shape as
/// `advect_humidity_layer_into`'s main flux: phase 1 (`dir_out`)
/// computes each source's outflow per direction in parallel, phase 2
/// gathers it back per destination (self-loss = `Σ_d dir_out[d][j]`,
/// inflow via `coord::opposite_direction`).
pub fn advect_cloud_water_into(
    current: &HexGrid,
    next: &mut HexGrid,
    wind_upper: &WindField,
    atmo_params: &AtmosphereParams,
    snap: &mut Vec<f32>,
    deltas: &mut Vec<f32>,
    dir_out: &mut [Vec<f32>; 6],
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
    for dir in dir_out.iter_mut() {
        dir.clear();
        dir.resize(n, 0.0);
    }

    {
        let snap_ref: &Vec<f32> = snap;
        let rate = atmo_params.cloud_advection_rate;
        for_each_chunk_mut6(dir_out, |start, chunks| {
            for local in 0..chunks[0].len() {
                let i = start + local;
                let wind = wind_upper[i];
                let cw = snap_ref[i];
                if cw <= 0.0 {
                    for chunk in chunks.iter_mut() {
                        chunk[local] = 0.0;
                    }
                    continue;
                }
                let mut weights = [0.0_f32; 6];
                let mut total_weight = 0.0_f32;
                for (di, w) in weights.iter_mut().enumerate() {
                    let (dx, dy) = hex_direction_to_world(di);
                    let dot = wind.x * dx + wind.y * dy;
                    if dot > 0.0 {
                        *w = dot;
                        total_weight += dot;
                    }
                }
                if total_weight < 1e-6 {
                    for chunk in chunks.iter_mut() {
                        chunk[local] = 0.0;
                    }
                    continue;
                }
                let inv_total = 1.0 / total_weight;
                let wind_mag = wind.magnitude();
                let fraction = (rate * wind_mag).min(0.95);
                let cw_out = fraction * cw;
                for (d, chunk) in chunks.iter_mut().enumerate() {
                    chunk[local] = if weights[d] > 0.0 {
                        cw_out * weights[d] * inv_total
                    } else {
                        0.0
                    };
                }
            }
        });
    }

    let dir_out_ref: &[Vec<f32>; 6] = dir_out;
    for_each_chunk_mut(deltas, |start, chunk| {
        for (local, d) in chunk.iter_mut().enumerate() {
            let j = start + local;
            let self_loss = sum_dir_out(dir_out_ref, j);
            let neighbors = current.neighbor_indices_toric(j);
            let mut gathered = 0.0_f32;
            for (dd, &k) in neighbors.iter().enumerate() {
                gathered += dir_out_ref[opposite_direction(dd)][k];
            }
            *d = gathered - self_loss;
        }
    });

    // Apply pass: reads only `deltas[i]`, writes only `next[i]` — a pure
    // per-cell map, parallelizable (`par::for_each_chunk_mut`).
    for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            let delta = deltas[start + local];
            if delta != 0.0 {
                cell.cloud_water = (cell.cloud_water + delta).max(0.0);
            }
        }
    });
}

/// Directional advection of temperature by the surface wind.
/// Symmetric transfer proportional to the temperature gradient and the
/// directional weight. Fill-and-gather half only: computes `temp_deltas`,
/// never touches `next`'s cells — the caller applies them (r250 perf
/// effort, chunk B2: `step_atmosphere_transport` fuses the apply with the
/// per-cell passes that immediately follow it, `step_cloud_dynamics` and
/// `step_surface_condensation`, into one sweep — see
/// `atmosphere::apply_temperature_advection_then_cloud_and_condensation` —
/// none of the three ever reads a neighbor's post-fusion value, each only
/// touches cell `i`'s own fields, so the fold is bit-identical to running
/// them as three full-grid passes). The caller is expected to gate the
/// call itself on `wind_params.temperature_advection_rate > 0.0` (no
/// internal early return here: unlike the historical single function,
/// `next` is only read here, so there is nothing to leave untouched on
/// the disabled path).
///
/// Two-phase scatter -> gather (r250 perf effort): unlike the other
/// converted passes, the per-direction outflow here can be negative (a
/// colder source draws heat FROM a warmer downwind neighbor via
/// `temp - target_temp`); the gather sums it with sign exactly as the
/// historical `temp_deltas[i] -= t_delta; temp_deltas[j] += t_delta;`
/// pair did, self-loss included (`Σ_d dir_out[d][j]`, possibly
/// negative), via `coord::opposite_direction`.
pub(crate) fn fill_temp_deltas(
    current: &HexGrid,
    next: &HexGrid,
    wind_field: &WindField,
    wind_params: &WindParams,
    temp_snap: &mut Vec<f32>,
    temp_deltas: &mut Vec<f32>,
    dir_out: &mut [Vec<f32>; 6],
) {
    let n = next.len();
    temp_snap.resize(n, 0.0);
    {
        let next_cells = next.cells_slice();
        for i in 0..n {
            temp_snap[i] = next_cells[i].temperature;
        }
    }
    temp_deltas.resize(n, 0.0);
    for dir in dir_out.iter_mut() {
        dir.clear();
        dir.resize(n, 0.0);
    }

    {
        let temp_snap_ref: &Vec<f32> = temp_snap;
        let rate = wind_params.temperature_advection_rate;
        for_each_chunk_mut6(dir_out, |start, chunks| {
            for local in 0..chunks[0].len() {
                let i = start + local;
                let wind = wind_field[i];
                let temp = temp_snap_ref[i];
                let neighbors = current.neighbor_indices_toric(i);

                let mut weights = [0.0_f32; 6];
                let mut total_weight = 0.0_f32;
                for (di, _dir) in DIRECTIONS.iter().enumerate() {
                    let (dx, dy) = hex_direction_to_world(di);
                    let dot = wind.x * dx + wind.y * dy;
                    if dot > 0.0 {
                        weights[di] = dot;
                        total_weight += dot;
                    }
                }
                if total_weight < 1e-6 {
                    for chunk in chunks.iter_mut() {
                        chunk[local] = 0.0;
                    }
                    continue;
                }
                let inv_total = 1.0 / total_weight;

                for (d, &j) in neighbors.iter().enumerate() {
                    chunks[d][local] = if weights[d] > 0.0 {
                        let target_temp = temp_snap_ref[j];
                        rate * weights[d] * inv_total * (temp - target_temp)
                    } else {
                        0.0
                    };
                }
            }
        });
    }

    let dir_out_ref: &[Vec<f32>; 6] = dir_out;
    for_each_chunk_mut(temp_deltas, |start, chunk| {
        for (local, d) in chunk.iter_mut().enumerate() {
            let j = start + local;
            let self_loss = sum_dir_out(dir_out_ref, j);
            let neighbors = current.neighbor_indices_toric(j);
            let mut gathered = 0.0_f32;
            for (dd, &k) in neighbors.iter().enumerate() {
                gathered += dir_out_ref[opposite_direction(dd)][k];
            }
            *d = gathered - self_loss;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::scaling::temp_advection_boosted_wind_params;
    use crate::atmosphere::test_support::{
        assert_lcl_slack, default_temp_params, default_wind_params, oro_pump_world,
    };
    use crate::atmosphere::{surface_means, total_humidity};
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
            let (mean_t, mean_z) = surface_means(grid);
            scratch.fill_upper_air(grid, mean_t, mean_z, &params, &temp_params);
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

    /// r250 perf effort: mass conservation of the scatter -> gather split
    /// on flat terrain (elevation uniform, so every direction's LCL lift
    /// is inert — isolates the plain directional gather from
    /// `uphill_advection_...` above, which exercises the lift-active
    /// path instead). Wind not aligned to any hex direction, so the flux
    /// spreads over several of the center's 6 neighbors at once (several
    /// non-zero `dir_out` directions), on a radius-2 grid.
    #[test]
    fn advect_humidity_surface_conserves_total_with_multidirectional_wind() {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.elevation = 100.0;
                cell.temperature = 15.0;
            }
        }
        grid.get_mut(HexCoord::new(0, 0)).unwrap().humidity_surface = 40.0;
        let params = AtmosphereParams::default();
        let temp_params = default_temp_params();
        let wind_params = default_wind_params();
        let wind: WindField = vec![WindVec { x: 0.08, y: 0.03 }; grid.len()];
        let before = total_humidity(&grid);

        let mut next = grid.clone();
        let mut scratch = AtmoScratch::new(grid.len());
        let (mean_t, mean_z) = surface_means(&grid);
        scratch.fill_upper_air(&grid, mean_t, mean_z, &params, &temp_params);
        advect_humidity_layer_into(
            &grid,
            &mut next,
            &wind,
            &wind_params,
            &params,
            HumidityLayer::Surface,
            &mut scratch,
        );

        let after = total_humidity(&next);
        assert!(
            (before - after).abs() < 1e-4,
            "conservation violated: before={before}, after={after}"
        );
        let center = next.get(HexCoord::new(0, 0)).unwrap().humidity_surface;
        assert!(
            center < 40.0,
            "center must lose humidity to its downwind neighbors, got {center}"
        );
        for &c in &[
            HexCoord::new(1, 0),
            HexCoord::new(1, -1),
            HexCoord::new(0, -1),
        ] {
            // Bit comparison (precedent time.rs): flat terrain means the
            // lift never fires, so `humidity_upper` stays at its exact
            // untouched default, not merely "close to" zero.
            assert_eq!(
                next.get(c).unwrap().humidity_upper.to_bits(),
                0.0f32.to_bits(),
                "flat terrain: nothing should rise into upper at {c:?}"
            );
        }
    }

    /// r250 perf effort: mass conservation of `advect_cloud_water_into`'s
    /// scatter -> gather split, radius-2, multidirectional wind (see
    /// above).
    #[test]
    fn advect_cloud_water_conserves_total_with_multidirectional_wind() {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.elevation = 100.0;
            }
        }
        grid.get_mut(HexCoord::new(0, 0)).unwrap().cloud_water = 5.0;
        let params = AtmosphereParams::default();
        let wind: WindField = vec![WindVec { x: 0.08, y: 0.03 }; grid.len()];
        let before: f32 = grid.iter().map(|(_, c)| c.cloud_water).sum();

        let mut next = grid.clone();
        let mut snap = Vec::new();
        let mut deltas = Vec::new();
        let mut dir_out: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
        advect_cloud_water_into(
            &grid,
            &mut next,
            &wind,
            &params,
            &mut snap,
            &mut deltas,
            &mut dir_out,
        );

        let after: f32 = next.iter().map(|(_, c)| c.cloud_water).sum();
        assert!(
            (before - after).abs() < 1e-4,
            "conservation violated: before={before}, after={after}"
        );
        let center = next.get(HexCoord::new(0, 0)).unwrap().cloud_water;
        assert!(
            center < 5.0,
            "center must lose cloud water to its downwind neighbors, got {center}"
        );
    }

    /// r250 perf effort: mass conservation of `fill_temp_deltas`'s
    /// scatter -> gather split (the apply is a plain `+=` here, exactly
    /// what the production caller — `atmosphere::
    /// apply_temperature_advection_then_cloud_and_condensation`, chunk
    /// B2 — does per cell, fused into a bigger sweep). Unlike the other
    /// converted passes, a per-direction outflow here can be negative (a
    /// colder cell draws heat back from a warmer neighbor); this pins
    /// that the sign survives the gather (`Σ T` unchanged) on a radius-2
    /// grid with multidirectional wind.
    #[test]
    fn advect_temperature_conserves_total_with_multidirectional_wind() {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.temperature = 10.0;
            }
        }
        grid.get_mut(HexCoord::new(0, 0)).unwrap().temperature = 30.0;
        let wind_params = default_wind_params();
        let wind: WindField = vec![WindVec { x: 0.08, y: 0.03 }; grid.len()];
        let before: f32 = grid.iter().map(|(_, c)| c.temperature).sum();

        let mut next = grid.clone();
        let mut temp_snap = Vec::new();
        let mut temp_deltas = Vec::new();
        let mut dir_out: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
        fill_temp_deltas(
            &grid,
            &next,
            &wind,
            &wind_params,
            &mut temp_snap,
            &mut temp_deltas,
            &mut dir_out,
        );
        for (cell, &delta) in next.cells_slice_mut().iter_mut().zip(temp_deltas.iter()) {
            cell.temperature += delta;
        }

        let after: f32 = next.iter().map(|(_, c)| c.temperature).sum();
        assert!(
            (before - after).abs() < 1e-3,
            "conservation violated: before={before}, after={after}"
        );
        let center = next.get(HexCoord::new(0, 0)).unwrap().temperature;
        assert!(
            center < 30.0,
            "warm center must cool by advecting heat downwind, got {center}"
        );
    }

    /// Ablation #tadv (subsampled cadence): a boosted gather (`sub = 3`,
    /// `temperature_advection_rate` tripled via
    /// `temp_advection_boosted_wind_params`) must move exactly 3x the
    /// heat of the plain hourly gather at `sub = 1` — the transfer is
    /// linear in `rate` (every `dir_out[d]` term above is `rate *
    /// weights[d] * inv_total * (temp - target_temp)`) — and the boosted
    /// result must still conserve the grid's total temperature. Same
    /// world/wind as
    /// `advect_temperature_conserves_total_with_multidirectional_wind`.
    #[test]
    fn temp_advection_boosted_x3_moves_exactly_3x_the_hourly_gather() {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.temperature = 10.0;
            }
        }
        grid.get_mut(HexCoord::new(0, 0)).unwrap().temperature = 30.0;
        let wind_params = default_wind_params();
        let wind: WindField = vec![WindVec { x: 0.08, y: 0.03 }; grid.len()];
        let next = grid.clone();

        let mut temp_snap = Vec::new();
        let mut deltas_hourly = Vec::new();
        let mut dir_out: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
        fill_temp_deltas(
            &grid,
            &next,
            &wind,
            &wind_params,
            &mut temp_snap,
            &mut deltas_hourly,
            &mut dir_out,
        );

        let boosted = temp_advection_boosted_wind_params(&wind_params, 3);
        let mut deltas_boosted = Vec::new();
        fill_temp_deltas(
            &grid,
            &next,
            &wind,
            &boosted,
            &mut temp_snap,
            &mut deltas_boosted,
            &mut dir_out,
        );

        for (i, (&hourly, &tripled)) in deltas_hourly.iter().zip(deltas_boosted.iter()).enumerate()
        {
            assert!(
                (tripled - hourly * 3.0).abs() < 1e-5,
                "cell {i}: boosted delta {tripled} must be 3x the hourly delta {hourly}"
            );
        }

        let before: f32 = grid.iter().map(|(_, c)| c.temperature).sum();
        let mut applied = next.clone();
        for (cell, &delta) in applied
            .cells_slice_mut()
            .iter_mut()
            .zip(deltas_boosted.iter())
        {
            cell.temperature += delta;
        }
        let after: f32 = applied.iter().map(|(_, c)| c.temperature).sum();
        assert!(
            (before - after).abs() < 1e-3,
            "conservation violated at sub=3: before={before}, after={after}"
        );
    }
}
