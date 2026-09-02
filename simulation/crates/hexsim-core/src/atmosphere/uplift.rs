use serde::Serialize;

use crate::grid::HexGrid;
use crate::physics::meyer_evaporation;
use crate::species::SPECIES;
use crate::temperature::{TemperatureParams, local_t_ref};
use crate::time::TICKS_PER_DAY_F32;
use crate::wind::wind_magnitude_to_meters_per_second;

use super::{AtmoScratch, AtmosphereParams, SOIL_GW_REFERENCE_MM, saturation_upper};

/// Open-water evaporation stats for the tick, aggregated by `step_evaporation`
/// over the exact cells and formula it uses to move water: no separate
/// recomputation exists anywhere else (a diagnostics-side clone of this
/// formula used to drift from it silently, see #29/#89 history). `mean_mm_day`
/// etc. are the physical rate (Dalton/Meyer ET₀), comparable to observed
/// open-water evaporation (temperate ≈ 2-4 mm/day, warm windy > 8 mm/day),
/// not the per-tick mass actually transferred (which is this rate divided by
/// `TICKS_PER_DAY` and capped to the available surplus).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct EvapStats {
    /// Average over the cells taken into account (true open water: surplus
    /// above `water_capacity`, thawed).
    pub mean_mm_day: f32,
    pub min_mm_day: f32,
    pub max_mm_day: f32,
    /// Number of cells taken into account. 0 when no cell has open water,
    /// in which case every field above is 0 (never a NaN from an empty mean).
    pub cell_count: usize,
}

/// Evaporation of liquid water + plant transpiration + snow sublimation.
/// Feeds `humidity_surface`: freshly evaporated vapor must be lifted by
/// uplift before it can precipitate. This temporal separation avoids the
/// captive lake → rain → lake cycle.
///
/// Phase 2 (#31): free-water evaporation goes through Dalton's law
/// (Meyer 1915) via `meyer_evaporation`. Phase 3 (#32): `humidity_surface`
/// is now directly in mm, so Meyer's mm/day output feeds the stock after
/// division by `TICKS_PER_DAY` (Tier 1, v0.3.0). #77: plant transpiration
/// (FAO-56, cf `transpiration_coef`) reuses this same Meyer demand as ET₀
/// and closes the vegetation → atmosphere loop.
/// Snow sublimation remains phenomenological.
///
/// `out` reports open-water evaporation stats (`EvapStats`) for whoever
/// needs to display them (diagnostics): filled from the same loop, same
/// gate (`open_water > 0`, i.e. a real lake, not any puddle under
/// capacity) and same memoized wind (`wind_mag`) that drive the flux
/// actually applied below. This is now the sole computation of open-water
/// evaporation; nothing else may recompute it (anti-pattern #2).
pub fn step_evaporation(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    wind_mag: &[f32],
    out: &mut EvapStats,
) {
    let mut evap_sum = 0.0_f32;
    let mut evap_min = f32::INFINITY;
    let mut evap_max = 0.0_f32;
    let mut evap_count = 0_usize;

    let cur = current.cells_slice();
    let next_cells = next.cells_slice_mut();
    for (i, cell) in cur.iter().enumerate() {
        // Reference evaporative demand ET₀ (Dalton/Meyer), in mm/day.
        // Shared by free-water evaporation and plant transpiration: same
        // vapor transfer physics, modulated differently downstream.
        // Magnitude precomputed at the cadence of the wind field (#89): the
        // field only changes one hour out of N, the per-cell-hour sqrt was
        // pure recomputation.
        let wind_ms = wind_magnitude_to_meters_per_second(wind_mag.get(i).copied().unwrap_or(0.0));
        let cap = saturation_upper(cell.temperature, params).max(1e-6);
        let rh = (cell.humidity_surface / cap).clamp(0.0, 1.0);
        let evap_demand_per_day = if cell.temperature >= 0.0 {
            meyer_evaporation(cell.temperature, cell.temperature, rh, wind_ms).0
        } else {
            0.0
        };

        // Free-water evaporation (lakes): the demand applies to the open
        // surface, drawn from `water_level`.
        let open_water = (cell.water_level - cell.water_capacity).max(0.0);
        if open_water > 0.0 && cell.temperature >= 0.0 {
            let evap = (evap_demand_per_day / TICKS_PER_DAY_F32).min(open_water);
            next_cells[i].water_level -= evap;
            next_cells[i].humidity_surface += evap;

            // `out`: same gate, same value, no separate pass over the grid.
            evap_sum += evap_demand_per_day;
            evap_min = evap_min.min(evap_demand_per_day);
            evap_max = evap_max.max(evap_demand_per_day);
            evap_count += 1;
        }

        // Plant transpiration (#77, replaces the `ground_evap_rate` proxy).
        // FAO-56: `ET = Kc × ET₀ × water_stress`, with Kc = Kc_max × biomass
        // (biomass [0,1] proxies LAI / cover fraction) and water stress =
        // groundwater saturation. Water is drawn *from* `groundwater` and
        // returned to `humidity_surface`: strict conservation (uptake =
        // transpiration), no double counting.
        // Cover weighted by each species' crop coefficient (FAO-56,
        // #83): Σ crop_coef_i × biomass_i. A forest transpires more than a
        // lawn at equal cover.
        let weighted_cover: f32 = cell
            .vegetation
            .iter()
            .zip(SPECIES.iter())
            .map(|(&v, s)| v * s.crop_coef)
            .sum();
        if weighted_cover > 0.0 && params.transpiration_coef > 0.0 && cell.temperature >= 0.0 {
            let gw_capacity = (cell.permeability * SOIL_GW_REFERENCE_MM).max(1e-6);
            let water_stress = (cell.groundwater / gw_capacity).clamp(0.0, 1.0);
            let kc = params.transpiration_coef * weighted_cover;
            let transp_per_day = evap_demand_per_day * kc * water_stress;
            let transp = (transp_per_day / TICKS_PER_DAY_F32).min(cell.groundwater);
            next_cells[i].groundwater -= transp;
            next_cells[i].humidity_surface += transp;
        }
        if cell.snow_level > 0.0 && cell.temperature < 0.0 {
            let cold_factor = (-cell.temperature / 10.0).clamp(0.0, 1.0);
            let sublim = (params.sublimation_rate * cold_factor).min(cell.snow_level);
            // Exact transfer (cf. `snow::step_snow`): credit humidity with
            // what ACTUALLY left the snow stock. On a glacier several
            // meters deep, f32 ULP (~0.5 mm at 4 m) means the rounded
            // decrement differs from the computed `sublim`; the gap,
            // constant and sign-biased, leaked into the strict
            // conservation test (#60 Phase 3, diagnosed via knockout).
            let new_snow = next_cells[i].snow_level - sublim;
            let departed = next_cells[i].snow_level - new_snow;
            next_cells[i].snow_level = new_snow;
            next_cells[i].humidity_surface += departed;
        }
    }

    *out = if evap_count == 0 {
        EvapStats::default()
    } else {
        let nf = f32::from(u16::try_from(evap_count).unwrap_or(u16::MAX));
        EvapStats {
            mean_mm_day: evap_sum / nf,
            min_mm_day: evap_min,
            max_mm_day: evap_max,
            cell_count: evap_count,
        }
    };
}

/// Intra-cell vertical uplift: transfers a fraction of `humidity_surface`
/// to `humidity_upper`, modulated by thermal convection, orographic
/// uplift, and (issue #46) diurnal convective drive. Vertical transport
/// stays within the same column.
///
/// Diurnal drive (#46): `(T - t_ref).max(0) × sin_elev_pos × coef`. Strong
/// in the afternoon over dry summer plains, zero at night (`sin_elev` =
/// 0). Creates the diurnal cumulus that struggled to emerge under the
/// static regime.
pub(crate) fn step_uplift(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    temp_params: &TemperatureParams,
    sin_elev_pos: f32,
) {
    let lat_rad = temp_params.latitude_deg.to_radians();
    for cell in next.cells_slice_mut().iter_mut() {
        if cell.humidity_surface <= 0.0 {
            continue;
        }
        let temp_boost = cell.temperature.max(0.0) * params.uplift_thermal_coef;
        // Diurnal drive: active only when the sun is above the horizon. At
        // T_excess=25 K and sin_elev=0.9, contribution ≈ 25 × 0.9 ×
        // convective_diurnal_coef. No extra cap here; the final clamp to
        // 0.9 on `rate` guarantees stability.
        let diurnal_drive = if sin_elev_pos > 0.0 && params.convective_diurnal_coef > 0.0 {
            let t_ref = local_t_ref(cell.elevation, cell.water_level, temp_params, lat_rad);
            let t_excess = (cell.temperature - t_ref).max(0.0);
            t_excess * sin_elev_pos * params.convective_diurnal_coef
        } else {
            0.0
        };
        let rate = (params.uplift_rate + temp_boost + diurnal_drive).clamp(0.0, 0.9);
        let transfer = cell.humidity_surface * rate;
        cell.humidity_surface -= transfer;
        cell.humidity_upper += transfer;
    }
}

/// Orographic convection: independent of wind, each cell sends a
/// fraction of `humidity_surface` to the `humidity_upper` of its
/// higher-altitude neighbors. Physically: vapor-laden air near the
/// ground is unstable close to relief (sharp vertical thermal
/// contrasts) → turbulent rise along the slope. Necessary in a closed
/// terrarium where thermal breezes would otherwise flow down relief
/// instead of up it. The split between neighbors is proportional to
/// the positive elevation difference.
pub(crate) fn step_orographic_convection(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    scratch: &mut AtmoScratch,
) {
    if params.orographic_lift_coef <= 0.0 {
        return;
    }
    let n = current.len();
    let next_cells = next.cells_slice_mut();
    let cur_cells = current.cells_slice();

    // Saturation per upper neighbor: precomputed once per tick (#97), read
    // here by index. The temperature consumed (pre-advection) is the one
    // this snapshot captured earlier: same value, bit-identical.
    let sat_upper_offset = &scratch.sat_upper_offset;

    // Snapshot of the fields before deltas. Deltas are computed in pure
    // read mode and applied in pass 2: no write contention possible.
    // Scratch buffers reused (#88/#65): clear() keeps the capacity, so
    // extend() never reallocates after the first tick.
    let src_surface = &mut scratch.oro_src_surface;
    let src_upper = &mut scratch.oro_src_upper;
    let elev_snap = &mut scratch.oro_elev;
    src_surface.clear();
    src_upper.clear();
    elev_snap.clear();
    for i in 0..n {
        src_surface.push(next_cells[i].humidity_surface);
        src_upper.push(next_cells[i].humidity_upper);
        elev_snap.push(cur_cells[i].elevation);
    }

    let delta_surface = &mut scratch.oro_delta_surface;
    let delta_upper_out = &mut scratch.oro_delta_upper_out; // upper losses from the source
    let delta_upper_in = &mut scratch.oro_delta_upper_in; // upper gains for the target
    delta_surface.clear();
    delta_surface.resize(n, 0.0);
    delta_upper_out.clear();
    delta_upper_out.resize(n, 0.0);
    delta_upper_in.clear();
    delta_upper_in.resize(n, 0.0);

    for i in 0..n {
        let src_elev = elev_snap[i];
        let surf = src_surface[i];
        let upper = src_upper[i];
        if surf <= 0.0 && upper <= 0.0 {
            continue;
        }
        // Toric neighborhood: orographic uplift also sees the relief on
        // the other side of the seam (periodic terrain). Without this,
        // edge cells had a truncated neighborhood → ring bias.
        let neighbors = current.neighbor_indices_toric(i);
        let mut total_positive_delta = 0.0_f32;
        for j in neighbors {
            let n_elev = elev_snap[j];
            if n_elev > src_elev {
                total_positive_delta += n_elev - src_elev;
            }
        }
        if total_positive_delta < 1e-6 {
            continue;
        }
        // Cap 0.30: at most 30% of humidity_surface exported per tick.
        // At 0.80 (initial value) the pump sucked everything toward the
        // peaks, creating pathological glaciers that captured ~95% of the
        // water stock within a few years. 0.30 leaves enough humidity for
        // the local cycle without enslaving the system.
        let rate = (params.orographic_lift_coef * total_positive_delta).clamp(0.0, 0.30);
        // Lift surface -> upper(higher neighbor): main flux, rate `rate`.
        let lift_surface_brut = surf * rate;
        // Pump upper -> upper(higher neighbor): 0.25x, secondary suction,
        // much weaker so as not to dry out `humidity_upper` in the
        // lowlands.
        let lift_upper_brut = upper * rate * 0.25;
        let total_in_brut = lift_surface_brut + lift_upper_brut;
        if total_in_brut < 1e-9 {
            continue;
        }

        // LCL bound (#63 Phase 4 Step 3): orographic transport toward a
        // higher neighbor cannot exceed the saturation deficit of
        // `humidity_upper` at the destination. An air parcel rising
        // adiabatically precipitates locally at the LCL: it cannot carry
        // more than `sat_upper - hu_dest`. Without this bound the pump
        // injected HR_upper p99 = 16-288 onto the peaks (cf JOURNAL pivot
        // #63 Phase 4 Step 3).
        //
        // Conservative algorithm: find the attenuation factor `scale ≤ 1`
        // such that for EACH higher neighbor j, `total_in × share_j ≤
        // deficit_j`. The surplus stays at the source in humidity_surface
        // AND humidity_upper proportionally (the physical mechanism =
        // compensating downdraft).
        let mut max_total_in_allowed = f32::INFINITY;
        for j in neighbors {
            let n_elev = elev_snap[j];
            if n_elev <= src_elev {
                continue;
            }
            let share_j = (n_elev - src_elev) / total_positive_delta;
            if share_j < 1e-9 {
                continue;
            }
            let sat_j = sat_upper_offset[j];
            let deficit_j = (sat_j - src_upper[j]).max(0.0);
            let limit = deficit_j / share_j;
            if limit < max_total_in_allowed {
                max_total_in_allowed = limit;
            }
        }
        let scale = (max_total_in_allowed / total_in_brut).clamp(0.0, 1.0);

        let lift_surface = lift_surface_brut * scale;
        let lift_upper = lift_upper_brut * scale;
        let total_in = lift_surface + lift_upper;

        delta_surface[i] -= lift_surface;
        delta_upper_out[i] -= lift_upper;
        for j in neighbors {
            let n_elev = elev_snap[j];
            if n_elev > src_elev {
                let share = (n_elev - src_elev) / total_positive_delta;
                delta_upper_in[j] += total_in * share;
            }
        }
    }

    for (i, cell) in next_cells.iter_mut().enumerate() {
        let ds = delta_surface[i];
        let du = delta_upper_out[i] + delta_upper_in[i];
        if ds == 0.0 && du == 0.0 {
            continue;
        }
        cell.humidity_surface = (cell.humidity_surface + ds).max(0.0);
        cell.humidity_upper = (cell.humidity_upper + du).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::scaling::scale_atmosphere_for_hourly_tick;
    use crate::atmosphere::test_support::{assert_lcl_slack, default_temp_params, oro_pump_world};
    use crate::atmosphere::total_humidity;
    use crate::coord::DIRECTIONS;
    use crate::coord::HexCoord;
    use crate::units::MetersPerSecond;

    #[test]
    fn step_evaporation_stats_count_only_true_open_water() {
        // Pins the bug fix: the old diagnostics-side observer gated on
        // `water_level > 0` (any surface water, including a puddle under
        // capacity), while the engine only ever evaporates the surplus
        // above `water_capacity` (a real lake). `EvapStats` must reflect
        // exactly the cells and formula the engine itself uses to move
        // water, no separate recomputation. Evaporation is cell-local
        // (no neighbor reads), so radius 0-transport caveats don't apply.
        let mut current = HexGrid::from_radius(1);
        let coords: Vec<HexCoord> = current.coords().copied().collect();

        // Lake: surplus above capacity, thawed -> counted.
        if let Some(c) = current.get_mut(coords[0]) {
            c.water_capacity = 1.0;
            c.water_level = 5.0;
            c.temperature = 20.0;
            c.humidity_surface = 5.0;
        }
        // Puddle: water_level > 0 but under capacity -> NOT open water,
        // excluded (this is exactly what the old buggy gate got wrong).
        if let Some(c) = current.get_mut(coords[1]) {
            c.water_capacity = 10.0;
            c.water_level = 3.0;
            c.temperature = 20.0;
        }
        // Frozen lake: surplus above capacity but T < 0 -> excluded.
        if let Some(c) = current.get_mut(coords[2]) {
            c.water_capacity = 1.0;
            c.water_level = 5.0;
            c.temperature = -5.0;
        }
        // Remaining cells stay at the default (water_level=0 < capacity=1):
        // dry, excluded.

        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let wind_mag = vec![0.0_f32; current.len()];
        let mut stats = EvapStats::default();

        step_evaporation(&current, &mut next, &params, &wind_mag, &mut stats);

        assert_eq!(
            stats.cell_count, 1,
            "only the true lake cell (surplus above capacity, thawed) counts as open water"
        );
        // Ground truth computed independently from the same physical
        // formula step_evaporation applies (Dalton/Meyer), not by calling
        // back into the engine: this is what "the flux the engine applied"
        // means for that single cell.
        let cap = saturation_upper(20.0, &params).max(1e-6);
        let rh = (5.0_f32 / cap).clamp(0.0, 1.0);
        let expected = meyer_evaporation(20.0, 20.0, rh, MetersPerSecond(0.0)).0;
        assert!(
            (stats.mean_mm_day - expected).abs() < 1e-4,
            "mean={} expected={expected}",
            stats.mean_mm_day
        );
        assert!((stats.min_mm_day - expected).abs() < 1e-4);
        assert!((stats.max_mm_day - expected).abs() < 1e-4);
    }

    #[test]
    fn step_evaporation_stats_empty_when_no_open_water() {
        // No cell has open water: EvapStats must fall back to all-zero
        // fields (same convention the removed diagnostics-side observer
        // used), never a NaN from dividing by zero cells.
        let current = HexGrid::from_radius(0);
        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let wind_mag = vec![0.0_f32; current.len()];
        let mut stats = EvapStats::default();

        step_evaporation(&current, &mut next, &params, &wind_mag, &mut stats);

        assert_eq!(stats.cell_count, 0);
        assert!(stats.mean_mm_day.abs() < 1e-6);
        assert!(stats.min_mm_day.abs() < 1e-6);
        assert!(stats.max_mm_day.abs() < 1e-6);
    }

    #[test]
    fn diurnal_convection_pushes_more_humidity_at_noon_than_at_night() {
        // Issue #46: with sin_elev_pos > 0 and T > t_ref, the diurnal
        // convective drive adds a boost to `step_uplift`. Compares a call
        // at sin_elev_pos = 0.0 (night) vs 0.95 (summer noon, 44.5°N
        // plain) and checks that humidity_surface decreased more under
        // the diurnal regime.
        fn run_with_sin_elev(sin_elev: f32) -> f32 {
            let mut grid = HexGrid::from_radius(0);
            let c0 = HexCoord::new(0, 0);
            if let Some(cell) = grid.get_mut(c0) {
                cell.humidity_surface = 100.0;
                // T = 30 °C >> t_ref(plain 44.5°N) ≈ 2 °C ⇒ t_excess ≈ 28 K
                cell.temperature = 30.0;
                cell.elevation = 0.0;
                cell.water_level = 0.0;
            }
            let params = AtmosphereParams::default();
            // Convert to "hourly" regime to test the real value of a
            // single tick (otherwise convective_diurnal_coef = 0.0005 per
            // day gives almost nothing on a single call).
            let params_hourly = scale_atmosphere_for_hourly_tick(&params);
            let tp = default_temp_params();
            step_uplift(&mut grid, &params_hourly, &tp, sin_elev);
            grid.get(c0).unwrap().humidity_surface
        }

        let after_night = run_with_sin_elev(0.0);
        let after_noon = run_with_sin_elev(0.95);
        let drop_night = 100.0 - after_night;
        let drop_noon = 100.0 - after_noon;
        assert!(
            drop_noon > drop_night * 1.05,
            "drive diurne devait pousser plus d'humidite en haut a midi qu'a minuit : night drop={drop_night:.4} noon drop={drop_noon:.4}"
        );
    }

    #[test]
    fn phys_oro_lift_bounded_by_saturation() {
        // Issue #63 Phase 4 Step 3: LCL bound on the orographic pump.
        //
        // Setup: 1 low cell (radius 0) surrounded by 6 high neighbors
        // already at HR_upper ≈ 0.99. Source full of humidity_surface.
        // Without the LCL bound, `step_orographic_convection` injects 30%
        // of surf into the neighbor's upper on every tick → HR_upper >> 1
        // immediately. With the bound, transport is capped by the
        // downstream saturation deficit, so the neighbor's HR_upper stays
        // ≤ 1 + epsilon.
        let mut current = HexGrid::from_radius(1);
        let center = HexCoord::new(0, 0);
        let coords: Vec<HexCoord> = current.coords().copied().collect();
        let neighbors: Vec<HexCoord> = coords.iter().copied().filter(|&c| c != center).collect();
        let temp_params = default_temp_params();
        let params = AtmosphereParams::default();
        let t_offset = temp_params.lapse_rate * params.upper_layer_altitude_m / 1000.0;

        // Low source, saturated with surface humidity
        if let Some(c) = current.get_mut(center) {
            c.elevation = 0.0;
            c.humidity_surface = 100.0;
            c.humidity_upper = 0.0;
            c.temperature = 20.0;
        }
        // High neighbors, already close to upper saturation
        for &nc in &neighbors {
            if let Some(c) = current.get_mut(nc) {
                c.elevation = 200.0;
                c.temperature = 18.0; // slightly colder T
                let t_upper = 18.0 - t_offset;
                let sat = saturation_upper(t_upper, &params);
                c.humidity_upper = sat * 0.99;
                c.humidity_surface = 0.0;
            }
        }
        let mut next = current.clone();

        let mut scratch = AtmoScratch::new(current.len());
        // Orographic convection consumes the precomputed `sat_upper_offset`
        // (#97); on a direct call outside `step_atmosphere_into`, fill it
        // here.
        scratch.fill_sat_upper_offset(&current, &params, &temp_params);
        step_orographic_convection(&current, &mut next, &params, &mut scratch);

        // LCL invariant: no high neighbor should be oversaturated.
        for &nc in &neighbors {
            let cell = next.get(nc).unwrap();
            let t_upper = cell.temperature - t_offset;
            let sat = saturation_upper(t_upper, &params);
            let hr = cell.humidity_upper / sat;
            assert!(
                hr <= 1.05,
                "high neighbor oversaturated: hr={hr}, hu={}, sat={sat}",
                cell.humidity_upper
            );
        }

        // Conservation invariant: the source lost exactly what the
        // neighbors gained (conservative pump + surplus falls back to the
        // source).
        let total_before = 100.0_f32
            + neighbors.iter().fold(0.0_f32, |acc, &nc| {
                acc + current.get(nc).unwrap().humidity_upper
            });
        let total_after: f32 = coords
            .iter()
            .map(|&c| {
                let cell = next.get(c).unwrap();
                cell.humidity_surface + cell.humidity_upper
            })
            .sum();
        assert!(
            (total_after - total_before).abs() < 1e-3,
            "conservation violated: before={total_before}, after={total_after}"
        );
    }

    /// Harness for the orographic pump micro-tests: builds the world via
    /// `oro_pump_world` (`test_support`) then runs
    /// `step_orographic_convection` in isolation.
    fn run_oro_pump(current: &HexGrid) -> HexGrid {
        let params = AtmosphereParams::default();
        let temp_params = default_temp_params();
        let mut next = current.clone();
        let mut scratch = AtmoScratch::new(current.len());
        scratch.fill_sat_upper_offset(current, &params, &temp_params);
        step_orographic_convection(current, &mut next, &params, &mut scratch);
        next
    }

    /// The orographic pump is an elevator, not a diffuser: surface vapor
    /// rises into `humidity_upper` of the HIGHER neighbor, and exactly
    /// nothing goes to the lower neighbor or to neighbors at the same
    /// elevation. Directional complement to
    /// `phys_oro_lift_bounded_by_saturation` (which pins the LCL bound and
    /// conservation, not the direction of transport).
    #[test]
    fn orographic_pump_lifts_only_toward_higher_neighbors() {
        let (mut grid, coords) = oro_pump_world(10.0);
        let center = HexCoord::new(0, 0);
        let uphill = center + DIRECTIONS[0];
        let downhill = center + DIRECTIONS[3];
        grid.get_mut(uphill).unwrap().elevation = 500.0;
        grid.get_mut(downhill).unwrap().elevation = 0.0;
        assert_lcl_slack(10.0 * 0.30); // rate capped at 30%/tick

        let next = run_oro_pump(&grid);

        let up = next.get(uphill).unwrap();
        assert!(
            up.humidity_upper > 0.0,
            "the high neighbor must receive vapor in the upper layer"
        );
        assert!(
            up.humidity_surface == 0.0,
            "the pump delivers aloft, not at the surface: surf={}",
            up.humidity_surface
        );
        for &c in &coords {
            if c == center || c == uphill {
                continue;
            }
            let cell = next.get(c).unwrap();
            assert!(
                cell.humidity_surface == 0.0 && cell.humidity_upper == 0.0,
                "only the HIGH neighbor receives, leak toward {c:?} \
                 (surf={}, upper={})",
                cell.humidity_surface,
                cell.humidity_upper
            );
        }
        let lost = 10.0 - next.get(center).unwrap().humidity_surface;
        assert!(
            (lost - up.humidity_upper).abs() < 1e-4,
            "the center must lose exactly what the peak gains: \
             lost={lost}, gained={}",
            up.humidity_upper
        );
    }

    /// The split between higher neighbors is proportional to the
    /// positive elevation difference: a neighbor at +400 m receives 4x
    /// what a neighbor at +100 m receives (`share_j = Δz_j / ΣΔz⁺`). Same
    /// temperatures and empty upper → the LCL bound doesn't bite, the
    /// ratio is exact.
    #[test]
    fn orographic_pump_share_scales_with_elevation_gap() {
        let (mut grid, _coords) = oro_pump_world(10.0);
        let center = HexCoord::new(0, 0);
        let tall = center + DIRECTIONS[0];
        let short = center + DIRECTIONS[2];
        grid.get_mut(tall).unwrap().elevation = 500.0; // +400 m
        grid.get_mut(short).unwrap().elevation = 200.0; // +100 m
        assert_lcl_slack(10.0 * 0.30);

        let next = run_oro_pump(&grid);

        let g_tall = next.get(tall).unwrap().humidity_upper;
        let g_short = next.get(short).unwrap().humidity_upper;
        assert!(
            g_tall > 0.0 && g_short > 0.0,
            "both high neighbors must receive: tall={g_tall}, short={g_short}"
        );
        let ratio = g_tall / g_short;
        assert!(
            (ratio - 4.0).abs() < 1e-3,
            "share ∝ elevation gap: ratio measured {ratio}, expected 4.0 (400 m / 100 m)"
        );
    }

    /// Ablation: without relief, the pump is inert. It really is the
    /// ELEVATION DIFFERENCE that causes the transport in the two previous
    /// tests, not the mere presence of humidity: a flat world comes out
    /// bit-identical, no diffuse leak.
    #[test]
    fn orographic_pump_is_inert_on_flat_terrain() {
        let (grid, coords) = oro_pump_world(50.0);
        let next = run_oro_pump(&grid);
        for &c in &coords {
            let before = grid.get(c).unwrap();
            let after = next.get(c).unwrap();
            // Bit comparison (precedent time.rs): keeps the strict
            // "bit-identical" invariant without clippy's float_cmp.
            assert!(
                before.humidity_surface.to_bits() == after.humidity_surface.to_bits()
                    && before.humidity_upper.to_bits() == after.humidity_upper.to_bits(),
                "flat terrain: {c:?} moved (surf {} → {}, upper {} → {})",
                before.humidity_surface,
                after.humidity_surface,
                before.humidity_upper,
                after.humidity_upper
            );
        }
    }

    #[test]
    fn uplift_conserves_total_humidity() {
        let mut grid = HexGrid::from_radius(3);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.humidity_surface = 60.0;
                cell.temperature = 15.0;
            }
        }
        let before = total_humidity(&grid);
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        step_uplift(&mut grid, &params, &tp, 0.5);
        let after = total_humidity(&grid);
        // Relative tolerance (≤ 1e-5) vs absolute: the Phase 3 ×200
        // rescale amplifies numerical noise in absolute value while
        // preserving f32 relative precision.
        let drift = (before - after).abs() / before.max(1.0);
        assert!(drift < 1e-5, "Uplift non conservatif : {before} -> {after}");
    }

    #[test]
    fn uplift_moves_surface_to_upper() {
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.humidity_surface = 100.0;
            cell.temperature = 20.0;
        }
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        step_uplift(&mut grid, &params, &tp, 0.5);
        let after = grid.get(c0).unwrap();
        assert!(after.humidity_surface < 100.0);
        assert!(after.humidity_upper > 0.0);
    }
}
