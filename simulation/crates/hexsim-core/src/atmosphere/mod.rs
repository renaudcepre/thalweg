use crate::climate::DayRecord;
use crate::grid::HexGrid;
use crate::temperature::{TemperatureParams, solar_declination_rad, solar_elevation_at_hour};
use crate::wind::{WindField, WindParams, compute_upper_wind_field_into};

mod advection;
mod condensation;
mod fog;
mod params;
mod precipitation;
mod scaling;
mod scratch;
mod updraft;
mod uplift;

#[cfg(test)]
pub(crate) mod test_support;

use advection::{HumidityLayer, advect_humidity_layer_into, advect_temperature_by_wind_into};
use condensation::{step_cloud_diffusion, step_cloud_dynamics};
use fog::step_surface_condensation;
use precipitation::step_precipitation_into;
use scaling::{
    scale_atmosphere_for_hourly_tick, scale_wind_for_hourly_tick, transport_boosted_params,
    transport_subsample,
};
use updraft::fill_updraft_into;
use uplift::{step_orographic_convection, step_uplift};

// Symbols imported elsewhere as `hexsim_core::atmosphere::X` (stable public
// surface), re-exported here from their implementation sub-module.
pub use advection::advect_cloud_water_into;
pub use condensation::{saturation_surface, saturation_upper, saturation_upper_pw};
pub use params::AtmosphereParams;
pub use precipitation::cloud_water_to_qc;
pub use scratch::AtmoScratch;
pub use uplift::step_evaporation;

/// Water vapor specific constant (J/kg/K). Used to convert a saturation
/// vapor pressure into a density via the ideal gas law:
/// `rho_vap = e_s / (R_vap × T_K)`. Standard value.
pub(crate) const R_VAP: f32 = 461.5;

/// Map of precipitation events produced by one atmospheric tick, indexed by
/// `HexGrid::cell_index` (size = `grid.len()`). Cells with no precipitation:
/// `DayRecord::default()` (rain = 0, snow = 0).
pub type PrecipitationMap = Vec<DayRecord>;

/// Height of the near-ground boundary layer for radiative fog (m).
/// 50 m = typical order of magnitude for a stable night above a humid valley
/// (Stull 1988, *An Introduction to Boundary Layer Meteorology*,
/// Sect 12.3 on nocturnal inversions).
pub const SURFACE_LAYER_M: f32 = 50.0;

/// Full water table reference for normalizing transpiration water stress
/// (cf. `transpiration_coef`, `step_evaporation`). Re-export of
/// `groundwater::DEFAULT_MAX_CAPACITY_MM` (single source of truth).
pub use crate::groundwater::DEFAULT_MAX_CAPACITY_MM as SOIL_GW_REFERENCE_MM;

/// Read-only tick forcing consumed by the atmosphere: wind field and
/// magnitudes memoized at the subsample cadence (#89), params of neighboring
/// phenomena (temperature for lapse rate / LCL, wind for advection), absolute
/// hour for the diurnal cycle. Pattern common to phenomena (cf.
/// `SnowForcing`, `ErosionForcing`): shared inputs travel grouped together
/// and are never mutated (#61).
#[derive(Clone, Copy)]
pub struct AtmoForcing<'a> {
    pub temp_params: &'a TemperatureParams,
    pub wind_params: &'a WindParams,
    pub wind_field: &'a WindField,
    pub wind_mag: &'a [f32],
    pub hour_tick: u64,
}

/// Two-layer atmospheric cycle, strictly closed terrarium.
///
/// Pipeline for one tick:
/// 1. Copy `current` → `next`
/// 2. Compute upper wind (Ekman drift)
/// 3. `step_evaporation`: water + snow → `humidity_surface`
/// 4. `advect_humidity_layer(Surface)`: fresh vapor dispersed BEFORE rising,
///    breaks the captive lake → rain → lake cycle
/// 5. `step_uplift`: `humidity_surface` → `humidity_upper` (thermal + diurnal
///    drive)
/// 6. `advect_humidity_layer(Upper)` advected by the upper wind
/// 7. `advect_temperature_by_wind`
/// 8. `step_precipitation`: consumes `humidity_upper` based on `T_upper`, with
///    a global hysteresis gate
pub fn step_atmosphere(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    forcing: &AtmoForcing<'_>,
    precip_gate_open: &mut bool,
) -> PrecipitationMap {
    let n = current.len();
    let mut scratch = AtmoScratch::new(n);
    let mut events: PrecipitationMap = vec![DayRecord::default(); n];
    step_atmosphere_into(
        current,
        next,
        params,
        forcing,
        precip_gate_open,
        &mut scratch,
        &mut events,
    );
    events
}

/// Zero-malloc variant: uses the scratch buffers supplied (`AtmoScratch`),
/// their capacity is reused from one tick to the next. `forcing.wind_mag` =
/// wind field magnitudes, precomputed by the caller at the field's cadence
/// (subsample #89) instead of one `sqrt` per cell per hour here.
pub fn step_atmosphere_into(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    forcing: &AtmoForcing<'_>,
    precip_gate_open: &mut bool,
    scratch: &mut AtmoScratch,
    events: &mut PrecipitationMap,
) {
    let AtmoForcing {
        temp_params,
        wind_params,
        wind_field,
        wind_mag,
        hour_tick,
    } = *forcing;

    // v0.3.0 PR2: shadow with scaled params for the hourly regime.
    // The caller passes "per day" params (v0.2.x convention); all
    // sub-functions here consume the scaled versions.
    let params_hourly = scale_atmosphere_for_hourly_tick(params);
    let wind_params_hourly = scale_wind_for_hourly_tick(wind_params);
    let params = &params_hourly;
    let wind_params = &wind_params_hourly;

    // Copy current → next via indexed slice: 1 memcpy instead of N HashMap
    // lookups.
    next.cells_slice_mut()
        .clone_from_slice(current.cells_slice());

    compute_upper_wind_field_into(wind_field, wind_params, &mut scratch.wind_upper);

    // Issue #46: positive sin_elev shared across all cells (single
    // latitude for the terrarium). Computed once here, passed to
    // step_uplift to drive the diurnal convective forcing.
    let lat_rad = temp_params.latitude_deg.to_radians();
    let day_of_year = crate::time::day_of_year(hour_tick);
    // Real clock hour (#sub-tick-agnostic, cf. clock_hour_of_day):
    // the diurnal convective drive must see all 24 h even if
    // TICKS_PER_DAY < 24.
    let hour_f = crate::time::clock_hour_of_day(hour_tick);
    let dec_rad = solar_declination_rad(day_of_year);
    let sin_elev_pos = solar_elevation_at_hour(lat_rad, dec_rad, hour_f)
        .sin()
        .max(0.0);

    // Tetens memoization (#97): `saturation_upper(T - t_offset)` on the
    // pre-advection temperature, shared identically by the orographic
    // convection (LCL bound by upper neighbor) and the Surface advection's
    // lift (LCL bound by upward flux). Both read the `current` temperature
    // (temperature advection only happens afterward), so the value is
    // bit-identical to the inline call it replaces.
    scratch.fill_sat_upper_offset(current, params, temp_params);

    step_evaporation(current, next, params, wind_mag);
    // Orographic convection BEFORE advection: fresh vapor evaporated near a
    // relief must rise orographically before being swept away by downslope
    // thermal breezes (which flow down reliefs in a closed terrarium).
    // Critical ordering, breaks if inverted.
    step_orographic_convection(current, next, params, scratch);
    // Horizontal transport subsampling: the humidity/cloud advection and
    // cloud diffusion passes only run one hour out of `sub`, with their
    // rates ×sub (daily transport ≈ conserved). The boosted copies are
    // local to the gated passes: `step_uplift`, `step_orographic_convection`
    // (which shares `orographic_lift_coef`), and temperature advection stay
    // in strict hourly regime.
    let sub = transport_subsample();
    let on_transport_tick = hour_tick.is_multiple_of(u64::from(sub));
    let (params_t, wind_params_t) = transport_boosted_params(params, wind_params, sub);
    let params_t = &params_t;
    let wind_params_t = &wind_params_t;

    if on_transport_tick {
        advect_humidity_layer_into(
            current,
            next,
            wind_field,
            wind_params_t,
            params_t,
            HumidityLayer::Surface,
            scratch,
        );
    }
    step_uplift(next, params, temp_params, sin_elev_pos);
    if on_transport_tick {
        advect_humidity_layer_into(
            current,
            next,
            wind_field,
            wind_params_t,
            params_t,
            HumidityLayer::Upper,
            scratch,
        );
    }
    advect_temperature_by_wind_into(
        current,
        next,
        wind_field,
        wind_params,
        &mut scratch.snap,
        &mut scratch.temp_deltas,
    );

    // After advection/diffusion: humidity_upper masses are spatially
    // stable. We do the vapor ↔ droplet transition before
    // precipitation so cloud_water can accumulate/discharge in
    // equilibrium with the local vapor field.
    step_cloud_dynamics(next, params, temp_params);
    // Issue #45: surface condensation (radiative fog), after the upper
    // cloud_water dynamics, so fog adds its "low" cloud_water to the
    // stock. The cloud_water diffusion + advection that follow will
    // integrate it with the upper one (Option A from the issue,
    // no separate fog_water).
    step_surface_condensation(next, params);
    // Directional advection of cloud_water by the upper wind:
    // droplets that have formed travel with the flow before precipitation,
    // a necessary condition so rain doesn't systematically fall
    // on the vapor source (cell-lake cycle).
    if on_transport_tick {
        advect_cloud_water_into(
            current,
            next,
            &scratch.wind_upper,
            params_t,
            &mut scratch.snap,
            &mut scratch.deltas,
        );
        // Spatial diffusion of cloud_water: smooths the checkerboard pattern.
        // Each cell shares a fraction of its cloud with its neighbors.
        step_cloud_diffusion(
            current,
            next,
            params_t,
            &mut scratch.snap,
            &mut scratch.deltas,
        );
    }

    // Ascent trigger (synoptic Phase 3, ex-design C #69): filled
    // only when active; consumed by precipitation.
    if params.updraft_ref_ms > 0.0 {
        fill_updraft_into(
            current,
            wind_field,
            params.upper_layer_altitude_m,
            &mut scratch.convergence,
        );
    }

    events.resize(next.len(), DayRecord::default());
    events.fill(DayRecord::default());
    step_precipitation_into(next, params, precip_gate_open, events, scratch);
}

/// Total humidity in the grid (surface + upper).
#[must_use]
pub fn total_humidity(grid: &HexGrid) -> f32 {
    grid.iter().map(|(_, cell)| cell.humidity_total()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::wind::compute_wind_field;
    use test_support::{
        default_temp_params, default_wind_params, make_wet_grid, total_moisture, wind_mags,
        zero_wind,
    };

    #[test]
    fn moisture_is_conserved() {
        let current = make_wet_grid();
        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        let wf = zero_wind(&current);
        let wp = default_wind_params();
        let wm = wind_mags(&wf);

        let before = total_moisture(&current);
        step_atmosphere(
            &current,
            &mut next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                hour_tick: 0,
            },
            &mut false,
        );
        let after = total_moisture(&next);

        assert!(
            (before - after).abs() < 1e-2,
            "Conservation violee : {before} -> {after}"
        );
    }

    #[test]
    fn moisture_conserved_with_wind() {
        let current = make_wet_grid();
        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        let wp = default_wind_params();
        let wf = compute_wind_field(&current, &wp, 0);
        let wm = wind_mags(&wf);

        let before = total_moisture(&current);
        step_atmosphere(
            &current,
            &mut next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                hour_tick: 0,
            },
            &mut false,
        );
        let after = total_moisture(&next);

        assert!(
            (before - after).abs() < 1e-1,
            "Conservation with wind violated: {before} -> {after}"
        );
    }

    #[test]
    fn evaporation_feeds_surface_humidity() {
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 10.0;
                cell.temperature = 20.0;
            }
        }

        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                hour_tick: 0,
            },
            &mut false,
        );

        let center = next.get(HexCoord::new(0, 0)).unwrap();
        assert!(center.water_level < 10.0, "water must decrease");
        assert!(
            center.humidity_total() > 0.0,
            "total humidity must increase"
        );
    }

    #[test]
    fn sublimation_below_freezing() {
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.snow_level = 2.0;
            cell.temperature = -5.0;
        }

        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                hour_tick: 0,
            },
            &mut false,
        );

        let center = next.get(c0).unwrap();
        assert!(center.snow_level < 2.0, "snow must decrease");
        assert!(
            center.humidity_total() > 0.0,
            "total humidity must increase"
        );
    }

    #[test]
    fn precipitation_moves_upper_humidity_to_water() {
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                // Phase 3: rescale ×200 (2.0 → 400.0), well above
                // the 24 mm saturation to trigger condensation + precipitation.
                cell.humidity_upper = 400.0;
            }
        }

        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                hour_tick: 0,
            },
            &mut false,
        );

        let center = next.get(HexCoord::new(0, 0)).unwrap();
        assert!(center.humidity_upper < 400.0);
        assert!(center.water_level > 0.0);
    }

    #[test]
    fn precipitation_phase_follows_cell_temperature() {
        // Same oversaturated sky (humidity_upper well above
        // saturation), only the cell temperature differs: below 0°C
        // precipitation must fall as snow and NEVER as liquid water,
        // above as liquid water and NEVER as snow (`step_precipitation_into`,
        // branch `is_snow = nc.temperature < 0.0`).
        //
        // Complement to `phys_wet_peak_snows.rs` (integration test, verifies
        // only the cold-side accumulation over 200 ticks through the full
        // pipeline): here we isolate the phase branching in a single tick of
        // `step_atmosphere`, on two single-cell worlds (radius 0, purely
        // local, the snow/water partition depends only on the source
        // cell's temperature, not on transport), and we
        // explicitly verify that the OTHER stock stays at zero (not
        // just that the right stock increases).
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        let wp = default_wind_params();

        let mut cold = HexGrid::from_radius(0);
        if let Some(c) = cold.get_mut(HexCoord::new(0, 0)) {
            c.temperature = -10.0;
            c.humidity_upper = 400.0;
        }
        let wf_cold = zero_wind(&cold);
        let wind_mag_cold = wind_mags(&wf_cold);
        let mut cold_next = cold.clone();
        step_atmosphere(
            &cold,
            &mut cold_next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf_cold,
                wind_mag: &wind_mag_cold,
                hour_tick: 0,
            },
            &mut false,
        );
        let cold_cell = cold_next.get(HexCoord::new(0, 0)).unwrap();

        let mut warm = HexGrid::from_radius(0);
        if let Some(c) = warm.get_mut(HexCoord::new(0, 0)) {
            c.temperature = 15.0;
            c.humidity_upper = 400.0;
        }
        let wf_warm = zero_wind(&warm);
        let wind_mag_warm = wind_mags(&wf_warm);
        let mut warm_next = warm.clone();
        step_atmosphere(
            &warm,
            &mut warm_next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf_warm,
                wind_mag: &wind_mag_warm,
                hour_tick: 0,
            },
            &mut false,
        );
        let warm_cell = warm_next.get(HexCoord::new(0, 0)).unwrap();

        assert!(
            cold_cell.snow_level > 0.0,
            "cold column (-10°C) under saturated sky must accumulate snow, snow={}",
            cold_cell.snow_level
        );
        assert!(
            cold_cell.water_level.abs() < 1e-6,
            "cold column must NOT receive liquid water, water={}",
            cold_cell.water_level
        );

        assert!(
            warm_cell.water_level > 0.0,
            "warm column (+15°C) under saturated sky must receive liquid water, water={}",
            warm_cell.water_level
        );
        assert!(
            warm_cell.snow_level.abs() < 1e-6,
            "warm column must NOT produce snow, snow={}",
            warm_cell.snow_level
        );
    }

    #[test]
    fn dry_grid_no_change() {
        let grid = HexGrid::from_radius(2);
        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                hour_tick: 0,
            },
            &mut false,
        );

        for (coord, cell) in next.iter() {
            let orig = grid.get(*coord).unwrap();
            assert!(
                (cell.water_level - orig.water_level).abs() < 1e-6
                    && (cell.humidity_surface - orig.humidity_surface).abs() < 1e-6
                    && (cell.humidity_upper - orig.humidity_upper).abs() < 1e-6,
                "state must stay unchanged on dry grid at {coord:?}"
            );
        }
    }

    #[test]
    fn conservation_after_many_steps() {
        let mut current = make_wet_grid();
        let initial = total_moisture(&current);
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        let wp = default_wind_params();

        for tick in 0..100 {
            let wf = compute_wind_field(&current, &wp, tick);
            let wm = wind_mags(&wf);
            let mut next = current.clone();
            step_atmosphere(
                &current,
                &mut next,
                &params,
                &AtmoForcing {
                    temp_params: &tp,
                    wind_params: &wp,
                    wind_field: &wf,
                    wind_mag: &wm,
                    hour_tick: 0,
                },
                &mut false,
            );
            current = next;
        }

        let final_moisture = total_moisture(&current);
        assert!(
            (initial - final_moisture).abs() < 1e-1,
            "Conservation apres 100 steps : {initial} -> {final_moisture}"
        );
    }
}
