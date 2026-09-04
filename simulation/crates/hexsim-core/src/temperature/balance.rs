//! The SI radiative balance itself: [`step_temperature`] and its read-only
//! forcing ([`TemperatureForcing`]), plus the small helpers shared with
//! neighboring phenomena (`local_t_ref` for #46 diurnal convection,
//! `cloud_cover_fraction`/`absorbed_solar_flux` for the snowmelt balance).
//!
//! Split out of the former single `temperature.rs` (#52 pattern): this is
//! the local, per-cell energy balance (no raymarch, no ephemeris) that
//! consumes the outputs of [`super::solar`] (the beam) and
//! [`super::illumination`] (the `flux_factor`) without depending on their
//! implementation.

use crate::atmosphere::surface_means;
use crate::grid::HexGrid;
use crate::snow::SnowParams;

use super::{
    ATMO_IR_BACK_CLEAR, ATMO_IR_BACK_CLOUDY_BOOST, LIN_RADIATIVE_COEF, SECONDS_PER_HOUR,
    SENSIBLE_EXCHANGE_COEF, SOLAR_CONSTANT, STEFAN_BOLTZMANN_AT_T0, TemperatureParams,
    cached_annual_mean_insolation_factor, local_heat_capacity, solar_beam_at_tick,
};

/// Cloud cover [0, 1] derived from the `cloud_water` stock (mm PW),
/// normalized to 1 mm PW = dense cloud. Single source of truth,
/// consumed by the radiative balance (`step_temperature`,
/// `absorbed_solar_flux`) and by the snowmelt balance
/// (`snow::step_snow`, #60).
#[must_use]
pub fn cloud_cover_fraction(cloud_water: f32) -> f32 {
    (cloud_water / 1.0).clamp(0.0, 1.0)
}

/// Solar flux absorbed at the surface after cloud modulation (W/m²):
/// `clear_sky × (1 − cloud_albedo)`. Shared source of truth
/// (see `clear_sky_solar_flux`).
#[must_use]
pub fn absorbed_solar_flux(cloud_water: f32, clear_sky_flux: f32, cloud_albedo_coef: f32) -> f32 {
    let cloud_cover = cloud_cover_fraction(cloud_water);
    let cloud_albedo = (cloud_cover * cloud_albedo_coef).min(0.95);
    clear_sky_flux * (1.0 - cloud_albedo)
}

/// Local radiative equilibrium temperature (°C) for the given cell,
/// modulated by altitude (`lapse_rate`) and open water
/// (`water_cooling`). Single source consumed by `step_temperature` AND
/// by the atmosphere phenomena that need the `(T - t_ref)` delta to
/// drive diurnal convection (#46) without duplicating the formula.
#[must_use]
pub fn local_t_ref(
    elevation_m: f32,
    water_level_mm: f32,
    params: &TemperatureParams,
    lat_rad: f32,
) -> f32 {
    let offset = calibration_offset(params, lat_rad);
    let water_cooling_term = params.water_cooling * (1.0 + water_level_mm / 1000.0).ln();
    offset - (elevation_m / 1000.0) * params.lapse_rate - water_cooling_term
}

/// Structural `t_ref` offset (°C) to set the annual 24h average of
/// `T_dry_flat` to `params.base_temp`. Solves:
///
/// ```text
///   net_radiative_avg = LIN_COEF × (base_temp - calibration_offset)
///   => calibration_offset = base_temp - net_radiative_avg / LIN_COEF
///
///   where net_radiative = solar + back_rad - σT0⁴  (linearized around T0=288K)
/// ```
///
/// On dry flat ground at 44.5°N (default, average cloud=0.5):
/// `mean_factor` ≈ 0.226, `solar_avg` ≈ 150 W/m², `back_rad_avg` =
/// 280 + 0.5×60 = 310 W/m², `σT0⁴` = 390 W/m² → `net_avg` ≈ 70 W/m² →
/// offset ≈ +2 °C.
///
/// With #44 (explicit downward IR back-radiation), this offset no
/// longer absorbs the greenhouse effect (now physical), only the
/// non-radiative losses (latent ~−50 W/m², turbulent sensible
/// ~−50 W/m²). Before #44 it was −12.85 °C; after #44 it is ~+2 °C,
/// which is the sign of the greenhouse effect decoupling.
///
/// `mean_factor` composes the flat-world astronomical average
/// (`cached_annual_mean_insolation_factor`) with two terrain
/// corrections: `terrain_insolation_factor` (multiplicative, the real
/// relief's occlusion + diffuse-sky deficit against the flat beam,
/// [`super::terrain_annual_mean_insolation_factor`]) and `aspect_correction`
/// (additive, the pure sunny/shaded-slope tilt). Both default to the
/// flat-world identity (1.0 and 0.0), so `mean_factor` is bit-identical
/// to the pre-terrain-calibration value on a flat map.
fn calibration_offset(params: &TemperatureParams, lat_rad: f32) -> f32 {
    let mean_factor = cached_annual_mean_insolation_factor(lat_rad)
        * params.terrain_insolation_factor
        + params.aspect_correction;
    let mean_solar_flux_annual = SOLAR_CONSTANT
        * params.atmospheric_transmittance
        * (1.0 - params.ground_albedo)
        * mean_factor;
    let mean_cloud = params.mean_cloud_cover_for_calibration.clamp(0.0, 1.0);
    let mean_back_rad = ATMO_IR_BACK_CLEAR + mean_cloud * ATMO_IR_BACK_CLOUDY_BOOST;
    let net_radiative_avg = mean_solar_flux_annual + mean_back_rad - STEFAN_BOLTZMANN_AT_T0;
    params.base_temp - net_radiative_avg / LIN_RADIATIVE_COEF
}

// Strict-SI energy balance (issues #43 + #44, milestone #42).
//
//   net_radiative = solar + back_rad - σT⁴
//                 ≈ solar + back_rad - σT0⁴ - LIN_COEF × (T - T0)   [linearized]
//                 = (solar + back_rad - σT0⁴) - LIN_COEF × (T - T0)
//                                                                  [W/m²]
//
// With:
//   solar     = SOLAR × τ × (1-α_ground) × (1-cloud_albedo) × sin_elev_pos
//   back_rad  = ATMO_IR_BACK_CLEAR + cloud_cover × ATMO_IR_BACK_CLOUDY_BOOST
//   σT0⁴      = STEFAN_BOLTZMANN_AT_T0 ≈ 390 W/m²
//
//   delta_T = thermal_coupling × net_radiative × 3600 / c_local       [K/h]
//
// The historical calibration (`base_temp = target mean annual T`) is
// preserved via `calibration_offset`, which absorbs the non-radiative
// losses (latent, turbulent sensible). Before #44 it also absorbed the
// greenhouse effect (~+33 K), now the greenhouse effect is physical
// via back_rad.
//
// `t_ref` remains the local equilibrium temperature modulated by
// altitude (lapse_rate) and water (water_cooling). `c_local`
// differentiates soil and water cell by cell, which is what makes the
// reduced diurnal amplitude over lakes and the clear-night/cloudy-night
// contrast emerge.
/// Read-only tick forcing consumed by `step_temperature`: the absolute
/// hour (drives the solar beam), a field memoized once per tick by the
/// caller, and the params of a neighboring phenomenon. Pattern common to
/// phenomena (cf. `SnowForcing`, `AtmoForcing`): shared inputs travel
/// grouped together and are never mutated (#61).
#[derive(Clone, Copy)]
pub struct TemperatureForcing<'a> {
    /// Absolute simulation hour (v0.3.0, 24 sub-ticks/day). Indexes the
    /// solar ephemeris (`solar_beam_at_tick`) and the calibration.
    pub hour_tick: u64,
    /// Per-cell illumination factor (aspect × relief occlusion × cloud
    /// shadow), the `scratch_flux_factor` computed once per tick by
    /// `compute_illumination` — `step_temperature` stays local (reads
    /// `flux_factor[i]`), the toric raymarch is isolated there.
    pub flux_factor: &'a [f32],
    /// Snow params of the neighboring phenomenon (`snow::step_snow`),
    /// consumed only for the dry/melt albedo switch and masking depth of
    /// the ice-albedo feedback (#60 Phase 2, single source of truth for
    /// snow albedo shared between the two phenomena).
    pub snow: &'a SnowParams,
}

pub fn step_temperature(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &TemperatureParams,
    forcing: &TemperatureForcing<'_>,
) {
    let TemperatureForcing {
        hour_tick,
        flux_factor,
        snow,
    } = *forcing;

    // Full current → next propagation: without this, fields other
    // than `temperature` (cloud_water, water_level, humidity_*, etc.)
    // that `step_temperature` doesn't explicitly touch are lost at the
    // next buffer swap (see swap pattern in `Simulation`). Symptom:
    // all of the previous tick's `step_atmosphere` work is erased,
    // clouds frozen, rain that doesn't accumulate. Aligns the
    // behavior with `step_snow` and `step_atmosphere_into`, which
    // already do a full copy before their modifications.
    next.cells_slice_mut()
        .clone_from_slice(current.cells_slice());

    let lat_rad = params.latitude_deg.to_radians();

    // This hour's solar beam (clear-sky magnitude `beam`). The
    // per-cell geometric factor (aspect × relief occlusion × cloud
    // shadow) is precomputed in `flux_factor` by
    // `compute_illumination`, step_temperature stays local (reading
    // `flux_factor[i]`), the toric raymarch is isolated.
    let solar = solar_beam_at_tick(params, hour_tick);

    let offset = calibration_offset(params, lat_rad);

    // Mixed boundary-layer air the surface exchanges sensible heat with:
    // map-mean temperature, standard lapse from the map-mean ground
    // (`SENSIBLE_EXCHANGE_COEF`). One pass over the grid per tick.
    let (mean_t, mean_z) = surface_means(current);
    let air_lapse_per_m = params.lapse_rate / 1000.0;

    // Local radiative balance plus the exchange with the shared air
    // (no neighbor lookup) → parallelizable per cell.
    // Index-based (cur[i] → next[i]): zero HashMap lookup, zero coord alloc.
    let cur = current.cells_slice();
    next.cells_slice_mut()
        .iter_mut()
        .zip(cur.iter())
        .zip(flux_factor.iter())
        .for_each(|((nc, cell), &ff)| {
            // LOCAL cloud cover for the IR back-radiation (#44): the
            // downward IR comes from the cloud ABOVE the cell, not from
            // the lateral solar shadow (already counted in `flux_factor`).
            let cloud_cover = cloud_cover_fraction(cell.cloud_water);
            // Absorbed solar flux = beam × illumination factor (aspect +
            // relief occlusion + cloud shadow), computed by
            // `compute_illumination`, corrected by the surface's
            // EFFECTIVE albedo (#60 Phase 2, ice-albedo feedback):
            // `beam` embeds (1−α_sol); a snow-covered cell reflects
            // `α_snow` (0.75) instead of `α_sol` (0.3) in proportion to
            // its optical cover `f = S/(S+S_half)` (masking, Bonan
            // §13.2). Zero snow → factor exactly 1, historical path
            // bit-identical. It's THIS term that makes a snowpack
            // protect itself: it absorbs ~2.8x less solar, the cell
            // stays cold, the snow persists, the ice-albedo loop
            // (scenario gap §5.9, closed).
            let snow_albedo_factor = if cell.snow_level > 0.0 {
                // Cold pack = dry snow (0.80); melting pack (T > 0) =
                // wet snow (0.60, USACE), liquid water absorbs. Same
                // switch as the melt balance (`snow::step_snow`, Phase 4).
                let snow_albedo = if cell.temperature > 0.0 {
                    snow.snow_albedo_melt
                } else {
                    snow.snow_albedo_dry
                };
                let snow_cover = cell.snow_level / (cell.snow_level + snow.snow_masking_half_mm);
                let albedo_eff =
                    params.ground_albedo + (snow_albedo - params.ground_albedo) * snow_cover;
                (1.0 - albedo_eff) / (1.0 - params.ground_albedo).max(1e-6)
            } else {
                1.0
            };
            let solar_in = solar.beam * ff * snow_albedo_factor;
            let back_rad = ATMO_IR_BACK_CLEAR + cloud_cover * ATMO_IR_BACK_CLOUDY_BOOST;

            // Local reference temperature (°C): calibration offset +
            // adiabatic correction + water cooling. Formula shared
            // with `local_t_ref` (public helper used by #46 diurnal
            // convection), changes must stay in sync.
            let water_cooling_term = params.water_cooling * (1.0 + cell.water_level / 1000.0).ln();
            let t_ref = offset - (cell.elevation / 1000.0) * params.lapse_rate - water_cooling_term;

            // Full linearized radiative balance (issue #44):
            //   net = solar + back_rad - σT0⁴ - LIN×(T - t_ref)
            // Note: `t_ref` plays the role of the local T_eq after
            // accounting for the structural offset (which absorbs
            // latent + sensible heat). In fully consistent SI we'd have
            // `(T - T0_C)`; equivalent here since t_ref is calibrated
            // to give the right T_avg by construction.
            // Turbulent sensible heat toward the mixed air at this
            // elevation (bulk aerodynamic formula, `SENSIBLE_EXCHANGE_COEF`):
            // damps the aspect/shadow/cloud anomalies of the surface
            // toward the boundary layer, energy-neutral over the map.
            let t_air = mean_t + air_lapse_per_m * (mean_z - cell.elevation);
            let sensible = SENSIBLE_EXCHANGE_COEF * (cell.temperature - t_air);
            let net_radiative = solar_in + back_rad
                - STEFAN_BOLTZMANN_AT_T0
                - LIN_RADIATIVE_COEF * (cell.temperature - t_ref)
                - sensible;

            // Local surface heat capacity (J/(m²·K)). Fixed soil +
            // water proportional to depth. A 5 m lake: c_local =
            // 360,000 + 5 × 4.186e6 ≈ 21,290,000 J/(m²·K), i.e. ×60 vs
            // dry soil, dominant damping that flattens the diurnal
            // cycle over water.
            let c_local = local_heat_capacity(cell.water_level, cell.groundwater);

            let delta_temp_k = params.thermal_coupling * net_radiative * SECONDS_PER_HOUR / c_local;
            nc.temperature = cell.temperature + delta_temp_k;
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::dynamics::CELL_SPACING_M;
    use crate::temperature::compute_illumination;
    use crate::time;
    use crate::time::TICKS_PER_DAY;

    /// Converts a grid radius (cells, bounded to a few thousand even
    /// for a very small `CELL_SPACING_M`) to a grid index. `as i32` is
    /// the only std path to round a bounded float to an integer (no
    /// `TryFrom<f32>` in std), isolated here, documented, rather than
    /// scattered.
    #[allow(clippy::cast_possible_truncation)]
    fn cells_to_radius(cells: f32) -> i32 {
        cells.ceil() as i32
    }

    /// Test helper: computes illumination (aspect × occlusion × cloud)
    /// then advances temperature. On a grid with null normals and no
    /// cloud, `flux_factor` equals `max(0, sin_elev)` → historical
    /// horizontal behavior, so the thermal balance tests remain valid
    /// unchanged.
    ///
    /// `cloud_altitude_m` deliberately tiny (not `CELL_SPACING_M`):
    /// these tests want the LOCAL (zenith) cloud, never ray-marched
    /// toward a neighbor, otherwise the result would depend on the
    /// engine's resolution (`d_cloud < CELL_SPACING_M` could flip
    /// depending on the constant's value).
    const LOCAL_ZENITH_CLOUD_ALTITUDE_M: f32 = 1.0;

    fn step_temp_lit(
        grid: &HexGrid,
        next: &mut HexGrid,
        params: &TemperatureParams,
        hour_tick: u64,
    ) {
        let beam = solar_beam_at_tick(params, hour_tick);
        let mut ff = Vec::new();
        let mut il = Vec::new();
        compute_illumination(
            grid,
            &beam,
            params.cloud_albedo_coef,
            LOCAL_ZENITH_CLOUD_ALTITUDE_M,
            &mut ff,
            &mut il,
        );
        step_temperature(
            grid,
            next,
            params,
            &TemperatureForcing {
                hour_tick,
                flux_factor: &ff,
                snow: &SnowParams::default(),
            },
        );
    }

    #[test]
    fn high_elevation_is_cold() {
        let mut grid = HexGrid::from_radius(1);
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.elevation = 3000.0;
            c.temperature = 20.0;
        }
        let mut next = grid.clone();
        let params = TemperatureParams::default();

        // 200 days with a fixed day (winter = day 0), full diurnal cycle per
        // iteration via `h % TICKS_PER_DAY`: stay in winter regime to force
        // convergence toward a negative T_ref independent of seasonality.
        for h in 0..(200 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, h % TICKS_PER_DAY);
            std::mem::swap(&mut grid, &mut next);
        }

        let temp = grid.get(HexCoord::new(0, 0)).unwrap().temperature;
        // 3000 m with lapse 6.5 + winter (day 0) at 44.5°N: T_eq expected
        // ≈ calibration_offset (-12.85) - 19.5 + solar_avg_day0/LIN ≈ -21°C.
        // Loose test (< 0), tolerates residual diurnal oscillation.
        assert!(temp < 0.0, "3000m summit should be below zero, got {temp}");
    }

    #[test]
    fn water_cools_cell_in_annual_mean() {
        // Annual mean test: with a full cycle, thermal inertia (×60) no
        // longer affects the mean, only the `water_cooling` effect remains,
        // which shifts t_ref by `water_cooling × ln(1 + depth_mm/1000)`. For
        // a 5 m lake, the expected offset is ln(6) × 1.0 ≈ 1.79 °C.
        let mut grid = HexGrid::from_radius(1);
        let wet_coord = HexCoord::new(0, 0);
        let dry_coord = HexCoord::new(1, 0);
        if let Some(c) = grid.get_mut(wet_coord) {
            c.elevation = 100.0;
            c.temperature = 15.0;
            c.water_level = 5000.0;
        }
        if let Some(c) = grid.get_mut(dry_coord) {
            c.elevation = 100.0;
            c.temperature = 15.0;
            c.water_level = 0.0;
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        // 3 years of warm-up (τ_lake ≈ 45 d → 3 years = ~24 e-folds,
        // saturated), then 1 year measuring the means.
        let warmup = 3 * 365 * TICKS_PER_DAY;
        for h in 0..warmup {
            step_temp_lit(&grid, &mut next, &params, h);
            std::mem::swap(&mut grid, &mut next);
        }
        let mut sum_wet: f32 = 0.0;
        let mut sum_dry: f32 = 0.0;
        let measure = 365 * TICKS_PER_DAY;
        for h in 0..measure {
            step_temp_lit(&grid, &mut next, &params, warmup + h);
            std::mem::swap(&mut grid, &mut next);
            sum_wet += grid.get(wet_coord).unwrap().temperature;
            sum_dry += grid.get(dry_coord).unwrap().temperature;
        }
        let n = f32::from(u16::try_from(measure).expect("365×24 fits u16"));
        let wet_mean = sum_wet / n;
        let dry_mean = sum_dry / n;
        // The `t_ref` offset of a 5 m lake is `water_cooling × ln(6)` ≈
        // 1.79 °C, and the sensible exchange with the mixed air damps
        // every surface anomaly by `LIN / (LIN + H)` (≈ 0.47 with H ≈ 6
        // W/(m²·K)): expected ≈ 0.85 °C, measured 0.85 on 2026-09-02.
        // Half of it is the floor, the sign is the property.
        let nominal = params.water_cooling * (1.0_f32 + 5000.0 / 1000.0).ln();
        let expected = nominal * LIN_RADIATIVE_COEF / (LIN_RADIATIVE_COEF + SENSIBLE_EXCHANGE_COEF);
        assert!(
            wet_mean < dry_mean - 0.5 * expected,
            "annual mean lake must be < plain by at least {:.2}°C (half the damped \
             water_cooling offset {expected:.2}): wet={wet_mean:.2} dry={dry_mean:.2}",
            0.5 * expected
        );
    }

    #[test]
    fn seasons_modulate_temperature() {
        // With the new astro formula, seasons emerge from latitude +
        // declination. At 44.5°N, summer (day 172) must be warmer than
        // winter (day 0).
        let mut grid = HexGrid::from_radius(0);
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.elevation = 0.0;
            c.temperature = 15.0;
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        // 10 days in July to let the cell adjust (240 hourly ticks).
        let summer_base = time::days_to_ticks(172);
        for h in 0..(10 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, summer_base + h);
            std::mem::swap(&mut grid, &mut next);
        }
        let summer = grid.get(HexCoord::new(0, 0)).unwrap().temperature;

        // Reset and 10 days in January.
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.temperature = 15.0;
        }
        for h in 0..(10 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, h);
            std::mem::swap(&mut grid, &mut next);
        }
        let winter = grid.get(HexCoord::new(0, 0)).unwrap().temperature;

        assert!(
            summer > winter + 3.0,
            "44.5°N summer should be warmer than winter by at least 3°C: summer={summer} winter={winter}"
        );
    }

    #[test]
    fn equator_has_minimal_seasonality() {
        // At the equator, seasonality is dampened (h0 varies little between
        // solstices, cos(dec_max) ≈ 0.92 vs 1.0). The summer/winter gap must
        // be << 3°C.
        let mut grid = HexGrid::from_radius(0);
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.elevation = 0.0;
            c.temperature = 15.0;
        }
        let params = TemperatureParams {
            latitude_deg: 0.0,
            ..TemperatureParams::default()
        };

        let mut next = grid.clone();
        let summer_base = time::days_to_ticks(172);
        for h in 0..(100 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, summer_base + h);
            std::mem::swap(&mut grid, &mut next);
        }
        let summer = grid.get(HexCoord::new(0, 0)).unwrap().temperature;

        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.temperature = 15.0;
        }
        for h in 0..(100 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, h);
            std::mem::swap(&mut grid, &mut next);
        }
        let winter = grid.get(HexCoord::new(0, 0)).unwrap().temperature;

        let delta = (summer - winter).abs();
        assert!(
            delta < 3.0,
            "Equator should have low seasonality (<3°C), got delta={delta} (summer={summer} winter={winter})"
        );
    }

    #[test]
    fn annual_mean_converges_toward_base_temp() {
        // In strict SI, `calibration_offset` is calibrated on the *annual
        // mean*, not a fixed day. So we simulate a full annual cycle and
        // measure the 24h mean over the last year. Dry plain, no cloud:
        // T_avg_annual must converge toward `base_temp` by construction of
        // the offset.
        let mut grid = HexGrid::from_radius(0);
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.elevation = 0.0;
            c.temperature = 15.0;
        }
        let params = TemperatureParams {
            latitude_deg: 0.0,
            water_cooling: 0.0,
            // Cell without cloud → align the calibration on 0% mean cloud
            // so the offset stays consistent with the test conditions
            // (otherwise the calibration assumes 50% mean cover, which
            // shifts T_avg by ~5°C in a cloud-free test).
            mean_cloud_cover_for_calibration: 0.0,
            ..TemperatureParams::default()
        };

        let mut next = grid.clone();
        // 3 years of warm-up + 1 year of measurement: dry soil thermal
        // τ ≈ 18 h, plenty of time to converge toward the stationary
        // annual cycle.
        let warmup_ticks = 3 * 365 * TICKS_PER_DAY;
        for h in 0..warmup_ticks {
            step_temp_lit(&grid, &mut next, &params, h);
            std::mem::swap(&mut grid, &mut next);
        }
        let mut sum: f32 = 0.0;
        let measure_ticks = 365 * TICKS_PER_DAY;
        for h in 0..measure_ticks {
            step_temp_lit(&grid, &mut next, &params, warmup_ticks + h);
            std::mem::swap(&mut grid, &mut next);
            sum += grid.get(HexCoord::new(0, 0)).unwrap().temperature;
        }
        // Range of the sum: ~8760 × ±50 °C ≈ ±440k, well within f32
        // precision (exact integers up to 16M).
        let mean_t = sum / f32::from(u16::try_from(measure_ticks).expect("365×24 fits u16"));

        // Tolerance 0.5°C: the discrete mean over 365×24 samples can
        // oscillate slightly relative to the continuous integral.
        assert!(
            (mean_t - params.base_temp).abs() < 0.5,
            "annual mean must converge toward base_temp: mean_t={mean_t:.2}, target={}",
            params.base_temp
        );
    }

    #[test]
    fn cloud_albedo_cools_at_summer_noon() {
        // With #44, the net cloud effect depends on the moment:
        // - Summer noon, full sun: cloud_albedo (-50% solar ≈ -310 W/m²)
        //   largely outweighs the IR boost (+60 W/m²) → cloudy cell
        //   colder.
        // - Night or winter: back_rad dominates → cloudy cell warmer.
        //
        // This test isolates the solar component by measuring the diurnal
        // T_max over 1 summer day after a common warmup (no cloud). At 14h
        // (ticks=14), cloud_water=1 is imposed on the cloudy cell; the
        // afternoon peak shows the irradiance drop.
        let mut grid = HexGrid::from_radius(1);
        let cloudy = HexCoord::new(0, 0);
        let clear = HexCoord::new(1, 0);
        // Uniform elevation on THE WHOLE grid (not just cloudy/clear):
        // otherwise the cells not mentioned stay at the default elevation
        // (0), creating a 200 m step at the grid edge that the relief
        // occlusion of compute_illumination can detect depending on
        // CELL_SPACING_M. The test wants to isolate the cloud effect
        // alone, not relief.
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.elevation = 200.0;
                cell.temperature = 25.0;
                cell.cloud_water = 0.0;
                cell.water_level = 0.0;
            }
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        // Cloud-free summer warmup (both cells at the same T_day).
        let summer = time::days_to_ticks(172);
        for h in 0..(20 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, summer + (h % TICKS_PER_DAY));
            std::mem::swap(&mut grid, &mut next);
        }
        // At 8h, switch cloud on for cloudy. Measure T_max over the rest
        // of the day (8h-20h, so 12 ticks covering noon and afternoon).
        if let Some(c) = grid.get_mut(cloudy) {
            c.cloud_water = 1.0;
        }
        let mut t_cloudy_max = f32::NEG_INFINITY;
        let mut t_clear_max = f32::NEG_INFINITY;
        for offset_h in 8..20_u64 {
            step_temp_lit(&grid, &mut next, &params, summer + offset_h);
            std::mem::swap(&mut grid, &mut next);
            if let Some(c) = grid.get_mut(cloudy) {
                c.cloud_water = 1.0;
            }
            t_cloudy_max = t_cloudy_max.max(grid.get(cloudy).unwrap().temperature);
            t_clear_max = t_clear_max.max(grid.get(clear).unwrap().temperature);
        }
        assert!(
            t_cloudy_max < t_clear_max - 1.0,
            "cloudy cell at summer noon must be colder than clear (cloud_albedo outweighs IR boost): cloudy_max={t_cloudy_max:.2} clear_max={t_clear_max:.2}"
        );
    }

    #[test]
    fn cloud_back_radiation_warms_cloudy_cell_at_night() {
        // Issue #44: even with `cloud_albedo_coef=0` (solar effect disabled),
        // a cloudy cell receives more IR back-radiation
        // (`ATMO_IR_BACK_CLOUDY_BOOST = 60 W/m²`) than a clear cell.
        // Effect visible at night when solar is zero. This test replaces
        // the former `cloud_albedo_zero_isolates_from_clouds` which assumed
        // cloud_water only acted on the solar component.
        let mut grid_cloudy = HexGrid::from_radius(0);
        let mut grid_clear = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        for g in [&mut grid_cloudy, &mut grid_clear] {
            if let Some(c) = g.get_mut(c0) {
                c.elevation = 0.0;
                c.temperature = 10.0;
            }
        }
        if let Some(c) = grid_cloudy.get_mut(c0) {
            c.cloud_water = 1.0;
        }

        let params = TemperatureParams {
            cloud_albedo_coef: 0.0, // solar isolation to measure IR only
            ..TemperatureParams::default()
        };
        let mut next_cloudy = grid_cloudy.clone();
        let mut next_clear = grid_clear.clone();
        // Midnight (sin_elev=0) → solar is zero, isolating the IR component.
        let midnight_tick = time::days_to_ticks(100);
        step_temp_lit(&grid_cloudy, &mut next_cloudy, &params, midnight_tick);
        step_temp_lit(&grid_clear, &mut next_clear, &params, midnight_tick);

        let t_cloudy = next_cloudy.get(c0).unwrap().temperature;
        let t_clear = next_clear.get(c0).unwrap().temperature;
        assert!(
            t_cloudy > t_clear,
            "cloud IR back-radiation must warm (#44): cloudy={t_cloudy} clear={t_clear}"
        );
        // Expected boost: 60 W/m² × 3600s / 360k J/m²K = 0.6 K in 1 tick.
        let delta = t_cloudy - t_clear;
        assert!(
            (delta - 0.6).abs() < 0.05,
            "IR boost per tick expected ≈ 0.6 K, got {delta}"
        );
    }

    #[test]
    fn diurnal_amplitude_dry_plain_summer_is_significant() {
        // Dry plain 44.5°N summer solstice: expected diurnal T amplitude
        // ≥ 5°C (cf. validation criterion #43). The amplitude emerges
        // physically from C_SOIL_SURFACE (time constant τ ≈ 18 h) which
        // does not smooth out the 24 h diurnal forcing.
        let mut grid = HexGrid::from_radius(0);
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.elevation = 0.0;
            c.temperature = 20.0;
            c.water_level = 0.0;
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        // 30 days of warm-up on a fixed day (repeated summer solstice) to
        // let the stationary diurnal cycle settle.
        let summer = time::days_to_ticks(172);
        for h in 0..(30 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, summer + (h % TICKS_PER_DAY));
            std::mem::swap(&mut grid, &mut next);
        }

        // Measure the min/max over the 31st day.
        let mut t_min = f32::INFINITY;
        let mut t_max = f32::NEG_INFINITY;
        for h in 0..TICKS_PER_DAY {
            step_temp_lit(&grid, &mut next, &params, summer + h);
            std::mem::swap(&mut grid, &mut next);
            let t = grid.get(HexCoord::new(0, 0)).unwrap().temperature;
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }
        let amplitude = t_max - t_min;
        assert!(
            amplitude >= 5.0,
            "dry plain summer diurnal amplitude expected ≥ 5°C, got {amplitude:.2} (min={t_min:.2}, max={t_max:.2})"
        );
    }

    #[test]
    fn diurnal_amplitude_lake_smaller_than_dry_plain() {
        // Over a lake (5 m), C_WATER × 5 m = 21 MJ/(m²·K) ≈ 60× larger
        // than C_SOIL → diurnal amplitude crushed. Expected ratio
        // (dry/lake) ≥ 2 (criterion #43, in practice ≫).
        let mut grid = HexGrid::from_radius(1);
        let dry_coord = HexCoord::new(0, 0);
        let wet_coord = HexCoord::new(1, 0);
        for c in [dry_coord, wet_coord] {
            if let Some(cell) = grid.get_mut(c) {
                cell.elevation = 0.0;
                cell.temperature = 20.0;
                cell.water_level = 0.0;
            }
        }
        if let Some(cell) = grid.get_mut(wet_coord) {
            cell.water_level = 5000.0; // 5 m deep
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        // Lake: τ ≈ 1100 h ≈ 45 days. 90-day warm-up to stabilize.
        let summer = time::days_to_ticks(172);
        for h in 0..(90 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, summer + (h % TICKS_PER_DAY));
            std::mem::swap(&mut grid, &mut next);
        }

        let mut dry_min = f32::INFINITY;
        let mut dry_max = f32::NEG_INFINITY;
        let mut wet_min = f32::INFINITY;
        let mut wet_max = f32::NEG_INFINITY;
        for h in 0..TICKS_PER_DAY {
            step_temp_lit(&grid, &mut next, &params, summer + h);
            std::mem::swap(&mut grid, &mut next);
            let t_dry = grid.get(dry_coord).unwrap().temperature;
            let t_wet = grid.get(wet_coord).unwrap().temperature;
            dry_min = dry_min.min(t_dry);
            dry_max = dry_max.max(t_dry);
            wet_min = wet_min.min(t_wet);
            wet_max = wet_max.max(t_wet);
        }
        let dry_amp = dry_max - dry_min;
        let wet_amp = wet_max - wet_min;
        assert!(
            dry_amp >= 2.0 * wet_amp,
            "amplitude ratio (dry/lake) expected ≥ 2: dry_amp={dry_amp:.3} (min={dry_min:.2} max={dry_max:.2}), wet_amp={wet_amp:.3} (min={wet_min:.2} max={wet_max:.2})"
        );
    }

    #[test]
    fn clear_night_colder_than_cloudy_night_in_plain() {
        // Issue #44 criterion: the stronger IR back-radiation under cloud
        // cover (+60 W/m²) must produce a noticeably higher nocturnal T_min
        // than under a clear night. Expected gap ≥ 3 °C.
        //
        // Method: COMMON warmup with no cloud (identical thermal state at
        // dusk), then at night switch cloud_water on for the cloudy cell.
        // T_min is measured over the night (12h).
        let mut grid = HexGrid::from_radius(1);
        let cloudy = HexCoord::new(0, 0);
        let clear = HexCoord::new(1, 0);
        for c in [cloudy, clear] {
            if let Some(cell) = grid.get_mut(c) {
                cell.elevation = 0.0;
                cell.temperature = 25.0;
                cell.cloud_water = 0.0;
                cell.water_level = 0.0;
            }
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        // 30-day summer warmup to reach the stationary daytime thermal
        // state at dusk (both cells at the same T).
        let summer = time::days_to_ticks(172);
        for h in 0..(30 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, summer + (h % TICKS_PER_DAY));
            std::mem::swap(&mut grid, &mut next);
        }
        // At 18h (dusk), switch cloud_water on for the "cloudy" cell.
        // From 18h to 6h = 12 nocturnal ticks (with sin_elev_pos low/zero).
        // The clear cell cools sharply, the cloudy one less (back_rad
        // boost +60 W/m²).
        if let Some(c) = grid.get_mut(cloudy) {
            c.cloud_water = 1.0;
        }
        let mut clear_min = f32::INFINITY;
        let mut cloudy_min = f32::INFINITY;
        // 18 nocturnal ticks (from 18h to 12h the next day, covering the
        // whole night plus early morning when T_min is typically reached).
        for offset_h in 18..36_u64 {
            step_temp_lit(&grid, &mut next, &params, summer + offset_h);
            std::mem::swap(&mut grid, &mut next);
            // re-impose cloud_water (no atmosphere here)
            if let Some(c) = grid.get_mut(cloudy) {
                c.cloud_water = 1.0;
            }
            let t_clear = grid.get(clear).unwrap().temperature;
            let t_cloudy = grid.get(cloudy).unwrap().temperature;
            clear_min = clear_min.min(t_clear);
            cloudy_min = cloudy_min.min(t_cloudy);
        }
        let delta = cloudy_min - clear_min;
        assert!(
            delta >= 3.0,
            "T_min gap cloudy vs clear expected ≥ 3 °C: clear={clear_min:.2} cloudy={cloudy_min:.2} (delta={delta:.2})"
        );
    }

    #[test]
    fn cloudy_summer_night_avoids_frost_in_plain() {
        // Issue #44 criterion: no frost in a plain under a cloudy summer
        // night. back_rad cloudy ≈ 340 W/m² nearly fully offsets the σT⁴
        // nocturnal losses → T stays well positive.
        let mut grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        if let Some(c) = grid.get_mut(coord) {
            c.elevation = 0.0;
            c.temperature = 20.0;
            c.cloud_water = 1.0;
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        let summer = time::days_to_ticks(172);
        for h in 0..(30 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, summer + (h % TICKS_PER_DAY));
            std::mem::swap(&mut grid, &mut next);
            if let Some(c) = grid.get_mut(coord) {
                c.cloud_water = 1.0;
            }
        }
        let mut t_min = f32::INFINITY;
        for h in 0..TICKS_PER_DAY {
            step_temp_lit(&grid, &mut next, &params, summer + h);
            std::mem::swap(&mut grid, &mut next);
            if let Some(c) = grid.get_mut(coord) {
                c.cloud_water = 1.0;
            }
            t_min = t_min.min(grid.get(coord).unwrap().temperature);
        }
        assert!(
            t_min > 0.0,
            "summer plain cloudy night must not freeze: T_min={t_min:.2}"
        );
    }

    #[test]
    fn clear_night_at_altitude_can_freeze_in_winter() {
        // Issue #44 criterion: frost possible at altitude > 1000 m, clear
        // night. At 1500 m + day 0 (winter) + cloud=0, t_ref drops low and
        // the clear back-rad (280 W/m²) is not enough to prevent T < 0 °C.
        let mut grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        if let Some(c) = grid.get_mut(coord) {
            c.elevation = 1500.0;
            c.temperature = 5.0;
            c.cloud_water = 0.0;
        }
        let params = TemperatureParams::default();

        let mut next = grid.clone();
        let winter = 0_u64;
        for h in 0..(30 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, winter + (h % TICKS_PER_DAY));
            std::mem::swap(&mut grid, &mut next);
        }
        let mut t_min = f32::INFINITY;
        for h in 0..TICKS_PER_DAY {
            step_temp_lit(&grid, &mut next, &params, winter + h);
            std::mem::swap(&mut grid, &mut next);
            t_min = t_min.min(grid.get(coord).unwrap().temperature);
        }
        assert!(
            t_min < 0.0,
            "altitude 1500 m winter clear night must be able to freeze: T_min={t_min:.2}"
        );
    }

    #[test]
    fn thermal_coupling_zero_freezes_temperature() {
        // `thermal_coupling = 0.0` must freeze the temperature at its
        // initial value (kill switch used by integration tests that want
        // to isolate the thermal dynamics).
        let mut grid = HexGrid::from_radius(0);
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.elevation = 1500.0;
            c.temperature = 7.5;
            c.water_level = 0.0;
        }
        let params = TemperatureParams {
            thermal_coupling: 0.0,
            ..TemperatureParams::default()
        };
        let mut next = grid.clone();
        for h in 0..(10 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, h);
            std::mem::swap(&mut grid, &mut next);
        }
        let t = grid.get(HexCoord::new(0, 0)).unwrap().temperature;
        assert!(
            (t - 7.5).abs() < 1e-5,
            "thermal_coupling=0 must freeze T at 7.5, got {t}"
        );
    }

    #[test]
    fn effective_lapse_rate_matches_params_at_equator() {
        // On a grid without seasons (equator, equinox), no water, no
        // clouds: the temp-vs-elev slope must equal exactly
        // -lapse_rate/1000. Isolated reference test: the ABSOLUTE radius
        // (not the number of cells) must stay ~5.37 km (5 cells at the
        // original 1074.569 m spacing) so relief occlusion
        // (compute_illumination, absolute threshold ILLUM_FULL_M=30 m)
        // stays negligible as it was at the test's origin. Radius and
        // resolution are two distinct axes (cf feat/dem-terrain-
        // validation): shrinking CELL_SPACING_M without growing the radius
        // in cells narrows the real domain and makes relief non-negligible.
        const SLOPE_M_PER_M: f32 = 500.0 / 1074.569;
        const REFERENCE_RADIUS_M: f32 = 5.0 * 1074.569;
        let radius = cells_to_radius(REFERENCE_RADIUS_M / CELL_SPACING_M);
        let mut grid = HexGrid::from_radius(radius);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                let dist = f32::from(
                    i16::try_from(coord.distance(HexCoord::new(0, 0))).expect("distance fits i16"),
                );
                c.elevation = dist * SLOPE_M_PER_M * CELL_SPACING_M;
                c.temperature = 0.0;
            }
        }
        let params = TemperatureParams {
            latitude_deg: 0.0,
            water_cooling: 0.0,
            ..TemperatureParams::default()
        };

        let mut next = grid.clone();
        let base = time::days_to_ticks(80); // equinox
        for h in 0..(500 * TICKS_PER_DAY) {
            step_temp_lit(&grid, &mut next, &params, base + h);
            std::mem::swap(&mut grid, &mut next);
        }

        let pts: Vec<(f32, f32)> = grid
            .iter()
            .map(|(_, c)| (c.elevation, c.temperature))
            .collect();
        let n = f32::from(u16::try_from(pts.len()).expect("grid fits u16"));
        let mean_x: f32 = pts.iter().map(|(x, _)| x).sum::<f32>() / n;
        let mean_y: f32 = pts.iter().map(|(_, y)| y).sum::<f32>() / n;
        let num: f32 = pts.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
        let den: f32 = pts.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
        let slope = num / den;
        let expected_slope = -params.lapse_rate / 1000.0;

        assert!(
            (slope - expected_slope).abs() < 0.001,
            "effective lapse rate {slope} != expected {expected_slope}"
        );
    }

    // ---- Ice-albedo feedback (#60 Phase 2) ----

    /// Temperature after one summer-noon tick for a dry cell carrying
    /// `snow_mm` of snow. Everything else is identical between runs.
    fn temp_after_noon_tick(snow_mm: f32) -> f32 {
        let noon_summer = 172 * TICKS_PER_DAY + 12;
        let params = TemperatureParams::default();
        let mut grid = HexGrid::from_radius(0);
        let c = grid.get_mut(HexCoord::new(0, 0)).unwrap();
        c.temperature = 5.0;
        c.snow_level = snow_mm;
        let mut next = grid.clone();
        step_temp_lit(&grid, &mut next, &params, noon_summer);
        next.get(HexCoord::new(0, 0)).unwrap().temperature
    }

    /// The ice-albedo loop (scenario gap §5.9, now closed): at noon, the
    /// thicker the snow cover, the less solar the cell absorbs, the less
    /// it heats up, strictly ordered bare > thin > thick. This is the
    /// mechanism that makes a snowpack protect itself (and lets fresh
    /// snow "hold" for several days).
    #[test]
    fn snow_cover_reflects_sunlight_thicker_cover_heats_less() {
        let t_bare = temp_after_noon_tick(0.0);
        let t_thin = temp_after_noon_tick(5.0);
        let t_thick = temp_after_noon_tick(500.0);
        assert!(
            t_thick < t_thin && t_thin < t_bare,
            "noon: warming expected strictly decreasing with snow \
             (bare={t_bare:.4}, 5mm={t_thin:.4}, 500mm={t_thick:.4})"
        );
    }

    /// Ablation control: at night, albedo plays no role (no solar), so
    /// the bare cell and the snowy cell follow EXACTLY the same IR/sensible
    /// balance. Bit-identical: guarantees the feedback introduced no other
    /// snow→temperature coupling besides the solar term.
    #[test]
    fn snow_albedo_has_no_effect_at_night() {
        let midnight_summer = 172 * TICKS_PER_DAY;
        let params = TemperatureParams::default();
        let run = |snow_mm: f32| -> f32 {
            let mut grid = HexGrid::from_radius(0);
            let c = grid.get_mut(HexCoord::new(0, 0)).unwrap();
            c.temperature = 5.0;
            c.snow_level = snow_mm;
            let mut next = grid.clone();
            step_temp_lit(&grid, &mut next, &params, midnight_summer);
            next.get(HexCoord::new(0, 0)).unwrap().temperature
        };
        let t_bare = run(0.0);
        let t_snowy = run(500.0);
        assert!(
            t_bare.to_bits() == t_snowy.to_bits(),
            "night: snow albedo should change nothing (bare={t_bare}, snowy={t_snowy})"
        );
    }
}
