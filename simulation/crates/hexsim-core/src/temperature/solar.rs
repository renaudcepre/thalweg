//! Solar astronomy (24h-average model, no diurnal sub-tick): declination,
//! day length, solar elevation, and the instantaneous sun direction vector
//! (`SolarBeam`) with the clear-sky flux it delivers to a surface of a
//! given orientation.
//!
//! Split out of the former single `temperature.rs` (#52 pattern): this is
//! pure ephemeris — a function of latitude, day of year and hour, no grid
//! state — kept separate from [`super::illumination`] (which turns a
//! `SolarBeam` into a per-cell shadowed flux) and [`super::balance`] (which
//! turns that flux into a temperature delta).

use crate::coord::hex_direction_to_world;
use crate::dynamics::CELL_SPACING_M;
use crate::grid::HexGrid;
use crate::time;

use super::{SOLAR_CONSTANT, TemperatureParams};

// ====================================================================
// Solar astronomy (24h-average model, no diurnal sub-tick)
// ====================================================================

/// Solar declination (angle between the sun's rays and the equatorial
/// plane) in radians, for the given day of year. Cooper 1969
/// approximation, error ~0.2° over the year, well below the engine's
/// climate precision.
///
/// Convention: day 0 = January 1st (declination ~-23° = boreal
/// winter), day 172 ≈ June 21 = boreal summer solstice (declination
/// +23.45°).
#[must_use]
pub fn solar_declination_rad(day_of_year: u16) -> f32 {
    let doy = f32::from(day_of_year);
    23.45_f32.to_radians() * (std::f32::consts::TAU * (284.0 + doy) / 365.0).sin()
}

/// Sunrise hour angle, in radians within [0, π]. Classic convention:
/// π/2 = 12h of daylight (equinox), 0 = polar night, π = 24h polar
/// day.
#[must_use]
pub fn hour_angle_sunrise_rad(latitude_rad: f32, declination_rad: f32) -> f32 {
    let cos_omega = -latitude_rad.tan() * declination_rad.tan();
    if cos_omega >= 1.0 {
        0.0
    } else if cos_omega <= -1.0 {
        std::f32::consts::PI
    } else {
        cos_omega.acos()
    }
}

/// Day length in hours for a given latitude and day.
///
/// Reference checks: equator = 12h all year, 45°N summer solstice
/// ≈ 15.4h, 45°N winter solstice ≈ 8.6h, 66.5°N summer solstice = 24h
/// (midnight sun), 66.5°N winter solstice = 0h (polar night).
#[must_use]
pub fn day_length_hours(latitude_deg: f32, day_of_year: u16) -> f32 {
    let lat_rad = latitude_deg.to_radians();
    let dec_rad = solar_declination_rad(day_of_year);
    let omega_0 = hour_angle_sunrise_rad(lat_rad, dec_rad);
    24.0 * omega_0 / std::f32::consts::PI
}

/// Dimensionless 24h-average insolation (Duffie & Beckman 2013, eq
/// 1.10.3). Not a temperature, a *factor* between 0 (polar night) and
/// ~1.25 (polar midnight sun, "compensated" 6 months later by polar
/// night). Reference: equator equinox = 1.0.
#[must_use]
pub fn daily_insolation_factor(latitude_rad: f32, day_of_year: u16) -> f32 {
    let dec = solar_declination_rad(day_of_year);
    let omega_0 = hour_angle_sunrise_rad(latitude_rad, dec);
    omega_0 * latitude_rad.sin() * dec.sin() + latitude_rad.cos() * dec.cos() * omega_0.sin()
}

/// 24h annual average of `max(0, sin(solar_elevation))` at the given
/// latitude, a dimensionless factor in `[0, 1/π]`.
///
/// Computation: average over 365 days of `daily_insolation_factor / π`
/// (the division by π converts D&B's `H0` integral into the 24h
/// average of `sin_elev_pos`). Checks: equator ≈ 0.305, 44.5°N ≈ 0.226,
/// winter pole ≈ 0.
///
/// Used in `step_temperature` to calibrate the structural thermal
/// offset: we set `mean_24h_annual(T_dry_flat) = base_temp` by
/// solving `solar_in_avg = LIN_COEF × (base_temp - calibration_offset)`.
/// This offset absorbs the missing greenhouse effect (#44 will
/// decouple it).
///
/// **Perf note**: 365 iterations × trig per call. To avoid repeating
/// this each tick (+23 ms/tick measured on `scale_ten_year` before the
/// cache), `step_temperature` uses `cached_annual_mean_insolation_factor`.
#[must_use]
pub fn annual_mean_insolation_factor(latitude_rad: f32) -> f32 {
    let mut sum: f32 = 0.0;
    for day in 0..365_u16 {
        sum += daily_insolation_factor(latitude_rad, day);
    }
    sum / 365.0 / std::f32::consts::PI
}

// Thread-local cache for the annual mean insolation: the world's
// latitude only changes at creation or via `update_param`. Without
// this cache, the 365 iterations of `annual_mean_insolation_factor`
// were rerun every tick by all the main loops (atmo + temp +
// climate_history), adding ~23 ms/tick to the budget.
thread_local! {
    static ANNUAL_FACTOR_CACHE: std::cell::Cell<(f32, f32)> =
        const { std::cell::Cell::new((f32::NAN, 0.0)) };
}

// Crate-internal only: `balance::calibration_offset` is the sole other
// caller (`step_temperature`'s calibration), never part of the public API.
#[must_use]
pub(crate) fn cached_annual_mean_insolation_factor(latitude_rad: f32) -> f32 {
    ANNUAL_FACTOR_CACHE.with(|cache| {
        let (cached_lat, cached_factor) = cache.get();
        if (cached_lat - latitude_rad).abs() < 1e-6 {
            cached_factor
        } else {
            let factor = annual_mean_insolation_factor(latitude_rad);
            cache.set((latitude_rad, factor));
            factor
        }
    })
}

/// Solar elevation angle in radians at a precise hour of the day.
/// Sign: positive above the horizon, negative below (night).
/// Reference: Duffie & Beckman 2013, eq 1.6.2.
///
/// `hour_of_day` ∈ [0, 24): 0 = midnight, 12 = local solar noon. The
/// model assumes the sun peaks at 12h (no equation of time, no
/// longitude). v0.3.0 groundwork (#38): to be consumed by
/// `step_temperature` in PR2 to replace the 24h average
/// `daily_insolation_factor` with an instantaneous flux, which is what
/// will produce nighttime freezing at altitude.
///
/// Checks: equinox noon at the equator = π/2 (90°). Summer solstice
/// noon at 44.5°N ≈ 68.95° ≈ 1.203 rad. Midnight at 44.5°N = negative
/// elevation.
#[must_use]
pub fn solar_elevation_at_hour(latitude_rad: f32, declination_rad: f32, hour_of_day: f32) -> f32 {
    // Hour angle ω: solar noon = 0, morning < 0, afternoon > 0.
    // 1h = 15° = π/12 rad.
    let omega = (hour_of_day - 12.0) * std::f32::consts::PI / 12.0;
    // sin(elevation) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
    let sin_elev = latitude_rad.sin() * declination_rad.sin()
        + latitude_rad.cos() * declination_rad.cos() * omega.cos();
    sin_elev.clamp(-1.0, 1.0).asin()
}

/// Solar geometry for one tick: unit vector to the sun in ENU frame
/// (East, North, Up) + clear-sky beam magnitude `beam = S₀·τ·(1−α_sol)`
/// (W/m²). Computed once per tick (cell-independent); the flux
/// received by a surface of a given orientation is obtained via
/// `clear_sky_flux_for_normal`.
#[derive(Debug, Clone, Copy)]
pub struct SolarBeam {
    /// East component of the unit sun vector.
    pub s_e: f32,
    /// North (astronomical) component of the unit sun vector.
    pub s_n: f32,
    /// Up component = `sin(elevation)`. ≤ 0 ⇒ sun below horizon (night).
    pub s_u: f32,
    /// Clear-sky beam before projection: `S₀·τ·(1−α_sol)` (W/m²).
    pub beam: f32,
}

/// Sun vector + clear-sky beam for the current hour. The geometry
/// (declination, hour angle, latitude) is that of
/// `solar_elevation_at_hour`; `s_u` is explicitly routed through it to
/// stay bit-identical to the historical horizontal path
/// (`clear_sky_solar_flux`).
#[must_use]
pub fn solar_beam_at_tick(params: &TemperatureParams, hour_tick: u64) -> SolarBeam {
    let lat_rad = params.latitude_deg.to_radians();
    let dec_rad = solar_declination_rad(time::day_of_year(hour_tick));
    // Actual clock hour (sub-tick-agnostic): needed when
    // `TICKS_PER_DAY` < 24, otherwise the sun would rise over an N-hour day.
    let hour_f = time::clock_hour_of_day(hour_tick);
    let omega = (hour_f - 12.0) * std::f32::consts::PI / 12.0;
    let (sin_phi, cos_phi) = (lat_rad.sin(), lat_rad.cos());
    let (sin_dec, cos_dec) = (dec_rad.sin(), dec_rad.cos());
    let (sin_omega, cos_omega) = (omega.sin(), omega.cos());
    SolarBeam {
        s_e: -cos_dec * sin_omega,
        s_n: sin_dec * cos_phi - cos_dec * cos_omega * sin_phi,
        s_u: solar_elevation_at_hour(lat_rad, dec_rad, hour_f).sin(),
        beam: SOLAR_CONSTANT * params.atmospheric_transmittance * (1.0 - params.ground_albedo),
    }
}

/// Absorbed clear-sky solar flux (W/m²) for a surface of normal
/// `(normal_east, normal_north)` in ENU (Up component reconstructed:
/// `√(1−nₑ²−n_n²)`), before cloud modulation. This is
/// `beam · cos(incidence)` with `cos(incidence) = max(0, S⃗·N⃗)`, the
/// geometry gives zero on its own when the slope faces away from the
/// sun (no arbitrary clamp, anti-pattern #4). Night (`s_u ≤ 0`) ⇒ 0
/// for all cells.
///
/// **Invariant**: flat surface `(0, 0)` ⇒ `beam · max(0, s_u)` = the
/// historical horizontal flux, bit-identical.
#[must_use]
pub fn clear_sky_flux_for_normal(beam: &SolarBeam, normal_east: f32, normal_north: f32) -> f32 {
    if beam.s_u <= 0.0 {
        return 0.0;
    }
    let n_u = (1.0 - normal_east * normal_east - normal_north * normal_north)
        .max(0.0)
        .sqrt();
    let cos_incidence =
        (beam.s_e * normal_east + beam.s_n * normal_north + beam.s_u * n_u).max(0.0);
    beam.beam * cos_incidence
}

/// Shortwave solar flux absorbed at a **horizontal** surface under
/// **clear sky** (W/m²): `S₀ × τ × (1−α_sol) × sin(elevation)⁺`.
/// Shorthand for the flat case and tests; delegates to
/// `clear_sky_flux_for_normal` (single source of truth). The actual
/// per-cell flux depends on its slope orientation (see `SolarBeam`).
#[must_use]
pub fn clear_sky_solar_flux(params: &TemperatureParams, hour_tick: u64) -> f32 {
    clear_sky_flux_for_normal(&solar_beam_at_tick(params, hour_tick), 0.0, 0.0)
}

/// Precomputes each cell's surface normal (ENU components
/// `normal_east`, `normal_north`) from the elevation gradient over its
/// 6 toric neighbors. Call **once** after terrain generation: elevation
/// is fixed (no erosion), so the normal is too.
///
/// Hexagonal finite-volume gradient `∇z = (1/3d)·Σ_k n̂_k·(z_j − z_i)`
/// (idiom from `dynamics.rs`, `d = CELL_SPACING_M`), exact for a
/// linear field. The upward normal of a surface `z(E, y)` is
/// `(−∂z/∂E, −∂z/∂y, 1)/L`; since the world frame has **+y = South**
/// (`hex_direction_to_world`), the South component becomes the North
/// component on sign flip: `normal_north = +∂z/∂y_south / L`. A south
/// facing slope (sunny slope) descends toward the South ⇒
/// `∂z/∂y_south < 0` ⇒ `normal_north < 0`.
pub fn compute_surface_normals(grid: &mut HexGrid) {
    let n = grid.len();
    let inv = 1.0 / (3.0 * CELL_SPACING_M);
    let mut normals: Vec<(f32, f32)> = Vec::with_capacity(n);
    {
        let cells = grid.cells_slice();
        for i in 0..n {
            let neighbors = grid.neighbor_indices_toric(i);
            let elev_i = cells[i].elevation;
            let (mut sum_x, mut sum_y) = (0.0_f32, 0.0_f32);
            for (k, &j) in neighbors.iter().enumerate() {
                let (dx, dy) = hex_direction_to_world(k);
                let delta = cells[j].elevation - elev_i;
                sum_x += dx * delta;
                sum_y += dy * delta;
            }
            let grad_east = inv * sum_x; // ∂z/∂East (m/m)
            let grad_south = inv * sum_y; // ∂z/∂(world +y = South) (m/m)
            let len = (1.0 + grad_east * grad_east + grad_south * grad_south).sqrt();
            normals.push((-grad_east / len, grad_south / len));
        }
    }
    for (cell, (ne, nn)) in grid.cells_slice_mut().iter_mut().zip(normals) {
        cell.normal_east = ne;
        cell.normal_north = nn;
    }
}

/// Correction to the average insolation factor from slope orientation
/// (sunny slope/shaded slope, #102): map × year average (365 d × 24 h)
/// of `max(0, S⃗·N⃗) − max(0, s_u)`. Dimensionless (same unit as
/// `annual_mean_insolation_factor`); added to the calibration offset
/// to recenter the map's average temperature on `base_temp` despite
/// the localization of the flux. Flat terrain ⇒ 0 (bit-identical
/// offset).
///
/// Expensive (≈ N × 8760 dot products) but computed **once** at
/// construction, elevation (and thus the normals) is fixed.
#[must_use]
pub fn aspect_insolation_correction(grid: &HexGrid, params: &TemperatureParams) -> f32 {
    let cells = grid.cells_slice();
    if cells.is_empty() {
        return 0.0;
    }
    // Counter in f32 (avoids a usize→f32 cast): normalize the average
    // per cell at EVERY hour to keep magnitudes small, summing 24.5M
    // raw terms in f32 would lose increments below the granularity.
    let mut n_cells = 0.0_f32;
    for _ in cells {
        n_cells += 1.0;
    }
    let mut sum_hours = 0.0_f32; // sum of hourly per-cell averages (night = 0)
    for day in 0..365_u64 {
        for hour in 0..24_u64 {
            let beam = solar_beam_at_tick(params, day * 24 + hour);
            if beam.s_u <= 0.0 {
                continue; // night: tilted = horiz = 0 everywhere → 0
            }
            let mut sum_cells = 0.0_f32;
            for cell in cells {
                let n_u = (1.0
                    - cell.normal_east * cell.normal_east
                    - cell.normal_north * cell.normal_north)
                    .max(0.0)
                    .sqrt();
                let tilted =
                    (beam.s_e * cell.normal_east + beam.s_n * cell.normal_north + beam.s_u * n_u)
                        .max(0.0);
                // horiz = max(0, s_u) = s_u here (s_u > 0 from the guard).
                sum_cells += tilted - beam.s_u;
            }
            sum_hours += sum_cells / n_cells;
        }
    }
    sum_hours / (365.0 * 24.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::time::TICKS_PER_DAY;

    // ---- Aspect / sun exposure depending on orientation (#102) ----

    #[test]
    fn flat_cell_matches_horizontal() {
        // Exact backward compat: flat cell (vertical normal) ⇒ historical
        // horizontal flux, swept over hour/day of the year.
        let params = TemperatureParams::default();
        for hour_tick in (0..8760u64).step_by(37) {
            let beam = solar_beam_at_tick(&params, hour_tick);
            let flat = clear_sky_flux_for_normal(&beam, 0.0, 0.0);
            // Literal horizontal formula (before refactor).
            let lat = params.latitude_deg.to_radians();
            let dec = solar_declination_rad(time::day_of_year(hour_tick));
            let hour = time::clock_hour_of_day(hour_tick);
            let sin_pos = solar_elevation_at_hour(lat, dec, hour).sin().max(0.0);
            let literal = SOLAR_CONSTANT
                * params.atmospheric_transmittance
                * (1.0 - params.ground_albedo)
                * sin_pos;
            // EXACT equality (bit-identical): compare on the bits to avoid
            // the float_cmp lint while keeping the invariant strict.
            assert_eq!(
                flat.to_bits(),
                literal.to_bits(),
                "flat != horizontal @ hour_tick={hour_tick}"
            );
            assert_eq!(
                flat.to_bits(),
                clear_sky_solar_flux(&params, hour_tick).to_bits()
            );
        }
    }

    #[test]
    fn sun_vector_unit_and_up() {
        let params = TemperatureParams::default();
        for hour_tick in (0..8760u64).step_by(13) {
            let b = solar_beam_at_tick(&params, hour_tick);
            let mag = (b.s_e * b.s_e + b.s_n * b.s_n + b.s_u * b.s_u).sqrt();
            assert!((mag - 1.0).abs() < 1e-3, "|S| = {mag} @ {hour_tick}");
            let lat = params.latitude_deg.to_radians();
            let dec = solar_declination_rad(time::day_of_year(hour_tick));
            let hour = time::clock_hour_of_day(hour_tick);
            let sin_elev = solar_elevation_at_hour(lat, dec, hour).sin();
            assert!((b.s_u - sin_elev).abs() < 1e-6);
        }
    }

    #[test]
    fn aspect_orders_flux() {
        // Summer solstice noon (~day 172), 44.5°N: sunny slope (south) >
        // flat > shaded slope (north).
        let params = TemperatureParams::default();
        let hour_tick = 172_u64 * TICKS_PER_DAY + 12;
        let b = solar_beam_at_tick(&params, hour_tick);
        let tilt = 30.0_f32.to_radians().sin(); // |horizontal component| 30° slope
        let adret = clear_sky_flux_for_normal(&b, 0.0, -tilt); // normal facing south
        let flat = clear_sky_flux_for_normal(&b, 0.0, 0.0);
        let ubac = clear_sky_flux_for_normal(&b, 0.0, tilt); // normal facing north
        assert!(adret > flat, "adret {adret} <= flat {flat}");
        assert!(flat > ubac, "flat {flat} <= ubac {ubac}");
        assert!(
            ubac > 0.0,
            "shaded slope {ubac} should stay lit at summer noon"
        );
        // Very steep north (80°): slope facing away from the sun → 0 net,
        // no clamp.
        let steep_north = clear_sky_flux_for_normal(&b, 0.0, 80.0_f32.to_radians().sin());
        // Flux is ≥ 0 by construction (max(0,·)); ≤ 0 ⟺ exactly 0.
        assert!(
            steep_north <= 0.0,
            "north 80° should drop to 0, got {steep_north}"
        );
    }

    #[test]
    fn normal_from_planar_terrain() {
        // Planar elevation field z = gE·E + gS·y_south. compute_surface_normals
        // must recover (gE, gS) at the center; a south-facing slope (gS<0) ⇒
        // normal_north<0. Locks the 1/(3d) factor and the world axis +y=South.
        let grad_east = 0.03_f32;
        let grad_south = -0.05_f32; // descends south → sunny slope (adret)
        let mut grid = HexGrid::from_radius(1);
        let center = HexCoord::new(0, 0);
        if let Some(c) = grid.get_mut(center) {
            c.elevation = 0.0;
        }
        for (k, dir) in crate::coord::DIRECTIONS.iter().enumerate() {
            let (dx, dy) = hex_direction_to_world(k);
            let elev = CELL_SPACING_M * (grad_east * dx + grad_south * dy);
            if let Some(c) = grid.get_mut(center + *dir) {
                c.elevation = elev;
            }
        }
        compute_surface_normals(&mut grid);
        let c = grid.get(center).unwrap();
        let n_u = (1.0 - c.normal_east * c.normal_east - c.normal_north * c.normal_north)
            .max(0.0)
            .sqrt();
        let recovered_east = -c.normal_east / n_u;
        let recovered_south = c.normal_north / n_u;
        assert!(
            (recovered_east - grad_east).abs() < 1e-4,
            "gE recovered {recovered_east}"
        );
        assert!(
            (recovered_south - grad_south).abs() < 1e-4,
            "gS recovered {recovered_south}"
        );
        assert!(
            c.normal_north < 0.0,
            "south-facing slope ⇒ normal_north<0, got {}",
            c.normal_north
        );
    }

    #[test]
    fn declination_solstices_and_equinoxes() {
        // Northern summer solstice ~ day 172 (June 21): declination = +23.45°
        let dec_summer = solar_declination_rad(172).to_degrees();
        assert!(
            (dec_summer - 23.45).abs() < 0.5,
            "northern summer solstice declination expected ~23.45°, got {dec_summer}"
        );
        // Northern winter solstice ~ day 355 (Dec 21): declination = -23.45°
        let dec_winter = solar_declination_rad(355).to_degrees();
        assert!(
            (dec_winter - -23.45).abs() < 0.5,
            "northern winter solstice declination expected ~-23.45°, got {dec_winter}"
        );
    }

    #[test]
    fn day_length_at_equator_is_always_12_hours() {
        for day in 0..365_u16 {
            let length = day_length_hours(0.0, day);
            assert!(
                (length - 12.0).abs() < 0.01,
                "day {day} at equator should be 12h, got {length}"
            );
        }
    }

    #[test]
    fn day_length_varies_at_mid_latitude() {
        // 45°N: summer solstice ~15.4h, winter solstice ~8.6h
        let summer = day_length_hours(45.0, 172);
        let winter = day_length_hours(45.0, 355);
        assert!(
            (summer - 15.4).abs() < 0.3,
            "45°N summer solstice expected ~15.4h, got {summer}"
        );
        assert!(
            (winter - 8.6).abs() < 0.3,
            "45°N winter solstice expected ~8.6h, got {winter}"
        );
    }

    #[test]
    fn polar_night_and_midnight_sun() {
        // Above the polar circle (~66.5°N), winter solstice = polar night
        // (0h), summer solstice = midnight sun (24h).
        let polar_night = day_length_hours(70.0, 355);
        let midnight_sun = day_length_hours(70.0, 172);
        assert!(
            polar_night < 0.5,
            "70°N winter solstice should be polar night, got {polar_night}h"
        );
        assert!(
            midnight_sun > 23.5,
            "70°N summer solstice should be midnight sun, got {midnight_sun}h"
        );
    }

    #[test]
    fn solar_elevation_equator_equinox_noon_is_zenith() {
        // At the equator, March equinox (~day 80), noon sun = zenith.
        let dec = solar_declination_rad(80);
        let elev = solar_elevation_at_hour(0.0, dec, 12.0);
        // day 80 declination ~ 0 (actually ~ -0.01 rad), so elev expected ~ π/2.
        assert!(
            (elev - std::f32::consts::FRAC_PI_2).abs() < 0.02,
            "equator noon equinox expected ~π/2, got {elev}"
        );
    }

    #[test]
    fn solar_elevation_summer_noon_at_drome() {
        // 44.5°N, summer solstice (day 172), noon: expected elevation
        // ~90° - (44.5° - 23.45°) = 68.95° = 1.2034 rad.
        let lat = 44.5_f32.to_radians();
        let dec = solar_declination_rad(172);
        let elev = solar_elevation_at_hour(lat, dec, 12.0);
        assert!(
            (elev.to_degrees() - 68.95).abs() < 0.5,
            "44.5°N summer solstice noon expected ~68.95°, got {}°",
            elev.to_degrees()
        );
    }

    #[test]
    fn solar_elevation_is_negative_at_midnight() {
        // 44.5°N, summer solstice (short nights): midnight must still be
        // below the horizon. 44.5° > 23.45° = outside the polar circle, no
        // midnight sun.
        let lat = 44.5_f32.to_radians();
        let dec = solar_declination_rad(172);
        let elev = solar_elevation_at_hour(lat, dec, 0.0);
        assert!(
            elev < 0.0,
            "44.5°N midnight should be below horizon, got {}° (elev={elev})",
            elev.to_degrees()
        );
    }

    #[test]
    fn solar_elevation_is_symmetric_around_noon() {
        // By symmetry of the model (no equation of time): elev(12-h) ==
        // elev(12+h). Checked at 44.5°N, summer solstice, offsets 2h / 4h.
        let lat = 44.5_f32.to_radians();
        let dec = solar_declination_rad(172);
        for offset in [2.0_f32, 4.0, 6.0] {
            let morning = solar_elevation_at_hour(lat, dec, 12.0 - offset);
            let afternoon = solar_elevation_at_hour(lat, dec, 12.0 + offset);
            assert!(
                (morning - afternoon).abs() < 1e-5,
                "noon asymmetry at offset {offset}h: morning={morning} evening={afternoon}"
            );
        }
    }

    #[test]
    fn solar_elevation_summer_higher_than_winter() {
        // At 44.5°N, summer solstice noon > winter solstice noon (trivially).
        let lat = 44.5_f32.to_radians();
        let summer_noon = solar_elevation_at_hour(lat, solar_declination_rad(172), 12.0);
        let winter_noon = solar_elevation_at_hour(lat, solar_declination_rad(355), 12.0);
        assert!(
            summer_noon > winter_noon,
            "summer noon ({summer_noon}) should be > winter noon ({winter_noon})"
        );
        // Expected gap: 2 × 23.45° = 46.9° between the two solstices at noon.
        let gap_deg = (summer_noon - winter_noon).to_degrees();
        assert!(
            (gap_deg - 46.9).abs() < 0.5,
            "summer/winter gap expected ~46.9°, got {gap_deg}°"
        );
    }

    #[test]
    fn irradiance_at_equator_equinox_noon_is_realistic() {
        // Checks that the solar flux absorbed at the surface at the
        // equator, equinox, noon (sin_elev = 1) is in the realistic
        // W/m² range. With τ=0.7, α=0.3: 1361 × 0.49 × 1 ≈ 666.9 W/m².
        // Observed reference: 600-900 W/m² clear sky, tropical noon.
        let lat = 0.0_f32.to_radians();
        let dec = solar_declination_rad(80); // spring equinox
        let sin_elev = solar_elevation_at_hour(lat, dec, 12.0).sin().max(0.0);
        let params = TemperatureParams::default();
        let flux = SOLAR_CONSTANT
            * params.atmospheric_transmittance
            * (1.0 - params.ground_albedo)
            * sin_elev;
        assert!(
            (flux - 666.9).abs() < 5.0,
            "equator equinox noon flux expected ~666 W/m², got {flux:.2}"
        );
    }
}
