use serde::{Deserialize, Serialize};

use crate::coord::hex_direction_to_world;
use crate::dynamics::CELL_SPACING_M;
use crate::grid::HexGrid;
use crate::time;

// ====================================================================
// Physical constants (strict SI: W/m², J/(m²·K), kg/m³, Pa, K — constants
// sourced from the literature, no unnamed dimensionless coefficients)
// ====================================================================

/// Solar constant, average irradiance at the top of Earth's atmosphere
/// (W/m²). Reference: World Radiation Center, value stable to within
/// ±0.1% over the 11-year solar cycle.
pub const SOLAR_CONSTANT: f32 = 1361.0;

/// Stefan-Boltzmann constant (W/(m²·K⁴)). Reference: CODATA 2018.
pub const STEFAN_BOLTZMANN: f32 = 5.67e-8;

/// Reference temperature for the linearization of radiative cooling
/// around 15 °C (Earth's global mean `T_eq`). In kelvin.
pub const T0_REF_KELVIN: f32 = 288.0;

/// Linearized radiative cooling coefficient: `4·σ·T0³`.
/// = 4 × 5.67e-8 × 288³ ≈ 5.4096 W/(m²·K). Appears when expanding
/// `σT⁴ ≈ σT0⁴ + 4σT0³(T-T0)` to 1st order around the global mean T.
/// Reference: Hartmann D. (1994), *Global Physical Climatology*, eq 2.62.
pub const LIN_RADIATIVE_COEF: f32 =
    4.0 * STEFAN_BOLTZMANN * T0_REF_KELVIN * T0_REF_KELVIN * T0_REF_KELVIN;

/// Surface heat capacity of soil (J/(m²·K)).
/// 0.3 m × 1500 kg/m³ × 800 J/(kg·K) = 360,000 J/(m²·K). Reference:
/// Bonan G. (2008), *Ecological Climatology* 2nd ed., table 12.1
/// (typical mineral soil, diurnal heat-exchange depth ~30 cm).
pub const C_SOIL_SURFACE: f32 = 360_000.0;

/// Surface heat capacity of water, per meter of depth
/// (J/(m²·K)·m⁻¹). = `c_p_water` × `ρ_water` = 4186 × 1000 = 4,186,000.
/// For a cell with `water_level` mm, multiply by
/// `water_level / 1000.0` (depth in meters).
pub const C_WATER_PER_METER: f32 = 4_186_000.0;

/// Seconds per hourly tick. Our tick = 1 h (v0.3.0 #38).
pub const SECONDS_PER_HOUR: f32 = 3600.0;

/// Local surface heat capacity (J/(m²·K)): fixed soil + surface free
/// water (`water_level`) + soil water (`groundwater`), both in mm.
/// The water table in this model is soil water (root zone, capacity
/// ~60 mm), not a deep aquifer, so it sits in the diurnal exchange
/// layer and counts toward thermal inertia. Important physical
/// consequence: moving water surface→soil (e.g. melt percolating,
/// `snow::melt_recharge_frac`) does NOT change the cell's thermal
/// mass, so there is no spurious melt→warming→melt feedback that
/// would make high-altitude snowpack vanish. Single source of truth
/// shared between `step_temperature` and the fire heat injection
/// (`fire::step_fire`), no divergent recomputation (anti-pattern #2).
#[must_use]
pub fn local_heat_capacity(water_level_mm: f32, groundwater_mm: f32) -> f32 {
    C_SOIL_SURFACE + C_WATER_PER_METER * ((water_level_mm + groundwater_mm) / 1000.0)
}

/// Reference IR emission from a black body at T0 = 288 K (W/m²).
/// = σT0⁴ ≈ 390 W/m². Appears in the radiative balance when expanding
/// `σT⁴ ≈ σT0⁴ + LIN_COEF × (T - T0)` to 1st order.
pub const STEFAN_BOLTZMANN_AT_T0: f32 =
    STEFAN_BOLTZMANN * T0_REF_KELVIN * T0_REF_KELVIN * T0_REF_KELVIN * T0_REF_KELVIN;

/// Downward atmospheric IR back-radiation under clear sky (W/m²).
/// = `ε_atmo` × σ × `T_atmo⁴` with `ε_atmo` ≈ 0.75 (average humid
/// atmosphere) and `T_atmo` ≈ 268 K (~ emitting layer `T_eff`).
/// Reference: Brutsaert W. (1975), *On a derivable formula for
/// long-wave radiation from clear skies*, Water Resources Research
/// 11(5), 742-744.
///
/// Issue #44: this is the flux that keeps the ground surface from
/// cooling to -20 °C at night. Without an atmosphere, the soil would
/// lose σT⁴ (~390 W/m² at 15 °C) and drop sharply.
pub const ATMO_IR_BACK_CLEAR: f32 = 280.0;

/// Back-radiation bonus for full cloud cover (W/m²).
/// Thick clouds act as an IR black body (ε ≈ 1, vs 0.75 for clear
/// atmosphere) → +60-90 W/m² extra under cloud. Notable asymmetry:
/// `280 + 60×cover` rather than `340 - 60×(1-cover)` because water
/// vapour (clear atmosphere) is less efficient in IR than condensed
/// liquid (cloud). Reference: Held I. & Soden B. (2000), *Water
/// vapor feedback and global warming*, Annual Review of Energy and
/// the Environment 25, 441-475.
pub const ATMO_IR_BACK_CLOUDY_BOOST: f32 = 60.0;

#[derive(Clone, Serialize, Deserialize)]
pub struct TemperatureParams {
    /// Target mean annual temperature (°C) on dry flat ground (no
    /// water, no cloud), at latitude `latitude_deg`. Acts as a climate
    /// target: a structural offset is computed each tick so that
    /// `mean_24h_annual(T) = base_temp` by construction. Absorbs the
    /// greenhouse effect missing from the model (to come, #44) and
    /// unmodeled losses (latent, turbulent sensible heat).
    pub base_temp: f32,
    /// Adiabatic lapse rate (°C/km). 6.5 = standard atmosphere.
    pub lapse_rate: f32,
    /// Cooling from open water (°C). Modulates `t_ref` by
    /// `water_cooling × ln(1 + water_level/1000)`, a log so a deep
    /// lake doesn't crash the temperature.
    pub water_cooling: f32,
    /// Latitude of the simulated world, in degrees (signed: positive =
    /// North, negative = South). Defines the solar geometry: seasonal
    /// declination, day length, zenith angle at noon.
    ///
    /// The whole map has a SINGLE latitude, no North-South biome
    /// gradient on one terrarium (design choice: the world is one closed
    /// terrarium, not a plane spanning multiple latitudes). Longitude is
    /// ignored: the whole terrarium lives in the same time zone.
    pub latitude_deg: f32,
    /// Cloud feedback on solar forcing. Each cell attenuates its
    /// irradiance by `(1 - cloud_albedo_coef * cloud_frac)`. Creates a
    /// negative feedback loop: more condensed humidity → less sun →
    /// T drops → less evaporation → dissipation → sun comes back.
    /// Main source of weather variability in the closed terrarium.
    pub cloud_albedo_coef: f32,
    /// Average atmospheric transmittance (dimensionless, [0,1]).
    /// Effective Beer-Lambert: fraction of the solar flux that
    /// reaches the ground under clear sky. ~0.7 = standard atmosphere
    /// with aerosols (Duffie & Beckman 2013, eq 2.11.1, continental
    /// climate tables).
    pub atmospheric_transmittance: f32,
    /// Average ground albedo (dimensionless, [0,1]). Fraction of the
    /// flux reflected back to space. 0.3 = typical grass/forest/bare
    /// soil mix (Bonan 2008, table 9.1).
    pub ground_albedo: f32,
    /// Global multiplier applied to `delta_T` each tick. 1.0 = normal
    /// physics. 0.0 = freezes T (kill switch for thermal ablation
    /// tests). Replaces the former `relax_rate`, which mixed thermal
    /// inertia and coupling coefficient into one dimensionless
    /// parameter.
    pub thermal_coupling: f32,
    /// Expected average cloud cover [0,1] for calibrating
    /// `calibration_offset`. The mean annual IR back-radiation depends
    /// on the mean `cloud_cover`, which the engine cannot know ahead
    /// of time, so it is parameterized here. 0.5 = "average" temperate
    /// continental atmosphere (ISCCP climatology). Lower for desert
    /// climate, raise for oceanic climate.
    ///
    /// Issue #44: only used in computing the structural offset; the
    /// instantaneous back-radiation in `step_temperature` reads the
    /// cell's actual `cloud_water`, not this average.
    pub mean_cloud_cover_for_calibration: f32,
    /// Correction to the average insolation factor from slope aspect
    /// (sunny slope/shaded slope, #102). **Value derived from
    /// terrain**, not a user setting: computed once by
    /// `aspect_insolation_correction` in `Simulation::new` and
    /// injected here so `calibration_offset` recenters the map
    /// average on `base_temp` despite the localization of the flux.
    /// Flat world (or default params) ⇒ 0.0 ⇒ offset bit-identical to
    /// the historical behavior.
    pub aspect_correction: f32,
}

impl Default for TemperatureParams {
    fn default() -> Self {
        Self {
            base_temp: 15.0,
            lapse_rate: 6.5,
            water_cooling: 1.0,
            // 44.5: Drome, central France. Winters ~4°C, summers ~24°C,
            // day max ~15.4h (summer solstice), min ~8.6h (winter solstice).
            latitude_deg: 44.5,
            cloud_albedo_coef: 0.5,
            atmospheric_transmittance: 0.7,
            ground_albedo: 0.3,
            thermal_coupling: 1.0,
            mean_cloud_cover_for_calibration: 0.5,
            aspect_correction: 0.0,
        }
    }
}

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

#[must_use]
fn cached_annual_mean_insolation_factor(latitude_rad: f32) -> f32 {
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

/// Diffuse fraction of clear sky (dimensionless). Upstream relief
/// blocks the DIRECT beam but not the diffuse radiation from the whole
/// sky: a fully occluded cell still keeps this fraction (not an
/// arbitrary safeguard, it's the diffuse component, ~15-25% under
/// clear sky, Duffie & Beckman 2013). Coarse model (no sky-view
/// factor) to refine.
pub const DIFFUSE_SKY_FRACTION: f32 = 0.2;

/// Max number of steps in the shadow march (beyond this, potential
/// obstruction is negligible and the cost climbs). Step = 1 cell.
const ILLUM_MAX_STEPS: usize = 64;

/// Raymarch ablation switch (`HEXSIM_ILLUM_KO=1`), read once.
/// Perf measurement only, never active by default.
fn illum_ko() -> bool {
    use std::sync::OnceLock;
    static KO: OnceLock<bool> = OnceLock::new();
    *KO.get_or_init(|| std::env::var("HEXSIM_ILLUM_KO").is_ok_and(|v| v == "1"))
}
/// Elevation gain (m) of an upstream relief above the solar ray that
/// gives full occlusion; soft penumbra below that.
const ILLUM_FULL_M: f32 = 30.0;

/// Hex direction most aligned with the horizontal sun + ray slope
/// (`tan(elevation) = s_u / ‖s_horiz‖`). `None` if the sun is too
/// close to the zenith to define an azimuth (`‖s_horiz‖ < 1e-6`), no
/// march then, local cloud. Shared by the reference march and the
/// cached path so the 6-direction quantization stays identical in
/// both.
fn sun_march_geometry(beam: &SolarBeam) -> Option<(usize, f32)> {
    let horiz = (beam.s_e * beam.s_e + beam.s_n * beam.s_n).sqrt();
    if horiz < 1e-6 {
        return None;
    }
    let ray_slope = beam.s_u / horiz; // tan(elevation)
    // World: x=East, y=South = -North. Argmax of the dot product →
    // direction index (usize), no float→int cast.
    let (sx, sy) = (beam.s_e, -beam.s_n);
    let mut sun_dir = 0_usize;
    let mut best = f32::NEG_INFINITY;
    for k in 0..6 {
        let (dx, dy) = hex_direction_to_world(k);
        let dot = dx * sx + dy * sy;
        if dot > best {
            best = dot;
            sun_dir = k;
        }
    }
    Some((sun_dir, ray_slope))
}

/// Illumination pass (#102, final): for each cell, computes
/// `flux_factor = max(0, S⃗·N⃗) · occlusion · cloud_transmission`
/// (physics: absorbed flux = `beam · flux_factor`) and its display
/// counterpart `illumination` ∈ [0,1] (fraction of full sun for a
/// flat, clear, cloudless cell).
///
/// - **aspect**: `max(0, S⃗·N⃗)` against the local normal (sunny
///   slope/shaded slope).
/// - **relief occlusion**: march toward the sun on the elevation grid;
///   an upstream relief that exceeds the ray darkens it (toward the
///   diffuse floor).
/// - **cloud shadow**: samples `cloud_water` at the layer crossing
///   (distance `d = H/tan(elevation)`, shifted farther out when the
///   sun is low).
///
/// Marches in **integer hex coordinates** via `neighbor_indices_toric`:
/// native toric wrap (the world has no edge), zero float→cell
/// conversion. The solar azimuth is quantized to the nearest hex
/// direction (coarse v1, 6 orientations, to refine into a 2-direction
/// DDA). Night (`s_u ≤ 0`) → `flux_factor = 0`, `illumination = 1`
/// (darkness is handled by scene lighting, not albedo).
///
/// **Role since #65**: executable specification. Production goes
/// through [`compute_illumination_cached`] (same outputs, proven
/// bit-identical by `tests/phys_illum_cache_equiv.rs`); this naive
/// march remains the readable reference and the arbiter for the
/// equivalence micro-test. Any evolution of the illumination physics
/// happens HERE first, the cached path follows.
pub fn compute_illumination(
    grid: &HexGrid,
    beam: &SolarBeam,
    cloud_albedo_coef: f32,
    cloud_altitude_m: f32,
    flux_factor: &mut Vec<f32>,
    illumination: &mut Vec<f32>,
) {
    let cells = grid.cells_slice();
    let n = cells.len();
    flux_factor.clear();
    flux_factor.resize(n, 0.0);
    illumination.clear();
    illumination.resize(n, 1.0);
    if beam.s_u <= 0.0 {
        return; // sun below the horizon
    }
    let march = sun_march_geometry(beam);
    let has_azimuth = march.is_some();
    let (sun_dir, ray_slope) = march.unwrap_or((0, 0.0));

    let max_elev = cells
        .iter()
        .map(|c| c.elevation)
        .fold(f32::NEG_INFINITY, f32::max);
    let d_cloud = if has_azimuth {
        cloud_altitude_m / ray_slope
    } else {
        0.0
    };
    // Ablation switch (perf measurement): `HEXSIM_ILLUM_KO=1`
    // short-circuits the raymarch (relief occlusion + shifted cloud
    // shadow), illumination becomes aspect × LOCAL cloud. Temporary,
    // for the visual A/B.
    let ko_raymarch = illum_ko();
    for i in 0..n {
        let (ne, nn) = (cells[i].normal_east, cells[i].normal_north);
        let n_u = (1.0 - ne * ne - nn * nn).max(0.0).sqrt();
        let cos_inc = (beam.s_e * ne + beam.s_n * nn + beam.s_u * n_u).max(0.0);
        if cos_inc <= 0.0 {
            flux_factor[i] = 0.0;
            illumination[i] = 0.0; // slope facing away from the sun: no direct
            continue;
        }
        let cy = cells[i].elevation;
        let mut over = 0.0_f32; // max exceedance of the ray by an upstream relief
        let mut eff_cloud = cells[i].cloud_water; // default: local (zenith) cloud
        if has_azimuth && !ko_raymarch {
            let mut idx = i;
            let mut dist = 0.0_f32;
            // Sub-cell cloud shadow (offset < 1 cell, sun high) → keep
            // the LOCAL cloud (the cloud overhead darkens its own
            // cell); lateral projection only matters when the sun is
            // low (offset ≥ 1 cell).
            let mut cloud_sampled = d_cloud < CELL_SPACING_M;
            for _ in 0..ILLUM_MAX_STEPS {
                idx = grid.neighbor_indices_toric(idx)[sun_dir];
                dist += CELL_SPACING_M;
                let ray_h = cy + dist * ray_slope;
                over = over.max(cells[idx].elevation - ray_h);
                // Cell whose footprint contains the layer crossing
                // (the one closest to d_cloud, not the first exceeded).
                if !cloud_sampled && dist + 0.5 * CELL_SPACING_M >= d_cloud {
                    eff_cloud = cells[idx].cloud_water;
                    cloud_sampled = true;
                }
                if ray_h >= max_elev && cloud_sampled {
                    break; // nothing else can occlude, cloud sampled
                }
            }
        }
        let t = (over / ILLUM_FULL_M).min(1.0);
        let occlusion = 1.0 - (1.0 - DIFFUSE_SKY_FRACTION) * t; // ∈ [DIFFUSE, 1]
        let cover = eff_cloud.clamp(0.0, 1.0); // cloud_water normalized to 1 mm PW
        let transm = 1.0 - (cover * cloud_albedo_coef).min(0.95);
        let ff = cos_inc * occlusion * transm;
        flux_factor[i] = ff;
        illumination[i] = (ff / beam.s_u).clamp(0.0, 1.0);
    }
}

/// Levels of the doubling jump tables: `2^0 … 2^6` steps, enough to
/// compose any offset `≤ ILLUM_MAX_STEPS` (guaranteed by the const
/// assert).
const ILLUM_SHIFT_LEVELS: usize = 7;
const _: () = assert!(ILLUM_MAX_STEPS == 1 << (ILLUM_SHIFT_LEVELS - 1));

/// Upper bound (m) on the f32 rounding error of the reference march on
/// `elev − (cy + dist·slope)`: magnitudes ≤ ~25,000 m ⇒ error ≤ ~5 mm;
/// 5 cm = 10x margin. The precomputed tangents are biased by this
/// margin so that "no occlusion" / "full occlusion" decided in f64
/// imply the same result as the f32 march, cells within the margin
/// simply fall into the penumbra band and march as before.
const ILLUM_EXACT_MARGIN_M: f64 = 0.05;

/// Terrain precomputations for `compute_illumination_cached` (#65).
/// The shadow raymarch mixes two dynamics: **relief** occlusion
/// (function of terrain alone, immutable outside erosion) and
/// **cloud** shadow shifting (fresh every tick). This cache freezes
/// everything that depends only on terrain, per cell and per hex
/// direction:
///
/// - `s_clear`: solar tangent above which NONE of the 64 upstream
///   steps occludes (`over = 0`, march unnecessary);
/// - `s_full`: tangent below which occlusion is FULL
///   (`over ≥ ILLUM_FULL_M`, `t = 1` without marching);
/// - `dir_max`: max elevation of the 64 upstream steps, a tight stop
///   bound for the residual march (penumbra band between the two
///   tangents);
/// - `shift`: doubling jump tables (`2^j` toric steps) to sample the
///   cloud at the layer crossing in O(popcount) instead of marching
///   to it.
///
/// Measured (`tests/perf_illum_march_stats.rs`, r45 seed 42): 92 to
/// 99.9% of lit cells exit via one of the two tangents depending on
/// the hour, the march only remains for the penumbra band.
///
/// **Invalidation**: terrain only moves through erosion,
/// [`crate::simulation::Simulation`] calls [`IllumCache::mark_dirty`]
/// at the same spot as the surface normal recompute. `ensure` then
/// rebuilds on the next tick. The cache is tied to the last grid seen
/// by `ensure`.
#[derive(Debug, Clone)]
pub struct IllumCache {
    /// Number of grid cells at the last `rebuild`.
    len: usize,
    /// Terrain changed since the last `rebuild` (or never built).
    dirty: bool,
    /// `shift[d][j][i]` = cell `2^j` toric steps from `i` in direction `d`.
    shift: [[Vec<usize>; ILLUM_SHIFT_LEVELS]; 6],
    /// Tangent (f64, `+ILLUM_EXACT_MARGIN_M` bias) of "clear sky" by (dir, cell).
    s_clear: [Vec<f64>; 6],
    /// Tangent (f64, `−ILLUM_EXACT_MARGIN_M` bias) of "full shadow" by (dir, cell).
    s_full: [Vec<f64>; 6],
    /// Max elevation of the `ILLUM_MAX_STEPS` upstream steps by (dir, cell).
    dir_max: [Vec<f32>; 6],
    /// Flat elevations (contiguous copy, march without touching the bulky `CellProperties`).
    elev: Vec<f32>,
}

impl Default for IllumCache {
    fn default() -> Self {
        Self {
            len: 0,
            dirty: true,
            shift: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            s_clear: std::array::from_fn(|_| Vec::new()),
            s_full: std::array::from_fn(|_| Vec::new()),
            dir_max: std::array::from_fn(|_| Vec::new()),
            elev: Vec::new(),
        }
    }
}

impl IllumCache {
    /// Empty cache, rebuilt on the first [`IllumCache::ensure`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Signals that elevation changed (erosion): the next `ensure`
    /// rebuilds. O(1).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Rebuilds if needed (cache never built, terrain modified via
    /// [`IllumCache::mark_dirty`], or grid size changed). Rebuild
    /// cost: `6 dirs × ILLUM_MAX_STEPS × n` flat reads, on the order
    /// of a single tick of the old march, paid once per relief change
    /// (never in steady state without erosion).
    pub fn ensure(&mut self, grid: &HexGrid) {
        if !self.dirty && self.len == grid.len() {
            return;
        }
        self.rebuild(grid);
    }

    fn rebuild(&mut self, grid: &HexGrid) {
        let n = grid.len();
        self.len = n;
        self.dirty = false;
        self.elev.clear();
        self.elev
            .extend(grid.cells_slice().iter().map(|c| c.elevation));
        let spacing = f64::from(CELL_SPACING_M);
        for d in 0..6 {
            // Level 0: 1 toric step. Following levels by composition.
            let lvl0: Vec<usize> = (0..n).map(|i| grid.neighbor_indices_toric(i)[d]).collect();
            self.shift[d][0] = lvl0;
            for j in 1..ILLUM_SHIFT_LEVELS {
                let (built, rest) = self.shift[d].split_at_mut(j);
                let prev = &built[j - 1];
                let cur = &mut rest[0];
                cur.clear();
                cur.extend((0..n).map(|i| prev[prev[i]]));
            }
            // Tangents + upstream max: single march of ILLUM_MAX_STEPS steps.
            let step1 = &self.shift[d][0];
            let s_clear = &mut self.s_clear[d];
            let s_full = &mut self.s_full[d];
            let dir_max = &mut self.dir_max[d];
            s_clear.clear();
            s_full.clear();
            dir_max.clear();
            for i in 0..n {
                let cy = f64::from(self.elev[i]);
                let mut idx = i;
                let mut dist = 0.0_f64;
                let mut best_clear = f64::NEG_INFINITY;
                let mut best_full = f64::NEG_INFINITY;
                let mut dmax = f32::NEG_INFINITY;
                for _ in 0..ILLUM_MAX_STEPS {
                    idx = step1[idx];
                    dist += spacing;
                    let e = self.elev[idx];
                    dmax = dmax.max(e);
                    let b = f64::from(e) - cy;
                    best_clear = best_clear.max((b + ILLUM_EXACT_MARGIN_M) / dist);
                    best_full =
                        best_full.max((b - f64::from(ILLUM_FULL_M) - ILLUM_EXACT_MARGIN_M) / dist);
                }
                s_clear.push(best_clear);
                s_full.push(best_full);
                dir_max.push(dmax);
            }
        }
    }
}

/// Step (1..=`ILLUM_MAX_STEPS`) whose footprint contains the cloud
/// layer crossing, exactly replicates the f32 discovery of the
/// reference march (`dist + 0.5·step ≥ d_cloud`, `dist` accumulated in
/// f32). `0` = local cloud: sub-cell crossing (sun high) or layer
/// never reached within `ILLUM_MAX_STEPS` steps (sun very low).
fn cloud_sample_step(d_cloud: f32) -> usize {
    if d_cloud < CELL_SPACING_M {
        return 0;
    }
    let mut dist = 0.0_f32;
    for step in 1..=ILLUM_MAX_STEPS {
        dist += CELL_SPACING_M;
        if dist + 0.5 * CELL_SPACING_M >= d_cloud {
            return step;
        }
    }
    0
}

/// Production illumination pass (#65): same outputs as
/// [`compute_illumination`] (the equivalence micro-test compares them
/// bit for bit), separating the two dynamics the reference march
/// conflates:
///
/// - **relief** (immutable outside erosion): decided by the
///   precomputed tangents of the [`IllumCache`], the march only
///   remains for the penumbra band (`s_full < slope < s_clear`, a few
///   % of cells), on flat arrays with a stop at `dir_max`;
/// - **cloud** (fresh every tick): sampled in O(popcount) via the
///   jump tables, at the same step as the reference march.
///
/// # Panics
/// If the cache doesn't match the grid (`ensure` forgotten after a
/// relief change), a stale cache would silently produce wrong
/// shadows, we prefer to fail loudly.
pub fn compute_illumination_cached(
    grid: &HexGrid,
    beam: &SolarBeam,
    cloud_albedo_coef: f32,
    cloud_altitude_m: f32,
    cache: &IllumCache,
    flux_factor: &mut Vec<f32>,
    illumination: &mut Vec<f32>,
) {
    let cells = grid.cells_slice();
    let n = cells.len();
    assert!(
        !cache.dirty && cache.len == n,
        "IllumCache stale (dirty={}, len={} vs grid {n}): missing ensure() call",
        cache.dirty,
        cache.len
    );
    flux_factor.clear();
    flux_factor.resize(n, 0.0);
    illumination.clear();
    illumination.resize(n, 1.0);
    if beam.s_u <= 0.0 {
        return; // sun below the horizon
    }
    let march = sun_march_geometry(beam).filter(|_| !illum_ko());
    let kstar = match march {
        Some((_, ray_slope)) => cloud_sample_step(cloud_altitude_m / ray_slope),
        None => 0,
    };
    for i in 0..n {
        let (ne, nn) = (cells[i].normal_east, cells[i].normal_north);
        let n_u = (1.0 - ne * ne - nn * nn).max(0.0).sqrt();
        let cos_inc = (beam.s_e * ne + beam.s_n * nn + beam.s_u * n_u).max(0.0);
        if cos_inc <= 0.0 {
            flux_factor[i] = 0.0;
            illumination[i] = 0.0; // slope facing away from the sun: no direct
            continue;
        }
        let mut t = 0.0_f32; // normalized occlusion = (over / ILLUM_FULL_M).min(1)
        let mut eff_cloud = cells[i].cloud_water; // default: local (zenith) cloud
        if let Some((sun_dir, ray_slope)) = march {
            // Cloud shadow: cell at the layer crossing, via 2^j jumps.
            if kstar > 0 {
                let mut idx = i;
                let mut bits = kstar;
                let mut level = 0;
                while bits != 0 {
                    if bits & 1 == 1 {
                        idx = cache.shift[sun_dir][level][idx];
                    }
                    bits >>= 1;
                    level += 1;
                }
                eff_cloud = cells[idx].cloud_water;
            }
            // Relief occlusion: precomputed tangents, march only in penumbra.
            let slope_w = f64::from(ray_slope);
            if slope_w >= cache.s_clear[sun_dir][i] {
                // no upstream step occludes: over = 0, t = 0 (majority path)
            } else if slope_w <= cache.s_full[sun_dir][i] {
                t = 1.0; // full occlusion guaranteed: the march would give min(1)
            } else {
                let cy = cache.elev[i];
                let dmax = cache.dir_max[sun_dir][i];
                let step1 = &cache.shift[sun_dir][0];
                let mut over = 0.0_f32;
                let mut idx = i;
                let mut dist = 0.0_f32;
                for _ in 0..ILLUM_MAX_STEPS {
                    idx = step1[idx];
                    dist += CELL_SPACING_M;
                    let ray_h = cy + dist * ray_slope;
                    over = over.max(cache.elev[idx] - ray_h);
                    if ray_h >= dmax {
                        break; // nothing tall enough left in this direction
                    }
                }
                t = (over / ILLUM_FULL_M).min(1.0);
            }
        }
        let occlusion = 1.0 - (1.0 - DIFFUSE_SKY_FRACTION) * t; // ∈ [DIFFUSE, 1]
        let cover = eff_cloud.clamp(0.0, 1.0); // cloud_water normalized to 1 mm PW
        let transm = 1.0 - (cover * cloud_albedo_coef).min(0.95);
        let ff = cos_inc * occlusion * transm;
        flux_factor[i] = ff;
        illumination[i] = (ff / beam.s_u).clamp(0.0, 1.0);
    }
}

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
fn calibration_offset(params: &TemperatureParams, lat_rad: f32) -> f32 {
    let mean_factor = cached_annual_mean_insolation_factor(lat_rad) + params.aspect_correction;
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
pub fn step_temperature(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &TemperatureParams,
    hour_tick: u64,
    flux_factor: &[f32],
    snow: &crate::snow::SnowParams,
) {
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

    // Purely local radiative balance (no neighbor) → parallelizable per cell.
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
            let net_radiative = solar_in + back_rad
                - STEFAN_BOLTZMANN_AT_T0
                - LIN_RADIATIVE_COEF * (cell.temperature - t_ref);

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
            hour_tick,
            &ff,
            &crate::snow::SnowParams::default(),
        );
    }

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
    fn illumination_relief_occludes_toward_sun() {
        // Sun in the east, 45° elevation. A cell with a tall wall to the EAST
        // (toward the sun) receives less than the same cell without a wall
        // (occlusion), but keeps the diffuse floor (never absolute zero).
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let beam = SolarBeam {
            s_e: s,
            s_n: 0.0,
            s_u: s,
            beam: 1000.0,
        };
        let center = HexCoord::new(0, 0);
        let east = HexCoord::new(1, 0);

        let mut grid = HexGrid::from_radius(2);
        if let Some(c) = grid.get_mut(east) {
            c.elevation = 4000.0;
        }
        let ci = grid.index_of(center).unwrap();
        let (mut ff, mut il) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, 1500.0, &mut ff, &mut il);
        let occluded = ff[ci];

        if let Some(c) = grid.get_mut(east) {
            c.elevation = 0.0; // wall removed, same world
        }
        let (mut ff2, mut il2) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, 1500.0, &mut ff2, &mut il2);

        assert!(
            occluded < ff2[ci],
            "occluded {occluded} should be < clear {}",
            ff2[ci]
        );
        assert!(occluded > 0.0, "diffuse floor preserved, got {occluded}");
    }

    #[test]
    fn illumination_cloud_dims_along_ray() {
        // Sun east/45°: d_cloud = cloud_altitude_m/tan45 = cloud_altitude_m,
        // so taking cloud_altitude_m = CELL_SPACING_M, the nearest cell is
        // the 1st east neighbor regardless of the engine's resolution. A
        // cloud THERE (not above the center) dims the cell: lateral cloud
        // shadow.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let beam = SolarBeam {
            s_e: s,
            s_n: 0.0,
            s_u: s,
            beam: 1000.0,
        };
        let center = HexCoord::new(0, 0);
        let cloud_cell = HexCoord::new(1, 0);

        let mut grid = HexGrid::from_radius(3);
        if let Some(c) = grid.get_mut(cloud_cell) {
            c.cloud_water = 1.0;
        }
        let ci = grid.index_of(center).unwrap();
        let (mut ff, mut il) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, CELL_SPACING_M, &mut ff, &mut il);
        let shaded = ff[ci];

        if let Some(c) = grid.get_mut(cloud_cell) {
            c.cloud_water = 0.0;
        }
        let (mut ff2, mut il2) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, CELL_SPACING_M, &mut ff2, &mut il2);

        assert!(
            shaded < ff2[ci],
            "lateral cloud should dim: {shaded} vs {}",
            ff2[ci]
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
        assert!(
            wet_mean < dry_mean - 1.0,
            "annual mean lake must be < plain by at least 1°C: wet={wet_mean:.2} dry={dry_mean:.2}"
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
