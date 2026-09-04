//! Temperature phenomenon: SI energy balance for the ground/water surface.
//!
//! Split (#52 pattern, reused here) out of a former single 2200+ line
//! `temperature.rs` into one sub-module per concern, mirroring how
//! `atmosphere/` is organized (`mod.rs` re-exports, sub-modules implement):
//!
//! - `solar`: solar astronomy — declination, day length, solar elevation,
//!   the sun direction vector (`SolarBeam`) and the clear-sky flux for a
//!   given surface orientation. Pure ephemeris, no grid state.
//! - `illumination`: per-cell shadow raymarch (relief occlusion + cloud
//!   shadow) that turns a `SolarBeam` into a `flux_factor` per cell, plus
//!   its terrain cache (`IllumCache`). Consumed once per tick by the
//!   simulation loop to feed [`crate::temperature::step_temperature`].
//! - `balance`: the SI radiative balance itself
//!   ([`crate::temperature::step_temperature`]/[`crate::temperature::TemperatureForcing`]) and the helpers it
//!   shares with neighboring phenomena (`local_t_ref`, `cloud_cover_fraction`).
//!
//! This top-level module owns what all three share: the physical constants
//! and [`crate::temperature::TemperatureParams`]. Everything below is re-exported here via
//! `pub use` so `hexsim_core::temperature::X` stays the stable public path
//! regardless of which sub-module implements `X` — the split moves code,
//! it does not change the crate's public surface.

use serde::{Deserialize, Serialize};

use crate::snow::{AIR_DENSITY_KG_PER_M3, AIR_SPECIFIC_HEAT_J_PER_KG_K};

mod balance;
mod illumination;
mod solar;

pub use balance::{
    TemperatureForcing, absorbed_solar_flux, cloud_cover_fraction, local_t_ref, step_temperature,
};
pub use illumination::{
    DIFFUSE_SKY_FRACTION, IllumCache, compute_illumination, compute_illumination_cached,
    terrain_annual_mean_insolation_factor,
};
// Crate-internal only: consumed by `ablation::Ablation::defaults` to build
// the compiled-in default without duplicating the constant.
pub(crate) use illumination::ILLUM_KO_DEFAULT;
// Crate-internal only: consumed by `balance::calibration_offset`, never
// part of the public API.
pub(crate) use solar::cached_annual_mean_insolation_factor;
pub use solar::{
    SolarBeam, annual_mean_insolation_factor, aspect_insolation_correction,
    clear_sky_flux_for_normal, clear_sky_solar_flux, compute_surface_normals,
    daily_insolation_factor, day_length_hours, hour_angle_sunrise_rad, solar_beam_at_tick,
    solar_declination_rad, solar_elevation_at_hour,
};

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

/// Bulk transfer coefficient for sensible heat `C_H` (dimensionless,
/// ratio of the turbulent heat flux to `ρ·c_p·U·ΔT`). 2e-3 = moderately
/// rough vegetated land in near-neutral stratification (Garratt J.
/// 1992, *The Atmospheric Boundary Layer*, table 4.1: 1-3e-3; Stull R.
/// 1988, *An Introduction to Boundary Layer Meteorology*, §7.4). Same
/// value as the snowpack melt balance (`SnowParams::sensible_exchange_coef`),
/// the two exchanges are the same physics on the same air.
pub const BULK_HEAT_TRANSFER_COEF: f32 = 2.0e-3;

/// Reference 10 m wind speed (m/s) driving the turbulent exchange:
/// climatological mean over inland temperate terrain (Drôme stations,
/// 2-3 m/s). A single map-wide value keeps the exchange energy-neutral
/// (see `SENSIBLE_EXCHANGE_COEF`); a per-cell wind is a later refinement.
pub const MIXING_WIND_REF_MS: f32 = 2.5;

/// Sensible heat exchange coefficient between the surface and the
/// mixed boundary-layer air (W/(m²·K)): `H = ρ·c_p·C_H·U ≈ 6`. This is
/// the bulk aerodynamic formula `Q_H = ρ·c_p·C_H·U·(T_s − T_air)`
/// (Garratt 1992 §4.3), the term the balance lacked: without it every
/// local radiative anomaly (aspect, relief shadow, cloud) maps into
/// surface temperature through the radiative damping alone
/// (`LIN_RADIATIVE_COEF` ≈ 5.4 W/(m²·K)), so a 40° south face at
/// 1000 m sat 8-11 °C above its band and a shaded winter plain 5 °C
/// below it (JOURNAL 2026-09-02, bisected to the 130 m spacing d6be105,
/// relief 8x steeper than at calibration). The exchange partner is the
/// horizontally mixed air of the terrarium at the cell's elevation,
/// `T̄ + Γ·(z̄ − z)/1000`: the same mixing that the atmosphere assumes
/// for its upper layer (`atmosphere::upper_air_temperature`). Summed
/// over the map the exchange is exactly zero (`Σ(T_i − T_air(z_i)) =
/// 0`), a pure redistribution: the calibration of `base_temp` through
/// `calibration_offset` is untouched.
pub const SENSIBLE_EXCHANGE_COEF: f32 = AIR_DENSITY_KG_PER_M3
    * AIR_SPECIFIC_HEAT_J_PER_KG_K
    * BULK_HEAT_TRANSFER_COEF
    * MIXING_WIND_REF_MS;

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
    /// Average atmospheric transmittance (dimensionless, `[0,1]`).
    /// Effective Beer-Lambert: fraction of the solar flux that
    /// reaches the ground under clear sky. ~0.7 = standard atmosphere
    /// with aerosols (Duffie & Beckman 2013, eq 2.11.1, continental
    /// climate tables).
    pub atmospheric_transmittance: f32,
    /// Average ground albedo (dimensionless, `[0,1]`). Fraction of the
    /// flux reflected back to space. 0.3 = typical grass/forest/bare
    /// soil mix (Bonan 2008, table 9.1).
    pub ground_albedo: f32,
    /// Global multiplier applied to `delta_T` each tick. 1.0 = normal
    /// physics. 0.0 = freezes T (kill switch for thermal ablation
    /// tests). Replaces the former `relax_rate`, which mixed thermal
    /// inertia and coupling coefficient into one dimensionless
    /// parameter.
    pub thermal_coupling: f32,
    /// Expected average cloud cover `[0,1]` for calibrating
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
    /// Annual mean ratio of the REAL terrain's illumination to the flat
    /// horizontal beam (relief occlusion + diffuse sky, clouds
    /// ignored), dimensionless in `(0, ~1]`. **Value derived from
    /// terrain**, not a user setting: computed once by
    /// [`terrain_annual_mean_insolation_factor`] in `Simulation::new`
    /// (and again wherever erosion invalidates the [`IllumCache`], since
    /// the relief changed) and injected here so `calibration_offset`
    /// targets `base_temp` on the terrain the map actually has, not on
    /// an assumed flat one. At `CELL_SPACING_M = 130`, mean slope ≈29°
    /// (r30 seed 42), only ≈0.856 of the flat beam gets through
    /// (`tests/diag_illumination_budget.rs`), so a flat-world
    /// calibration ran the whole map ≈4 K below `base_temp` (JOURNAL
    /// 2026-09-02/03). Flat world (or default params) ⇒ 1.0 ⇒ offset
    /// bit-identical to the historical flat-terrain behavior.
    #[serde(default = "default_terrain_insolation_factor")]
    pub terrain_insolation_factor: f32,
}

/// `serde(default)` value for [`TemperatureParams::terrain_insolation_factor`]:
/// 1.0 = flat terrain, so a checkpoint or params file predating this
/// field restores the historical flat-calibration behavior exactly.
fn default_terrain_insolation_factor() -> f32 {
    1.0
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
            terrain_insolation_factor: default_terrain_insolation_factor(),
        }
    }
}
