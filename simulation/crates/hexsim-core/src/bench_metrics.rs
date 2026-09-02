// Descriptive stats module (means, variances, percentiles, ratios).
// The u32/u64/usize -> f32 numeric casts are expected: we aggregate a
// small number of observations (a few thousand at most) with f32
// precision, which is largely sufficient to characterize a climate.
// Not a bug, this is standard practice for observational metrics.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

//! Benchmark / optimization harness: lets you run the simulation with a
//! custom parameter set (partial override of the defaults) and extract
//! a set of quantifiable climatological metrics from it.
//!
//! Usage: consumed by the `hexsim-bench` binary, itself driven by a
//! Python script (`scripts/optim/random_search.py`) that explores the
//! parameter space.
//!
//! Design:
//! - **No `Deserialize` added to the prod structs**: we use mirror
//!   structs `*ParamsOverride` with fields as `Option<T>`. This avoids
//!   any risk of silently resetting a parameter forgotten in the input
//!   JSON, and leaves the prod structs unchanged.
//! - **`deny_unknown_fields`** everywhere: a typo in the JSON crashes
//!   immediately instead of being silently ignored.
//! - **Temporary duplication of metric calculations** already present
//!   in the scale tests. Optional refactor later (phase 2).

use serde::{Deserialize, Serialize};

use crate::atmosphere::AtmosphereParams;
use crate::dynamics::SynopticParams;
use crate::grid::HexGrid;
use crate::groundwater::GroundwaterParams;
use crate::hydro::HydroParams;
use crate::simulation::Simulation;
use crate::snow::SnowParams;
use crate::temperature::TemperatureParams;
use crate::terrain::{TerrainParams, generate_terrain};
use crate::wind::WindParams;

// ====================================================================
// Overrides (mirror structs, all fields as Option<T>)
// ====================================================================

macro_rules! apply_opt {
    ($override:expr, $base:expr, $( $field:ident ),+ $(,)?) => {
        $(
            if let Some(v) = $override.$field {
                $base.$field = v;
            }
        )+
    };
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AtmosphereParamsOverride {
    pub transpiration_coef: Option<f32>,
    pub sublimation_rate: Option<f32>,
    pub uplift_rate: Option<f32>,
    pub uplift_thermal_coef: Option<f32>,
    pub upper_layer_altitude_m: Option<f32>,
    pub global_precip_gate: Option<f32>,
    pub initial_humidity_floor: Option<f32>,
    pub orographic_lift_coef: Option<f32>,
    pub condensation_rate: Option<f32>,
    pub cloud_evap_hr_threshold: Option<f32>,
    pub cloud_evap_rate: Option<f32>,
    pub cloud_diffusion_rate: Option<f32>,
    pub cloud_advection_rate: Option<f32>,
    pub kk2000_droplet_count: Option<f32>,
    pub precip_neighbor_share: Option<f32>,
    pub max_precip_per_tick: Option<f32>,
    pub fog_condensation_threshold: Option<f32>,
    pub fog_condensation_rate: Option<f32>,
    pub updraft_ref_ms: Option<f32>,
    pub updraft_floor: Option<f32>,
    pub precip_crit_mm: Option<f32>,
}

impl AtmosphereParamsOverride {
    pub fn apply(&self, base: &mut AtmosphereParams) {
        apply_opt!(
            self,
            base,
            transpiration_coef,
            sublimation_rate,
            uplift_rate,
            uplift_thermal_coef,
            upper_layer_altitude_m,
            global_precip_gate,
            initial_humidity_floor,
            orographic_lift_coef,
            condensation_rate,
            cloud_evap_hr_threshold,
            cloud_evap_rate,
            cloud_diffusion_rate,
            cloud_advection_rate,
            kk2000_droplet_count,
            precip_neighbor_share,
            max_precip_per_tick,
            fog_condensation_threshold,
            fog_condensation_rate,
            updraft_ref_ms,
            updraft_floor,
            precip_crit_mm,
        );
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct HydroParamsOverride {
    pub flow_rate: Option<f32>,
    pub slope_full_mobility: Option<f32>,
    pub flow_concentration: Option<f32>,
}

impl HydroParamsOverride {
    pub fn apply(&self, base: &mut HydroParams) {
        apply_opt!(
            self,
            base,
            flow_rate,
            slope_full_mobility,
            flow_concentration
        );
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemperatureParamsOverride {
    pub base_temp: Option<f32>,
    pub lapse_rate: Option<f32>,
    pub water_cooling: Option<f32>,
    pub thermal_coupling: Option<f32>,
    pub latitude_deg: Option<f32>,
    pub cloud_albedo_coef: Option<f32>,
    pub atmospheric_transmittance: Option<f32>,
    pub ground_albedo: Option<f32>,
}

impl TemperatureParamsOverride {
    pub fn apply(&self, base: &mut TemperatureParams) {
        apply_opt!(
            self,
            base,
            base_temp,
            lapse_rate,
            water_cooling,
            thermal_coupling,
            latitude_deg,
            cloud_albedo_coef,
            atmospheric_transmittance,
            ground_albedo,
        );
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WindParamsOverride {
    pub noise_direction_amplitude: Option<f32>,
    pub noise_strength_amplitude: Option<f32>,
    pub noise_time_scale: Option<f32>,
    pub thermal_strength: Option<f32>,
    pub terrain_deflection: Option<f32>,
    pub terrain_speed_factor: Option<f32>,
    pub smoothing_passes: Option<u8>,
    pub humidity_advection_rate: Option<f32>,
    pub temperature_advection_rate: Option<f32>,
    pub wind_upper_rotation_deg: Option<f32>,
    pub wind_upper_speed_ratio: Option<f32>,
}

impl WindParamsOverride {
    pub fn apply(&self, base: &mut WindParams) {
        apply_opt!(
            self,
            base,
            noise_direction_amplitude,
            noise_strength_amplitude,
            noise_time_scale,
            thermal_strength,
            terrain_deflection,
            terrain_speed_factor,
            smoothing_passes,
            humidity_advection_rate,
            temperature_advection_rate,
            wind_upper_rotation_deg,
            wind_upper_speed_ratio,
        );
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroundwaterParamsOverride {
    pub infiltration_rate: Option<f32>,
    pub diffusion_rate: Option<f32>,
    pub max_capacity: Option<f32>,
}

impl GroundwaterParamsOverride {
    pub fn apply(&self, base: &mut GroundwaterParams) {
        apply_opt!(self, base, infiltration_rate, diffusion_rate, max_capacity);
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SnowParamsOverride {
    pub snow_albedo_dry: Option<f32>,
    pub snow_albedo_melt: Option<f32>,
    pub snow_emissivity: Option<f32>,
    pub sensible_exchange_coef: Option<f32>,
    pub free_convection_wind_ms: Option<f32>,
    pub snow_masking_half_mm: Option<f32>,
    pub freeze_threshold: Option<f32>,
}

impl SnowParamsOverride {
    pub fn apply(&self, base: &mut SnowParams) {
        apply_opt!(
            self,
            base,
            snow_albedo_dry,
            snow_albedo_melt,
            snow_emissivity,
            sensible_exchange_coef,
            free_convection_wind_ms,
            snow_masking_half_mm,
            freeze_threshold,
        );
    }
}

/// Override for the synoptic dynamics (Phase 4 of the synoptic-dynamics
/// design: climate calibration once the pressure-driven wind is coupled
/// to precipitation).
///
/// `enabled` overrides the runtime param `synoptic.enabled` (ON by default,
/// hardcoded since #108): when present, it forces activation (or not)
/// independently of the default, which makes the A/B bench (baseline vs
/// synoptic) reproducible.
///
/// `deformation_radius_cells` is NOT exposed: it fixes the scale thickness
/// `H = h0` set at init of the synoptic state, not hot-tunable (cf.
/// `simulation::update_param`). It stays at the Phase 0 spike default (~10).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SynopticParamsOverride {
    pub enabled: Option<bool>,
    pub mean_flow_ms: Option<f32>,
    pub thermal_anomaly_days: Option<f32>,
    pub thermal_coupling: Option<f32>,
    pub viscosity: Option<f32>,
    pub friction_days: Option<f32>,
    pub relax_days: Option<f32>,
}

/// Root of the input JSON. All groups are optional (default = empty
/// override = all params at the sim defaults).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BenchParams {
    pub atmosphere: AtmosphereParamsOverride,
    pub hydro: HydroParamsOverride,
    pub temperature: TemperatureParamsOverride,
    pub wind: WindParamsOverride,
    pub groundwater: GroundwaterParamsOverride,
    pub snow: SnowParamsOverride,
    pub synoptic: SynopticParamsOverride,
}

/// Dump of the effective parameters after applying the overrides. Serialized
/// into the JSON output so that the user (or a script) can recover the
/// exact config that produced the result without depending on the default.
///
/// No `Debug` because the prod structs don't derive it and we refuse to
/// modify them for a strictly bench-only need.
#[derive(Clone, Serialize)]
pub struct EffectiveParams {
    pub atmosphere: AtmosphereParams,
    pub hydro: HydroParams,
    pub temperature: TemperatureParams,
    pub wind: WindParams,
    pub groundwater: GroundwaterParams,
    pub snow: SnowParams,
    pub synoptic: SynopticParams,
    pub synoptic_enabled: bool,
}

// ====================================================================
// Build helper, equivalent to common::build_prod_sim but parameterizable
// ====================================================================

/// Builds a sim by applying the overrides to the default params.
/// Returns the sim and the effective params (to include in the JSON output).
///
/// # Panics
///
/// Panics if an expected synoptic parameter key becomes unknown to
/// `Simulation::update_param` (a field rename not reflected here); this is
/// a static programming error, not a user input case.
#[must_use]
pub fn build_bench_sim(
    seed: u32,
    radius: i32,
    overrides: &BenchParams,
) -> (Simulation, EffectiveParams) {
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed,
            ..TerrainParams::default()
        },
    );

    let mut atmosphere = AtmosphereParams::default();
    overrides.atmosphere.apply(&mut atmosphere);
    let mut hydro = HydroParams::default();
    overrides.hydro.apply(&mut hydro);
    let mut temperature = TemperatureParams::default();
    overrides.temperature.apply(&mut temperature);
    let mut wind = WindParams {
        seed,
        ..WindParams::default()
    };
    overrides.wind.apply(&mut wind);
    let mut groundwater = GroundwaterParams::default();
    overrides.groundwater.apply(&mut groundwater);
    let mut snow = SnowParams::default();
    overrides.snow.apply(&mut snow);

    // Synoptic (Phase 4): seed + latitude inherited from the world, as
    // `Simulation::new` does. We build the effective config here to report
    // it, then apply it to the sim via `update_param` after construction
    // (the synoptic fields are safe to hot-tune; the anomaly mode's `t_ref`
    // seeds itself on the first step).
    let mut synoptic = SynopticParams {
        seed,
        latitude_deg: temperature.latitude_deg,
        ..SynopticParams::default()
    };
    let so = &overrides.synoptic;
    apply_opt!(
        so,
        synoptic,
        mean_flow_ms,
        thermal_anomaly_days,
        thermal_coupling,
        viscosity,
        friction_days,
        relax_days,
    );
    // Activation: the explicit override takes priority; otherwise we respect
    // the default hardcoded in `Simulation::new` (ON, #108).
    let synoptic_enabled_override = so.enabled;

    let effective = EffectiveParams {
        atmosphere: atmosphere.clone(),
        hydro: hydro.clone(),
        temperature: temperature.clone(),
        wind: wind.clone(),
        groundwater: groundwater.clone(),
        snow: snow.clone(),
        synoptic: synoptic.clone(),
        synoptic_enabled: false, // filled in below from the sim
    };

    let mut sim = Simulation::new(
        grid,
        hydro,
        atmosphere,
        groundwater,
        snow,
        temperature,
        wind,
    );

    // Applies the hot-tunable synoptic overrides (no-op if all None).
    // `update_param` returns `true` if the key exists: we assert on it so we
    // don't silently mask a field rename (the keys are static).
    let mut set = |key: &str, v: f32| {
        assert!(sim.update_param(key, v), "unknown synoptic key: {key}");
    };
    if let Some(v) = so.mean_flow_ms {
        set("synoptic.mean_flow_ms", v);
    }
    if let Some(v) = so.thermal_anomaly_days {
        set("synoptic.thermal_anomaly_days", v);
    }
    if let Some(v) = so.thermal_coupling {
        set("synoptic.thermal_coupling", v);
    }
    if let Some(v) = so.viscosity {
        set("synoptic.viscosity", v);
    }
    if let Some(v) = so.friction_days {
        set("synoptic.friction_days", v);
    }
    if let Some(v) = so.relax_days {
        set("synoptic.relax_days", v);
    }
    if let Some(on) = synoptic_enabled_override {
        set("synoptic.enabled", if on { 1.0 } else { 0.0 });
    }
    // `set` borrows `sim` mutably; its last use above releases the borrow
    // (NLL), so the shared access `synoptic_enabled()` right after is fine.

    let effective = EffectiveParams {
        synoptic_enabled: sim.synoptic_enabled(),
        ..effective
    };
    (sim, effective)
}

// ====================================================================
// Metrics
// ====================================================================

/// Status of a run. Distinct from the HTTP status; `"ok"` means "the sim
/// ran to completion and the metrics are valid".
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Ok,
    NanInf,
}

/// Signature of a rain "hotspot" cell: its coordinate, its elevation, and
/// its water level at the end of the run. Lets us diagnose whether
/// persistent rain is falling on a lake (`water_level` > 0) or on relief
/// (elevation > 1000m).
#[derive(Debug, Clone, Serialize)]
pub struct RainHotspot {
    pub q: i32,
    pub r: i32,
    pub rain_days: f32,
    pub elevation: f32,
    pub water_level: f32,
}

/// Result of a bench run, serializable to JSON.
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    /// |final - initial| / initial, on total `water_budget`.
    pub water_drift_pct: f32,
    /// Max over all cells of the number of rainy ticks (rain > 1e-4).
    /// Expressed in days/year (normalized by `mesure_ticks` then x 365).
    pub rain_days_max_per_cell: f32,
    /// Top 3 rainiest cells. Spatial diagnostic: lets us know where the
    /// persistent rain falls (lake, relief, plain...).
    pub rain_hotspots: Vec<RainHotspot>,
    /// Median number of **rain** days/year per elevation band
    /// (`rain > 1e-4`, snow is NOT counted; it was inflating the cold
    /// high-elevation bands and simulating a fake orographic drizzle).
    /// Order: [<300m, 300-800m, 800-1500m, >1500m].
    /// An empty band is reported as NaN.
    pub rain_days_median_by_altitude: [f32; 4],
    /// Average precip/tick in summer (days 146-237) divided by the average
    /// in winter (days 0-54 and 335-364). 1.0 = no seasonality.
    pub ratio_precip_summer_winter: f32,
    /// Median over a sample of plain/hill cells (50-600m) of the longest
    /// consecutive dry streak, in days.
    pub dry_streak_median_per_cell_days: f32,
    /// Average rainfall over plains/hills (50-600m), in mm/day; temporal
    /// average of rain (snow excluded) over the band. Phase 4 point of
    /// attention: plains falling to 0.01-0.02 mm/day under the synoptic
    /// regime threaten vegetation (issue #69/#86). Target: keep the band
    /// watered (> ~0.05 mm/day) while still keeping dry days.
    pub plains_precip_mm_per_day: f32,
    /// Median fraction (over plain/hill cells 50-600m) of measured days
    /// receiving a rain **gust** (> 1 mm/day). The temporal
    /// re-concentration targeted by the synoptic regime: few days, but
    /// days that are clearly rainy rather than a continuous drizzle.
    /// Phase 4 point of attention: measured at 1-2% at the end of Phase 3.
    pub gust_days_frac_median: f32,
    /// **Mean** fraction (50-600m band) of gust days (> 1 mm/day).
    /// Unlike the median (~0 because the event is rare), the mean is
    /// sensitive to the re-concentration: this is the gust calibration
    /// metric.
    pub gust_days_frac_mean: f32,
    /// Median (50-600m band) of the **maximum daily rain** seen over the
    /// whole window, in mm/day. "When it rains hardest, how much?", a
    /// direct intensity signal: a uniform drizzle plateaus low, a
    /// re-concentrated weather pattern makes it climb.
    pub plains_max_daily_rain_median_mm: f32,
    /// Days in the measurement window where NO cell receives any
    /// precipitation, neither rain NOR snow (fully dry global phase).
    /// Warning: confounded by alpine snow: the cold high-elevation band
    /// snows nearly year-round, so this metric stays pinned at 0 even
    /// when the plains experience long high-pressure phases. For the
    /// "high-pressure phase" question (#86), read
    /// `fully_rain_free_days_total` instead, which ignores snow.
    pub fully_dry_days_total: f32,
    /// Days in the measurement window where NO cell receives **rain**
    /// (alpine snow is allowed). This is the relevant indicator of a
    /// global high-pressure phase: high-elevation snow is a near-permanent
    /// process that must not mask the drying out of the lowlands.
    pub fully_rain_free_days_total: f32,
    /// `min(snow_total)` observed in summer / `max(snow_total)` observed
    /// in winter. Low (< 0.5) = clear seasonal cycle.
    pub snow_ratio_winter_max_summer_min: f32,
    /// Fraction of cells >1000m with RH > 0.6 for at least 5% of the time.
    pub cloud_cover_mountain_pct: f32,
    /// Temp-vs-elevation slope measured on the final grid (deg C/km,
    /// linear regression). Compare to `params.temperature.lapse_rate` to
    /// detect thermal inversions or the gradient being flattened by
    /// advection.
    pub effective_lapse_rate_c_per_km: f32,
    /// Perf: average time per measurement tick (excluding warmup).
    pub ms_per_tick: f32,
    /// Number of cells in the grid (sanity check).
    pub cell_count: usize,
}

/// Tick-by-tick observation accumulator. Doesn't use `climate_history` so
/// as not to be coupled to its API.
pub struct MetricsAccumulator {
    cell_count: usize,
    measure_ticks: u64,
    // Cell order: coords_slice() of the grid. The indexing stays stable
    // over the run's duration (grid not modified).
    elevations: Vec<f32>,
    // Per-cell (index = cell_index)
    rain_ticks: Vec<u32>,
    // Days (ticks) with a gust > 1 mm/day, temporal re-concentration.
    gust_ticks: Vec<u32>,
    // Cumulative rain (mm) per cell, averaged -> mm/day for the plain band.
    rain_total_mm: Vec<f32>,
    // Maximum daily rain (mm) seen per cell, gust intensity.
    max_daily_rain: Vec<f32>,
    cloudy_ticks: Vec<u32>, // RH > 0.6 at the tick
    current_dry_streak: Vec<u32>,
    max_dry_streak: Vec<u32>,
    // Per-tick globals (len = ticks observed after warmup)
    snow_total_per_tick: Vec<f32>,
    precip_per_tick: Vec<f32>,
    // Days where NO cell receives any precipitation (neither rain NOR snow).
    fully_dry_ticks: u64,
    // Days where NO cell receives rain (alpine snow tolerated), the true
    // indicator of a high-pressure phase, de-confounded from snow.
    fully_rain_free_ticks: u64,
    // Atmo config needed for the RH calculation (Tetens, Phase 6)
    upper_layer_altitude_m: f32,
    lapse_rate: f32,
    // Conservation
    initial_water: f32,
    // Detection of a NaN/Inf in the main stocks
    nan_inf_seen: bool,
    // Tick index (0..measure_ticks)
    ticks_observed: u64,
}

impl MetricsAccumulator {
    /// Creates the accumulator from the sim after warmup.
    /// `initial_water` is fixed to the value of `water_budget` at this
    /// exact moment; conservation is measured relative to that.
    #[must_use]
    pub fn start(sim: &Simulation, effective: &EffectiveParams, measure_ticks: u64) -> Self {
        let grid = sim.grid();
        let cell_count = grid.len();
        let elevations: Vec<f32> = grid.iter().map(|(_, c)| c.elevation).collect();
        Self {
            cell_count,
            measure_ticks,
            elevations,
            rain_ticks: vec![0; cell_count],
            gust_ticks: vec![0; cell_count],
            rain_total_mm: vec![0.0; cell_count],
            max_daily_rain: vec![0.0; cell_count],
            cloudy_ticks: vec![0; cell_count],
            current_dry_streak: vec![0; cell_count],
            max_dry_streak: vec![0; cell_count],
            snow_total_per_tick: Vec::with_capacity(measure_ticks as usize),
            precip_per_tick: Vec::with_capacity(measure_ticks as usize),
            fully_dry_ticks: 0,
            fully_rain_free_ticks: 0,
            upper_layer_altitude_m: effective.atmosphere.upper_layer_altitude_m,
            lapse_rate: effective.temperature.lapse_rate,
            initial_water: total_water_budget(sim),
            nan_inf_seen: false,
            ticks_observed: 0,
        }
    }

    /// Called after each `sim.step()` during the measurement phase.
    pub fn observe(&mut self, sim: &Simulation) {
        let grid = sim.grid();
        let precip = sim.last_precipitation();
        let mut precip_this_tick = 0.0_f32;
        let mut snow_total = 0.0_f32;
        let mut any_rain_this_tick = false;
        let mut any_rain_only_this_tick = false;
        let t_offset = self.lapse_rate * self.upper_layer_altitude_m / 1000.0;
        let altitude_m = self.upper_layer_altitude_m;

        for (i, (_, cell)) in grid.iter().enumerate() {
            // NaN/Inf check on the main stocks
            if !cell.water_level.is_finite()
                || !cell.temperature.is_finite()
                || !cell.humidity_upper.is_finite()
                || !cell.cloud_water.is_finite()
                || !cell.snow_level.is_finite()
            {
                self.nan_inf_seen = true;
            }

            let p = precip.get(i);
            let rain = p.map_or(0.0, |d| d.rain);
            let snow = p.map_or(0.0, |d| d.snow);
            let rained = rain > 1e-4;
            let wet = rained || snow > 1e-4;
            precip_this_tick += rain + snow;
            snow_total += cell.snow_level;
            self.rain_total_mm[i] += rain;
            if rain > self.max_daily_rain[i] {
                self.max_daily_rain[i] = rain;
            }
            if rain > 1.0 {
                self.gust_ticks[i] += 1;
            }

            // `any_rain_this_tick` includes snow (total precip, used for
            // `fully_dry`); `any_rain_only_this_tick` is rain alone (used
            // for `fully_rain_free` = high-pressure phase, cf. #86).
            if wet {
                any_rain_this_tick = true;
            }
            if rained {
                any_rain_only_this_tick = true;
            }

            // `rain_ticks` and `dry_streak` count RAIN alone (consistent
            // with their docstrings: "rain > 1e-4" / "streak without
            // rain"). Alpine snow, near-permanent at altitude, was
            // inflating these metrics and making a fake drizzle appear
            // on relief.
            if rained {
                self.rain_ticks[i] += 1;
                // End of dry streak: update the max and reset.
                if self.current_dry_streak[i] > self.max_dry_streak[i] {
                    self.max_dry_streak[i] = self.current_dry_streak[i];
                }
                self.current_dry_streak[i] = 0;
            } else {
                self.current_dry_streak[i] += 1;
            }

            // Visible cloud: RH > 0.6 (same Tetens saturation as
            // step_cloud_dynamics). Phase 6 (#29): Clausius-Clapeyron via
            // `saturation_upper_pw` (pure, depends only on altitude).
            let t_upper = cell.temperature - t_offset;
            let sat = crate::atmosphere::saturation_upper_pw(t_upper, altitude_m);
            let hr = if sat > 0.0 {
                cell.humidity_upper / sat
            } else {
                0.0
            };
            if hr > 0.6 {
                self.cloudy_ticks[i] += 1;
            }
        }

        if !any_rain_this_tick {
            self.fully_dry_ticks += 1;
        }
        if !any_rain_only_this_tick {
            self.fully_rain_free_ticks += 1;
        }
        self.precip_per_tick.push(precip_this_tick);
        self.snow_total_per_tick.push(snow_total);
        self.ticks_observed += 1;
    }

    /// Computes the final metrics. `elapsed` = total duration of the
    /// measurement loop (not warmup).
    #[must_use]
    pub fn finalize(
        mut self,
        sim: &Simulation,
        elapsed: std::time::Duration,
    ) -> (Metrics, RunStatus) {
        // For each cell, close out the current streak (count the ongoing
        // streak if it's longer than the max seen).
        for i in 0..self.cell_count {
            if self.current_dry_streak[i] > self.max_dry_streak[i] {
                self.max_dry_streak[i] = self.current_dry_streak[i];
            }
        }

        let final_water = total_water_budget(sim);
        let water_drift_pct = if self.initial_water.abs() > 1e-6 {
            (final_water - self.initial_water).abs() / self.initial_water
        } else {
            0.0
        };

        // rain_days normalization: rain_ticks over `measure_ticks` -> days/year
        let scale = 365.0 / self.measure_ticks as f32;
        let rain_days_max_per_cell =
            self.rain_ticks.iter().copied().max().unwrap_or(0) as f32 * scale;

        // Top 3 rainiest cells (spatial diagnostic).
        let mut indexed: Vec<(usize, u32)> = self.rain_ticks.iter().copied().enumerate().collect();
        indexed.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let coords = sim.grid().coords_slice();
        let grid = sim.grid();
        let rain_hotspots: Vec<RainHotspot> = indexed
            .iter()
            .take(3)
            .filter_map(|(idx, ticks)| {
                let coord = *coords.get(*idx)?;
                let cell = grid.get(coord)?;
                Some(RainHotspot {
                    q: coord.q,
                    r: coord.r,
                    rain_days: *ticks as f32 * scale,
                    elevation: cell.elevation,
                    water_level: cell.water_level,
                })
            })
            .collect();

        // Median days/year per elevation band
        let mut bands: [Vec<f32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (i, &elev) in self.elevations.iter().enumerate() {
            let days = self.rain_ticks[i] as f32 * scale;
            let bi = band_index(elev);
            bands[bi].push(days);
        }
        let rain_days_median_by_altitude = std::array::from_fn(|i| median(&mut bands[i]));

        // Seasonality: we split the observed ticks into seasons based on
        // the day of the year at the time of measurement. Tick 0 of the
        // measurement = first measurement day; we assume the sim calendar
        // starts at day 0 and cycles modulo 365. So tick t modulo 365.
        // Warmup = 365 ticks by default -> tick 0 of the measurement =
        // day 0 of year 1. Consistent for a well-aligned seasonal cycle.
        let mut precip_summer = 0.0_f64;
        let mut n_summer = 0_u32;
        let mut precip_winter = 0.0_f64;
        let mut n_winter = 0_u32;
        for (t, &p) in self.precip_per_tick.iter().enumerate() {
            let day = (t as u64) % 365;
            if (146..=237).contains(&day) {
                precip_summer += f64::from(p);
                n_summer += 1;
            } else if day <= 54 || day >= 335 {
                precip_winter += f64::from(p);
                n_winter += 1;
            }
        }
        let mean_summer = if n_summer > 0 {
            precip_summer / f64::from(n_summer)
        } else {
            0.0
        };
        let mean_winter = if n_winter > 0 {
            precip_winter / f64::from(n_winter)
        } else {
            0.0
        };
        let ratio_precip_summer_winter = if mean_winter > 1e-9 {
            (mean_summer / mean_winter) as f32
        } else if mean_summer > 1e-9 {
            // Winter fully dry, summer rainy -> arbitrarily large ratio
            999.0
        } else {
            // Both dry, no signal
            1.0
        };

        // Median of the per-cell max streak, over a plain/hill sample
        // (50-600m) to be comparable to the existing check in
        // scale_dry_periods.
        let mut streaks: Vec<f32> = self
            .elevations
            .iter()
            .zip(self.max_dry_streak.iter())
            .filter_map(|(&e, &s)| {
                if (50.0..600.0).contains(&e) {
                    Some(s as f32)
                } else {
                    None
                }
            })
            .collect();
        let dry_streak_median_per_cell_days = median(&mut streaks);

        // Average rainfall of the plain/hill band (50-600m) and fraction
        // of gust days (> 1 mm/day), over the same spatial sample as the
        // dry_streak. mm/day = cumulative rain / number of measured days.
        let measure_days = self.measure_ticks.max(1) as f32;
        let mut plains_precip: Vec<f32> = Vec::new();
        let mut gust_fracs: Vec<f32> = Vec::new();
        let mut plains_max_rain: Vec<f32> = Vec::new();
        for (i, &e) in self.elevations.iter().enumerate() {
            if (50.0..600.0).contains(&e) {
                plains_precip.push(self.rain_total_mm[i] / measure_days);
                gust_fracs.push(self.gust_ticks[i] as f32 / measure_days);
                plains_max_rain.push(self.max_daily_rain[i]);
            }
        }
        let plains_precip_mm_per_day = if plains_precip.is_empty() {
            f32::NAN
        } else {
            plains_precip.iter().sum::<f32>() / plains_precip.len() as f32
        };
        let gust_days_frac_mean = if gust_fracs.is_empty() {
            f32::NAN
        } else {
            gust_fracs.iter().sum::<f32>() / gust_fracs.len() as f32
        };
        let gust_days_frac_median = median(&mut gust_fracs);
        let plains_max_daily_rain_median_mm = median(&mut plains_max_rain);

        // Snow cycle: min observed in summer / max observed in winter
        let mut snow_max_winter = f32::NEG_INFINITY;
        let mut snow_min_summer = f32::INFINITY;
        for (t, &s) in self.snow_total_per_tick.iter().enumerate() {
            let day = (t as u64) % 365;
            if (146..=237).contains(&day) && s < snow_min_summer {
                snow_min_summer = s;
            } else if (day <= 54 || day >= 335) && s > snow_max_winter {
                snow_max_winter = s;
            }
        }
        let snow_ratio_winter_max_summer_min = if snow_max_winter > 1e-6 {
            (snow_min_summer / snow_max_winter).max(0.0)
        } else {
            // No winter snow, return 1.0 (= no cycle)
            1.0
        };

        // Mountain cloud cover: cells >1000m with >5% cloudy ticks
        let cloud_threshold = (self.measure_ticks as f32 * 0.05) as u32;
        let mut mountain_cells = 0_u32;
        let mut mountain_cloudy = 0_u32;
        for (i, &elev) in self.elevations.iter().enumerate() {
            if elev > 1000.0 {
                mountain_cells += 1;
                if self.cloudy_ticks[i] > cloud_threshold {
                    mountain_cloudy += 1;
                }
            }
        }
        let cloud_cover_mountain_pct = if mountain_cells > 0 {
            f32::from(u16::try_from(mountain_cloudy).unwrap_or(0))
                / f32::from(u16::try_from(mountain_cells).unwrap_or(0))
        } else {
            0.0
        };

        let ms_per_tick = if self.ticks_observed > 0 {
            (elapsed.as_secs_f64() * 1000.0 / self.ticks_observed as f64) as f32
        } else {
            0.0
        };

        let effective_lapse_rate_c_per_km =
            crate::diagnostics::effective_lapse_rate_c_per_km(sim.grid());

        let metrics = Metrics {
            water_drift_pct,
            rain_days_max_per_cell,
            rain_hotspots,
            rain_days_median_by_altitude,
            ratio_precip_summer_winter,
            dry_streak_median_per_cell_days,
            plains_precip_mm_per_day,
            gust_days_frac_median,
            gust_days_frac_mean,
            plains_max_daily_rain_median_mm,
            fully_dry_days_total: self.fully_dry_ticks as f32,
            fully_rain_free_days_total: self.fully_rain_free_ticks as f32,
            snow_ratio_winter_max_summer_min,
            cloud_cover_mountain_pct,
            effective_lapse_rate_c_per_km,
            ms_per_tick,
            cell_count: self.cell_count,
        };

        let status = if self.nan_inf_seen {
            RunStatus::NanInf
        } else {
            RunStatus::Ok
        };
        (metrics, status)
    }
}

fn total_water_budget(sim: &Simulation) -> f32 {
    sim.grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum()
}

fn band_index(elev: f32) -> usize {
    if elev < 300.0 {
        0
    } else if elev < 800.0 {
        1
    } else if elev < 1500.0 {
        2
    } else {
        3
    }
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n.is_multiple_of(2) {
        f32::midpoint(values[n / 2 - 1], values[n / 2])
    } else {
        values[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_overrides_preserve_defaults() {
        let (_, effective) = build_bench_sim(42, 3, &BenchParams::default());
        let def = AtmosphereParams::default();
        // upper_layer_altitude_m is not scaled by TICKS_PER_DAY
        // (cf. scale_atmosphere_for_hourly_tick), so it stays identical to
        // the default after bench setup.
        assert!(
            (effective.atmosphere.upper_layer_altitude_m - def.upper_layer_altitude_m).abs() < 1e-6
        );
    }

    #[test]
    fn override_applies_single_field() {
        // upper_layer_altitude_m is not scaled by TICKS_PER_DAY, so the
        // override is applied identically (vs default).
        let overrides = BenchParams {
            atmosphere: AtmosphereParamsOverride {
                upper_layer_altitude_m: Some(2000.0),
                ..AtmosphereParamsOverride::default()
            },
            ..BenchParams::default()
        };
        let (_, effective) = build_bench_sim(42, 3, &overrides);
        assert!((effective.atmosphere.upper_layer_altitude_m - 2000.0).abs() < 1e-6);
        let def = AtmosphereParams::default();
        assert!(
            (effective.atmosphere.upper_layer_altitude_m - def.upper_layer_altitude_m).abs() > 1.0,
            "override must actually shift the effective value vs default"
        );
    }

    #[test]
    fn deny_unknown_fields_rejects_typo() {
        let json = r#"{ "atmosphere": { "precip_rate_typo": 0.05 } }"#;
        let parsed: Result<BenchParams, _> = serde_json::from_str(json);
        assert!(
            parsed.is_err(),
            "an unknown field must fail deserialization"
        );
    }

    #[test]
    fn accumulator_runs_without_panic() {
        let (mut sim, effective) = build_bench_sim(42, 3, &BenchParams::default());
        for _ in 0..30 {
            sim.step();
        }
        let mut acc = MetricsAccumulator::start(&sim, &effective, 30);
        for _ in 0..30 {
            sim.step();
            acc.observe(&sim);
        }
        let (metrics, status) = acc.finalize(&sim, std::time::Duration::from_millis(100));
        assert_eq!(status, RunStatus::Ok);
        assert!(metrics.ms_per_tick > 0.0);
    }
}
