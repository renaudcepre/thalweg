use crate::atmosphere::{
    AtmoForcing, AtmoScratch, AtmosphereParams, PrecipitationMap, step_atmosphere_into,
};
use crate::checkpoint::{CHECKPOINT_FORMAT_VERSION, Checkpoint, CheckpointError, MAGIC};
use crate::climate::{ClimateHistory, DayRecord};
use crate::climate_normals::{CellClimateNormals, ClimateNormalsAccumulator};
use crate::diagnostics::{Diagnostics, compute_diagnostics};
use crate::dynamics::{SynopticParams, SynopticState};
use crate::erosion::{
    DischargeEmaMap, EdgeFluxEmaMap, ErosionForcing, ErosionParams, step_erosion, update_edge_ema,
    update_ema,
};
use crate::fire::{FireParams, step_fire};
use crate::grid::{GridState, HexGrid};
use crate::groundwater::{GroundwaterParams, step_groundwater};
use crate::hydro::{
    DischargeMap, EdgeFluxMap, FlowVecMap, FluxMap, HydroMaps, HydroParams, step_hydro_mfd_into,
};
use crate::lake::{LakeParams, step_lake_leveling};
use crate::phase_timing::{PhaseTimings, elapsed_s, mark};
use crate::snow::{SnowForcing, SnowParams, step_snow};
use crate::synoptic_mesh::SynopticMesh;
use crate::temperature::{
    IllumCache, TemperatureParams, aspect_insolation_correction, compute_illumination_cached,
    compute_surface_normals, solar_beam_at_tick, step_temperature,
};
use crate::time::{self, TICKS_PER_DAY};
use crate::vegetation::{VegetationParams, step_vegetation};
use crate::wind::{
    WindField, WindParams, WindVec, compute_wind_field_into, compute_wind_magnitudes_into,
};
use serde::Serialize;

/// Fire metrics accumulated since the simulation started (#wildfire).
/// `peak_burning` = largest number of cells simultaneously on fire, the
/// guard against "the whole map burned down" (must stay small vs. the total).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct FireStats {
    pub ignitions_total: u64,
    pub cell_days_total: u64,
    pub peak_burning: u32,
    pub currently_burning: usize,
}

/// MFD integration budget: 8 passes/day to converge the surface water
/// transfer (`flow_rate=0.12` per pass -> CFL ~1/7, each pass moves at most
/// a fraction of the cell). **This is not a time proxy**: the 8 passes do
/// not represent 8 physical hours. Legacy from v0.2.x where 1 tick = 1 day;
/// under the hourly regime (v0.3.0+) these passes are still executed in a
/// burst once a day in `step_daily_tail`, after `step_groundwater`, to
/// preserve the infiltration -> runoff order that closes the mass balance.
///
/// Issue #48: option (1), cosmetic, promoting to Tier 2 (1 pass/hour) broke
/// conservation via gw/hydro decoupling. Options (2) recalibrate 24/day and
/// (3) Tier 2 refactor are to be evaluated separately.
const HYDRO_MFD_PASSES_PER_DAY: u32 = 8;

/// Wind field subsampling (issue #89). `compute_wind_field_into` used to run
/// every hour (24x/day) even though its dominant components are *day-scale*:
/// the noise (`compute_noise_wind`) is indexed by day, the westerly bias and
/// terrain deflection are static. Only the thermal breeze
/// (`add_thermal_component`, weight `thermal_strength`) varies hourly. So we
/// only recompute the field one hour out of N, reusing `wind_field` in
/// between. `N=1` = historical behaviour (hourly recompute).
///
/// A/B ablation radius 30, seed 42, 3 years (warmup 365), transport already
/// at N=3:
///
/// | N | ms/tick         | byAlt (plains/hills/mid/high) | s/w   | mtnClouds |
/// |---|-----------------|--------------------------------|-------|-----------|
/// | 1 | 48.38           | 247/106/334/365                | 34.27 | 66.7 %    |
/// | 2 | 39.63 (−18 %)   | 246/106/334/365                | 34.26 | 66.7 %    |
/// | 3 | 36.38 (−24.8 %) | 247/107/334/365                | 34.29 | 66.7 %    |
///
/// Identical climate (±1 day = noise), conservation intact. `N=3` = sweet
/// spot: aligns wind with the transport cadence (refreshed every 3 h) and
/// keeps the diurnal breeze sampled finely enough. Going higher is not
/// justified by these metrics (rain/clouds), which don't capture local
/// breeze fidelity. Overridable via `HEXSIM_WIND_SUBSAMPLE` for A/B testing.
const WIND_SUBSAMPLE_HOURS: u64 = 3;

/// Effective value: `WIND_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_WIND_SUBSAMPLE` (parametric A/B without recompiling). Read once.
fn wind_subsample() -> u64 {
    use std::sync::OnceLock;
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("HEXSIM_WIND_SUBSAMPLE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(WIND_SUBSAMPLE_HOURS)
    })
}

/// Synoptic ODE subsampling. The prognostic integrator
/// (`SynopticState::step_hour`) is the dominant cost of the tick when the
/// synoptic layer is active (~68 %: 41 ms ON vs. 13 ms OFF, r30): 20
/// substeps/hour imposed by the CFL (advection `U=3` 73 %, waves `c≈1.1`
/// 27 %), hence *irreducible* by lowering the `substeps` floor. So we advance
/// the state by one hour of physical time one hour out of M, and freeze
/// `synoptic_base` in between; the wind that consumes it is already
/// subsampled to 3 h (#89), and the synoptic field evolves at day-scale
/// (relaxation, low `c`), so running its clock at 1/M doesn't move the
/// climatology.
///
/// Unlike wind (#89, *diagnostic* field recomputed-and-reused with no loss),
/// the synoptic state is *prognostic*: this subsample is a real physics
/// change (systems slowed to 1/M), validated by climate ablation and not by
/// construction. Ablation r30, seed {42,7,99}, warmup 365, 3 years:
///
/// | M | ms/tick        | s/w (42/7/99)   | byAlt drift | conservation |
/// |---|----------------|-----------------|-------------|--------------|
/// | 1 | 41.2           | 63.8/40.1/51.6  | —           | ✓            |
/// | 2 | 27.0 (−34 %)   | 64.3/39.6/52.0  | ≤1 d        | ✓            |
/// | 3 | 22.3 (−46 %)   | 65.0/39.7/52.7  | ≤3 d        | ✓            |
/// | 4 | 20.0 (−51 %)   | 64.9/  — /  —   | ≤2 d        | ✓            |
///
/// `M=3` = sweet spot: near-max gain and *alignment* with the wind subsample
/// (wind then always reads a freshly-written `synoptic_base`). `plainsP`
/// drifts ±1-4 % with no consistent sign across seeds -> noise, not a bias.
/// `M=1` = historical behaviour. Overridable via `HEXSIM_SYNOPTIC_SUBSAMPLE`.
const SYNOPTIC_SUBSAMPLE_HOURS: u64 = 3;

/// Effective value: `SYNOPTIC_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_SYNOPTIC_SUBSAMPLE` (parametric A/B without recompiling). Read once.
fn synoptic_subsample() -> u64 {
    use std::sync::OnceLock;
    static M: OnceLock<u64> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("HEXSIM_SYNOPTIC_SUBSAMPLE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(SYNOPTIC_SUBSAMPLE_HOURS)
    })
}

/// Effective subsample cadences (#89, synoptic), exposed so that instruments
/// that replay the tick phase by phase (`perf_phase_breakdown`) measure at
/// production cadences, including environment overrides. An instrument that
/// hardcodes its own cadence silently drifts from production (measured
/// 2026-07-10: the breakdown ignored the synoptic phase, 82 % of the tick,
/// and recomputed wind every hour).
#[must_use]
pub fn wind_subsample_hours() -> u64 {
    wind_subsample()
}

/// See [`wind_subsample_hours`].
#[must_use]
pub fn synoptic_subsample_hours() -> u64 {
    synoptic_subsample()
}

pub struct Simulation {
    current: HexGrid,
    next: HexGrid,
    /// Cumulative hour counter since creation (v0.3.0, issue #38). 1 internal
    /// tick = 1 hour. The external API `tick()` returns days (v0.2.x compat);
    /// `hour_tick()` exposes this counter directly.
    hour_tick: u64,
    hydro_params: HydroParams,
    atmosphere_params: AtmosphereParams,
    groundwater_params: GroundwaterParams,
    snow_params: SnowParams,
    temperature_params: TemperatureParams,
    wind_params: WindParams,
    /// First cut (epic #71): defaulted internally rather than passed to
    /// `new()`, to avoid breaking the ~30 existing call sites. Promotion to a
    /// constructor argument to be evaluated once the layer stabilizes.
    vegetation_params: VegetationParams,
    /// Emergent wildfire (#wildfire). Dormant by default (`enabled = false`).
    fire_params: FireParams,
    /// Seed for the fire's deterministic randomness (ignition/spread).
    /// Defaults to 0; the server sets it to the world seed (`set_seed`) so
    /// "1 seed = 1 world" holds down to the fires.
    seed: u32,
    /// Cumulative fire counters (calibration metrics).
    fire_ignitions_total: u64,
    fire_cell_days_total: u64,
    fire_peak_burning: u32,
    discharge_map: DischargeMap,
    flow_vec_map: FlowVecMap,
    /// Outgoing flux per edge (order `coord::DIRECTIONS`), same lifecycle as
    /// `discharge_map`: reset then accumulated over the 8 passes of the daily
    /// hydro slice, stable until the next slice (#103). Source of truth for
    /// the edges of `diag_water_flows`, never reconstructed by projecting
    /// `flow_vec`. Since #106, the front-end render consumes `edge_flux_ema`
    /// (smoother), not this raw slice.
    edge_flux_map: EdgeFluxMap,
    /// River erosion (#105). Enabled by default, background geophysics, not
    /// an optional mode. Defaulted internally like `vegetation_params`.
    erosion_params: ErosionParams,
    /// Hydrostatic leveling of multi-hex lakes (#106). Defaulted internally
    /// like `erosion_params`; preserved on reset via `set_lake_params`.
    lake_params: LakeParams,
    /// EMA of daily discharge (mm/day), updated after each hydro slice, even
    /// with erosion disabled. It's the flux history that drives stream power
    /// and, since #106, the front-end render (`snapshot`), never the
    /// instantaneous value (anti-pattern #3).
    discharge_ema: DischargeEmaMap,
    /// EMA of flux per edge, same cadence: the directional memory that routes
    /// sediment load downstream and the displayed network (#106).
    edge_flux_ema: EdgeFluxEmaMap,
    /// Cumulative erosion counters (m of bedrock incised / redeposited since
    /// creation), diag instrumentation, f64 to absorb the years.
    erosion_incised_total: f64,
    erosion_deposited_total: f64,
    wind_field: WindField,
    /// Surface wind field forced for tests/diagnostics (see
    /// `set_uniform_wind`). `Some` short-circuits the normal wind recompute
    /// (noise/thermal/terrain/synoptic) and disables the synoptic dynamics.
    /// Never serialized to the checkpoint, reconstructed by the test's
    /// explicit call on each run.
    uniform_wind: Option<WindVec>,
    /// Magnitudes of `wind_field`, recomputed at the same cadence as it
    /// (subsample #89). Consumed by Meyer evaporation, avoids a `sqrt` per
    /// cell and per hour on a field that only changes every N hours (perf
    /// project #88).
    wind_mag: Vec<f32>,
    /// Prognostic synoptic dynamics (Phase 1 of the synoptic-dynamics
    /// design: f-plane shallow-water core, not yet coupled to precip).
    /// Evolved every hour when `synoptic_enabled`; its geostrophic wind then
    /// replaces the noisy base of `wind_field` (via `synoptic_base`).
    synoptic_params: SynopticParams,
    synoptic_state: SynopticState,
    /// Activation flag. **ON by default, hardcoded since #108**: the
    /// background wind and its dominant direction emerge from the pressure
    /// field. The ON default is no longer an env flag, only the runtime
    /// param `update_param("synoptic.enabled", 0.0)` can disable it
    /// (advection tests use this to isolate a deterministic scripted wind
    /// via `set_uniform_wind`).
    synoptic_enabled: bool,
    /// Synoptic wind expressed in `WindVec` units, rewritten every hour when
    /// enabled, injected as the base into `compute_wind_field_into`.
    synoptic_base: WindField,
    /// Fine-to-coarse coupling of the synoptic solver (#88): the solver
    /// integrates on a torus dedicated to ~calibration spacing (~1 km), not on
    /// the fine grid. Never serialized, deterministically reconstructed from
    /// the grid and coarse radius in the checkpoint.
    synoptic_mesh: SynopticMesh,
    /// Base wind on the coarse torus before interpolation to `synoptic_base`
    /// (scratch, coarse size).
    synoptic_coarse_base: WindField,
    climate_history: ClimateHistory,
    last_precipitation: PrecipitationMap,
    /// Persistent state of the global precipitation gate (hysteresis).
    /// See `AtmosphereParams.global_precip_gate`.
    precip_gate_open: bool,
    // Scratch buffers reused each tick (zero malloc in step()).
    scratch_wind_snap: WindField,
    scratch_atmo: AtmoScratch,
    scratch_flux: FluxMap,
    scratch_flow_vec: FlowVecMap,
    scratch_edge_flux: EdgeFluxMap,
    /// Current hour precipitation (written by `step_atmosphere_into`).
    /// Accumulated into `last_precipitation` after each Tier 1 sub-tick, reset
    /// at the start of each day. Allows `climate_history` to see 24h total
    /// precipitation rather than just the last tick.
    scratch_precip_tick: PrecipitationMap,
    /// Illumination per cell (#102), recalculated each tick: `scratch_flux_factor`
    /// = physical factor (absorbed flux = beam × factor), `scratch_illumination`
    /// ∈ [0,1] for display. Scratch, no allocation per tick.
    scratch_flux_factor: Vec<f32>,
    scratch_illumination: Vec<f32>,
    /// Terrain precalculations for illumination (#65): horizon tangents + jump
    /// tables. Derived from elevation alone, lazily reconstructed (`ensure`) at
    /// the first tick and after effective erosion (`mark_dirty` at the same place
    /// as normal recalculation). Never serialized, derived not state.
    illum_cache: IllumCache,
    /// Climate normals per cell (#79, foundation of niche model #78).
    /// Accumulates T/water/insolation each hour, frozen at year rollover.
    /// Read-only, does not affect conservation.
    climate_normals: ClimateNormalsAccumulator,
    /// Cumulative wall-clock per tick phase (see `phase_timing`). Transient,
    /// never serialized to checkpoint, reset at loading. Inert on wasm32
    /// (no-op clock).
    timings: PhaseTimings,
}

impl Simulation {
    #[must_use]
    pub fn new(
        mut grid: HexGrid,
        hydro_params: HydroParams,
        atmosphere_params: AtmosphereParams,
        groundwater_params: GroundwaterParams,
        snow_params: SnowParams,
        temperature_params: TemperatureParams,
        wind_params: WindParams,
    ) -> Self {
        let n = grid.len();
        // Closed terrarium: no external humidity input. We bootstrap the cycle
        // by applying a floor to `humidity_upper` from tick 0, otherwise the
        // system starts entirely dry and takes years to start.
        let floor = atmosphere_params.initial_humidity_floor;
        if floor > 0.0 {
            let coords: Vec<_> = grid.coords().copied().collect();
            for coord in coords {
                if let Some(cell) = grid.get_mut(coord) {
                    cell.humidity_upper = cell.humidity_upper.max(floor);
                }
            }
        }

        // Surface normal per cell (aspect sunny/shaded slope, #102): precalculated
        // once, elevation frozen after generation (no erosion). Both buffers
        // inherit it (clone below), and each tick preserves it via the
        // clone_from_slice in `step_temperature`.
        compute_surface_normals(&mut grid);
        // Thermal offset calibration: localizing flux by aspect shifts the map
        // mean of the geometric factor; we absorb it in the offset to keep
        // `mean_annual(T) = base_temp`. Flat terrain -> 0 -> unchanged.
        let mut temperature_params = temperature_params;
        temperature_params.aspect_correction =
            aspect_insolation_correction(&grid, &temperature_params);
        let next = grid.clone();

        // Synoptic dynamics: seed + latitude inherited from world to stay
        // consistent with "one world". Solver integrates on its dedicated torus
        // (~calibration spacing, `SynopticMesh`), not on the fine grid at 130 m
        // where CFL imposed 163 substeps/h on all cells, 82 % of tick (task #88).
        // `HEXSIM_SYNOPTIC_COARSE=0` forces identity mesh (historical fine-grid
        // behavior, bit-for-bit) for ablation A/B.
        //
        // Coarse ON toggle validated by climate ablation (2026-07-10, M=3
        // subsample protocol: hexsim-bench r30, seeds {42,7,99}, warmup 365 d,
        // measure 3 years, fine vs coarse):
        //
        // | metric                      | 42          | 7           | 99         |
        // |-----------------------------|-------------|-------------|------------|
        // | water_drift (x1e-5)         | 1.93->1.92  | 4.46->4.54  | 1.10->1.02 |
        // | byAlt (d/yr, 4 bands)       | ≤3 d        | ≤2 d        | ≤1 d       |
        // | plainsP (mm/d)              | -0.1 %      | -1.3 %      | +0.5 %     |
        // | dry_streak (d)              | -0.4 %      | +9.4 %*     | -3.5 %     |
        // | summer/winter ratio         | -0.6 %      | -0.5 %      | +0.2 %     |
        // | gust_days_frac_mean         | +0.0 %      | +0.4 %      | -1.5 %     |
        // | ms/tick (6-proc contention) | -74 %       | -74 %       | -72 %      |
        //
        // (*) metric nearly saturated (971-1095 d over 1095 window).
        // plains_max_daily_rain_median (~0.0003-0.003 mm/d, 300x below gust
        // threshold) oscillates -100 % to +124 % with no constant sign by seed,
        // so floor noise, same verdict as plainsP on 07-07. Perf outside
        // contention: synoptic 83.2 % -> 2.0 % of tick r30 (7.131 -> 0.029
        // ms/hour-tick), r45 end-to-end 427 -> 75.3 ms/tick.
        let synoptic_coarse = std::env::var("HEXSIM_SYNOPTIC_COARSE")
            .map_or(true, |v| v != "0" && !v.eq_ignore_ascii_case("false"));
        let mut synoptic_mesh = if synoptic_coarse {
            SynopticMesh::build(&grid)
        } else {
            SynopticMesh::identity(&grid)
        };
        let synoptic_params = SynopticParams {
            seed: wind_params.seed,
            latitude_deg: temperature_params.latitude_deg,
            ..SynopticParams::for_spacing(synoptic_mesh.spacing_m())
        };
        let n_coarse = synoptic_mesh.grid().len();
        let synoptic_state = SynopticState::new(n_coarse, &synoptic_params);
        // Prod default hardcoded since #108 (ex-env flag `HEXSIM_SYNOPTIC`,
        // exact duplicate of runtime param `synoptic.enabled`). Override via
        // `update_param("synoptic.enabled", ...)`.
        let synoptic_enabled = true;
        synoptic_mesh.aggregate_temperature(&grid);
        let mut synoptic_coarse_base: WindField = vec![WindVec::default(); n_coarse];
        synoptic_state.write_base_wind(&synoptic_params, &mut synoptic_coarse_base);
        let mut synoptic_base: WindField = vec![WindVec::default(); n];
        synoptic_mesh.interpolate_wind(&synoptic_coarse_base, &mut synoptic_base);

        let mut wind_field: WindField = vec![WindVec::default(); n];
        let mut scratch_wind_snap: WindField = vec![WindVec::default(); n];
        compute_wind_field_into(
            &grid,
            &wind_params,
            0,
            &mut wind_field,
            &mut scratch_wind_snap,
            synoptic_enabled.then_some(&synoptic_base),
        );
        let mut wind_mag: Vec<f32> = Vec::with_capacity(n);
        compute_wind_magnitudes_into(&wind_field, &mut wind_mag);
        Self {
            current: grid,
            next,
            hour_tick: 0,
            hydro_params,
            atmosphere_params,
            groundwater_params,
            snow_params,
            temperature_params,
            wind_params,
            vegetation_params: VegetationParams::default(),
            fire_params: FireParams::default(),
            seed: 0,
            fire_ignitions_total: 0,
            fire_cell_days_total: 0,
            fire_peak_burning: 0,
            discharge_map: vec![0.0; n],
            flow_vec_map: vec![(0.0, 0.0); n],
            edge_flux_map: vec![[0.0; 6]; n],
            erosion_params: ErosionParams::default(),
            lake_params: LakeParams::default(),
            discharge_ema: vec![0.0; n],
            edge_flux_ema: vec![[0.0; 6]; n],
            erosion_incised_total: 0.0,
            erosion_deposited_total: 0.0,
            wind_field,
            uniform_wind: None,
            wind_mag,
            synoptic_params,
            synoptic_state,
            synoptic_enabled,
            synoptic_base,
            synoptic_mesh,
            synoptic_coarse_base,
            climate_history: ClimateHistory::new(),
            last_precipitation: vec![DayRecord::default(); n],
            precip_gate_open: false,
            scratch_wind_snap,
            scratch_atmo: AtmoScratch::new(n),
            scratch_flux: vec![0.0; n],
            scratch_flow_vec: vec![(0.0, 0.0); n],
            scratch_edge_flux: vec![[0.0; 6]; n],
            scratch_precip_tick: vec![DayRecord::default(); n],
            scratch_flux_factor: vec![0.0; n],
            scratch_illumination: vec![1.0; n],
            illum_cache: IllumCache::new(),
            climate_normals: ClimateNormalsAccumulator::new(n),
            timings: PhaseTimings::default(),
        }
    }

    /// Advance the simulation by 1 day = `TICKS_PER_DAY` hours.
    ///
    /// Historical external API preserved: 1 call to `step()` advances the sim
    /// by one day. Internally, the loop iterates over hourly sub-ticks
    /// (`step_hour`) to establish v0.3.0 infrastructure (#38). In PR1, all
    /// phenomena remain Tier 3 (once per day), physics unchanged. PR2 will
    /// promote wind/temp/atmo to Tier 1.
    pub fn step(&mut self) {
        for _ in 0..TICKS_PER_DAY {
            self.step_hour();
        }
    }

    /// Advance the simulation by 1 hour. Tiered scheduler:
    /// - Tier 1 (each tick): `step_temperature`, `compute_wind_field_into`,
    ///   `step_snow`, `step_atmosphere_into` — complete radiative-atmo dynamics
    ///   at diurnal resolution. Precipitation accumulated in `last_precipitation`
    ///   over 24 sub-ticks.
    /// - Tier 3 (start of next day): `climate_history.record_tick`,
    ///   `step_groundwater`, then `step_hydro_mfd_into` x8. Consume the freshly
    ///   closed 24h precipitation sum. The gw -> hydro order preserves mass
    ///   conservation (strict 10-year test).
    pub fn step_hour(&mut self) {
        // Start of a new day: reset the precip accumulator. Hydro flux maps
        // (`discharge_map` & co) are NOT reset here; they reset at the start of
        // the hydro slice itself (`step_hydro_tranche`), otherwise they'd be
        // empty 23 h out of 24 for any reader not at midnight sharp (#103).
        if time::hour_of_day(self.hour_tick) == 0 {
            for record in &mut self.last_precipitation {
                *record = DayRecord::default();
            }
        }

        // Year rollover: freeze the climate normals of the elapsed year (#79).
        // `hour_tick` multiple of TICKS_PER_YEAR = N complete years accumulated.
        // Normals N-1 serve year N (assumed 1-year lag).
        if self.hour_tick > 0 && self.hour_tick.is_multiple_of(time::TICKS_PER_YEAR) {
            self.climate_normals.finalize_year();
        }

        // Illumination per cell (#102): aspect x relief occlusion x cloud shadow,
        // calculated once per tick (toroidal raymarch) from `current`.
        // `scratch_flux_factor` drives physics (absorbed flux = beam x factor),
        // `scratch_illumination` [0,1] will serve display.
        let solar = solar_beam_at_tick(&self.temperature_params, self.hour_tick);
        let t0 = mark();
        // Terrain precalculations (#65): no-op in steady state, rebuild at first
        // tick and after effective erosion (`mark_dirty` below).
        self.illum_cache.ensure(&self.current);
        compute_illumination_cached(
            &self.current,
            &solar,
            self.temperature_params.cloud_albedo_coef,
            self.atmosphere_params.upper_layer_altitude_m,
            &self.illum_cache,
            &mut self.scratch_flux_factor,
            &mut self.scratch_illumination,
        );
        self.timings.illumination += elapsed_s(t0);

        // Tier 1: the whole atmo-radiative pipeline, every hour.
        let t0 = mark();
        step_temperature(
            &self.current,
            &mut self.next,
            &self.temperature_params,
            self.hour_tick,
            &self.scratch_flux_factor,
            &self.snow_params,
        );
        std::mem::swap(&mut self.current, &mut self.next);
        self.timings.temperature += elapsed_s(t0);

        self.refresh_wind();

        // Melt SI balance forcings (#60 Phase 1): same beam and illumination
        // per cell as `step_temperature` (single source of truth), memoized wind
        // (#89), and rain from previous tick for rain-on-snow heat advection.
        let t0 = mark();
        step_snow(
            &self.current,
            &mut self.next,
            &self.snow_params,
            self.groundwater_params.max_capacity,
            &SnowForcing {
                beam_w_m2: solar.beam,
                ground_albedo: self.temperature_params.ground_albedo,
                flux_factor: &self.scratch_flux_factor,
                wind_mag: &self.wind_mag,
                rain_last_tick: &self.scratch_precip_tick,
            },
        );
        std::mem::swap(&mut self.current, &mut self.next);
        self.timings.snow += elapsed_s(t0);

        let t0 = mark();
        step_atmosphere_into(
            &self.current,
            &mut self.next,
            &self.atmosphere_params,
            &AtmoForcing {
                temp_params: &self.temperature_params,
                wind_params: &self.wind_params,
                wind_field: &self.wind_field,
                wind_mag: &self.wind_mag,
                hour_tick: self.hour_tick,
            },
            &mut self.precip_gate_open,
            &mut self.scratch_atmo,
            &mut self.scratch_precip_tick,
        );
        std::mem::swap(&mut self.current, &mut self.next);
        self.timings.atmosphere += elapsed_s(t0);

        // Accumulate tick precip in the daily accumulator.
        for (total, tick) in self
            .last_precipitation
            .iter_mut()
            .zip(self.scratch_precip_tick.iter())
        {
            total.rain += tick.rain;
            total.snow += tick.snow;
        }

        // Climate normals (#79): accumulate hourly state (T + water) and
        // absorbed insolation = beam x flux_factor (aspect + occlusion + cloud,
        // calculated at tick start, reused here, pre/post-atmo cloud difference
        // per tick negligible for annual aggregate). Read-only.
        let t0 = mark();
        self.climate_normals
            .record_tick(&self.current, &self.scratch_flux_factor, solar.beam);
        self.timings.normals += elapsed_s(t0);

        self.hour_tick += 1;
        self.timings.hours += 1;

        // Tier 3: once per day, at day boundary (hour_of_day transitions to 0
        // after 24 sub-ticks). At this moment `last_precipitation` holds the
        // complete sum of the 24 hours just elapsed.
        if time::hour_of_day(self.hour_tick) == 0 {
            self.step_daily_tail();
        }
    }

    /// Produce the tick's wind field, consumed thereafter by snow and
    /// atmosphere via `AtmoForcing`/`SnowForcing`. Reads temperature from
    /// `current` (post `step_temperature`).
    fn refresh_wind(&mut self) {
        // Synoptic dynamics (Phase 1): prognostic state evolved one hour out
        // of M (`SYNOPTIC_SUBSAMPLE_HOURS`, dominant cost of the tick,
        // subsample validated by climate ablation). Off by default -> zero cost.
        if self.synoptic_enabled && self.hour_tick.is_multiple_of(synoptic_subsample()) {
            let t0 = mark();
            self.synoptic_mesh.aggregate_temperature(&self.current);
            self.synoptic_state
                .step_hour(self.synoptic_mesh.grid(), &self.synoptic_params);
            self.synoptic_state
                .write_base_wind(&self.synoptic_params, &mut self.synoptic_coarse_base);
            self.synoptic_mesh
                .interpolate_wind(&self.synoptic_coarse_base, &mut self.synoptic_base);
            self.timings.synoptic += elapsed_s(t0);
        }

        // Wind subsample (#89): the wind field is only recomputed one hour
        // out of N (dominant components are day-scale, cf.
        // `WIND_SUBSAMPLE_HOURS`). `hour_tick % N == 0` recomputes at hour 0
        // (init already done in the constructor) then at a fixed cadence;
        // otherwise the previous hour's `wind_field` is reused by the
        // atmosphere and thermal advection.
        //
        // Test seam (#108, `set_uniform_wind`): if a uniform wind is forced,
        // it systematically takes precedence over the normal recompute, no
        // subsample, no noise/thermal/terrain/synoptic.
        if let Some(w) = self.uniform_wind {
            self.wind_field.fill(w);
            compute_wind_magnitudes_into(&self.wind_field, &mut self.wind_mag);
        } else if self.hour_tick.is_multiple_of(wind_subsample()) {
            let t0 = mark();
            let base = if self.synoptic_enabled {
                Some(&self.synoptic_base)
            } else {
                None
            };
            compute_wind_field_into(
                &self.current,
                &self.wind_params,
                self.hour_tick,
                &mut self.wind_field,
                &mut self.scratch_wind_snap,
                base,
            );
            // Magnitudes memoized at the field's cadence: consumed every hour
            // by evaporation, valid as long as the field doesn't change.
            compute_wind_magnitudes_into(&self.wind_field, &mut self.wind_mag);
            self.timings.wind += elapsed_s(t0);
        }
    }

    /// 8 MFD passes once a day, an integration budget to converge the
    /// surface water transfer (CFL ~1/7 per pass with `flow_rate=0.12`).
    /// **Not a time proxy**, cf. the doc on `HYDRO_MFD_PASSES_PER_DAY`. The
    /// symmetric MFD recomputes the topology on each pass.
    /// `discharge_map`/`flow_vec_map`/`edge_flux_map` are reset HERE, right
    /// before the passes, not at the start of the day: between two slices
    /// they keep the last flow, so `diag` and snapshots read at any hour see
    /// non-zero flux (#103; before, a misaligned world showed 0 outflow
    /// permanently).
    fn step_hydro_tranche(&mut self) {
        let n = self.current.len();
        self.discharge_map.resize(n, 0.0);
        self.discharge_map.fill(0.0);
        self.flow_vec_map.resize(n, (0.0, 0.0));
        self.flow_vec_map.fill((0.0, 0.0));
        self.edge_flux_map.resize(n, [0.0; 6]);
        self.edge_flux_map.fill([0.0; 6]);
        for _ in 0..HYDRO_MFD_PASSES_PER_DAY {
            step_hydro_mfd_into(
                &self.current,
                &mut self.next,
                &self.hydro_params,
                &mut self.scratch_flux,
                &mut self.scratch_flow_vec,
                &mut self.scratch_edge_flux,
            );
            for i in 0..n {
                self.discharge_map[i] += self.scratch_flux[i];
                self.flow_vec_map[i].0 += self.scratch_flow_vec[i].0;
                self.flow_vec_map[i].1 += self.scratch_flow_vec[i].1;
                for d in 0..6 {
                    self.edge_flux_map[i][d] += self.scratch_edge_flux[i][d];
                }
            }
            std::mem::swap(&mut self.current, &mut self.next);
        }
    }

    /// Hydrostatic lake leveling (#106), conservative. A flowing river stays
    /// below the threshold and keeps its dynamic gradient.
    fn level_lakes(&mut self) {
        step_lake_leveling(&self.current, &mut self.next, &self.lake_params);
        std::mem::swap(&mut self.current, &mut self.next);
    }

    fn step_daily_tail(&mut self) {
        let t0 = mark();
        self.climate_history
            .record_tick(&self.last_precipitation, &self.current);
        self.timings.history += elapsed_s(t0);

        let t0 = mark();
        step_groundwater(&self.current, &mut self.next, &self.groundwater_params);
        std::mem::swap(&mut self.current, &mut self.next);
        self.timings.groundwater += elapsed_s(t0);

        // Hydro in Tier 3 (8 MFD passes once a day after gw): coupling
        // preserved with the v0.2.x order (infiltration/resurgence then
        // runoff). The 8 passes are an MFD integration budget (CFL ~1/7),
        // not 8 hours of physics. An attempt to promote to Tier 2 (PR3)
        // broke mass conservation: the time decoupling between gw and hydro
        // introduces a cumulative drift above the strict 10-year tolerance.
        // Issue #48 tracks the options (recalibrate 24/day or a proper Tier 2
        // refactor); this PR only renames + clarifies.
        let t0 = mark();
        self.step_hydro_tranche();
        self.timings.hydro += elapsed_s(t0);

        // Hydro flux EMA (#105): daily smoothing of the slice that just
        // closed. Updated even with erosion off, stream power AND the
        // network render (#106) both consume the same history.
        let t0 = mark();
        update_ema(
            &mut self.discharge_ema,
            &self.discharge_map,
            self.erosion_params.tau_days,
        );
        update_edge_ema(
            &mut self.edge_flux_ema,
            &self.edge_flux_map,
            self.erosion_params.tau_days,
        );
        self.timings.ema += elapsed_s(t0);

        // Lake leveling (Tier 3, #106): after the MFD slice, every deep
        // free-water component regains a flat surface (MFD alone no longer
        // levels since the SI pass #104).
        let t0 = mark();
        self.level_lakes();
        self.timings.lakes += elapsed_s(t0);

        // River erosion (Tier 3, #105): stream power incision -> load ->
        // deposit, driven by the freshly updated EMA. Elevation is no longer
        // frozen: the surface normals (sunny/shaded slope aspect, #102) are
        // recomputed after an effective step, O(n), negligible next to the 8
        // MFD passes. `aspect_correction` (N×8760, expensive) stays frozen at
        // construction: the normals' drift is second-order on the annual map
        // average that it recalibrates.
        if self.erosion_params.enabled {
            let t0 = mark();
            let totals = step_erosion(
                &self.current,
                &mut self.next,
                &self.erosion_params,
                &ErosionForcing {
                    discharge_ema: &self.discharge_ema,
                    edge_flux_ema: &self.edge_flux_ema,
                },
            );
            std::mem::swap(&mut self.current, &mut self.next);
            if totals.incised_m > 0.0 || totals.deposited_m > 0.0 {
                compute_surface_normals(&mut self.current);
                // Elevation moved: the illumination's horizon tangents are
                // stale (#65), rebuild on the next tick.
                self.illum_cache.mark_dirty();
            }
            self.erosion_incised_total += f64::from(totals.incised_m);
            self.erosion_deposited_total += f64::from(totals.deposited_m);
            self.timings.erosion += elapsed_s(t0);
        }

        // Multi-species vegetation (Tier 3, after hydro, #81): evolves
        // biomass per species based on climate fitness (normals #79) and
        // available space. Biomass drives transpiration (Tier 1,
        // `step_evaporation`, FAO-56): that's what writes to the atmosphere,
        // not this step. Normals are read as-is (year N-1).
        let t0 = mark();
        step_vegetation(
            &self.current,
            &mut self.next,
            &self.vegetation_params,
            self.climate_normals.normals(),
        );
        std::mem::swap(&mut self.current, &mut self.next);
        self.timings.vegetation += elapsed_s(t0);

        // Fire (Tier 3, after vegetation: the fuel, biomass times age, is
        // known). Destroys biomass and injects heat; water stays intact
        // (drying emerges via evaporation on the next tick).
        let t0 = mark();
        let ft = step_fire(
            &self.current,
            &mut self.next,
            &self.fire_params,
            self.seed,
            time::ticks_to_days(self.hour_tick),
        );
        self.fire_ignitions_total += u64::from(ft.ignitions);
        self.fire_cell_days_total += u64::from(ft.burning);
        self.fire_peak_burning = self.fire_peak_burning.max(ft.burning);
        std::mem::swap(&mut self.current, &mut self.next);
        self.timings.fire += elapsed_s(t0);
    }

    #[must_use]
    pub fn snapshot(&self) -> GridState {
        // `discharge`/`edge_flux` come from the EMA (#105), not the day's
        // slice: the displayed network drifts with the seasons instead of
        // rearranging on every rainy day (#106 point 2). `flow_vec` stays
        // the instantaneous slice, no EMA exists for this field, out of
        // scope for #106 (it drives neither river nor lake in the render).
        let mut state = self.current.snapshot(
            time::ticks_to_days(self.hour_tick),
            self.hour_tick,
            &HydroMaps {
                discharge: &self.discharge_ema,
                flow_vec: &self.flow_vec_map,
                edge_flux: &self.edge_flux_ema,
            },
            &self.wind_field,
            &self.last_precipitation,
        );
        // Synoptic fields (Phase 2): filled here, not in `grid.snapshot`, the
        // synoptic state lives in the sim, the grid doesn't know about the
        // dynamics. Total wind `(u + U, v)` in m/s: the base that the wind
        // pipeline consumes when `synoptic.enabled` (source of truth on the
        // core side, the front-end displays it blindly, anti-pattern #2).
        // The state lives on the coarse torus: sampled per fine cell with the
        // same barycentric weights as the base wind.
        let (u, v) = self.synoptic_state.velocity();
        let h = self.synoptic_state.height();
        let u0 = self.synoptic_params.mean_flow_ms;
        for (i, cell) in state.cells.iter_mut().enumerate() {
            cell.synoptic_h = self.synoptic_mesh.sample_scalar(h, i);
            cell.synoptic_u = self.synoptic_mesh.sample_scalar(u, i) + u0;
            cell.synoptic_v = self.synoptic_mesh.sample_scalar(v, i);
        }
        // Illumination (#102): display factor [0,1] per cell (aspect x
        // occlusion x cloud shadow), computed by `compute_illumination` this
        // tick. Lives in the sim; the front-end colors the albedo with it,
        // no recompute (#2).
        for (cell, &illum) in state.cells.iter_mut().zip(self.scratch_illumination.iter()) {
            cell.illumination = illum;
        }
        state
    }

    #[must_use]
    pub fn diagnostics(&self) -> Diagnostics {
        let mut diag = compute_diagnostics(
            &self.current,
            time::ticks_to_days(self.hour_tick),
            &self.discharge_map,
            &self.wind_field,
            &self.last_precipitation,
            &self.atmosphere_params,
        );
        diag.synoptic = Some(
            self.synoptic_state
                .stats(&self.synoptic_params, self.synoptic_enabled),
        );
        diag.erosion = Some(crate::diagnostics::ErosionStats {
            enabled: self.erosion_params.enabled,
            gini_discharge_ema: crate::erosion::discharge_gini(&self.discharge_ema),
            sediment_in_transit_m: self
                .current
                .cells_slice()
                .iter()
                .map(|c| f64::from(c.sediment_load))
                .sum(),
            incised_total_m: self.erosion_incised_total,
            deposited_total_m: self.erosion_deposited_total,
            closed_depressions: crate::erosion::closed_depression_indices(&self.current).len(),
        });
        diag
    }

    #[must_use]
    pub fn climate_history(&self) -> &ClimateHistory {
        &self.climate_history
    }

    /// Wall-clock cumulative totals per real tick phase (cf.
    /// [`PhaseTimings`]). Always zero on wasm32.
    #[must_use]
    pub fn phase_timings(&self) -> &PhaseTimings {
        &self.timings
    }

    /// Resets the phase counters to zero (windowing a measurement).
    pub fn reset_phase_timings(&mut self) {
        self.timings = PhaseTimings::default();
    }

    /// Climate normals per cell (#79), indexed like `grid().cells_slice()`.
    /// Last complete year; default values as long as no year has finished
    /// (cf. `climate_normals_ready`).
    #[must_use]
    pub fn climate_normals(&self) -> &[CellClimateNormals] {
        self.climate_normals.normals()
    }

    /// `true` as soon as at least one year has been simulated (normals
    /// available).
    #[must_use]
    pub fn climate_normals_ready(&self) -> bool {
        self.climate_normals.has_normals()
    }

    #[must_use]
    pub fn last_precipitation(&self) -> &PrecipitationMap {
        &self.last_precipitation
    }

    #[must_use]
    pub fn grid(&self) -> &HexGrid {
        &self.current
    }

    /// Day counter elapsed since creation (v0.2.x API compat). Derived from
    /// `hour_tick / TICKS_PER_DAY`, the current hour of day isn't visible
    /// through it. Use `hour_tick()`, `day_of_year()` or `hour_of_day()` for
    /// hourly resolution.
    #[must_use]
    pub fn tick(&self) -> u64 {
        time::ticks_to_days(self.hour_tick)
    }

    /// Cumulative hour counter since creation (v0.3.0, issue #38).
    /// 1 unit = 1 hour. Exposes the sub-tick temporal resolution for
    /// consumers that want to trace dawn/dusk tick by tick.
    #[must_use]
    pub fn hour_tick(&self) -> u64 {
        self.hour_tick
    }

    /// Day of year [0, 364] for the current tick.
    #[must_use]
    pub fn day_of_year(&self) -> u16 {
        time::day_of_year(self.hour_tick)
    }

    /// Local hour [0, 23] for the current tick. In PR1 always 0 when
    /// observed from the external API (`step()` advances by 24 hours).
    #[must_use]
    pub fn hour_of_day(&self) -> u8 {
        time::hour_of_day(self.hour_tick)
    }

    #[must_use]
    pub fn discharge_map(&self) -> &DischargeMap {
        &self.discharge_map
    }

    #[must_use]
    pub fn flow_vec_map(&self) -> &FlowVecMap {
        &self.flow_vec_map
    }

    /// Outgoing flux per edge (order `coord::DIRECTIONS`), accumulated over
    /// the last daily hydro slice (#103). `Σ_dir == discharge_map[i]` up to
    /// f32 epsilon.
    #[must_use]
    pub fn edge_flux_map(&self) -> &EdgeFluxMap {
        &self.edge_flux_map
    }

    /// EMA of daily discharge (mm/day, τ = `erosion.tau_days`), the flux
    /// history that drives stream power (#105) and the stabilized network
    /// rendering consumed by `snapshot` (#106).
    #[must_use]
    pub fn discharge_ema_map(&self) -> &DischargeEmaMap {
        &self.discharge_ema
    }

    /// EMA of flux per edge, same cadence as [`discharge_ema_map`](Self::discharge_ema_map).
    #[must_use]
    pub fn edge_flux_ema_map(&self) -> &EdgeFluxEmaMap {
        &self.edge_flux_ema
    }

    /// Cumulative erosion counters since creation: (bedrock incised,
    /// load redeposited), in m accumulated over the map.
    #[must_use]
    pub fn erosion_totals(&self) -> (f64, f64) {
        (self.erosion_incised_total, self.erosion_deposited_total)
    }

    /// Total ascent `w = H·(−∇·v) + v·∇z` (m/s) per cell, buffer for the
    /// ascent trigger (synoptic Phase 3). Empty if `updraft_ref_ms = 0`
    /// (the scratch is only filled when the trigger is active). Diagnostic
    /// only.
    #[must_use]
    pub fn updraft_field(&self) -> &[f32] {
        &self.scratch_atmo.convergence
    }

    #[must_use]
    pub fn wind_field(&self) -> &WindField {
        &self.wind_field
    }

    /// Synoptic state (Phase 1), geopotential height / prognostic wind.
    /// Exposed for diagnostics (isobars, L/H). Read-only.
    #[must_use]
    pub fn synoptic_state(&self) -> &SynopticState {
        &self.synoptic_state
    }

    /// Does synoptic dynamics drive the wind (param `synoptic.enabled`,
    /// hardcoded ON by default)?
    #[must_use]
    pub fn synoptic_enabled(&self) -> bool {
        self.synoptic_enabled
    }

    /// Test/diagnostic seam: forces a uniform surface wind field, short-
    /// circuiting the whole pipeline (noise/thermal/relief/synoptic) and
    /// disabling synoptic dynamics. Replaces the old `west_bias=… +
    /// synoptic.enabled=0` fixture from the advection tests. Upper-level
    /// wind is still derived from this uniform field (rotation + ratio) in
    /// the atmosphere pass.
    pub fn set_uniform_wind(&mut self, wind: WindVec) {
        self.uniform_wind = Some(wind);
        self.synoptic_enabled = false;
        self.wind_field.fill(wind);
        compute_wind_magnitudes_into(&self.wind_field, &mut self.wind_mag);
    }

    #[must_use]
    pub fn atmosphere_params(&self) -> &AtmosphereParams {
        &self.atmosphere_params
    }

    #[must_use]
    pub fn hydro_params(&self) -> &HydroParams {
        &self.hydro_params
    }

    #[must_use]
    pub fn groundwater_params(&self) -> &GroundwaterParams {
        &self.groundwater_params
    }

    #[must_use]
    pub fn snow_params(&self) -> &SnowParams {
        &self.snow_params
    }

    #[must_use]
    pub fn temperature_params(&self) -> &TemperatureParams {
        &self.temperature_params
    }

    #[must_use]
    pub fn wind_params(&self) -> &WindParams {
        &self.wind_params
    }

    #[must_use]
    pub fn vegetation_params(&self) -> &VegetationParams {
        &self.vegetation_params
    }

    #[must_use]
    pub fn synoptic_params(&self) -> &SynopticParams {
        &self.synoptic_params
    }

    #[must_use]
    pub fn erosion_params(&self) -> &ErosionParams {
        &self.erosion_params
    }

    /// Replaces the erosion parameters (preserved on reset, like fire).
    pub fn set_erosion_params(&mut self, params: ErosionParams) {
        self.erosion_params = params;
    }

    #[must_use]
    pub fn lake_params(&self) -> &LakeParams {
        &self.lake_params
    }

    /// Replaces the lake leveling parameters (preserved on reset).
    pub fn set_lake_params(&mut self, params: LakeParams) {
        self.lake_params = params;
    }

    #[must_use]
    pub fn fire_params(&self) -> &FireParams {
        &self.fire_params
    }

    /// Cumulative fire metrics, for calibration (#wildfire).
    #[must_use]
    pub fn fire_stats(&self) -> FireStats {
        let currently_burning = self
            .current
            .cells_slice()
            .iter()
            .filter(|c| c.fire_intensity > 1e-3)
            .count();
        FireStats {
            ignitions_total: self.fire_ignitions_total,
            cell_days_total: self.fire_cell_days_total,
            peak_burning: self.fire_peak_burning,
            currently_burning,
        }
    }

    /// Sets the fire randomness seed (determinism "1 seed = 1 world").
    pub fn set_seed(&mut self, seed: u32) {
        self.seed = seed;
    }

    /// Replaces the fire parameters (preserved on reset, like the others).
    pub fn set_fire_params(&mut self, params: FireParams) {
        self.fire_params = params;
    }

    /// Updates a simulation parameter by "group.field" key.
    /// Returns true if the key is recognized.
    pub fn update_param(&mut self, key: &str, value: f32) -> bool {
        if key.starts_with("atmosphere.") {
            return self.set_atmosphere_param(key, value);
        }
        if key.starts_with("erosion.") {
            return self.set_erosion_param(key, value);
        }
        match key {
            // Hydrology
            "hydro.flow_rate" => self.hydro_params.flow_rate = value,
            "hydro.slope_full_mobility" => self.hydro_params.slope_full_mobility = value,
            "hydro.flow_concentration" => self.hydro_params.flow_concentration = value,
            // Lake leveling (#106)
            "lake.enabled" => self.lake_params.enabled = value != 0.0,
            "lake.min_surplus_mm" => self.lake_params.min_surplus_mm = value,
            // Groundwater
            "groundwater.infiltration_rate" => {
                self.groundwater_params.infiltration_rate = value;
            }
            "groundwater.diffusion_rate" => self.groundwater_params.diffusion_rate = value,
            "groundwater.max_capacity" => self.groundwater_params.max_capacity = value,
            "groundwater.baseflow_coef" => self.groundwater_params.baseflow_coef = value,
            // Snow
            "snow.snow_albedo_dry" => self.snow_params.snow_albedo_dry = value,
            "snow.snow_albedo_melt" => self.snow_params.snow_albedo_melt = value,
            "snow.snow_emissivity" => self.snow_params.snow_emissivity = value,
            "snow.sensible_exchange_coef" => self.snow_params.sensible_exchange_coef = value,
            "snow.free_convection_wind_ms" => self.snow_params.free_convection_wind_ms = value,
            "snow.snow_masking_half_mm" => self.snow_params.snow_masking_half_mm = value,
            "snow.freeze_threshold" => self.snow_params.freeze_threshold = value,
            "snow.melt_recharge_frac" => self.snow_params.melt_recharge_frac = value,
            // Temperature
            "temperature.base_temp" => self.temperature_params.base_temp = value,
            "temperature.lapse_rate" => self.temperature_params.lapse_rate = value,
            "temperature.water_cooling" => self.temperature_params.water_cooling = value,
            "temperature.thermal_coupling" => self.temperature_params.thermal_coupling = value,
            "temperature.latitude_deg" => self.temperature_params.latitude_deg = value,
            "temperature.cloud_albedo_coef" => {
                self.temperature_params.cloud_albedo_coef = value;
            }
            "temperature.atmospheric_transmittance" => {
                self.temperature_params.atmospheric_transmittance = value;
            }
            "temperature.ground_albedo" => {
                self.temperature_params.ground_albedo = value;
            }
            // Wind
            "wind.noise_direction_amplitude" => {
                self.wind_params.noise_direction_amplitude = value;
            }
            "wind.noise_strength_amplitude" => {
                self.wind_params.noise_strength_amplitude = value;
            }
            "wind.noise_time_scale" => self.wind_params.noise_time_scale = value,
            "wind.thermal_strength" => self.wind_params.thermal_strength = value,
            "wind.terrain_deflection" => self.wind_params.terrain_deflection = value,
            "wind.terrain_speed_factor" => self.wind_params.terrain_speed_factor = value,
            "wind.humidity_advection_rate" => self.wind_params.humidity_advection_rate = value,
            "wind.temperature_advection_rate" => {
                self.wind_params.temperature_advection_rate = value;
            }
            "wind.wind_upper_rotation_deg" => {
                self.wind_params.wind_upper_rotation_deg = value;
            }
            "wind.wind_upper_speed_ratio" => {
                self.wind_params.wind_upper_speed_ratio = value;
            }
            // Synoptic dynamics (Phase 1). `deformation_radius_cells` is NOT
            // hot-reloadable: it fixes H = h₀, frozen at state init
            // (changing it requires a reset). The others are safe at runtime.
            "synoptic.enabled" => self.synoptic_enabled = value != 0.0,
            "synoptic.mean_flow_ms" => self.synoptic_params.mean_flow_ms = value,
            "synoptic.thermal_anomaly_days" => {
                self.synoptic_params.thermal_anomaly_days = value;
            }
            "synoptic.thermal_coupling" => self.synoptic_params.thermal_coupling = value,
            "synoptic.viscosity" => self.synoptic_params.viscosity = value,
            "synoptic.friction_days" => self.synoptic_params.friction_days = value,
            "synoptic.relax_days" => self.synoptic_params.relax_days = value,
            // Vegetation (transition to SI, finalized by #77)
            "vegetation.growth_rate" => self.vegetation_params.growth_rate = value,
            "vegetation.colonization_rate" => self.vegetation_params.colonization_rate = value,
            "vegetation.base_mortality" => self.vegetation_params.base_mortality = value,
            "vegetation.lethal_mortality" => self.vegetation_params.lethal_mortality = value,
            "vegetation.succession_rate" => self.vegetation_params.succession_rate = value,
            "vegetation.k_total" => self.vegetation_params.k_total = value,
            "vegetation.open_water_excess" => self.vegetation_params.open_water_excess = value,
            // Fire (#wildfire). `fire.enabled`: 0 = off, otherwise on.
            "fire.enabled" => self.fire_params.enabled = value != 0.0,
            "fire.ignition_rate" => self.fire_params.ignition_rate = value,
            "fire.spread_rate" => self.fire_params.spread_rate = value,
            "fire.moisture_ref_mm" => self.fire_params.moisture_ref_mm = value,
            "fire.temp_ignite_lo" => self.fire_params.temp_ignite_lo = value,
            "fire.temp_ignite_hi" => self.fire_params.temp_ignite_hi = value,
            "fire.fuel_age_half_years" => self.fire_params.fuel_age_half_years = value,
            "fire.combustion_fraction_per_day" => {
                self.fire_params.combustion_fraction_per_day = value;
            }
            "fire.extinguish_fuel_min" => self.fire_params.extinguish_fuel_min = value,
            "fire.fuel_load_kg_per_m2" => self.fire_params.fuel_load_kg_per_m2 = value,
            "fire.combustion_heat_ground_fraction" => {
                self.fire_params.combustion_heat_ground_fraction = value;
            }
            _ => return false,
        }
        true
    }

    /// Applies an `erosion.*` parameter (#105). Extracted from `update_param`
    /// to keep it readable (like `set_atmosphere_param`).
    fn set_erosion_param(&mut self, key: &str, value: f32) -> bool {
        match key {
            "erosion.enabled" => self.erosion_params.enabled = value != 0.0,
            "erosion.k_incision" => self.erosion_params.k_incision = value,
            "erosion.k_transport" => self.erosion_params.k_transport = value,
            "erosion.m_exponent" => self.erosion_params.m_exponent = value,
            "erosion.n_exponent" => self.erosion_params.n_exponent = value,
            "erosion.tau_days" => self.erosion_params.tau_days = value,
            "erosion.accel_years_per_day" => self.erosion_params.accel_years_per_day = value,
            "erosion.cfl_drop_frac" => self.erosion_params.cfl_drop_frac = value,
            _ => return false,
        }
        true
    }

    fn set_atmosphere_param(&mut self, key: &str, value: f32) -> bool {
        match key {
            "atmosphere.transpiration_coef" => self.atmosphere_params.transpiration_coef = value,
            // Ascent trigger + critical mass (synoptic Phase 3).
            "atmosphere.updraft_ref_ms" => self.atmosphere_params.updraft_ref_ms = value,
            "atmosphere.updraft_floor" => self.atmosphere_params.updraft_floor = value,
            "atmosphere.precip_crit_mm" => self.atmosphere_params.precip_crit_mm = value,
            "atmosphere.condensation_rate" => {
                self.atmosphere_params.condensation_rate = value;
            }
            "atmosphere.cloud_evap_hr_threshold" => {
                self.atmosphere_params.cloud_evap_hr_threshold = value;
            }
            "atmosphere.cloud_evap_rate" => {
                self.atmosphere_params.cloud_evap_rate = value;
            }
            "atmosphere.kk2000_droplet_count" => {
                self.atmosphere_params.kk2000_droplet_count = value;
            }
            "atmosphere.cloud_diffusion_rate" => {
                self.atmosphere_params.cloud_diffusion_rate = value;
            }
            "atmosphere.precip_neighbor_share" => {
                self.atmosphere_params.precip_neighbor_share = value;
            }
            "atmosphere.max_precip_per_tick" => {
                self.atmosphere_params.max_precip_per_tick = value;
            }
            "atmosphere.fog_condensation_threshold" => {
                self.atmosphere_params.fog_condensation_threshold = value;
            }
            "atmosphere.fog_condensation_rate" => {
                self.atmosphere_params.fog_condensation_rate = value;
            }
            "atmosphere.sublimation_rate" => self.atmosphere_params.sublimation_rate = value,
            "atmosphere.uplift_rate" => self.atmosphere_params.uplift_rate = value,
            "atmosphere.uplift_thermal_coef" => {
                self.atmosphere_params.uplift_thermal_coef = value;
            }
            "atmosphere.upper_layer_altitude_m" => {
                self.atmosphere_params.upper_layer_altitude_m = value;
            }
            "atmosphere.global_precip_gate" => {
                self.atmosphere_params.global_precip_gate = value;
            }
            "atmosphere.initial_humidity_floor" => {
                self.atmosphere_params.initial_humidity_floor = value;
            }
            "atmosphere.orographic_lift_coef" => {
                self.atmosphere_params.orographic_lift_coef = value;
            }
            _ => return false,
        }
        true
    }
}

impl Simulation {
    /// Serializes the full simulation state to `MessagePack` (see
    /// [`crate::checkpoint`]). The blob can be reloaded via
    /// [`Simulation::load_state`] to resume the simulation **identically**;
    /// bit-identical resumption is proven by test.
    ///
    /// # Errors
    /// Returns [`CheckpointError::Encode`] if `MessagePack` serialization
    /// fails, which doesn't happen on a valid simulation state, but the API
    /// stays honest rather than masking the failure with an `unwrap`.
    pub fn save_state(&self) -> Result<Vec<u8>, CheckpointError> {
        let checkpoint = Checkpoint {
            magic: MAGIC.to_string(),
            format_version: CHECKPOINT_FORMAT_VERSION,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            grid: self.current.clone(),
            hour_tick: self.hour_tick,
            seed: self.seed,
            fire_ignitions_total: self.fire_ignitions_total,
            fire_cell_days_total: self.fire_cell_days_total,
            fire_peak_burning: self.fire_peak_burning,
            discharge_map: self.discharge_map.clone(),
            flow_vec_map: self.flow_vec_map.clone(),
            edge_flux_map: self.edge_flux_map.clone(),
            discharge_ema: self.discharge_ema.clone(),
            edge_flux_ema: self.edge_flux_ema.clone(),
            erosion_incised_total: self.erosion_incised_total,
            erosion_deposited_total: self.erosion_deposited_total,
            wind_field: self.wind_field.clone(),
            wind_mag: self.wind_mag.clone(),
            synoptic_params: self.synoptic_params.clone(),
            synoptic_state: self.synoptic_state.clone(),
            synoptic_enabled: self.synoptic_enabled,
            synoptic_base: self.synoptic_base.clone(),
            synoptic_coarse_radius: self.synoptic_mesh.grid().radius(),
            climate_history: self.climate_history.clone(),
            last_precipitation: self.last_precipitation.clone(),
            precip_gate_open: self.precip_gate_open,
            climate_normals: self.climate_normals.clone(),
            hydro_params: self.hydro_params.clone(),
            atmosphere_params: self.atmosphere_params.clone(),
            groundwater_params: self.groundwater_params.clone(),
            snow_params: self.snow_params.clone(),
            temperature_params: self.temperature_params.clone(),
            wind_params: self.wind_params.clone(),
            vegetation_params: self.vegetation_params.clone(),
            fire_params: self.fire_params,
            erosion_params: self.erosion_params.clone(),
            lake_params: self.lake_params.clone(),
        };
        checkpoint.encode()
    }

    /// Rebuilds a simulation from a blob produced by
    /// [`Simulation::save_state`]. The authoritative state is restored verbatim;
    /// derived fields (double-buffer `next`, scratch buffers) are
    /// rebuilt on the fly, never depended on from the file.
    ///
    /// # Errors
    /// Returns [`CheckpointError`] if the blob isn't a valid `HexSim`
    /// checkpoint ([`CheckpointError::Decode`] / [`CheckpointError::BadMagic`])
    /// or has an incompatible format version ([`CheckpointError::Version`]).
    pub fn load_state(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let ckpt = Checkpoint::decode(bytes)?;
        let current = ckpt.grid;
        let n = current.len();
        // Field absent from pre-#103 v2 checkpoints (`serde(default)`): empty
        // map -> sized to the grid, filled on the next hydro slice.
        let mut edge_flux_map = ckpt.edge_flux_map;
        edge_flux_map.resize(n, [0.0; 6]);
        // Same contract for pre-#105 EMAs: empty -> sized, the EMA
        // refills over ~3τ (warm-up assumed, see `erosion.rs`).
        let mut discharge_ema = ckpt.discharge_ema;
        discharge_ema.resize(n, 0.0);
        let mut edge_flux_ema = ckpt.edge_flux_ema;
        edge_flux_ema.resize(n, [0.0; 6]);
        // `next` is a double-buffer: it must mirror `current` before each
        // phase (exact parity with `Simulation::new`, which does `grid.clone()`).
        let next = current.clone();
        // Mesh rebuilt at the PERSISTED radius (not the current env's): the
        // verbatim-restored synoptic state stays aligned with its torus.
        let mut synoptic_mesh =
            SynopticMesh::with_coarse_radius(&current, ckpt.synoptic_coarse_radius);
        synoptic_mesh.aggregate_temperature(&current);
        let mut synoptic_coarse_base: WindField =
            vec![WindVec::default(); synoptic_mesh.grid().len()];
        ckpt.synoptic_state
            .write_base_wind(&ckpt.synoptic_params, &mut synoptic_coarse_base);
        Ok(Self {
            current,
            next,
            hour_tick: ckpt.hour_tick,
            hydro_params: ckpt.hydro_params,
            atmosphere_params: ckpt.atmosphere_params,
            groundwater_params: ckpt.groundwater_params,
            snow_params: ckpt.snow_params,
            temperature_params: ckpt.temperature_params,
            wind_params: ckpt.wind_params,
            vegetation_params: ckpt.vegetation_params,
            fire_params: ckpt.fire_params,
            seed: ckpt.seed,
            fire_ignitions_total: ckpt.fire_ignitions_total,
            fire_cell_days_total: ckpt.fire_cell_days_total,
            fire_peak_burning: ckpt.fire_peak_burning,
            discharge_map: ckpt.discharge_map,
            flow_vec_map: ckpt.flow_vec_map,
            edge_flux_map,
            erosion_params: ckpt.erosion_params,
            lake_params: ckpt.lake_params,
            discharge_ema,
            edge_flux_ema,
            erosion_incised_total: ckpt.erosion_incised_total,
            erosion_deposited_total: ckpt.erosion_deposited_total,
            wind_field: ckpt.wind_field,
            wind_mag: ckpt.wind_mag,
            uniform_wind: None,
            synoptic_params: ckpt.synoptic_params,
            synoptic_state: ckpt.synoptic_state,
            synoptic_enabled: ckpt.synoptic_enabled,
            synoptic_base: ckpt.synoptic_base,
            synoptic_mesh,
            synoptic_coarse_base,
            climate_history: ckpt.climate_history,
            last_precipitation: ckpt.last_precipitation,
            precip_gate_open: ckpt.precip_gate_open,
            scratch_wind_snap: vec![WindVec::default(); n],
            scratch_atmo: AtmoScratch::new(n),
            scratch_flux: vec![0.0; n],
            scratch_flow_vec: vec![(0.0, 0.0); n],
            scratch_edge_flux: vec![[0.0; 6]; n],
            scratch_precip_tick: vec![DayRecord::default(); n],
            scratch_flux_factor: vec![0.0; n],
            scratch_illumination: vec![1.0; n],
            illum_cache: IllumCache::new(),
            climate_normals: ckpt.climate_normals,
            timings: PhaseTimings::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{TerrainParams, generate_terrain};

    fn default_sim(radius: i32) -> Simulation {
        let grid = HexGrid::from_radius(radius);
        Simulation::new(
            grid,
            HydroParams::default(),
            AtmosphereParams::default(),
            GroundwaterParams::default(),
            SnowParams::default(),
            TemperatureParams::default(),
            WindParams::default(),
        )
    }

    fn sim_with_terrain(radius: i32, seed: u32) -> Simulation {
        let mut grid = HexGrid::from_radius(radius);
        generate_terrain(
            &mut grid,
            &TerrainParams {
                seed,
                ..TerrainParams::default()
            },
        );
        Simulation::new(
            grid,
            HydroParams::default(),
            AtmosphereParams::default(),
            GroundwaterParams::default(),
            SnowParams::default(),
            TemperatureParams::default(),
            WindParams {
                seed,
                ..WindParams::default()
            },
        )
    }

    /// The core of step 1: `save_state` -> `load_state` -> continuation
    /// **bit-identical**. Saves at a non-aligned instant (mid-day, mid-year)
    /// to exercise all the hidden state: prognostic synoptic, in-progress
    /// yearly normals accumulator, intra-day flux maps, precipitation
    /// hysteresis, retained subsampled wind field. If just one of these
    /// fields weren't restored, the grid would diverge within a few hours via
    /// the evaporation/wind/precipitation chain.
    #[test]
    fn checkpoint_restart_is_bit_identical() {
        let mut a = sim_with_terrain(6, 42);
        // Force synoptic ON (already the hardcoded default since #108, set
        // explicitly so the prognostic state is part of the tested
        // round-trip, independent of any future default change).
        a.update_param("synoptic.enabled", 1.0);

        // 20 days + 7 h: instant not aligned on a day/year boundary.
        for _ in 0..(20 * 24 + 7) {
            a.step_hour();
        }

        let bytes = a.save_state().expect("save_state must not fail");
        let mut b = Simulation::load_state(&bytes).expect("load_state of a valid blob");
        assert_eq!(a.hour_tick(), b.hour_tick(), "restored clock");

        // Identical continuation on both sides.
        for _ in 0..(3 * 24 + 5) {
            a.step_hour();
            b.step_hour();
        }

        // Strong, order-stable comparison (Vec of cells, not a HashMap whose
        // iteration order is non-deterministic): all per-cell physics must
        // be bit-identical. `CellProperties` doesn't implement `PartialEq`,
        // so we compare via `MessagePack` encoding, which is deterministic.
        let cells_a = rmp_serde::to_vec(a.grid().cells_slice()).expect("encode cells a");
        let cells_b = rmp_serde::to_vec(b.grid().cells_slice()).expect("encode cells b");
        assert_eq!(
            cells_a, cells_b,
            "grid diverged after restart: a hidden state field was not restored"
        );
    }

    /// A blob that isn't a `HexSim` checkpoint must be rejected cleanly,
    /// never silently misinterpreted.
    #[test]
    fn load_state_rejects_foreign_bytes() {
        let result = Simulation::load_state(b"this is not a checkpoint");
        assert!(result.is_err(), "a foreign blob must be rejected");
    }

    #[test]
    fn tick_increments() {
        let mut sim = default_sim(2);
        assert_eq!(sim.tick(), 0);
        sim.step();
        assert_eq!(sim.tick(), 1);
        sim.step();
        assert_eq!(sim.tick(), 2);
    }

    #[test]
    fn snapshot_returns_current_tick() {
        let mut sim = default_sim(1);
        sim.step();
        sim.step();
        sim.step();
        let snap = sim.snapshot();
        assert_eq!(snap.tick, 3);
    }

    /// #106 point 2: the displayed network should drift with the seasons, not
    /// rearrange itself with every rain slice. Pins that `snapshot` sources
    /// `outflow_flux`/`edge_flux` from the EMA (#105, τ=60d), never from the
    /// daily slice; otherwise an isolated instantaneous flow makes the
    /// network flicker in `play` instead of drifting slowly.
    #[test]
    fn snapshot_flux_uses_ema_not_daily_tranche() {
        let mut sim = sim_with_terrain(6, 42);
        for _ in 0..(3 * 24) {
            sim.step_hour();
        }

        let discharge_map = sim.discharge_map().clone();
        let discharge_ema = sim.discharge_ema_map().clone();
        assert!(
            discharge_map.iter().any(|&d| d > 0.0),
            "no outflow after 3 days, fixture not meaningful"
        );

        let snap = sim.snapshot();
        for (i, cell) in snap.cells.iter().enumerate() {
            assert!(
                (cell.outflow_flux - discharge_ema[i]).abs() < 1e-6,
                "cell {i}: outflow_flux={} should follow discharge_ema={}, not discharge_map={}",
                cell.outflow_flux,
                discharge_ema[i],
                discharge_map[i]
            );
        }

        // τ=60d by default: after only 3 days the EMA should still lag well
        // behind the instantaneous value, otherwise the test doesn't
        // distinguish EMA from raw.
        let sum_map: f64 = discharge_map.iter().map(|&v| f64::from(v)).sum();
        let sum_ema: f64 = discharge_ema.iter().map(|&v| f64::from(v)).sum();
        assert!(
            sum_ema < sum_map * 0.5,
            "EMA should be damped vs raw slice after 3 days (map={sum_map}, ema={sum_ema})"
        );
    }

    #[test]
    fn update_param_sets_each_group() {
        // 1 representative key per group: the setter must be reflected
        // by the corresponding getter (round-trip).
        let mut sim = default_sim(1);

        assert!(sim.update_param("atmosphere.uplift_rate", 0.123));
        assert!((sim.atmosphere_params().uplift_rate - 0.123).abs() < 1e-6);

        assert!(sim.update_param("hydro.flow_rate", 0.456));
        assert!((sim.hydro_params().flow_rate - 0.456).abs() < 1e-6);

        assert!(sim.update_param("groundwater.max_capacity", 7.5));
        assert!((sim.groundwater_params().max_capacity - 7.5).abs() < 1e-6);

        assert!(sim.update_param("snow.snow_albedo_dry", 0.7));
        assert!((sim.snow_params().snow_albedo_dry - 0.7).abs() < 1e-6);

        assert!(sim.update_param("temperature.base_temp", 15.0));
        assert!((sim.temperature_params().base_temp - 15.0).abs() < 1e-6);

        assert!(sim.update_param("wind.thermal_strength", 0.8));
        assert!((sim.wind_params().thermal_strength - 0.8).abs() < 1e-6);

        assert!(sim.update_param("vegetation.growth_rate", 0.33));
        assert!((sim.vegetation_params().growth_rate - 0.33).abs() < 1e-6);

        assert!(sim.update_param("erosion.accel_years_per_day", 50.0));
        assert!((sim.erosion_params().accel_years_per_day - 50.0).abs() < 1e-6);
        assert!(sim.update_param("erosion.enabled", 0.0));
        assert!(!sim.erosion_params().enabled);
    }

    #[test]
    fn update_param_unknown_key_returns_false() {
        let mut sim = default_sim(1);
        assert!(!sim.update_param("not.a.key", 0.0));
        assert!(!sim.update_param("atmosphere.unknown", 0.0));
        assert!(!sim.update_param("", 0.0));
    }

    #[test]
    fn pipeline_produces_no_nan() {
        // Radius 3, default terrain, 50 ticks: no property should
        // become NaN or Inf. Guard against a numerical glitch
        // in any step of the pipeline.
        let mut sim = sim_with_terrain(3, 42);
        for _ in 0..50 {
            sim.step();
        }
        for (coord, cell) in sim.grid().iter() {
            assert!(
                cell.water_level.is_finite(),
                "water_level NaN at {coord:?}: {}",
                cell.water_level
            );
            assert!(
                cell.humidity_surface.is_finite() && cell.humidity_upper.is_finite(),
                "humidity NaN at {coord:?}: surface={} upper={}",
                cell.humidity_surface,
                cell.humidity_upper
            );
            assert!(
                cell.groundwater.is_finite(),
                "groundwater NaN at {coord:?}: {}",
                cell.groundwater
            );
            assert!(
                cell.snow_level.is_finite(),
                "snow_level NaN at {coord:?}: {}",
                cell.snow_level
            );
            assert!(
                cell.temperature.is_finite(),
                "temperature NaN at {coord:?}: {}",
                cell.temperature
            );
            let veg_total: f32 = cell.vegetation.iter().sum();
            assert!(
                cell.vegetation.iter().all(|v| v.is_finite() && *v >= 0.0)
                    && veg_total <= 1.0 + 1e-3,
                "vegetation hors borne at {coord:?}: {:?} (total {veg_total})",
                cell.vegetation
            );
        }
    }

    #[test]
    fn simulation_is_reproducible_given_seed() {
        // Two sims built with the same seeds should produce totals very
        // close to each other after N ticks. 1% tolerance: `HashMap`
        // iteration order (default RandomState) varies between two
        // instances, which introduces drift from non-commutative floating-
        // point summations. A drift > 1% would signal actual
        // non-determinism (e.g. `rand::thread_rng` in the sim).
        let mut sim_a = sim_with_terrain(3, 42);
        let mut sim_b = sim_with_terrain(3, 42);
        for _ in 0..30 {
            sim_a.step();
            sim_b.step();
        }
        let snap_a = sim_a.snapshot();
        let snap_b = sim_b.snapshot();

        let rel_drift = |a: f32, b: f32| (a - b).abs() / a.abs().max(b.abs()).max(1.0);
        assert!(
            rel_drift(snap_a.total_surface_water, snap_b.total_surface_water) < 0.01,
            "surface A={} B={}",
            snap_a.total_surface_water,
            snap_b.total_surface_water
        );
        assert!(
            rel_drift(snap_a.total_humidity, snap_b.total_humidity) < 0.01,
            "humidity A={} B={}",
            snap_a.total_humidity,
            snap_b.total_humidity
        );
        assert!(
            rel_drift(snap_a.total_groundwater, snap_b.total_groundwater) < 0.01,
            "gw A={} B={}",
            snap_a.total_groundwater,
            snap_b.total_groundwater
        );
    }

    #[test]
    fn climate_normals_ready_after_one_year() {
        // #79: before 1 year, no normals; after, they're consistent.
        let mut sim = sim_with_terrain(3, 42);
        assert!(!sim.climate_normals_ready());
        // 366 days: the year rollover (finalize) triggers at the start of
        // the 366th step (hour_tick == TICKS_PER_YEAR).
        for _ in 0..366 {
            sim.step();
        }
        assert!(sim.climate_normals_ready());

        let normals = sim.climate_normals();
        assert_eq!(normals.len(), sim.grid().len());
        for n in normals {
            assert!(
                n.t_mean.is_finite() && n.t_min.is_finite() && n.t_max.is_finite(),
                "normales T non finies : {n:?}"
            );
            assert!(n.t_min <= n.t_mean + 1e-3 && n.t_mean <= n.t_max + 1e-3);
            assert!(
                n.moisture_min <= n.moisture_mean + 1e-3
                    && n.moisture_mean <= n.moisture_max + 1e-3
            );
            assert!(n.insolation_mean >= 0.0, "insolation negative : {n:?}");
        }
    }

    #[test]
    fn water_budget_equals_sum_of_components() {
        // `water_budget.total` must be exactly the sum of the 4 stocks.
        // If this test fails, a new reservoir was added without updating
        // `compute_diagnostics`.
        let mut sim = sim_with_terrain(3, 42);
        for _ in 0..10 {
            sim.step();
        }
        let diag = sim.diagnostics();
        let sum = diag.water_budget.surface
            + diag.water_budget.humidity
            + diag.water_budget.groundwater
            + diag.water_budget.snow;
        assert!(
            (diag.water_budget.total - sum).abs() < 1e-3,
            "total={} != surface+humidity+gw+snow={}",
            diag.water_budget.total,
            sum
        );
    }
}
