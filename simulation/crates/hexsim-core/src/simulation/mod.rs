//! The simulation engine: per-tick orchestration, the `Simulation` state,
//! and the API surface around it.
//!
//! Split (same pattern as `temperature/`: `mod.rs` owns the shared state,
//! sub-modules add `impl Simulation` blocks) out of a former single
//! 1700+ line `simulation.rs` into one file per concern:
//!
//! - This top-level module: [`crate::simulation::FireStats`], the [`crate::simulation::Simulation`] struct
//!   itself, and the orchestration — `new`, `step`, `step_hour` and the
//!   private helpers they call (`refresh_wind`, `step_hydro_tranche`,
//!   `level_lakes`, `step_daily_tail`). The call order inside
//!   `step_hour`/`step_daily_tail` is physics (convection before
//!   advection, groundwater before hydro), written in hand rather than
//!   dispatched through a generic trait (issue #61: no `Phenomenon`
//!   trait).
//! - `accessors`: the read/write API surface — `snapshot`,
//!   `diagnostics`, `grid`, `tick`, the `*_params` getters, the `set_*`
//!   setters. Passthrough onto the fields declared here, kept out of the
//!   orchestration so it doesn't drown it.
//! - `params`: `update_param`, the stringly-typed runtime tuning
//!   dispatch (`group.field` keys), and its two per-group helpers.
//! - `persistence`: `save_state`/`load_state`, the checkpoint glue.
//!
//! `Simulation`'s fields stay private to this module. Every sub-module
//! below is a child of `simulation`, so an `impl Simulation` block there
//! sees the same private fields as this file — no re-export needed, and
//! `hexsim_core::simulation::X` stays the exact same public path for
//! every `X` regardless of which sub-module implements it: the split
//! moves code, it does not change the crate's public surface.

use crate::ablation::Ablation;
use crate::atmosphere::{
    AtmoForcing, AtmoScratch, AtmosphereParams, PrecipitationMap, step_atmosphere_into,
};
use crate::climate::{ClimateHistory, DayRecord};
use crate::climate_normals::ClimateNormalsAccumulator;
use crate::dynamics::{SynopticParams, SynopticState};
use crate::erosion::{
    DischargeEmaMap, EdgeFluxEmaMap, ErosionForcing, ErosionParams, step_erosion, update_edge_ema,
    update_ema,
};
use crate::fire::{FireParams, step_fire};
use crate::grid::HexGrid;
use crate::groundwater::{GroundwaterParams, step_groundwater};
use crate::hydro::{
    DischargeMap, EdgeFluxMap, FlowVecMap, FluxMap, HydroParams, step_hydro_mfd_into,
};
use crate::lake::{LakeParams, step_lake_leveling};
use crate::phase_timing::{PhaseTimings, elapsed_s, mark};
use crate::snow::{SnowForcing, SnowParams, step_snow};
use crate::synoptic_mesh::SynopticMesh;
use crate::temperature::{
    IllumCache, TemperatureForcing, TemperatureParams, aspect_insolation_correction,
    compute_illumination_cached, compute_surface_normals, solar_beam_at_tick, step_temperature,
};
use crate::time::{self, TICKS_PER_DAY};
use crate::vegetation::{VegetationParams, step_vegetation};
use crate::wind::{
    WindField, WindParams, WindVec, compute_wind_field_into, compute_wind_magnitudes_into,
};
use serde::Serialize;

mod accessors;
mod params;
mod persistence;

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
pub(crate) const WIND_SUBSAMPLE_HOURS: u64 = 3;

/// Effective value: `WIND_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_WIND_SUBSAMPLE` (parametric A/B without recompiling). Delegates
/// to [`Ablation::effective`], which reads the environment once for the
/// whole process.
fn wind_subsample() -> u64 {
    Ablation::effective().wind_subsample
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
pub(crate) const SYNOPTIC_SUBSAMPLE_HOURS: u64 = 3;

/// Effective value: `SYNOPTIC_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_SYNOPTIC_SUBSAMPLE` (parametric A/B without recompiling).
/// Delegates to [`Ablation::effective`], which reads the environment once
/// for the whole process.
fn synoptic_subsample() -> u64 {
    Ablation::effective().synoptic_subsample
}

/// Compiled-in default for the coarse synoptic mesh toggle: coarse ON. See
/// the ablation rationale and A/B table at its call site in
/// `Simulation::new` (`synoptic_coarse`, `HEXSIM_SYNOPTIC_COARSE`).
pub(crate) const SYNOPTIC_COARSE_DEFAULT: bool = true;

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
        //
        // Effective value: `SYNOPTIC_COARSE_DEFAULT` unless overridden by
        // `HEXSIM_SYNOPTIC_COARSE`, resolved through [`Ablation::effective`]
        // (reads the environment once for the whole process).
        let synoptic_coarse = Ablation::effective().synoptic_coarse;
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
            &TemperatureForcing {
                hour_tick: self.hour_tick,
                flux_factor: &self.scratch_flux_factor,
                snow: &self.snow_params,
            },
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
            &SnowForcing {
                beam_w_m2: solar.beam,
                ground_albedo: self.temperature_params.ground_albedo,
                flux_factor: &self.scratch_flux_factor,
                wind_mag: &self.wind_mag,
                rain_last_tick: &self.scratch_precip_tick,
                gw_max_capacity: self.groundwater_params.max_capacity,
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
}
