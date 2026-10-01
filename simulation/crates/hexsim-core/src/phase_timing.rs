//! Per-phase timing of the real tick (`Simulation::step_hour`).
//!
//! Unlike the `perf_phase_breakdown` bench (which REPLAYS the tick phase
//! by phase and must be kept as a mirror, a blind spot documented in its
//! header), these counters live INSIDE the orchestrator: every phase
//! that runs is measured, at production cadences, including the Tier 3
//! phases (vegetation, fire, lakes, erosion, normals) that the mirror
//! bench ignores.
//!
//! Cost: 2 clock reads per phase per tick (~15 ns each on Apple
//! Silicon), negligible next to phases running in the hundreds of µs.
//! On wasm32 (`Instant` unavailable), the clock is a no-op and all
//! counters stay at zero.

use crate::atmosphere::MoistCoarseTimings;

/// Wall-clock cumulative totals (seconds) per tick phase, since the
/// [`crate::simulation::Simulation`] was created or the last
/// [`crate::simulation::Simulation::reset_phase_timings`].
#[derive(Debug, Default, Clone, Copy)]
pub struct PhaseTimings {
    /// `compute_illumination` (shadow raymarch + cloud shadow), hourly.
    pub illumination: f64,
    /// `atmosphere::surface_means` on the pre-temperature state: the
    /// map-mean surface temperature and elevation that anchor the mixed
    /// boundary-layer air `step_temperature` exchanges sensible heat with
    /// (`TemperatureForcing::mean_surface_t`/`mean_elevation`). One
    /// full-grid reduction, hourly, computed by the orchestrator so its
    /// cost is read apart from the balance sweep.
    pub temp_means: f64,
    /// `step_temperature`, hourly: the per-cell balance sweep alone (its
    /// map means are `temp_means` above).
    pub temperature: f64,
    /// Synoptic dynamics (aggregate + ODE + interpolation), 1 h out of M.
    pub synoptic: f64,
    /// `compute_wind_field_into` + magnitudes, 1 h out of N.
    pub wind: f64,
    /// `step_snow`, hourly.
    pub snow: f64,
    /// `atmosphere::surface_means` on the post-snow state, the
    /// instantaneous map-mean surface temperature that steps the
    /// upper-air anchor EMA (`Simulation::upper_air_mean_t`) and the
    /// map-mean elevation `fill_upper_air` reads
    /// (`AtmoForcing::mean_elevation`). One full-grid reduction, hourly.
    pub atmo_means: f64,
    /// `step_atmosphere_into` (evap, uplift, advection, condensation, precip).
    pub atmosphere: f64,
    /// Moist upper layer, fine → coarse (coarse upper layer): the hourly
    /// transfer of what the fine passes moved (two interpolations, one
    /// delta sweep, three gathers) on the coarse path, or step 1's plain
    /// mirror gather (`MoistCoarseState::gather_from_fine`) on the fine
    /// one. Runs right after `step_atmosphere_into` returns, on the state
    /// it just wrote, before the `current`/`next` swap.
    ///
    /// This and the two rows below are top-level phases like `synoptic`,
    /// NOT `atmo_*` sub-buckets (those are measured INSIDE
    /// `step_atmosphere_into`; these run in the orchestrator, after it
    /// returns) — counted once in [`PhaseTimings::total`], not part of
    /// [`PhaseTimings::atmo_rows`]. On the coarse path `atmo_precipitation`
    /// reads 0 and its work is in `atmo_moist_coarse` below.
    pub atmo_moist_gather: f64,
    /// The coarse passes themselves (imposed regime, vapour ↔ droplet
    /// transition, KK2000) plus the coarse → fine precipitation
    /// distribution. Zero on the fine path.
    pub atmo_moist_coarse: f64,
    /// Rewriting the fine `humidity_upper`/`cloud_water` views from the
    /// coarse stock (two interpolations and one write sweep). Zero on the
    /// fine path.
    pub atmo_moist_views: f64,
    /// `ClimateNormalsAccumulator::record_tick`, hourly.
    pub normals: f64,
    /// `ClimateHistory::record_tick`, daily.
    pub history: f64,
    /// `step_groundwater`, daily.
    pub groundwater: f64,
    /// MFD slice (8 passes + map accumulation), daily.
    pub hydro: f64,
    /// EMA discharge + edge flux (#105), daily.
    pub ema: f64,
    /// `step_lake_leveling` (#106), daily.
    pub lakes: f64,
    /// `step_erosion` + surface normals recompute, daily.
    pub erosion: f64,
    /// `step_vegetation`, daily.
    pub vegetation: f64,
    /// `step_fire`, daily.
    pub fire: f64,
    /// `scratch.fill_upper_air`, part of `atmosphere`, hourly: the
    /// per-cell fill alone since the map-mean elevation it needs travels
    /// in `AtmoForcing::mean_elevation` (computed once under
    /// `atmo_means`) instead of being re-reduced from the grid here.
    pub atmo_upper_air: f64,
    /// `step_evaporation`, part of `atmosphere`, hourly.
    pub atmo_evaporation: f64,
    /// `step_orographic_convection`, part of `atmosphere`, hourly.
    pub atmo_orographic: f64,
    /// `advect_humidity_layer_into`, both layers, part of `atmosphere`,
    /// only on a transport tick (`atmosphere::scaling::transport_subsample`).
    pub atmo_advection_humidity: f64,
    /// `step_uplift`, part of `atmosphere`, hourly.
    pub atmo_uplift: f64,
    /// `advect_temperature_by_wind_into`'s gather AND apply, fused with
    /// `step_cloud_dynamics` and `step_surface_condensation` into one
    /// sweep since the r250 perf effort's chunk B2 (see
    /// `atmosphere::apply_temperature_advection_then_cloud_and_condensation`):
    /// this bucket now carries all three, attributed to the first one.
    /// Part of `atmosphere`, hourly.
    pub atmo_advection_temperature: f64,
    /// `step_cloud_dynamics` (vapor ↔ droplet transition). Folded into
    /// `atmo_advection_temperature` since chunk B2 (see its doc): this
    /// bucket reads 0 from then on, kept for the row layout.
    pub atmo_cloud_dynamics: f64,
    /// `step_surface_condensation` (radiative fog). Folded into
    /// `atmo_advection_temperature` since chunk B2 (see its doc): this
    /// bucket reads 0 from then on, kept for the row layout.
    pub atmo_condensation: f64,
    /// `advect_cloud_water_into` + `step_cloud_diffusion`, part of
    /// `atmosphere`, only on a transport tick.
    pub atmo_advection_cloud: f64,
    /// `step_precipitation_into`, part of `atmosphere`, only on a
    /// precipitation tick (`PRECIP_SUBSAMPLE_HOURS`, hourly by default).
    pub atmo_precipitation: f64,
    /// Remaining `atmosphere` passes not broken out above (currently
    /// `fill_updraft_into`, the synoptic ascent trigger, active only when
    /// `updraft_ref_ms > 0`).
    pub atmo_other: f64,
    /// `fill_hydro_outflow` (per-source MFD split toward the 6
    /// neighbors), summed over the 8 daily substeps, part of `hydro`.
    pub hydro_outflow: f64,
    /// `fill_hydro_source_aggregates` (`flux_out` + `flow_vec` per
    /// source), summed over the substeps, part of `hydro`.
    pub hydro_aggregates: f64,
    /// `fill_hydro_edge_flux` (transposition to `edge_flux_out`), summed
    /// over the substeps, part of `hydro`.
    pub hydro_edge_flux: f64,
    /// `gather_hydro_water` (fixed-order gather into `next`, carries the
    /// full-grid `current → next` copy), summed over the substeps, part
    /// of `hydro`.
    pub hydro_gather: f64,
    /// Bookkeeping of the day's maps in `Simulation::step_hydro_tranche`:
    /// the reset before the substeps and the accumulation of each
    /// substep's `scratch_*` into `discharge_map`/`flow_vec_map`/
    /// `edge_flux_map`, part of `hydro`.
    pub hydro_accumulate: f64,
    /// Number of simulated hours covered by the cumulative totals.
    pub hours: u64,
}

impl PhaseTimings {
    /// Sum of the measured phases (s). Slightly lower than the full
    /// tick's wall clock: the inter-phase glue (swaps, precip
    /// accumulation) is not counted.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.rows().iter().map(|&(_, s)| s).sum()
    }

    /// `(name, seconds)` rows in tick execution order.
    #[must_use]
    pub fn rows(&self) -> [(&'static str, f64); 20] {
        [
            ("illumination", self.illumination),
            ("temp_means", self.temp_means),
            ("temperature", self.temperature),
            ("synoptic", self.synoptic),
            ("wind", self.wind),
            ("snow", self.snow),
            ("atmo_means", self.atmo_means),
            ("atmosphere", self.atmosphere),
            ("atmo_moist_gather", self.atmo_moist_gather),
            ("atmo_moist_coarse", self.atmo_moist_coarse),
            ("atmo_moist_views", self.atmo_moist_views),
            ("normals", self.normals),
            ("history", self.history),
            ("groundwater", self.groundwater),
            ("hydro", self.hydro),
            ("ema", self.ema),
            ("lakes", self.lakes),
            ("erosion", self.erosion),
            ("vegetation", self.vegetation),
            ("fire", self.fire),
        ]
    }

    /// `(name, seconds)` rows for the `atmosphere` sub-phases, in the
    /// order `step_atmosphere_into` runs them. A breakdown of the single
    /// `atmosphere` row above, not an addition to [`PhaseTimings::total`]:
    /// summing these double-counts the parent bucket.
    #[must_use]
    pub fn atmo_rows(&self) -> [(&'static str, f64); 11] {
        [
            ("atmo_upper_air", self.atmo_upper_air),
            ("atmo_evaporation", self.atmo_evaporation),
            ("atmo_orographic", self.atmo_orographic),
            ("atmo_advection_humidity", self.atmo_advection_humidity),
            ("atmo_uplift", self.atmo_uplift),
            (
                "atmo_advection_temperature",
                self.atmo_advection_temperature,
            ),
            ("atmo_cloud_dynamics", self.atmo_cloud_dynamics),
            ("atmo_condensation", self.atmo_condensation),
            ("atmo_advection_cloud", self.atmo_advection_cloud),
            ("atmo_precipitation", self.atmo_precipitation),
            ("atmo_other", self.atmo_other),
        ]
    }

    /// `(name, seconds)` rows for the `hydro` sub-phases, in the order one
    /// MFD substep runs them, the orchestrator's map bookkeeping last. A
    /// breakdown of the single `hydro` row above, like
    /// [`PhaseTimings::atmo_rows`] for `atmosphere`: not an addition to
    /// [`PhaseTimings::total`].
    #[must_use]
    pub fn hydro_rows(&self) -> [(&'static str, f64); 5] {
        [
            ("hydro_outflow", self.hydro_outflow),
            ("hydro_aggregates", self.hydro_aggregates),
            ("hydro_edge_flux", self.hydro_edge_flux),
            ("hydro_gather", self.hydro_gather),
            ("hydro_accumulate", self.hydro_accumulate),
        ]
    }

    /// Folds one substep's (non-cumulative) [`HydroStepTimings`] into the
    /// matching cumulative `hydro_*` fields. Called by
    /// `Simulation::step_hydro_tranche` right after each
    /// `step_hydro_mfd_into` returns (8 times a day), alongside the `hydro`
    /// bucket's own `elapsed_s` accumulation around the whole slice.
    pub fn accumulate_hydro(&mut self, st: &HydroStepTimings) {
        self.hydro_outflow += st.outflow;
        self.hydro_aggregates += st.aggregates;
        self.hydro_edge_flux += st.edge_flux;
        self.hydro_gather += st.gather;
    }

    /// Folds this tick's (non-cumulative) [`AtmoStepTimings`] into the
    /// matching cumulative `atmo_*` fields. Called by `Simulation::step_hour`
    /// right after `step_atmosphere_into` returns, alongside the `atmosphere`
    /// bucket's own `elapsed_s` accumulation.
    pub fn accumulate_atmo(&mut self, st: &AtmoStepTimings) {
        self.atmo_upper_air += st.upper_air;
        self.atmo_evaporation += st.evaporation;
        self.atmo_orographic += st.orographic;
        self.atmo_advection_humidity += st.advection_humidity;
        self.atmo_uplift += st.uplift;
        self.atmo_advection_temperature += st.advection_temperature;
        self.atmo_cloud_dynamics += st.cloud_dynamics;
        self.atmo_condensation += st.condensation;
        self.atmo_advection_cloud += st.advection_cloud;
        self.atmo_precipitation += st.precipitation;
        self.atmo_other += st.other;
    }

    /// Folds this tick's (non-cumulative) [`MoistCoarseTimings`] into the
    /// three top-level moist-layer rows. Called by `Simulation` right
    /// after `atmosphere::step_moist_coarse` returns, the same pattern as
    /// [`Self::accumulate_atmo`].
    pub fn accumulate_moist(&mut self, st: &MoistCoarseTimings) {
        self.atmo_moist_gather += st.transfer;
        self.atmo_moist_coarse += st.coarse;
        self.atmo_moist_views += st.views;
    }
}

/// Per-tick durations (s) of the `atmosphere` sub-phases, measured inside
/// `step_atmosphere_into` and read back by the caller right after the call
/// (same pattern as [`crate::atmosphere::EvapStats`] on `AtmoScratch`).
/// Unlike [`PhaseTimings`], not cumulative: every field is fully
/// overwritten at the top of each `step_atmosphere_into` call, so the
/// struct carries no meaning between two ticks — the caller accumulates
/// what it reads into the matching `PhaseTimings::atmo_*` field. Kept on
/// `AtmoScratch` (an existing argument) rather than added as its own
/// parameter: `step_atmosphere_into` is already at the 7-argument ceiling
/// (convention #61).
#[derive(Debug, Default, Clone, Copy)]
pub struct AtmoStepTimings {
    pub upper_air: f64,
    pub evaporation: f64,
    pub orographic: f64,
    pub advection_humidity: f64,
    pub uplift: f64,
    pub advection_temperature: f64,
    pub cloud_dynamics: f64,
    pub condensation: f64,
    pub advection_cloud: f64,
    pub precipitation: f64,
    pub other: f64,
}

/// Per-substep durations (s) of the `hydro` sub-phases, measured inside
/// `hydro::step_hydro_mfd_into` and read back by the caller right after
/// the call, the same pattern as [`AtmoStepTimings`] on `AtmoScratch`:
/// not cumulative, fully overwritten at the top of every call, kept on
/// `HydroScratch` (an existing argument) because `step_hydro_mfd_into`
/// already sits at the 7-argument ceiling (convention #61). The
/// orchestrator's own map bookkeeping (`hydro_accumulate`) is measured
/// there, not here.
#[derive(Debug, Default, Clone, Copy)]
pub struct HydroStepTimings {
    pub outflow: f64,
    pub aggregates: f64,
    pub edge_flux: f64,
    pub gather: f64,
}

/// Opaque clock mark. `Instant` on native, unit on wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub type Mark = std::time::Instant;
/// Opaque clock mark. `Instant` on native, unit on wasm32.
#[cfg(target_arch = "wasm32")]
pub type Mark = ();

/// Start a phase measurement.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
#[must_use]
pub fn mark() -> Mark {
    std::time::Instant::now()
}

/// Start a phase measurement (wasm no-op).
#[cfg(target_arch = "wasm32")]
#[inline]
#[must_use]
pub fn mark() -> Mark {}

/// Seconds elapsed since `m`.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
#[must_use]
pub fn elapsed_s(m: Mark) -> f64 {
    m.elapsed().as_secs_f64()
}

/// Seconds elapsed since `m` (always 0 on wasm).
#[cfg(target_arch = "wasm32")]
#[inline]
#[must_use]
pub fn elapsed_s(m: Mark) -> f64 {
    let () = m;
    0.0
}
