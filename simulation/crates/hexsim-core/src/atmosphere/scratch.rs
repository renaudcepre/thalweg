use crate::grid::HexGrid;
use crate::temperature::TemperatureParams;
use crate::wind::{WindField, WindVec};

use super::{AtmosphereParams, EvapStats, saturation_upper, surface_means, upper_air_temperature};

/// Scratch buffers for `step_atmosphere_into`, owned by the caller
/// (`Simulation`) and reused every tick: zero malloc in the hot path
/// (perf effort #88/#65: orography, precipitation and advection lift
/// used to allocate ~6 fresh `Vec`s per simulated hour). Content
/// between two ticks is undefined: each sub-phase resizes/fills
/// whatever it consumes.
pub struct AtmoScratch {
    /// Generic snapshot shared sequentially by advection and
    /// diffusion (historical `snap` pattern).
    pub snap: Vec<f32>,
    /// Generic deltas, same sequential sharing as `snap`.
    pub deltas: Vec<f32>,
    pub temp_deltas: Vec<f32>,
    pub wind_upper: WindField,
    /// Precomputed `saturation_upper(T_upper)` per cell
    /// (#97). `T_current` = pre-advection temperature, shared
    /// identically by orographic convection and the Surface advection
    /// lift (both gather the same `sat_upper(T_neighbor)`). Memoizing
    /// this here kills the inter-cell redundancy (a high neighbor
    /// evaluated once instead of once per cell referencing it); loop
    /// iterations prevent LLVM's CSE, unlike intra-cell redundancy
    /// (cf. the powf lesson, JOURNAL 05-07).
    pub sat_upper_offset: Vec<f32>,
    /// Upper-air temperature per cell (`upper_air_temperature`), filled
    /// with `sat_upper_offset` by `fill_upper_air`; consumed by the
    /// vapor ↔ droplet transition.
    pub t_upper: Vec<f32>,
    // Surface advection with orographic lift.
    pub lift_deltas_upper: Vec<f32>,
    pub lift_upper_snap: Vec<f32>,
    // Orographic convection (6 parallel arrays).
    pub oro_src_surface: Vec<f32>,
    pub oro_src_upper: Vec<f32>,
    pub oro_elev: Vec<f32>,
    pub oro_delta_surface: Vec<f32>,
    pub oro_delta_upper_out: Vec<f32>,
    pub oro_delta_upper_in: Vec<f32>,
    // Precipitation.
    pub precip_cloud_delta: Vec<f32>,
    pub precip_water_delta: Vec<f32>,
    pub precip_snow_delta: Vec<f32>,
    /// Total ascent `w = H·(−∇·v) + v·∇z` per cell, in m/s (Phase 3
    /// ascent trigger; filled only when `updraft_ref_ms > 0`).
    pub convergence: Vec<f32>,
    /// Open-water evaporation stats for the tick, written by
    /// `step_evaporation`. Unlike the other fields here, this one is read
    /// back by the caller after `step_atmosphere_into` returns (same
    /// pattern as `convergence`/`updraft_field`): it is the diagnostics
    /// layer's sole source for evaporation, never recomputed there.
    pub evap: EvapStats,
}

impl AtmoScratch {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            snap: Vec::with_capacity(n),
            deltas: Vec::with_capacity(n),
            temp_deltas: Vec::with_capacity(n),
            wind_upper: vec![WindVec::default(); n],
            sat_upper_offset: Vec::with_capacity(n),
            t_upper: Vec::with_capacity(n),
            lift_deltas_upper: Vec::with_capacity(n),
            lift_upper_snap: Vec::with_capacity(n),
            oro_src_surface: Vec::with_capacity(n),
            oro_src_upper: Vec::with_capacity(n),
            oro_elev: Vec::with_capacity(n),
            oro_delta_surface: Vec::with_capacity(n),
            oro_delta_upper_out: Vec::with_capacity(n),
            oro_delta_upper_in: Vec::with_capacity(n),
            precip_cloud_delta: Vec::with_capacity(n),
            precip_water_delta: Vec::with_capacity(n),
            precip_snow_delta: Vec::with_capacity(n),
            convergence: Vec::with_capacity(n),
            evap: EvapStats::default(),
        }
    }

    /// Precomputes the upper-air temperature and `saturation_upper` per cell
    /// (#97). Single source of truth consumed by orographic convection
    /// and the Surface advection lift (LCL bound): both passes gather
    /// the same `sat_upper(T_neighbor)` on the pre-advection
    /// temperature, so memoizing it here kills the inter-cell
    /// redundancy (a high neighbor evaluated once instead of once per
    /// cell referencing it). Called at the top of
    /// `step_atmosphere_into`; direct callers of orographic convection
    /// (unit tests) must invoke it first.
    ///
    /// Since 2026-09-02 the upper-air temperature is horizontally
    /// homogeneous (`upper_air_temperature`: map-mean surface T and
    /// standard lapse from the map-mean ground), so both buffers only
    /// depend on each cell's elevation, on the map-mean elevation and on
    /// `upper_air_mean_t`, the diurnally smoothed map-mean surface
    /// temperature owned by the simulation (`AtmoForcing::upper_air_mean_t`,
    /// see `UPPER_AIR_SMOOTHING_TAU_S`): the free atmosphere does not
    /// follow the day/night swing of the surface. The mean elevation is
    /// read from the grid each call (constant until erosion moves it).
    pub(crate) fn fill_upper_air(
        &mut self,
        current: &HexGrid,
        upper_air_mean_t: f32,
        params: &AtmosphereParams,
        temp_params: &TemperatureParams,
    ) {
        let (_, mean_z) = surface_means(current);
        self.t_upper.clear();
        self.t_upper.extend(current.cells_slice().iter().map(|c| {
            upper_air_temperature(upper_air_mean_t, mean_z, c.elevation, params, temp_params)
        }));
        self.sat_upper_offset.clear();
        self.sat_upper_offset
            .extend(self.t_upper.iter().map(|&t| saturation_upper(t, params)));
    }
}
