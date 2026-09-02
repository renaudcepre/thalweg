use crate::grid::HexGrid;
use crate::temperature::TemperatureParams;
use crate::wind::{WindField, WindVec};

use super::{AtmosphereParams, saturation_upper};

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
    /// Precomputed `saturation_upper(T_current - t_offset)` per cell
    /// (#97). `T_current` = pre-advection temperature, shared
    /// identically by orographic convection and the Surface advection
    /// lift (both gather the same `sat_upper(T_neighbor)`). Memoizing
    /// this here kills the inter-cell redundancy (a high neighbor
    /// evaluated once instead of once per cell referencing it); loop
    /// iterations prevent LLVM's CSE, unlike intra-cell redundancy
    /// (cf. the powf lesson, JOURNAL 05-07).
    pub sat_upper_offset: Vec<f32>,
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
        }
    }

    /// Precomputes `saturation_upper(T_current - t_offset)` per cell
    /// (#97). Single source of truth consumed by orographic convection
    /// and the Surface advection lift (LCL bound): both passes gather
    /// the same `sat_upper(T_neighbor)` on the pre-advection
    /// temperature, so memoizing it here kills the inter-cell
    /// redundancy (a high neighbor evaluated once instead of once per
    /// cell referencing it). Called at the top of
    /// `step_atmosphere_into`; direct callers of orographic convection
    /// (unit tests) must invoke it first.
    pub(crate) fn fill_sat_upper_offset(
        &mut self,
        current: &HexGrid,
        params: &AtmosphereParams,
        temp_params: &TemperatureParams,
    ) {
        let t_offset = temp_params.lapse_rate * params.upper_layer_altitude_m / 1000.0;
        self.sat_upper_offset.clear();
        self.sat_upper_offset.extend(
            current
                .cells_slice()
                .iter()
                .map(|c| saturation_upper(c.temperature - t_offset, params)),
        );
    }
}
