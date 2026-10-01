use crate::grid::HexGrid;
use crate::par::for_each_chunk_mut2;
use crate::phase_timing::AtmoStepTimings;
use crate::temperature::TemperatureParams;
use crate::wind::{WindField, WindVec};

use super::coarse::MoistCoarseScratch;
use super::{
    AtmosphereParams, EvapCell, EvapStats, PrecipOutflow, saturation_upper, upper_air_temperature,
};

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
    /// This hour's **signed vapour ↔ droplet transfer** of each fine
    /// column (mm): `> 0` condensation, `< 0` the saturation adjustment
    /// giving droplets back to vapour, `0` nothing moved. Written by the
    /// fused sweep (`apply_temperature_advection_then_cloud_and_
    /// condensation`), which simply stores what `cloud_dynamics` returns.
    ///
    /// One reader, off by default: the by-altitude cloud budget instrument
    /// (`tests/diag_cloud_budget_by_altitude.rs`), which splits it into
    /// its two branches per elevation band.
    ///
    /// Written only when `AtmoForcing::track_cloud_transfer` says a
    /// reader exists: the store is loop-invariant, so it disappears from
    /// the sweep otherwise (measured at r120,
    /// `atmo_advection_temperature` 0.316 ms/h-tick with the gate against
    /// 0.326 storing unconditionally). One body either way — the gate is
    /// a store, not a second copy of the physics.
    pub cloud_transfer: Vec<f32>,
    // Surface advection with orographic lift.
    pub lift_deltas_upper: Vec<f32>,
    pub lift_upper_snap: Vec<f32>,
    // Orographic convection (parallel arrays).
    pub oro_src_surface: Vec<f32>,
    pub oro_src_upper: Vec<f32>,
    pub oro_elev: Vec<f32>,
    pub oro_delta_surface: Vec<f32>,
    /// Net `humidity_upper` delta of the orographic pump: the gathered
    /// inflow from lower neighbors minus this cell's own upward-pump
    /// loss (`dir_out` carries the per-direction outflow both terms are
    /// derived from — see `step_orographic_convection`'s gather phase).
    /// Replaces the historical separate `oro_delta_upper_out`/`_in` pair
    /// now that both are computed together by the same gather pass.
    pub oro_delta_upper: Vec<f32>,
    // Precipitation.
    /// Per-source outflow of the current precipitation tick (see
    /// `precipitation::PrecipOutflow`), filled by a parallel per-cell
    /// pass over the heavy KK2000 math, consumed by the gather pass that
    /// fills `precip_water_delta`/`precip_snow_delta`.
    pub precip_outflow: Vec<PrecipOutflow>,
    pub precip_water_delta: Vec<f32>,
    pub precip_snow_delta: Vec<f32>,
    /// Ping-pong buffers for the extra diffusion passes
    /// (`precipitation::spread_precip_further`) that carry the footprint
    /// past the first ring when `precip_spread_radius > 1`: each pass
    /// reads `precip_water_delta`/`precip_snow_delta` and writes here,
    /// then the two pairs are swapped (`std::mem::swap` on the `Vec`s
    /// themselves, no copy) so the next pass reads what this one wrote.
    /// Unused, left at whatever capacity they last had, when the radius
    /// rounds to 1 (no extra pass).
    pub precip_water_delta_tmp: Vec<f32>,
    pub precip_snow_delta_tmp: Vec<f32>,
    /// Total ascent `w = H·(−∇·v) + v·∇z` per cell, in m/s (Phase 3
    /// ascent trigger; filled only when `updraft_ref_ms > 0`).
    pub convergence: Vec<f32>,
    /// Per-direction outflow of the scatter pass currently being run in
    /// two phases (r250 perf effort): `dir_out[d][i]` is what cell `i`
    /// sends toward its toric neighbor in direction `d`
    /// (`coord::DIRECTIONS[d]`). Filled by a parallel "outflow" pass
    /// over source cells, read back by the immediately following
    /// "gather" pass over destination cells via
    /// `coord::opposite_direction` (see `HexGrid::neighbor_indices_toric`
    /// for the identity this relies on). Reused sequentially by every
    /// directional scatter of the atmosphere (orographic convection,
    /// both humidity layers, cloud water, temperature advection);
    /// content is undefined between two passes, like `snap`/`deltas`.
    pub dir_out: [Vec<f32>; 6],
    /// Second per-direction outflow set, needed only when a pass splits
    /// a single directional flux into two destination pools within the
    /// same tick. Surface-layer humidity advection is the only user
    /// today: `dir_out` carries the total flux toward each neighbor,
    /// `dir_out_secondary` the part of that same flux the destination's
    /// LCL bound converts to `humidity_upper` there instead of
    /// `humidity_surface` (forced orographic condensation).
    pub dir_out_secondary: [Vec<f32>; 6],
    /// Per-cell share sent identically to each of the 6 toric neighbors,
    /// for a scatter that distributes without directional weighting
    /// (cloud diffusion, precipitation dispersion): no per-direction
    /// storage is needed since every direction carries the same value —
    /// `Σ_{k ∈ neighbors(j)} uniform_share[k]` gathers it directly, the
    /// "k is my neighbor" relation being symmetric on the toric lattice
    /// (`coord::opposite_direction` isn't needed here). Reused
    /// sequentially; a pass needing two independent uniform shares in
    /// the same tick (precipitation: rain vs snow) also uses
    /// `uniform_share_secondary`.
    pub uniform_share: Vec<f32>,
    pub uniform_share_secondary: Vec<f32>,
    /// Vapour each cell exported to the sky reservoir this hour
    /// (`regime::step_weather_regime`, #63), written by the per-cell pass
    /// and read back by the deterministic block reduction that totals it.
    /// Untouched — never even sized — while the regime is disabled.
    pub regime_moved: Vec<f32>,
    /// Block partials of that reduction (`par::reduce_blocks`), so the
    /// total leaving the map has the same bits whatever the thread count.
    pub regime_partials: Vec<f32>,
    /// Per-cell output of `step_evaporation`'s parallel pass: the
    /// evaporative demand (mm/day, Dalton/Meyer) of each cell counted as
    /// open water, or a negative sentinel for a cell that isn't (dry,
    /// under capacity, or frozen), reduced serially in index order into
    /// `evap` (same subset and summation order as the historical single
    /// serial loop, bit-identical); and the vapour each cell emitted this
    /// tick by source, read back by `Simulation` into its daily
    /// accumulator right after the call, like `evap`.
    pub evap_cells: Vec<EvapCell>,
    /// Open-water evaporation stats for the tick, written by
    /// `step_evaporation`. Unlike the other fields here, this one is read
    /// back by the caller after `step_atmosphere_into` returns (same
    /// pattern as `convergence`/`updraft_field`): it is the diagnostics
    /// layer's sole source for evaporation, never recomputed there.
    pub evap: EvapStats,
    /// This tick's `atmosphere` sub-phase durations, filled by
    /// `step_atmosphere_into` and read back by the caller (`Simulation`)
    /// right after the call, same pattern as `evap`. See
    /// [`AtmoStepTimings`] for why this isn't cumulative.
    pub step_timings: AtmoStepTimings,
    /// Buffers of the coarse moist step (`atmosphere::coarse`,
    /// `step_moist_precip`): the fine-sized staging buffer, and the
    /// coarse-sized fields the ~1 km passes read and write. Lives here
    /// rather than on `Simulation` so the coarse step keeps reading the
    /// fine `sat_upper_offset`/`convergence` this struct already owns
    /// without a second `&mut` borrow. Same contract as every other field:
    /// content undefined between two ticks — the coarse mirror itself
    /// is NOT here, it is `Simulation::moist_coarse`.
    pub moist: MoistCoarseScratch,
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
            cloud_transfer: Vec::with_capacity(n),
            lift_deltas_upper: Vec::with_capacity(n),
            lift_upper_snap: Vec::with_capacity(n),
            oro_src_surface: Vec::with_capacity(n),
            oro_src_upper: Vec::with_capacity(n),
            oro_elev: Vec::with_capacity(n),
            oro_delta_surface: Vec::with_capacity(n),
            oro_delta_upper: Vec::with_capacity(n),
            precip_outflow: Vec::with_capacity(n),
            precip_water_delta: Vec::with_capacity(n),
            precip_snow_delta: Vec::with_capacity(n),
            // No `with_capacity(n)`: only allocated (by the first extra
            // pass) when `precip_spread_radius` rounds above 1.
            precip_water_delta_tmp: Vec::new(),
            precip_snow_delta_tmp: Vec::new(),
            convergence: Vec::with_capacity(n),
            dir_out: std::array::from_fn(|_| Vec::with_capacity(n)),
            dir_out_secondary: std::array::from_fn(|_| Vec::with_capacity(n)),
            uniform_share: Vec::with_capacity(n),
            uniform_share_secondary: Vec::with_capacity(n),
            // No `with_capacity(n)`: when the regime is off, this pass
            // never runs, and a run without it must not pay an
            // allocation for scratch buffers it never fills.
            regime_moved: Vec::new(),
            regime_partials: Vec::new(),
            evap_cells: Vec::with_capacity(n),
            evap: EvapStats::default(),
            step_timings: AtmoStepTimings::default(),
            // No `with_capacity`: sized on first use, and never sized at
            // all on the fine path (`HEXSIM_MOIST_COARSE=0`), which must
            // not pay an allocation for buffers it never fills.
            moist: MoistCoarseScratch::default(),
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
    /// follow the day/night swing of the surface. `mean_z` is the
    /// map-mean elevation of `current` (`surface_means(current).1`,
    /// `AtmoForcing::mean_elevation`): the caller already reduced it
    /// with the mean temperature that steps the anchor, so it travels
    /// here instead of costing this pass a second full-grid reduction
    /// of the same state.
    pub(crate) fn fill_upper_air(
        &mut self,
        current: &HexGrid,
        upper_air_mean_t: f32,
        mean_z: f32,
        params: &AtmosphereParams,
        temp_params: &TemperatureParams,
    ) {
        let cells = current.cells_slice();
        // Two elementwise maps fused into one sweep (r250 perf effort,
        // chunk B2): `sat_upper_offset[i] = g(t_upper[i])` reads only
        // cell `i`'s OWN just-computed `t_upper[i]`, never a neighbor's,
        // so computing both in the same per-cell closure invocation
        // (`par::for_each_chunk_mut2`) is bit-identical to running them
        // as two full-grid passes with a barrier in between — same two
        // operations, same order, one sweep instead of two.
        self.t_upper.clear();
        self.t_upper.resize(cells.len(), 0.0);
        self.sat_upper_offset.clear();
        self.sat_upper_offset.resize(cells.len(), 0.0);
        for_each_chunk_mut2(
            &mut self.t_upper,
            &mut self.sat_upper_offset,
            |start, t_chunk, sat_chunk| {
                for (local, (t, s)) in t_chunk.iter_mut().zip(sat_chunk.iter_mut()).enumerate() {
                    let elevation = cells[start + local].elevation;
                    *t = upper_air_temperature(
                        upper_air_mean_t,
                        mean_z,
                        elevation,
                        params,
                        temp_params,
                    );
                    *s = saturation_upper(*t, params);
                }
            },
        );
    }
}
