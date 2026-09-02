//! Read/write API surface of [`Simulation`]: `snapshot`, `diagnostics`,
//! `grid`, `tick`, the `*_params` getters, the `set_*` setters. Plain
//! passthrough onto the fields declared in [`super`], kept out of the
//! orchestration (`new`/`step`/`step_hour`) so it doesn't drown it.

use super::{FireStats, Simulation};
use crate::atmosphere::{AtmosphereParams, PrecipitationMap};
use crate::climate::ClimateHistory;
use crate::climate_normals::CellClimateNormals;
use crate::diagnostics::{Diagnostics, compute_diagnostics};
use crate::dynamics::{SynopticParams, SynopticState};
use crate::erosion::{DischargeEmaMap, EdgeFluxEmaMap, ErosionParams};
use crate::fire::FireParams;
use crate::grid::HexGrid;
use crate::groundwater::GroundwaterParams;
use crate::hydro::{DischargeMap, EdgeFluxMap, FlowVecMap, HydroMaps, HydroParams};
use crate::lake::LakeParams;
use crate::phase_timing::PhaseTimings;
use crate::snapshot::GridState;
use crate::snow::SnowParams;
use crate::temperature::TemperatureParams;
use crate::time;
use crate::vegetation::VegetationParams;
use crate::wind::{WindField, WindParams, WindVec, compute_wind_magnitudes_into};

impl Simulation {
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
        );
        // Single source of truth for evaporation: `step_evaporation`'s `out`
        // param, written this tick into the scratch buffer it shares with
        // `Simulation` (same pattern as `updraft_field`/`scratch_atmo.convergence`).
        // No recomputation here (anti-pattern #2).
        diag.evap_observer = self.scratch_atmo.evap;
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
