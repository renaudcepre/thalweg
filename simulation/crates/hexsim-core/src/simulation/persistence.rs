//! Checkpoint glue: [`Simulation::save_state`]/[`Simulation::load_state`],
//! see [`crate::checkpoint`] for the format itself.

use super::Simulation;
use crate::ablation::Ablation;
use crate::atmosphere::{AtmoScratch, surface_means};
use crate::checkpoint::{CHECKPOINT_FORMAT_VERSION, Checkpoint, CheckpointError, MAGIC};
use crate::climate::DayRecord;
use crate::phase_timing::PhaseTimings;
use crate::synoptic_mesh::SynopticMesh;
use crate::temperature::IllumCache;
use crate::wind::{WindField, WindVec};

impl Simulation {
    /// Serializes the full simulation state to `MessagePack` (see
    /// [`crate::checkpoint`]). The blob can be reloaded via
    /// [`Simulation::load_state`] to resume the simulation **identically**;
    /// bit-identical resumption is proven by test. This includes the process's
    /// [`Ablation`] (env-var A/B switches): [`Simulation::load_state`] refuses
    /// a blob captured under a different ablation config rather than silently
    /// resuming in different physics (see [`crate::ablation`]).
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
            upper_air_mean_t: Some(self.upper_air_mean_t),
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
            ablation: Ablation::effective().clone(),
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
        // Field absent from checkpoints predating the smoothed upper-air
        // anchor (`serde(default)` → `None`): restart on the instantaneous
        // mean of the loaded grid, exactly like `Simulation::new`; the EMA
        // settles within ~3τ (3 days).
        let upper_air_mean_t = ckpt
            .upper_air_mean_t
            .unwrap_or_else(|| surface_means(&current).0);
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
            upper_air_mean_t,
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
    use crate::atmosphere::AtmosphereParams;
    use crate::grid::HexGrid;
    use crate::groundwater::GroundwaterParams;
    use crate::hydro::HydroParams;
    use crate::snow::SnowParams;
    use crate::temperature::TemperatureParams;
    use crate::terrain::{TerrainParams, generate_terrain};
    use crate::wind::WindParams;

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
}
