//! Full-state serialization (checkpoint): save/restore of the entire world
//! to a reloadable file.
//!
//! # Why
//! Spinning up a mature world (climax forest) takes decades of simulated
//! time. A checkpoint lets you pay that cost **once**, then reload the
//! exact state, to resume a long run after a crash (ephemeral server), or
//! to give scale tests an "advanced forest" fixture without a 30-year
//! wait.
//!
//! # Fidelity
//! The checkpoint captures ALL authoritative state of the
//! [`Simulation`](crate::simulation::Simulation): the grid, the clock, the
//! prognostic synoptic state, the in-progress yearly normals accumulator,
//! the precipitation hysteresis, the retained downsampled wind field, the
//! fire counters. *Derived* fields (double-buffer `next`, reconstructible
//! neighbor caches, scratch buffers) are rebuilt on load, never relied
//! upon. Since fire is a **stateless** random draw
//! (`hash01(seed, day, cell)`), the seed plus the clock are enough to
//! reproduce it; no generator state to store.
//!
//! # Format
//! `MessagePack` (`rmp-serde`, same conventions as the wire format):
//! compact binary, and above all it accepts **arbitrary map keys**,
//! needed for `HexGrid::coord_index` and `ClimateHistory`
//! (`HashMap<HexCoord, _>`), which JSON refuses (string keys only).
//! Versioned envelope: loading a file with an incompatible format version
//! is **refused with a clear message** ([`CheckpointError::Version`])
//! rather than silently misread.

use serde::{Deserialize, Serialize};

use crate::atmosphere::AtmosphereParams;
use crate::climate::{ClimateHistory, DayRecord};
use crate::climate_normals::ClimateNormalsAccumulator;
use crate::dynamics::{SynopticParams, SynopticState};
use crate::erosion::ErosionParams;
use crate::fire::FireParams;
use crate::grid::HexGrid;
use crate::groundwater::GroundwaterParams;
use crate::hydro::HydroParams;
use crate::lake::LakeParams;
use crate::snow::SnowParams;
use crate::temperature::TemperatureParams;
use crate::vegetation::VegetationParams;
use crate::wind::{WindParams, WindVec};

/// Checkpoint format version. Bump on any breaking change to the
/// [`Checkpoint`] schema. Loading a different version is refused
/// ([`CheckpointError::Version`]); no silent migration.
///
/// v2 (issue #88): synoptic state now lives on the coarse torus
/// (`synoptic_coarse_radius` added, `synoptic_state` vectors at coarse
/// size); a v1 carries fine-grid state that can't be converted.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// Header marker: distinguishes a `HexSim` checkpoint from some unrelated
/// file dropped in by mistake.
pub(crate) const MAGIC: &str = "HEXSIM_CKPT";

/// Errors saving/loading a checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// The blob isn't `MessagePack` decodable into a [`Checkpoint`].
    #[error("checkpoint deserialization failed: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    /// `MessagePack` serialization failed (doesn't happen on a valid
    /// state, but the API stays honest).
    #[error("checkpoint serialization failed: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    /// The blob decodes fine but doesn't carry the `HexSim` marker.
    #[error("file not recognized as a HexSim checkpoint")]
    BadMagic,
    /// Format version differs from the engine's: explicit refusal.
    #[error("incompatible checkpoint version: file v{found}, engine v{expected} (no migration)")]
    Version {
        /// Version read from the file.
        found: u32,
        /// Version expected by this engine.
        expected: u32,
    },
}

/// Full authoritative state of a
/// [`Simulation`](crate::simulation::Simulation). See the module doc for
/// what's captured here vs. rebuilt on load.
///
/// Fields are `pub(crate)`: `Simulation::save_state`/`load_state` handle
/// the mapping to its private fields (same crate).
#[derive(Serialize, Deserialize)]
pub(crate) struct Checkpoint {
    pub(crate) magic: String,
    pub(crate) format_version: u32,
    /// Engine version at dump time (traceability; not binding).
    pub(crate) engine_version: String,

    pub(crate) grid: HexGrid,
    pub(crate) hour_tick: u64,

    pub(crate) seed: u32,
    pub(crate) fire_ignitions_total: u64,
    pub(crate) fire_cell_days_total: u64,
    pub(crate) fire_peak_burning: u32,

    pub(crate) discharge_map: Vec<f32>,
    pub(crate) flow_vec_map: Vec<(f32, f32)>,
    /// Per-edge flux (#103). `serde(default)`: v2 checkpoints predating
    /// this field stay loadable, empty map on load, resized by
    /// `load_state` and filled in at the next hydro slice (cosmetic only,
    /// no physical state depends on it).
    #[serde(default)]
    pub(crate) edge_flux_map: Vec<[f32; 6]>,
    /// Hydro EMA + erosion counters (#105). `serde(default)`: a
    /// pre-#105 checkpoint loads empty maps (resized by `load_state`,
    /// the EMA fills back in over ~3τ) and zeroed counters, same
    /// precedent as `edge_flux_map`.
    #[serde(default)]
    pub(crate) discharge_ema: Vec<f32>,
    #[serde(default)]
    pub(crate) edge_flux_ema: Vec<[f32; 6]>,
    #[serde(default)]
    pub(crate) erosion_incised_total: f64,
    #[serde(default)]
    pub(crate) erosion_deposited_total: f64,
    pub(crate) wind_field: Vec<WindVec>,
    pub(crate) wind_mag: Vec<f32>,

    pub(crate) synoptic_params: SynopticParams,
    pub(crate) synoptic_state: SynopticState,
    pub(crate) synoptic_enabled: bool,
    pub(crate) synoptic_base: Vec<WindVec>,
    /// Radius of coarse synoptic torus (#88). Mesh not serialized (deterministic
    /// from grid + this radius); persisting it makes load independent of
    /// `HEXSIM_SYNOPTIC_COARSE` at load time, restored state stays aligned
    /// with mesh no matter what.
    pub(crate) synoptic_coarse_radius: i32,

    pub(crate) climate_history: ClimateHistory,
    pub(crate) last_precipitation: Vec<DayRecord>,
    pub(crate) precip_gate_open: bool,
    pub(crate) climate_normals: ClimateNormalsAccumulator,

    pub(crate) hydro_params: HydroParams,
    pub(crate) atmosphere_params: AtmosphereParams,
    pub(crate) groundwater_params: GroundwaterParams,
    pub(crate) snow_params: SnowParams,
    pub(crate) temperature_params: TemperatureParams,
    pub(crate) wind_params: WindParams,
    pub(crate) vegetation_params: VegetationParams,
    pub(crate) fire_params: FireParams,
    /// `serde(default)`: pre-#105 checkpoint restarts with defaults (erosion
    /// active), consistent with fresh world.
    #[serde(default)]
    pub(crate) erosion_params: ErosionParams,
    /// `serde(default)`: pre-#106 checkpoint restarts with lake leveling active
    /// (default), consistent with fresh world.
    #[serde(default)]
    pub(crate) lake_params: LakeParams,
}

impl Checkpoint {
    /// Encodes to `MessagePack` with named struct keys (robust to field
    /// reordering at constant format).
    pub(crate) fn encode(&self) -> Result<Vec<u8>, CheckpointError> {
        Ok(rmp_serde::to_vec_named(self)?)
    }

    /// Decodes then validates marker + version. Foreign file fails at either
    /// `MessagePack` decode ([`CheckpointError::Decode`]) or marker
    /// ([`CheckpointError::BadMagic`]).
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let ckpt: Self = rmp_serde::from_slice(bytes)?;
        if ckpt.magic != MAGIC {
            return Err(CheckpointError::BadMagic);
        }
        if ckpt.format_version != CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointError::Version {
                found: ckpt.format_version,
                expected: CHECKPOINT_FORMAT_VERSION,
            });
        }
        Ok(ckpt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Simulation;
    use crate::terrain::{TerrainParams, generate_terrain};

    /// Same pattern as `sim_with_terrain` in `simulation.rs` (not reusable
    /// here: private to its module), a world with real relief so
    /// `discharge`/`edge_flux`/erosion have something non-trivial to
    /// accumulate.
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

    /// Reproduces an "old" checkpoint (pre-#105/#106): a ghost struct that
    /// carries EXACTLY the mandatory fields of [`Checkpoint`], same names,
    /// since `to_vec_named` maps by field name, omitting the 7
    /// `#[serde(default)]` fields added since (`edge_flux_map`,
    /// `discharge_ema`, `edge_flux_ema`, `erosion_incised_total`,
    /// `erosion_deposited_total`, `erosion_params`, `lake_params`).
    /// Encoding this faithfully simulates a file produced by an engine
    /// predating those PRs: the missing keys don't appear at all in the
    /// `MessagePack` map, exactly like a real old file.
    #[derive(Serialize)]
    struct OldCheckpoint {
        magic: String,
        format_version: u32,
        engine_version: String,
        grid: HexGrid,
        hour_tick: u64,
        seed: u32,
        fire_ignitions_total: u64,
        fire_cell_days_total: u64,
        fire_peak_burning: u32,
        discharge_map: Vec<f32>,
        flow_vec_map: Vec<(f32, f32)>,
        wind_field: Vec<WindVec>,
        wind_mag: Vec<f32>,
        synoptic_params: SynopticParams,
        synoptic_state: SynopticState,
        synoptic_enabled: bool,
        synoptic_base: Vec<WindVec>,
        synoptic_coarse_radius: i32,
        climate_history: ClimateHistory,
        last_precipitation: Vec<DayRecord>,
        precip_gate_open: bool,
        climate_normals: ClimateNormalsAccumulator,
        hydro_params: HydroParams,
        atmosphere_params: AtmosphereParams,
        groundwater_params: GroundwaterParams,
        snow_params: SnowParams,
        temperature_params: TemperatureParams,
        wind_params: WindParams,
        vegetation_params: VegetationParams,
        fire_params: FireParams,
    }

    /// Strips the 7 `#[serde(default)]` fields from a complete
    /// [`Checkpoint`] and re-encodes: faithfully simulates a file
    /// produced by an engine predating #105/#106 (the missing keys don't
    /// exist at all in the `MessagePack` map, not just set to a default
    /// value).
    fn omit_serde_default_fields(full: Checkpoint) -> Vec<u8> {
        let Checkpoint {
            magic,
            format_version,
            engine_version,
            grid,
            hour_tick,
            seed,
            fire_ignitions_total,
            fire_cell_days_total,
            fire_peak_burning,
            discharge_map,
            flow_vec_map,
            wind_field,
            wind_mag,
            synoptic_params,
            synoptic_state,
            synoptic_enabled,
            synoptic_base,
            synoptic_coarse_radius,
            climate_history,
            last_precipitation,
            precip_gate_open,
            climate_normals,
            hydro_params,
            atmosphere_params,
            groundwater_params,
            snow_params,
            temperature_params,
            wind_params,
            vegetation_params,
            fire_params,
            ..
        } = full;

        let old = OldCheckpoint {
            magic,
            format_version,
            engine_version,
            grid,
            hour_tick,
            seed,
            fire_ignitions_total,
            fire_cell_days_total,
            fire_peak_burning,
            discharge_map,
            flow_vec_map,
            wind_field,
            wind_mag,
            synoptic_params,
            synoptic_state,
            synoptic_enabled,
            synoptic_base,
            synoptic_coarse_radius,
            climate_history,
            last_precipitation,
            precip_gate_open,
            climate_normals,
            hydro_params,
            atmosphere_params,
            groundwater_params,
            snow_params,
            temperature_params,
            wind_params,
            vegetation_params,
            fire_params,
        };
        rmp_serde::to_vec_named(&old).expect("encode of the old blob")
    }

    /// The gap left by `checkpoint_restart_is_bit_identical` (which only
    /// round-trips a COMPLETE checkpoint): a pre-#105/#106 checkpoint,
    /// where the 7 `serde(default)` fields are absent from the
    /// `MessagePack` keys, must stay loadable and the defaults must
    /// actually apply, not just coincide with values already at default
    /// in the original.
    ///
    /// To prove this unambiguously, we force the affected fields to
    /// non-default values BEFORE saving (`erosion.enabled=1`,
    /// `lake.min_surplus_mm` marker), let it run long enough for the
    /// derived counters/EMA to become non-zero, then strip those 7 keys
    /// from the blob before loading. If `load_state` mistakenly restored
    /// the original's values (or panicked), the test would catch it.
    #[test]
    fn load_state_accepts_pre_105_106_checkpoint_and_applies_defaults() {
        let mut sim = sim_with_terrain(6, 7);
        // Values deliberately non-default for the fields we're about to
        // omit: if the restore sees them again, the blob didn't really
        // omit the keys (invalid test); if the restore sees the real
        // defaults, `serde(default)` deserialization did its job.
        assert!(sim.update_param("erosion.enabled", 1.0));
        assert!(sim.update_param("lake.min_surplus_mm", 12345.0));

        // Enough days for the daily hydro slice to feed
        // discharge_ema/edge_flux_ema and for erosion (now active) to
        // incise/deposit a measurable total on real relief.
        for _ in 0..(5 * 24) {
            sim.step_hour();
        }

        let n = sim.grid().len();

        // Sanity check: the 7 quantities we're about to omit are indeed
        // non-trivial in the original, otherwise the test would prove
        // nothing.
        let (incised, deposited) = sim.erosion_totals();
        assert!(
            incised > 0.0 || deposited > 0.0,
            "precondition: erosion (enabled) must have moved something \
             before we omit the counters, otherwise the test is vacuous"
        );
        assert!(
            sim.discharge_ema_map().iter().any(|&d| d > 0.0),
            "precondition: discharge EMA must be nonzero after 5 days"
        );
        assert!(
            (sim.lake_params().min_surplus_mm - 12345.0).abs() < f32::EPSILON,
            "precondition: the non-default marker must be set on the original"
        );

        let bytes = sim.save_state().expect("save_state must not fail");
        let full = Checkpoint::decode(&bytes).expect("decode of the complete checkpoint");
        let old_bytes = omit_serde_default_fields(full);

        let mut restored = Simulation::load_state(&old_bytes)
            .expect("a pre-#105/#106 checkpoint must load via serde(default)");

        // The clock (mandatory field, present) is restored verbatim.
        assert_eq!(
            sim.hour_tick(),
            restored.hour_tick(),
            "clock restored from an old checkpoint"
        );

        // The real defaults are applied, not the (non-default) values we
        // explicitly forced on the original before omitting them.
        assert!(
            !restored.erosion_params().enabled,
            "erosion_params absent → default (enabled=false), not the value \
             forced (true) on the original"
        );
        assert!(
            (restored.lake_params().min_surplus_mm - LakeParams::default().min_surplus_mm).abs()
                < f32::EPSILON,
            "lake_params absent → default (50 mm), not the marker (12345) from the original"
        );
        let (restored_incised, restored_deposited) = restored.erosion_totals();
        assert!(
            restored_incised.abs() < f64::EPSILON && restored_deposited.abs() < f64::EPSILON,
            "erosion counters absent → reset to zero, not the original's cumulative total"
        );
        assert_eq!(
            restored.edge_flux_map().len(),
            n,
            "edge_flux_map absent → resized to grid size"
        );
        assert!(
            restored
                .edge_flux_map()
                .iter()
                .all(|f| f.iter().all(|x| x.abs() < f32::EPSILON)),
            "edge_flux_map absent → filled with zeros, not the original's accumulated flux"
        );
        assert_eq!(
            restored.discharge_ema_map().len(),
            n,
            "discharge_ema absent → resized to grid size"
        );
        assert!(
            restored
                .discharge_ema_map()
                .iter()
                .all(|d| d.abs() < f32::EPSILON),
            "discharge_ema absent → filled with zeros, not the original's accumulated EMA"
        );

        // Final proof that loading isn't just "accepted" but actually
        // usable: the restored sim runs without panic or NaN over the
        // following day (hydro slice + erosion, now disabled by default,
        // land back on their feet).
        for _ in 0..24 {
            restored.step_hour();
        }
        for cell in restored.grid().cells_slice() {
            assert!(
                cell.temperature.is_finite(),
                "temperature NaN after restoring an old checkpoint"
            );
            assert!(
                cell.water_level.is_finite(),
                "water_level NaN after restoring an old checkpoint"
            );
        }
    }
}
