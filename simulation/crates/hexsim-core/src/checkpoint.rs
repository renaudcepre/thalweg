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
//! prognostic synoptic state, the moist-layer coarse mirror (coarse upper
//! layer, step 1; `atmosphere::coarse`), the in-progress yearly normals
//! accumulator, the precipitation hysteresis, the retained downsampled
//! wind field, the fire counters, and the process's
//! [`crate::ablation::Ablation`] (env-var A/B switches, see
//! [`crate::ablation`]) — process-global state that lives outside the seed
//! but still changes the physics. *Derived* fields (double-buffer `next`,
//! reconstructible neighbor caches, scratch buffers) are rebuilt on load,
//! never relied upon. Since fire is a **stateless** random draw
//! (`hash01(seed, day, cell)`), the seed plus the clock are enough to
//! reproduce it; no generator state to store.
//!
//! # Format
//! `MessagePack` (`rmp-serde`, same conventions as the wire format):
//! compact binary, and above all it accepts **arbitrary map keys**,
//! needed for `HexGrid::coord_index` and `ClimateHistory`
//! (`HashMap<HexCoord, _>`), which JSON refuses (string keys only).
//! Versioned envelope: loading a file with an incompatible format version
//! is **refused with a clear message** ([`crate::checkpoint::CheckpointError::Version`])
//! rather than silently misread. Loading a file whose [`crate::ablation::Ablation`] doesn't
//! match the running process is refused the same way
//! ([`crate::checkpoint::CheckpointError::Ablation`]).
//!
//! # Species columns
//! `CellProperties::vegetation` is one column per species, in the order of
//! `species::SPECIES`, and that order changes when the table grows (5
//! species up to #161, 16 since). The header records the order the file
//! was written in (`Checkpoint::species_order`); `Checkpoint::decode`
//! moves every column to its species' place **by id**, so a decoded
//! checkpoint always speaks the engine's order and a file from a smaller
//! table loads with zeros for the species it didn't know.

use serde::{Deserialize, Serialize};

use crate::ablation::Ablation;
use crate::atmosphere::{AtmosphereParams, MoistCoarseState, WeatherRegime};
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
use crate::species::{SPECIES, SPECIES_COUNT, SpeciesId, species_index};
use crate::temperature::TemperatureParams;
use crate::vegetation::VegetationParams;
use crate::wind::{WindParams, WindVec};

/// Checkpoint format version. Bump on any breaking change to the
/// `Checkpoint` schema. Loading a different version is refused
/// ([`CheckpointError::Version`]); no silent migration.
///
/// v2 (issue #88): synoptic state now lives on the coarse torus
/// (`synoptic_coarse_radius` added, `synoptic_state` vectors at coarse
/// size); a v1 carries fine-grid state that can't be converted.
///
/// `ablation` is additive and carries `serde(default)`, so it did **not**
/// bump the version: see its field doc for why a missing value is read as
/// the compiled defaults rather than refused. Same for `species_order`
/// (#161): a missing value is the 5-species table every earlier file was
/// written with, and the columns are remapped rather than refused.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// Header marker: distinguishes a `HexSim` checkpoint from some unrelated
/// file dropped in by mistake.
pub(crate) const MAGIC: &str = "HEXSIM_CKPT";

/// Errors saving/loading a checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// The blob isn't `MessagePack` decodable into a `Checkpoint`.
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
    /// The checkpoint's [`Ablation`] (env-var A/B switches) differs from the
    /// running process's: explicit refusal, same precedent as
    /// [`CheckpointError::Version`]. Resuming under a different ablation
    /// silently resumes the same seed in different physics (see
    /// [`crate::ablation`]) — refuse, don't warn.
    #[error(
        "checkpoint ablation differs from the running process, refusing to resume in a \
         different physics: {differences} (match the environment variables to the ones the \
         checkpoint was saved under, or start a fresh world)"
    )]
    Ablation {
        /// One entry per diverging field, from [`Ablation::differences`].
        differences: String,
    },
    /// The header's `species_order` can't place the vegetation columns
    /// without losing biomass: a species listed twice, or biomass in a
    /// column the order doesn't name. Refused rather than dropped, same
    /// precedent as a row longer than the engine's species count
    /// ([`CheckpointError::Decode`]).
    #[error("checkpoint species order can't place its vegetation columns: {0}")]
    SpeciesOrder(String),
}

/// Species order of the vegetation columns in every checkpoint written
/// before `Checkpoint::species_order` existed (#161): the 5-species table
/// of #81, in its row order. The `serde(default)` of that field.
fn legacy_species_order() -> Vec<SpeciesId> {
    vec![
        SpeciesId::OakPubescent,
        SpeciesId::Pine,
        SpeciesId::Beech,
        SpeciesId::Fir,
        SpeciesId::AlpineGrass,
    ]
}

/// The engine's own column order, what `save_state` writes and what a
/// decoded checkpoint is remapped to.
pub(crate) fn engine_species_order() -> Vec<SpeciesId> {
    SPECIES.iter().map(|s| s.id).collect()
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
    /// Species of each `vegetation` column of `grid`'s cells, in the order
    /// they were written (`species::SPECIES` at save time). Decoding
    /// remaps the columns by id to the running engine's order
    /// ([`Checkpoint::decode`]), so a table that gains, loses or reorders
    /// species doesn't strand a saved world.
    ///
    /// `serde(default)`: a file predating the field was written by the
    /// 5-species engine ([`legacy_species_order`]), which is the only
    /// order it can have. Additive, no version bump, same precedent as
    /// `ablation` below: `frontend/worlds/aged.ckptz`, the 42-year world
    /// the public embed booted on until v0.14.0, is such a file.
    #[serde(default = "legacy_species_order")]
    pub(crate) species_order: Vec<SpeciesId>,
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
    /// Radius of the moist-layer coarse mesh (coarse upper layer, step 1;
    /// `atmosphere::coarse`), same precedent as `synoptic_coarse_radius`
    /// above but additive (`serde(default)` → `None` on a checkpoint
    /// predating it): `load_state` then rebuilds the mesh at the
    /// current env's natural radius and regains the mirror by a direct
    /// gather, exact for a mirror (unlike the synoptic solver's
    /// prognostic state, nothing here is lost by re-deriving it from the
    /// fine grid) — no format version bump needed.
    #[serde(default)]
    pub(crate) moist_coarse_radius: Option<i32>,
    /// Coarse mirror of `humidity_upper`/`cloud_water` (coarse upper
    /// layer, step 1). `serde(default)` → `None` on a checkpoint
    /// predating it, same fallback as `moist_coarse_radius` above (the
    /// two travel together: either both are present or neither is).
    #[serde(default)]
    pub(crate) moist_coarse_state: Option<MoistCoarseState>,

    pub(crate) climate_history: ClimateHistory,
    pub(crate) last_precipitation: Vec<DayRecord>,
    pub(crate) precip_gate_open: bool,
    /// Imposed weather regime (#63): the phase of the synoptic Markov
    /// chain and the sky reservoir (`Simulation`'s `atmo_state.regime`).
    /// Real state, not derivable from the grid: without it a restart
    /// would redraw the phase mid-episode and, worse, the terrarium would
    /// silently gain or lose whatever the sky was holding.
    /// `serde(default)` → `None` on a checkpoint predating it; the world
    /// restarts on a dry chain with an empty sky, which is the state any
    /// world with the regime disabled is in — no longer the shipped
    /// default since 2026-09-06 (#63/#146), but still what such an old
    /// checkpoint is, since it also predates `regime_enabled` itself
    /// (`AtmosphereParams::regime_enabled`'s own bare `#[serde(default)]`
    /// reads it back as `0.0`, off).
    #[serde(default)]
    pub(crate) weather_regime: Option<WeatherRegime>,
    /// Diurnally smoothed map-mean surface temperature anchoring the
    /// upper layer (`Simulation::upper_air_mean_t`, EMA τ = 24 h). Real
    /// state, not derivable from the grid: without it a restart would
    /// see the upper air jump by the diurnal anomaly and drift for ~3τ
    /// (JOURNAL 2026-07-07: the state is NOT just the grid).
    /// `serde(default)` → `None` on a checkpoint predating it; `load_state`
    /// then restarts on the loaded grid's instantaneous mean, exactly
    /// like `Simulation::new`.
    #[serde(default)]
    pub(crate) upper_air_mean_t: Option<f32>,
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
    /// Env-var A/B switches in effect when this checkpoint was saved.
    /// `serde(default)`: a file predating the field is read as
    /// [`Ablation::defaults`], which is the only information it carries —
    /// the switches are perf experiments run deliberately, so a file with
    /// no record of one was produced under the compiled configuration.
    ///
    /// Refusing such a file instead would cost the one that exists:
    /// `frontend/worlds/aged.ckptz`, the 42-year world the public embed
    /// boots on, saved by engine 0.10.0 before this field existed. Same
    /// precedent as the `serde(default)` fields above. See
    /// [`crate::ablation`].
    ///
    /// # A retired switch inside an existing checkpoint (measured, not assumed)
    ///
    /// `Ablation` itself has no `#[serde(deny_unknown_fields)]`. When a
    /// switch is deleted from the struct (`oro_legacy_clamp` and
    /// `moist_coarse_stock`, both retired 2026-09-07: the levers they
    /// gated, the pre-#156 orographic clamp and the legacy coarse-stock
    /// moist mode, both lost their A/B and were deleted with them), a
    /// checkpoint saved with either forced on decodes here **without
    /// error**: the two keys are silently dropped, unlike a switch that
    /// still exists and merely differs, which the `differences()` check
    /// two paragraphs up refuses on sight. Measured by
    /// `ablation::tests::checkpoint_with_a_retired_switch_decodes_silently_as_if_it_were_absent`:
    /// encoding the pre-removal shape with either field forced on and
    /// decoding it as today's `Ablation` reads back bit-for-bit as
    /// [`Ablation::defaults`], no trace left that the file asked for
    /// something else.
    ///
    /// Judged negligible rather than fixed. Who could have produced such
    /// a file: only someone deliberately running the specific A/B session
    /// each lever existed for — `oro_legacy_clamp` for #156's few days
    /// between merge (2026-09-05) and retirement (2026-09-07),
    /// `moist_coarse_stock` for the coarse-upper-layer steps 2/2b
    /// bench — never a shipped default a long-running world would have
    /// drifted into by accident. Whether one remains: the checkpoints this
    /// module actually promises to keep loading are the reference
    /// `frontend/worlds/*.ckptz` files, which represent the *shipped*
    /// physics and so were never saved under either lever. And unlike
    /// `HEXSIM_SYNOPTIC_SUBSAMPLE` (`crate::ablation`'s module doc: "a
    /// real physics change… not a cosmetic one"), what a mismatch here
    /// would silently switch a resumed world onto is itself a small,
    /// already-characterized delta (the bounded exponential pump agrees
    /// with the legacy clamp at gentle slopes and only diverges where the
    /// old cap saturated; `CoarsePrecip` was already measured milder than
    /// `CoarseStock`'s departure from the fine path), not a class of bug
    /// this checkpoint format exists to catch.
    #[serde(default)]
    pub(crate) ablation: Ablation,
}

impl Checkpoint {
    /// Encodes to `MessagePack` with named struct keys (robust to field
    /// reordering at constant format).
    pub(crate) fn encode(&self) -> Result<Vec<u8>, CheckpointError> {
        Ok(rmp_serde::to_vec_named(self)?)
    }

    /// Decodes, validates marker + version + ablation, then brings the
    /// vegetation columns into the engine's species order
    /// ([`Checkpoint::remap_species_columns`]): a decoded `Checkpoint`
    /// always speaks the engine's order, whoever decodes it. Foreign file
    /// fails at either `MessagePack` decode ([`CheckpointError::Decode`])
    /// or marker ([`CheckpointError::BadMagic`]).
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let mut ckpt: Self = rmp_serde::from_slice(bytes)?;
        if ckpt.magic != MAGIC {
            return Err(CheckpointError::BadMagic);
        }
        if ckpt.format_version != CHECKPOINT_FORMAT_VERSION {
            return Err(CheckpointError::Version {
                found: ckpt.format_version,
                expected: CHECKPOINT_FORMAT_VERSION,
            });
        }
        let differences = ckpt.ablation.differences(Ablation::effective());
        if !differences.is_empty() {
            return Err(CheckpointError::Ablation {
                differences: differences.join("; "),
            });
        }
        ckpt.remap_species_columns()?;
        Ok(ckpt)
    }

    /// Moves every cell's vegetation column `i` to the engine column of
    /// `species_order[i]` (`species::species_index`), zero for the species
    /// the file didn't know, then records the engine's order. A no-op when
    /// the file was written in the engine's order, which is every
    /// checkpoint this engine saves.
    ///
    /// Refuses ([`CheckpointError::SpeciesOrder`]) a species listed twice
    /// (two columns would land on one) and biomass in a column past the
    /// end of the order (nobody to give it to): both would drop a stock
    /// silently. A duplicate-free order can't be longer than the engine's
    /// table, since every listed id already decoded as a known species.
    fn remap_species_columns(&mut self) -> Result<(), CheckpointError> {
        let engine = engine_species_order();
        if self.species_order == engine {
            return Ok(());
        }
        let mut target = Vec::with_capacity(self.species_order.len());
        for (i, &id) in self.species_order.iter().enumerate() {
            if self.species_order[..i].contains(&id) {
                return Err(CheckpointError::SpeciesOrder(format!(
                    "{id:?} is listed twice in {:?}",
                    self.species_order
                )));
            }
            target.push(species_index(id));
        }
        let listed = target.len();
        for (cell_index, cell) in self.grid.cells_slice_mut().iter_mut().enumerate() {
            let raw = cell.vegetation;
            if let Some(column) = raw[listed..].iter().position(|v| v.abs() > 0.0) {
                return Err(CheckpointError::SpeciesOrder(format!(
                    "cell {cell_index} holds biomass {} in column {} but the order names \
                     only {listed} species",
                    raw[listed + column],
                    listed + column
                )));
            }
            cell.vegetation = [0.0; SPECIES_COUNT];
            for (&column, &biomass) in target.iter().zip(raw.iter()) {
                cell.vegetation[column] = biomass;
            }
        }
        self.species_order = engine;
        Ok(())
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
    /// accumulate. The biomass of the species the pre-#161 table did not
    /// have is cleared: worldgen seeds all sixteen (#152), and the legacy
    /// files these tests emulate were written by an engine that could only
    /// carry the five of [`legacy_species_order`].
    fn sim_with_terrain(radius: i32, seed: u32) -> Simulation {
        let mut grid = HexGrid::from_radius(radius);
        generate_terrain(
            &mut grid,
            &TerrainParams {
                seed,
                ..TerrainParams::default()
            },
        );
        let legacy: Vec<usize> = legacy_species_order()
            .into_iter()
            .map(species_index)
            .collect();
        for cell in grid.cells_slice_mut() {
            for (column, v) in cell.vegetation.iter_mut().enumerate() {
                if !legacy.contains(&column) {
                    *v = 0.0;
                }
            }
        }
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
    /// since `to_vec_named` maps by field name, omitting the
    /// `#[serde(default)]` fields added since (`edge_flux_map`,
    /// `discharge_ema`, `edge_flux_ema`, `erosion_incised_total`,
    /// `erosion_deposited_total`, `erosion_params`, `lake_params`,
    /// `upper_air_mean_t`, `species_order`, …).
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

    /// Strips the `#[serde(default)]` fields from a complete
    /// [`Checkpoint`] and re-encodes: faithfully simulates a file
    /// produced by an engine predating #105/#106, the `ablation` field and
    /// the 16-species table (the missing keys don't exist at all in the
    /// `MessagePack` map, not just set to a default value, and every
    /// cell's `vegetation` row is the 5-column row of the legacy table,
    /// see [`to_legacy_species_rows`]). `frontend/worlds/aged.ckptz` is
    /// exactly such a file.
    fn omit_serde_default_fields(full: Checkpoint) -> Vec<u8> {
        let cells = full.grid.len();
        let ghost = omit_serde_default_keys(full);
        to_legacy_species_rows(&ghost, cells)
    }

    /// The key half of [`omit_serde_default_fields`]: the ghost struct's
    /// encoding, rows still in the engine's 16-column layout.
    fn omit_serde_default_keys(full: Checkpoint) -> Vec<u8> {
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

    /// The `vegetation` key as `to_vec_named` writes it inside each cell's
    /// map: a 10-byte `fixstr` (`0xa0 | 10`). `vegetation_params` is a
    /// 17-byte one (`0xb1` prefix), so this prefix can't match it.
    const VEGETATION_KEY: &[u8] = b"\xaavegetation";

    /// A `MessagePack` `f32` element: `0xca` then 4 big-endian bytes, the
    /// only float encoding `rmp_serde` writes for an `f32`.
    fn f32_element(v: f32) -> [u8; 5] {
        let b = v.to_be_bytes();
        [0xca, b[0], b[1], b[2], b[3]]
    }

    /// Rewrites every cell's `vegetation` row inside a `to_vec_named`
    /// checkpoint blob, the way a file written by an engine with another
    /// species table would carry it. `edit` receives the row as its
    /// `MessagePack` `f32` elements and returns the row to write. Returns
    /// the blob and the number of rows rewritten, which the caller checks
    /// against the cell count (a byte pattern matched anywhere else would
    /// show up there).
    fn rewrite_vegetation_rows(
        bytes: &[u8],
        mut edit: impl FnMut(&[[u8; 5]]) -> Vec<[u8; 5]>,
    ) -> (Vec<u8>, usize) {
        let mut out = Vec::with_capacity(bytes.len());
        let mut rows = 0;
        let mut i = 0;
        while i < bytes.len() {
            if !bytes[i..].starts_with(VEGETATION_KEY) {
                out.push(bytes[i]);
                i += 1;
                continue;
            }
            let mut j = i + VEGETATION_KEY.len();
            let len = match bytes[j] {
                marker @ 0x90..=0x9f => {
                    j += 1;
                    usize::from(marker & 0x0f)
                }
                0xdc => {
                    j += 3;
                    usize::from(u16::from_be_bytes([bytes[j - 2], bytes[j - 1]]))
                }
                other => panic!("vegetation row: unexpected MessagePack marker {other:#x}"),
            };
            let row: Vec<[u8; 5]> = bytes[j..j + 5 * len].as_chunks::<5>().0.to_vec();
            let new_row = edit(&row);
            out.extend_from_slice(VEGETATION_KEY);
            match u8::try_from(new_row.len()) {
                Ok(n) if n <= 15 => out.push(0x90 | n),
                _ => {
                    out.push(0xdc);
                    let n = u16::try_from(new_row.len()).expect("row fits array16");
                    out.extend_from_slice(&n.to_be_bytes());
                }
            }
            for e in &new_row {
                out.extend_from_slice(e);
            }
            i = j + 5 * len;
            rows += 1;
        }
        (out, rows)
    }

    /// Rewrites the engine's 16-column rows into the 5-column rows of the
    /// pre-#161 table ([`legacy_species_order`]): column `i` of the result
    /// is the engine column of `legacy_species_order()[i]`. The biomass of
    /// a species the legacy table didn't have is dropped: an old engine
    /// never carried it, and since #152 a fresh world holds some in every
    /// column from day one (seeded at worldgen, colonizing from the first
    /// daily tail), so a blob an old engine could have written is one
    /// without those columns, not one where they happen to be empty.
    fn to_legacy_species_rows(bytes: &[u8], cells: usize) -> Vec<u8> {
        let legacy: Vec<usize> = legacy_species_order()
            .into_iter()
            .map(species_index)
            .collect();
        let (out, rows) = rewrite_vegetation_rows(bytes, |row| {
            assert_eq!(row.len(), SPECIES_COUNT, "engine row");
            legacy.iter().map(|&c| row[c]).collect()
        });
        assert_eq!(rows, cells, "one vegetation row per cell");
        out
    }

    /// The gap left by `checkpoint_restart_is_bit_identical` (which only
    /// round-trips a COMPLETE checkpoint): a pre-#105/#106 checkpoint,
    /// where the 9 `serde(default)` fields are absent from the
    /// `MessagePack` keys, must stay loadable and the defaults must
    /// actually apply, not just coincide with values already at default
    /// in the original.
    ///
    /// To prove this unambiguously, we force the affected fields to
    /// non-default values BEFORE saving (`erosion.enabled=1`,
    /// `lake.min_surplus_mm` marker), let it run long enough for the
    /// derived counters/EMA to become non-zero, then strip those 8 keys
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

        // Sanity check: the 8 quantities we're about to omit are indeed
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
        // After 5 days the smoothed upper-air anchor (EMA τ = 24 h) lags
        // the instantaneous mean by a good part of the diurnal swing at
        // midnight: omitting it must be observable.
        let instantaneous_mean_t = crate::atmosphere::surface_means(sim.grid()).0;
        assert!(
            (sim.upper_air_mean_t() - instantaneous_mean_t).abs() > 1e-3,
            "precondition: the smoothed upper-air anchor must differ from the \
             instantaneous mean, otherwise omitting it is unobservable"
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
        // Upper-air anchor absent → restarted on the loaded grid's
        // instantaneous mean (the `Simulation::new` rule), not the
        // original's EMA.
        let restored_mean_t = crate::atmosphere::surface_means(restored.grid()).0;
        assert!(
            (restored.upper_air_mean_t() - restored_mean_t).abs() < f32::EPSILON,
            "upper_air_mean_t absent → instantaneous mean of the loaded grid \
             ({restored_mean_t}), got {}",
            restored.upper_air_mean_t()
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

    /// A checkpoint saved and reloaded within the same process (hence the
    /// same [`Ablation::effective`]) must decode: the happy path that
    /// `checkpoint_with_different_ablation_is_refused` contrasts with.
    #[test]
    fn checkpoint_with_matching_ablation_decodes() {
        let sim = sim_with_terrain(4, 1);
        let bytes = sim.save_state().expect("save_state must not fail");
        Checkpoint::decode(&bytes)
            .expect("a checkpoint saved under the running process's own ablation must decode");
    }

    /// A checkpoint whose ablation config doesn't match the running
    /// process's must be refused, not silently loaded under a different
    /// physics (the bug this module exists to close). Built by hand-editing
    /// a decoded [`Checkpoint`]'s `ablation` field, never
    /// `std::env::set_var` (unsafe in Rust 2024, and would leak into every
    /// other test in this binary).
    #[test]
    fn checkpoint_with_different_ablation_is_refused() {
        let sim = sim_with_terrain(4, 1);
        let bytes = sim.save_state().expect("save_state must not fail");
        let mut ckpt: Checkpoint =
            rmp_serde::from_slice(&bytes).expect("raw decode of a checkpoint we just produced");
        ckpt.ablation.wind_subsample += 1;
        let tampered_bytes = ckpt.encode().expect("re-encode must not fail");

        let err = Checkpoint::decode(&tampered_bytes)
            .err()
            .expect("a checkpoint with a mismatched ablation must be refused");
        match err {
            CheckpointError::Ablation { differences } => {
                assert!(
                    differences.contains("wind_subsample"),
                    "refusal message must name the diverging field, got: {differences}"
                );
            }
            other => panic!("expected CheckpointError::Ablation, got {other:?}"),
        }
    }

    /// A blob written before `ablation` existed must still decode, and
    /// decode as [`Ablation::defaults`].
    ///
    /// This is a sentinel, not a nicety: `frontend/worlds/aged.ckptz` is
    /// such a blob (engine 0.10.0, no `ablation` key), still shipped and
    /// loaded by `?world=aged` (the public embed's default until v0.14.0). No test of the default suite loads that
    /// file — 2 MB gzipped, 52 MB decompressed; the `#[ignore]`d
    /// `aged_world_ckptz_loads_with_its_species_remapped` does, on demand —
    /// so nothing else in the default suite notices when a schema change
    /// locks it out. Bumping `CHECKPOINT_FORMAT_VERSION` for
    /// `ablation` did exactly that, and 388 green tests said nothing.
    #[test]
    fn checkpoint_without_an_ablation_key_decodes_as_defaults() {
        let sim = sim_with_terrain(4, 1);
        let bytes = sim.save_state().expect("save_state must not fail");
        let stripped = omit_serde_default_fields(
            rmp_serde::from_slice(&bytes).expect("raw decode of a checkpoint we just produced"),
        );

        let ckpt = Checkpoint::decode(&stripped)
            .expect("a blob predating the `ablation` field must still decode");
        assert_eq!(
            ckpt.ablation,
            Ablation::defaults(),
            "a missing `ablation` key must read as the compiled defaults"
        );
    }

    /// Distinct, exactly representable biomass for column `column` of
    /// cell `cell`: a swapped column or a shifted cell shows as a
    /// mismatch.
    fn marker_biomass(cell: usize, column: usize) -> f32 {
        let cell = u16::try_from(cell % 64).expect("< 64");
        let column = u16::try_from(column).expect("small column");
        f32::from(column + 1) / 32.0 + f32::from(cell) / 4096.0
    }

    /// (a) A checkpoint written by this engine round-trips its 16
    /// vegetation columns bit for bit: the header carries the engine's
    /// order, the remap is the no-op, and nothing is padded or moved.
    #[test]
    fn sixteen_species_checkpoint_round_trips_bit_identical() {
        let sim = sim_with_terrain(4, 1);
        let mut ckpt = Checkpoint::decode(&sim.save_state().expect("save_state"))
            .expect("decode of a fresh checkpoint");
        for (i, cell) in ckpt.grid.cells_slice_mut().iter_mut().enumerate() {
            for (column, v) in cell.vegetation.iter_mut().enumerate() {
                *v = marker_biomass(i, column);
            }
        }
        let expected: Vec<[f32; SPECIES_COUNT]> = ckpt
            .grid
            .cells_slice()
            .iter()
            .map(|c| c.vegetation)
            .collect();
        let bytes = ckpt.encode().expect("encode");

        let raw: Checkpoint = rmp_serde::from_slice(&bytes).expect("raw decode");
        assert_eq!(
            raw.species_order,
            engine_species_order(),
            "the header records the engine's own order"
        );

        let restored = Simulation::load_state(&bytes).expect("load of a 16-species checkpoint");
        for (i, (cell, want)) in restored
            .grid()
            .cells_slice()
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert_eq!(
                cell.vegetation.map(f32::to_bits),
                want.map(f32::to_bits),
                "cell {i}: vegetation not restored bit for bit"
            );
        }
    }

    /// (b) A blob with the pre-#161 layout (5-column rows in the legacy
    /// order, no `species_order` key, `frontend/worlds/aged.ckptz`'s
    /// shape) loads, and every column lands on its species by id: oak,
    /// pine, beech, fir and alpine grass in their new columns, zero for
    /// the 11 species the legacy table didn't have.
    #[test]
    fn legacy_five_species_checkpoint_lands_columns_by_species_id() {
        let sim = sim_with_terrain(4, 1);
        let mut full = Checkpoint::decode(&sim.save_state().expect("save_state"))
            .expect("decode of a fresh checkpoint");
        let legacy = legacy_species_order();
        for (i, cell) in full.grid.cells_slice_mut().iter_mut().enumerate() {
            cell.vegetation = [0.0; SPECIES_COUNT];
            for (rank, &id) in legacy.iter().enumerate() {
                cell.vegetation[species_index(id)] = marker_biomass(i, rank);
            }
        }
        let cells = full.grid.len();
        let old_bytes = omit_serde_default_fields(full);

        // Precondition: the blob really is the legacy layout, raw columns
        // in the legacy order and the key absent (read as its default).
        let raw: Checkpoint = rmp_serde::from_slice(&old_bytes).expect("raw decode");
        assert_eq!(raw.species_order, legacy, "missing key → legacy order");
        for (i, cell) in raw.grid.cells_slice().iter().enumerate() {
            for (rank, id) in legacy.iter().enumerate() {
                assert_eq!(
                    cell.vegetation[rank].to_bits(),
                    marker_biomass(i, rank).to_bits(),
                    "precondition: raw column {rank} of cell {i} is the legacy {id:?}"
                );
            }
            assert!(
                cell.vegetation[legacy.len()..]
                    .iter()
                    .all(|v| v.abs() < f32::EPSILON),
                "precondition: a 5-column row is zero-padded"
            );
        }

        let restored =
            Simulation::load_state(&old_bytes).expect("a 5-species checkpoint must load");
        assert_eq!(restored.grid().len(), cells);
        for (i, cell) in restored.grid().cells_slice().iter().enumerate() {
            for (column, v) in cell.vegetation.iter().enumerate() {
                let id = SPECIES[column].id;
                let want = legacy
                    .iter()
                    .position(|&l| l == id)
                    .map_or(0.0, |rank| marker_biomass(i, rank));
                assert_eq!(
                    v.to_bits(),
                    want.to_bits(),
                    "cell {i}: {id:?} (column {column}) holds {v}, expected {want}"
                );
            }
        }
    }

    /// (c) A row longer than the engine's species count was written by an
    /// engine with species this one doesn't know: refused with a message
    /// that says so, never truncated.
    #[test]
    fn a_seventeen_column_row_is_refused_with_a_clear_error() {
        let sim = sim_with_terrain(2, 1);
        let bytes = sim.save_state().expect("save_state");
        let (longer, rows) = rewrite_vegetation_rows(&bytes, |row| {
            let mut row = row.to_vec();
            row.push(f32_element(0.25));
            row
        });
        assert_eq!(rows, sim.grid().len(), "one vegetation row per cell");

        let err = Checkpoint::decode(&longer)
            .err()
            .expect("a 17-column row must be refused");
        let msg = err.to_string();
        assert!(
            matches!(err, CheckpointError::Decode(_)),
            "expected a decode error, got {err:?}"
        );
        assert!(
            msg.contains("invalid length 17")
                && msg.contains(&format!("at most {SPECIES_COUNT}"))
                && msg.contains("more species"),
            "the refusal must name the length, the limit and why: {msg}"
        );
    }

    /// A `species_order` that can't place its columns without losing
    /// biomass is refused: a species listed twice (two columns would land
    /// on one) or biomass in a column past the end of the order.
    #[test]
    fn a_species_order_that_would_drop_biomass_is_refused() {
        let sim = sim_with_terrain(2, 1);
        let bytes = sim.save_state().expect("save_state");

        let mut twice: Checkpoint = rmp_serde::from_slice(&bytes).expect("raw decode");
        twice.species_order = vec![SpeciesId::Fir, SpeciesId::Pine, SpeciesId::Fir];
        let err = Checkpoint::decode(&twice.encode().expect("encode"))
            .err()
            .expect("a duplicated species must be refused");
        assert!(
            matches!(&err, CheckpointError::SpeciesOrder(m) if m.contains("Fir is listed twice")),
            "got {err:?}"
        );

        let mut short: Checkpoint = rmp_serde::from_slice(&bytes).expect("raw decode");
        short.species_order = legacy_species_order();
        short.grid.cells_slice_mut()[0].vegetation[7] = 0.5;
        let err = Checkpoint::decode(&short.encode().expect("encode"))
            .err()
            .expect("biomass in a column the order doesn't name must be refused");
        assert!(
            matches!(&err, CheckpointError::SpeciesOrder(m) if m.contains("column 7")),
            "got {err:?}"
        );
    }

    /// Sentinel on the real file: `frontend/worlds/aged.ckptz`, the
    /// 42-year world the public embed booted on (#147) until v0.14.0, written by the
    /// 5-species engine. It must load under the 16-species table with
    /// every legacy column's biomass on its species (per-species totals
    /// bit-identical to the raw columns), zero for the new species, and
    /// run a day.
    ///
    /// `#[ignore]`: 2 MB gzipped, 52 MB decompressed, and gunzipped
    /// through the system `gzip` (no inflate crate in `hexsim-core`'s
    /// dependencies). Run it whenever the checkpoint schema, `CellProperties`
    /// or the species table changes:
    /// `cargo test -p hexsim-core --lib aged_world -- --ignored`.
    #[test]
    #[ignore = "loads the 52 MB shipped world through the system gzip"]
    fn aged_world_ckptz_loads_with_its_species_remapped() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../frontend/worlds/aged.ckptz"
        );
        let out = std::process::Command::new("gzip")
            .args(["-dc", path])
            .output()
            .expect("run gzip");
        assert!(out.status.success(), "gzip -dc {path} failed");
        let bytes = out.stdout;

        // Raw decode, no remap: what the file itself says.
        let raw: Checkpoint = rmp_serde::from_slice(&bytes).expect("raw decode of aged.ckptz");
        let legacy = legacy_species_order();
        assert_eq!(
            raw.species_order, legacy,
            "aged.ckptz predates `species_order`: read as the legacy order"
        );
        let mut raw_totals = vec![0.0_f64; legacy.len()];
        for cell in raw.grid.cells_slice() {
            assert!(
                cell.vegetation[legacy.len()..]
                    .iter()
                    .all(|v| v.abs() < f32::EPSILON),
                "a 5-column row is zero-padded"
            );
            for (rank, total) in raw_totals.iter_mut().enumerate() {
                *total += f64::from(cell.vegetation[rank]);
            }
        }
        assert!(
            raw_totals.iter().all(|&t| t > 0.0),
            "a 42-year world holds every legacy species somewhere: {raw_totals:?}"
        );

        let mut sim = Simulation::load_state(&bytes).expect("aged.ckptz must load");
        let mut totals = [0.0_f64; SPECIES_COUNT];
        for cell in sim.grid().cells_slice() {
            for (total, &v) in totals.iter_mut().zip(cell.vegetation.iter()) {
                *total += f64::from(v);
            }
        }
        for (column, &total) in totals.iter().enumerate() {
            let id = SPECIES[column].id;
            let want = legacy
                .iter()
                .position(|&l| l == id)
                .map_or(0.0, |rank| raw_totals[rank]);
            assert_eq!(
                total.to_bits(),
                want.to_bits(),
                "{id:?}: total biomass {total} after load, {want} in the file"
            );
        }

        for _ in 0..24 {
            sim.step_hour();
        }
        for cell in sim.grid().cells_slice() {
            assert!(cell.temperature.is_finite() && cell.water_level.is_finite());
            assert!(cell.vegetation.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
    }
}
