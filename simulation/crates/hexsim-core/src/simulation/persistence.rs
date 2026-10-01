//! Checkpoint glue: [`Simulation::save_state`]/[`Simulation::load_state`],
//! see [`crate::checkpoint`] for the format itself.

use super::Simulation;
use crate::ablation::Ablation;
use crate::atmosphere::{
    AtmoScratch, AtmoState, MoistCoarseState, moist_coarse_radius, surface_means,
};
use crate::checkpoint::{
    CHECKPOINT_FORMAT_VERSION, Checkpoint, CheckpointError, MAGIC, engine_species_order,
};
use crate::climate::DayRecord;
use crate::groundwater::GroundwaterScratch;
use crate::hydro::HydroScratch;
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
            species_order: engine_species_order(),
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
            moist_coarse_radius: Some(self.moist_mesh.grid().radius()),
            moist_coarse_state: Some(self.moist_coarse.clone()),
            climate_history: self.climate_history.clone(),
            last_precipitation: self.last_precipitation.clone(),
            precip_gate_open: self.atmo_state.precip_gate_open,
            weather_regime: Some(self.atmo_state.regime),
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
    /// A checkpoint written with a different species table (the 5-species
    /// engine before #161, `frontend/worlds/aged.ckptz` among them) loads
    /// with its vegetation columns moved to their species by id and zeros
    /// for the species it didn't know: see [`crate::checkpoint`],
    /// "Species columns".
    ///
    /// # Errors
    /// Returns [`CheckpointError`] if the blob isn't a valid `HexSim`
    /// checkpoint ([`CheckpointError::Decode`] / [`CheckpointError::BadMagic`]),
    /// has an incompatible format version ([`CheckpointError::Version`]),
    /// was saved under a different ablation ([`CheckpointError::Ablation`])
    /// or carries vegetation columns its species order can't place
    /// ([`CheckpointError::SpeciesOrder`], or [`CheckpointError::Decode`]
    /// for a row longer than this engine's species count).
    pub fn load_state(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let ckpt = Checkpoint::decode(bytes)?;
        // Already in the engine's species order: `Checkpoint::decode`
        // remapped the vegetation columns by id (#161).
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
        // Mesh rebuilt at the PERSISTED radius (not the current env's): the
        // verbatim-restored synoptic state stays aligned with its torus.
        let mut synoptic_mesh =
            SynopticMesh::with_coarse_radius(&current, ckpt.synoptic_coarse_radius);
        synoptic_mesh.aggregate_temperature(&current);
        let mut synoptic_coarse_base: WindField =
            vec![WindVec::default(); synoptic_mesh.grid().len()];
        ckpt.synoptic_state
            .write_base_wind(&ckpt.synoptic_params, &mut synoptic_coarse_base);
        // Moist-layer coarse mesh (coarse upper layer): rebuilt at the
        // PERSISTED radius when present, same precedent as `synoptic_mesh`
        // above. A checkpoint predating this field (`serde(default)` ->
        // None) rebuilds at the current env's natural radius instead, and
        // the mirror below is regained by a direct gather rather than
        // restored verbatim — exact either way, since it is a pure mean of
        // the fine grid on both live modes (see `atmosphere::coarse`).
        let moist_coarse_mode = Ablation::effective().moist_coarse_mode();
        let moist_coarse_rc = ckpt.moist_coarse_radius.unwrap_or_else(|| {
            if moist_coarse_mode.precipitates_coarse() {
                moist_coarse_radius(current.radius())
            } else {
                current.radius()
            }
        });
        let moist_mesh = SynopticMesh::with_coarse_radius(&current, moist_coarse_rc);
        let moist_coarse = if let Some(state) = ckpt.moist_coarse_state {
            state
        } else {
            let mut state = MoistCoarseState::new(moist_mesh.coarse_len());
            state.gather_from_fine(&moist_mesh, &current);
            state
        };
        // `next` is a double-buffer: it must mirror `current` before each
        // phase (exact parity with `Simulation::new`, which does
        // `grid.clone()`).
        let next = current.clone();
        // Derived, rebuilt on load like the double-buffer: the daily memo
        // of the hourly transpiration starts from the loaded biomass.
        let mut transpiration_cover = Vec::with_capacity(n);
        crate::vegetation::fill_transpiration_cover_into(&current, &mut transpiration_cover);
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
            transpiration_cover,
            uniform_wind: None,
            synoptic_params: ckpt.synoptic_params,
            synoptic_state: ckpt.synoptic_state,
            synoptic_enabled: ckpt.synoptic_enabled,
            synoptic_base: ckpt.synoptic_base,
            synoptic_mesh,
            synoptic_coarse_base,
            moist_mesh,
            moist_coarse,
            moist_coarse_mode,
            // Instrument, not physics: a reloaded world starts with the
            // probe off, whatever the process that saved it did.
            cloud_transfer_probe: false,
            climate_history: ckpt.climate_history,
            last_precipitation: ckpt.last_precipitation,
            vapor_today: vec![crate::atmosphere::VaporSources::default(); n],
            // Flat in the file, grouped in memory (see `AtmoState`): a
            // checkpoint predating the weather regime has no
            // `weather_regime` key at all and restarts on a dry chain
            // with an empty sky, which is exactly the state an OFF world
            // is in anyway.
            atmo_state: AtmoState {
                precip_gate_open: ckpt.precip_gate_open,
                regime: ckpt.weather_regime.unwrap_or_default(),
            },
            upper_air_mean_t,
            scratch_wind_snap: vec![WindVec::default(); n],
            scratch_atmo: AtmoScratch::new(n),
            scratch_flux: vec![0.0; n],
            scratch_flow_vec: vec![(0.0, 0.0); n],
            scratch_edge_flux: vec![[0.0; 6]; n],
            hydro_scratch: HydroScratch::new(n),
            groundwater_scratch: GroundwaterScratch::new(n),
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
    use crate::checkpoint::Checkpoint;
    #[cfg(feature = "parallel")]
    use crate::climate::Window;
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
        // Same for the imposed weather regime (#63), ON by default since
        // 2026-09-06 (#146) but forced explicitly here too, same reason as
        // synoptic above: with it off, its state is constant and the
        // round-trip would say nothing about it. On, the phase of the
        // Markov chain and the sky reservoir are two more pieces of
        // hidden state that make the continuation diverge within hours
        // if either is dropped.
        a.update_param("atmosphere.regime_enabled", 1.0);

        // 20 days + 7 h: instant not aligned on a day/year boundary.
        for _ in 0..(20 * 24 + 7) {
            a.step_hour();
        }

        let bytes = a.save_state().expect("save_state must not fail");
        let mut b = Simulation::load_state(&bytes).expect("load_state of a valid blob");
        assert_eq!(a.hour_tick(), b.hour_tick(), "restored clock");
        assert_eq!(
            a.sky_water_total().to_bits(),
            b.sky_water_total().to_bits(),
            "restored sky reservoir"
        );
        assert_eq!(
            a.weather_regime_is_wet(),
            b.weather_regime_is_wet(),
            "restored weather regime phase"
        );
        assert!(
            a.sky_water_total() > 0.0,
            "20 days with the regime on must have put something in the sky, \
             otherwise this round-trip does not exercise it"
        );
        // Moist-layer coarse mirror (coarse upper layer, step 1): restored
        // verbatim, bit for bit, not just re-derivable (the mesh's radius
        // must also match, otherwise the two `Vec<f32>` could coincidentally
        // have the same length without being the same mesh).
        assert_eq!(
            a.moist_mesh.grid().radius(),
            b.moist_mesh.grid().radius(),
            "restored moist mesh radius"
        );
        assert_eq!(
            a.moist_coarse.humidity_upper, b.moist_coarse.humidity_upper,
            "restored moist coarse humidity_upper"
        );
        assert_eq!(
            a.moist_coarse.cloud_water, b.moist_coarse.cloud_water,
            "restored moist coarse cloud_water"
        );

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

    /// Coarse upper layer, step 1: a checkpoint predating
    /// `moist_coarse_radius`/`moist_coarse_state` (both `serde(default)`)
    /// must still load, rebuilding the mesh at the radius the running
    /// process's path calls for (`moist_coarse_radius` on the shipped
    /// coarse mode, the identity under `HEXSIM_MOIST_COARSE=0` — see
    /// `atmosphere::coarse::MOIST_COARSE_DEFAULT`) and regaining
    /// the stock by a direct gather — the mean is exact for a stock
    /// (unlike the synoptic solver's prognostic state, nothing here is
    /// lost by re-deriving it from the fine grid, see `load_state`'s doc).
    /// Built by decoding a real checkpoint then setting the two `Option`
    /// fields to `None` before re-encoding: `Option<T>` round-trips `None`
    /// through `MessagePack` the same way an old file omitting the key
    /// entirely does (`serde(default)` handles both), which is simpler
    /// here than a hand-rolled ghost struct (cf. `checkpoint.rs`'s
    /// `OldCheckpoint`, needed there because `to_vec_named` would
    /// otherwise emit ALL fields including the mandatory ones this test
    /// doesn't care about).
    #[test]
    fn load_state_without_moist_coarse_fields_regains_the_mirror_by_gather() {
        let mut sim = sim_with_terrain(30, 42);
        for _ in 0..(3 * 24) {
            sim.step_hour();
        }
        assert!(
            sim.moist_coarse.humidity_upper.iter().any(|&v| v > 0.0),
            "precondition: 3 days must have put something in the moist mirror, \
             otherwise omitting it is unobservable"
        );

        let bytes = sim.save_state().expect("save_state must not fail");
        let mut ckpt = Checkpoint::decode(&bytes).expect("decode of the complete checkpoint");
        ckpt.moist_coarse_radius = None;
        ckpt.moist_coarse_state = None;
        let old_bytes = ckpt.encode().expect("re-encode without the moist fields");

        let restored = Simulation::load_state(&old_bytes)
            .expect("a checkpoint predating the moist mirror must still load");

        // Mesh rebuilt at the natural radius for this grid (the running
        // process's ablation config, same as a fresh `Simulation::new`).
        let expected_rc = if restored.moist_coarse_path() {
            moist_coarse_radius(restored.grid().radius())
        } else {
            restored.grid().radius()
        };
        assert_eq!(restored.moist_mesh.grid().radius(), expected_rc);
        // Mirror regained by a direct gather on the restored grid: equal to
        // what `gather_from_fine` would produce right now, not to the
        // original's (unreachable, deliberately omitted) mirror.
        let mut expected = MoistCoarseState::new(restored.moist_mesh.coarse_len());
        expected.gather_from_fine(&restored.moist_mesh, restored.grid());
        assert_eq!(
            restored.moist_coarse.humidity_upper,
            expected.humidity_upper
        );
        assert_eq!(restored.moist_coarse.cloud_water, expected.cloud_water);
    }

    /// The gate of the r250 parallelization chunk: every loop parallelized
    /// via `par::for_each_chunk_mut`/`for_each_chunk_mut2` must be a true
    /// pure per-cell map, so the simulation's result cannot depend on how
    /// many threads computed it. Two identical worlds, one stepped inside
    /// a single-thread rayon pool, the other inside a 4-thread pool; if
    /// their state diverges, one of the parallelized loops secretly reads
    /// or writes another cell's output (a scatter or a reduction slipped
    /// through) — find it and revert it, don't weaken this test.
    ///
    /// Checks the FULL checkpoint, field by field, not just the grid's
    /// cells — with two deliberate exceptions for fields backed by a
    /// `HashMap<HexCoord, _>` (`HexGrid::coord_index`,
    /// `ClimateHistory::history`): the default hasher draws a fresh random
    /// seed per `HashMap` instance, so their serialized byte layout
    /// differs between the two independently-constructed simulations
    /// regardless of thread count (confirmed by bisection: it still
    /// differs when BOTH runs use `num_threads(1)`, i.e. with no parallel
    /// code path involved at all). Comparing raw bytes there would flag
    /// that pre-existing, unrelated non-determinism as a false positive,
    /// so this test checks their actual CONTENT instead, through
    /// `HexGrid`'s and `ClimateHistory`'s public, order-independent
    /// accessors.
    #[test]
    #[cfg(feature = "parallel")]
    fn parallel_tick_is_bit_identical_to_serial() {
        let mut serial_sim = sim_with_terrain(10, 42);
        serial_sim.update_param("synoptic.enabled", 1.0);
        let mut parallel_sim = sim_with_terrain(10, 42);
        parallel_sim.update_param("synoptic.enabled", 1.0);

        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build 1-thread pool")
            .install(|| {
                for _ in 0..(3 * 24) {
                    serial_sim.step_hour();
                }
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build 4-thread pool")
            .install(|| {
                for _ in 0..(3 * 24) {
                    parallel_sim.step_hour();
                }
            });

        let serial_bytes = serial_sim.save_state().expect("save_state must not fail");
        let parallel_bytes = parallel_sim.save_state().expect("save_state must not fail");
        let s = Checkpoint::decode(&serial_bytes).expect("decode serial checkpoint");
        let p = Checkpoint::decode(&parallel_bytes).expect("decode parallel checkpoint");

        assert_eq!(s.grid.radius(), p.grid.radius(), "grid radius diverged");
        assert_eq!(
            rmp_serde::to_vec(s.grid.coords_slice()).unwrap(),
            rmp_serde::to_vec(p.grid.coords_slice()).unwrap(),
            "grid coordinate order diverged"
        );
        assert_eq!(
            rmp_serde::to_vec(s.grid.cells_slice()).unwrap(),
            rmp_serde::to_vec(p.grid.cells_slice()).unwrap(),
            "grid cells diverged: a parallelized loop is not a pure per-cell map"
        );
        for coord in s.grid.coords().copied() {
            for window in [Window::Last30, Window::Last180, Window::Last365] {
                assert_eq!(
                    s.climate_history.rain_days(coord, window),
                    p.climate_history.rain_days(coord, window),
                    "rain_days diverged at {coord:?}/{window:?}"
                );
                assert_eq!(
                    s.climate_history.snow_days(coord, window),
                    p.climate_history.snow_days(coord, window),
                    "snow_days diverged at {coord:?}/{window:?}"
                );
                assert!(
                    (s.climate_history.total_rain(coord, window)
                        - p.climate_history.total_rain(coord, window))
                    .abs()
                        < 1e-6,
                    "total_rain diverged at {coord:?}/{window:?}"
                );
                assert!(
                    (s.climate_history.total_snow(coord, window)
                        - p.climate_history.total_snow(coord, window))
                    .abs()
                        < 1e-6,
                    "total_snow diverged at {coord:?}/{window:?}"
                );
            }
        }

        macro_rules! assert_field_eq {
            ($field:ident) => {
                assert_eq!(
                    rmp_serde::to_vec(&s.$field).unwrap(),
                    rmp_serde::to_vec(&p.$field).unwrap(),
                    "{} diverged: a parallelized loop is not a pure per-cell map",
                    stringify!($field)
                );
            };
        }
        assert_field_eq!(hour_tick);
        assert_field_eq!(discharge_map);
        assert_field_eq!(flow_vec_map);
        assert_field_eq!(edge_flux_map);
        assert_field_eq!(discharge_ema);
        assert_field_eq!(edge_flux_ema);
        assert_field_eq!(erosion_incised_total);
        assert_field_eq!(erosion_deposited_total);
        assert_field_eq!(wind_field);
        assert_field_eq!(wind_mag);
        assert_field_eq!(synoptic_state);
        assert_field_eq!(synoptic_base);
        assert_field_eq!(last_precipitation);
        assert_field_eq!(precip_gate_open);
        assert_field_eq!(upper_air_mean_t);
        assert_field_eq!(climate_normals);
    }

    /// A blob that isn't a `HexSim` checkpoint must be rejected cleanly,
    /// never silently misinterpreted.
    #[test]
    fn load_state_rejects_foreign_bytes() {
        let result = Simulation::load_state(b"this is not a checkpoint");
        assert!(result.is_err(), "a foreign blob must be rejected");
    }
}
