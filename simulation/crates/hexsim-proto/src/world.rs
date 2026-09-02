//! The simulated world and command execution.
//!
//! [`World`] groups what a shell needs to keep alive between two
//! commands: the [`Simulation`], the [`TerrainParams`] (which only apply
//! on the next `reset`) and the grid radius. [`World::apply`] executes
//! a [`Command`] and returns an [`Outcome`], *what remains to be done*,
//! not what has been done.
//!
//! This split is the transport/protocol boundary: `World` knows nothing
//! about broadcasting a snapshot, logging, or sleeping between ticks. The
//! WebSocket server does it with tokio and a `broadcast`, the WASM module
//! with `postMessage` from a Web Worker. Both call the same
//! `apply`.

use hexsim_core::grid::HexGrid;
use hexsim_core::simulation::Simulation;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use serde_json::Value;
use thiserror::Error;

use crate::command::{Command, Query};
use crate::query::{self, BuildInfo};
use crate::wire;

/// Number of steps in a multi-tick `step`: enough for a smooth progress
/// bar, without flooding the client with snapshots (costly at large
/// radius).
pub const STEP_BATCHES: u64 = 20;

/// What remains for the shell to do after a command. The world itself is
/// already up to date.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Nothing to broadcast.
    Nothing,
    /// The world changed: broadcast a fresh snapshot to all clients.
    Snapshot,
    /// Targeted reply to the requesting client.
    Reply(Value),
    /// To broadcast to all clients (sync sliders after `set_param`).
    Broadcast(Value),
    /// Auto-tick scheduling change, lives outside the world: an atomic
    /// on the server side, a `setTimeout` on the browser side.
    Schedule(Schedule),
    /// Long advance: up to the shell to split it into batches (see
    /// [`batch_stride`]) and emit the intermediate snapshots.
    Advance { n: u64, hourly: bool },
}

/// Scheduling change requested by the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    Play,
    Pause,
    /// `tick_ms` is already clamped; `requested` is the raw value received.
    Speed {
        tick_ms: u64,
        requested: u64,
    },
}

/// A valid command that cannot be applied to this world.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ApplyError {
    #[error("unknown parameter: `{key}`")]
    UnknownParam { key: String },
}

/// Simulation plus what surrounds it: generation parameters, radius, build
/// identity.
///
/// `BuildInfo` is stored here because the `meta` query is part of the
/// protocol: without it, each shell would have to rebuild this JSON on its
/// own side, and the two would diverge.
pub struct World {
    sim: Simulation,
    terrain: TerrainParams,
    radius: i32,
    build: BuildInfo,
}

impl World {
    /// Generates a fresh world: terrain from `terrain`, then simulation at
    /// default parameters.
    #[must_use]
    pub fn generate(radius: i32, terrain: TerrainParams, build: BuildInfo) -> Self {
        let mut grid = HexGrid::from_radius(radius);
        generate_terrain(&mut grid, &terrain);
        Self::from_grid(grid, radius, terrain, build)
    }

    /// Variant of [`World::generate`] for an already-built grid, this is
    /// the server's entry point when an external DEM terrain has been
    /// applied to the grid before startup.
    #[must_use]
    pub fn from_grid(grid: HexGrid, radius: i32, terrain: TerrainParams, build: BuildInfo) -> Self {
        let wind = hexsim_core::wind::WindParams {
            seed: terrain.seed,
            ..hexsim_core::wind::WindParams::default()
        };
        let mut sim = Simulation::new(
            grid,
            hexsim_core::hydro::HydroParams::default(),
            hexsim_core::atmosphere::AtmosphereParams::default(),
            hexsim_core::groundwater::GroundwaterParams::default(),
            hexsim_core::snow::SnowParams::default(),
            hexsim_core::temperature::TemperatureParams::default(),
            wind,
        );
        // Live world: emergent fire active, randomness tied to the world seed (#wildfire).
        sim.set_seed(terrain.seed);
        sim.update_param("fire.enabled", 1.0);
        Self {
            sim,
            terrain,
            radius,
            build,
        }
    }

    /// Rebuilds a [`World`] around a simulation loaded from a checkpoint.
    /// The terrain comes from the checkpoint, not from `terrain`.
    #[must_use]
    pub fn from_simulation(sim: Simulation, terrain: TerrainParams, build: BuildInfo) -> Self {
        let radius = sim.grid().radius();
        Self {
            sim,
            terrain,
            radius,
            build,
        }
    }

    #[must_use]
    pub fn sim(&self) -> &Simulation {
        &self.sim
    }

    pub fn sim_mut(&mut self) -> &mut Simulation {
        &mut self.sim
    }

    #[must_use]
    pub fn radius(&self) -> i32 {
        self.radius
    }

    #[must_use]
    pub fn seed(&self) -> u32 {
        self.terrain.seed
    }

    #[must_use]
    pub fn terrain_params(&self) -> &TerrainParams {
        &self.terrain
    }

    /// Replaces the simulation (checkpoint import). The radius follows that
    /// of the loaded grid, which may differ from the current world's.
    pub fn replace_simulation(&mut self, sim: Simulation) {
        self.radius = sim.grid().radius();
        self.sim = sim;
    }

    /// Column-packed msgpack snapshot, ready to send.
    ///
    /// # Errors
    ///
    /// Propagates [`wire::encode_snapshot`]'s error: see its doc for when
    /// that happens and why it isn't masked here either.
    pub fn snapshot_bytes(&self) -> Result<bytes::Bytes, rmp_serde::encode::Error> {
        wire::encode_snapshot(&self.sim.snapshot())
    }

    /// Advances by `n` units: hours if `hourly`, days otherwise.
    pub fn advance(&mut self, n: u64, hourly: bool) {
        for _ in 0..n {
            if hourly {
                self.sim.step_hour();
            } else {
                self.sim.step();
            }
        }
    }

    /// Regenerates the world while preserving physics parameters tuned at
    /// runtime: otherwise a `reset` after tuning would keep the fresh
    /// terrain but lose all the tuning.
    pub fn reset(&mut self, seed: Option<u32>) {
        if let Some(seed) = seed {
            self.terrain.seed = seed;
        }
        let mut grid = HexGrid::from_radius(self.radius);
        generate_terrain(&mut grid, &self.terrain);

        let hydro = self.sim.hydro_params().clone();
        let atmosphere = self.sim.atmosphere_params().clone();
        let groundwater = self.sim.groundwater_params().clone();
        let snow = self.sim.snow_params().clone();
        let temperature = self.sim.temperature_params().clone();
        let wind = self.sim.wind_params().clone();
        let fire = *self.sim.fire_params();
        let erosion = self.sim.erosion_params().clone();
        let lake = self.sim.lake_params().clone();

        let mut sim = Simulation::new(
            grid,
            hydro,
            atmosphere,
            groundwater,
            snow,
            temperature,
            wind,
        );
        sim.set_seed(self.terrain.seed);
        sim.set_fire_params(fire);
        sim.set_erosion_params(erosion);
        sim.set_lake_params(lake);
        self.sim = sim;
    }

    /// Applies a parameter at runtime.
    ///
    /// `terrain.*` keys target world generation: they only take effect on
    /// the next `reset` (the terrain is frozen once generated), and so do
    /// not trigger a slider resync.
    ///
    /// # Errors
    ///
    /// [`ApplyError::UnknownParam`] if no module recognizes the key.
    pub fn set_param(&mut self, key: &str, value: f32) -> Result<Outcome, ApplyError> {
        if let Some(field) = key.strip_prefix("terrain.") {
            return if self.set_terrain_param(field, value) {
                Ok(Outcome::Nothing)
            } else {
                Err(ApplyError::UnknownParam {
                    key: key.to_owned(),
                })
            };
        }
        if self.sim.update_param(key, value) {
            // Sync UI (#2): the change may come from another client or
            // from the CLI (`just param`), each front realigns its sliders
            // on the single source of truth rather than on its static HTML
            // values.
            Ok(Outcome::Broadcast(query::params(&self.sim)))
        } else {
            Err(ApplyError::UnknownParam {
                key: key.to_owned(),
            })
        }
    }

    /// Terrain generation parameters. `false` = unknown key.
    fn set_terrain_param(&mut self, field: &str, value: f32) -> bool {
        match field {
            // One-shot erosion (#105): intensity of the network carved at
            // generation time. `as u32` on an f32 has been saturating since
            // Rust 1.45 (NaN → 0, out of bounds → bound) and `max(0.0)`
            // discards the sign: the conversion is intentional, not an
            // accidental truncation.
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            "erosion_iterations" => {
                self.terrain.erosion_iterations = value.max(0.0) as u32;
                true
            }
            "erosion_accel" => {
                self.terrain.erosion_accel_years = value.max(0.0);
                true
            }
            // Water budget seeded at worldgen (#107). The terrarium is closed →
            // these two values fix the total liquid water budget, conserved
            // forever. Raising `initial_water`/`initial_groundwater` is the
            // only measured lever that grows water body persistence (JOURNAL
            // 2026-07-12). See also `atmosphere.initial_humidity_floor` (the
            // other line item of the budget, ~80%).
            "initial_water" => {
                self.terrain.initial_water = value.max(0.0);
                true
            }
            "initial_groundwater" => {
                self.terrain.initial_groundwater = value.max(0.0);
                true
            }
            _ => false,
        }
    }

    /// Executes a command and tells the shell what remains to be done.
    ///
    /// # Errors
    ///
    /// [`ApplyError`] if the command is well-formed but cannot be applied
    /// (unknown parameter key).
    pub fn apply(&mut self, cmd: &Command) -> Result<Outcome, ApplyError> {
        match cmd {
            Command::Play => Ok(Outcome::Schedule(Schedule::Play)),
            Command::Pause => Ok(Outcome::Schedule(Schedule::Pause)),
            Command::Speed { tick_ms, requested } => Ok(Outcome::Schedule(Schedule::Speed {
                tick_ms: *tick_ms,
                requested: *requested,
            })),
            Command::Step { n, hourly } => Ok(Outcome::Advance {
                n: *n,
                hourly: *hourly,
            }),
            Command::Reset { seed } => {
                self.reset(*seed);
                Ok(Outcome::Snapshot)
            }
            Command::SetParam { key, value } => self.set_param(key, *value),
            Command::Query(q) => Ok(Outcome::Reply(self.answer(q))),
        }
    }

    /// Answer to a query, without going through [`World::apply`], useful
    /// for shells that expose reads through another route (HTTP, MCP).
    #[must_use]
    pub fn answer(&self, query: &Query) -> Value {
        query::answer(&self.sim, query, &self.build)
    }
}

/// Batch size for an advance of `n` units.
///
/// A `+1 year` (365 days ~15s at R=30) executed as a single block holds
/// the lock from start to finish and only emits a final snapshot, the
/// front stays frozen with no feedback. Split into [`STEP_BATCHES`] steps,
/// each batch broadcasts a snapshot (the map animates during the
/// computation) and a progress point. `max(1)` keeps a single batch for
/// small `n`.
#[must_use]
pub fn batch_stride(n: u64) -> u64 {
    (n / STEP_BATCHES).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{test_build, tiny_world};

    #[test]
    fn play_pause_speed_do_not_touch_the_world() {
        let mut w = tiny_world();
        let tick = w.sim().tick();
        assert_eq!(
            w.apply(&Command::Play),
            Ok(Outcome::Schedule(Schedule::Play))
        );
        assert_eq!(
            w.apply(&Command::Pause),
            Ok(Outcome::Schedule(Schedule::Pause))
        );
        assert_eq!(
            w.apply(&Command::Speed {
                tick_ms: 30,
                requested: 30
            }),
            Ok(Outcome::Schedule(Schedule::Speed {
                tick_ms: 30,
                requested: 30
            }))
        );
        assert_eq!(w.sim().tick(), tick, "scheduling must not advance the sim");
    }

    /// `step` is delegated to the shell: `apply` must absolutely not
    /// advance the sim itself, otherwise a `+1 year` would block the
    /// server for the whole duration.
    #[test]
    fn step_is_delegated_to_the_shell() {
        let mut w = tiny_world();
        let tick = w.sim().tick();
        assert_eq!(
            w.apply(&Command::Step {
                n: 100,
                hourly: true
            }),
            Ok(Outcome::Advance {
                n: 100,
                hourly: true
            })
        );
        assert_eq!(w.sim().tick(), tick);
    }

    #[test]
    fn hourly_advance_advances_correctly() {
        let mut w = tiny_world();
        let before = w.sim().hour_tick();
        w.advance(3, true);
        assert_eq!(w.sim().hour_tick(), before + 3);
    }

    /// The case that motivated the preservation: setting a parameter then
    /// `reset` must keep the setting, otherwise all tuning is lost on the
    /// first seed change.
    #[test]
    fn reset_preserves_hot_reloaded_params() {
        let mut w = tiny_world();
        w.set_param("wind.humidity_advection_rate", 2.5)
            .expect("known key");
        w.apply(&Command::Reset { seed: Some(7) }).expect("reset");
        assert_eq!(w.seed(), 7);
        let rate = w.answer(&Query::Params)["wind"]["humidity_advection_rate"]
            .as_f64()
            .expect("field present");
        assert!((rate - 2.5).abs() < 1e-6, "params lost on reset: {rate}");
    }

    #[test]
    fn reset_without_seed_keeps_the_current_seed() {
        let mut w = tiny_world();
        let seed = w.seed();
        w.apply(&Command::Reset { seed: None }).expect("reset");
        assert_eq!(w.seed(), seed);
    }

    /// A physics `set_param` resynchronizes the sliders of all clients; a
    /// `terrain.*` one does not (it does not appear in the `params`
    /// response, its effect is deferred to the next reset).
    #[test]
    fn set_param_physique_broadcast_terrain_non() {
        let mut w = tiny_world();
        assert!(matches!(
            w.set_param("atmosphere.cloud_evap_rate", 0.2),
            Ok(Outcome::Broadcast(_))
        ));
        assert_eq!(
            w.set_param("terrain.initial_water", 5.0),
            Ok(Outcome::Nothing)
        );
        assert!((w.terrain_params().initial_water - 5.0).abs() < 1e-6);
    }

    #[test]
    fn set_param_unknown_is_an_error() {
        let mut w = tiny_world();
        assert_eq!(
            w.set_param("licorne.magie", 1.0),
            Err(ApplyError::UnknownParam {
                key: "licorne.magie".to_owned()
            })
        );
        assert_eq!(
            w.set_param("terrain.licorne", 1.0),
            Err(ApplyError::UnknownParam {
                key: "terrain.licorne".to_owned()
            })
        );
    }

    /// Negative values are floored to 0: `erosion_iterations` is a count,
    /// `initial_water` a stock, neither one makes sense negative, and an
    /// `as u32` on a negative would give an enormous value.
    #[test]
    fn params_terrain_plancher_a_zero() {
        let mut w = tiny_world();
        w.set_param("terrain.erosion_iterations", -5.0)
            .expect("known key");
        w.set_param("terrain.initial_water", -1.0)
            .expect("known key");
        assert_eq!(w.terrain_params().erosion_iterations, 0);
        assert!((w.terrain_params().initial_water - 0.0).abs() < 1e-6);
    }

    #[test]
    fn batch_stride_keeps_a_minimum_batch() {
        assert_eq!(batch_stride(0), 1);
        assert_eq!(batch_stride(1), 1);
        assert_eq!(batch_stride(19), 1);
        assert_eq!(batch_stride(200), 10);
        assert_eq!(batch_stride(8760), 438);
    }

    /// The `meta` query is used to spot a forgotten `just rebuild`: the
    /// hash it returns must be the shell's, not a protocol constant.
    #[test]
    fn meta_reports_the_shell_build() {
        let w = tiny_world();
        let val = w.answer(&Query::Meta);
        assert_eq!(val["build_hash"], test_build().hash);
        assert_eq!(val["version"], test_build().version);
    }
}
