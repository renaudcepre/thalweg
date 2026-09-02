//! WASM shell for the `HexSim` protocol; the browser twin of `hexsim-cli`.
//!
//! The native server and this module do the **same** job: parse a
//! command, apply it to the world, then execute what the protocol asks
//! for. Only the transport differs; tokio and a `broadcast` over there,
//! `postMessage` from a Web Worker here. All the logic lives in
//! `hexsim-proto`; this crate only translates its types to JavaScript.
//!
//! Nothing decided here is protocol. If a validation rule, a field name,
//! or a response format appears in this file, it's missing from
//! `hexsim-proto`.
//!
//! # JS-side usage loop
//!
//! ```js
//! const sim = new HexSim(42, 45);
//! const out = JSON.parse(sim.command('{"cmd":"reset","seed":7}'));
//! if (out.kind === "snapshot") render(sim.snapshot());
//! ```

use hexsim_core::simulation::Simulation;
use hexsim_core::terrain::TerrainParams;
use hexsim_proto::command::Command;
use hexsim_proto::query::BuildInfo;
use hexsim_proto::world::{Outcome, Schedule, World};
use serde::Serialize;
use serde_json::Value;
use wasm_bindgen::prelude::*;

/// What the JS shell must do with a command; the direct translation of
/// [`Outcome`], and thus of what `run_outcome` does on the server side.
///
/// Deliberately **flat**: on the JS side it's a `switch (out.kind)` with
/// no nesting. `Play`/`Pause`/`Speed` all three come from [`Schedule`];
/// they are bumped up a level because a browser shell treats them like
/// the other cases, not as a sub-family.
///
/// `Snapshot` does **not carry** the bytes: a snapshot weighs several MB
/// at large radius, passing it as base64 in JSON would be absurd. The JS
/// calls back [`HexSim::snapshot`] when it wants one.
#[derive(Serialize, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Envelope {
    /// Nothing to do.
    Nothing,
    /// The world changed: request a new snapshot and render it.
    Snapshot,
    /// Reply to a read; to be delivered to the code that requested it.
    Reply { payload: Value },
    /// Slider resync after `set_param` (#2). On the server this goes out
    /// to every client; in a single tab there's only one recipient, but
    /// the message stays the same.
    Broadcast { payload: Value },
    /// Resume the auto-tick.
    Play,
    /// Suspend the auto-tick.
    Pause,
    /// Change the auto-tick period. `requested` lets the shell signal a
    /// clamp, just like the server's `warn!`.
    Speed { tick_ms: u64, requested: u64 },
    /// Long advance: it's up to the JS to split it with [`batch_stride`]
    /// and emit one snapshot per batch, otherwise the tab freezes.
    Advance { n: u64, hourly: bool },
    /// Command rejected. The server logs a `warn!`; here the message goes
    /// up to the JS, which displays it in the log panel.
    Error { message: String },
}

impl From<Outcome> for Envelope {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Outcome::Nothing => Self::Nothing,
            Outcome::Snapshot => Self::Snapshot,
            Outcome::Reply(payload) => Self::Reply { payload },
            Outcome::Broadcast(payload) => Self::Broadcast { payload },
            Outcome::Schedule(Schedule::Play) => Self::Play,
            Outcome::Schedule(Schedule::Pause) => Self::Pause,
            Outcome::Schedule(Schedule::Speed { tick_ms, requested }) => {
                Self::Speed { tick_ms, requested }
            }
            Outcome::Advance { n, hourly } => Self::Advance { n, hourly },
        }
    }
}

impl Envelope {
    /// Serializes for the JS boundary.
    ///
    /// A serialization failure is treated as a command error rather than
    /// a panic: an `unwrap` here would bring down the whole wasm module
    /// for a case that doesn't happen.
    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!(r#"{{"kind":"error","message":"serialization failed: {e}"}}"#)
        })
    }
}

/// Identity of this wasm build, injected by `build.rs`. Distinct from the
/// server's: these are two separate binaries, and the front's badge
/// exists precisely to tell which one is being served.
fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        hash: env!("HEXSIM_BUILD_HASH"),
        unix: env!("HEXSIM_BUILD_UNIX").parse().unwrap_or(0),
    }
}

/// The simulation as seen from JavaScript.
#[wasm_bindgen]
pub struct HexSim {
    world: World,
}

#[wasm_bindgen]
impl HexSim {
    /// Generates a fresh world. `radius` 45 ≈ 6200 cells, 120 ≈ 43000.
    ///
    /// Installs the panic hook along the way: without it, a panic in the
    /// wasm shows up in JS as a `RuntimeError: unreachable` with no stack.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(seed: u32, radius: i32) -> Self {
        console_error_panic_hook::set_once();
        let terrain = TerrainParams {
            seed,
            ..TerrainParams::default()
        };
        Self {
            world: World::generate(radius, terrain, build_info()),
        }
    }

    /// Executes a protocol command (`{"cmd":"diagnostics"}`) and returns
    /// the JSON `Envelope` describing what to do next.
    ///
    /// Never panics: an unreadable or inapplicable command comes back as
    /// `{"kind":"error"}`, exactly where the server would emit a `warn!`.
    pub fn command(&mut self, json: &str) -> String {
        let envelope = match Command::parse(json) {
            Ok(cmd) => match self.world.apply(&cmd) {
                Ok(outcome) => Envelope::from(outcome),
                Err(e) => Envelope::Error {
                    message: e.to_string(),
                },
            },
            Err(e) => Envelope::Error {
                message: e.to_string(),
            },
        };
        envelope.to_json()
    }

    /// Advances by one simulated hour and renders the corresponding
    /// snapshot.
    ///
    /// This is the auto-tick step; the exact counterpart of the server's
    /// `spawn_tick_loop`, which advances by one **hour** and not one day
    /// so that the front sees the diurnal cycle (#47).
    ///
    /// # Errors
    ///
    /// Error message if encoding the snapshot fails; see
    /// [`HexSim::snapshot`].
    pub fn step_hour(&mut self) -> Result<Vec<u8>, String> {
        self.world.sim_mut().step_hour();
        self.snapshot()
    }

    /// Advances by `n` units (hours if `hourly`, days otherwise)
    /// **without** producing a snapshot.
    ///
    /// Reserved for splitting up an `Advance`: the JS chains
    /// `advance(batch)` + `snapshot()` per step, yielding control back to
    /// the browser in between, the same way the server releases its lock.
    pub fn advance(&mut self, n: u32, hourly: bool) {
        self.world.advance(u64::from(n), hourly);
    }

    /// Column-packed msgpack snapshot of the current state; byte-for-byte
    /// identical to the WebSocket one.
    ///
    /// # Errors
    ///
    /// Error message if `MessagePack` encoding fails (see
    /// `hexsim_proto::wire::encode_snapshot`): this doesn't happen on a
    /// valid state, but the error is surfaced to JS rather than masked,
    /// same rationale as [`HexSim::export_checkpoint`].
    pub fn snapshot(&self) -> Result<Vec<u8>, String> {
        self.world
            .snapshot_bytes()
            .map(|bytes| bytes.to_vec())
            .map_err(|e| format!("snapshot_bytes: {e}"))
    }

    /// Radius of the current grid (it changes if a checkpoint from a
    /// different radius is loaded).
    #[must_use]
    pub fn radius(&self) -> i32 {
        self.world.radius()
    }

    /// Seed of the current world.
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.world.seed()
    }

    /// Full serialized state; the equivalent of `GET /checkpoint`. The JS
    /// turns it into a `Blob` to download.
    ///
    /// # Errors
    ///
    /// Error message if serializing the world fails. The error is a
    /// `String` and not a `JsError`: the latter cannot be *constructed*
    /// outside wasm (it calls an imported function and panics), which
    /// would make this path untestable natively. The message reuses the
    /// body the server returns on `POST /checkpoint` with a 400.
    pub fn export_checkpoint(&self) -> Result<Vec<u8>, String> {
        self.world
            .sim()
            .save_state()
            .map_err(|e| format!("save_state: {e}"))
    }

    /// Loads an exported state; the equivalent of `POST /checkpoint`. An
    /// invalid blob is rejected **without touching the current world**.
    ///
    /// # Errors
    ///
    /// Error message if the bytes aren't a checkpoint readable by this
    /// version. See [`HexSim::export_checkpoint`] for the choice of
    /// `String`.
    pub fn import_checkpoint(&mut self, bytes: &[u8]) -> Result<(), String> {
        let sim = Simulation::load_state(bytes).map_err(|e| format!("load_state: {e}"))?;
        self.world.replace_simulation(sim);
        Ok(())
    }
}

/// Batch size for splitting up an `Advance`, identical to the server's.
///
/// Exposed so that WASM and WebSocket produce the **same** number of
/// progress steps: a bar that advances differently depending on the mode
/// would be a protocol discrepancy disguised as a UI detail.
#[wasm_bindgen]
#[must_use]
pub fn batch_stride(n: u32) -> u32 {
    u32::try_from(hexsim_proto::world::batch_stride(u64::from(n))).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Radius 2 (19 cells): enough for neighbors everywhere, small enough
    /// for generation to be instant.
    fn tiny() -> HexSim {
        HexSim::new(42, 2)
    }

    fn kind(json: &str) -> String {
        serde_json::from_str::<Value>(json).expect("enveloppe JSON")["kind"]
            .as_str()
            .expect("champ kind")
            .to_owned()
    }

    /// Each protocol `Outcome` has one translation, and only one. This
    /// test fails if `Outcome` gains a variant without the JS learning to
    /// handle it.
    #[test]
    fn every_outcome_has_its_envelope() {
        let cases = [
            (Outcome::Nothing, "nothing"),
            (Outcome::Snapshot, "snapshot"),
            (Outcome::Reply(Value::Null), "reply"),
            (Outcome::Broadcast(Value::Null), "broadcast"),
            (Outcome::Schedule(Schedule::Play), "play"),
            (Outcome::Schedule(Schedule::Pause), "pause"),
            (
                Outcome::Schedule(Schedule::Speed {
                    tick_ms: 30,
                    requested: 30,
                }),
                "speed",
            ),
            (
                Outcome::Advance {
                    n: 10,
                    hourly: true,
                },
                "advance",
            ),
        ];
        for (outcome, expected) in cases {
            let json = Envelope::from(outcome).to_json();
            assert_eq!(kind(&json), expected, "envelope for {expected}");
        }
    }

    /// A query passes through the wrapper without being reformatted: the
    /// `payload` is exactly what the WebSocket would have sent.
    #[test]
    fn query_comes_back_as_reply_with_the_protocol_payload() {
        let mut sim = tiny();
        let out = sim.command(r#"{"cmd":"params"}"#);
        let v: Value = serde_json::from_str(&out).expect("JSON");
        assert_eq!(v["kind"], "reply");
        assert_eq!(v["payload"]["type"], "params");
        assert!(v["payload"]["atmosphere"].is_object());
    }

    /// `meta` must report the build of the **wasm module**, not a
    /// protocol constant: this is what makes it possible to spot a stale
    /// wasm in browser cache.
    #[test]
    fn meta_reports_the_module_build() {
        let mut sim = tiny();
        let v: Value = serde_json::from_str(&sim.command(r#"{"cmd":"meta"}"#)).expect("JSON");
        assert_eq!(v["payload"]["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(v["payload"]["build_hash"], env!("HEXSIM_BUILD_HASH"));
    }

    /// Both failure families (unreadable JSON and inapplicable command)
    /// come back as `error` rather than a panic. A panic here would kill
    /// the entire wasm module.
    #[test]
    fn failures_do_not_panic() {
        let mut sim = tiny();
        for bad in [
            "not json",
            "{}",
            r#"{"cmd":"danser"}"#,
            r#"{"cmd":"set_param","key":"licorne.magie","value":1}"#,
        ] {
            let out = sim.command(bad);
            assert_eq!(kind(&out), "error", "commande `{bad}`");
            let v: Value = serde_json::from_str(&out).expect("JSON");
            assert!(
                !v["message"].as_str().unwrap_or_default().is_empty(),
                "empty error message for `{bad}`"
            );
        }
    }

    /// `step_hour` advances by one hour and renders a decodable snapshot;
    /// this is the auto-tick step, the most heavily traveled path in the
    /// module.
    #[test]
    fn step_hour_advances_and_returns_a_valid_snapshot() {
        let mut sim = tiny();
        let before = sim.world.sim().hour_tick();
        let bytes = sim.step_hour().expect("step_hour snapshot encode");
        assert_eq!(sim.world.sim().hour_tick(), before + 1);

        let decoded: Value = rmp_serde::from_slice(&bytes).expect("snapshot msgpack");
        assert!(decoded["cell_fields"].is_array(), "expected wire format");
        assert_eq!(decoded["hour_tick"], before + 1);
    }

    /// `advance` doesn't produce a snapshot: this is what lets the JS pay
    /// for one only per batch instead of per tick.
    #[test]
    fn advance_advances_without_a_snapshot() {
        let mut sim = tiny();
        let before = sim.world.sim().hour_tick();
        sim.advance(5, true);
        assert_eq!(sim.world.sim().hour_tick(), before + 5);
    }

    /// A checkpoint round-trip preserves the tick; an invalid blob is
    /// rejected **without damaging the current world**.
    #[test]
    fn checkpoint_round_trip_and_clean_rejection() {
        let mut sim = tiny();
        sim.advance(3, true);
        let saved = sim.export_checkpoint().expect("export");
        let tick_avant = sim.world.sim().hour_tick();

        sim.advance(10, true);
        sim.import_checkpoint(&saved).expect("import");
        assert_eq!(sim.world.sim().hour_tick(), tick_avant);

        assert!(sim.import_checkpoint(b"nawak").is_err());
        assert_eq!(
            sim.world.sim().hour_tick(),
            tick_avant,
            "a rejected import must not touch the world"
        );
    }

    /// The batching must be the same as in server mode, otherwise the
    /// progress bar doesn't advance the same way depending on the
    /// transport.
    #[test]
    fn batch_stride_matches_the_protocol() {
        for n in [0_u32, 1, 19, 200, 8760] {
            assert_eq!(
                u64::from(batch_stride(n)),
                hexsim_proto::world::batch_stride(u64::from(n)),
                "stride for n={n}"
            );
        }
    }

    /// The world stays deterministic across the wrapper: same seed, same
    /// snapshot. This is the project's fundamental invariant, and the
    /// wrapper is a plausible place to break it (hidden state, init
    /// order).
    #[test]
    fn same_seed_same_world() {
        let mut a = HexSim::new(7, 2);
        let mut b = HexSim::new(7, 2);
        a.advance(4, true);
        b.advance(4, true);
        assert_eq!(
            a.snapshot().expect("snapshot a"),
            b.snapshot().expect("snapshot b")
        );

        let c = HexSim::new(8, 2);
        assert_ne!(
            a.snapshot().expect("snapshot a"),
            c.snapshot().expect("snapshot c"),
            "different seeds, same world?"
        );
    }
}
