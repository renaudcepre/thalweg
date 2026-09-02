//! `HexSim` protocol: the shared vocabulary between the engine and its
//! consumers.
//!
//! This crate is the **transport / protocol boundary**. It answers three
//! questions, and no others:
//!
//! - what is a valid command? ([`command`])
//! - what does the engine reply to a read? ([`query`])
//! - what does a snapshot look like on the wire? ([`wire`])
//!
//! It doesn't know how to open a socket, log, or wait. That's what
//! lets it run identically in the native WebSocket server
//! (`hexsim-cli`, with tokio) and in the browser (WASM module, with no
//! `std::net` or `std::thread` at all). Without it, every shell would
//! reimplement the protocol, and the two would diverge at the first
//! field added to a snapshot.
//!
//! ```no_run
//! use hexsim_proto::{command::Command, query::BuildInfo, world::{Outcome, World}};
//! use hexsim_core::terrain::TerrainParams;
//!
//! let build = BuildInfo { version: "0.0.0", hash: "dev", unix: 0 };
//! let mut world = World::generate(45, TerrainParams::default(), build);
//!
//! let cmd = Command::parse(r#"{"cmd":"diagnostics"}"#).expect("commande valide");
//! if let Ok(Outcome::Reply(json)) = world.apply(&cmd) {
//!     println!("{json}");
//! }
//! ```

pub mod command;
pub mod query;
pub mod wire;
pub mod world;

#[cfg(test)]
pub(crate) mod test_support;

pub use command::{Command, ParseError, Query};
pub use query::BuildInfo;
pub use world::{ApplyError, Outcome, Schedule, World};
