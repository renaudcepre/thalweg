//! A geophysical cellular automaton on a hexagonal grid.
//!
//! Rivers, lakes, microclimates and forest succession are not drawn and not
//! scripted: they are what falls out of local rules applied to a cell and its
//! six neighbours. Three properties hold the whole engine together.
//!
//! **A closed terrarium.** Water and energy are conserved quantities. If a
//! total drifts, that is a bug, and it is the most fundamental test invariant
//! in the project.
//!
//! **Double buffering.** Every tick reads from `current` and writes into
//! `next` ([`grid::HexGrid`]), so the order cells are visited in cannot change
//! the result. One tick is one hour, twenty-four per simulated day
//! ([`time`]).
//!
//! **Phenomena are pure functions.** Each `step_*` reads a cell and its
//! neighbours and writes into the next buffer. They never call each other;
//! they interact only through the properties they mutate. Their argument
//! order is fixed by convention: `(current, next, params, forcing, …)`, where
//! a forcing groups the tick's read-only inputs. The call order in
//! [`simulation`] is physics — convection before advection, groundwater
//! before hydrology — so it stays written out, not hidden behind a dispatch
//! trait.
//!
//! A single seed determines the entire world, terrain included. The one piece
//! of state that escapes it is [`ablation`], the environment-variable A/B
//! switches, which is why a checkpoint records them and refuses to resume
//! under different ones.
//!
//! Physics is written in strict SI units, with the fundamental constants
//! declared explicitly and their source named at the point of use.

// --- Foundations: the grid and the vocabulary everything else is written in.
/// The properties one cell carries from tick to tick.
pub mod cell;
/// Axial `(q, r)` coordinates, the six directions, and the toric lattice.
pub mod coord;
/// The `HexGrid` itself: cells, neighbour indices, double buffering.
pub mod grid;
/// The serialised view an external consumer renders. Depends on the
/// phenomena; the grid does not depend on it.
pub mod snapshot;
/// The tick contract: hours, days, and how the two relate.
pub mod time;
/// Dimensioned newtypes, so a millimetre cannot be added to a metre.
pub mod units;

// --- World generation: what exists before the first tick.
/// The mineral substrate beneath each cell, which sets what erodes and how
/// fast.
pub mod lithology;
/// Procedural relief from a seed.
pub mod terrain;

// --- Phenomena. Listed alphabetically because `rustfmt` reorders module
// declarations; the order they actually run in is physics and is written
// out in `simulation::Simulation::step_hour`.
/// Humidity, clouds, precipitation, and their horizontal transport.
pub mod atmosphere;
/// Synoptic dynamics: the shallow-water solver that produces weather systems.
pub mod dynamics;
/// Fluvial erosion: stream power incision and deposition.
pub mod erosion;
/// Forest fire ignition and spread.
pub mod fire;
/// The water table: infiltration, storage, resurgence.
pub mod groundwater;
/// Surface water flowing downhill, and the flux history rivers are read from.
pub mod hydro;
/// Hydrostatic levelling of a lake spanning several cells.
pub mod lake;
/// Physical laws shared by several phenomena (saturation, evaporation).
pub mod physics;
/// Snowpack: freeze, melt, and the ice-albedo feedback.
pub mod snow;
/// The species themselves and their ecological niches.
pub mod species;
/// The coarse mesh the synoptic solver integrates on.
pub mod synoptic_mesh;
/// SI energy balance for the ground and water surface: solar astronomy, the
/// illumination cache, and the temperature step itself.
pub mod temperature;
/// Multi-species biomass, growth and competition.
pub mod vegetation;
/// The wind field the atmosphere is advected by.
pub mod wind;

// --- Orchestration: what turns the phenomena into a running world.
/// The environment-variable switches that change the physics, gathered in one
/// place so a checkpoint can record them.
pub mod ablation;
/// Full-state save and restore.
pub mod checkpoint;
/// The engine: the `Simulation` state and the per-tick call order.
pub mod simulation;

// --- Observation: reading the world without changing it.
/// Descriptive statistics used by the calibration instruments.
pub mod bench_metrics;
/// Per-cell climate history, a 365-day ring buffer.
pub mod climate;
/// Climate normals, the multi-year averages species niches are judged
/// against.
pub mod climate_normals;
/// Compact per-tick diagnostics.
pub mod diagnostics;
/// Where the time in a tick actually goes, phase by phase.
pub mod phase_timing;
