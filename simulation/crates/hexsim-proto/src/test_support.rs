//! Fixtures shared by the crate's tests.

use hexsim_core::simulation::Simulation;
use hexsim_core::terrain::TerrainParams;

use crate::query::BuildInfo;
use crate::world::World;

/// Fake build identity, tests verify it *travels through* the
/// protocol, not its value.
pub fn test_build() -> BuildInfo {
    BuildInfo {
        version: "0.0.0-test",
        hash: "testhash",
        unix: 0,
    }
}

/// Radius-2 world (19 cells): enough to have neighbors everywhere,
/// small enough for terrain generation to be instantaneous.
///
/// Radius 2, not 0: on the torus, a radius-0 cell is its own neighbor
/// ×6, so any transport there becomes a silent self-transfer.
pub fn tiny_world() -> World {
    World::generate(2, TerrainParams::default(), test_build())
}

/// The simulation alone, for serialization tests that don't need the
/// world around it.
pub fn tiny_sim() -> Simulation {
    let mut grid = hexsim_core::grid::HexGrid::from_radius(2);
    hexsim_core::terrain::generate_terrain(&mut grid, &TerrainParams::default());
    Simulation::new(
        grid,
        hexsim_core::hydro::HydroParams::default(),
        hexsim_core::atmosphere::AtmosphereParams::default(),
        hexsim_core::groundwater::GroundwaterParams::default(),
        hexsim_core::snow::SnowParams::default(),
        hexsim_core::temperature::TemperatureParams::default(),
        hexsim_core::wind::WindParams::default(),
    )
}
