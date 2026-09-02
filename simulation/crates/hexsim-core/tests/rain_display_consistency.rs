//! Guard against "display != physics" drift.
//!
//! History: `grid.rs::snapshot` used obsolete hardcoded threshold to
//! determine `is_raining`. When wet convection and orography were added to
//! `step_precipitation`, snapshot kept using old threshold, causing hours of
//! wild goose chase debugging because display underestimated actual rain.
//!
//! This test ensures `CellSnapshot.is_raining` matches **exactly** cells that
//! received precipitation in last tick (via solver `PrecipitationMap`). Any
//! future threshold logic duplication that re-diverges would fail this test.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

fn build_sim(radius: i32, seed: u32) -> Simulation {
    let terrain = TerrainParams {
        seed,
        ..TerrainParams::default()
    };
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(&mut grid, &terrain);
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

/// For each simulated tick, cells marked `is_raining = true` in snapshot
/// must be exactly those with positive precipitation in solver
/// `PrecipitationMap`.
///
/// If this test fails, either a new precipitation branch (thunderstorm,
/// front, etc.) was added to `atmosphere.rs` without updating snapshot's
/// source of truth, or snapshot uses a hardcoded threshold that diverges
/// from actual physics.
#[test]
fn snapshot_is_raining_matches_precipitation_map() {
    let mut sim = build_sim(8, 42);

    // Let sim start - a few ticks for humidity to build.
    for _ in 0..60 {
        sim.step();
    }

    // Check consistency over consecutive ticks: want to capture both
    // dry ticks and ticks with events.
    for _ in 0..20 {
        sim.step();

        let precip = sim.last_precipitation();
        let snapshot = sim.snapshot();
        let grid = sim.grid();

        // Build the set of cells "that precipitated according to the physics"
        let actually_precipitated: std::collections::HashSet<HexCoord> = precip
            .iter()
            .enumerate()
            .filter(|(_, day)| day.rain > 1e-4 || day.snow > 1e-4)
            .map(|(i, _)| grid.coords_slice()[i])
            .collect();

        // Compare with the cells "displayed as raining"
        let displayed_as_raining: std::collections::HashSet<HexCoord> = snapshot
            .cells
            .iter()
            .filter(|c| c.is_raining)
            .map(|c| HexCoord::new(c.q, c.r))
            .collect();

        let only_in_snapshot: Vec<_> = displayed_as_raining
            .difference(&actually_precipitated)
            .collect();
        let only_in_physics: Vec<_> = actually_precipitated
            .difference(&displayed_as_raining)
            .collect();

        assert!(
            only_in_snapshot.is_empty(),
            "Snapshot shows rainy cells that did NOT precipitate (tick {}): {} cells. Example: {:?}",
            snapshot.tick,
            only_in_snapshot.len(),
            only_in_snapshot.first(),
        );
        assert!(
            only_in_physics.is_empty(),
            "Cells precipitated but are missing from the is_raining snapshot (tick {}): {} cells. Example: {:?}",
            snapshot.tick,
            only_in_physics.len(),
            only_in_physics.first(),
        );
    }
}
