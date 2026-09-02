//! Integration test: water flows through a lake/spillway/outflow topography.
//!
//! Simple topology:
//!   SINK (0,0, elev=0) → SPILLWAY (1,0, elev=5) → OUTFLOW (2,0, elev=-5)
//!   Walls at elev=10 everywhere else.
//!
//! Water injected into the SINK must:
//!   1. Reach the OUTFLOW (lowest global point) in significant quantity.
//!   2. Be strictly conserved (closed terrarium, no evaporation, no
//!      infiltration).
//!   3. Not accumulate pathologically in the SPILLWAY (a transit
//!      zone, not storage).
//!
//! If this test fails, the `step_hydro_mfd` + `identify_basins` +
//! `spill_basins` + `drain_edges` pipeline no longer correctly handles water
//! passing over a saddle into a deeper basin.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const SINK: HexCoord = HexCoord { q: 0, r: 0 };
const SPILLWAY: HexCoord = HexCoord { q: 1, r: 0 };
const OUTFLOW: HexCoord = HexCoord { q: 2, r: 0 };
// Phase 3 (#32): rescale ×200 (100 → 20000 mm = 20 m). water_level in mm;
// a large lake to survive Meyer evaporation during the 50 transit ticks.
const INITIAL_WATER: f32 = 20000.0;

fn build_lake_sim() -> Simulation {
    // Linear topology: SINK (0) → SPILLWAY (5) → OUTFLOW (-5). Walls at 10
    // elsewhere. The outflow is the lowest global point; without the spillway
    // at elev=5, water from the sink could not reach it.
    let atmo = AtmosphereParams {
        sublimation_rate: 0.0,
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    };

    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = if coord == SINK {
                0.0
            } else if coord == SPILLWAY {
                5.0
            } else if coord == OUTFLOW {
                -5.0
            } else {
                10.0
            };
            cell.water_level = if coord == SINK { INITIAL_WATER } else { 0.0 };
            cell.water_capacity = 0.0;
            cell.permeability = 0.0;
            cell.humidity_upper = 0.0;
            cell.temperature = 1.0;
            cell.snow_level = 0.0;
            cell.groundwater = 0.0;
        }
    }

    let hydro = HydroParams::default();
    let wind = WindParams {
        thermal_strength: 0.0,
        noise_direction_amplitude: 0.0,
        noise_strength_amplitude: 0.0,
        ..WindParams::default()
    };
    let temp = TemperatureParams {
        latitude_deg: 0.0,
        base_temp: 1.0,
        water_cooling: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };

    Simulation::new(
        grid,
        hydro,
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        temp,
        wind,
    )
}

#[test]
fn water_reaches_outflow_through_spillway() {
    let mut sim = build_lake_sim();
    let before = sim.grid().get(OUTFLOW).unwrap().water_level;
    assert!(before < 1e-3, "outflow must be dry at the start: {before}");

    // 50 ticks: enough for water to cross the spillway (bottleneck
    // at elev=5) and start filling the outflow (-5).
    // Pragmatic threshold: > 10 units = clear signal that the
    // S -> Sp -> O path works. The outflow can't store more than
    // ~15 units anyway before its eff reaches the level of the
    // neighboring walls (10), which in turn makes it an exporter.
    for _ in 0..50 {
        sim.step();
    }

    let after = sim.grid().get(OUTFLOW).unwrap().water_level;
    assert!(
        after > 10.0,
        "outflow did not receive water: got {after:.2}, expected > 10 after 50 ticks"
    );
}

#[test]
fn lake_system_conserves_mass() {
    // Closed terrarium + permeability=0: the total water mass (across all
    // water/humidity/snow/groundwater compartments) must be strictly
    // conserved. Phase 2 (#31): Meyer evaporates from `water_level` to
    // `humidity_surface`, so we do sum the whole water budget, not just
    // the free surface.
    let mut sim = build_lake_sim();
    let total_before = grid_water_budget(&sim);

    for _ in 0..50 {
        sim.step();
    }

    let total_after = grid_water_budget(&sim);
    let drift = (total_after - total_before).abs() / total_before.max(1.0);
    assert!(
        drift < 0.02,
        "conservation broken after 50 ticks: {total_before:.4} -> {total_after:.4} \
         (drift {:.3} %)",
        drift * 100.0
    );
}

fn grid_water_budget(sim: &Simulation) -> f32 {
    sim.grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.snow_level + c.groundwater)
        .sum()
}

#[test]
fn spillway_stays_below_sink_during_transit() {
    // Early phase (tick 5): the sink still has plenty of water, the
    // spillway passes it through but doesn't accumulate it. The spillway
    // must stay below the sink's level at this stage.
    let mut sim = build_lake_sim();

    for _ in 0..5 {
        sim.step();
    }

    let sink_w = sim.grid().get(SINK).unwrap().water_level;
    let spill_w = sim.grid().get(SPILLWAY).unwrap().water_level;
    assert!(
        spill_w < sink_w,
        "during transit, the spillway ({spill_w:.2}) must not exceed \
         the sink ({sink_w:.2})"
    );
}
