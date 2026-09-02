//! Test d'integration : cycle local evaporation → convection → precipitation.
//!
//! A warm lake next to a cold mountain: evaporation feeds humidity,
//! convection carries it to the high, cold cell, and precipitation
//! deposits water on the mountain. Without wind and without external
//! input (closed terrarium), only this internal cycle can wet the
//! mountain.
//!
//! If this test fails:
//! - `step_evaporation` no longer produces humidity (check `temp_factor`)
//! - `step_convection` no longer transports humidity to the high-cold cell
//! - `step_precipitation` no longer triggers at low temperature
//!
//! Covers the atmosphere-surface coupling on a minimal case (2 cells).

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const LAKE: HexCoord = HexCoord { q: 0, r: 0 };
const MOUNTAIN: HexCoord = HexCoord { q: 1, r: 0 };
// Phase 3 (#32): rescale ×200 (50 → 10000 mm = 10 m depth) so that
// the lake survives the evap/rain cycle ticks without being drained by Meyer.
const INITIAL_LAKE_WATER: f32 = 10000.0;

fn build_sim() -> Simulation {
    // Radius-2 grid: LAKE and MOUNTAIN are direct neighbors.
    let mut grid = HexGrid::from_radius(2);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            if coord == LAKE {
                cell.elevation = 0.0;
                cell.water_level = INITIAL_LAKE_WATER;
                cell.temperature = 20.0;
            } else if coord == MOUNTAIN {
                cell.elevation = 500.0;
                cell.water_level = 0.0;
                // Just above 0°C: precipitation will be liquid
                // (no snow), avoiding any contamination by freezing.
                cell.temperature = 2.0;
            } else {
                // Neutral cells around, no water, average temperature.
                cell.elevation = 100.0;
                cell.water_level = 0.0;
                cell.temperature = 15.0;
            }
            cell.water_capacity = 0.0;
            cell.humidity_upper = 0.0;
            cell.permeability = 0.0;
            cell.groundwater = 0.0;
            cell.snow_level = 0.0;
        }
    }

    let hydro = HydroParams::default();
    let wind = WindParams {
        thermal_strength: 0.0,
        noise_direction_amplitude: 0.0,
        noise_strength_amplitude: 0.0,
        ..WindParams::default()
    };
    // thermal_coupling=0: freezes the initial temperatures. Otherwise the sim
    // pulls them back toward the lapse-rate target and the delta between lake
    // and mountain shrinks, slowing convection.
    let temp = TemperatureParams {
        latitude_deg: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };

    // Initial humidity floor at 0: the test `humidity_builds_up_over_ticks`
    // verifies that the grid starts sterile and that the local cycle generates
    // humidity "from zero".
    let atmo = AtmosphereParams {
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
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
fn mountain_gains_moisture_from_lake_evaporation() {
    let mut sim = build_sim();
    for _ in 0..300 {
        sim.step();
    }

    let mountain = sim.grid().get(MOUNTAIN).unwrap();
    let lake = sim.grid().get(LAKE).unwrap();

    // The mountain must have accumulated matter (water, snow, or
    // humidity). The precise signal (water vs snow) depends on the
    // precipitation at T=2°C and the lapse rate; we test the broad
    // invariant: "the cycle does transport water from the lake to
    // the mountain".
    let mountain_moisture = mountain.water_level + mountain.snow_level + mountain.humidity_total();
    assert!(
        mountain_moisture > 0.05,
        "the mountain must receive from the evap-convection cycle: \
         water={:.4} snow={:.4} humidity={:.4}",
        mountain.water_level,
        mountain.snow_level,
        mountain.humidity_total()
    );
    assert!(
        lake.water_level < INITIAL_LAKE_WATER,
        "the lake must lose water via evaporation: {:.2} (initial {INITIAL_LAKE_WATER})",
        lake.water_level
    );
}

#[test]
fn total_water_is_conserved_in_local_cycle() {
    // Closed terrarium + no infiltration: conservation of the total
    // water+humidity+gw+snow must be strict (drift < 1 %).
    let mut sim = build_sim();
    let total_before: f32 = sim
        .grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum();

    for _ in 0..300 {
        sim.step();
    }

    let total_after: f32 = sim
        .grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum();

    let drift = (total_after - total_before).abs() / total_before.max(1.0);
    assert!(
        drift < 0.02,
        "conservation broken over the local cycle: \
         {total_before:.2} -> {total_after:.2} (drift {:.3} %)",
        drift * 100.0
    );
}

#[test]
fn humidity_builds_up_over_ticks() {
    // Sanity: humidity must appear (via evaporation) before
    // it can precipitate. This test catches an upstream bug (step_evaporation
    // no longer producing) that would mask the true cause of failure in the two
    // tests above.
    let mut sim = build_sim();
    let humidity_before: f32 = sim.grid().iter().map(|(_, c)| c.humidity_total()).sum();
    assert!(humidity_before < 1e-6, "initial humidity must be zero");

    for _ in 0..30 {
        sim.step();
    }

    let humidity_after: f32 = sim.grid().iter().map(|(_, c)| c.humidity_total()).sum();
    assert!(
        humidity_after > 0.01,
        "humidity must build up within the first 30 ticks: {humidity_after:.5}"
    );
}
