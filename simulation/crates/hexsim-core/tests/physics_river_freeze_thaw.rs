//! Integration test: river freeze and thaw.
//!
//! Active runoff on a slope freezes when temperature drops below 0C,
//! water becomes snow, discharge drops. When temperature rises, snow melts
//! and water flows again.
//!
//! Extreme case of a fundamental property: matter of frozen stock doesn't
//! move (no discharge), thaw restores flow.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const SOURCE: HexCoord = HexCoord { q: 0, r: 0 };
const MID: HexCoord = HexCoord { q: 1, r: 0 };
const SINK: HexCoord = HexCoord { q: 2, r: 0 };

fn build_sim(temperature: f32) -> Simulation {
    // Slope profile: SOURCE elev=100, MID elev=50, SINK elev=0.
    // Water 20 at source; rest at 0. No explicit walls, other cells at
    // elev=200 (shouldn't receive water, they're higher than all line neighbors).
    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            if coord == SOURCE {
                cell.elevation = 100.0;
                // Phase 3 (#32): rescale x200 (20 to 4000 mm = 4 m).
                cell.water_level = 4000.0;
            } else if coord == MID {
                cell.elevation = 50.0;
                cell.water_level = 0.0;
            } else if coord == SINK {
                cell.elevation = 0.0;
                cell.water_level = 0.0;
            } else {
                cell.elevation = 200.0;
                cell.water_level = 0.0;
            }
            cell.temperature = temperature;
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
    // thermal_coupling=0: temperature stays frozen at initial value.
    let temp = TemperatureParams {
        latitude_deg: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };
    // Humidity floor + sublimation + orographic convection at 0 for pure
    // freeze/flow mechanics test. Meyer evap can fire on thawed water; tests
    // verify freeze/flow equilibria not absolute mass conservation, so impact acceptable.
    let atmo = AtmosphereParams {
        sublimation_rate: 0.0,
        orographic_lift_coef: 0.0,
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
fn warm_river_has_positive_discharge() {
    // Sanity: in warm regime, water flows and discharge is nonzero.
    let mut sim = build_sim(15.0);
    sim.step();

    let d_source = sim
        .grid()
        .cell_index(SOURCE)
        .and_then(|i| sim.discharge_map().get(i).copied())
        .unwrap_or(0.0);
    assert!(
        d_source > 0.0,
        "source must have nonzero discharge at T=15C: {d_source}"
    );
}

#[test]
fn cold_climate_locks_water_into_snow() {
    // At T=-10C for 500 ticks: with limited freeze rate (0.05/tick max),
    // water from initial 20 units ends mostly trapped as snow (freeze >
    // surface water conservation).
    //
    // Test macro invariant: total_snow > total_water after convergence,
    // not fine-grained kinetics of a given cell, which depends on hydro/snow
    // order in tick and flow speed downstream.
    let mut sim = build_sim(-10.0);
    for _ in 0..500 {
        sim.step();
    }

    let total_water: f32 = sim.grid().iter().map(|(_, c)| c.water_level).sum();
    let total_snow: f32 = sim.grid().iter().map(|(_, c)| c.snow_level).sum();

    assert!(
        total_snow > total_water,
        "at T=-10 after 500 ticks, snow must dominate: \
         snow={total_snow:.2}, water={total_water:.2}"
    );
    // Phase 3: rescale x200 (18 to 3600, close to 4000 initial with tolerance).
    assert!(
        total_snow + total_water > 3600.0,
        "conservation: total water+snow must stay close to 4000: \
         {total_snow:.2} + {total_water:.2} = {:.2}",
        total_snow + total_water
    );
}

/// Variant of `build_sim` with snow (instead of water) at source.
fn build_sim_with_snow(temperature: f32, initial_snow: f32) -> Simulation {
    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            if coord == SOURCE {
                cell.elevation = 100.0;
                cell.snow_level = initial_snow;
            } else if coord == MID {
                cell.elevation = 50.0;
            } else if coord == SINK {
                cell.elevation = 0.0;
            } else {
                cell.elevation = 200.0;
            }
            cell.temperature = temperature;
            cell.water_level = 0.0;
            cell.water_capacity = 0.0;
            cell.humidity_upper = 0.0;
            cell.permeability = 0.0;
            cell.groundwater = 0.0;
        }
    }
    let hydro = HydroParams::default();
    let wind = WindParams {
        thermal_strength: 0.0,
        ..WindParams::default()
    };
    let temp = TemperatureParams {
        latitude_deg: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };
    let atmo = AtmosphereParams::default();
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
fn thawing_releases_water_from_snow() {
    // Inverse setup: snow at summit, T=+15. Snow melts, water flows to sink.
    // Phase 3 (#32): rescale x200 (20 to 4000 mm = 4 m snow cover).
    let mut sim = build_sim_with_snow(15.0, 4000.0);

    for _ in 0..30 {
        sim.step();
    }

    let source = sim.grid().get(SOURCE).unwrap();
    let sink = sim.grid().get(SINK).unwrap();

    assert!(
        source.snow_level < 4000.0,
        "source snow must melt: snow={:.2} (initial 20)",
        source.snow_level
    );
    assert!(
        sink.water_level > 0.0,
        "meltwater must flow to sink: water={:.2}",
        sink.water_level
    );
}
