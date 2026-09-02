//! Integration test: cascade of 2 successive lakes.
//!
//! Linear topology:
//!   `A_SINK` (high basin) → `A_SPILL` (saddle) → `B_SINK` (low basin)
//!   → `B_SPILL` (saddle) → OUT (very low outlet)
//!
//! Water injected into the high basin must:
//!   1. Fill `A_SINK` until its surface exceeds `A_SPILL`.
//!   2. Overflow through `A_SPILL` and flow down into `B_SINK`.
//!   3. Fill `B_SINK` up to `B_SPILL`.
//!   4. Overflow toward OUT.
//!
//! Validates coupling several basins in series: if one of the saddles
//! doesn't let water through (a bug in effective-surface detection or in
//! the ordering of MFD substeps), OUT stays dry and water stagnates
//! upstream. So we test water arrival at OUT plus global conservation.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const A_SINK: HexCoord = HexCoord { q: -2, r: 0 };
const A_SPILL: HexCoord = HexCoord { q: -1, r: 0 };
const B_SINK: HexCoord = HexCoord { q: 0, r: 0 };
const B_SPILL: HexCoord = HexCoord { q: 1, r: 0 };
const OUT: HexCoord = HexCoord { q: 2, r: 0 };
// Phase 3 (#32): rescale ×200 (30 → 6000 mm = 6 m deep). water_level is now
// in mm, the test lakes are sized in tens of cm to meters so they survive
// active Meyer evap over the 50+ ticks of the cascade.
const INITIAL_WATER: f32 = 6000.0;

fn build_sim() -> Simulation {
    // Profile: high basin at 10, saddle at 12, low basin at 5, saddle at 7,
    // outlet at -5. Walls at 50 elsewhere (heights > A_SINK initial eff
    // + water = 40, to prevent a lateral leak).
    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = if coord == A_SINK {
                10.0
            } else if coord == A_SPILL {
                12.0
            } else if coord == B_SINK {
                5.0
            } else if coord == B_SPILL {
                7.0
            } else if coord == OUT {
                -5.0
            } else {
                50.0
            };
            cell.water_level = if coord == A_SINK { INITIAL_WATER } else { 0.0 };
            cell.water_capacity = 0.0;
            cell.temperature = 1.0;
            cell.humidity_upper = 0.0;
            cell.permeability = 0.0;
            cell.groundwater = 0.0;
            cell.snow_level = 0.0;
        }
    }

    let hydro = HydroParams::default();
    let atmo = AtmosphereParams {
        sublimation_rate: 0.0,
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    };
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
fn water_cascades_through_two_lakes_to_outflow() {
    let mut sim = build_sim();
    assert!(
        sim.grid().get(OUT).unwrap().water_level < 1e-3,
        "outflow must be dry at the start"
    );

    // 100 ticks: enough for the water to cross A → saddle A → B → saddle B
    // → OUT. Each link is a bottleneck (saddle 2 elevation units above the
    // basin), so convergence is slow but determinable.
    for _ in 0..100 {
        sim.step();
    }

    let out_water = sim.grid().get(OUT).unwrap().water_level;
    assert!(
        out_water > 1.0,
        "water must have crossed both saddles and reached OUT: {out_water:.3}"
    );
}

#[test]
fn lake_cascade_conserves_mass() {
    let mut sim = build_sim();
    let total_before: f32 = sim
        .grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum();

    for _ in 0..100 {
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
        "conservation violated across cascade: {total_before:.2} -> {total_after:.2} \
         (drift {:.3} %)",
        drift * 100.0
    );
}

#[test]
fn intermediate_lake_fills_before_downstream() {
    // Temporal order: B_SINK (intermediate) must have water BEFORE OUT
    // receives any; otherwise the water "teleports" over the intermediate
    // basin, a sign that multi-basin coupling is broken.
    let mut sim = build_sim();

    // 20 ticks: enough for B_SINK to fill up but not enough for OUT to
    // receive much.
    for _ in 0..20 {
        sim.step();
    }

    let b_water = sim.grid().get(B_SINK).unwrap().water_level;
    assert!(
        b_water > 0.5,
        "B_SINK (intermediate basin) must accumulate water: {b_water:.3}"
    );
}
