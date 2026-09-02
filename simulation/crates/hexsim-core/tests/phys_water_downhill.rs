//! Micro e2e-unit tests: transport of surface water flowing downhill.
//!
//! Radius 2 or 3, a few dozen steps, isolated brick. Pins the "obvious but
//! hard-won" properties of MFD runoff (`hydro.rs`): water flows downhill,
//! never uphill; a basin concentrates water at the bottom, not at the
//! rim; the total water budget is conserved while it transits between
//! compartments.
//!
//! Radius >= 2 rule: on the torus, a radius-0 cell is its own neighbor
//! ×6, so any transport there would be a silent self-transfer that would
//! test nothing.
//!
//! Doesn't duplicate `emergent_lake.rs` (multi-cell emergence without
//! orchestration), `physics_lake_overflow.rs`/`physics_lake_cascade.rs`
//! (pass crossing, basin cascade), `mfd_stability.rs` (anti-oscillation
//! guard) nor `total_mass_conservation_strict.rs` (10-year drift on
//! production terrain): here the geometry is as simple as possible (line,
//! 2-ring basin) and the sole goal of each test is a single brick of
//! behavior.

mod common;

use common::total_water_budget;
use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::cell::CellProperties;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::{HydroParams, step_hydro_mfd, total_water};
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

// --- Test 1: water flows downhill, never uphill ------------------------

const HIGH: HexCoord = HexCoord { q: -1, r: 0 };
const MID: HexCoord = HexCoord { q: 0, r: 0 };
const LOW: HexCoord = HexCoord { q: 1, r: 0 };
/// Deliberately modest: with `water_capacity = 0`, all `water_level`
/// counts toward `effective_elevation` (free surface, mm converted to m
/// since #104). A source large enough to flood MID above HIGH
/// (> 10,000 mm = 10 m of water depth) would send water both ways: a lake
/// finding its level, not a bug, but it would no longer test the same
/// thing. `MID_ELEV(10 m) + 0.005 m < HIGH(20 m)`: far from the edge case.
const SOURCE_WATER: f32 = 5.0;

/// Line HIGH(elev 20) - MID(elev 10, source) - LOW(elev 0), walls at
/// elev=1000 on the other 4 neighbors of MID to isolate the path.
fn build_line_grid() -> HexGrid {
    let mut grid = HexGrid::from_radius(2);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(c) = grid.get_mut(coord) {
            *c = CellProperties {
                elevation: if coord == HIGH {
                    20.0
                } else if coord == MID {
                    10.0
                } else if coord == LOW {
                    0.0
                } else {
                    1000.0
                },
                water_level: if coord == MID { SOURCE_WATER } else { 0.0 },
                water_capacity: 0.0,
                ..Default::default()
            };
        }
    }
    grid
}

#[test]
fn water_flows_downhill_never_uphill_on_a_simple_line() {
    let mut current = build_line_grid();
    let before = total_water(&current);
    let params = HydroParams::default();

    for _ in 0..3 {
        let mut next = current.clone();
        step_hydro_mfd(&current, &mut next, &params);
        current = next;
    }

    let high_after = current.get(HIGH).unwrap().water_level;
    let mid_after = current.get(MID).unwrap().water_level;
    let low_after = current.get(LOW).unwrap().water_level;

    assert!(
        high_after < 1e-6,
        "water flowed back up to the high cell HIGH: {high_after}"
    );
    assert!(
        low_after > 0.0,
        "water never reached the low cell LOW: {low_after}"
    );
    assert!(
        mid_after < SOURCE_WATER,
        "the MID source didn't lose water downhill: {mid_after}"
    );

    let after = total_water(&current);
    assert!(
        (after - before).abs() < 1e-3,
        "conservation violated on the line: {before} -> {after}"
    );
}

// --- Test 2: a basin concentrates water at the bottom, not at the rim --

/// 2-ring basin (center=0, ring1=5, ring2=10, radius 2 = 19 cells): same
/// profile as `emergent_lake.rs`/`mfd_stability.rs`, but initial water
/// spread over the 12 RIM cells (not at the center): the question asked
/// is different: does water CONVERGE toward the low point, not just "does
/// a lake emerge if we already put it at the bottom".
const PERIPHERY_WATER_EACH: f32 = 2.0;

fn build_basin_grid() -> (HexGrid, Vec<HexCoord>) {
    let mut grid = HexGrid::from_radius(2);
    let center = HexCoord::new(0, 0);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        let d = coord.distance(center);
        if let Some(c) = grid.get_mut(coord) {
            *c = CellProperties {
                elevation: match d {
                    0 => 0.0,
                    1 => 5.0,
                    _ => 10.0,
                },
                water_level: if d == 2 { PERIPHERY_WATER_EACH } else { 0.0 },
                water_capacity: 0.0,
                ..Default::default()
            };
        }
    }
    let periphery: Vec<HexCoord> = grid
        .coords()
        .copied()
        .filter(|c| c.distance(center) == 2)
        .collect();
    (grid, periphery)
}

#[test]
fn water_spread_on_the_rim_converges_to_the_basin_floor() {
    let (grid, periphery) = build_basin_grid();
    let center = HexCoord::new(0, 0);
    let total_initial = total_water(&grid);
    assert!(
        total_initial > 1.0,
        "invalid setup, no initial water: {total_initial}"
    );

    let params = HydroParams::default();
    let mut current = grid;
    for _ in 0..60 {
        let mut next = current.clone();
        step_hydro_mfd(&current, &mut next, &params);
        current = next;
    }

    let center_after = current.get(center).unwrap().water_level;
    let periphery_total_after: f32 = periphery
        .iter()
        .map(|c| current.get(*c).unwrap().water_level)
        .sum();
    let periphery_avg_after =
        periphery_total_after / f32::from(u16::try_from(periphery.len()).expect("fits u16"));

    assert!(
        periphery_total_after < total_initial * 0.9,
        "the periphery didn't lose water significantly: {periphery_total_after:.3} \
         (initial {total_initial:.3})"
    );
    assert!(
        center_after > periphery_avg_after,
        "the basin floor ({center_after:.3}) should concentrate more water than the peripheral \
         average ({periphery_avg_after:.3})"
    );

    let total_after = total_water(&current);
    assert!(
        (total_after - total_initial).abs() < 1e-2,
        "conservation violated on the basin: {total_initial:.3} -> {total_after:.3}"
    );
}

// --- Test 3: the total water budget is conserved during transit --------

const RAMP_TOP: HexCoord = HexCoord { q: -3, r: 0 };
const RAMP_WATER: f32 = 80.0;

/// Full simulation (not just `step_hydro_mfd`) on an open slope with 7
/// steps (radius 3). Unlike `physics_lake_overflow.rs`/
/// `physics_lake_cascade.rs` (lake/pass/outlet topology, 2% tolerance,
/// 50-100 days), here the geometry is a simple inclined plane with no
/// threshold, the duration is short (3 days) and the tolerance is tight
/// (~1e-3 relative): a finer sentinel, dedicated to active transport
/// rather than to the emergence/crossing of a lake.
fn build_ramp_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(c) = grid.get_mut(coord) {
            let q = f32::from(i16::try_from(coord.q).expect("q fits i16"));
            c.elevation = 20.0 - 5.0 * q;
            c.water_level = if coord == RAMP_TOP { RAMP_WATER } else { 0.0 };
            c.water_capacity = 0.0;
            c.permeability = 0.3;
            c.humidity_upper = 0.0;
        }
    }

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
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        temp,
        wind,
    )
}

#[test]
fn water_budget_is_conserved_while_actively_flowing_downhill() {
    let mut sim = build_ramp_sim();
    let before = total_water_budget(&sim);
    assert!(
        before > 1.0,
        "setup invalide, budget initial nul : {before}"
    );

    for _ in 0..3 {
        sim.step();
    }

    // Sanity: transport did indeed happen (otherwise the test proves nothing).
    let top_after = sim.grid().get(RAMP_TOP).unwrap().water_level;
    assert!(
        top_after < RAMP_WATER,
        "the source didn't transport water downhill, empty test: {top_after}"
    );

    let after = total_water_budget(&sim);
    let drift = (after - before).abs();
    let relative = drift / before;
    assert!(
        relative < 1e-3,
        "conservation violated during transit: {before:.6} -> {after:.6} \
         (drift {drift:.6}, {:.4} %)",
        relative * 100.0
    );
}
