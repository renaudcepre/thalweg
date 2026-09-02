//! Micro e2e-unit tests: fluvial erosion (#105) in the full simulation.
//!
//! Radius >= 2 (transport rule: a radius-0 cell is its own neighbor ×6 on
//! the torus, any transport there would self-cancel), a few days at inflated
//! geological acceleration to make the phenomenon visible in ms. Pins the
//! "obvious but hard-won" properties: a flowing river incises its bed;
//! the detached matter travels downstream (>= 2 cells) and redeposits;
//! the rock budget `Σ(elevation)+Σ(sediment_load)` is conserved while the
//! WATER budget is too; erosion off = bit-identical relief.
//!
//! Doesn't duplicate the unit tests of `erosion.rs` (hand-built EMA
//! forcing, a single step): here the EMA fills itself from the real
//! hydro slice, and the full coupling (rain, groundwater, vegetation…)
//! runs around it.

mod common;

use common::total_water_budget;
use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::cell::CellProperties;
use hexsim_core::erosion::total_rock;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

/// East->West ramp (world x ∝ 2q+r) with a perpetual water source at
/// the top: a permanent "river" flows down the q line. Radius 3 to
/// leave >= 2 cells of downstream travel from the top.
fn build_river_sim(accel_years_per_day: f32) -> Simulation {
    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(c) = grid.get_mut(coord) {
            *c = CellProperties {
                elevation: f32::from(
                    i16::try_from(500 + 15 * (2 * coord.q + coord.r)).expect("fits i16"),
                ),
                // Generous initial water depth across the whole map: the
                // hydro slice makes it flow from day 1, the EMA fills up.
                water_level: 40.0,
                water_capacity: 0.0,
                permeability: 0.0,
                ..Default::default()
            };
        }
    }

    // World as inert as possible around the hydro: no evaporation
    // driven by wind/excessive heat, no snow.
    let atmo = AtmosphereParams {
        sublimation_rate: 0.0,
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    };
    let temp = TemperatureParams {
        latitude_deg: 0.0,
        base_temp: 5.0,
        water_cooling: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams {
            infiltration_rate: 0.0,
            ..GroundwaterParams::default()
        },
        SnowParams::default(),
        temp,
        WindParams {
            thermal_strength: 0.0,
            noise_direction_amplitude: 0.0,
            noise_strength_amplitude: 0.0,
            ..WindParams::default()
        },
    );
    // LIVE erosion explicitly enabled: since the one-shot pivot (#105)
    // it's OFF by default (the world is pre-eroded at worldgen, cf.
    // `erode_terrain`); these tests exercise the opt-in live mode.
    assert!(sim.update_param("erosion.enabled", 1.0));
    // Test geological acceleration: make incision measurable within a
    // few simulated days. Short τ so the EMA fills up fast.
    assert!(sim.update_param("erosion.accel_years_per_day", accel_years_per_day));
    assert!(sim.update_param("erosion.tau_days", 2.0));
    sim
}

/// The river incises its bed, matter travels >= 2 cells downstream and
/// the rock budget is conserved: all of it through the real hydro slice
/// (self-filling EMA), not a hand-built forcing.
#[test]
fn river_incises_and_rock_travels_downstream() {
    let mut sim = build_river_sim(200_000.0);
    let rock_before = total_rock(sim.grid());
    let elev_before: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation)
        .collect();

    for _ in 0..12 {
        sim.step();
    }

    // Strict rock conservation while the work happens.
    let rock_after = total_rock(sim.grid());
    let rel = ((rock_after - rock_before) / rock_before.abs().max(1.0)).abs();
    assert!(
        rel < 1e-6,
        "rock budget not conserved: {rock_before} → {rock_after} ({rel:e} relative)"
    );

    // The bed has incised somewhere (water flows, the EMA is loaded).
    let (incised, deposited) = sim.erosion_totals();
    assert!(
        incised > 0.0,
        "no incision after 12 days of river: {incised}"
    );

    // Transport >= 2 cells: matter has been deposited (or is in
    // transit) >= 2 cells downstream of an incised cell. On the ramp
    // 2q+r, "downstream" = smaller world x. We compare per cell: gain
    // in elevation (deposit) or load in transit to the west, loss to the east.
    let cells = sim.grid().cells_slice();
    let mut easternmost_incised: Option<i32> = None;
    let mut westernmost_touched: Option<i32> = None;
    for (i, coord) in sim.grid().coords_slice().iter().enumerate() {
        let x2 = 2 * coord.q + coord.r; // ∝ x monde
        let d_elev = cells[i].elevation - elev_before[i];
        if d_elev < -1e-4 {
            easternmost_incised = Some(easternmost_incised.map_or(x2, |v| v.max(x2)));
        }
        if d_elev > 1e-4 || cells[i].sediment_load > 1e-4 {
            westernmost_touched = Some(westernmost_touched.map_or(x2, |v| v.min(x2)));
        }
    }
    let src = easternmost_incised.expect("at least one incised cell");
    let dst = westernmost_touched.expect("at least one deposit or load in transit");
    // 1 west-neighbor step = Δ(2q+r) = 2 → >= 2 cells = Δ >= 4.
    assert!(
        src - dst >= 4,
        "matter didn't travel ≥ 2 cells: incision up to x2={src}, \
         deposit/load up to x2={dst} (cumulative deposit {deposited})"
    );
}

/// The WATER budget stays conserved while erosion works: the two
/// terrarium invariants hold together, not against each other.
#[test]
fn water_budget_survives_active_erosion() {
    let mut sim = build_river_sim(200_000.0);
    let water_before = total_water_budget(&sim);
    for _ in 0..8 {
        sim.step();
    }
    let (incised, _) = sim.erosion_totals();
    assert!(incised > 0.0, "empty test: erosion didn't do any work");
    let water_after = total_water_budget(&sim);
    let rel = ((water_after - water_before) / water_before.max(1.0)).abs();
    assert!(
        rel < 1e-3,
        "water conservation violated under erosion: {water_before} → {water_after}"
    );
}

/// `erosion.enabled = 0` ⇒ the relief doesn't move a single bit: the
/// switch really isolates the phenomenon (and pre-#105 worlds stay
/// reproducible).
#[test]
fn disabled_erosion_leaves_bedrock_bit_identical() {
    let mut sim = build_river_sim(200_000.0);
    assert!(sim.update_param("erosion.enabled", 0.0));
    let elev_before: Vec<u32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation.to_bits())
        .collect();
    for _ in 0..5 {
        sim.step();
    }
    let (incised, deposited) = sim.erosion_totals();
    assert!(incised == 0.0 && deposited == 0.0);
    for (i, c) in sim.grid().cells_slice().iter().enumerate() {
        assert_eq!(
            c.elevation.to_bits(),
            elev_before[i],
            "elevation changed with erosion off, cell {i}"
        );
        assert_eq!(
            c.sediment_load.to_bits(),
            0.0f32.to_bits(),
            "sediment_load non-zero with erosion off, cell {i}"
        );
    }
}
