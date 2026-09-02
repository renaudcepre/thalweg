//! Integration test: confluence of two springs under MFD.
//!
//! Two upstream sources (A, B) flow into a confluence cell (C) that is
//! a topographic sink (all neighbors higher). In MFD, a cell's
//! `discharge` = `flux_out` = sum of outgoing transfers, so:
//!   - `discharge[A]` and `discharge[B]` > 0 (they send toward C)
//!   - `discharge[C]` = 0 (C is a sink, no lower neighbor)
//!   - `water_level[C]_after` ≈ `water_level[C]_before` + `discharge[A]` + `discharge[B]`
//!     (C receives exactly what A and B send it: conservation).
//!
//! If this test fails, a parasitic flux (infiltration, precip, evap) is
//! contaminating the confluence, or the `flux_out`/`discharge` count has
//! diverged.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const SRC_A: HexCoord = HexCoord { q: -1, r: 0 };
const SRC_B: HexCoord = HexCoord { q: 1, r: -1 };
const CONFLUENCE: HexCoord = HexCoord { q: 0, r: 0 };

fn build_confluence_sim() -> Simulation {
    // Topology: A and B at elev=100 (both neighbors of C), C at elev=0.
    // Non-source neighbors are at elev=200 (walls) so the MFD mobile
    // water from A and B only flows down toward C.
    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = if coord == CONFLUENCE {
                0.0
            } else if coord == SRC_A || coord == SRC_B {
                100.0
            } else {
                200.0
            };
            cell.water_level = if coord == SRC_A || coord == SRC_B {
                10.0
            } else {
                0.0
            };
            // water_capacity=0: all of water_level is mobile. Otherwise
            // the sub-capacity residue of A, B would stay in place, and
            // extend_basins_to_overflowing_neighbors would pull it into
            // C's basin through equalization, adding a "leak" outside
            // the MFD flow we want to measure.
            cell.water_capacity = 0.0;
            // Phase 3 (#32): T < 0 to disable Meyer evap and keep the
            // test focused on pure MFD flow conservation. The
            // step_evaporation filter ignores cells at T < 0 (frozen
            // surface water, no liquid evap).
            cell.temperature = -1.0;
            cell.humidity_upper = 0.0;
            cell.permeability = 0.0;
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
        // Phase 3 (#32): T < 0 to disable Meyer evap (see cell setup).
        base_temp: -1.0,
        water_cooling: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };

    Simulation::new(
        grid,
        hydro,
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        temp,
        wind,
    )
}

#[test]
fn confluence_sink_has_zero_discharge() {
    // Sanity: in MFD, a sink can't send anything (no lower neighbor).
    let mut sim = build_confluence_sim();
    sim.step();

    let grid = sim.grid();
    let d_a = grid
        .cell_index(SRC_A)
        .and_then(|i| sim.discharge_map().get(i).copied())
        .unwrap_or(0.0);
    let d_b = grid
        .cell_index(SRC_B)
        .and_then(|i| sim.discharge_map().get(i).copied())
        .unwrap_or(0.0);
    let d_c = grid
        .cell_index(CONFLUENCE)
        .and_then(|i| sim.discharge_map().get(i).copied())
        .unwrap_or(0.0);

    assert!(d_a > 0.0, "source A must have nonzero discharge: {d_a}");
    assert!(d_b > 0.0, "source B must have nonzero discharge: {d_b}");
    assert!(
        d_c < 1e-4,
        "confluence (sink) must have zero discharge: {d_c}"
    );
}

#[test]
fn confluence_accumulates_sum_of_affluents() {
    // Conservation: the water received by C between before and after
    // the tick = flux_out[A] + flux_out[B] = discharge[A] + discharge[B].
    let mut sim = build_confluence_sim();
    let wl_before = sim.grid().get(CONFLUENCE).unwrap().water_level;

    sim.step();

    let wl_after = sim.grid().get(CONFLUENCE).unwrap().water_level;
    let grid = sim.grid();
    let d_a = grid
        .cell_index(SRC_A)
        .and_then(|i| sim.discharge_map().get(i).copied())
        .unwrap_or(0.0);
    let d_b = grid
        .cell_index(SRC_B)
        .and_then(|i| sim.discharge_map().get(i).copied())
        .unwrap_or(0.0);

    let received = wl_after - wl_before;
    let expected = d_a + d_b;
    let rel_err = (received - expected).abs() / expected.max(1e-6);
    assert!(
        rel_err < 0.02,
        "confluence received {received:.4}, expected d_A ({d_a:.4}) + d_B ({d_b:.4}) = {expected:.4} \
         (relative err {:.3} %)",
        rel_err * 100.0
    );
}
