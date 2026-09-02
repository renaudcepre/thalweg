//! KK2000 test #4: lifetime of an isolated cloud.
//!
//! Once formed, a cloud must stay visible for several hours before
//! disappearing. Today the linear drain empties `cloud_water` in
//! ~3-5 ticks (calculation: 0.5 - 0.05 = 0.45 mm initial, drain
//! `0.25/24` per hour → ~5 mm/hour linearized, empty in ~2-3 h). KK2000
//! must offer a persistence ≥ 12 h because the super-linear formula
//! becomes very weak as `cloud_water` drops.
//!
//! Setup:
//! - radius 2, flat terrain 200 m, T = 15 °C, calm wind.
//! - An initial puff `cloud_water = 0.5 mm` at (0, 0). No humidity
//!   elsewhere (the cloud can evaporate but not regenerate).
//!
//! Assertion: after 12 hours, the sum of `cloud_water` over the grid
//! stays > 0.05 mm. The cloud still exists somewhere (it may have
//! diffused to a few neighbors but not fully evaporated).

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

#[test]
fn isolated_cloud_persists_at_least_12_hours() {
    let mut grid = HexGrid::from_radius(2);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            cell.temperature = 15.0;
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
            cell.humidity_surface = 0.0;
            cell.humidity_upper = 0.0;
            cell.cloud_water = 0.0;
        }
    }
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.cloud_water = 0.5;
    }

    let atmo = AtmosphereParams {
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    };
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );

    for _ in 0..12 {
        sim.step_hour();
    }

    let total_cloud: f32 = sim.grid().cells_slice().iter().map(|c| c.cloud_water).sum();

    assert!(
        total_cloud > 0.05,
        "Isolated cloud vanished too fast: total cloud_water after 12h = \
         {total_cloud:.4} mm (expected > 0.05). KK2000 expects a \
         0.5 mm cloud to drain slowly because P_auto ∝ q_c^2.47 \
         drops sharply as cloud_water decreases."
    );
}
