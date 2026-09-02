//! Symmetry of the "dry land" test: a VEGETATED cell with a high water table
//! must lose groundwater and feed `humidity_surface` through transpiration
//! (#77, FAO-56), *without* open water.
//!
//! Setup:
//! - radius 3, flat terrain 200 m, T = 25 °C.
//! - `groundwater` = 80 mm everywhere (high water table, no water stress).
//! - `vegetation` = 0.6 everywhere (established cover, transpiration scales
//!   with biomass, bare soil does not transpire).
//! - `water_level` = 0 everywhere (no open-water Meyer evaporation).
//!
//! After 200 ticks (~8 simulated days):
//! - sum of `humidity_surface` > 5 mm (the cover transpires vapor).
//! - sum of `groundwater` drops by at least 5 mm (uptake = transpiration).

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
fn humid_vegetated_soil_feeds_atmosphere() {
    let mut grid = HexGrid::from_radius(3);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            cell.temperature = 25.0;
            cell.water_level = 0.0;
            cell.groundwater = 80.0;
            // Established cover (0.6 total biomass, one species): transpiration
            // scales with total cover.
            cell.vegetation = [0.6, 0.0, 0.0, 0.0, 0.0];
            cell.humidity_surface = 0.0;
            cell.humidity_upper = 0.0;
            cell.cloud_water = 0.0;
        }
    }

    let initial_gw_total: f32 = grid.cells_slice().iter().map(|c| c.groundwater).sum();

    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );

    for _ in 0..200 {
        sim.step();
    }

    let cells = sim.grid().cells_slice();
    let humidity_total: f32 = cells.iter().map(|c| c.humidity_surface).sum();
    let gw_total: f32 = cells.iter().map(|c| c.groundwater).sum();
    let gw_loss = initial_gw_total - gw_total;

    assert!(
        humidity_total > 5.0,
        "Vegetation cover did not transpire toward the atmosphere: humidity_surface \
         total = {humidity_total:.3} mm across the grid (expected > 5)."
    );
    assert!(
        gw_loss > 5.0,
        "Water table did not drop as expected: total delta = {gw_loss:.3} mm \
         (initial {initial_gw_total:.2} → final {gw_total:.2}, expected > 5). \
         Either transpiration is not drawing from groundwater (conservation broken), \
         or some mechanism is recharging it in parallel."
    );
}
