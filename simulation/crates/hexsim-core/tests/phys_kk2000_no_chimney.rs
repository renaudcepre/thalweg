//! KK2000 test #5 (guard rail): no permanent chimney over a warm lake.
//!
//! Anticipated risk: KK2000 delays rain, so a lake that evaporates
//! continuously could build up a `cloud_water` column that
//! self-feeds and never purges. Night cuts off daytime convection
//! (a good brake), but Meyer evaporation keeps going, and wind
//! advects the water away but can be refilled by the source faster
//! than it exports.
//!
//! Setup:
//! - radius 4, flat terrain 200 m, T = 18 °C (temperate climate).
//! - A lake (`water_level` = 600 mm) at the center. Normal wind (full
//!   defaults, not zero, we want a realistic climate).
//! - 30 days = 720 ticks.
//!
//! Assertion: over the whole run, no cell exceeds
//! `cloud_water = 3.0 mm` (= severe CB regime). 1.5 mm = plausible and
//! acceptable persistent cumulus. Higher = runaway column.
//!
//! If this test goes red, we'll know an additional mechanism is
//! needed (e.g. an implicit microphysical cap, or a more aggressive
//! formulation above a saturation threshold).

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
fn warm_lake_does_not_create_perpetual_chimney() {
    let mut grid = HexGrid::from_radius(4);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            cell.temperature = 18.0;
            cell.water_level = 0.0;
        }
    }
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.water_level = 600.0;
    }

    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );

    let mut max_observed: f32 = 0.0;
    for _ in 0..720 {
        sim.step_hour();
        let tick_max = sim
            .grid()
            .cells_slice()
            .iter()
            .map(|c| c.cloud_water)
            .fold(0.0_f32, f32::max);
        max_observed = max_observed.max(tick_max);
    }

    assert!(
        max_observed < 3.0,
        "Chimney detected over warm lake: cloud_water peak = \
         {max_observed:.3} mm over 30 days (expected < 3.0). KK2000 \
         lets the cloud grow without purging, add a microphysical cap \
         mechanism or a more aggressive supersaturation regime."
    );
}
