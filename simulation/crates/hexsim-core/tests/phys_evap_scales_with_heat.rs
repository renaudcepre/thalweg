//! Micro-e2e: with equal water and cover, a HOTTER world evaporates faster.
//!
//! "Obvious but hard-won" property (SI energy balance #42-47,
//! FAO-56 #77) that we set in stone: evaporative demand follows the
//! saturation deficit, which rises with temperature (Clausius-Clapeyron,
//! cf. `saturation_surface_rises_strictly_with_temperature`). Two worlds
//! identical except for `base_temp`; the hot one must have loaded the
//! atmosphere faster. A red result means the T → deficit → evaporation
//! chain is broken.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

/// Small flat world, high water table + cover everywhere, air initially dry.
/// Only `base_temp` changes between the two calls.
fn watered_world(base_temp: f32) -> Simulation {
    let mut grid = HexGrid::from_radius(2);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        let c = grid.get_mut(coord).unwrap();
        c.elevation = 200.0;
        c.groundwater = 80.0;
        c.water_level = 0.0;
        c.vegetation = [0.6, 0.0, 0.0, 0.0, 0.0];
        c.humidity_surface = 0.0;
        c.humidity_upper = 0.0;
        c.cloud_water = 0.0;
    }
    let temp = TemperatureParams {
        base_temp,
        ..TemperatureParams::default()
    };
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        temp,
        WindParams::default(),
    )
}

/// Total atmospheric humidity of the grid (direct product of evaporation
/// before it falls back as rain).
fn air_moisture(sim: &Simulation) -> f32 {
    sim.grid()
        .cells_slice()
        .iter()
        .map(|c| c.humidity_surface + c.humidity_upper + c.cloud_water)
        .sum()
}

#[test]
fn warmer_world_evaporates_faster() {
    let mut warm = watered_world(26.0);
    let mut cold = watered_world(6.0);
    // 30 days: enough to load the atmosphere, short enough for the
    // evaporation signature to dominate the full rain cycle.
    for _ in 0..30 {
        warm.step();
        cold.step();
    }
    // Atmospheric humidity produced = direct signature of the evaporation
    // rate. (We do NOT measure the remaining water table: under dense cover
    // it drains toward 0 in both worlds, so it doesn't discriminate the
    // rate; only the air stock does.)
    let (warm_air, cold_air) = (air_moisture(&warm), air_moisture(&cold));
    assert!(
        warm_air > cold_air,
        "the warm world must have evaporated more: warm air={warm_air:.3} mm vs cold={cold_air:.3} mm"
    );
}
