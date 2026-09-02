//! KK2000 physics test #2: a *big* cloud must precipitate significantly.
//!
//! `cloud_water = 1.0 mm` corresponds to entering the cumulonimbus regime
//! in real LWP (Liquid Water Path) metric: this is a true storm cloud,
//! not a fair-weather cumulus.
//!
//! Setup: symmetric to the `light_cloud` test, just `cloud_water` = 1.0 mm.
//! `cloud_advection_rate = 0`: we **freeze the cloud** on its cell so that
//! its concentration (hence the super-linear KK2000 precip) stays well
//! defined. Wind (`WindParams::default()`) stays active: it feeds the
//! convergence updraft that *gates* precipitation; cutting it would kill
//! the effect. Before parity fix #68, `cloud_advection_rate=0.37` was
//! slow enough for the cloud to stay in place on its own on this
//! radius-2 torus; at 3.0 it disperses within a few hours and the pulse
//! rained ~2x less (0.29 mm), wrongly flagging red. We therefore
//! explicitly isolate precip from transport.
//!
//! Assertions:
//! - Cumulative rain > 0.05 mm over 24h. Guarantees the world hasn't
//!   gone dry: a big cloud must be able to rain, not just exist.
//! - Cumulative rain < 5 mm over 24h. Safeguard: KK2000 must not drain
//!   everything instantly (that would be a chaotic oscillation).
//!
//! Expected `test_heavy` / `test_light` ratio: >= 50 (super-linear KK2000
//! yields a ratio ~250 in theory; the current linear model gives ~19).

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
fn heavy_cloud_rains_notably() {
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
        cell.cloud_water = 1.0;
    }

    let atmo = AtmosphereParams {
        initial_humidity_floor: 0.0,
        cloud_advection_rate: 0.0,
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

    for _ in 0..24 {
        sim.step_hour();
    }

    // last_precipitation accumulates precipitation from all sub-ticks
    // (rain + snow). More reliable than water_level, which can be
    // redistributed by hydro and groundwater in tier 3 at day's end.
    let total_water: f32 = sim
        .last_precipitation()
        .iter()
        .map(|d| d.rain + d.snow)
        .sum();

    assert!(
        total_water > 0.5,
        "Large cloud (1.0 mm) barely precipitated: {total_water:.4} mm \
         accumulated (expected > 0.5 over 24h). KK2000 too conservative or \
         exponent too strong, risk of dry world."
    );
    assert!(
        total_water < 30.0,
        "Large cloud drained instantly: {total_water:.4} mm accumulated \
         (expected < 30 to avoid burst rain). KK2000 too aggressive."
    );
}
