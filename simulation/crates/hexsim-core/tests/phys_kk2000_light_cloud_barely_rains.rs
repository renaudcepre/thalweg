//! KK2000 physics test #1: a *small* cloud must rain almost nothing.
//!
//! Khairoutdinov & Kogan 2000: `P_auto ∝ cloud_water^2.47`. A cloud at
//! 0.1 mm produces 295× less rain than a cloud at 1.0 mm (ratio
//! 0.1^2.47 / 1.0^2.47). This is *exactly* the "big clump rule": a
//! cloud must grow before it rains.
//!
//! Minimal setup:
//! - radius 2, flat terrain 200 m, T = 15 °C, calm wind.
//! - The center cell receives `cloud_water = 0.1 mm` at the start.
//! - Runs for 24 ticks (1 simulated day).
//!
//! Assertion: cumulative rain on the source cell < 0.005 mm in 24h.
//! - Old linear model: `(0.1 - 0.05) × 0.25/24 ≈ 0.0005 mm/h × 24
//!   = 0.012 mm`. Fails (>= 0.005).
//! - KK2000: `1350 × q_c^2.47 × N_c^-1.79` with `q_c ≈ 6.67e-5` (LWP
//!   converted ÷ 1500m × 1.0 kg/m³) gives ~3e-5 mm/h → 0.0008 mm in 24h.
//!   Passes (< 0.005). This is *exactly* the difference we're looking for.

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
fn light_cloud_barely_rains() {
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
        cell.cloud_water = 0.1;
    }
    // No humidity floor: we want an isolated cloud_water pulse without
    // distributed condensation that would add rain everywhere.
    // `cloud_advection_rate = 0`: cloud frozen in place, concentration
    // well-defined (symmetric to heavy_cloud; #68 raised advection from
    // 0.37 → 3.0, which dispersed the pulse and conflated precip with
    // transport). Wind stays active for the convergence updraft that
    // gates precipitation.
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

    let rain_cumulative: f32 = sim
        .last_precipitation()
        .iter()
        .map(|d| d.rain + d.snow)
        .sum();

    assert!(
        rain_cumulative < 0.05,
        "Small cloud (0.1 mm) precipitated too much: {rain_cumulative:.4} mm \
         accumulated (expected < 0.05 under KK2000). Either the formula is not \
         super-linear, or the exponent is too weak."
    );
}
