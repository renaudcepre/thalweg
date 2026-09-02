//! #103 (1): hydro flux maps stay readable at ANY hour.
//!
//! Historical instrumentation bug: `discharge_map`/`flow_vec_map` were
//! filled at the midnight tick (`step_daily_tail`) then reset at the
//! start of the following `step_hour`. Any read outside exact midnight
//! saw `outflow 0`, `river_cells 0`; a misaligned world (`hour_of_day
//! != 0`) displayed them empty permanently even though hydro was
//! actually running.
//!
//! The fix moves the reset into `step_hydro_tranche` (right before the
//! 8 MFD passes): between two daily tranches, the maps keep the last
//! flow. This test pins that behavior: if a `phys_flux_maps_*` goes
//! red, someone put the reset back at day start (or broke tranche
//! accumulation); the front-end ribbons would go empty again 23 hours
//! out of 24.
//!
//! Micro e2e-unit test: radius 2 (transport => radius >= 2, never 0; on
//! the torus a radius-0 cell is its own neighbor x6), a few days.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

/// Radius-2 world with an east-to-west ramp and a water stock at the
/// top: MFD flow is guaranteed there on every daily tranche.
fn sloped_wet_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(2);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(c) = grid.get_mut(coord) {
            // Ramp along world x (∝ 2q + r, integer): a clear slope, no
            // plateau where water would equilibrate on day one.
            c.elevation = f32::from(
                i16::try_from(500 + 40 * (2 * coord.q + coord.r)).expect("elevation fits i16"),
            );
            c.water_level = 5.0;
            c.water_capacity = 0.5;
        }
    }
    if let Some(c) = grid.get_mut(HexCoord::new(2, 0)) {
        c.water_level = 200.0; // reservoir at the top of the slope
    }
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    )
}

#[test]
fn discharge_map_stays_readable_on_desaligned_world() {
    let mut sim = sloped_wet_sim();

    // 2 full days: the 2nd daily hydro tranche has filled the maps.
    sim.step();
    sim.step();
    let at_midnight = sim.diagnostics().hydrology.max_discharge;
    assert!(
        at_midnight > 0.0,
        "broken setup: no flow at exact midnight (max_discharge = {at_midnight})"
    );

    // Misalign the world: +7h. No hydro tranche runs during these
    // hours (Tier 3 = daily); the maps must KEEP the last flow, not
    // get reset at the midnight tick.
    for hour in 1..=7 {
        sim.step_hour();
        let d = sim.diagnostics().hydrology.max_discharge;
        assert!(
            d > 0.0,
            "hour {hour}: max_discharge = {d}, flux maps were reset \
             outside the hydro tranche (regression #103)"
        );
    }

    // The value read outside midnight is indeed THAT of the last tranche.
    let desaligned = sim.diagnostics().hydrology.max_discharge;
    assert!(
        (desaligned - at_midnight).abs() < 1e-6,
        "maps must stay stable between two tranches: midnight {at_midnight} vs +7h {desaligned}"
    );
}

#[test]
fn edge_flux_export_matches_discharge_at_any_hour() {
    let mut sim = sloped_wet_sim();
    sim.step();
    sim.step_hour(); // world out of alignment (hour_of_day = 1)

    let discharge = sim.discharge_map();
    let edge_flux = sim.edge_flux_map();
    let mut any_flow = false;
    for (i, edges) in edge_flux.iter().enumerate() {
        let edge_sum: f32 = edges.iter().sum();
        any_flow |= edge_sum > 0.0;
        // Exact decomposition: the per-edge export (#103) sums over the
        // aggregated discharge, including outside midnight. Tolerance
        // = f32 epsilon from summation order (8 passes x 6 directions).
        assert!(
            (edge_sum - discharge[i]).abs() < 1e-3 * discharge[i].max(1.0),
            "cell {i}: Σ edge_flux = {edge_sum} != discharge = {}",
            discharge[i]
        );
    }
    assert!(any_flow, "broken setup: no edge flux exported");
}
