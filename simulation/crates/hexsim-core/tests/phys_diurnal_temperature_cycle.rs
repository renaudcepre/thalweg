//! Micro e2e: a cell's temperature follows a day/night cycle, hot in
//! the afternoon, cold before dawn.
//!
//! Pins the emergent result of the SI diurnal energy budget (#42-47): the
//! hourly solar forcing (`solar_beam_at_tick`) heats the day, net
//! radiation cools the night. Solar geometry is already tested
//! (elevation, day length); here we pin the THERMAL consequence, that
//! `step_temperature` does produce a diurnal wave of the right phase.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

fn single_dry_cell() -> Simulation {
    let mut grid = HexGrid::from_radius(1);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        let c = grid.get_mut(coord).unwrap();
        c.elevation = 200.0;
        c.water_level = 0.0;
        c.groundwater = 0.0; // dry soil: low inertia, sharp diurnal wave
        c.humidity_surface = 0.0;
        c.humidity_upper = 0.0;
        c.cloud_water = 0.0;
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
fn temperature_peaks_in_afternoon_and_bottoms_before_dawn() {
    let mut sim = single_dry_cell();
    let center = HexCoord::new(0, 0);
    let ci = sim.grid().index_of(center).unwrap();

    // Warm-up: 3 days to get past the initial transient and settle into
    // the steady-state diurnal wave.
    for _ in 0..(3 * 24) {
        sim.step_hour();
    }

    // Sample the 24 hours of the following day, indexed by local hour.
    let mut temp_by_hour = [0.0_f32; 24];
    for _ in 0..24 {
        let hour = (sim.hour_tick() % 24) as usize;
        temp_by_hour[hour] = sim.grid().cells_slice()[ci].temperature;
        sim.step_hour();
    }

    let (hot_hour, &hot) = temp_by_hour
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let (cold_hour, &cold) = temp_by_hour
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Non-trivial diurnal amplitude (the wave really exists).
    assert!(
        hot - cold > 2.0,
        "diurnal amplitude too low: max {hot:.1} °C, min {cold:.1} °C"
    );
    // Phase: the peak is in the afternoon (solar noon + thermal inertia lag
    // ⇒ ~12-18 h), the trough is late night (~3-8 h), never the reverse.
    assert!(
        (12..=18).contains(&hot_hour),
        "the peak must be in the afternoon, measured at {hot_hour} h"
    );
    assert!(
        (2..=8).contains(&cold_hour),
        "the trough must be late at night, measured at {cold_hour} h"
    );
}
