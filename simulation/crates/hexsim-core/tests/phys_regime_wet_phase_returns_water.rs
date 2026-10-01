//! Micro-test (#63): a wet episode gives the sky reservoir back to the
//! map, at the time constant it advertises, and the water that comes back
//! rains.
//!
//! The counterpart of `phys_regime_dry_phase_stops_rain`: a lever that
//! only ever takes water out would desertify the terrarium over a few
//! years, and the drift would be blamed on the hydrology. What is pinned
//! here is that the return is a first-order relaxation with
//! `regime_return_hours`, not "some water eventually comes back".
//!
//! Radius 4 for the same reason as the dry test: the pump and the
//! advection are transport, and transport at radius 0 is a silent
//! self-transfer on the torus.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 4;
const RETURN_HOURS: f32 = 12.0;
/// Sky stock (mm, map total) preloaded before the wet episode. Well above
/// what the map holds aloft, so the return dominates every other term of
/// the upper-layer budget over the window.
const SKY_PRELOAD_MM: f32 = 2000.0;

fn sim_with_regime() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let coords: Vec<_> = grid.coords().copied().collect();
    for coord in coords {
        let cell = grid.get_mut(coord).unwrap();
        cell.elevation = 100.0 * f32::from(u8::try_from(coord.r.abs()).unwrap());
        cell.temperature = 10.0;
        cell.water_capacity = 100.0;
        // A dry start: what falls afterwards came from the sky reservoir,
        // not from a lake that was there all along.
        cell.humidity_upper = 0.0;
        cell.humidity_surface = 0.0;
    }
    let atmosphere = AtmosphereParams {
        regime_enabled: 1.0,
        regime_return_hours: RETURN_HOURS,
        ..AtmosphereParams::default()
    };
    Simulation::new(
        grid,
        HydroParams::default(),
        atmosphere,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    )
}

#[test]
fn a_wet_episode_returns_the_sky_at_its_time_constant_and_it_rains() {
    let mut sim = sim_with_regime();
    sim.set_weather_regime(true, SKY_PRELOAD_MM);

    let mut rain_total = 0.0_f32;
    let mut sky_after_tau = f32::NAN;
    for hour in 0..72_u32 {
        // Re-forced every hour: `set_weather_regime` is a seam, the daily
        // draw would otherwise flip the phase at the first midnight.
        sim.set_weather_regime(true, sim.sky_water_total());
        sim.step_hour();
        rain_total += sim.precip_this_tick().iter().map(|d| d.rain).sum::<f32>();
        if hour + 1 == 12 {
            sky_after_tau = sim.sky_water_total();
        }
    }

    // After exactly τ = 12 h the residual of a first-order relaxation is
    // e^-1 of the initial stock. 2 % tolerance: the discrete hourly EMA
    // (gain `1 - e^{-1/τ}` per step) sits a hair off the continuous
    // filter, same argument as `upper_air_smoothing_converges_with_e_fold_time_tau`.
    let expected = SKY_PRELOAD_MM * (-1.0_f32).exp();
    let error = (sky_after_tau - expected).abs() / expected;
    assert!(
        error < 0.02,
        "after τ = {RETURN_HOURS} h the sky must hold e^-1 of its stock: \
         {sky_after_tau} mm vs {expected} mm ({:.1} % off)",
        error * 100.0
    );

    // 72 h = 6τ: e^-6 = 0.25 % must be left.
    let residual = sim.sky_water_total() / SKY_PRELOAD_MM;
    assert!(
        residual < 0.01,
        "after 6τ the sky must be nearly empty, {:.2} % left",
        residual * 100.0
    );

    assert!(
        rain_total > 0.0,
        "the returned water never precipitated: sky went from {SKY_PRELOAD_MM} to {} \
         with 0 mm of rain",
        sim.sky_water_total()
    );
}
