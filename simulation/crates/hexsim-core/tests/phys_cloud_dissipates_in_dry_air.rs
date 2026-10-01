//! Micro-test (#63 L2b): a cloud in subsaturated air is gone within the
//! hour, and nothing rains out of it.
//!
//! ## What it pins
//!
//! The vapour ↔ droplet transition became a saturation adjustment
//! (`atmosphere::condensation::saturation_adjustment_transfer`, Sundqvist
//! 1978 / Tiedtke 1993): below saturation the droplets go back to vapour,
//! bounded by the layer's own deficit and nothing else. On a well-mixed
//! 1500 m column over an hour that is the whole cloud, because the phase
//! relaxation time of a droplet in subsaturated air is seconds (τ ≈ 3 s,
//! Rogers & Yau 1989 ch. 7) against a 3600 s tick.
//!
//! It replaced `cloud_evap_rate` = 0.10/day, a ~7-day half-life. Under
//! that coefficient a 0.5 mm drizzle survived a 4.5-day dry episode and
//! kept raining, which is why `fully_rain_free_days_total` was 0 by
//! construction on every seed (JOURNAL 2026-09-06). The first assertion
//! here is exactly the case that used to survive.
//!
//! ## The control matters as much as the case
//!
//! The second test runs the identical fixture with the layer held
//! *saturated*. The cloud must then survive. Without it, "the cloud
//! disappeared" would also pass on a fixture that simply destroys
//! droplets, and the sentinel would pin nothing.
//!
//! ## Radius 3, not 0
//!
//! Cloud advection and diffusion are transport, and a radius-0 cell is its
//! own neighbour six times on the torus, so any transport there is a
//! silent self-transfer (micro-test rule: no transport at radius 0). The cloud is
//! seeded on every cell so the diffusion between them cannot be mistaken
//! for evaporation either way.

use hexsim_core::atmosphere::{AtmosphereParams, saturation_upper};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 3;
/// Cloud seeded on every cell (mm of liquid water path). The order of
/// magnitude of the residual drizzle measured on the r30 world during a
/// dry episode, 0.2-0.5 mm/cell, and comfortably above `precip_crit_mm`
/// (0.15 mm) so KK2000 would rain it if it were still there.
const CLOUD_MM: f32 = 0.5;
/// Hours run before the verdict. One is enough for the adjustment itself;
/// six leaves room for a droplet to come back from somewhere and be seen.
const HOURS: u32 = 6;

/// A flat, water-free world at a uniform temperature, with every path
/// that could refill `humidity_upper` or recreate droplets switched off:
/// only the vapour ↔ droplet transition and the transport of what already
/// exists are left running.
///
/// `humidity_upper` is set per cell as a fraction of `saturation_upper` at
/// the temperature the upper layer will actually have — on this flat map
/// with a uniform surface temperature, that is `T − Γ·H/1000` (the
/// map-mean formula of `upper_air_temperature` with `z = z̄`).
fn world_at_relative_humidity(rh: f32) -> Simulation {
    let atmosphere = AtmosphereParams {
        // No source aloft: no pump, no uplift, no thermal convection.
        uplift_rate: 0.0,
        uplift_thermal_coef: 0.0,
        convective_diurnal_coef: 0.0,
        orographic_lift_coef: 0.0,
        // No droplet source either: fog is the other way into
        // `cloud_water`, from the surface layer.
        fog_condensation_rate: 0.0,
        // The floor is applied at construction and would overwrite the
        // humidity this fixture is about.
        initial_humidity_floor: 0.0,
        // #63/#146: the control (saturated layer) needs to actually stay
        // saturated for `HOURS` with no wind — the imposed weather
        // regime (on by default since 2026-09-06) would drain the
        // surplus toward its own dry/wet target on its own clock,
        // regardless of the settings above, and desaturate the layer the
        // control is about.
        regime_enabled: 0.0,
        ..AtmosphereParams::default()
    };
    let temperature = TemperatureParams::default();
    let surface_t = 10.0_f32;
    let t_upper = surface_t - temperature.lapse_rate * atmosphere.upper_layer_altitude_m / 1000.0;
    let humidity_upper = rh * saturation_upper(t_upper, &atmosphere);

    let mut grid = HexGrid::from_radius(RADIUS);
    for cell in grid.cells_slice_mut() {
        cell.elevation = 0.0;
        cell.water_level = 0.0;
        cell.temperature = surface_t;
        cell.humidity_surface = 0.0;
        cell.humidity_upper = humidity_upper;
        cell.cloud_water = CLOUD_MM;
    }
    Simulation::new(
        grid,
        HydroParams::default(),
        atmosphere,
        GroundwaterParams::default(),
        SnowParams::default(),
        temperature,
        // No vapour advection: what is aloft stays above its own cell, so
        // "the layer is dry" holds all run long. Cloud advection and
        // diffusion stay on, they only move droplets around.
        WindParams {
            humidity_advection_rate: 0.0,
            ..WindParams::default()
        },
    )
}

fn total_cloud_water(sim: &Simulation) -> f32 {
    sim.grid().iter().fold(0.0, |a, (_, c)| a + c.cloud_water)
}

/// Runs `hours` and returns `(map-wide cloud left, map-wide rain fallen)`.
fn run(sim: &mut Simulation, hours: u32) -> (f32, f32) {
    let mut rain = 0.0_f32;
    for _ in 0..hours {
        sim.step_hour();
        rain += sim.precip_this_tick().iter().map(|d| d.rain).sum::<f32>();
    }
    (total_cloud_water(sim), rain)
}

/// The case: RH 0.3 aloft, the range a dry episode's subsidence produces.
/// The layer's deficit is then ~70 % of `saturation_upper` ≈ 4.6 mm at
/// this temperature, nine times the 0.5 mm of cloud, so the adjustment
/// takes all of it and the layer never even reaches saturation.
#[test]
fn a_cloud_in_subsaturated_air_is_gone_within_the_hour_and_never_rains() {
    let mut sim = world_at_relative_humidity(0.3);
    let seeded = total_cloud_water(&sim);
    assert!(seeded > 0.0, "the fixture seeded no cloud at all");

    // `cloud_water` and the rain accumulator are non-negative stocks, so
    // `<= 0.0` is "exactly zero" without an equality on floats (clippy
    // `float_cmp`, and no `#[allow]` in this project). Exact is the point:
    // the adjustment leaves a true zero, not a residue under a floor.
    let (after_one_hour, rain_one_hour) = run(&mut sim, 1);
    assert!(
        after_one_hour <= 0.0,
        "a cloud in air at RH 0.3 must be entirely gone after one hour, {after_one_hour} mm left \
         of {seeded} mm"
    );
    assert!(
        rain_one_hour <= 0.0,
        "the cloud rained {rain_one_hour} mm on its way out: KK2000 saw droplets the \
         adjustment should have evaporated first"
    );

    let (after, rain) = run(&mut sim, HOURS - 1);
    assert!(
        after <= 0.0,
        "droplets came back in a subsaturated layer: {after} mm after {HOURS} h"
    );
    assert!(rain <= 0.0, "it rained {rain} mm over the following hours");
}

/// The control: same fixture, same cloud, but the layer held at
/// saturation. The adjustment has no deficit to work with, so the cloud
/// must still be there — this is what makes the test above a statement
/// about the deficit and not about the fixture.
///
/// RH 1.2 rather than exactly 1.0: the condensation branch drains the
/// surplus into droplets in one tick, which lands the layer at saturation
/// and leaves the cloud strictly larger than it was seeded. Sitting
/// exactly on the boundary would make the test a race between f32 rounding
/// on either side of it.
#[test]
fn the_same_cloud_survives_when_the_layer_is_saturated() {
    let mut sim = world_at_relative_humidity(1.2);
    let seeded = total_cloud_water(&sim);
    let (after, _) = run(&mut sim, HOURS);
    assert!(
        after >= seeded,
        "a saturated layer cannot evaporate its own droplets: {seeded} mm -> {after} mm \
         after {HOURS} h"
    );
}
