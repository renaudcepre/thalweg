//! Micro-test (#63): what a dry episode of the imposed weather regime
//! actually guarantees, and what it does not.
//!
//! ## The guarantee
//!
//! While the synoptic chain says "dry", the map loses vapour aloft every
//! hour, the sky reservoir gains exactly that, the droplet reservoir
//! collapses and stays collapsed, and the rain goes to zero. Those four
//! are what the pass is answerable for and what this file pins.
//!
//! ## The droplet half of it changed at #63 L2b
//!
//! Until 2026-09-06 the third assertion read "no net droplet is created
//! anywhere", checked hour by hour as a monotone decrease of the map's
//! `cloud_water`. That was a statement about a stock draining slowly, not
//! about the pass: droplets left through KK2000 and through
//! `cloud_evap_rate` = 0.10/day, a ~7-day half-life against episodes
//! averaging 4.5 days, so the reservoir always dominated the total and
//! always went down. On the real r30 world it left 65 cells of 2 791
//! drizzling on the driest day of the year and
//! `fully_rain_free_days_total` at 0 on every seed.
//!
//! L2b replaced that coefficient by a saturation adjustment. Measured on
//! this fixture, dry phase forced, map totals per hour:
//!
//! | hour | 0 | 4 | 8 | 16 | 40 |
//! |---|---|---|---|---|---|
//! | `cloud_water`, before | 722 | 111 | 62.0 | 36.6 | 17.5 |
//! | `cloud_water`, after  | 722 | 55.5 | 0.33 | 0.002 | 0.0003 |
//! | rain mm/h, before | 244 | 39.3 | 6.95 | 1.86 | 0.50 |
//! | rain mm/h, after  | 244 | 8.06 | 0.005 | 0.000 | 0.000 |
//!
//! Two consequences for this file. The rain assertion gets to be
//! absolute: it really does reach zero, so the test asks for zero rather
//! than for a tenfold collapse. And the monotonicity assertion has to go,
//! because it is now false and was never the pass's promise: with the
//! reservoir at the noise floor, the orographic pump still re-saturates a
//! column here and there and condenses a fraction of a millimetre into it
//! (0.33 → 1.45 mm at hour 9, wiped again at hour 10). What is true, and
//! what replaces it, is that the reservoir stays *collapsed*.
//!
//! ## What it still does NOT guarantee
//!
//! **The export is a rate, and it can be outrun.** It moves
//! `excess × (1 − exp(−1/τ))` per hour, 15 % of it at the shipped
//! τ = 6 h. On a column the orographic pump refills faster than that, the
//! leftover still exceeds saturation when the transition reads it, so RH
//! comes out pinned at exactly 1.000 hour after hour even though the map
//! as a whole is drying. An assertion of the form "every column sits
//! below the target" is therefore false on a strongly-supplied world and
//! true on a weakly supplied one; it would be a statement about the
//! fixture.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 4;
const RH_TARGET: f32 = 0.5;

/// A world that rains hard: a loaded upper layer (30 mm against a
/// saturation of ~11 mm at this relief) over shallow water, with a ridge
/// along `r` so the orographic pump has something to lift against — the
/// pump being exactly what keeps summits raining in the real engine.
///
/// Radius 4, not 0: the pump and the humidity advection are transport, and
/// transport at radius 0 is a silent self-transfer on the torus (micro-test
/// rule).
fn rainy_world(regime_enabled: f32) -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let coords: Vec<_> = grid.coords().copied().collect();
    for coord in coords {
        let cell = grid.get_mut(coord).unwrap();
        cell.elevation = 100.0 * f32::from(u8::try_from(coord.r.abs()).unwrap());
        cell.water_level = 50.0;
        cell.water_capacity = 500.0;
        cell.temperature = 15.0;
        cell.humidity_upper = 30.0;
        cell.humidity_surface = 1.0;
    }
    let atmosphere = AtmosphereParams {
        regime_enabled,
        regime_dry_rh_target: RH_TARGET,
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

/// Runs `hours`, re-forcing the phase every hour, and returns the map-wide
/// rain (mm). `set_weather_regime` is a seam, not a lock: the daily draw
/// still runs at midnight, so a multi-day window has to keep re-forcing.
fn run_dry(sim: &mut Simulation, hours: u32) -> f32 {
    let mut total = 0.0_f32;
    for _ in 0..hours {
        sim.set_weather_regime(false, sim.sky_water_total());
        sim.step_hour();
        total += sim.precip_this_tick().iter().map(|d| d.rain).sum::<f32>();
    }
    total
}

fn run_free(sim: &mut Simulation, hours: u32) -> f32 {
    let mut total = 0.0_f32;
    for _ in 0..hours {
        sim.step_hour();
        total += sim.precip_this_tick().iter().map(|d| d.rain).sum::<f32>();
    }
    total
}

fn total_upper_humidity(sim: &Simulation) -> f32 {
    sim.grid()
        .iter()
        .fold(0.0_f32, |a, (_, c)| a + c.humidity_upper)
}

fn total_cloud_water(sim: &Simulation) -> f32 {
    sim.grid()
        .iter()
        .fold(0.0_f32, |a, (_, c)| a + c.cloud_water)
}

/// Hours the reservoir is given to collapse before the bound below
/// applies. Measured: it is under 1 mm from hour 8 (see the table in the
/// module doc); 12 leaves half again as much room.
const COLLAPSE_HOURS: u32 = 12;
/// Map-wide `cloud_water` (mm over 61 cells) the reservoir must stay
/// under once collapsed. Measured peak of the residual pulses over hours
/// 12-48: 1.4 mm, against 722 mm at the start of the episode. 5 mm is a
/// 3.5x margin on the pulses and 140x below the initial stock, so it
/// separates "collapsed with the pump still ticking over" from "the
/// reservoir survived the episode" without being a fit to the trace.
const COLLAPSED_CLOUD_MM: f32 = 5.0;

/// The mechanism, hour by hour: vapour leaves the columns, the sky gains
/// it, and the droplet reservoir collapses and stays collapsed. Not
/// "never grows" — see the module doc for why that assertion was about a
/// slowly draining stock rather than about this pass.
#[test]
fn a_dry_episode_drains_the_columns_into_the_sky_and_collapses_the_droplets() {
    let mut sim = rainy_world(1.0);
    sim.set_weather_regime(false, 0.0);
    // One hour first: the stocks below are compared against the state the
    // regime has already acted on once, not against the fixture's t0.
    run_dry(&mut sim, 1);
    let mut upper = total_upper_humidity(&sim);
    let mut sky = sim.sky_water_total();
    let seeded_cloud = total_cloud_water(&sim);

    for hour in 1..48_u32 {
        run_dry(&mut sim, 1);
        let now_upper = total_upper_humidity(&sim);
        let now_sky = sim.sky_water_total();
        let now_cloud = total_cloud_water(&sim);
        assert!(
            now_sky > sky,
            "hour {hour}: the sky stopped filling ({sky} -> {now_sky})"
        );
        assert!(
            now_upper < upper,
            "hour {hour}: the columns stopped draining ({upper} -> {now_upper})"
        );
        if hour >= COLLAPSE_HOURS {
            assert!(
                now_cloud < COLLAPSED_CLOUD_MM,
                "hour {hour}: the droplet reservoir did not stay collapsed \
                 ({now_cloud} mm, from {seeded_cloud} mm at the start of the \
                 episode) — droplets are outliving the dry phase again"
            );
        }
        upper = now_upper;
        sky = now_sky;
    }

    assert!(
        sky > 0.0,
        "the columns drained but nothing reached the sky: {sky} mm"
    );
    assert!(
        !sim.weather_regime_is_wet(),
        "the phase must still be dry at the end of the window"
    );
}

/// Map-wide rain (mm) tolerated over the third day of a dry episode.
/// Measured at 0.0016 mm on 2026-09-06 (61 cells, so 2.6e-5 mm/cell);
/// 0.01 keeps a 6x margin and still sits two orders of magnitude below
/// the 1e-4 mm/cell the bench needs to count a cell-day as rainy.
///
/// Both assertions of this file were re-run against the build before
/// #63 L2b to check they still bite. That build fails this one at
/// 5.91 mm of rain on day 3 (its own control: 133.7 mm) and the collapse
/// assertion at 45.9 mm of cloud still standing at hour 12.
const DRY_DAY_RAIN_CEILING_MM: f32 = 0.01;

/// The effect: the same world, same hours, with and without the regime.
///
/// The bound was "a tenfold collapse" while a drizzle survived every
/// episode. Since #63 L2b the third day of a dry episode is dry in the
/// only sense that means anything: **0.0016 mm map-wide over the whole
/// 24 h**, against 123.3 mm for the control — a factor of 77 000, and
/// 2.6e-5 mm per cell, two orders of magnitude below the 1e-4 mm the
/// bench metrics need to call a cell-day rainy.
///
/// The first draft of this assertion asked for an exact 0.0 and was red
/// on the first run: the pump still re-saturates a column now and then
/// and KK2000 takes a sliver out of it. Recorded rather than rounded
/// away — the bound below is the measured residual with a 6x margin, not
/// a zero the model does not deliver.
#[test]
fn a_dry_episode_stops_the_rain_dead_against_its_own_control() {
    let mut control = rainy_world(0.0);
    run_free(&mut control, 48);
    let control_last_day = run_free(&mut control, 24);

    let mut sim = rainy_world(1.0);
    sim.set_weather_regime(false, 0.0);
    run_dry(&mut sim, 48);
    let regime_last_day = run_dry(&mut sim, 24);

    assert!(
        control_last_day > 0.0,
        "the control stopped raining on its own: nothing to switch off"
    );
    assert!(
        regime_last_day < DRY_DAY_RAIN_CEILING_MM,
        "day 3 of a dry episode still produced {regime_last_day} mm of rain map-wide \
         (control over the same 24 h: {control_last_day} mm) — a drizzle is \
         surviving the episode again"
    );
}
