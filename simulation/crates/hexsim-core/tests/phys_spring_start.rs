//! Sentinel: a world can be born on another day than January 1st, with its
//! t0 computed for that day and its clock started there.
//!
//! Cell-local properties only (surface temperature at t0, the clock, the
//! primed climate normals), on a small relief.

mod common;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::time::TICKS_PER_DAY;
use hexsim_core::wind::WindParams;

const SEED: u32 = 42;
const RADIUS: i32 = 4;
const SPRING_EQUINOX: u16 = 79;

fn born_on(day: u16) -> Simulation {
    let terrain = TerrainParams {
        seed: SEED,
        start_day: day,
        ..TerrainParams::default()
    };
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(&mut grid, &terrain);
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed: SEED,
            ..WindParams::default()
        },
    );
    sim.set_start_day(terrain.start_day);
    sim
}

fn mean_temperature(sim: &Simulation) -> f32 {
    let cells = sim.grid().cells_slice();
    cells.iter().map(|c| c.temperature).sum::<f32>()
        / f32::from(u16::try_from(cells.len()).unwrap())
}

#[test]
fn a_spring_world_starts_on_its_day_and_warmer_than_a_january_one() {
    let january = born_on(0);
    let spring = born_on(SPRING_EQUINOX);
    assert_eq!(january.hour_tick(), 0);
    assert_eq!(
        spring.hour_tick(),
        u64::from(SPRING_EQUINOX) * TICKS_PER_DAY
    );
    let (tj, ts) = (mean_temperature(&january), mean_temperature(&spring));
    assert!(ts > tj + 3.0, "spring t0 {ts} vs january t0 {tj}");
}

#[test]
fn the_first_partial_year_does_not_replace_the_primed_normals() {
    let mut sim = born_on(SPRING_EQUINOX);
    let primed: Vec<_> = sim.climate_normals().to_vec();
    // Past the first rollover (hour_tick == TICKS_PER_YEAR): 286 days
    // recorded, a partial year.
    for _ in 0..(365 - u32::from(SPRING_EQUINOX) + 2) {
        sim.step();
    }
    assert!(sim.climate_normals_ready());
    assert!(
        sim.climate_normals()
            .iter()
            .zip(primed.iter())
            .all(|(a, b)| a == b),
        "a partial year must not replace the analytic normals"
    );
}
