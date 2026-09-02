//! Integration tests for climate history.
//!
//! Validates end-to-end pipeline: precipitation -> ring buffer -> aggregates
//! by altitude band. Includes an `#[ignore]` test documenting the target
//! improvement "mid-altitude must not be a desert" to enable when moist
//! convection is introduced.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::climate::{Window, default_bands};
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

fn build_sim(radius: i32, seed: u32) -> Simulation {
    let terrain = TerrainParams {
        seed,
        ..TerrainParams::default()
    };
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(&mut grid, &terrain);

    let wind = WindParams {
        seed,
        humidity_advection_rate: 0.20,
        ..WindParams::default()
    };
    let atmo = AtmosphereParams::default();
    Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind,
    )
}

/// After N ticks, the ring buffer must have accumulated N days (or 365 cap).
#[test]
fn ring_buffer_fills_over_ticks() {
    let mut sim = build_sim(3, 42);

    for _ in 0..50 {
        sim.step();
    }
    let history = sim.climate_history();
    let any_coord = HexCoord::new(0, 0);
    assert_eq!(
        history.days_recorded(any_coord),
        50,
        "after 50 ticks, buffer must contain 50 days",
    );

    for _ in 0..400 {
        sim.step();
    }
    let history = sim.climate_history();
    assert_eq!(
        history.days_recorded(any_coord),
        365,
        "after 450 ticks, buffer is capped at 365 days",
    );
}

/// After one year, at least one cell must have received rain. Otherwise,
/// either the module records nothing, or the simulation never rains.
#[test]
fn at_least_one_cell_receives_precipitation_after_a_year() {
    let mut sim = build_sim(8, 42);

    for _ in 0..365 {
        sim.step();
    }

    let bands = default_bands();
    let stats = sim
        .climate_history()
        .aggregate(sim.grid(), &bands, Window::Last365);

    let total_wet: u32 = stats.iter().map(|s| s.wet_cells).sum();
    assert!(
        total_wet > 0,
        "no cell received rain over 365 days: pipeline broken or sim desert",
    );
}

/// Coherence invariants on aggregates: `arid_cells` and others <= cells.
#[test]
fn aggregate_counts_are_consistent() {
    let mut sim = build_sim(5, 7);

    for _ in 0..100 {
        sim.step();
    }

    let bands = default_bands();
    for window in [Window::Last30, Window::Last180, Window::Last365] {
        let stats = sim.climate_history().aggregate(sim.grid(), &bands, window);
        for band in &stats {
            assert!(
                band.arid_cells <= band.cells,
                "arid_cells ({}) > cells ({}) in band {}",
                band.arid_cells,
                band.cells,
                band.name,
            );
            assert!(
                band.wet_cells <= band.cells,
                "wet_cells ({}) > cells ({}) in band {}",
                band.wet_cells,
                band.cells,
                band.name,
            );
        }
    }
}

/// Climatological invariant: mid-altitudes (100-800m) must not be desert.
/// Threshold: less than 50 % arid cells over 365 days.
///
/// Without moist convection, this test fails (~71 % arid measured). Convection
/// (summer storms in warm plains) precipitates below stable threshold when
/// temperature > 20°C and humidity > 0.10, wetting mid-altitudes that were
/// previously just atmospheric transport corridors.
#[test]
fn mid_altitude_should_not_be_desertic() {
    let mut sim = build_sim(25, 42);

    for _ in 0..365 {
        sim.step();
    }

    let bands = default_bands();
    let stats = sim
        .climate_history()
        .aggregate(sim.grid(), &bands, Window::Last365);

    let mid = stats
        .iter()
        .find(|s| s.name == "mid")
        .expect("band 'mid' must exist");

    assert!(mid.cells > 0, "no mid-altitude cell in this seed");

    #[allow(clippy::cast_precision_loss)]
    let arid_ratio = f64::from(mid.arid_cells) / f64::from(mid.cells);
    assert!(
        arid_ratio < 0.50,
        "mid-altitude arid_ratio = {arid_ratio:.2} (target < 0.50). {} cells out of {} are arid.",
        mid.arid_cells,
        mid.cells,
    );
}
