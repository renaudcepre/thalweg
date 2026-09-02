//! Guard rail: on perfectly dry soil (`groundwater = 0`), turning on
//! vegetation transpiration must add NOTHING compared to the disabled
//! case. Differential test to isolate transpiration's contribution from
//! other humidity sources (subsidence from upper, init floor, etc.) that
//! can inject vapor at the surface.
//!
//! Duplicated setup: two identical scenarios except for
//! `transpiration_coef` (0.0 vs the active default). On truly dry soil,
//! water stress is zero (`groundwater = 0`) AND biomass doesn't grow: the
//! average humidity gap between the two must be zero, up to floating-
//! point precision.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

fn build_dry_grid() -> HexGrid {
    let mut grid = HexGrid::from_radius(3);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            cell.temperature = 25.0;
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
        }
    }
    grid
}

fn run(atmo: AtmosphereParams) -> f32 {
    let mut sim = Simulation::new(
        build_dry_grid(),
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );
    for _ in 0..100 {
        sim.step();
    }
    sim.grid()
        .cells_slice()
        .iter()
        .map(|c| c.humidity_surface)
        .sum()
}

#[test]
fn dry_land_transpiration_adds_nothing() {
    let with_transp = run(AtmosphereParams::default());
    let without_transp = run(AtmosphereParams {
        transpiration_coef: 0.0,
        ..AtmosphereParams::default()
    });
    let delta = (with_transp - without_transp).abs();
    assert!(
        delta < 2e-3,
        "Transpiration added humidity on dry soil: total delta = \
         {delta:.6} mm (with {with_transp:.4}, without {without_transp:.4}). \
         A leak here means the pump is drawing from somewhere other than groundwater."
    );
}
