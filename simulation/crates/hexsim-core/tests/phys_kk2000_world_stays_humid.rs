//! KK2000 test #6 (reverse safeguard): the world doesn't become a global
//! desert because KK2000 would hold back too much rain.
//!
//! Risk: if the formula is too conservative, clouds travel
//! but never rain. Very pretty visually, but climatologically
//! absurd.
//!
//! Setup :
//! - radius 10 with generated terrain (mountains, plains, normal
//!   topographic gradient).
//! - 365 days = 8760 ticks.
//!
//! Assertion: **total rainfall** over the grid (sum of all
//! cells) over the year > 5000 mm. Threshold choice: we don't use
//! "% of cells > X mm" because rain is naturally concentrated
//! on a few cells (orography); a test on the fraction
//! would penalize a realistic climate with mountain/plain contrast. The
//! total water mass, on the other hand, is robust: if KK2000 is too
//! conservative, the total collapses.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

#[test]
fn world_does_not_become_a_desert() {
    let mut grid = HexGrid::from_radius(10);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: 42,
            ..TerrainParams::default()
        },
    );
    let n = grid.len();

    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );

    let mut yearly_precip: Vec<f32> = vec![0.0; n];

    // Simulation::step() advances 1 day (24 Tier 1 sub-ticks) and exposes
    // last_precipitation accumulated over these 24 ticks. After 365 days
    // we have the annual total per cell.
    for _ in 0..365 {
        sim.step();
        for (cumul, day) in yearly_precip.iter_mut().zip(sim.last_precipitation()) {
            *cumul += day.rain + day.snow;
        }
    }

    let total: f32 = yearly_precip.iter().sum();

    if std::env::var("KK2000_DEBUG").is_ok() {
        let mut sorted = yearly_precip.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("yearly_precip top 5: {:?}", &sorted[..5.min(sorted.len())]);
        eprintln!("total {total:.0} mm median {:.1} mm", sorted[n / 2]);
    }

    assert!(
        total > 5000.0,
        "World too dry: total annual rainfall = {total:.0} mm across the grid \
         (expected > 5000). KK2000 too conservative, either the main \
         coefficient is too low, `N_c` too high, or condensation \
         can't keep pace.",
    );
}
