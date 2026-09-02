//! **Dumb, simple physics test**: a lake must produce clouds
//! nearby. Not necessarily directly above it (depending on the current
//! uplift it can be around it), but in its immediate neighborhood.
//!
//! Minimal setup (radius 2, 19 cells):
//! - Uniform plain at 200m
//! - A lake at the center (q=0, r=0): `water_level=3.0`
//!
//! Expected property: after 100 ticks, at least 3 cells in the
//! grid (including at least one at distance ≤ 2 from the lake) must have
//! relative humidity > 70%, i.e. "a visible cloud".
//!
//! If the test fails, the lake isn't feeding the upper-altitude
//! humidity circulation: the main diagnostic for invisible clouds
//! above bodies of water.

use hexsim_core::atmosphere::{AtmosphereParams, saturation_upper};
use hexsim_core::coord::HexCoord;
use hexsim_core::dynamics::CELL_SPACING_M;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

// 300 ticks = 12.5 days. Since the SI energy budget #43, the dry plain
// cools sharply in winter (T_avg ~ -2°C at 44.5°N day 0) while
// the lake retains its initial T (τ ≈ 6 days for 60 cm). At 100 ticks the
// emitted vapor hadn't yet had time to envelop enough cells:
// 2/19 at RH > 0.7 instead of 3+. 300 ticks gives the humidity time
// to accumulate in humidity_upper even in winter conditions.
const TICKS: u64 = 300;
/// Visible cloud = relative humidity > 70%. RH threshold rather than
/// absolute because saturation depends strongly on `T_upper` in the new
/// model: a warm plain has saturation ~0.24, cold terrain ~0.09; the same
/// absolute threshold wouldn't measure the same thing.
const CLOUD_HR_THRESHOLD: f32 = 0.70;
const MIN_CLOUDY_CELLS: usize = 3;
/// Grid and neighborhood radius calibrated to the engine's original
/// spacing (2 cells ≈ 2.15 km). Rescaled by `CELL_SPACING_M` to
/// preserve the same REAL extent: otherwise (measured at 130 m) the
/// toroidal domain becomes so small (260 m) that the wind wraps around
/// the torus in a few ticks and homogenizes humidity instead of concentrating
/// it near the lake; the test no longer measures what it claims to measure.
const REFERENCE_CELL_SPACING_M: f32 = 1074.569;
const REFERENCE_RADIUS_CELLS: f32 = 2.0;

/// `as i32` is the only std path to round a bounded float (grid radius,
/// a few dozen even at very reduced `CELL_SPACING_M`) into an integer
/// (no `TryFrom<f32>` in std); isolated here, documented.
#[allow(clippy::cast_possible_truncation)]
fn scaled_radius() -> i32 {
    (REFERENCE_RADIUS_CELLS * REFERENCE_CELL_SPACING_M / CELL_SPACING_M).ceil() as i32
}

fn build_scene() -> Simulation {
    let mut grid = HexGrid::from_radius(scaled_radius());
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            cell.temperature = 15.0;
        }
    }
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        // Phase 3 (#32): rescale ×200 (3 → 600 mm = 60 cm depth).
        // water_level now interpreted as mm; a small, durable lake.
        cell.water_level = 600.0;
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

fn hex_distance(a: HexCoord, b: HexCoord) -> i32 {
    ((a.q - b.q).abs() + (a.r - b.r).abs() + (a.q + a.r - b.q - b.r).abs()) / 2
}

#[test]
fn lake_produces_clouds_in_its_neighborhood() {
    let mut sim = build_scene();
    for _ in 0..TICKS {
        sim.step();
    }
    let lake = HexCoord::new(0, 0);
    let neighborhood_radius = scaled_radius();
    let atmo = AtmosphereParams::default();
    let temp = TemperatureParams::default();
    let coords: Vec<HexCoord> = sim.grid().coords().copied().collect();
    let cloudy: Vec<(HexCoord, f32)> = coords
        .iter()
        .filter_map(|c| {
            sim.grid().get(*c).and_then(|cell| {
                let t_upper = cell.temperature - temp.lapse_rate * 1.5;
                let sat = saturation_upper(t_upper, &atmo);
                let hr = if sat > 0.0 {
                    cell.humidity_upper / sat
                } else {
                    0.0
                };
                if hr > CLOUD_HR_THRESHOLD {
                    Some((*c, hr))
                } else {
                    None
                }
            })
        })
        .collect();
    let nearby_cloudy = cloudy
        .iter()
        .filter(|(c, _)| hex_distance(*c, lake) <= neighborhood_radius)
        .count();

    eprintln!(
        "phys_lake_makes_clouds_nearby: after {TICKS} ticks, \
         {} cells with HR > {CLOUD_HR_THRESHOLD} ({} \
         within distance <= {neighborhood_radius} of the lake). Details: {:?}",
        cloudy.len(),
        nearby_cloudy,
        cloudy
            .iter()
            .take(5)
            .map(|(c, h)| format!("{c:?}=HR{h:.2}"))
            .collect::<Vec<_>>()
    );
    assert!(
        cloudy.len() >= MIN_CLOUDY_CELLS,
        "Not enough clouds: {}/{} cells with HR > {CLOUD_HR_THRESHOLD} \
         (min expected: {MIN_CLOUDY_CELLS}). The lake is not feeding the atmosphere.",
        cloudy.len(),
        sim.grid().len()
    );
    assert!(
        nearby_cloudy >= 1,
        "No cloud near the lake (distance <= {neighborhood_radius}). \
         Evaporated humidity travels too far before rising in altitude."
    );
}
