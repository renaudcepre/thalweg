//! Shared fixtures between the `atmosphere/*` test modules. Avoids
//! duplicating the same small worlds / default params in each
//! sub-module (`condensation`, `uplift`, `advection`, `fog`, `mod.rs`).

use crate::coord::HexCoord;
use crate::grid::HexGrid;
use crate::hydro::total_water;
use crate::temperature::TemperatureParams;
use crate::wind::{WindField, WindParams, WindVec, compute_wind_magnitudes_into};

use super::{AtmosphereParams, saturation_upper, total_humidity};

pub(crate) fn make_wet_grid() -> HexGrid {
    let mut grid = HexGrid::from_radius(3);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            cell.water_level = 5.0;
            cell.humidity_upper = 40.0;
        }
    }
    grid
}

pub(crate) fn total_moisture(grid: &HexGrid) -> f32 {
    total_water(grid) + total_humidity(grid)
}

pub(crate) fn zero_wind(grid: &HexGrid) -> WindField {
    vec![WindVec::default(); grid.len()]
}

/// Wind field magnitudes (#89), computed on demand for tests that
/// build an `AtmoForcing` by hand (the production caller memoizes
/// these magnitudes, cf `Simulation::wind_mag`).
pub(crate) fn wind_mags(wf: &WindField) -> Vec<f32> {
    let mut out = Vec::with_capacity(wf.len());
    compute_wind_magnitudes_into(wf, &mut out);
    out
}

pub(crate) fn default_wind_params() -> WindParams {
    WindParams::default()
}

pub(crate) fn default_temp_params() -> TemperatureParams {
    TemperatureParams::default()
}

/// Test harness for the orographic pump micro-tests: flat radius-2
/// grid (100 m, 15 °C, dry), only the center cell receives
/// `humidity_surface`. Each test then sculpts the elevation of the
/// neighbors it targets. Radius 2 (not 0-1): the pump is a transport
/// between neighbors, and on the torus a radius-0 cell is its own neighbor
/// ×6, which would make the transport a silent self-transfer.
pub(crate) fn oro_pump_world(center_humidity: f32) -> (HexGrid, Vec<HexCoord>) {
    let mut grid = HexGrid::from_radius(2);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for &c in &coords {
        if let Some(cell) = grid.get_mut(c) {
            cell.elevation = 100.0;
            cell.temperature = 15.0;
            cell.humidity_surface = 0.0;
            cell.humidity_upper = 0.0;
        }
    }
    grid.get_mut(HexCoord::new(0, 0)).unwrap().humidity_surface = center_humidity;
    (grid, coords)
}

/// Shared calibration guard: the upper saturation deficit at the
/// summit must clearly dominate the test's transfer, otherwise the LCL
/// bound (tested separately by `phys_oro_lift_bounded_by_saturation`)
/// kicks in and skews the property measured here. Failure = setup to
/// revisit, not physics broken.
pub(crate) fn assert_lcl_slack(max_transfer_mm: f32) {
    let params = AtmosphereParams::default();
    let temp_params = default_temp_params();
    let t_offset = temp_params.lapse_rate * params.upper_layer_altitude_m / 1000.0;
    let sat = saturation_upper(15.0 - t_offset, &params);
    assert!(
        sat > 2.0 * max_transfer_mm,
        "setup: saturation upper ({sat:.1} mm) too close to transfer \
         ({max_transfer_mm} mm), the LCL bound would contaminate the test"
    );
}
