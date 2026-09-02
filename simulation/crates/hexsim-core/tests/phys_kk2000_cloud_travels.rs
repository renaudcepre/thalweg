//! KK2000 test #3 (the motivating test): a cloud formed by condensation
//! of a humidity pulse *must* drift in the wind direction before
//! raining, not fall right back on the source.
//!
//! This is *exactly* the visual bug reported by the user: "the clouds
//! don't move!". Root cause: with the current linear drain, `cloud_water`
//! precipitates as soon as it exceeds 0.05 mm, so the cloud dies before
//! it has traveled. Super-linear KK2000 lets the cloud grow without
//! raining while it's small, so the wind has time to carry it away.
//!
//! Setup:
//! - radius 8, flat terrain 200 m, T = 15 °C.
//! - Unidirectional wind toward the West forced via the
//!   `Simulation::set_uniform_wind` seam (#108: mapping `west_bias = 0.5`
//!   to `WindVec { x: -0.5, y: 0.0 }` ≈ 5 m/s, thermal/terrain/noise
//!   disabled in `WindParams`, `initial_humidity_floor` = 0 so as not to
//!   saturate the grid with uniform humidity). `set_uniform_wind` also
//!   disables synoptic dynamics along the way: before #108, this test did
//!   NOT do so explicitly (latent bug: the synoptic field, active by
//!   default, silently masked `west_bias` with a seed-dependent
//!   geostrophic wind), now fixed automatically by the seam.
//! - Initial pulse: `cloud_water` = 1.0 mm directly on (q=8, r=0)
//!   (East edge). No `humidity_upper`, no persistent source: the cloud
//!   is isolated, must drift and eventually precipitate.
//!
//! Assertion: after 60h, the **leading edge** of the cloud (the
//! westernmost cell with `cloud_water` > 0.001) must be at least 3 cells
//! West of the source (q ≤ 5). We don't look at the peak: the source
//! always dominates in mass for a finite pulse, but the advected matter
//! must have traveled some distance.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::{WindParams, WindVec};

const SOURCE: HexCoord = HexCoord { q: 8, r: 0 };
/// Magnitude of the former `west_bias` (cf. mapping #108, applied via
/// `Simulation::set_uniform_wind` right after `sim` is built).
const WEST_BIAS: f32 = 0.5;

#[test]
fn cloud_drifts_downwind_before_raining() {
    let mut grid = HexGrid::from_radius(8);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            cell.temperature = 15.0;
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
            cell.humidity_surface = 0.0;
            cell.humidity_upper = 0.0;
            cell.cloud_water = 0.0;
        }
    }
    if let Some(cell) = grid.get_mut(SOURCE) {
        cell.cloud_water = 1.0;
    }

    let atmo = AtmosphereParams {
        // No humidity floor: we want an isolated cloud without
        // distributed condensation that would mask the drift.
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    };
    let wind = WindParams {
        noise_direction_amplitude: 0.0,
        noise_strength_amplitude: 0.0,
        thermal_strength: 0.0,
        terrain_deflection: 0.0,
        ..WindParams::default()
    };

    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind,
    );
    // Uniform wind forced toward the West + automatic disabling of the
    // synoptic dynamics (cf. module doc-comment, mapping #108).
    sim.set_uniform_wind(WindVec {
        x: -WEST_BIAS,
        y: 0.0,
    });

    for _ in 0..60 {
        sim.step_hour();
    }

    // Leading edge of the cloud = westernmost cell (minimum q) with
    // detectable cloud_water. Excludes toroidal wraps by filtering on q
    // near the opposite edge: we only consider q >= -1 (the source is at
    // 8, we don't want to confuse it with a cloud that wrapped around the
    // torus).
    let coords_slice = sim.grid().coords_slice();
    let cells = sim.grid().cells_slice();
    let leading_q = cells
        .iter()
        .enumerate()
        .filter(|(i, c)| c.cloud_water > 0.001 && coords_slice[*i].q >= -1)
        .map(|(i, _)| coords_slice[i].q)
        .min()
        .unwrap_or(SOURCE.q);
    let drift_west = SOURCE.q - leading_q;

    assert!(
        drift_west >= 3,
        "Cloud leading edge did not travel any distance: leading_q = \
         {leading_q} (source q={}), west drift = {drift_west} cells \
         (expected >= 3). KK2000 expects the cloud to grow before raining, \
         wind carries matter far enough before it vanishes.",
        SOURCE.q,
    );
}
