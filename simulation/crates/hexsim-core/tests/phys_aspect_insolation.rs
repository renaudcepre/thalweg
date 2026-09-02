//! Micro e2e #102: at equal altitude, a south-facing slope (sunny
//! slope / adret) receives more sun and warms more than a
//! north-facing slope (shaded slope / ubac).
//!
//! Replaces `scale_aspect_insolation` (r24, procedural terrain,
//! statistical binning per elevation band over 1 year): the property
//! is purely local, we don't need a whole world to prove it. Here a
//! BUILT east-west ridge gives two twin cells (same altitude, opposite
//! normals computed by the real `compute_surface_normals`), and we
//! read the climate produced by the full pipeline (illumination ->
//! energy balance -> climate normals). A red points to the faulty
//! cell, not "somewhere in a 1729-cell world".

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 6;
/// Ridge centered on `r = 0`, sloping down on both sides: `z = PEAK −
/// SLOPE·|r|`. World coordinate `y` grows with `r` (south), so cells
/// at `r > 0` face south (sunny slope / adret) and those at `r < 0`
/// face north (shaded slope / ubac). The two slopes at equal `|r|` are
/// at the SAME altitude: lapse rate is neutralized by construction,
/// not by a statistical filter.
const PEAK_M: f32 = 500.0;
const SLOPE_M_PER_R: f32 = 60.0;
/// Sampled slope rank (far enough from the `r=0` ridge and the
/// `r=±6`/toric seam edge for a clean normal).
const FLANK: i32 = 3;

fn ridge_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        let z = PEAK_M - SLOPE_M_PER_R * f32::from(i16::try_from(coord.r.abs()).unwrap_or(0));
        grid.get_mut(coord).unwrap().elevation = z;
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

#[test]
fn adret_is_sunnier_and_warmer_than_ubac() {
    let mut sim = ridge_sim();
    let adret = HexCoord::new(0, FLANK); // sud
    let ubac = HexCoord::new(0, -FLANK); // nord
    let ai = sim.grid().index_of(adret).unwrap();
    let ui = sim.grid().index_of(ubac).unwrap();

    // Geometry guard: the ridge did produce two opposite slopes at the
    // same altitude (otherwise the rest of the test wouldn't measure
    // what we think it does).
    let cells = sim.grid().cells_slice();
    assert!(
        (cells[ai].elevation - cells[ui].elevation).abs() < 1e-3,
        "sunny slope and shaded slope must be at equal elevation: {} vs {}",
        cells[ai].elevation,
        cells[ui].elevation
    );
    assert!(
        cells[ai].normal_north < -0.05,
        "sunny slope (r>0) must face south (normal_north < 0), got {}",
        cells[ai].normal_north
    );
    assert!(
        cells[ui].normal_north > 0.05,
        "shaded slope (r<0) must face north (normal_north > 0), got {}",
        cells[ui].normal_north
    );

    // One year: year-0 normals frozen (insolation nearly insensitive
    // to the ~30-tick thermal transient).
    for _ in 0..=365 {
        sim.step();
    }
    let n = sim.climate_normals();

    // Primary assertion: absorbed INSOLATION, upstream of any advective
    // damping; this is the quantity that aspect drives directly (cos
    // of incidence angle on the surface normal).
    assert!(
        n[ai].insolation_mean > n[ui].insolation_mean,
        "sunny slope must be sunnier: {} vs {} W/m²",
        n[ai].insolation_mean,
        n[ui].insolation_mean
    );
    // Secondary, softer assertion: the energy surplus translates into
    // mean temperature (advection smooths out part of the contrast).
    assert!(
        n[ai].t_mean >= n[ui].t_mean,
        "sunny slope must not be colder than the shaded slope: {} vs {} °C",
        n[ai].t_mean,
        n[ui].t_mean
    );
}
