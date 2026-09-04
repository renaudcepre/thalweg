//! Micro e2e (2026-09-03): `base_temp` means what it says on the REAL
//! terrain, not on an assumed flat one.
//!
//! `calibration_offset` sets the structural thermal offset so the
//! annual mean of dry flat ground equals `base_temp`. Since
//! `CELL_SPACING_M` went to 130 m (d6be105) the procedural relief is
//! steep (mean slope ~29° at r30 seed 42) and the illumination pass
//! (aspect × relief occlusion × diffuse sky) only lets ~0.856 of the
//! flat beam through on average (`tests/diag_illumination_budget.rs`,
//! clouds = 0): a flat-world calibration ran the whole map ~4 K below
//! `base_temp` (JOURNAL 2026-09-02/03). `terrain_insolation_factor`
//! (`TemperatureParams`, computed by
//! `terrain::terrain_annual_mean_insolation_factor` in
//! `Simulation::new`) folds the REAL deficit into the offset.
//!
//! Built WASHBOARD relief: adjacent rows (`r` even/odd) alternate
//! `±AMPLITUDE_M`, short parallel ridges (row-flat, jump at every row
//! boundary) rather than one continuous tilt, then numerically
//! zero-centered — this fixture's map-mean elevation is (checked) 0, so
//! the lapse rate contributes nothing to the map mean and the
//! comparison below isolates the insolation term `calibration_offset`
//! actually targets. 100 m of row-to-row jump (slope ≈ 37° at
//! `CELL_SPACING_M` = 130 m, close to the procedural map's own ~29°
//! mean, JOURNAL 2026-09-02) so the relief genuinely self-shadows. 1
//! year warm-up, 1 year measuring the map-mean surface temperature
//! every hour.
//!
//! Bone dry on purpose (`water_level = 0`, `initial_humidity_floor =
//! 0`, `mean_cloud_cover_for_calibration = 0.0`): clouds are a SEPARATE
//! calibration axis (issue #44, `TemperatureParams::mean_cloud_cover_for_calibration`
//! doc) this fixture isn't built to control for — without this the init
//! humidity floor (10 mm PW by default, unconditional, see
//! `phys_dry_land_no_evap.rs`) condenses an uncontrolled amount of cloud
//! over a year and its back-radiation boost (`ATMO_IR_BACK_CLOUDY_BOOST`)
//! confounds the measurement.
//!
//! Three prior fixtures were diagnosed and discarded, not patched into
//! the tolerance (JOURNAL 2026-09-03): a `|r|`-ridge with default cloud
//! calibration measured 4.16 °C below `base_temp` (the actual simulated
//! cloud cover isn't the assumed 0.5, issue #44); the same ridge with
//! clouds pinned to 0 still measured 2.12 °C below the LAPSE-ADJUSTED
//! target (a ridge's map-mean elevation is never 0: `mean(|r|) > 0`
//! whatever the sign); a monotonic signed-`r` tilt (one direction for
//! the WHOLE domain, chosen to zero the mean cheaply) measured +10.6 °C
//! — `aspect_insolation_correction`'s pure-tilt approximation breaks on
//! a single map-wide gradient, a case no real (locally-varied) relief
//! produces. The washboard's short, alternating-direction ridges avoid
//! that failure mode.

use hexsim_core::atmosphere::{AtmosphereParams, surface_means};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::time::TICKS_PER_YEAR;
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 6;
/// Half-amplitude of the row-to-row elevation jump: adjacent rows (`r`
/// even vs odd) alternate `+AMPLITUDE_M`/`-AMPLITUDE_M`, a washboard of
/// short parallel ridges (row-flat, jump at every row boundary) rather
/// than one continuous tilt: real terrain (and `aspect_insolation_correction`,
/// which this fixture also exercises) is built on LOCAL, alternating-
/// aspect slopes, not a single monotonic half-map gradient. A first
/// attempt with a monotonic signed tilt (one direction for the whole
/// domain) measured +10.6 °C: `aspect_correction`'s uniform-direction
/// worst case, not representative of any real relief and not what this
/// fixture means to exercise (JOURNAL 2026-09-03).
const AMPLITUDE_M: f32 = 100.0;
/// Tolerance on the map-mean elevation: the raw washboard pattern's
/// mean isn't exactly 0 (odd row count around `r = 0`), so it's
/// zero-centered numerically below; this only guards that the
/// zero-centering itself worked.
const MAX_MEAN_Z_M: f32 = 1e-3;
/// Tolerance on the annual map-mean surface temperature vs `base_temp`.
/// A genuine physical margin (diurnal thermal inertia, the discrete
/// 8760-sample annual mean, the sensible-exchange redistribution), not
/// slack for a known residual bias: a red here is a real miscalibration.
const MAX_DEVIATION_C: f32 = 1.0;

fn steep_slope_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let coords: Vec<_> = grid.coords().copied().collect();
    for &coord in &coords {
        let z = if coord.r.rem_euclid(2) == 0 {
            AMPLITUDE_M
        } else {
            -AMPLITUDE_M
        };
        grid.get_mut(coord).unwrap().elevation = z;
    }
    // Zero-center: the washboard's raw mean isn't exactly 0 (odd count
    // of `+`/`-` rows around `r = 0`), and any residual here would
    // reintroduce the very lapse-rate confound this fixture is built to
    // avoid (see the module doc).
    let mean_z = coords
        .iter()
        .map(|&c| grid.get(c).unwrap().elevation)
        .sum::<f32>()
        / f32::from(u16::try_from(coords.len()).expect("small test grid fits u16"));
    for &coord in &coords {
        grid.get_mut(coord).unwrap().elevation -= mean_z;
    }
    let atmosphere = AtmosphereParams {
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    };
    let temperature = TemperatureParams {
        mean_cloud_cover_for_calibration: 0.0,
        ..TemperatureParams::default()
    };
    Simulation::new(
        grid,
        HydroParams::default(),
        atmosphere,
        GroundwaterParams::default(),
        SnowParams::default(),
        temperature,
        WindParams::default(),
    )
}

#[test]
fn base_temp_holds_on_steep_terrain() {
    let mut sim = steep_slope_sim();
    // Construction guard: the lapse rate must contribute exactly 0 to
    // the map mean, or the assertion below wouldn't isolate the
    // insolation term.
    let (_, mean_z_check) = surface_means(sim.grid());
    assert!(
        mean_z_check.abs() < MAX_MEAN_Z_M,
        "fixture must have zero map-mean elevation, got {mean_z_check} m"
    );

    // 1 year warm-up: dry-soil thermal time constant is ~18h, but the
    // lapse-rate spread across the slope needs a full annual cycle to
    // settle into its stationary regime.
    for _ in 0..TICKS_PER_YEAR {
        sim.step_hour();
    }
    // 1 year measuring the annual map-mean surface temperature.
    let mut sum: f32 = 0.0;
    for _ in 0..TICKS_PER_YEAR {
        sim.step_hour();
        let (mean_t, _mean_z) = surface_means(sim.grid());
        sum += mean_t;
    }
    let measured = sum / f32::from(u16::try_from(TICKS_PER_YEAR).expect("365x24 fits u16"));
    let base_temp = sim.temperature_params().base_temp;
    let factor = sim.temperature_params().terrain_insolation_factor;
    eprintln!(
        "phys_terrain_calibration: measured annual map-mean = {measured:.2} °C, \
         base_temp = {base_temp:.2} °C, terrain_insolation_factor = {factor:.3}"
    );
    assert!(
        (measured - base_temp).abs() < MAX_DEVIATION_C,
        "annual map-mean temperature must land within {MAX_DEVIATION_C} °C of base_temp \
         on steep terrain: measured {measured:.2} °C vs target {base_temp:.2} °C \
         (terrain_insolation_factor={factor:.3})"
    );
}
