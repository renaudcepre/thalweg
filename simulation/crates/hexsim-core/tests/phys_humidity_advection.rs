//! e2e-unit micro-tests: wind-driven atmospheric advection.
//!
//! Pins the elementary brick: "`humidity_upper` (the advected reservoir)
//! travels IN the direction of the wind, never upwind, transport
//! conserves mass, and it really is the wind that causes it".
//!
//! Does not duplicate: `phys_kk2000_cloud_travels.rs` (derived from
//! `cloud_water`, isolated pulse on radius 8 / 60h, a different reservoir,
//! already covered); `phys_kk2000_cloud_persists.rs` (lifetime, not
//! direction); `synoptic_wind_integration.rs` (the full synoptic wind field,
//! not the transport it drives); `uplift_conserves_total_humidity` in
//! `atmosphere.rs` (VERTICAL intra-cell transfer surface->upper, no
//! horizontal transport). Here: `humidity_upper` alone, small grid (radius
//! 5), short horizon (1 day) to stay far from toroidal wraparound (the
//! advection-specific pitfall: too small a torus homogenizes before the
//! upwind/downwind gap has any meaning).
//!
//! Scripted, deterministic wind: `build_sim` forces a UNIFORM surface field
//! via the `Simulation::set_uniform_wind` seam (#108, mapping the old
//! `west_bias = v` to `WindVec { x: -v, y: 0.0 }`, wind toward -x), which
//! also disables synoptic dynamics at the same time. The "downwind"
//! direction stays determined EMPIRICALLY, never assumed hardcoded
//! "East"/"West": the real upper-level wind is derived via
//! `compute_upper_wind_field` (the same function the engine uses internally
//! to advect `humidity_upper`, Ekman rotation + speed ratio), then the
//! "downwind" cell is the one whose hex direction maximizes the dot product
//! with that wind.

mod common;

use hexsim_core::atmosphere::{AtmosphereParams, total_humidity};
use hexsim_core::cell::CellProperties;
use hexsim_core::coord::{DIRECTIONS, HexCoord, hex_direction_to_world};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::{WindField, WindParams, WindVec, compute_upper_wind_field};

const RADIUS: i32 = 5;
const SOURCE: HexCoord = HexCoord { q: 0, r: 0 };
/// mm of initial upper-level vapor, concentrated on `SOURCE`. Stays under
/// saturation even fully concentrated on a single cell: at 15°C surface
/// temperature, `t_upper = 15 - lapse_rate(6.5) * 1.5 = 5.25°C`, and
/// `saturation_upper(5.25°C, H=1500m)` ≈ 10 mm (cf `saturation_upper`
/// doc-comment: 7.3 mm at 0°C, 19.3 mm at 15°C). 5.0 mm stays under this
/// threshold: no condensation should trigger, otherwise the test would be
/// measuring KK2000 microphysics instead of advection alone.
const SOURCE_HUMIDITY: f32 = 5.0;

/// Flat grid, temperature frozen at 15°C everywhere (`thermal_coupling: 0.0`
/// freezes temperature at its initial value, cf `temperature.rs`
/// `thermal_coupling_zero_freezes_temperature`): eliminates all thermal
/// dynamics (breeze, diurnal cycle) that could interfere with the sole
/// advection we want to isolate. Flat terrain => orographic uplift and
/// wind deflection by relief are no-ops by construction (zero elevation
/// gradient), no need to disable them explicitly.
fn build_flat_grid() -> HexGrid {
    let mut grid = HexGrid::from_radius(RADIUS);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            *cell = CellProperties {
                elevation: 200.0,
                temperature: 15.0,
                ..CellProperties::default()
            };
        }
    }
    if let Some(cell) = grid.get_mut(SOURCE) {
        cell.humidity_upper = SOURCE_HUMIDITY;
    }
    grid
}

fn temp_params_frozen() -> TemperatureParams {
    TemperatureParams {
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    }
}

/// "Clean" scripted wind: noise/thermal/relief cut off so the wind field
/// applied via `Simulation::set_uniform_wind` (cf `build_sim`) is the sole
/// source of air movement over the whole grid (same pattern as
/// `phys_kk2000_cloud_travels.rs`). A zero vector (`west_bias` equivalent to
/// 0, cf `build_sim`) gives a strictly zero wind everywhere (used for the
/// ablation control, test 3).
fn wind_params_scripted() -> WindParams {
    WindParams {
        noise_direction_amplitude: 0.0,
        noise_strength_amplitude: 0.0,
        thermal_strength: 0.0,
        terrain_deflection: 0.0,
        ..WindParams::default()
    }
}

fn atmo_params_no_floor() -> AtmosphereParams {
    // initial_humidity_floor at 0: without this, `Simulation::new` raises
    // humidity_upper of ALL cells to 10 mm (closed terrarium floor),
    // drowning out the isolated source we want to track.
    AtmosphereParams {
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    }
}

/// `west_bias`: magnitude of the old West bias, converted here into a
/// uniform wind vector via the `Simulation::set_uniform_wind` seam (mapping
/// #108: `west_bias = v` becomes `WindVec { x: -v, y: 0.0 }`, wind toward
/// -x).
fn build_sim(west_bias: f32) -> Simulation {
    let mut sim = Simulation::new(
        build_flat_grid(),
        HydroParams::default(),
        atmo_params_no_floor(),
        GroundwaterParams::default(),
        SnowParams::default(),
        temp_params_frozen(),
        wind_params_scripted(),
    );
    // `set_uniform_wind` forces a UNIFORM surface wind field and
    // automatically disables synoptic dynamics (active BY DEFAULT
    // otherwise, cf `simulation.rs`), no need for a separate `update_param`.
    // Isolates the sole brick under test: advection of a scripted,
    // deterministic wind field. Without this call, the "zero wind" of the
    // ablation control (test 3) would not really be zero (synoptic would
    // take back over).
    sim.set_uniform_wind(WindVec {
        x: -west_bias,
        y: 0.0,
    });
    sim
}

/// Hex direction index (0..6, cf `DIRECTIONS`/`hex_direction_to_world`)
/// that maximizes the dot product with `wind`, the "downwind" direction
/// measured on the real field, never assumed.
fn downwind_direction_index(wind: WindVec) -> usize {
    (0..6_usize)
        .max_by(|&a, &b| {
            let (ax, ay) = hex_direction_to_world(a);
            let (bx, by) = hex_direction_to_world(b);
            let dot_a = wind.x * ax + wind.y * ay;
            let dot_b = wind.x * bx + wind.y * by;
            dot_a.total_cmp(&dot_b)
        })
        .expect("6 directions, never empty")
}

fn humidity_upper_at(sim: &Simulation, coord: HexCoord) -> f32 {
    sim.grid().get(coord).unwrap().humidity_upper
}

// --- Test 1: it travels WITH the wind, never against it -----------------

#[test]
fn humidity_upper_travels_downwind_not_upwind() {
    const WEST_BIAS: f32 = 0.5;
    let wind_params = wind_params_scripted();
    let mut sim = build_sim(WEST_BIAS);

    // The surface wind no longer needs to be "measured" empirically via
    // `compute_wind_field`: `set_uniform_wind` (cf `build_sim`) forces it to
    // EXACTLY the uniform vector set, over the whole grid. We rebuild this
    // uniform field then derive the REAL upper-level wind that transports
    // `humidity_upper` via `compute_upper_wind_field` (Ekman rotation, scaled
    // by `wind_upper_speed_ratio`), the same function the engine uses
    // internally. The "downwind" direction stays determined EMPIRICALLY
    // (max dot product over the 6 hex directions), never assumed hardcoded
    // "West".
    let uniform_wind = WindVec {
        x: -WEST_BIAS,
        y: 0.0,
    };
    let surface_field: WindField = vec![uniform_wind; sim.grid().len()];
    let wind_upper = compute_upper_wind_field(&surface_field, &wind_params);
    let source_idx = sim.grid().cell_index(SOURCE).unwrap();
    let measured = wind_upper[source_idx];
    assert!(
        measured.magnitude() > 0.05,
        "invalid setup: near-zero upper-level wind, nothing to transport ({measured:?})"
    );

    let downwind_idx = downwind_direction_index(measured);
    let upwind_idx = (downwind_idx + 3) % 6;
    let downwind_coord = SOURCE + DIRECTIONS[downwind_idx];
    let upwind_coord = SOURCE + DIRECTIONS[upwind_idx];

    sim.step(); // 1 day, cf module doc-comment (short horizon, radius 5)

    let downwind_after = humidity_upper_at(&sim, downwind_coord);
    let upwind_after = humidity_upper_at(&sim, upwind_coord);
    let source_after = humidity_upper_at(&sim, SOURCE);

    assert!(
        source_after < SOURCE_HUMIDITY,
        "the source did not lose humidity, no transport took place: {source_after}"
    );
    assert!(
        downwind_after > 1e-3,
        "the downwind cell ({downwind_coord:?}) received almost nothing: {downwind_after}"
    );
    assert!(
        upwind_after < 1e-6,
        "the upwind cell ({upwind_coord:?}) received humidity against the wind: {upwind_after}"
    );
    assert!(
        downwind_after > upwind_after,
        "downwind ({downwind_after}) should have gained strictly more than upwind \
         ({upwind_after})"
    );

    // Sanity check: microphysics did not interfere (cf SOURCE_HUMIDITY
    // doc-comment), otherwise this test would be measuring KK2000, not
    // advection.
    let cloud_total: f32 = sim.grid().cells_slice().iter().map(|c| c.cloud_water).sum();
    assert!(
        cloud_total < 1e-6,
        "condensation was triggered (cloud_total={cloud_total}): SOURCE_HUMIDITY \
         calibration to revisit, this test must isolate pure advection"
    );
}

// --- Test 2: advection creates no mass, destroys no mass -----------------

#[test]
fn humidity_advection_conserves_total_mass_while_the_field_moves() {
    let mut sim = build_sim(0.5);
    let before = total_humidity(sim.grid());
    assert!(before > 1.0, "invalid setup, no initial humidity: {before}");

    sim.step(); // 1 day: the field has time to move (cf test 1)

    // Sanity check: the field really did move (otherwise conservation
    // would be trivial, the test would prove nothing).
    let source_after = humidity_upper_at(&sim, SOURCE);
    assert!(
        source_after < SOURCE_HUMIDITY * 0.99,
        "the source did not transport humidity, test is void: {source_after}"
    );

    let after = total_humidity(sim.grid());
    let drift = (after - before).abs();
    let relative = drift / before;
    assert!(
        relative < 1e-4,
        "conservation violated during wind transport: {before:.6} -> {after:.6} \
         (drift {drift:.6}, {:.6} %)",
        relative * 100.0
    );
}

// --- Test 3: no wind, no transport (ablation) -----------------------------

#[test]
fn no_wind_no_advection_source_stays_put() {
    // Same grid, same duration: only the wind changes. Confirms that it
    // really is THE WIND that causes the transport observed in test 1, not
    // some other mechanism (diffusion, uplift, orographic convection...)
    // that could have produced a similar result by accident.
    let mut sim_calm = build_sim(0.0);
    let mut sim_windy = build_sim(0.5);

    for _ in 0..24 {
        sim_calm.step_hour();
        sim_windy.step_hour();
    }

    // Calm grid: `advect_humidity_layer_into` only acts if the total
    // directional weight (dot product with the wind) exceeds 1e-6. Zero
    // wind => literal no-op on every cell, the source keeps all its mass
    // and no neighbor receives even a crumb.
    let source_calm = humidity_upper_at(&sim_calm, SOURCE);
    assert!(
        (source_calm - SOURCE_HUMIDITY).abs() < 1e-6,
        "with no wind, the source should not lose humidity: {source_calm}"
    );
    let neighbors_calm_total: f32 = SOURCE
        .neighbors()
        .iter()
        .map(|&c| humidity_upper_at(&sim_calm, c))
        .sum();
    assert!(
        neighbors_calm_total < 1e-6,
        "with no wind, neighbors should receive no humidity: {neighbors_calm_total}"
    );

    // Same setup, wind active: the source has measurably lost humidity
    // over the same horizon.
    let source_windy = humidity_upper_at(&sim_windy, SOURCE);
    assert!(
        source_windy < SOURCE_HUMIDITY - 1e-3,
        "with wind, the source should have lost humidity (positive control): {source_windy}"
    );
}
