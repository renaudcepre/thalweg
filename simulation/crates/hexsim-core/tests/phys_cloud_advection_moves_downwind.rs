//! Micro e2e-unit test (#68): advection carries clouds *downwind*.
//!
//! Anti-regression sentinel for the parity fix on `cloud_advection_rate`
//! 0.37 → 3.0 (2026-07-12). Context: vapor had been bumped to 3.0
//! (drizzle-on-lakes, 2026-07-05) without the droplets following suit,
//! which crept along at 0.37, visually static clouds ("smoke columns",
//! issue #68). The exploratory diagnostic lives in
//! `diag_cloud_advection_lifetime.rs`; here we **pin** two properties:
//!
//!   1. **Directional**: the uniform wind vector (set via the seam
//!      `Simulation::set_uniform_wind`, #108) points toward `-x`
//!      (decreasing `world_x`); since advection follows the wind vector,
//!      the cloud moves off on that side, markedly more than toward
//!      `+x`. A strong asymmetry (`> 3×`) distinguishes wind-driven
//!      transport from plain symmetric diffusion and catches a sign
//!      flip in advection. (On the radius-5 torus, part wraps around to
//!      the `+x` side, hence we compare the two half-planes rather than
//!      requiring `+x` to be strictly zero.)
//!   2. **Anti-regression**: after 24 h of uniform strong wind, the
//!      source keeps very little of the pulse. Measured (scripted wind,
//!      synoptic OFF): 0.37 → **51%** remains, 3.0 → **0.3%**. Threshold
//!      at 25%: wide margin on both sides; a regression to 0.37 (or any
//!      notable weakening of advection) turns this test red.
//!
//! Isolation (cf `phys_humidity_advection.rs`): flat terrain radius 5
//! (transport ⇒ radius ≥ 2 mandatory: on the torus a radius-0 cell is its
//! own neighbor ×6), uniform scripted
//! wind set via `Simulation::set_uniform_wind` (noise/thermal/relief
//! disabled in `WindParams`); the seam automatically disables synoptic
//! dynamics (ON BY DEFAULT otherwise), which would silently replace the
//! scripted wind with its emergent, seed-dependent geostrophic field,
//! not the intended vector. `cloud_water` regeneration disabled
//! (condensation/evap/fog=0), KK2000 disabled
//! (`kk2000_droplet_count=0`) to isolate pure advection.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::{WindParams, WindVec};

const RADIUS: i32 = 5;
const SEED: u32 = 42;
const PULSE_MM: f32 = 5.0;
const HOURS: u64 = 24;

/// Magnitude of the old `west_bias`, converted to a uniform wind vector
/// via `Simulation::set_uniform_wind` (mapping #108: `west_bias = v` →
/// `WindVec { x: -v, y: 0.0 }`) ⇒ wind vector toward `-x` (~10 m/s,
/// `WindField` convention: magnitude × 10 = m/s). Advection follows the
/// wind vector, so clouds are carried downwind = decreasing `world_x`.
const WEST_BIAS: f32 = 1.0;

/// World x position of an axial cell: `x = q + r/2` (cf
/// `hex_direction_to_world`, East=(1,0), Southeast=(0.5, √3/2)).
fn world_x(c: HexCoord) -> f32 {
    f32::from(i16::try_from(c.q).unwrap_or(0)) + 0.5 * f32::from(i16::try_from(c.r).unwrap_or(0))
}

/// Atmosphere: no `cloud_water` regeneration, KK2000 disabled.
fn build_atmosphere() -> AtmosphereParams {
    AtmosphereParams {
        transpiration_coef: 0.0,
        sublimation_rate: 0.0,
        uplift_rate: 0.0,
        uplift_thermal_coef: 0.0,
        condensation_rate: 0.0,
        cloud_evap_rate: 0.0,
        fog_condensation_rate: 0.0,
        orographic_lift_coef: 0.0,
        convective_diurnal_coef: 0.0,
        initial_humidity_floor: 0.0,
        kk2000_droplet_count: 0.0,
        ..AtmosphereParams::default()
    }
}

/// Pure west wind: all field mechanisms (noise, thermal, relief
/// deflection, vapor/temp advection) neutralized; the uniform vector
/// itself is set afterward via `set_uniform_wind` (cf `build_sim`), not
/// via a `WindParams` field.
fn build_wind() -> WindParams {
    WindParams {
        seed: SEED,
        noise_direction_amplitude: 0.0,
        noise_strength_amplitude: 0.0,
        thermal_strength: 0.0,
        terrain_deflection: 0.0,
        humidity_advection_rate: 0.0,
        temperature_advection_rate: 0.0,
        ..WindParams::default()
    }
}

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.cloud_water = PULSE_MM;
    }
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        build_atmosphere(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        build_wind(),
    );
    // `set_uniform_wind` forces the surface wind field to the desired
    // uniform vector AND automatically disables synoptic dynamics
    // (cf phys_humidity_advection.rs, JOURNAL gotcha): ON BY DEFAULT
    // otherwise, it would SILENTLY replace the scripted vector with the
    // emergent geostrophic wind (seed-dependent), not the uniform wind
    // we want to isolate to test advection alone.
    sim.set_uniform_wind(WindVec {
        x: -WEST_BIAS,
        y: 0.0,
    });
    sim
}

#[test]
fn cloud_advection_carries_clouds_downwind_at_hourly_ticks() {
    let mut sim = build_sim();
    let center_idx = sim
        .grid()
        .cell_index(HexCoord::new(0, 0))
        .expect("(0,0) exists in radius-5 grid");

    for _ in 0..HOURS {
        sim.step_hour();
    }

    let coords = sim.grid().coords_slice();
    let cells = sim.grid().cells_slice();

    // Wind toward -x ⇒ downwind = west (world_x < 0), upwind = east
    // (world_x > 0).
    let (mut downwind, mut upwind) = (0.0_f32, 0.0_f32);
    for (i, c) in coords.iter().enumerate() {
        if i == center_idx {
            continue;
        }
        let x = world_x(*c);
        if x < -1e-3 {
            downwind += cells[i].cloud_water;
        } else if x > 1e-3 {
            upwind += cells[i].cloud_water;
        }
    }

    // (1) Directional transport: the cloud clearly moves downwind (-x),
    // not symmetrically. The 3× factor separates wind-driven transport
    // from isotropic diffusion and turns red if the sign of advection
    // flips. Measured at 3.0: downwind ≈ 4.5 mm, upwind ≈ 0.5 mm (part
    // wrapped around the torus).
    assert!(
        downwind > 3.0 * upwind,
        "advection must carry the cloud downwind (-x): downwind={downwind:.4} mm must exceed 3x upwind={upwind:.4} mm"
    );

    // (2) Anti-regression: after 24 h of strong wind the source keeps
    // very little of the pulse. Measured (synoptic OFF, scripted wind):
    // 3.0 → 0.3%, 0.37 → 51%. Threshold 25% = wide margin on both sides;
    // a regression to 0.37 turns this red.
    let source_frac = cells[center_idx].cloud_water / PULSE_MM;
    assert!(
        source_frac < 0.25,
        "after {HOURS} h of strong wind the source must keep < 25% of the pulse (advection strong enough), measured {:.1} %",
        source_frac * 100.0
    );
}
