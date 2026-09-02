//! Diagnostic #68: cloud advection speed at the hourly tick.
//!
//! RESOLVED (2026-07-12, first building block): `cloud_advection_rate`
//! bumped 0.37 → 3.0 to restore parity with `humidity_advection_rate`
//! (vapor alone had been bumped on 2026-07-05, leaving the droplets
//! crawling). Measured here (synoptic OFF, no-precip, uniform scripted
//! wind): source remaining after 24 h **51.1% → 0.3%**, speed
//! **0.025 → 0.151 cells/h**. Anti-regression sentinel:
//! `tests/phys_cloud_advection_moves_downwind.rs`. This diagnostic
//! remains useful to re-explore further if we want to push beyond
//! parity (Courant ≫ 1 at 1 km/hourly tick, parity is not the physical
//! limit).
//!
//! Hypothesis: `cloud_advection_rate` divided by 24 in
//! `scale_atmosphere_for_hourly_tick` (v0.3.0 legacy) under-scales
//! horizontal advection. At ~10 m/s over a ~1 km cell, crossing takes
//! < 100 s, so the fraction transported per hour should approach 100%,
//! not ~15%. Visual symptom: "smoke column" clouds that take ~12 h to
//! react to a wind change (the visually-static-clouds bug, #68, flagged
//! during visual validation).
//!
//! Setup:
//! - radius 5 grid (91 cells, flat terrain, no water, no initial
//!   humidity)
//! - uniform west wind forced via `Simulation::set_uniform_wind` (#108),
//!   all other `WindParams` mechanisms (Perlin noise, thermal, relief
//!   deflection, vapor advection) neutralized
//! - atmosphere: no `cloud_water` regeneration (`condensation_rate=0`,
//!   `cloud_evap_rate=0`, `fog_condensation_rate=0`), no transpiration
//!   (`transpiration_coef=0`), no initial humidity
//!   (`initial_humidity_floor=0`)
//! - KK2000 precipitation left active = normal part of the decay
//!
//! Pulse: `cloud_water = 5.0 mm` on cell (0,0), rest of the grid at 0.
//! We observe 24 hourly sub-ticks (`Simulation::step_hour`), not
//! `Simulation::step()` which would do 24 ticks at once and hide the
//! hour-by-hour dynamics.
//!
//! Eval style (cf project memory `scale_tests_eval_style`): no assert,
//! just `eprintln!` + `#[ignore]`. We read, draw the hypothesis, and
//! act.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test scale_cloud_advection_lifetime \
//!     -- --ignored --nocapture
//! ```

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
const TOTAL_HOURS: u64 = 24;

/// `WindField` convention (cf `wind.rs`): `magnitude × 10 = m/s`.
/// Magnitude of the old `west_bias`, set via `Simulation::set_uniform_wind`
/// (mapping #108: `west_bias = v` → `WindVec { x: -v, y: 0.0 }`) ⇒
/// `wind.x = -1.0` ⇒ 10 m/s west at the surface, ~14 m/s upper after
/// `wind_upper_speed_ratio = 1.4`.
const WEST_BIAS: f32 = 1.0;

/// To really disable KK2000 (autoconversion to rain), you have to set
/// `kk2000_droplet_count = 0`: the `max_precip_per_tick` cap only
/// disables the ceiling, not KK2000 itself (cf the guard line
/// `droplet_count_cm3 <= 0.0` in `kk2000_drain_mm_per_hour`). The
/// `max_precip_per_tick` field's doc says "at 0: disabled" but that's
/// misleading, debt to fix.
fn build_atmosphere(kk2000_droplet_count: f32) -> AtmosphereParams {
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
        kk2000_droplet_count,
        ..AtmosphereParams::default()
    }
}

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

fn build_sim_with_pulse(kk2000_droplet_count: f32) -> (Simulation, usize, HexCoord) {
    let mut grid = HexGrid::from_radius(RADIUS);
    let center_coord = HexCoord::new(0, 0);
    if let Some(cell) = grid.get_mut(center_coord) {
        cell.cloud_water = PULSE_MM;
    }
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        build_atmosphere(kk2000_droplet_count),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        build_wind(),
    );
    // `set_uniform_wind` forces the surface wind field to the desired
    // vector AND automatically disables synoptic dynamics (ON BY
    // DEFAULT otherwise, `unwrap_or(true)`), which would otherwise
    // replace the scripted wind with its emergent geostrophic field
    // (seed-dependent); isolates advection under the documented
    // uniform wind (cf phys_humidity_advection.rs).
    sim.set_uniform_wind(WindVec {
        x: -WEST_BIAS,
        y: 0.0,
    });
    let center_idx = sim
        .grid()
        .cell_index(center_coord)
        .expect("(0,0) exists in radius-5 grid");
    (sim, center_idx, center_coord)
}

/// Hex metric distance (in cells) from `center_coord`, by dense index.
/// With `RADIUS = 5`, `d ∈ [0, 10]` ⇒ fits in `i16` without loss.
fn cell_distances_from(sim: &Simulation, center_coord: HexCoord) -> Vec<f32> {
    sim.grid()
        .coords_slice()
        .iter()
        .map(|c| f32::from(i16::try_from(c.distance(center_coord)).unwrap_or(0)))
        .collect()
}

#[derive(Clone, Copy)]
struct HourSample {
    hour: u64,
    cw_total: f32,
    cw_source: f32,
    cw_downwind: f32,
    weighted_dist_cells: f32,
}

fn sample(hour: u64, sim: &Simulation, distances: &[f32], center_idx: usize) -> HourSample {
    let cells = sim.grid().cells_slice();
    let mut cw_total = 0.0_f32;
    let mut weighted = 0.0_f32;
    for (i, c) in cells.iter().enumerate() {
        cw_total += c.cloud_water;
        weighted += c.cloud_water * distances[i];
    }
    let cw_source = cells[center_idx].cloud_water;
    let cw_downwind = (cw_total - cw_source).max(0.0);
    let weighted_dist_cells = if cw_total > 1e-6 {
        weighted / cw_total
    } else {
        0.0
    };
    HourSample {
        hour,
        cw_total,
        cw_source,
        cw_downwind,
        weighted_dist_cells,
    }
}

fn print_row(s: HourSample) {
    eprintln!(
        "  {:>5} {:>10.4} {:>10.4} {:>10.4} {:>12.3}",
        s.hour, s.cw_total, s.cw_source, s.cw_downwind, s.weighted_dist_cells,
    );
}

fn run_pulse_decay(label: &str, kk2000_droplet_count: f32) {
    let (mut sim, center_idx, center_coord) = build_sim_with_pulse(kk2000_droplet_count);
    let distances = cell_distances_from(&sim, center_coord);
    let n_cells = sim.grid().len();
    let upper_speed_ratio = sim.wind_params().wind_upper_speed_ratio;
    let cloud_adv_rate_per_day = sim.atmosphere_params().cloud_advection_rate;
    let cloud_adv_rate_per_hour = cloud_adv_rate_per_day / 24.0;
    let cloud_diff_rate_per_day = sim.atmosphere_params().cloud_diffusion_rate;
    let cloud_diff_rate_per_hour = cloud_diff_rate_per_day / 24.0;
    let max_precip_per_tick = sim.atmosphere_params().max_precip_per_tick;

    eprintln!(
        "\n=== Cloud advection pulse decay (#68 diag) [{label}], radius {RADIUS}, {n_cells} cells, uniform west wind ==="
    );
    eprintln!(
        "  uniform wind (set_uniform_wind) magnitude {WEST_BIAS:.2} → surface wind ~{:.0} m/s, upper wind ~{:.0} m/s (ratio×{upper_speed_ratio:.1})",
        WEST_BIAS * 10.0,
        WEST_BIAS * 10.0 * upper_speed_ratio,
    );
    eprintln!(
        "  cloud_advection_rate per-day (default) = {cloud_adv_rate_per_day:.3} → per-hour after /24 = {cloud_adv_rate_per_hour:.4}"
    );
    eprintln!(
        "  cloud_diffusion_rate per-day (default) = {cloud_diff_rate_per_day:.3} → per-hour after /24 = {cloud_diff_rate_per_hour:.4}"
    );
    eprintln!(
        "  kk2000_droplet_count                    = {kk2000_droplet_count:.1} cm⁻³ (0 = KK2000 disabled)"
    );
    eprintln!(
        "  max_precip_per_tick                     = {max_precip_per_tick:.2} mm/tick (NB: not scaled /24, legacy debt from v0.3.0)"
    );
    eprintln!(
        "  initial pulse: cloud_water = {PULSE_MM:.2} mm on cell (0,0), rest of the grid at 0\n"
    );
    eprintln!(
        "  {:>5} {:>10} {:>10} {:>10} {:>12}",
        "hour", "cw_total", "cw_source", "cw_downw", "dist_pond_h"
    );
    eprintln!("  {}", "-".repeat(58));

    let cap = usize::try_from(TOTAL_HOURS).unwrap_or(0) + 1;
    let mut samples: Vec<HourSample> = Vec::with_capacity(cap);
    let s0 = sample(0, &sim, &distances, center_idx);
    samples.push(s0);
    print_row(s0);

    for h in 1..=TOTAL_HOURS {
        sim.step_hour();
        let s = sample(h, &sim, &distances, center_idx);
        samples.push(s);
        print_row(s);
    }

    let half_threshold = PULSE_MM / 2.0;
    let half_life = samples
        .iter()
        .find(|s| s.cw_total < half_threshold)
        .map(|s| s.hour);

    let last = samples.last().copied().expect("at least h=0 sampled");
    let total_hours_f = f32::from(u16::try_from(TOTAL_HOURS).unwrap_or(0));
    let eff_velocity_cph = if total_hours_f > 0.0 {
        last.weighted_dist_cells / total_hours_f
    } else {
        0.0
    };
    let frac_source = if PULSE_MM > 0.0 {
        last.cw_source / PULSE_MM
    } else {
        0.0
    };

    eprintln!("\n=== Key figures ===");
    match half_life {
        Some(h) => eprintln!("  cw_total half-life        : {h} h"),
        None => eprintln!(
            "  cw_total half-life        : >{TOTAL_HOURS} h (final cw_total = {:.3} mm out of {PULSE_MM:.2})",
            last.cw_total
        ),
    }
    eprintln!(
        "  effective adv speed       : {eff_velocity_cph:.3} cells/h (weighted mass displacement / {TOTAL_HOURS} h)"
    );
    eprintln!(
        "  source fraction left      : {:.1}% at h={TOTAL_HOURS}",
        frac_source * 100.0
    );
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn cloud_advection_pulse_decay_baseline() {
    // kk2000_droplet_count = 30 cm⁻³ (default) → KK2000 actif, cap 4 mm/h.
    let default_kk = AtmosphereParams::default().kk2000_droplet_count;
    run_pulse_decay("baseline (KK2000 active)", default_kk);
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn cloud_advection_pulse_decay_no_precip() {
    // kk2000_droplet_count = 0 -> KK2000 returns 0.0 (guard around line
    // ~1404 of atmosphere.rs). Isolates pure advection + diffusion.
    run_pulse_decay("no precip (pure advection)", 0.0);
}

// The Legend block is printed only once at the bottom of both runs
// via the test runner (eprintln in each run_pulse_decay), to avoid
// duplicating the doc for every variant.
//
// Legend:
//   cw_total      : Σ cloud_water over the whole grid (mm).
//   cw_source     : cloud_water on cell (0,0) alone.
//   cw_downw      : cw_total - cw_source = what got advected elsewhere.
//   dist_pond_h   : Σ (cw × dist_hex) / cw_total = center of mass in
//                   hex cells from (0,0). 1 cell ≈ 1 km.
//
// Hypothesis:
//   At 10 m/s, crossing a cell (~1 km) takes < 100 s. So the fraction
//   per hour → ~100%. If speed < 0.3 cells/h or source fraction > 50%
//   after 24 h in the "no precip" variant, the /24 scaling on
//   `cloud_advection_rate` is too conservative for the hourly tick.
