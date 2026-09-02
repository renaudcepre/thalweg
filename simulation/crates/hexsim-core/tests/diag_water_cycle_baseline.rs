//! Water cycle baseline BEFORE #77 (SI transpiration coupling).
//!
//! #77 will remove the evapotranspiration proxy `ground_evap_rate`
//! (`atmosphere.rs:596-606`, which pumps `groundwater → humidity_surface`)
//! and replace it with transpiration driven by **vegetation**. This
//! reopens the water cycle: this diag freezes the BEFORE state, to
//! compare against the AFTER.
//!
//! **What to watch after #77** (same lines, compared):
//! - **Terrarium**: conservation drift must stay ~0 (hard invariant).
//! - **Humidity sources**: `humidity` stock + average `humidity_surface`,
//!   the proxy injects into it today; transpiration will take over. The
//!   total flux must stay the same order of magnitude, or rain/clouds tip over.
//! - **Water table**: the proxy AND future transpiration both drain
//!   `groundwater`.
//! - **Rain by band**: directly downstream of the source change.
//! - **Open-water evaporation** (`evap_observer`): control, #77 doesn't
//!   touch it.
//! - **Vegetation**: the feedback loop (more vegetation → more
//!   transpiration → more rain → more vegetation?), watch for runaway
//!   growth or drying out.
//!
//! **Eval style**: `#[ignore]`, no assert. Run via `just diag-tools` or:
//! ```text
//! cargo test --release -p hexsim-core --test diag_water_cycle_baseline \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::climate::{Window, default_bands};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::vegetation::{cell_total_vegetation, is_open_water};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const WARMUP_DAYS: u64 = 365;
const MEASURE_YEARS: u64 = 3;

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: SEED,
            ..TerrainParams::default()
        },
    );
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed: SEED,
            ..WindParams::default()
        },
    )
}

#[test]
#[ignore = "eval-style baseline (slow): run via just diag-tools"]
fn diag_water_cycle_baseline() {
    let mut sim = build_sim();
    println!("== Water cycle baseline BEFORE #77 (seed {SEED}, radius {RADIUS}) ==");
    println!("warmup {WARMUP_DAYS} d then {MEASURE_YEARS} years measured\n");

    for _ in 0..WARMUP_DAYS {
        sim.step();
    }
    let total_after_warmup = sim.diagnostics().water_budget.total;

    // Annual biomass peak over the last year (biome definition).
    let n = sim.grid().len();
    let mut peak = vec![0.0_f32; n];
    for y in 0..MEASURE_YEARS {
        let last_year = y == MEASURE_YEARS - 1;
        for _ in 0..365 {
            sim.step();
            if last_year {
                for (p, c) in peak.iter_mut().zip(sim.grid().cells_slice()) {
                    *p = p.max(cell_total_vegetation(c));
                }
            }
        }
    }

    let d = sim.diagnostics();
    let wb = &d.water_budget;

    println!("== TERRARIUM (invariant, must stay ~0 after #77) ==");
    let drift = (wb.total - total_after_warmup) / total_after_warmup.abs().max(1.0) * 100.0;
    println!(
        "  total water  end warmup={total_after_warmup:.1}  end measure={:.1}  drift={drift:.3}%",
        wb.total
    );
    println!(
        "  budget     surface={:.1}  humidity={:.1}  groundwater={:.1}  snow={:.1}",
        wb.surface, wb.humidity, wb.groundwater, wb.snow
    );

    println!("\n== HUMIDITY SOURCES (what #77 replaces) ==");
    println!(
        "  humidity stock   mean={:.3}  total={:.1}",
        d.humidity.mean, d.humidity.total
    );
    println!(
        "  humidity_surface mean={:.3} mm  (the proxy injects here → transpiration next)",
        d.humidity_surface_mm.mean
    );
    println!(
        "  groundwater      mean={:.3}  total={:.1}  (drained by proxy + future transpiration)",
        d.groundwater.mean, d.groundwater.total
    );
    println!(
        "  open water evap  mean={:.3} mm/d over {} cells  (control, #77 doesn't touch it)",
        d.evap_observer.mean_mm_day, d.evap_observer.cell_count
    );
    println!(
        "  humidity high={:.3}  low={:.3}  lapse_eff={:.2} °C/km",
        d.altitude.humidity_high, d.altitude.humidity_low, d.altitude.effective_lapse_rate_c_per_km
    );

    println!("\n== RAIN by elevation band (downstream of the source change) ==");
    let bands = default_bands();
    let stats = sim
        .climate_history()
        .aggregate(sim.grid(), &bands, Window::Last365);
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>10}",
        "band", "cells", "d_rain/yr", "mm_rain/yr", "arid"
    );
    for s in &stats {
        println!(
            "{:>6} {:>10} {:>12.1} {:>12.2} {:>10}",
            s.name, s.cells, s.avg_rain_days, s.avg_total_rain, s.arid_cells
        );
    }

    println!("\n== VEGETATION (feedback to monitor) ==");
    // Cover density at the annual peak (species composition is measured
    // by diag_species_distribution; here we track aggregate cover).
    let (mut water, mut bare, mut sparse, mut dense) = (0u32, 0u32, 0u32, 0u32);
    for (cell, &v) in sim.grid().cells_slice().iter().zip(&peak) {
        if is_open_water(cell) {
            water += 1;
        } else if v < 0.12 {
            bare += 1;
        } else if v < 0.5 {
            sparse += 1;
        } else {
            dense += 1;
        }
    }
    let peak_total: f32 = peak.iter().sum();
    let nf = f64::from(u32::try_from(n).unwrap_or(u32::MAX).max(1));
    let pct = |c: u32| f64::from(c) / nf * 100.0;
    println!(
        "  cover (peak)  water={:.1}%  bare={:.1}%  sparse={:.1}%  dense={:.1}%",
        pct(water),
        pct(bare),
        pct(sparse),
        pct(dense)
    );
    println!("  total peak biomass={peak_total:.0}");
}
