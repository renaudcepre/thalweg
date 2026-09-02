//! Diagnostic #107 blocker 1: why is the water table draining?
//!
//! Anti-pattern #1 (overengineering before diagnostic) + #5 (measure before
//! touching). Before recalibrating or remodeling the water table, isolate
//! the cause by knock-out ablation of each lever:
//!   - `infiltration_rate`: does surface fill the table adequately?
//!   - `diffusion_rate`: does piezometric diffusion drain the peaks?
//!   - `max_capacity`: does the ceiling constrain the stock?
//!
//! For each config: warmup then measure mean water table stock (total mm),
//! its concentration (share in top 5 % richest cells; table "pooled" at lows
//! vs distributed), and winter base flow (total discharge at mid-winter; this
//! is what the table should sustain between rains).
//!
//! Run: `cargo test -p hexsim-core --release --test diag_groundwater_ablation -- --ignored --nocapture`

mod common;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

fn build(seed: u32, radius: i32, gw: GroundwaterParams) -> Simulation {
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed,
            ..TerrainParams::default()
        },
    );
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        gw,
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed,
            ..WindParams::default()
        },
    )
}

struct Metrics {
    gw_total_mean: f32,
    gw_top5pct_share: f32,
    winter_discharge_mean: f32,
    summer_discharge_mean: f32,
}

fn gw_total(sim: &Simulation) -> f32 {
    sim.grid().iter().map(|(_, c)| c.groundwater).sum()
}

/// Share of water table stock concentrated in top 5 % richest cells.
fn gw_top5pct_share(sim: &Simulation) -> f32 {
    let mut v: Vec<f32> = sim.grid().iter().map(|(_, c)| c.groundwater).collect();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f32 = v.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let k = (v.len() / 20).max(1);
    v[..k].iter().sum::<f32>() / total
}

fn discharge_total(sim: &Simulation) -> f32 {
    sim.discharge_map().iter().sum()
}

fn run_case(label: &str, seed: u32, radius: i32, gw: GroundwaterParams) -> Metrics {
    let mut sim = build(seed, radius, gw);
    // Warmup: water table takes several years to reach equilibrium.
    for _ in 0..(4 * 365) {
        sim.step();
    }
    // Measure over 2 years, monthly sampling. Separate winter (T < 0) and
    // summer (T > 10) by mean map temperature of the sample.
    let mut gw_sum = 0.0;
    let mut share_sum = 0.0;
    let mut n_samples = 0.0;
    let mut winter_d = 0.0;
    let mut winter_n = 0.0;
    let mut summer_d = 0.0;
    let mut summer_n = 0.0;
    for _ in 0..24 {
        for _ in 0..30 {
            sim.step();
        }
        gw_sum += gw_total(&sim);
        share_sum += gw_top5pct_share(&sim);
        n_samples += 1.0;
        let count = f64::from(u32::try_from(sim.grid().len()).expect("cell count fits u32"));
        let mean_t = sim
            .grid()
            .iter()
            .map(|(_, c)| f64::from(c.temperature))
            .sum::<f64>()
            / count;
        let d = discharge_total(&sim);
        if mean_t < 0.0 {
            winter_d += d;
            winter_n += 1.0;
        } else if mean_t > 10.0 {
            summer_d += d;
            summer_n += 1.0;
        }
    }
    let m = Metrics {
        gw_total_mean: gw_sum / n_samples,
        gw_top5pct_share: share_sum / n_samples,
        winter_discharge_mean: if winter_n > 0.0 {
            winter_d / winter_n
        } else {
            0.0
        },
        summer_discharge_mean: if summer_n > 0.0 {
            summer_d / summer_n
        } else {
            0.0
        },
    };
    eprintln!(
        "  {label:<28} gw_total {:>8.0}  top5% {:>5.1}%  winter_Q {:>7.1}  summer_Q {:>7.1}",
        m.gw_total_mean,
        m.gw_top5pct_share * 100.0,
        m.winter_discharge_mean,
        m.summer_discharge_mean,
    );
    m
}

#[test]
#[ignore = "diagnostic #107 blocker 1, water table ablation (seed 7, r20)"]
fn groundwater_ablation() {
    let seed = 7;
    let radius = 20;
    eprintln!(
        "\n=== #107 water table ablation - seed {seed} r{radius}, 4yr warmup + 2yr measure ==="
    );
    let d = GroundwaterParams::default();
    eprintln!(
        "  (default: infiltration {}, diffusion {}, max_capacity {})",
        d.infiltration_rate, d.diffusion_rate, d.max_capacity
    );

    run_case("default", seed, radius, GroundwaterParams::default());
    run_case(
        "diffusion=0 (knock-out)",
        seed,
        radius,
        GroundwaterParams {
            diffusion_rate: 0.0,
            ..GroundwaterParams::default()
        },
    );
    run_case(
        "diffusion/10",
        seed,
        radius,
        GroundwaterParams {
            diffusion_rate: 0.003,
            ..GroundwaterParams::default()
        },
    );
    run_case(
        "infiltration x4",
        seed,
        radius,
        GroundwaterParams {
            infiltration_rate: 0.2,
            ..GroundwaterParams::default()
        },
    );
    run_case(
        "max_capacity x5",
        seed,
        radius,
        GroundwaterParams {
            max_capacity: 500.0,
            ..GroundwaterParams::default()
        },
    );
    run_case(
        "infil x4 + diff/10",
        seed,
        radius,
        GroundwaterParams {
            infiltration_rate: 0.2,
            diffusion_rate: 0.003,
            ..GroundwaterParams::default()
        },
    );
}
