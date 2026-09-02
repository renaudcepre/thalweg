//! Helpers shared by scale tests (`scale_*.rs`).
//!
//! Pattern `tests/common/mod.rs` + `mod common;` at the top of each test
//! so cargo doesn't compile this file as a standalone test.

#![allow(dead_code)] // each test consumes only a subset of the helpers

use std::time::Instant;

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

/// Sim "prod-like": generated terrain, defaults everywhere. Since transition
/// to strictly closed terrarium, identical to old `build_closed_sim` (no edge
/// flux, strict conservation guaranteed).
pub fn build_prod_sim(seed: u32, radius: i32) -> Simulation {
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed,
            ..TerrainParams::default()
        },
    );
    let wind = WindParams {
        seed,
        ..WindParams::default()
    };
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind,
    )
}

/// Sum of 4 water stocks (surface + humidity + groundwater + snow).
pub fn total_water_budget(sim: &Simulation) -> f32 {
    sim.grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum()
}

/// Minimal stopwatch: `start` -> multiple `lap(label)` -> `report(name)`.
/// Each `lap` is independent (interval since previous `lap` or `start`).
pub struct PerfTimer {
    label: String,
    start: Instant,
    last: Instant,
    laps: Vec<(String, f64)>,
    ticks: Option<u64>,
}

impl PerfTimer {
    pub fn start(label: &str) -> Self {
        let now = Instant::now();
        Self {
            label: label.to_string(),
            start: now,
            last: now,
            laps: Vec::new(),
            ticks: None,
        }
    }

    /// Records elapsed time since previous `lap` (or `start`).
    pub fn lap(&mut self, name: &str) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        self.laps.push((name.to_string(), dt));
        self.last = now;
    }

    /// Associates total tick count to execution for ms/tick calculation.
    pub fn ticks(&mut self, n: u64) {
        self.ticks = Some(n);
    }

    /// Prints report to stderr (visible with `cargo test -- --nocapture`).
    pub fn report(&self) {
        let total = self.start.elapsed().as_secs_f64();
        eprintln!("=== Perf {} ===", self.label);
        for (name, dt) in &self.laps {
            eprintln!("  {name:<24} {dt:>8.3} s");
        }
        eprintln!("  {:<24} {:>8.3} s", "TOTAL", total);
        if let Some(n) = self.ticks {
            let n_f = f64::from(u32::try_from(n).expect("tick count fits u32"));
            let ms_per_tick = (total * 1000.0) / n_f;
            eprintln!(
                "  {:<24} {:>8.2} ms/tick ({} ticks)",
                "ms/tick moyen", ms_per_tick, n
            );
        }
        eprintln!("{}", "=".repeat(34 + self.label.len()));
    }
}

/// Small helper to format readable percentage.
pub fn pct(x: f32) -> String {
    format!("{:.1} %", x * 100.0)
}
