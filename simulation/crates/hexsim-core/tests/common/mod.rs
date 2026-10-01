//! Helpers shared by scale tests (`scale_*.rs`) and diags (`diag_*.rs`).
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
use hexsim_core::species::SpeciesId;
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

/// The terrarium's whole water stock: `Simulation::water_budget_total`,
/// read and never re-assembled here (anti-pattern 2). Surface
/// stocks per cell (free water, surface humidity, groundwater, snowpack
/// and lake ice), the moist upper layer through its coarse REFERENCE
/// stock (coarse upper layer, step 2: the fine `humidity_upper`/
/// `cloud_water` are views rewritten by a non-conservative interpolation
/// every hour, summing them is not the mass — see
/// `Simulation::upper_water_total`), and the sky reservoir of the imposed
/// weather regime (#63), which left the cells but not the world.
pub fn total_water_budget(sim: &Simulation) -> f32 {
    sim.water_budget_total()
}

/// `humidity_upper` (mm of precipitable water) that
/// [`freeze_phase_transition`] loads every cell with.
///
/// Well past `saturation_upper` at any temperature the fixtures using it
/// reach: `PW_sat(20 °C)` ≈ 26 mm over the 1500 m layer, and the upper
/// air is colder than the ground.
pub const FROZEN_TRANSITION_UPPER_MM: f32 = 60.0;

/// Holds the vapour ↔ droplet transition still, so a fixture can watch
/// what *else* acts on `cloud_water` — KK2000 autoconversion, advection,
/// diffusion — with a cloud whose mass only those can change.
///
/// Both directions have to be pinned and, since #63 L2b, they are pinned
/// differently. The forward one is still a rate, so `condensation_rate =
/// 0` stops it. The reverse one has no rate any more: it is a saturation
/// adjustment bounded by `sat − humidity_upper`, so the only way to stop
/// it is to leave no deficit. Hence the floor: every cell starts
/// supersaturated ([`FROZEN_TRANSITION_UPPER_MM`]), the surplus branch is
/// the live one, and at rate 0 it transfers nothing.
///
/// Before L2b the second half was `cloud_evap_rate = 0`, and the fixtures
/// that used it also set `initial_humidity_floor = 0` — a cloud sitting
/// in air at RH 0. The old 0.10/day coefficient let such a cloud live for
/// a week; the physics does not (JOURNAL 2026-09-06), so the fixture had
/// to change or the tests would have measured the advection and the
/// autoconversion of a cloud that no longer exists after hour one.
///
/// #63/#146 (regime default flip, 2026-09-06): the imposed weather
/// regime also reaches into `humidity_upper` directly, on its own clock,
/// regardless of `condensation_rate` or the floor above — a dry episode
/// exports the supersaturation this helper relies on straight to the sky
/// reservoir, and once the deficit follows, the saturation adjustment
/// evaporates the very cloud these fixtures are trying to isolate.
/// `regime_enabled = 0` closes that third door, so the isolation this
/// helper promises actually holds.
///
/// Only touches those three fields, so every other knob the caller set
/// survives.
pub fn freeze_phase_transition(atmosphere: &mut AtmosphereParams) {
    atmosphere.condensation_rate = 0.0;
    atmosphere.initial_humidity_floor = FROZEN_TRANSITION_UPPER_MM;
    atmosphere.regime_enabled = 0.0;
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

/// Column label of a species in the diag tables (≤ 6 chars so a row of
/// 16 species stays readable). Exhaustive on purpose: a species added to
/// `species::SPECIES` fails to compile here until it gets a label, rather
/// than showing up as a blank column.
pub fn species_label(id: SpeciesId) -> &'static str {
    match id {
        SpeciesId::DryGrassland => "dgrass",
        SpeciesId::Meadow => "meadow",
        SpeciesId::AlpineGrass => "agrass",
        SpeciesId::Boxwood => "box",
        SpeciesId::Juniper => "junip",
        SpeciesId::Broom => "broom",
        SpeciesId::Hazel => "hazel",
        SpeciesId::Heath => "heath",
        SpeciesId::HolmOak => "holm",
        SpeciesId::OakPubescent => "oak",
        SpeciesId::Beech => "beech",
        SpeciesId::Fir => "fir",
        SpeciesId::Pine => "pine",
        SpeciesId::Larch => "larch",
        SpeciesId::Maple => "maple",
        SpeciesId::Riparian => "ripar",
    }
}
