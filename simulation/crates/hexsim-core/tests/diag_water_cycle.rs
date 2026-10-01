//! Diagnostic: where does the box's water leave the ground, and where does
//! it fall back? (#107 follow-up, 2026-10-01.)
//!
//! #107 measured that the root zone outside lakes never reaches field
//! capacity and that 85 % of the box's water sits in lakes covering 1 % of
//! the map. Adding water (×3) only fed the lakes. The question this
//! instrument answers before any tuning: is the cycle a loop over the
//! lakes (they evaporate, it rains back on them) or does their vapour
//! reach the land?
//!
//! One year of daily fluxes per cell, after a 3-year warmup (past the
//! drainage transient #152 named), attributed each day to the cell's
//! class that day: lake (the core's water-body predicate) or land, land
//! split by the core's altitude bands. Precipitation is the core's daily
//! accumulator, vapour the exact masses `step_evaporation` moved, by
//! source. Nothing is recomputed here (anti-pattern #2).
//!
//! Run: `cargo test -p hexsim-core --release --test diag_water_cycle -- --ignored --nocapture`

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::climate::default_bands;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

/// Free-water depth (mm above `water_capacity`, liquid or ice) that makes
/// a cell a lake, as in `diag_perennial_water`.
const LAKE_SURPLUS_MM: f32 = 5.0;

/// Yearly fluxes summed over the cell-days of one class (mm × cell-day).
#[derive(Default, Clone, Copy)]
struct ClassFlux {
    cell_days: f64,
    precipitation: f64,
    open_water: f64,
    transpiration: f64,
    sublimation: f64,
}

impl ClassFlux {
    fn evaporation(self) -> f64 {
        self.open_water + self.transpiration + self.sublimation
    }

    /// Per cell-year (mm/yr on an average cell of the class).
    fn per_cell_year(self, flux: f64) -> f64 {
        if self.cell_days > 0.0 {
            flux / self.cell_days * 365.0
        } else {
            0.0
        }
    }
}

fn record_day(sim: &Simulation, classes: &mut [ClassFlux], class_of: &dyn Fn(usize) -> usize) {
    let precipitation = sim.last_precipitation();
    let vapor = sim.vapor_sources_today();
    for i in 0..sim.grid().len() {
        let class = &mut classes[class_of(i)];
        class.cell_days += 1.0;
        class.precipitation += f64::from(precipitation[i].rain + precipitation[i].snow);
        class.open_water += f64::from(vapor[i].open_water);
        class.transpiration += f64::from(vapor[i].transpiration);
        class.sublimation += f64::from(vapor[i].sublimation);
    }
}

/// Production sim whose surface water at worldgen (the runoff sheet routed
/// into the basins, #152) is multiplied by `water_mult`: the closed box's
/// budget grows by that much, conserved forever. `1.0` is `build_prod_sim`.
fn build_sim(seed: u32, radius: i32, water_mult: f32) -> Simulation {
    build_sim_with(seed, radius, water_mult, GroundwaterParams::default())
}

fn build_sim_with(
    seed: u32,
    radius: i32,
    water_mult: f32,
    groundwater: GroundwaterParams,
) -> Simulation {
    let mut grid = HexGrid::from_radius(radius);
    let terrain = TerrainParams {
        seed,
        ..TerrainParams::default()
    };
    generate_terrain(
        &mut grid,
        &TerrainParams {
            initial_water: terrain.initial_water * water_mult,
            ..terrain
        },
    );
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        groundwater,
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed,
            ..WindParams::default()
        },
    )
}

fn run_seed(seed: u32, radius: i32, warmup_years: u64, water_mult: f32) {
    eprintln!(
        "\n=== water cycle / seed {seed} r{radius}, surface water x{water_mult} ({warmup_years} year warmup + 1 year) ==="
    );
    report(build_sim(seed, radius, water_mult), warmup_years);
}

fn report(mut sim: Simulation, warmup_years: u64) {
    for _ in 0..(warmup_years * 365) {
        sim.step();
    }

    let bands = default_bands();
    let mut labels = vec!["lake".to_string()];
    labels.extend(bands.iter().map(|b| format!("land {}", b.range())));
    let mut classes = vec![ClassFlux::default(); labels.len()];

    for _ in 0..365 {
        sim.step();
        let cells = sim.grid().cells_slice();
        let class_of = |i: usize| -> usize {
            let cell = &cells[i];
            if cell.is_open_water_at(LAKE_SURPLUS_MM) {
                return 0;
            }
            1 + bands
                .iter()
                .position(|b| b.contains(cell.elevation))
                .expect("default bands cover every elevation")
        };
        record_day(&sim, &mut classes, &class_of);
    }

    let total_p: f64 = classes.iter().map(|c| c.precipitation).sum();
    let total_e: f64 = classes.iter().map(|c| c.evaporation()).sum();
    let total_cell_days: f64 = classes.iter().map(|c| c.cell_days).sum();
    eprintln!(
        "  {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7}",
        "class", "area%", "P mm/yr", "Eow", "T", "S", "P-E", "P%", "E%"
    );
    for (label, c) in labels.iter().zip(&classes) {
        eprintln!(
            "  {:<18} {:>6.2} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>7.1} {:>7.1}",
            label,
            100.0 * c.cell_days / total_cell_days,
            c.per_cell_year(c.precipitation),
            c.per_cell_year(c.open_water),
            c.per_cell_year(c.transpiration),
            c.per_cell_year(c.sublimation),
            c.per_cell_year(c.precipitation - c.evaporation()),
            100.0 * c.precipitation / total_p.max(1e-12),
            100.0 * c.evaporation() / total_e.max(1e-12),
        );
    }
    let cells = total_cell_days / 365.0;
    eprintln!(
        "  map: P {:.1} mm/yr, E {:.1} mm/yr per cell, budget {:.1} mm/cell",
        total_p / cells,
        total_e / cells,
        f64::from(sim.water_budget_total()) / cells
    );
}

/// The three seeds of #152 / #107, r30, 3-year warmup + 1 measured year.
#[test]
#[ignore = "diagnostic, water cycle by surface class (three seeds, r30)"]
fn water_cycle_three_seeds_r30() {
    for seed in [42, 7, 123] {
        run_seed(seed, 30, 3, 1.0);
    }
}

/// Does the cycle's intensity follow the open-water area? In a closed box
/// it only rains what evaporates; lakes are 1 % of the map and 60 % of
/// the evaporation. Same seeds and protocol, surface water ×3 and ×10.
#[test]
#[ignore = "diagnostic, water cycle against the box's surface water (three seeds, r30)"]
fn water_cycle_against_surface_water_r30() {
    for water_mult in [3.0, 10.0] {
        for seed in [42, 7, 123] {
            run_seed(seed, 30, 3, water_mult);
        }
    }
}

/// Infiltration capacity (2026-10-01): the SI default (loam, Ks 317
/// mm/day) against a clay surface (Ks 14 mm/day, Rawls et al. 1982).
/// Before the SI rewrite the fraction-of-ponded-water rate let ~0.25
/// mm/day in; its ×20 ablation doubled the cycle (26-32 → 60-66 mm/yr).
#[test]
#[ignore = "diagnostic, water cycle against infiltration (three seeds, r30)"]
fn water_cycle_against_infiltration_r30() {
    for seed in [42, 7, 123] {
        eprintln!("\n=== water cycle / seed {seed} r30, Ks 14 (clay) (3 year warmup + 1 year) ===");
        let groundwater = GroundwaterParams {
            saturated_conductivity_mm_per_day: 14.0,
            ..GroundwaterParams::default()
        };
        report(build_sim_with(seed, 30, 1.0, groundwater), 3);
    }
}

/// SI infiltration with more water in the box: surface water ×3 and ×10.
#[test]
#[ignore = "diagnostic, water cycle against infiltration and water (three seeds, r30)"]
fn water_cycle_infiltration_and_water_r30() {
    for water_mult in [3.0, 10.0] {
        for seed in [42, 7, 123] {
            eprintln!(
                "\n=== water cycle / seed {seed} r30, Ks 317, surface water x{water_mult} (3 year warmup + 1 year) ==="
            );
            report(build_sim(seed, 30, water_mult), 3);
        }
    }
}
