//! Diagnostic #107: are the bodies of water (lakes + rivers) PERENNIAL?
//!
//! Goal: human agents who build on the water. Constraint: water stable
//! in TIME and SPACE. This diag is the numeric "done-when": how many
//! water cells stay wet at EVERY sample of a year, over several years
//! (multi-seed).
//!
//! Anti-pattern #5 (measure before touching): this instrument establishes
//! the baseline BEFORE any recalibration. It measures three things:
//!
//! 1. **Seasonal budget per stock** (surface / water table / snow / rivers),
//!    sampled summer vs winter each year; reproduces the issue's table
//!    and shows the two blockers (water table draining, winter freezing
//!    everything).
//! 2. **Water table trajectory**: total gw month by month; base flow
//!    comes from there. If it collapses and never comes back up, no
//!    perennial river.
//! 3. **Persistence metric**: cells in water (lake OR river) at ALL the
//!    monthly samples of a year, then across ALL the years.
//!
//! Run: `cargo test -p hexsim-core --release --test diag_perennial_water -- --ignored --nocapture`

mod common;

use common::build_prod_sim;
use hexsim_core::simulation::Simulation;

/// Discharge threshold that defines a "river cell" (def. front/diag #105).
const RIVER_THRESHOLD: f32 = 0.5;
/// Free water depth (mm above `water_capacity`) that defines a
/// "lake cell": a real standing body of water, not a sub-hex puddle.
const LAKE_SURPLUS_MM: f32 = 5.0;

/// Hydrological state of a cell at a sampling instant.
#[derive(Clone, Copy)]
struct CellSample {
    is_lake: bool,
    is_river: bool,
}

impl CellSample {
    fn is_water(self) -> bool {
        self.is_lake || self.is_river
    }
}

/// Snapshot of the water masks + stock totals at the current instant.
#[derive(Clone)]
struct Snapshot {
    day: u64,
    mean_temp: f64,
    surface_total: f32,
    groundwater_total: f32,
    snow_total: f32,
    river_total: f32,
    cells: Vec<CellSample>,
}

fn snapshot(sim: &Simulation) -> Snapshot {
    let grid = sim.grid();
    let discharge = sim.discharge_map();
    let n = grid.len();

    let mut surface_total = 0.0;
    let mut groundwater_total = 0.0;
    let mut snow_total = 0.0;
    let mut river_total = 0.0;
    let mut temp_sum = 0.0;
    let mut cells = Vec::with_capacity(n);

    for (i, (_, cell)) in grid.iter().enumerate() {
        surface_total += cell.water_level;
        groundwater_total += cell.groundwater;
        snow_total += cell.snow_level;
        temp_sum += f64::from(cell.temperature);
        let d = discharge.get(i).copied().unwrap_or(0.0);
        river_total += d;
        let surplus = cell.water_level - cell.water_capacity;
        cells.push(CellSample {
            is_lake: surplus > LAKE_SURPLUS_MM,
            is_river: d > RIVER_THRESHOLD,
        });
    }

    Snapshot {
        day: sim.tick(),
        mean_temp: temp_sum / f64::from(u32::try_from(n).expect("cell count fits u32")),
        surface_total,
        groundwater_total,
        snow_total,
        river_total,
        cells,
    }
}

/// Counts the cells "in water" at ALL the samples in the list.
fn perennial_count(samples: &[Snapshot]) -> usize {
    if samples.is_empty() {
        return 0;
    }
    let n = samples[0].cells.len();
    (0..n)
        .filter(|&i| samples.iter().all(|s| s.cells[i].is_water()))
        .count()
}

/// Peak of cells in water (the best instant): upper bound of the potential.
fn peak_water_cells(samples: &[Snapshot]) -> usize {
    samples
        .iter()
        .map(|s| s.cells.iter().filter(|c| c.is_water()).count())
        .max()
        .unwrap_or(0)
}

fn run_seed(seed: u32, radius: i32, warmup_years: u64, measure_years: u64) {
    eprintln!("\n=== seed {seed} (r{radius}), perennial water (#107) ===");
    let mut sim = build_prod_sim(seed, radius);

    // Warmup: the terrarium starts dry, it takes several years for the
    // cycle to establish itself (water table, snow, rivers).
    for _ in 0..(warmup_years * 365) {
        sim.step();
    }

    // Monthly sampling (~30 d) over the measurement window.
    let mut all_samples: Vec<Snapshot> = Vec::new();

    eprintln!(
        "  {:>5} {:>6} {:>9} {:>8} {:>8} {:>8} {:>7} {:>7}",
        "day", "T°C", "surface", "table", "snow", "rivers", "lakes", "rivC"
    );
    for _year in 0..measure_years {
        let mut year_samples: Vec<Snapshot> = Vec::new();
        for _month in 0..12 {
            for _ in 0..30 {
                sim.step();
            }
            let s = snapshot(&sim);
            let lakes = s.cells.iter().filter(|c| c.is_lake).count();
            let rivers = s.cells.iter().filter(|c| c.is_river).count();
            eprintln!(
                "  {:>5} {:>6.1} {:>9.0} {:>8.0} {:>8.0} {:>8.1} {:>7} {:>7}",
                s.day,
                s.mean_temp,
                s.surface_total,
                s.groundwater_total,
                s.snow_total,
                s.river_total,
                lakes,
                rivers,
            );
            year_samples.push(s);
        }
        let py = perennial_count(&year_samples);
        let peak = peak_water_cells(&year_samples);
        eprintln!("    → year: {py} perennial cells / {peak} at peak");
        all_samples.append(&mut year_samples);
    }

    let overall = perennial_count(&all_samples);
    let peak = peak_water_cells(&all_samples);
    eprintln!("  ===> PERENNIAL over {measure_years} years: {overall} cells (instant peak {peak})");
}

/// Baseline #107: seed 7 r30 (the seed/radius from the issue), short window
/// for iteration. `--ignored --nocapture`.
#[test]
#[ignore = "diagnostic #107, perennial water baseline (seed 7, r30)"]
fn perennial_baseline_seed7() {
    run_seed(7, 30, 3, 5);
}

/// Multi-seed baseline, reduced radius to fit in a fast local CI.
#[test]
#[ignore = "diagnostic #107, multi-seed perennial water (r20)"]
fn perennial_multiseed() {
    for seed in [7, 42, 99] {
        run_seed(seed, 20, 3, 5);
    }
}
