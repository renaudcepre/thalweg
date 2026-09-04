//! Structural diagnostic, distribution of emergent **species** (epic #78,
//! step D: #82). Replaces the former `diag_vegetation_biomes` (abstract biomes).
//!
//! Objective metric **before any tuning** of the niches (`species::SPECIES`)
//! and rates (`VegetationParams`) - no physical balance change without
//! global metrics at scale. Measured over
//! 10 years (daily resolution):
//! - **temporal succession**: surface fraction per dominant species, year
//!   by year (pioneers first, then climax, or steady-state);
//! - **species × altitude band distribution** (band × category matrix);
//! - biomass bounding / NaN-free check;
//! - determinism (same seed → same total biomass per species).
//!
//! **Known limitation to quantify here**: v0 competition = shared space,
//! without succession (shade tolerance) → a generalist species with a wide
//! niche (pine) can dominate everywhere. This diag is built to show it.
//!
//! **Eval style** (`scale_tests_eval_style`): `#[ignore]`, no assert,
//! structured output meant to be read to calibrate `Species` / `VegetationParams`.
//!
//! ```text
//! cargo test --release -p hexsim-core --test diag_species_distribution \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::cell::CellProperties;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::species::{SPECIES, SpeciesId};
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::vegetation::{cell_total_vegetation, dominant_species, is_open_water};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const DEFAULT_SEED: u32 = 42;

/// Seed of the run, overridable so the #151 guard metric (bare fraction
/// per band on 3 seeds) does not need an edit between runs:
/// `HEXSIM_DIAG_SEED=7 cargo test --release ... -- --ignored --nocapture`.
fn seed() -> u32 {
    std::env::var("HEXSIM_DIAG_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SEED)
}
const WARMUP_DAYS: u64 = 365;
const RUN_YEARS: u64 = 10;

const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

/// Cover categories: open water, bare soil, then one per species.
/// `N_CAT = 2 + number of species`.
const N_CAT: usize = 2 + SPECIES.len();

fn cat_labels() -> [String; N_CAT] {
    let mut out: [String; N_CAT] = core::array::from_fn(|_| String::new());
    out[0] = "water".to_string();
    out[1] = "bare".to_string();
    for (i, s) in SPECIES.iter().enumerate() {
        out[2 + i] = species_label(s.id).to_string();
    }
    out
}

fn species_label(id: SpeciesId) -> &'static str {
    match id {
        SpeciesId::OakPubescent => "oak",
        SpeciesId::Pine => "pine",
        SpeciesId::Beech => "beech",
        SpeciesId::Fir => "fir",
        SpeciesId::AlpineGrass => "grass",
    }
}

fn species_index(id: SpeciesId) -> usize {
    SPECIES.iter().position(|s| s.id == id).unwrap_or(0)
}

/// Category of a cell: 0 = water, 1 = bare soil, 2+i = dominant species i.
fn category(cell: &CellProperties) -> usize {
    if is_open_water(cell) {
        return 0;
    }
    match dominant_species(cell) {
        Some(id) => 2 + species_index(id),
        None => 1,
    }
}

fn band_index(elev: f32) -> usize {
    BANDS
        .iter()
        .position(|&(_, lo, hi)| elev >= lo && elev < hi)
        .unwrap_or(0)
}

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: seed(),
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
            seed: seed(),
            ..WindParams::default()
        },
    )
}

/// Surface fractions per category (on the current state).
fn fractions(sim: &Simulation) -> [f64; N_CAT] {
    let mut counts = [0u32; N_CAT];
    let mut n = 0u32;
    for (_, cell) in sim.grid().iter() {
        counts[category(cell)] += 1;
        n += 1;
    }
    let total = f64::from(n.max(1));
    let mut out = [0.0; N_CAT];
    for (o, c) in out.iter_mut().zip(counts.iter()) {
        *o = f64::from(*c) / total * 100.0;
    }
    out
}

/// Total biomass per species across the whole grid (determinism).
fn biomass_per_species(sim: &Simulation) -> [f64; 5] {
    let mut out = [0.0_f64; 5];
    for (_, cell) in sim.grid().iter() {
        for (o, &v) in out.iter_mut().zip(cell.vegetation.iter()) {
            *o += f64::from(v);
        }
    }
    out
}

#[test]
#[ignore = "eval-style diagnostic (slow): run via just diag-tools"]
fn diag_species_distribution() {
    let mut sim = build_sim();
    let labels = cat_labels();

    println!("== Emergent species (seed {}, radius {RADIUS}) ==", seed());
    println!("warmup {WARMUP_DAYS}d then {RUN_YEARS} years measured\n");

    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    // --- Temporal succession: fractions per dominant species, year by year ---
    println!("== Surface fractions by dominant cover (year end, %) ==");
    print!("{:>5}", "year");
    for l in &labels {
        print!("{l:>8}");
    }
    println!("{:>10}", "veg_tot");

    for year in 1..=RUN_YEARS {
        for _ in 0..365 {
            sim.step();
        }
        let frac = fractions(&sim);
        print!("{year:>5}");
        for f in frac {
            print!("{f:>8.1}");
        }
        let veg_tot: f32 = sim
            .grid()
            .iter()
            .map(|(_, c)| cell_total_vegetation(c))
            .sum();
        println!("{veg_tot:>10.0}");
    }

    // --- Final distribution: category × altitude band ---
    println!("\n== Final distribution: cover × altitude band (% of band) ==");
    let mut matrix = [[0u32; N_CAT]; 5];
    let mut band_totals = [0u32; 5];
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut nan_count = 0u32;
    for (_, cell) in sim.grid().iter() {
        let bi = band_index(cell.elevation);
        matrix[bi][category(cell)] += 1;
        band_totals[bi] += 1;
        for &v in &cell.vegetation {
            if v.is_finite() {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            } else {
                nan_count += 1;
            }
        }
    }

    print!("{:>12}", "band");
    for l in &labels {
        print!("{l:>8}");
    }
    println!("{:>7}", "n");
    for (bi, (label, _, _)) in BANDS.iter().enumerate() {
        let total = f64::from(band_totals[bi].max(1));
        print!("{label:>12}");
        for count in matrix[bi] {
            print!("{:>8.1}", f64::from(count) / total * 100.0);
        }
        println!("{:>7}", band_totals[bi]);
    }

    println!("\n== Biomass bounds (per species) ==");
    println!("  min={min_v:.3}  max={max_v:.3}  NaN={nan_count}");

    // --- Determinism: two short sims, same seed → same biomass/species ---
    let mut a = build_sim();
    let mut b = build_sim();
    for _ in 0..(WARMUP_DAYS + 60) {
        a.step();
        b.step();
    }
    let (ba, bb) = (biomass_per_species(&a), biomass_per_species(&b));
    let mut drift = 0.0_f64;
    for (x, y) in ba.iter().zip(bb.iter()) {
        drift += (x - y).abs();
    }
    println!("\n== Determinism ({} d) ==", WARMUP_DAYS + 60);
    print!("  biomass/species A=[");
    for v in ba {
        print!("{v:.1} ");
    }
    println!("]  drift_abs_total={drift:.2e}");
}
