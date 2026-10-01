//! Structural diagnostic, distribution of emergent **species** (epic #78,
//! step D: #82; strata and light, #161 steps 1-3). Replaces the former
//! `diag_vegetation_biomes` (abstract biomes).
//!
//! Objective metric **before any tuning** of the niches (`species::SPECIES`)
//! and rates (`VegetationParams`) - no physical balance change without
//! global metrics at scale. Measured over
//! 10 years (daily resolution):
//! - **temporal succession**: surface fraction per dominant species, year
//!   by year (pioneers first, then climax, or steady-state), next to the
//!   mean cover of each stratum (herbs first, then shrubs, then trees);
//! - **species × altitude band distribution** (band × category matrix);
//! - **strata × altitude band**: mean cover per stratum and canopy cover,
//!   the vertical structure the light coupling builds (#161 step 2);
//! - biomass bounding / NaN-free check, per species and per stratum;
//! - determinism (same seed → same total biomass per species).
//!
//! "Dominant" is `vegetation::dominant_species`, the species **seen from
//! the sky** (canopy first, then what its gaps let through): a meadow
//! under an oak canopy counts as oak in the matrix but still shows in the
//! herb cover column. The palette has 16 species over 3 strata; a
//! generalist that wins everywhere (pine before #85) or an understory that
//! never establishes under the trees is what this diag is built to show.
//!
//! **Eval style** (`scale_tests_eval_style`): `#[ignore]`, no assert,
//! structured output meant to be read to calibrate `Species` / `VegetationParams`.
//!
//! ```text
//! cargo test --release -p hexsim-core --test diag_species_distribution \
//!     -- --ignored --nocapture
//! ```

mod common;

use common::species_label;
use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::cell::CellProperties;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::species::{SPECIES, SPECIES_COUNT, STRATA, STRATUM_COUNT, Stratum, species_index};
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::vegetation::{canopy_cover, dominant_species, is_open_water, stratum_cover};
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

const N_BANDS: usize = BANDS.len();

/// Cover categories: open water, bare soil, then one per species.
/// `N_CAT = 2 + number of species`.
const N_CAT: usize = 2 + SPECIES_COUNT;

fn cat_labels() -> [&'static str; N_CAT] {
    let mut out = ["water"; N_CAT];
    out[1] = "bare";
    for (slot, s) in out[2..].iter_mut().zip(SPECIES.iter()) {
        *slot = species_label(s.id);
    }
    out
}

fn stratum_label(s: Stratum) -> &'static str {
    match s {
        Stratum::Herb => "herb",
        Stratum::Shrub => "shrub",
        Stratum::Tree => "tree",
    }
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
fn biomass_per_species(sim: &Simulation) -> [f64; SPECIES_COUNT] {
    let mut out = [0.0_f64; SPECIES_COUNT];
    for (_, cell) in sim.grid().iter() {
        for (o, &v) in out.iter_mut().zip(cell.vegetation.iter()) {
            *o += f64::from(v);
        }
    }
    out
}

/// Mean cover per stratum and mean canopy cover over the **land** cells
/// of a set (open water carries no terrestrial vegetation and would only
/// dilute the means). Both read from the core (`stratum_cover`,
/// `canopy_cover`), never re-summed here (anti-pattern #2).
#[derive(Default, Clone, Copy)]
struct CoverMeans {
    n_land: u32,
    strata: [f64; STRATUM_COUNT],
    canopy: f64,
}

impl CoverMeans {
    fn add(&mut self, cell: &CellProperties) {
        if is_open_water(cell) {
            return;
        }
        self.n_land += 1;
        for (acc, &s) in self.strata.iter_mut().zip(STRATA.iter()) {
            *acc += f64::from(stratum_cover(cell, s));
        }
        self.canopy += f64::from(canopy_cover(cell));
    }

    fn of(sim: &Simulation) -> Self {
        let mut out = Self::default();
        for (_, cell) in sim.grid().iter() {
            out.add(cell);
        }
        out
    }

    /// Prints the stratum means then the canopy mean, 7 chars each.
    fn print_columns(&self) {
        let n = f64::from(self.n_land.max(1));
        for acc in self.strata {
            print!("{:>7.3}", acc / n);
        }
        print!("{:>7.3}", self.canopy / n);
    }
}

fn print_cover_header() {
    for &s in &STRATA {
        print!("{:>7}", stratum_label(s));
    }
    print!("{:>7}", "canopy");
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

    // --- Temporal succession: fractions per dominant species, year by
    // year, then the mean cover per stratum over land cells ---
    println!(
        "== Surface fractions by dominant cover seen from the sky (year end, %) \
         | mean cover over land [0, 1] =="
    );
    print!("{:>5}", "year");
    for l in &labels {
        print!("{l:>7}");
    }
    print!(" |");
    print_cover_header();
    println!();

    for year in 1..=RUN_YEARS {
        for _ in 0..365 {
            sim.step();
        }
        print!("{year:>5}");
        for f in fractions(&sim) {
            print!("{f:>7.1}");
        }
        print!(" |");
        CoverMeans::of(&sim).print_columns();
        println!();
    }

    print_final_distribution(&sim, &labels);
    print_determinism();
}

/// Band × category matrix, stratum cover per band and biomass bounds of
/// the final state.
fn print_final_distribution(sim: &Simulation, labels: &[&str; N_CAT]) {
    // --- Final distribution: category × altitude band ---
    println!("\n== Final distribution: cover × altitude band (% of band) ==");
    let mut matrix = [[0u32; N_CAT]; N_BANDS];
    let mut band_totals = [0u32; N_BANDS];
    let mut band_cover = [CoverMeans::default(); N_BANDS];
    let mut min_v = [f32::INFINITY; SPECIES_COUNT];
    let mut max_v = [f32::NEG_INFINITY; SPECIES_COUNT];
    let mut max_stratum = [f32::NEG_INFINITY; STRATUM_COUNT];
    let mut nan_count = 0u32;
    for (_, cell) in sim.grid().iter() {
        let bi = band_index(cell.elevation);
        matrix[bi][category(cell)] += 1;
        band_totals[bi] += 1;
        band_cover[bi].add(cell);
        for ((&v, lo), hi) in cell.vegetation.iter().zip(&mut min_v).zip(&mut max_v) {
            if v.is_finite() {
                *lo = lo.min(v);
                *hi = hi.max(v);
            } else {
                nan_count += 1;
            }
        }
        for (m, &s) in max_stratum.iter_mut().zip(STRATA.iter()) {
            *m = m.max(stratum_cover(cell, s));
        }
    }

    print!("{:>12}", "band");
    for l in labels {
        print!("{l:>7}");
    }
    println!("{:>7}", "n");
    for (bi, (label, _, _)) in BANDS.iter().enumerate() {
        let total = f64::from(band_totals[bi].max(1));
        print!("{label:>12}");
        for count in matrix[bi] {
            print!("{:>7.1}", f64::from(count) / total * 100.0);
        }
        println!("{:>7}", band_totals[bi]);
    }

    // --- Vertical structure: mean cover per stratum × altitude band ---
    println!("\n== Final mean cover per stratum × altitude band (land cells, [0, 1]) ==");
    print!("{:>12}", "band");
    print_cover_header();
    println!("{:>7}", "n_land");
    for ((label, _, _), cover) in BANDS.iter().zip(band_cover.iter()) {
        print!("{label:>12}");
        cover.print_columns();
        println!("{:>7}", cover.n_land);
    }

    println!("\n== Biomass bounds (per species, per stratum; stratum cover ≤ k_total) ==");
    for (s, (lo, hi)) in SPECIES.iter().zip(min_v.iter().zip(max_v.iter())) {
        print!("  {}=[{lo:.3}, {hi:.3}]", species_label(s.id));
    }
    println!();
    print!(" ");
    for (&s, m) in STRATA.iter().zip(max_stratum.iter()) {
        print!(" max_{}={m:.3}", stratum_label(s));
    }
    println!("  NaN={nan_count}");
}

/// Two short sims, same seed → same biomass per species.
fn print_determinism() {
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
    for (s, v) in SPECIES.iter().zip(ba) {
        print!("{}={v:.1} ", species_label(s.id));
    }
    println!("]  drift_abs_total={drift:.2e}");
}
