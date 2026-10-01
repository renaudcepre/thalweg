//! Structural diagnostic: WHY is a cell bare (epic #136, P3 prep)?
//!
//! `diag_species_distribution` counts the bare fraction; this tool says
//! which niche cutoff produces it. For every non-water cell after
//! `RUN_YEARS`, per altitude band: the mean climate normals, the bare
//! fraction, and for the bare cells which lethal limit (frost, heat,
//! drought) excludes each species, or, when no species is excluded, which
//! response term (thermal, water, light) starves the best one.
//!
//! The limits and the responses are the core's (`Species::lethal_cause`,
//! `Species::responses`, `Species::suitability`), never re-derived here
//! (anti-pattern #2: a local copy had drifted, it missed the `> 0.0`
//! drought guard and the light compensation point). Each species reads
//! the light of its own stratum, `insolation_mean ×
//! light_transmittance_below` (#161 step 2), as the vegetation step does.
//!
//! Eval style: `#[ignore]`, `eprintln!`, no assert. Env `HEXSIM_DIAG_SEED`.
//!
//! ```text
//! cargo nextest run -p hexsim-core --run-ignored only \
//!     -E 'binary(diag_bare_anatomy)' --no-capture
//! ```

mod common;

use std::fmt::Write as _;

use common::species_label;
use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::species::{LethalCause, SPECIES, SPECIES_COUNT};
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::vegetation::{
    canopy_cover, dominant_species, is_open_water, light_transmittance_below,
};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const WARMUP_DAYS: u64 = 365;
const RUN_YEARS: u64 = 5;

const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

fn env_seed() -> u32 {
    std::env::var("HEXSIM_DIAG_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(42)
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn band_index(elev: f32) -> usize {
    BANDS
        .iter()
        .position(|&(_, lo, hi)| elev >= lo && elev < hi)
        .unwrap_or(BANDS.len() - 1)
}

#[derive(Default, Clone)]
struct BandStats {
    n: u32,
    bare: u32,
    t_mean: f64,
    t_min: f64,
    t_max: f64,
    moist_mean: f64,
    moist_min: f64,
    insol: f64,
    gw: f64,
    wl: f64,
    cap: f64,
    /// Sum of `canopy_cover` (cover seen from the sky, in [0, 1]).
    cover: f64,
    all_lethal: u32,
    /// Per species, bare cells whose FIRST lethal limit crossed
    /// (`Species::lethal_cause`, order frost, heat, drought) is this one.
    frost: [u32; SPECIES_COUNT],
    heat: [u32; SPECIES_COUNT],
    drought: [u32; SPECIES_COUNT],
    best_suit: f64,
    best_f_temp: f64,
    best_f_water: f64,
    best_f_sun: f64,
    non_lethal_bare: u32,
}

/// Terrain whose horizontal wavelengths are `k` times longer than the
/// default (same seed, same `elevation_scale`), cf. `diag_terrain_scale_climate`.
fn stretched_terrain(seed: u32, k: f64) -> TerrainParams {
    let d = TerrainParams::default();
    TerrainParams {
        seed,
        continent_frequency: d.continent_frequency / k,
        ridge_frequency: d.ridge_frequency / k,
        swiss_frequency: d.swiss_frequency / k,
        permeability_frequency: d.permeability_frequency / k,
        ..d
    }
}

fn run_sim(seed: u32) -> Simulation {
    let k = std::env::var("HEXSIM_DIAG_TERRAIN_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    eprintln!("terrain stretch k={k}");
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(&mut grid, &stretched_terrain(seed, k));
    let hydro = HydroParams {
        slope_full_mobility: env_f32("HEXSIM_DIAG_SLOPE_MOB")
            .unwrap_or(HydroParams::default().slope_full_mobility),
        ..HydroParams::default()
    };
    let gw = GroundwaterParams {
        saturated_conductivity_mm_per_day: env_f32("HEXSIM_DIAG_INFILTRATION")
            .unwrap_or(GroundwaterParams::default().saturated_conductivity_mm_per_day),
        diffusion_rate: env_f32("HEXSIM_DIAG_GW_DIFF")
            .unwrap_or(GroundwaterParams::default().diffusion_rate),
        ..GroundwaterParams::default()
    };
    eprintln!(
        "params: slope_full_mobility={} saturated_conductivity_mm_per_day={} gw_diffusion={}",
        hydro.slope_full_mobility, gw.saturated_conductivity_mm_per_day, gw.diffusion_rate
    );
    let mut sim = Simulation::new(
        grid,
        hydro,
        AtmosphereParams::default(),
        gw,
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams {
            seed,
            ..WindParams::default()
        },
    );
    for _ in 0..(WARMUP_DAYS + RUN_YEARS * 365) {
        sim.step();
    }
    sim
}

fn collect(sim: &Simulation) -> Vec<BandStats> {
    let mut stats = vec![BandStats::default(); BANDS.len()];
    let cells = sim.grid().cells_slice();
    let normals = sim.climate_normals();
    for (cell, n) in cells.iter().zip(normals.iter()) {
        if is_open_water(cell) {
            continue;
        }
        let b = &mut stats[band_index(cell.elevation)];
        b.n += 1;
        b.t_mean += f64::from(n.t_mean);
        b.t_min += f64::from(n.t_min);
        b.t_max += f64::from(n.t_max);
        b.moist_mean += f64::from(n.moisture_mean);
        b.moist_min += f64::from(n.moisture_min);
        b.insol += f64::from(n.insolation_mean);
        b.gw += f64::from(cell.groundwater);
        b.wl += f64::from(cell.water_level);
        b.cap += f64::from(cell.water_capacity);
        b.cover += f64::from(canopy_cover(cell));
        if dominant_species(cell).is_some() {
            continue;
        }
        b.bare += 1;
        let mut any_alive = false;
        let mut best: Option<(f32, f32, f32, f32)> = None;
        for (i, s) in SPECIES.iter().enumerate() {
            match s.lethal_cause(n) {
                Some(LethalCause::Frost) => b.frost[i] += 1,
                Some(LethalCause::Heat) => b.heat[i] += 1,
                Some(LethalCause::Drought) => b.drought[i] += 1,
                None => {
                    any_alive = true;
                    // The light this species' stratum receives under the
                    // layers above it, as `step_vegetation` reads it.
                    let light = n.insolation_mean * light_transmittance_below(cell, s.stratum);
                    let r = s.responses(n, light);
                    let suit = s.suitability(n, light);
                    if best.is_none_or(|(bs, _, _, _)| suit > bs) {
                        best = Some((suit, r.temp, r.water, r.sun));
                    }
                }
            }
        }
        if !any_alive {
            b.all_lethal += 1;
        }
        if let Some((suit, ft, fw, fs)) = best {
            b.non_lethal_bare += 1;
            b.best_suit += f64::from(suit);
            b.best_f_temp += f64::from(ft);
            b.best_f_water += f64::from(fw);
            b.best_f_sun += f64::from(fs);
        }
    }
    stats
}

fn print_inventory(sim: &Simulation) {
    let budget = sim.diagnostics().water_budget;
    let n_cells = f64::from(u32::try_from(sim.grid().cells_slice().len()).expect("fits u32"));
    let gw_capacity: f64 = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| {
            f64::from(c.permeability) * f64::from(hexsim_core::groundwater::DEFAULT_MAX_CAPACITY_MM)
        })
        .sum::<f64>()
        / n_cells;
    eprintln!(
        "== Water inventory per cell (mm): total={:.2} surface={:.2} humidity={:.2} groundwater={:.2} snow={:.2} | gw capacity={:.1} ==",
        f64::from(budget.total) / n_cells,
        f64::from(budget.surface) / n_cells,
        f64::from(budget.humidity) / n_cells,
        f64::from(budget.groundwater) / n_cells,
        f64::from(budget.snow) / n_cells,
        gw_capacity
    );
}

#[test]
#[ignore = "diagnostic tool, run on demand (see module doc)"]
fn diag_bare_anatomy() {
    let seed = env_seed();
    let sim = run_sim(seed);
    let stats = collect(&sim);

    print_inventory(&sim);
    eprintln!("== Bare anatomy (seed {seed}, r{RADIUS}, {RUN_YEARS} y after warmup) ==");
    eprintln!(
        "{:>10} {:>5} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>7} {:>7} {:>6} | {:>6} {:>6} {:>6}",
        "band",
        "n",
        "bare%",
        "cover",
        "Tmean",
        "Tmin",
        "Tmax",
        "Mmean",
        "Mmin",
        "Insol",
        "gw",
        "wl",
        "cap"
    );
    for (&(name, _, _), b) in BANDS.iter().zip(stats.iter()) {
        if b.n == 0 {
            continue;
        }
        let n = f64::from(b.n);
        eprintln!(
            "{:>10} {:>5} {:>6.1} {:>6.2} | {:>6.1} {:>6.1} {:>6.1} | {:>7.2} {:>7.2} {:>6.0} | {:>6.2} {:>6.2} {:>6.1}",
            name,
            b.n,
            f64::from(b.bare) / n * 100.0,
            b.cover / n,
            b.t_mean / n,
            b.t_min / n,
            b.t_max / n,
            b.moist_mean / n,
            b.moist_min / n,
            b.insol / n,
            b.gw / n,
            b.wl / n,
            b.cap / n
        );
    }
    eprintln!();
    eprintln!(
        "== Among BARE cells: % excluded per species by frost / heat / drought \
         (first limit crossed, in that order) =="
    );
    let labels = SPECIES.map(|s| species_label(s.id));
    for (&(name, _, _), b) in BANDS.iter().zip(stats.iter()) {
        if b.bare == 0 {
            continue;
        }
        let nb = f64::from(b.bare);
        let mut line = format!(
            "{name:>10} bare={:<4} all_lethal={:>5.1}% |",
            b.bare,
            f64::from(b.all_lethal) / nb * 100.0
        );
        for (((label, &frost), &heat), &drought) in labels
            .iter()
            .zip(b.frost.iter())
            .zip(b.heat.iter())
            .zip(b.drought.iter())
        {
            write!(
                line,
                " {label}:{:.0}/{:.0}/{:.0}",
                f64::from(frost) / nb * 100.0,
                f64::from(heat) / nb * 100.0,
                f64::from(drought) / nb * 100.0
            )
            .expect("write to String");
        }
        eprintln!("{line}");
    }
    eprintln!();
    eprintln!(
        "== Among bare cells with at least one viable species: mean best suitability and its terms =="
    );
    for (&(name, _, _), b) in BANDS.iter().zip(stats.iter()) {
        if b.non_lethal_bare == 0 {
            continue;
        }
        let k = f64::from(b.non_lethal_bare);
        eprintln!(
            "{name:>10} n={:<4} suit={:.3} f_temp={:.2} f_water={:.2} f_sun={:.2}",
            b.non_lethal_bare,
            b.best_suit / k,
            b.best_f_temp / k,
            b.best_f_water / k,
            b.best_f_sun / k
        );
    }
}
