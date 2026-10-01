// Diagnostic tool: statistical conversions (counts <-> floats,
// percentiles), ubiquitous and benign. Precedent: `diag_wind_rain_distribution`,
// `scale_climate_lapse_rate` use the same allow for stats code.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
//! Climatological initial state (#152), the two instruments of the effort.
//!
//! 1. `diag_aged_world_climatology`: what a **spun-up** world looks like,
//!    read on `frontend/worlds/aged.ckptz` (the 42-year world the embed
//!    boots on, #147): stocks by elevation band and by topographic
//!    wetness index, lakes against the spill levels of the relief, snow,
//!    the humidity profile, the climate normals against elevation, the
//!    vegetation per stratum and the canopy ages. The targets the
//!    climatological t0 is deduced from.
//! 2. `diag_spinup_drift`: the drift of every stock, year by year, on
//!    r30 x 3 seeds from the engine's t0 — the before/after metric of the
//!    issue ("if the model walks away from a realistic state fast, it is
//!    a leak to find, not a t0 to retouch").
//!
//! ```text
//! cargo test --release -p hexsim-core --test diag_initial_state \
//!     -- --ignored --nocapture
//! HEXSIM_DIAG_YEARS=10 cargo test --release ... diag_spinup_drift -- --ignored --nocapture
//! ```

mod common;

use std::collections::BinaryHeap;

use common::build_prod_sim;
use hexsim_core::cell::CellProperties;
use hexsim_core::climate_normals::CellClimateNormals;
use hexsim_core::dynamics::CELL_SPACING_M;
use hexsim_core::erosion::{WORLDGEN_FLOW_CONCENTRATION, accumulate_flow};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::DEFAULT_MAX_CAPACITY_MM;
use hexsim_core::simulation::Simulation;
use hexsim_core::species::{STRATA, STRATUM_COUNT};
use hexsim_core::vegetation::{canopy_cover, is_open_water, stratum_cover};

const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

fn env_u<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn band_index(elev: f32) -> usize {
    BANDS
        .iter()
        .position(|&(_, lo, hi)| elev >= lo && elev < hi)
        .unwrap_or(0)
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    (v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64) as f32
}

fn pct(v: &[f32], p: f64) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    let idx = ((p / 100.0) * (s.len() - 1) as f64).round() as usize;
    s[idx.min(s.len() - 1)]
}

/// Linear regression `y = intercept + slope x`: returns `(intercept,
/// slope, residual std)`.
fn regress(xs: &[f32], ys: &[f32]) -> (f32, f32, f32) {
    let count = xs.len() as f64;
    let mean_x = xs.iter().map(|&v| f64::from(v)).sum::<f64>() / count;
    let mean_y = ys.iter().map(|&v| f64::from(v)).sum::<f64>() / count;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (&xi, &yi) in xs.iter().zip(ys) {
        sxx += (f64::from(xi) - mean_x).powi(2);
        sxy += (f64::from(xi) - mean_x) * (f64::from(yi) - mean_y);
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let intercept = mean_y - slope * mean_x;
    let mut sres = 0.0;
    for (&xi, &yi) in xs.iter().zip(ys) {
        sres += (f64::from(yi) - intercept - slope * f64::from(xi)).powi(2);
    }
    (intercept as f32, slope as f32, (sres / count).sqrt() as f32)
}

/// Topographic wetness index `ln(a / tan beta)` (Beven & Kirkby 1979): `a` =
/// drained area in cells (MFD on the bare relief), `tan beta` = steepest
/// descent to a neighbour, floored at 1 m over the cell spacing.
fn wetness_index(grid: &HexGrid) -> Vec<f32> {
    let (drained, _) = accumulate_flow(grid, WORLDGEN_FLOW_CONCENTRATION);
    let cells = grid.cells_slice();
    (0..cells.len())
        .map(|i| {
            let drop = grid
                .neighbor_indices_toric(i)
                .iter()
                .map(|&j| cells[i].elevation - cells[j].elevation)
                .fold(0.0_f32, f32::max)
                .max(1.0);
            (drained[i] / (drop / CELL_SPACING_M)).ln()
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Flood {
    level_bits: u32,
    idx: usize,
}

impl Ord for Flood {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap on the level: the reversed comparison.
        f32::from_bits(other.level_bits)
            .total_cmp(&f32::from_bits(self.level_bits))
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

impl PartialOrd for Flood {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Spill level of every cell on the torus, priority-flood (Barnes et al.
/// 2014) seeded at the global minimum of the bedrock: `level[i] =
/// max(elevation[i], level of the cell it was reached from)`. Every
/// closed depression but the one holding the global minimum reads its
/// own spill height; the terminal basin stays at its bedrock.
fn spill_levels(grid: &HexGrid) -> Vec<f32> {
    let cells = grid.cells_slice();
    let n = cells.len();
    let mut level = vec![f32::NAN; n];
    let mut heap = BinaryHeap::new();
    let seed = (0..n)
        .min_by(|&a, &b| cells[a].elevation.total_cmp(&cells[b].elevation))
        .expect("non-empty grid");
    level[seed] = cells[seed].elevation;
    heap.push(Flood {
        level_bits: level[seed].to_bits(),
        idx: seed,
    });
    let mut done = vec![false; n];
    while let Some(Flood { level_bits, idx }) = heap.pop() {
        if done[idx] {
            continue;
        }
        done[idx] = true;
        let here = f32::from_bits(level_bits);
        for j in grid.neighbor_indices_toric(idx) {
            if done[j] || !level[j].is_nan() {
                continue;
            }
            level[j] = cells[j].elevation.max(here);
            heap.push(Flood {
                level_bits: level[j].to_bits(),
                idx: j,
            });
        }
    }
    level
}

fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

fn print_stock_tables(sim: &Simulation) {
    let grid = sim.grid();
    let cells = grid.cells_slice();
    let twi = wetness_index(grid);
    let spill = spill_levels(grid);

    println!(
        "cells {}  radius {}  hour_tick {}  (year {}, day {})",
        cells.len(),
        grid.radius(),
        sim.hour_tick(),
        sim.hour_tick() / (24 * 365),
        sim.day_of_year()
    );

    // --- stocks by elevation band ---
    println!(
        "\n== Stocks by elevation band (land cells unless noted; mm per cell, means) ==\n{:<10}{:>6}{:>6} {:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
        "band",
        "n",
        "lake",
        "wl_land",
        "wl/cap",
        "gw",
        "gw/cap",
        "snow",
        "hum_up",
        "hum_sf",
        "cloud",
        "T",
        "lakeSur",
        "lakeDep"
    );
    for (bi, (label, _, _)) in BANDS.iter().enumerate() {
        let in_band: Vec<&CellProperties> = cells
            .iter()
            .filter(|c| band_index(c.elevation) == bi)
            .collect();
        if in_band.is_empty() {
            continue;
        }
        let land: Vec<&&CellProperties> = in_band.iter().filter(|c| !is_open_water(c)).collect();
        let lakes: Vec<&&CellProperties> = in_band.iter().filter(|c| is_open_water(c)).collect();
        let wl_land: Vec<f32> = land.iter().map(|c| c.water_level).collect();
        let wl_cap: Vec<f32> = land
            .iter()
            .map(|c| c.water_level / c.water_capacity.max(1e-6))
            .collect();
        let gw: Vec<f32> = in_band.iter().map(|c| c.groundwater).collect();
        let gw_cap: Vec<f32> = in_band
            .iter()
            .map(|c| c.groundwater / (c.permeability * DEFAULT_MAX_CAPACITY_MM).max(1e-6))
            .collect();
        let snow: Vec<f32> = in_band.iter().map(|c| c.snow_level).collect();
        let hum_up: Vec<f32> = in_band.iter().map(|c| c.humidity_upper).collect();
        let hum_sf: Vec<f32> = in_band.iter().map(|c| c.humidity_surface).collect();
        let cloud: Vec<f32> = in_band.iter().map(|c| c.cloud_water).collect();
        let temp: Vec<f32> = in_band.iter().map(|c| c.temperature).collect();
        let lake_surplus: Vec<f32> = lakes.iter().map(|c| c.water_body_surplus()).collect();
        let lake_depth_m: Vec<f32> = lakes
            .iter()
            .map(|c| c.water_body_surplus() / 1000.0)
            .collect();
        println!(
            "{label:<10}{:>6}{:>6} {:>8.2}{:>8.3}{:>8.2}{:>8.3}{:>8.2}{:>8.2}{:>8.3}{:>8.3}{:>8.2}{:>8.0}{:>8.2}",
            in_band.len(),
            lakes.len(),
            mean(&wl_land),
            mean(&wl_cap),
            mean(&gw),
            mean(&gw_cap),
            mean(&snow),
            mean(&hum_up),
            mean(&hum_sf),
            mean(&cloud),
            mean(&temp),
            mean(&lake_surplus),
            pct(&lake_depth_m, 50.0)
        );
    }

    // --- groundwater and puddle water against the wetness index ---
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|&a, &b| twi[a].total_cmp(&twi[b]));
    println!(
        "\n== Land stocks by topographic wetness index sextile (TWI = ln(a/tanb)) ==\n{:<8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
        "sextile", "twi_lo", "twi_hi", "n", "gw/cap", "wl/cap", "lake%", "moist"
    );
    let n_sex = 6;
    for s in 0..n_sex {
        let lo = s * order.len() / n_sex;
        let hi = (s + 1) * order.len() / n_sex;
        let idx = &order[lo..hi];
        let land: Vec<usize> = idx
            .iter()
            .copied()
            .filter(|&i| !is_open_water(&cells[i]))
            .collect();
        let gw_cap: Vec<f32> = land
            .iter()
            .map(|&i| {
                cells[i].groundwater / (cells[i].permeability * DEFAULT_MAX_CAPACITY_MM).max(1e-6)
            })
            .collect();
        let wl_cap: Vec<f32> = land
            .iter()
            .map(|&i| cells[i].water_level / cells[i].water_capacity.max(1e-6))
            .collect();
        let moist: Vec<f32> = land
            .iter()
            .map(|&i| cells[i].groundwater + cells[i].water_level)
            .collect();
        println!(
            "{s:<8}{:>8.2}{:>8.2}{:>8}{:>8.3}{:>8.3}{:>8.1}{:>8.2}",
            twi[idx[0]],
            twi[idx[idx.len() - 1]],
            idx.len(),
            mean(&gw_cap),
            mean(&wl_cap),
            100.0 * (1.0 - count_f64(land.len()) / count_f64(idx.len())),
            mean(&moist)
        );
    }
    let twi_mean = mean(&twi);
    let gw_cap_all: Vec<f32> = (0..cells.len())
        .filter(|&i| !is_open_water(&cells[i]))
        .map(|i| cells[i].groundwater / (cells[i].permeability * DEFAULT_MAX_CAPACITY_MM).max(1e-6))
        .collect();
    let twi_land: Vec<f32> = (0..cells.len())
        .filter(|&i| !is_open_water(&cells[i]))
        .map(|i| twi[i])
        .collect();
    let (a, b, res) = regress(&twi_land, &gw_cap_all);
    println!(
        "gw/cap = {a:.3} + {b:.4} x TWI (residual std {res:.3}); TWI mean {twi_mean:.2} p5 {:.2} p95 {:.2}",
        pct(&twi, 5.0),
        pct(&twi, 95.0)
    );

    // --- lakes against the spill levels ---
    let mut lake_cells = 0_usize;
    let mut surplus_total = 0.0_f64;
    let mut depressions = 0_usize;
    let mut depression_filled = 0_usize;
    let mut depression_volume_mm = 0.0_f64;
    let mut over_spill = 0_usize;
    let mut under_spill = 0_usize;
    let mut fill_fracs: Vec<f32> = Vec::new();
    for (i, c) in cells.iter().enumerate() {
        let depth_to_spill_mm = (spill[i] - c.elevation) * 1000.0;
        if depth_to_spill_mm > 1.0 {
            depressions += 1;
            depression_volume_mm += f64::from(depth_to_spill_mm);
            let surplus = c.water_body_surplus().max(0.0);
            fill_fracs.push(surplus / depth_to_spill_mm);
            if surplus > 0.9 * depth_to_spill_mm {
                depression_filled += 1;
            }
        }
        if is_open_water(c) {
            lake_cells += 1;
            surplus_total += f64::from(c.water_body_surplus());
            let surface = c.elevation + c.water_body_surplus() / 1000.0;
            if surface > spill[i] + 0.05 {
                over_spill += 1;
            } else if surface < spill[i] - 0.05 {
                under_spill += 1;
            }
        }
    }
    println!(
        "\n== Lakes vs relief ==\nlake cells {lake_cells} ({:.2} % of the map), surplus total {:.0} mm.cell = {:.2} mm/cell map-wide",
        100.0 * count_f64(lake_cells) / count_f64(cells.len()),
        surplus_total,
        surplus_total / count_f64(cells.len())
    );
    println!(
        "depression cells (bedrock below its spill level by > 1 mm): {depressions}, volume to spill {:.0} mm.cell = {:.2} mm/cell; filled >= 90 %: {depression_filled}; fill fraction p10/p50/p90 {:.2}/{:.2}/{:.2}",
        depression_volume_mm,
        depression_volume_mm / count_f64(cells.len()),
        pct(&fill_fracs, 10.0),
        pct(&fill_fracs, 50.0),
        pct(&fill_fracs, 90.0)
    );
    println!(
        "lake surfaces: {over_spill} above their spill level (+5 cm), {under_spill} below (-5 cm), {} within",
        lake_cells - over_spill - under_spill
    );

    // --- vegetation ---
    println!(
        "\n== Vegetation by band (land) ==\n{:<10}{:>6}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
        "band", "n", "herb", "shrub", "tree", "canopy", "bare%", "age_mn", "age_p10", "age_p90"
    );
    for (bi, (label, _, _)) in BANDS.iter().enumerate() {
        let land: Vec<&CellProperties> = cells
            .iter()
            .filter(|c| band_index(c.elevation) == bi && !is_open_water(c))
            .collect();
        if land.is_empty() {
            continue;
        }
        let mut strata = [0.0_f32; STRATUM_COUNT];
        for c in &land {
            for (acc, &s) in strata.iter_mut().zip(STRATA.iter()) {
                *acc += stratum_cover(c, s);
            }
        }
        let n = land.len() as f32;
        let canopy: Vec<f32> = land.iter().map(|c| canopy_cover(c)).collect();
        let bare = land.iter().filter(|c| canopy_cover(c) < 0.05).count();
        let ages: Vec<f32> = land.iter().map(|c| c.stand_age).collect();
        println!(
            "{label:<10}{:>6}{:>8.3}{:>8.3}{:>8.3}{:>8.3}{:>8.1}{:>8.1}{:>8.1}{:>8.1}",
            land.len(),
            strata[0] / n,
            strata[1] / n,
            strata[2] / n,
            mean(&canopy),
            100.0 * count_f64(bare) / count_f64(land.len()),
            mean(&ages),
            pct(&ages, 10.0),
            pct(&ages, 90.0)
        );
    }
}

fn print_normals_tables(sim: &Simulation) {
    let grid = sim.grid();
    let cells = grid.cells_slice();
    let normals: &[CellClimateNormals] = sim.climate_normals();
    if !sim.climate_normals_ready() {
        println!("\n(no climate normals yet)");
        return;
    }
    let elev: Vec<f32> = cells.iter().map(|c| c.elevation).collect();
    let t_mean: Vec<f32> = normals.iter().map(|n| n.t_mean).collect();
    let (a, b, res) = regress(&elev, &t_mean);
    let z_mean = mean(&elev);
    println!(
        "\n== Climate normals ==\nt_mean = {a:.2} + {:.2} C/km x z (residual std {res:.2} C); at mean elevation {z_mean:.0} m: {:.2} C",
        b * 1000.0,
        a + b * z_mean
    );
    println!(
        "{:<10}{:>6}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
        "band",
        "n",
        "t_mean",
        "tmin-tm",
        "tmax-tm",
        "tmn_p10",
        "tmx_p90",
        "moist",
        "mmin/mm",
        "mmax/mm",
        "insol",
        "ins_p10"
    );
    for (bi, (label, _, _)) in BANDS.iter().enumerate() {
        let idx: Vec<usize> = (0..cells.len())
            .filter(|&i| band_index(cells[i].elevation) == bi && !is_open_water(&cells[i]))
            .collect();
        if idx.is_empty() {
            continue;
        }
        let tm: Vec<f32> = idx.iter().map(|&i| normals[i].t_mean).collect();
        let dmin: Vec<f32> = idx
            .iter()
            .map(|&i| normals[i].t_min - normals[i].t_mean)
            .collect();
        let dmax: Vec<f32> = idx
            .iter()
            .map(|&i| normals[i].t_max - normals[i].t_mean)
            .collect();
        let moist: Vec<f32> = idx.iter().map(|&i| normals[i].moisture_mean).collect();
        let rmin: Vec<f32> = idx
            .iter()
            .map(|&i| normals[i].moisture_min / normals[i].moisture_mean.max(1e-6))
            .collect();
        let rmax: Vec<f32> = idx
            .iter()
            .map(|&i| normals[i].moisture_max / normals[i].moisture_mean.max(1e-6))
            .collect();
        let insol: Vec<f32> = idx.iter().map(|&i| normals[i].insolation_mean).collect();
        println!(
            "{label:<10}{:>6}{:>8.2}{:>8.2}{:>8.2}{:>8.2}{:>8.2}{:>8.2}{:>8.3}{:>8.2}{:>8.1}{:>8.1}",
            idx.len(),
            mean(&tm),
            mean(&dmin),
            mean(&dmax),
            pct(&dmin, 10.0),
            pct(&dmax, 90.0),
            mean(&moist),
            mean(&rmin),
            mean(&rmax),
            mean(&insol),
            pct(&insol, 10.0)
        );
    }
    // Aspect anomaly: t_mean residual against the insolation residual.
    let insol: Vec<f32> = normals.iter().map(|n| n.insolation_mean).collect();
    let (ia, ib, _) = regress(&elev, &insol);
    let t_res: Vec<f32> = (0..cells.len())
        .map(|i| t_mean[i] - (a + b * elev[i]))
        .collect();
    let i_res: Vec<f32> = (0..cells.len())
        .map(|i| insol[i] - (ia + ib * elev[i]))
        .collect();
    let (_, k, kres) = regress(&i_res, &t_res);
    println!(
        "insolation = {ia:.1} + {:.3} W/m2 per km x z; t_mean residual = {k:.4} K per W/m2 of insolation residual (residual std {kres:.2} C)",
        ib * 1000.0
    );
    // Moisture normals against the current stocks (are the t0 stocks the climatology?).
    let now_moist: Vec<f32> = cells
        .iter()
        .map(|c| c.groundwater + c.water_level)
        .collect();
    let norm_moist: Vec<f32> = normals.iter().map(|n| n.moisture_mean).collect();
    let (ma, mb, mres) = regress(&now_moist, &norm_moist);
    println!(
        "moisture_mean(normals) = {ma:.2} + {mb:.3} x (gw + wl now) (residual std {mres:.2} mm)"
    );
}

fn load_aged() -> Simulation {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../frontend/worlds/aged.ckptz"
    );
    let out = std::process::Command::new("gzip")
        .args(["-dc", path])
        .output()
        .expect("run gzip");
    assert!(out.status.success(), "gzip -dc {path} failed");
    Simulation::load_state(&out.stdout).expect("aged.ckptz must load")
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_aged_world_climatology() {
    let sim = load_aged();
    println!("=== aged.ckptz, the spun-up world ===");
    print_stock_tables(&sim);
    print_normals_tables(&sim);
}

/// One line per year: map-wide means of every stock, land fractions.
fn print_year_line(sim: &Simulation, year: u64) {
    let cells = sim.grid().cells_slice();
    let n = count_f64(cells.len());
    let mut wl_land = 0.0_f64;
    let mut surplus = 0.0_f64;
    let mut gw = 0.0_f64;
    let mut gw_cap = 0.0_f64;
    let mut snow = 0.0_f64;
    let mut hum = 0.0_f64;
    let mut cloud = 0.0_f64;
    let mut lakes = 0_usize;
    let mut bare = 0_usize;
    let mut strata = [0.0_f64; STRATUM_COUNT];
    let mut canopy = 0.0_f64;
    let mut age = 0.0_f64;
    let mut land = 0_usize;
    for c in cells {
        gw += f64::from(c.groundwater);
        gw_cap += f64::from(c.groundwater / (c.permeability * DEFAULT_MAX_CAPACITY_MM).max(1e-6));
        snow += f64::from(c.snow_level);
        hum += f64::from(c.humidity_upper);
        cloud += f64::from(c.cloud_water);
        if is_open_water(c) {
            lakes += 1;
            surplus += f64::from(c.water_body_surplus());
        } else {
            land += 1;
            wl_land += f64::from(c.water_level);
            for (acc, &s) in strata.iter_mut().zip(STRATA.iter()) {
                *acc += f64::from(stratum_cover(c, s));
            }
            let cc = canopy_cover(c);
            canopy += f64::from(cc);
            if cc < 0.05 {
                bare += 1;
            }
            age += f64::from(c.stand_age);
        }
    }
    let nl = count_f64(land.max(1));
    println!(
        "{year:>4}{:>9.2}{:>9.2}{:>6}{:>8.2}{:>8.3}{:>8.2}{:>8.2}{:>8.3}{:>8.2}{:>8.2} |{:>7.3}{:>7.3}{:>7.3}{:>7.3}{:>7.1}{:>7.1}",
        f64::from(sim.water_budget_total()) / n,
        surplus / n,
        lakes,
        wl_land / nl,
        gw_cap / n,
        gw / n,
        snow / n,
        hum / n,
        cloud / n,
        f64::from(sim.sky_water_total()) / n,
        strata[0] / nl,
        strata[1] / nl,
        strata[2] / nl,
        canopy / nl,
        100.0 * count_f64(bare) / nl,
        age / nl
    );
}

#[test]
#[ignore = "exploratory diagnostic (slow), run with --ignored --nocapture"]
fn diag_spinup_drift() {
    let years: u64 = env_u("HEXSIM_DIAG_YEARS", 3);
    let radius: i32 = env_u("HEXSIM_DIAG_RADIUS", 30);
    for seed in [42_u32, 7, 123] {
        let mut sim = build_prod_sim(seed, radius);
        println!(
            "\n=== spin-up drift, seed {seed}, r{radius}, {years} years (state at Jan 1 of each year; mm per cell map-wide, covers over land) ==="
        );
        println!(
            "{:>4}{:>9}{:>9}{:>6}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8} |{:>7}{:>7}{:>7}{:>7}{:>7}{:>7}",
            "year",
            "total",
            "lakeSur",
            "lakes",
            "wl_land",
            "gw/cap",
            "gw",
            "snow",
            "hum_up",
            "cloud",
            "sky",
            "herb",
            "shrub",
            "tree",
            "canopy",
            "bare%",
            "age"
        );
        print_year_line(&sim, 0);
        for year in 1..=years {
            for _ in 0..365 {
                sim.step();
            }
            print_year_line(&sim, year);
        }
        print_stock_tables(&sim);
        print_normals_tables(&sim);
    }
}

/// Cost of the climatological worldgen: the climate sweep now runs twice
/// on a fresh world (once in `generate_terrain` to seed it, once in
/// `Simulation::new` to prime the normals), so the price of a `reset` is
/// what this prints. `HEXSIM_DIAG_RADIUS` (45).
#[test]
#[ignore = "timing instrument, run with --ignored --nocapture"]
fn diag_construction_cost() {
    use hexsim_core::atmosphere::AtmosphereParams;
    use hexsim_core::groundwater::GroundwaterParams;
    use hexsim_core::hydro::HydroParams;
    use hexsim_core::snow::SnowParams;
    use hexsim_core::temperature::TemperatureParams;
    use hexsim_core::terrain::{TerrainParams, generate_terrain};
    use hexsim_core::wind::WindParams;
    use std::time::Instant;

    let radius: i32 = env_u("HEXSIM_DIAG_RADIUS", 45);
    let mut grid = HexGrid::from_radius(radius);
    let t0 = Instant::now();
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: 42,
            ..TerrainParams::default()
        },
    );
    let gen_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let sim = Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );
    let new_s = t1.elapsed().as_secs_f64();
    println!(
        "r{radius} ({} cells): generate_terrain {gen_s:.2} s, Simulation::new {new_s:.2} s",
        sim.grid().len()
    );
    print_stock_tables(&sim);
    print_normals_tables(&sim);
}
