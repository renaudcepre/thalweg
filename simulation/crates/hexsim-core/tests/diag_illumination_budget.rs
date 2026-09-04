//! Structural diagnostic: annual solar budget of the illumination pass
//! on a procedural world, by elevation band.
//!
//! Answers "how much of the clear-sky beam does the relief actually
//! let through?" without running the full simulation: the terrain is
//! generated, its normals computed, and `compute_illumination` is
//! swept over one year of hourly sun positions (clouds = 0). For each
//! band we print the mean of `flux_factor / s_u` (1 = the horizontal
//! beam of a flat cell), split into its aspect part (`HEXSIM_ILLUM_KO`
//! equivalent: raymarch off) and the full pass (relief occlusion on),
//! plus the slope distribution that drives both.
//!
//! Built for the 2026-09-02 investigation of the lapse-rate collapse
//! at `CELL_SPACING_M = 130` (JOURNAL). Eval style: `#[ignore]`,
//! `eprintln!`, no assert.
//!
//! ```text
//! cargo test --release -p hexsim-core --test diag_illumination_budget \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::dynamics::CELL_SPACING_M;
use hexsim_core::grid::HexGrid;
use hexsim_core::temperature::{
    IllumCache, TemperatureParams, compute_illumination_cached, compute_surface_normals,
    solar_beam_at_tick,
};
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::time::TICKS_PER_YEAR;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const BANDS: [(&str, f32, f32); 5] = [
    ("<200", f32::NEG_INFINITY, 200.0),
    ("200-500", 200.0, 500.0),
    ("500-900", 500.0, 900.0),
    ("900-1400", 900.0, 1400.0),
    (">=1400", 1400.0, f32::INFINITY),
];

struct BandAcc {
    cells: usize,
    slope_sum: f64,
    slope_max: f32,
    beam_sum: f64,
    flux_sum: f64,
    /// December-February (index 0) and June-August (index 1) sums.
    beam_season: [f64; 2],
    flux_season: [f64; 2],
}

/// 0 = winter (Dec-Feb), 1 = summer (Jun-Aug), None otherwise.
fn season_of(hour_tick: u64) -> Option<usize> {
    let day = hexsim_core::time::day_of_year(hour_tick);
    match day {
        0..=58 | 334..=365 => Some(0),
        151..=242 => Some(1),
        _ => None,
    }
}

fn band_of(elevation: f32) -> usize {
    BANDS
        .iter()
        .position(|(_, lo, hi)| elevation >= *lo && elevation < *hi)
        .expect("bands cover the real line")
}

#[test]
#[ignore = "diagnostic tool, run on demand (see module doc)"]
fn illumination_budget_by_band() {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: SEED,
            ..TerrainParams::default()
        },
    );
    compute_surface_normals(&mut grid);
    let params = TemperatureParams::default();
    let mut cache = IllumCache::default();
    cache.ensure(&grid);

    let n = grid.len();
    let band_idx: Vec<usize> = grid
        .cells_slice()
        .iter()
        .map(|c| band_of(c.elevation))
        .collect();
    let mut acc: Vec<BandAcc> = BANDS
        .iter()
        .map(|_| BandAcc {
            cells: 0,
            slope_sum: 0.0,
            slope_max: 0.0,
            beam_sum: 0.0,
            flux_sum: 0.0,
            beam_season: [0.0; 2],
            flux_season: [0.0; 2],
        })
        .collect();
    for (i, c) in grid.cells_slice().iter().enumerate() {
        let (ne, nn) = (c.normal_east, c.normal_north);
        let n_u = (1.0 - ne * ne - nn * nn).max(0.0).sqrt();
        let slope_deg = n_u.acos().to_degrees();
        let a = &mut acc[band_idx[i]];
        a.cells += 1;
        a.slope_sum += f64::from(slope_deg);
        a.slope_max = a.slope_max.max(slope_deg);
    }

    let mut flux_factor = Vec::with_capacity(n);
    let mut illumination = Vec::with_capacity(n);
    let mut daylight_hours = 0_u64;
    for hour_tick in 0..TICKS_PER_YEAR {
        let beam = solar_beam_at_tick(&params, hour_tick);
        if beam.s_u <= 0.0 {
            continue;
        }
        daylight_hours += 1;
        let season = season_of(hour_tick);
        compute_illumination_cached(
            &grid,
            &beam,
            params.cloud_albedo_coef,
            1500.0,
            &cache,
            &mut flux_factor,
            &mut illumination,
        );
        for i in 0..n {
            let a = &mut acc[band_idx[i]];
            a.beam_sum += f64::from(beam.s_u);
            a.flux_sum += f64::from(flux_factor[i]);
            if let Some(k) = season {
                a.beam_season[k] += f64::from(beam.s_u);
                a.flux_season[k] += f64::from(flux_factor[i]);
            }
        }
    }

    eprintln!(
        "\n== Illumination budget, seed {SEED}, radius {RADIUS}, spacing {CELL_SPACING_M} m, \
         {daylight_hours} daylight hours, clouds = 0 =="
    );
    eprintln!(
        "  {:<10} {:>6} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "band", "cells", "slope_mean", "slope_max", "flux/flat", "winter", "summer"
    );
    for (b, a) in BANDS.iter().zip(&acc) {
        if a.cells == 0 {
            continue;
        }
        let cells_f = f64::from(u32::try_from(a.cells).expect("fits u32"));
        eprintln!(
            "  {:<10} {:>6} {:>9.1}° {:>9.1}° {:>12.3} {:>10.3} {:>10.3}",
            b.0,
            a.cells,
            a.slope_sum / cells_f,
            a.slope_max,
            a.flux_sum / a.beam_sum,
            a.flux_season[0] / a.beam_season[0],
            a.flux_season[1] / a.beam_season[1]
        );
    }
    let total_flux: f64 = acc.iter().map(|a| a.flux_sum).sum();
    let total_beam: f64 = acc.iter().map(|a| a.beam_sum).sum();
    eprintln!("  map mean flux/flat = {:.3}", total_flux / total_beam);
}
