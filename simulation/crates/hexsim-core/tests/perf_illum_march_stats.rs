//! Shadow march instrumentation (`compute_illumination`) - measurement, not
//! behavior test. Answers two questions before any optimization:
//!
//! 1. **Lead 2**: at what distance is max occlusion actually found? (if all
//!    under 20 steps, 64 ceiling is oversized)
//! 2. **Lead 1**: what fraction of cells exit with `over == 0` (no occlusion)?
//!    Potential for precalculated horizon test (`ray_slope >= horizon(cell, dir)`
//!    => entire march skippable).
//!
//! Relief march depends only on terrain + solar position: no need to simulate,
//! replay `temperature.rs` loop on generated terrain.
//!
//! `cargo test --release --test perf_illum_march_stats -- --ignored --nocapture`

use hexsim_core::coord::hex_direction_to_world;
use hexsim_core::dynamics::CELL_SPACING_M;
use hexsim_core::grid::HexGrid;
use hexsim_core::temperature::{TemperatureParams, compute_surface_normals, solar_beam_at_tick};
use hexsim_core::terrain::{TerrainParams, generate_terrain};

const ILLUM_MAX_STEPS: usize = 64;
const ILLUM_FULL_M: f32 = 30.0;

struct HourStats {
    label: String,
    slope: f32,
    dark_pct: f64,    // cos_inc <= 0 (local night on the slope)
    no_occl_pct: f64, // over == 0: march for nothing
    full_pct: f64,    // over >= 30: full shadow
    exit_steps_p50: usize,
    exit_steps_p95: usize,
    argmax_p50: usize,
    argmax_p95: usize,
    exit_dirmax_p50: usize, // steps until exit using max_elev PER DIRECTION
    exit_dirmax_p95: usize,
}

/// Percentile in integer arithmetic (p in %), rounded to nearest.
fn pct(sorted: &[usize], p: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) * p + 50) / 100;
    sorted[idx]
}

/// `100 x num / den` in f64, no usize-to-f64 cast (counts bounded by n <= u32).
fn as_percent(num: usize, den: usize) -> f64 {
    let num = u32::try_from(num).expect("count < 2^32");
    let den = u32::try_from(den.max(1)).expect("count < 2^32");
    100.0 * f64::from(num) / f64::from(den)
}

/// Result of march for a lit cell.
struct CellMarch {
    over: f32,
    argmax: usize,
    exited: usize,
    exited_dirmax: usize,
}

/// Replay `compute_illumination` march for cell `i` and note at what step
/// max occlusion is found and where two exit criteria (global max, max per
/// direction) would have stopped.
fn march_cell(
    grid: &HexGrid,
    i: usize,
    sun_dir: usize,
    ray_slope: f32,
    max_elev: f32,
) -> CellMarch {
    let cells = grid.cells_slice();
    let cy = cells[i].elevation;
    // max upstream per direction, computed on the fly (precalculable off-tick)
    let mut dir_max = f32::NEG_INFINITY;
    let mut idx2 = i;
    for _ in 0..ILLUM_MAX_STEPS {
        idx2 = grid.neighbor_indices_toric(idx2)[sun_dir];
        dir_max = dir_max.max(cells[idx2].elevation);
    }
    let mut out = CellMarch {
        over: 0.0,
        argmax: 0,
        exited: ILLUM_MAX_STEPS,
        exited_dirmax: ILLUM_MAX_STEPS,
    };
    let mut idx = i;
    let mut dist = 0.0_f32;
    let mut found_dirmax = false;
    for step in 1..=ILLUM_MAX_STEPS {
        idx = grid.neighbor_indices_toric(idx)[sun_dir];
        dist += CELL_SPACING_M;
        let ray_h = cy + dist * ray_slope;
        let o = cells[idx].elevation - ray_h;
        if o > out.over {
            out.over = o;
            out.argmax = step;
        }
        if !found_dirmax && ray_h >= dir_max {
            out.exited_dirmax = step;
            found_dirmax = true;
        }
        if ray_h >= max_elev {
            out.exited = step;
            break;
        }
    }
    out.exited_dirmax = out.exited_dirmax.min(out.exited);
    out
}

fn stats_for_hour(
    grid: &HexGrid,
    params: &TemperatureParams,
    tick: u64,
    label: &str,
) -> Option<HourStats> {
    let beam = solar_beam_at_tick(params, tick);
    if beam.s_u <= 0.0 {
        return None;
    }
    let cells = grid.cells_slice();
    let n = cells.len();
    let horiz = (beam.s_e * beam.s_e + beam.s_n * beam.s_n).sqrt();
    if horiz < 1e-6 {
        return None;
    }
    let ray_slope = beam.s_u / horiz;
    let mut sun_dir = 0_usize;
    let (sx, sy) = (beam.s_e, -beam.s_n);
    let mut best = f32::NEG_INFINITY;
    for k in 0..6 {
        let (dx, dy) = hex_direction_to_world(k);
        let dot = dx * sx + dy * sy;
        if dot > best {
            best = dot;
            sun_dir = k;
        }
    }
    let max_elev = cells
        .iter()
        .map(|c| c.elevation)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut dark = 0_usize;
    let mut no_occl = 0_usize;
    let mut full = 0_usize;
    let mut exit_steps = Vec::with_capacity(n);
    let mut exit_dirmax = Vec::with_capacity(n);
    let mut argmaxes = Vec::with_capacity(n);
    for (i, cell) in cells.iter().enumerate() {
        let (ne, nn) = (cell.normal_east, cell.normal_north);
        let n_u = (1.0 - ne * ne - nn * nn).max(0.0).sqrt();
        let cos_inc = (beam.s_e * ne + beam.s_n * nn + beam.s_u * n_u).max(0.0);
        if cos_inc <= 0.0 {
            dark += 1;
            continue;
        }
        let m = march_cell(grid, i, sun_dir, ray_slope, max_elev);
        exit_steps.push(m.exited);
        exit_dirmax.push(m.exited_dirmax);
        if m.over <= 0.0 {
            no_occl += 1;
        } else {
            argmaxes.push(m.argmax);
            if m.over >= ILLUM_FULL_M {
                full += 1;
            }
        }
    }
    exit_steps.sort_unstable();
    exit_dirmax.sort_unstable();
    argmaxes.sort_unstable();
    let lit = n - dark;
    Some(HourStats {
        label: label.to_string(),
        slope: ray_slope,
        dark_pct: as_percent(dark, n),
        no_occl_pct: as_percent(no_occl, lit),
        full_pct: as_percent(full, lit),
        exit_steps_p50: pct(&exit_steps, 50),
        exit_steps_p95: pct(&exit_steps, 95),
        argmax_p50: pct(&argmaxes, 50),
        argmax_p95: pct(&argmaxes, 95),
        exit_dirmax_p50: pct(&exit_dirmax, 50),
        exit_dirmax_p95: pct(&exit_dirmax, 95),
    })
}

#[test]
#[ignore = "measurement, cargo test --release --test perf_illum_march_stats -- --ignored --nocapture"]
fn illum_march_stats() {
    let radius = std::env::var("HEXSIM_PERF_RADIUS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(45_i32);
    let mut grid = HexGrid::from_radius(radius);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: 42,
            ..TerrainParams::default()
        },
    );
    compute_surface_normals(&mut grid);
    let params = TemperatureParams::default();

    // (day, hour, label): winter solstice, equinox, summer solstice,
    // each at grazing sun / mid-height / noon.
    let cases: &[(u64, u64, &str)] = &[
        (354, 9, "winter 09h"),
        (354, 12, "winter 12h"),
        (354, 15, "winter 15h"),
        (80, 8, "equinox 08h"),
        (80, 12, "equinox 12h"),
        (80, 17, "equinox 17h"),
        (172, 6, "summer 06h"),
        (172, 9, "summer 09h"),
        (172, 12, "summer 12h"),
    ];
    eprintln!(
        "r{radius} ({} cells) - step {} m, ceiling {} steps ({} km)",
        grid.len(),
        CELL_SPACING_M,
        ILLUM_MAX_STEPS,
        CELL_SPACING_M * f32::from(u16::try_from(ILLUM_MAX_STEPS).expect("64")) / 1000.0
    );
    eprintln!(
        "{:<14} {:>6} {:>6} {:>7} {:>6} | exit p50/p95 | argmax p50/p95 | exit-dir p50/p95",
        "case", "slope", "night%", "0-occl%", "full%"
    );
    for &(day, hour, label) in cases {
        let tick = day * 24 + hour;
        match stats_for_hour(&grid, &params, tick, label) {
            None => eprintln!("{label:<14} night (s_u <= 0)"),
            Some(s) => eprintln!(
                "{:<14} {:>6.3} {:>5.1}% {:>6.1}% {:>5.1}% |    {:>2} / {:>2}   |    {:>2} / {:>2}    |    {:>2} / {:>2}",
                s.label,
                s.slope,
                s.dark_pct,
                s.no_occl_pct,
                s.full_pct,
                s.exit_steps_p50,
                s.exit_steps_p95,
                s.argmax_p50,
                s.argmax_p95,
                s.exit_dirmax_p50,
                s.exit_dirmax_p95,
            ),
        }
    }
}
