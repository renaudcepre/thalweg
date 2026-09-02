//! Structural diagnostic, cloud clusters over time.
//!
//! Tool 1 of the `feat/structural-diag-toolkit` toolkit. Targets the open
//! questions:
//! - **#68** stationary clouds (how many hours does a cluster live, how
//!   fast does it move?)
//! - **#24** cell-lake cycle (are there cells systematically covered by a
//!   cluster that never leaves?)
//! - **#69** residual drizzle (how many simultaneous clusters, what
//!   fraction of the rain falls inside a cluster vs. outside one?)
//!
//! Method: at every **hourly** tick, the connected components of
//! `cloud_water > threshold` are extracted, matched frame-to-frame by overlap
//! (cf `diagnostics::structural::ClusterTracker`), and the trajectories are
//! aggregated.
//!
//! **Eval style** (project memory `scale_tests_eval_style`): `#[ignore]`,
//! no assert, dense output via `eprintln!`.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test diag_cloud_clusters \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::diagnostics::structural::{
    ClusterTracker, ClusterTrajectory, connected_components_hex,
};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const WARMUP_DAYS: u64 = 90;
const YEARS: u64 = 2;
/// Measurement duration in HOURS (post-warmup).
const TOTAL_HOURS: u64 = YEARS * 365 * 24;

const CLOUD_THRESHOLD_MM: f32 = 0.05;
const OVERLAP_THRESHOLD: f64 = 0.5;
const TOP_ATTRACTORS: usize = 20;

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed: SEED,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    let wind = WindParams {
        seed: SEED,
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

/// Percentile index via integer arithmetic (cf project memory
/// `clippy_strict_patterns`). `p_promille` ∈ [0, 1000].
fn percentile_idx(len: usize, p_promille: u32) -> usize {
    if len == 0 {
        return 0;
    }
    let p = usize::try_from(p_promille).expect("u32 fits usize");
    ((len - 1) * p / 1000).min(len - 1)
}

fn percentiles_u64(values: &mut [u64]) -> Vec<(u32, u64)> {
    values.sort_unstable();
    let labels = [100, 250, 500, 750, 990, 1000];
    labels
        .iter()
        .map(|p| {
            let idx = percentile_idx(values.len(), *p);
            (*p, values.get(idx).copied().unwrap_or(0))
        })
        .collect()
}

fn percentiles_f64(values: &mut [f64]) -> Vec<(u32, f64)> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let labels = [100, 250, 500, 750, 990, 1000];
    labels
        .iter()
        .map(|p| {
            let idx = percentile_idx(values.len(), *p);
            (*p, values.get(idx).copied().unwrap_or(0.0))
        })
        .collect()
}

fn print_percentiles_u64(label: &str, ps: &[(u32, u64)]) {
    eprintln!(
        "  {label:<32} p10={:>6}  p25={:>6}  p50={:>6}  p75={:>6}  p99={:>6}  max={:>6}",
        ps[0].1, ps[1].1, ps[2].1, ps[3].1, ps[4].1, ps[5].1
    );
}

fn print_percentiles_f64(label: &str, ps: &[(u32, f64)]) {
    eprintln!(
        "  {label:<32} p10={:>6.2}  p25={:>6.2}  p50={:>6.2}  p75={:>6.2}  p99={:>6.2}  max={:>6.2}",
        ps[0].1, ps[1].1, ps[2].1, ps[3].1, ps[4].1, ps[5].1
    );
}

struct RunOutput {
    trajectories: Vec<ClusterTrajectory>,
    simultaneity: Vec<u64>,
    rain_total: f64,
    rain_in: f64,
}

fn run_loop(sim: &mut Simulation) -> RunOutput {
    let n_cells = sim.grid().len();
    let mut tracker = ClusterTracker::new(OVERLAP_THRESHOLD);
    let mut simultaneity: Vec<u64> = Vec::with_capacity(usize::try_from(TOTAL_HOURS).unwrap_or(0));
    let mut rain_total_per_cell = vec![0.0_f64; n_cells];
    let mut rain_in_cluster_per_cell = vec![0.0_f64; n_cells];
    let mut in_cluster = vec![false; n_cells];

    for h in 0..TOTAL_HOURS {
        sim.step_hour();
        let clusters =
            connected_components_hex(sim.grid(), |_, c| c.cloud_water > CLOUD_THRESHOLD_MM);
        simultaneity.push(u64::try_from(clusters.len()).unwrap_or(u64::MAX));

        for c in &clusters {
            for &idx in &c.indices {
                in_cluster[idx] = true;
            }
        }
        let precip = sim.last_precipitation();
        for (idx, rec) in precip.iter().enumerate() {
            let rain = f64::from(rec.rain);
            rain_total_per_cell[idx] += rain;
            if in_cluster[idx] {
                rain_in_cluster_per_cell[idx] += rain;
            }
        }
        for c in &clusters {
            for &idx in &c.indices {
                in_cluster[idx] = false;
            }
        }
        tracker.update(h, clusters);
    }

    RunOutput {
        trajectories: tracker.finalize(),
        simultaneity,
        rain_total: rain_total_per_cell.iter().sum(),
        rain_in: rain_in_cluster_per_cell.iter().sum(),
    }
}

fn print_distributions(trajectories: &[ClusterTrajectory], simultaneity: &[u64]) {
    let mut lifetimes: Vec<u64> = trajectories
        .iter()
        .map(ClusterTrajectory::lifetime_ticks)
        .collect();
    let mut displacements: Vec<f64> = trajectories
        .iter()
        .map(ClusterTrajectory::displacement_cells)
        .collect();
    let mut path_lengths: Vec<f64> = trajectories
        .iter()
        .map(ClusterTrajectory::path_length_cells)
        .collect();
    let mut speeds: Vec<f64> = trajectories
        .iter()
        .map(|t| {
            let lt = t.lifetime_ticks();
            if lt == 0 {
                0.0
            } else {
                t.path_length_cells() / f64::from(u32::try_from(lt).unwrap_or(u32::MAX))
            }
        })
        .collect();
    let mut sim_sorted: Vec<u64> = simultaneity.to_vec();
    let sim_ps = percentiles_u64(&mut sim_sorted);
    let life_ps = percentiles_u64(&mut lifetimes);
    let disp_ps = percentiles_f64(&mut displacements);
    let path_ps = percentiles_f64(&mut path_lengths);
    let speed_ps = percentiles_f64(&mut speeds);

    eprintln!("\n== Distributions ==");
    print_percentiles_u64("simultaneity (clusters/h)", &sim_ps);
    print_percentiles_u64("lifetime (h)", &life_ps);
    print_percentiles_f64("net displacement (cells)", &disp_ps);
    print_percentiles_f64("path length (cells)", &path_ps);
    print_percentiles_f64("path/life speed (cells/h)", &speed_ps);
}

fn print_top_attractors(
    trajectories: &[ClusterTrajectory],
    n_cells: usize,
    coords: &[hexsim_core::coord::HexCoord],
    elevations: &[f32],
    water_levels: &[f32],
    water_capacities: &[f32],
) -> Vec<usize> {
    let mut visits_per_cell: Vec<u64> = vec![0; n_cells];
    for t in trajectories {
        for (&idx, &v) in &t.cell_visits {
            visits_per_cell[idx] += u64::from(v);
        }
    }
    let mut order: Vec<usize> = (0..n_cells).collect();
    order.sort_by(|&a, &b| visits_per_cell[b].cmp(&visits_per_cell[a]));
    let total_h_f = f64::from(u32::try_from(TOTAL_HOURS).unwrap_or(u32::MAX));

    eprintln!("\n== Top {TOP_ATTRACTORS} attractor cells (% hours covered by a cluster) ==");
    // is_lake: water_level > water_capacity (canonical threshold, beyond it
    // the surplus is open water, an evaporating surface; cf cell.rs:52-57).
    eprintln!(
        "  {:<5} {:<14} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "rank", "coord(q,r)", "elev_m", "visits", "pct_h", "water_mm", "is_lake"
    );
    let top: Vec<usize> = order.iter().take(TOP_ATTRACTORS).copied().collect();
    for (rank, &idx) in top.iter().enumerate() {
        let v = visits_per_cell[idx];
        let pct = 100.0 * f64::from(u32::try_from(v).unwrap_or(u32::MAX)) / total_h_f;
        let coord = coords[idx];
        let elev = elevations[idx];
        let wl = water_levels[idx];
        let wc = water_capacities[idx];
        let is_lake = wl > wc;
        eprintln!(
            "  {:<5} ({:>4},{:>4})    {:>8.0} {:>10} {:>9.2}% {:>10.2} {:>8}",
            rank + 1,
            coord.q,
            coord.r,
            elev,
            v,
            pct,
            wl,
            if is_lake { "YES" } else { "no" }
        );
    }
    eprintln!();
    top
}

/// Verdict #24: among the low-altitude attractors (`elev < 300 m`), what
/// fraction actually carries open water (`water_level >
/// water_capacity`)? ≥ 50% → cell-lake cycle confirmed by measurement.
fn print_lake_verdict(
    top: &[usize],
    elevations: &[f32],
    water_levels: &[f32],
    water_capacities: &[f32],
) {
    const LOW_ELEV_M: f32 = 300.0;
    let lows: Vec<usize> = top
        .iter()
        .copied()
        .filter(|&i| elevations[i] < LOW_ELEV_M)
        .collect();
    let lakes_low = lows
        .iter()
        .filter(|&&i| water_levels[i] > water_capacities[i])
        .count();
    let any_water_low = lows.iter().filter(|&&i| water_levels[i] > 0.0).count();
    eprintln!("== Verdict #24 (cell-lake cycle) ==");
    eprintln!(
        "  low-alt attractors (<{LOW_ELEV_M:.0} m) : {} / {}",
        lows.len(),
        top.len()
    );
    if lows.is_empty() {
        eprintln!("  → no low-altitude attractor, hypothesis not testable here.");
        return;
    }
    let lows_f = f64::from(u32::try_from(lows.len()).unwrap_or(u32::MAX));
    let pct_lake = 100.0 * f64::from(u32::try_from(lakes_low).unwrap_or(u32::MAX)) / lows_f;
    let pct_any = 100.0 * f64::from(u32::try_from(any_water_low).unwrap_or(u32::MAX)) / lows_f;
    eprintln!("  of which is_lake (wl > wc)        : {lakes_low} ({pct_lake:.1}%)");
    eprintln!("  of which water_level > 0          : {any_water_low} ({pct_any:.1}%)");
    let verdict = if pct_lake >= 50.0 {
        "CONFIRMED: ≥50% of low-alt attractors are lakes"
    } else {
        "INVALIDATED: <50% of low-alt attractors are lakes"
    };
    eprintln!("  → #24 : {verdict}");
    eprintln!();
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_cloud_clusters() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    let n_cells = sim.grid().len();
    let coords: Vec<_> = sim.grid().coords_slice().to_vec();
    let elevations: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation)
        .collect();

    eprintln!(
        "\n=== Cloud cluster diag (#68/#24/#69), radius {RADIUS}, seed {SEED}, {YEARS} years ({TOTAL_HOURS} h) ==="
    );
    eprintln!(
        "  cloud_water threshold = {CLOUD_THRESHOLD_MM:.3} mm   overlap matching = {OVERLAP_THRESHOLD:.2}"
    );
    eprintln!("  warmup = {WARMUP_DAYS} days, n_cells = {n_cells}");

    let out = run_loop(&mut sim);
    let hours_any_cluster = out.simultaneity.iter().filter(|&&n| n > 0).count();
    let mean_simul = if out.simultaneity.is_empty() {
        0.0
    } else {
        let sum: u64 = out.simultaneity.iter().sum();
        f64::from(u32::try_from(sum).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(out.simultaneity.len()).unwrap_or(u32::MAX))
    };

    eprintln!("\n== Inventory ==");
    eprintln!("  total trajectories         : {}", out.trajectories.len());
    eprintln!(
        "  hours with ≥1 cluster      : {hours_any_cluster} / {TOTAL_HOURS} ({:.1}%)",
        100.0 * f64::from(u32::try_from(hours_any_cluster).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(TOTAL_HOURS).unwrap_or(u32::MAX))
    );
    eprintln!("  mean simultaneity          : {mean_simul:.2} clusters/h");

    print_distributions(&out.trajectories, &out.simultaneity);

    let frac_in = if out.rain_total > 0.0 {
        out.rain_in / out.rain_total
    } else {
        0.0
    };
    let rain_out = out.rain_total - out.rain_in;
    eprintln!("\n== Total rain inside clusters vs outside clusters ==");
    eprintln!("  rain_total          : {:>12.1} mm·cell", out.rain_total);
    eprintln!(
        "  rain_in_cluster     : {:>12.1} mm·cell  ({:.1}%)",
        out.rain_in,
        100.0 * frac_in
    );
    eprintln!(
        "  rain_out_cluster    : {rain_out:>12.1} mm·cell  ({:.1}%)",
        100.0 * (1.0 - frac_in)
    );

    let water_levels: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.water_level)
        .collect();
    let water_capacities: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.water_capacity)
        .collect();
    let top = print_top_attractors(
        &out.trajectories,
        n_cells,
        &coords,
        &elevations,
        &water_levels,
        &water_capacities,
    );
    print_lake_verdict(&top, &elevations, &water_levels, &water_capacities);
}
