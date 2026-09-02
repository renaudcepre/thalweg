//! Structural diagnostic: effective vs topological catchments.
//!
//! Tool 5 of the `feat/structural-diag-toolkit`. Targets:
//! - **#24** lake-cell: orphan lakes (no effective tributaries) vs
//!   dominant lakes
//! - identifying **disagreements**: cells where water structurally ends
//!   up *elsewhere* than pure topography predicts. This is the core of
//!   the lake-cell diagnostic: if topo says "water flows through" but
//!   effective says "water stops here", the cell is a trap.
//!
//! Method:
//! - **Effective basins**: for each cell, follow the dominant neighbor
//!   according to `flow_vec_map` averaged over 1 year (weighted by
//!   discharge). The terminus of the chain identifies the effective
//!   basin.
//! - **Topological basins**: steepest descent on raw `cell.elevation`
//!   (no water). Terminus = topological pit.
//! - **Comparison**: for each cell, `terminus_eff` vs `terminus_topo`.
//!   Disagreement => cell "diverted" by effective hydrology.
//!
//! **Style eval** (`scale_tests_eval_style`): `#[ignore]`, no assert.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test diag_basins \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::{HexCoord, hex_direction_to_world};
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
const RUN_DAYS: u64 = 365;
const MAX_WALK: usize = 200;

const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

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

fn dominant_neighbor_dir(flow: (f64, f64)) -> Option<usize> {
    let (fx, fy) = flow;
    let mag2 = fx * fx + fy * fy;
    if mag2 < 1e-12 {
        return None;
    }
    let mut best = 0usize;
    let mut best_dot = f64::NEG_INFINITY;
    for k in 0..6 {
        let (dx, dy) = hex_direction_to_world(k);
        let dot = fx * f64::from(dx) + fy * f64::from(dy);
        if dot > best_dot {
            best_dot = dot;
            best = k;
        }
    }
    if best_dot > 0.0 { Some(best) } else { None }
}

/// Accumulate `flow_vec_map` weighted by discharge over `RUN_DAYS` days,
/// post-warmup. Returns the cumulative `(x, y)` vector per cell.
fn accumulate_effective_flow(sim: &mut Simulation) -> Vec<(f64, f64)> {
    let n = sim.grid().len();
    let mut acc = vec![(0.0_f64, 0.0_f64); n];
    for _ in 0..RUN_DAYS {
        sim.step();
        let discharge = sim.discharge_map();
        let flow_vec = sim.flow_vec_map();
        for i in 0..n {
            let d = f64::from(discharge[i]);
            acc[i].0 += d * f64::from(flow_vec[i].0);
            acc[i].1 += d * f64::from(flow_vec[i].1);
        }
    }
    acc
}

/// For each cell, returns the index of the successor neighbor in the
/// given topology, or `None` if the cell is a terminus (sink).
type SuccessorMap = Vec<Option<usize>>;

fn effective_successors(sim: &Simulation, flow_acc: &[(f64, f64)]) -> SuccessorMap {
    let n = sim.grid().len();
    let mut out: SuccessorMap = vec![None; n];
    for i in 0..n {
        if let Some(k) = dominant_neighbor_dir(flow_acc[i]) {
            // Toric: the flow successor can be on the other side of the
            // seam; the non-toric API used to fabricate a fake terminus
            // on every edge cell whose flow exits the map.
            out[i] = Some(sim.grid().neighbor_indices_toric(i)[k]);
        }
    }
    out
}

fn topological_successors(sim: &Simulation) -> SuccessorMap {
    let n = sim.grid().len();
    let cells = sim.grid().cells_slice();
    let mut out: SuccessorMap = vec![None; n];
    for i in 0..n {
        // Toric: the topological descent crosses the seam (periodic
        // terrain, same topology as hydro).
        let neighbors = sim.grid().neighbor_indices_toric(i);
        let mut best: Option<(usize, f32)> = None;
        let elev_i = cells[i].elevation;
        for j in neighbors {
            let elev_j = cells[j].elevation;
            if elev_j < elev_i && best.is_none_or(|(_, e)| elev_j < e) {
                best = Some((j, elev_j));
            }
        }
        out[i] = best.map(|(idx, _)| idx);
    }
    out
}

/// For each cell, follow the successor until reaching a terminus
/// (None) or a cycle. Returns the terminus found.
fn walk_to_terminus(succ: &SuccessorMap, start: usize) -> usize {
    let mut visited: Vec<bool> = vec![false; succ.len()];
    let mut cur = start;
    for _ in 0..MAX_WALK {
        if visited[cur] {
            return cur;
        }
        visited[cur] = true;
        match succ[cur] {
            Some(next) => cur = next,
            None => return cur,
        }
    }
    cur
}

fn build_terminus_map(succ: &SuccessorMap) -> Vec<usize> {
    (0..succ.len()).map(|i| walk_to_terminus(succ, i)).collect()
}

/// Counts the size of each basin (key = `terminus_idx`, value = number
/// of cells that belong to it).
fn basin_sizes(terminus: &[usize]) -> std::collections::HashMap<usize, u32> {
    let mut out = std::collections::HashMap::new();
    for &t in terminus {
        *out.entry(t).or_insert(0) += 1;
    }
    out
}

fn print_top_basins(
    label: &str,
    sizes: &std::collections::HashMap<usize, u32>,
    coords: &[HexCoord],
    elev: &[f32],
    cells_water_level: &[f32],
) {
    let mut pairs: Vec<(usize, u32)> = sizes.iter().map(|(&k, &v)| (k, v)).collect();
    pairs.sort_by_key(|&(_, size)| std::cmp::Reverse(size));
    eprintln!("\n== Top 10 basins {label} (size = nb of drained cells) ==");
    eprintln!(
        "  {:<5} {:<14} {:>8} {:>8} {:>10}",
        "rank", "terminus(q,r)", "elev_m", "size", "WL_term"
    );
    for (rank, (idx, size)) in pairs.iter().take(10).enumerate() {
        eprintln!(
            "  {:<5} ({:>4},{:>4})    {:>8.0} {:>8} {:>10.3}",
            rank + 1,
            coords[*idx].q,
            coords[*idx].r,
            elev[*idx],
            size,
            cells_water_level[*idx]
        );
    }
}

fn print_disagreements(term_eff: &[usize], term_topo: &[usize], coords: &[HexCoord], elev: &[f32]) {
    let n = term_eff.len();
    let mut disagree: Vec<usize> = (0..n).filter(|&i| term_eff[i] != term_topo[i]).collect();
    let pct = 100.0 * f64::from(u32::try_from(disagree.len()).unwrap_or(u32::MAX))
        / f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    eprintln!(
        "\n== Effective vs topological disagreement: {} / {n} cells ({:.1}%) ==",
        disagree.len(),
        pct
    );
    if disagree.is_empty() {
        return;
    }
    eprintln!("  -> these cells are 'rerouted' by hydrology: the water ends up elsewhere");
    eprintln!("     than what the terrain slope alone predicts.");
    // Sort by descending |elev_terminus_topo - elev_terminus_eff| (rerouting magnitude)
    disagree.sort_by(|&a, &b| {
        let drop_a = (elev[term_topo[a]] - elev[term_eff[a]]).abs();
        let drop_b = (elev[term_topo[b]] - elev[term_eff[b]]).abs();
        drop_b
            .partial_cmp(&drop_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("\n  Top 15 by |Δelev terminus topo vs eff| (most rerouted):");
    eprintln!(
        "  {:<5} {:<14} {:>8} {:<14} {:>8} {:<14} {:>8}",
        "rank", "cell", "elev_c", "term_topo", "elev_to", "term_eff", "elev_te"
    );
    for (rank, &i) in disagree.iter().take(15).enumerate() {
        let tt = term_topo[i];
        let te = term_eff[i];
        eprintln!(
            "  {:<5} ({:>4},{:>4})    {:>8.0} ({:>4},{:>4})    {:>8.0} ({:>4},{:>4})    {:>8.0}",
            rank + 1,
            coords[i].q,
            coords[i].r,
            elev[i],
            coords[tt].q,
            coords[tt].r,
            elev[tt],
            coords[te].q,
            coords[te].r,
            elev[te]
        );
    }
}

fn print_band_agreement(term_eff: &[usize], term_topo: &[usize], elev: &[f32]) {
    eprintln!("\n== Effective/topological agreement by elevation band ==");
    eprintln!(
        "  {:<10} {:>10} {:>14} {:>14}",
        "band", "n_cells", "n_agree", "pct_agree"
    );
    for (label, lo, hi) in BANDS {
        let mut total = 0u32;
        let mut agree = 0u32;
        for (i, &e) in elev.iter().enumerate() {
            if e >= *lo && e < *hi {
                total += 1;
                if term_eff[i] == term_topo[i] {
                    agree += 1;
                }
            }
        }
        let pct = if total > 0 {
            100.0 * f64::from(agree) / f64::from(total)
        } else {
            0.0
        };
        eprintln!("  {label:<10} {total:>10} {agree:>14} {pct:>13.1}%");
    }
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_basins() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    eprintln!(
        "\n=== Basin diag (#24 lake-cell), radius {RADIUS}, seed {SEED}, run {RUN_DAYS} days ==="
    );
    eprintln!(
        "  warmup = {WARMUP_DAYS} days, n_cells = {}",
        sim.grid().len()
    );

    let flow_acc = accumulate_effective_flow(&mut sim);

    let succ_eff = effective_successors(&sim, &flow_acc);
    let succ_topo = topological_successors(&sim);

    let term_eff = build_terminus_map(&succ_eff);
    let term_topo = build_terminus_map(&succ_topo);

    let sizes_eff = basin_sizes(&term_eff);
    let sizes_topo = basin_sizes(&term_topo);

    let coords: Vec<HexCoord> = sim.grid().coords_slice().to_vec();
    let elev: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation)
        .collect();
    let water_level: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.water_level)
        .collect();

    eprintln!("\n== Summary ==");
    eprintln!("  effective basins    : {}", sizes_eff.len());
    eprintln!("  topological basins : {}", sizes_topo.len());

    print_top_basins("effective", &sizes_eff, &coords, &elev, &water_level);
    print_top_basins("topological", &sizes_topo, &coords, &elev, &water_level);
    print_band_agreement(&term_eff, &term_topo, &elev);
    print_disagreements(&term_eff, &term_topo, &coords, &elev);
    eprintln!();
}
