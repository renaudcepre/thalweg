//! Structural diagnostic: real water flows and average atmospheric budgets.
//!
//! Tool 3 of the `feat/structural-diag-toolkit` toolkit. Target:
//! - **#69** residual drizzle: where humidity structurally accumulates
//! - **#24** cell-lake cycle: which cells are hydrological sinks
//!   (zero transit) vs transit-cells (rivers)
//!
//! Method:
//! - **`water_level`** side: `discharge_map` (total flow transiting through
//!   a cell) and `edge_flux_map` (per-edge flow, #103) are exposed by the
//!   engine. Pair-to-pair edges are therefore **exact**: no more
//!   approximate reconstruction by projecting `flow_vec` (one source, two
//!   consumers: the front draws the ribbons from the same export).
//! - **Atmosphere** side: without engine instrumentation, there's no
//!   pair-flux. So we report **average stocks** per cell
//!   (`humidity_surface/upper`, `cloud_water`) as a proxy for "where the
//!   water sits".
//!
//! Output: top transit cells, top edges (exact), 5 longest river
//! chains following the strongest-flow edge, discharge/stock distribution
//! by elevation band.
//!
//! **Eval style** (`scale_tests_eval_style`): `#[ignore]`, no assertions.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test diag_water_flows \
//!     -- --ignored --nocapture
//! ```

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

use std::collections::HashMap;
use std::fmt::Write as _;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const WARMUP_DAYS: u64 = 90;
const YEARS: u64 = 2;
const TOTAL_DAYS: u64 = YEARS * 365;

const TOP_N: usize = 20;
const N_RIVER_CHAINS: usize = 5;
const MAX_CHAIN_LEN: usize = 60;

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

/// Direction (0..6) carrying the strongest outgoing edge flow from a cell,
/// from the engine's exact export. `None` if nothing flows.
fn strongest_edge_dir(edges: &[f32; 6]) -> Option<usize> {
    let (dir, &best) = edges
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
    if best > 0.0 { Some(dir) } else { None }
}

struct Accum {
    discharge_total: Vec<f64>,
    flow_x_sum: Vec<f64>,
    flow_y_sum: Vec<f64>,
    mean_water_level: Vec<f64>,
    mean_cloud: Vec<f64>,
    mean_h_upper: Vec<f64>,
    mean_h_surface: Vec<f64>,
    /// Cumulative flux per edge `(src_idx, dst_idx)`, exact export from the
    /// engine.
    edge_flux: HashMap<(usize, usize), f64>,
}

fn collect(sim: &mut Simulation) -> Accum {
    let n = sim.grid().len();
    let mut a = Accum {
        discharge_total: vec![0.0; n],
        flow_x_sum: vec![0.0; n],
        flow_y_sum: vec![0.0; n],
        mean_water_level: vec![0.0; n],
        mean_cloud: vec![0.0; n],
        mean_h_upper: vec![0.0; n],
        mean_h_surface: vec![0.0; n],
        edge_flux: HashMap::new(),
    };

    for _ in 0..TOTAL_DAYS {
        sim.step();
        let discharge = sim.discharge_map();
        let flow_vec = sim.flow_vec_map();
        let edge_flux = sim.edge_flux_map();
        let cells = sim.grid().cells_slice();

        for i in 0..n {
            let d = f64::from(discharge[i]);
            a.discharge_total[i] += d;
            let (fx, fy) = flow_vec[i];
            a.flow_x_sum[i] += f64::from(fx);
            a.flow_y_sum[i] += f64::from(fy);
            a.mean_water_level[i] += f64::from(cells[i].water_level);
            a.mean_cloud[i] += f64::from(cells[i].cloud_water);
            a.mean_h_upper[i] += f64::from(cells[i].humidity_upper);
            a.mean_h_surface[i] += f64::from(cells[i].humidity_surface);

            for (dir_idx, &f) in edge_flux[i].iter().enumerate() {
                if f <= 0.0 {
                    continue;
                }
                // Toric neighborhood: MFD hydro routes water across the
                // seam; the tool must follow, otherwise it truncates the
                // flux edges at the border (fake terminus on the outer ring).
                let dst = sim.grid().neighbor_indices_toric(i)[dir_idx];
                *a.edge_flux.entry((i, dst)).or_insert(0.0) += f64::from(f);
            }
        }
    }

    let n_f = f64::from(u32::try_from(TOTAL_DAYS).unwrap_or(u32::MAX));
    for v in [
        &mut a.mean_water_level,
        &mut a.mean_cloud,
        &mut a.mean_h_upper,
        &mut a.mean_h_surface,
    ] {
        for x in v.iter_mut() {
            *x /= n_f;
        }
    }
    a
}

fn print_top_transit(a: &Accum, coords: &[HexCoord], elev: &[f32]) {
    let n = a.discharge_total.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        a.discharge_total[y]
            .partial_cmp(&a.discharge_total[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("\n== Top {TOP_N} transit cells (discharge accumulated over {TOTAL_DAYS} days) ==");
    eprintln!(
        "  {:<5} {:<14} {:>8} {:>14} {:>10} {:>10}",
        "rank", "coord(q,r)", "elev_m", "discharge_tot", "WL_mean", "fdir(°)"
    );
    for (rank, &idx) in order.iter().take(TOP_N).enumerate() {
        let fx = a.flow_x_sum[idx];
        let fy = a.flow_y_sum[idx];
        let dir_deg = if fx * fx + fy * fy > 1e-12 {
            fy.atan2(fx).to_degrees()
        } else {
            f64::NAN
        };
        eprintln!(
            "  {:<5} ({:>4},{:>4})    {:>8.0} {:>14.1} {:>10.3} {:>10.1}",
            rank + 1,
            coords[idx].q,
            coords[idx].r,
            elev[idx],
            a.discharge_total[idx],
            a.mean_water_level[idx],
            dir_deg
        );
    }
}

fn print_top_edges(a: &Accum, coords: &[HexCoord], elev: &[f32]) {
    let mut edges: Vec<((usize, usize), f64)> = a.edge_flux.iter().map(|(&k, &v)| (k, v)).collect();
    edges.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\n== Top {TOP_N} edges (exact export of edge_flux_map, #103) ==");
    eprintln!(
        "  {:<5} {:<14} → {:<14} {:>10} → {:>10} {:>14}",
        "rank", "src(q,r)", "dst(q,r)", "elev_src", "elev_dst", "edge_flux"
    );
    for (rank, ((src, dst), flux)) in edges.iter().take(TOP_N).enumerate() {
        eprintln!(
            "  {:<5} ({:>4},{:>4})    → ({:>4},{:>4})    {:>10.0} → {:>10.0} {:>14.1}",
            rank + 1,
            coords[*src].q,
            coords[*src].r,
            coords[*dst].q,
            coords[*dst].r,
            elev[*src],
            elev[*dst],
            flux
        );
    }
}

fn print_river_chains(a: &Accum, sim: &Simulation, coords: &[HexCoord], elev: &[f32]) {
    let n = a.discharge_total.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        a.discharge_total[y]
            .partial_cmp(&a.discharge_total[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!("\n== {N_RIVER_CHAINS} longest river chains from top transit cells ==");
    eprintln!(
        "  Follows the edge with the strongest flux, stops on zero flux, revisit, or len>{MAX_CHAIN_LEN}."
    );

    let mut printed = 0usize;
    let mut visited_starts: Vec<bool> = vec![false; n];
    let mut rank_idx = 0;
    while printed < N_RIVER_CHAINS && rank_idx < order.len() {
        let start = order[rank_idx];
        rank_idx += 1;
        if visited_starts[start] || a.discharge_total[start] <= 0.0 {
            continue;
        }
        let chain = build_chain(start, sim, &mut visited_starts);
        if chain.len() < 3 {
            continue;
        }
        printed += 1;
        let elev_drop = elev[chain[0]] - elev[*chain.last().expect("non-empty chain")];
        eprintln!(
            "\n  Chain #{printed} : len={} cells, elev_drop={elev_drop:.0}m",
            chain.len()
        );
        for (k, &idx) in chain.iter().enumerate() {
            eprintln!(
                "    {:>3}. ({:>4},{:>4})  elev={:>5.0}m  WL_mean={:>6.3}  disch_tot={:>8.1}",
                k,
                coords[idx].q,
                coords[idx].r,
                elev[idx],
                a.mean_water_level[idx],
                a.discharge_total[idx]
            );
        }
    }
}

fn build_chain(start: usize, sim: &Simulation, visited: &mut [bool]) -> Vec<usize> {
    let mut chain = vec![start];
    visited[start] = true;
    let mut cur = start;
    let edge_flux = sim.edge_flux_map();
    for _ in 0..MAX_CHAIN_LEN {
        let Some(dir_idx) = strongest_edge_dir(&edge_flux[cur]) else {
            break;
        };
        // Toric: a river chain continues on the other side of the
        // seam (the hydro really does route the water there). The `visited`
        // guard already bounds loops around the torus.
        let next = sim.grid().neighbor_indices_toric(cur)[dir_idx];
        if visited[next] {
            chain.push(next);
            break;
        }
        visited[next] = true;
        chain.push(next);
        cur = next;
    }
    chain
}

fn print_band_summary(a: &Accum, elev: &[f32]) {
    let n = elev.len();
    let mut bins: [Vec<usize>; 5] = Default::default();
    for (i, &e) in elev.iter().enumerate() {
        if let Some(b) = BANDS.iter().position(|(_, lo, hi)| e >= *lo && e < *hi) {
            bins[b].push(i);
        }
    }
    let _ = n;
    eprintln!("\n== Summary by elevation band (mean / p50 / p99) ==");
    eprintln!(
        "  {:<24} {:>20} {:>20} {:>20} {:>20} {:>20}",
        "metric", BANDS[0].0, BANDS[1].0, BANDS[2].0, BANDS[3].0, BANDS[4].0
    );
    eprintln!(
        "  {:<24} {:>20} {:>20} {:>20} {:>20} {:>20}",
        "n cells",
        bins[0].len(),
        bins[1].len(),
        bins[2].len(),
        bins[3].len(),
        bins[4].len()
    );
    let metrics: [(&str, &[f64]); 4] = [
        ("discharge_total (mm)", &a.discharge_total),
        ("water_level mean (mm)", &a.mean_water_level),
        ("cloud_water mean (mm)", &a.mean_cloud),
        ("humidity_upper mean", &a.mean_h_upper),
    ];
    for (label, vals) in metrics {
        let mut row = format!("  {label:<24}");
        for indices in &bins {
            let sub: Vec<f64> = indices.iter().map(|&i| vals[i]).collect();
            let stats = mean_p50_p99_f64(&sub);
            let _ = write!(
                row,
                " {:>5.2}/{:>5.2}/{:>6.2}    ",
                stats.0, stats.1, stats.2
            );
        }
        eprintln!("{row}");
    }
}

fn mean_p50_p99_f64(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let p50 = sorted[(n - 1) / 2];
    let p99 = sorted[((n - 1) * 99) / 100];
    let n_f = f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    let mean: f64 = sorted.iter().sum::<f64>() / n_f;
    (mean, p50, p99)
}

fn print_top_atmospheric_stocks(a: &Accum, coords: &[HexCoord], elev: &[f32]) {
    eprintln!("\n== Top 10 cells by cloud_water mean (where the cloud lingers) ==");
    let n = a.mean_cloud.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        a.mean_cloud[y]
            .partial_cmp(&a.mean_cloud[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!(
        "  {:<5} {:<14} {:>8} {:>10} {:>10} {:>10}",
        "rank", "coord(q,r)", "elev_m", "cloud(mm)", "h_up", "h_surf"
    );
    for (rank, &idx) in order.iter().take(10).enumerate() {
        eprintln!(
            "  {:<5} ({:>4},{:>4})    {:>8.0} {:>10.4} {:>10.3} {:>10.3}",
            rank + 1,
            coords[idx].q,
            coords[idx].r,
            elev[idx],
            a.mean_cloud[idx],
            a.mean_h_upper[idx],
            a.mean_h_surface[idx]
        );
    }
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_water_flows() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    let n = sim.grid().len();
    let coords: Vec<HexCoord> = sim.grid().coords_slice().to_vec();
    let elev: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation)
        .collect();

    eprintln!(
        "\n=== Water flow diag (#69/#24), radius {RADIUS}, seed {SEED}, {YEARS} years ({TOTAL_DAYS} days) ==="
    );
    eprintln!("  warmup = {WARMUP_DAYS} days, n_cells = {n}");

    let a = collect(&mut sim);

    let total_disch: f64 = a.discharge_total.iter().sum();
    let n_active = a.discharge_total.iter().filter(|&&d| d > 0.0).count();
    let n_active_f = f64::from(u32::try_from(n_active.max(1)).unwrap_or(u32::MAX));
    eprintln!(
        "\n== Global summary ==\n  total discharge         : {total_disch:.0} mm·cell\n  active cells            : {n_active} / {n} ({:.1}%)\n  unique edges observed   : {}",
        100.0 * f64::from(u32::try_from(n_active).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(n).unwrap_or(u32::MAX)),
        a.edge_flux.len()
    );
    let _ = n_active_f;

    print_top_transit(&a, &coords, &elev);
    print_top_edges(&a, &coords, &elev);
    print_river_chains(&a, &sim, &coords, &elev);
    print_band_summary(&a, &elev);
    print_top_atmospheric_stocks(&a, &coords, &elev);
    eprintln!();
}
