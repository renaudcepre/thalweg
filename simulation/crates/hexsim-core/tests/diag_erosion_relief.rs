//! Diagnostic #105: does erosion SMOOTH or CARVE the relief?
//!
//! Visual feedback "it's all melted" (2026-07-11): is my erosion
//! planing down the terrain (deposition that whitens the ridges) or is
//! the renderer/generator too soft? Anti-pattern #5: we MEASURE instead
//! of squinting at it. Roughness metric = TRI (Terrain Ruggedness
//! Index): per-cell average of the absolute elevation drop to its 6
//! toric neighbors (m). We take it on the GENERATED (raw) terrain, then
//! after N years of erosion, at several accels, and look at the slope
//! distribution (a real fluvial incision ADDS very steep cells, valley
//! walls, even if it flattens the bottoms).
//!
//! Style eval: no target assert, just printed measurements.
//! Lancer : `just diag-tool erosion_relief`

mod common;

use common::build_prod_sim;
use hexsim_core::grid::HexGrid;
use hexsim_core::simulation::Simulation;

const RADIUS: i32 = 30;

/// Average absolute elevation drop to toric neighbors (m) per cell =
/// local roughness. Also returns p50/p90/max of this distribution and
/// the number of "steep" cells (> 20 m drop to a neighbor).
struct Relief {
    tri_mean: f64,
    tri_p50: f32,
    tri_p90: f32,
    tri_max: f32,
    steep_cells: usize,
    elev_stddev: f64,
}

/// usize -> f64 counter without a pedantic cast (test grids fit in
/// u32: r30 = 2791 cells).
fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("cell counts fit u32"))
}

fn measure_relief(grid: &HexGrid) -> Relief {
    let cells = grid.cells_slice();
    let n = cells.len();
    let mut per_cell: Vec<f32> = Vec::with_capacity(n);
    let mut steep = 0usize;
    for i in 0..n {
        let zi = cells[i].elevation;
        let mut worst = 0.0_f32;
        for &j in &grid.neighbor_indices_toric(i) {
            let d = (zi - cells[j].elevation).abs();
            worst = worst.max(d);
        }
        if worst > 20.0 {
            steep += 1;
        }
        per_cell.push(worst);
    }
    let n_f = count_f64(n);
    let tri_mean = per_cell.iter().map(|&x| f64::from(x)).sum::<f64>() / n_f;
    per_cell.sort_by(f32::total_cmp);
    // Percentiles by integer index (no float×len): rank k/100.
    let pct = |num: usize| per_cell[(n * num / 100).min(n - 1)];
    let mean_elev = cells.iter().map(|c| f64::from(c.elevation)).sum::<f64>() / n_f;
    let var = cells
        .iter()
        .map(|c| {
            let d = f64::from(c.elevation) - mean_elev;
            d * d
        })
        .sum::<f64>()
        / n_f;
    Relief {
        tri_mean,
        tri_p50: pct(50),
        tri_p90: pct(90),
        tri_max: *per_cell.last().unwrap(),
        steep_cells: steep,
        elev_stddev: var.sqrt(),
    }
}

fn print_relief(label: &str, r: &Relief) {
    eprintln!(
        "  {label:<22} TRI mean {:.2} m | p50 {:.1} | p90 {:.1} | max {:.1} | \
         steep cells(>20m) {} | elev std-dev {:.1} m",
        r.tri_mean, r.tri_p50, r.tri_p90, r.tri_max, r.steep_cells, r.elev_stddev
    );
}

fn run_years(sim: &mut Simulation, years: u64) {
    for _ in 0..(years * 365) {
        sim.step();
    }
}

#[test]
#[ignore = "diagnostic #105, roughness before/after erosion (seed 7, r30)"]
fn erosion_relief_seed7() {
    eprintln!("=== #105 / relief roughness / seed 7 (r{RADIUS}) ===");
    eprintln!("  Question: does erosion SMOOTH (TRI ↓) or CARVE (TRI ↑, steep cells ↑)?");

    let brut = build_prod_sim(7, RADIUS);
    let r0 = measure_relief(brut.grid());
    print_relief("raw terrain (t0)", &r0);

    for accel in [20.0_f32, 100.0, 500.0] {
        let mut sim = build_prod_sim(7, RADIUS);
        // Live mode opt-in since the one-shot pivot (#105).
        assert!(sim.update_param("erosion.enabled", 1.0));
        assert!(sim.update_param("erosion.accel_years_per_day", accel));
        run_years(&mut sim, 10);
        let r = measure_relief(sim.grid());
        print_relief(&format!("after 10 years (accel {accel:.0})"), &r);
        let steep_delta = i64::try_from(r.steep_cells).expect("fits i64")
            - i64::try_from(r0.steep_cells).expect("fits i64");
        eprintln!(
            "    Δ vs raw: TRI mean {:+.1} % | steep cells {steep_delta:+} | elev std-dev {:+.1} %",
            (r.tri_mean - r0.tri_mean) / r0.tri_mean * 100.0,
            (r.elev_stddev - r0.elev_stddev) / r0.elev_stddev * 100.0,
        );
    }
    eprintln!(
        "  Reading: a sharp TRI ↓ = erosion PLANES (whitening deposit); \
         TRI ↑ or steep cells ↑ = it CARVES valleys (the 'melted' look is then \
         the hex-column rendering / a too-smooth terrain-gen, not erosion)."
    );
}
