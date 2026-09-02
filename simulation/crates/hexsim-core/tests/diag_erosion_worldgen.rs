//! Diagnostic #105 one-shot: does erosion at GENERATION time properly
//! dissect the terrain, and at what intensity (number of iterations)?
//!
//! The "one-shot" pivot (2026-07-11): erosion is computed once at
//! worldgen then frozen. This instrument sweeps `erosion_iterations`
//! and measures, on the generated terrain (seed 7, r30):
//!   - TRI roughness (max elevation drop to toric neighbors): a
//!     healthy dissection RAISES the p90 / the max (valley walls) even
//!     if the mean barely moves;
//!   - drainage concentration (Gini of flow accumulation on the
//!     resulting terrain): should RISE (rills -> tree);
//!   - closed depressions (future lakes);
//!   - rock conservation Σ(elevation) (erosion doesn't create/destroy
//!     matter: the residual load is deposited in place).
//!
//! Style eval: printed measurements, no target.
//! Lancer : `just diag-tool erosion_worldgen`

mod common;

use hexsim_core::erosion::{accumulate_flow, closed_depression_indices, discharge_gini};
use hexsim_core::grid::HexGrid;
use hexsim_core::terrain::{TerrainParams, generate_terrain};

const RADIUS: i32 = 30;

fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("cell counts fit u32"))
}

struct Stats {
    /// Mean local relief = (max − min elevation in the neighborhood)
    /// per cell. A healthy dissection RAISES it (valleys carved
    /// between ridges); smoothing lowers it. A more accurate metric
    /// than TRI (max drop, which follows the longitudinal profile: a
    /// channel's grading lowers it even while carving laterally).
    local_relief: f64,
    elev_stddev: f64,
    gini_flow: f64,
    depressions: usize,
    total_rock: f64,
}

fn measure(grid: &HexGrid) -> Stats {
    let cells = grid.cells_slice();
    let n = cells.len();
    let mut relief_sum = 0.0_f64;
    for i in 0..n {
        let zi = cells[i].elevation;
        let mut lo = zi;
        let mut hi = zi;
        for &j in &grid.neighbor_indices_toric(i) {
            lo = lo.min(cells[j].elevation);
            hi = hi.max(cells[j].elevation);
        }
        relief_sum += f64::from(hi - lo);
    }
    let mean_elev = cells.iter().map(|c| f64::from(c.elevation)).sum::<f64>() / count_f64(n);
    let var = cells
        .iter()
        .map(|c| {
            let d = f64::from(c.elevation) - mean_elev;
            d * d
        })
        .sum::<f64>()
        / count_f64(n);
    let (discharge, _) = accumulate_flow(grid, 4.0);
    Stats {
        local_relief: relief_sum / count_f64(n),
        elev_stddev: var.sqrt(),
        gini_flow: discharge_gini(&discharge),
        depressions: closed_depression_indices(grid).len(),
        total_rock: cells.iter().map(|c| f64::from(c.elevation)).sum(),
    }
}

/// Acceleration overridable via env for A/B (`HEXSIM_WORLDGEN_ACCEL`):
/// injected into `TerrainParams` (accel is no longer a global env
/// var, it's a per-world field since the one-shot pivot).
fn accel() -> f32 {
    std::env::var("HEXSIM_WORLDGEN_ACCEL")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(hexsim_core::erosion::WORLDGEN_ACCEL_YEARS)
}

fn build(iterations: u32) -> HexGrid {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: 7,
            erosion_iterations: iterations,
            erosion_accel_years: accel(),
            ..TerrainParams::default()
        },
    );
    grid
}

#[test]
#[ignore = "diagnostic #105 one-shot, worldgen dissection vs iterations (seed 7, r30)"]
fn erosion_worldgen_sweep() {
    eprintln!(
        "=== #105 one-shot / worldgen dissection / seed 7 (r{RADIUS}, accel {}) ===",
        accel()
    );
    eprintln!("  iter ; relief_local ; σ_elev ; gini_flow ; basins ; Σrock (sediment export)");
    let base = measure(&build(0));
    for iter in [0_u32, 5, 15, 25, 40, 80] {
        let s = measure(&build(iter));
        let rock_drift = (s.total_rock - base.total_rock) / base.total_rock.abs().max(1.0);
        eprintln!(
            "  {iter:>4} ; {:>11.2} ; {:>6.1} ; {:.4} ; {:>3} ; {:.2e} rel",
            s.local_relief, s.elev_stddev, s.gini_flow, s.depressions, rock_drift
        );
    }
    eprintln!(
        "  Reading: relief_local ↑ + gini_flow ↑ = carved valleys, concentrated drainage (good); \
         relief_local ↓ + σ ↓ sharp = smoothing (melted). Σrock ↓ = rock exported out of the basin \
         (non-conservative incision, accepted at worldgen, see erode_terrain)."
    );
}
