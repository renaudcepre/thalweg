//! Hydrostatic leveling of multi-hex lakes (#106).
//!
//! **Measured problem** (`tests/diag_lake_flatness.rs`): the surface of a
//! lake spread over several cells is NOT flat, with an `effective_
//! elevation` gap of tens of meters. Leveling used to go through the MFD
//! surplus transfer (`step_hydro_mfd`), whose rate collapsed with the SI
//! pass of #104 (surplus is in meters, so `transfer ∝ flow_rate × Δ(m)`,
//! ~1000x weaker). A lake spanning tens of meters of relief never reaches a
//! single flat level: it stays a staircase of deep puddles.
//!
//! **Fix**: a dedicated pass that, for each connected component of DEEP free
//! water (above a threshold, so as NOT to freeze a flowing river), directly
//! solves the common hydrostatic level `H` and redistributes the surplus, a
//! fill-to-flat that is exact and conservative. Once flat, re-solving gives
//! the same `H`: idempotent, no oscillation.
//!
//! **Strict conservation**: the surplus (free water above `water_
//! capacity`) is redistributed within the component; trapped water
//! (≤ capacity) and cells outside the component don't move. `Σ surplus` is
//! invariant by construction of the solver.

use serde::{Deserialize, Serialize};

use crate::grid::HexGrid;
use crate::units::{Meters, Mm};

#[derive(Clone, Serialize, Deserialize)]
pub struct LakeParams {
    /// Enables leveling. `true` by default: this is a correctness fix (a
    /// lake MUST have a flat surface), not an optional mode.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Minimum free-water depth (surplus mm above `water_capacity`) for a
    /// cell to count as a "lake" to be leveled. Above the thin sheet of a
    /// flowing river (which we do NOT want to freeze into a puddle), below a
    /// true stagnant body of water. A river keeps its dynamic gradient via
    /// the MFD; only full basins get leveled.
    #[serde(default = "default_min_surplus")]
    pub min_surplus_mm: f32,
}

fn default_true() -> bool {
    true
}

fn default_min_surplus() -> f32 {
    // 50 mm of free water: well above the transient surplus of a
    // river cell (a few mm), well below a lake (cm to m).
    50.0
}

impl Default for LakeParams {
    fn default() -> Self {
        Self {
            enabled: true,
            min_surplus_mm: default_min_surplus(),
        }
    }
}

/// Free-water `surplus` level (mm) of a cell (0 if below capacity). Returns
/// `Mm`, not a bare `f32`: every caller that needs it in meters (to compare
/// or add against an elevation) must go through `Mm::to_meters()`, which is
/// exactly the compile-time check #104 needed.
#[inline]
fn surplus_mm(water_level: f32, water_capacity: f32) -> Mm {
    (Mm(water_level) - Mm(water_capacity)).non_negative()
}

/// Levels each connected lake: reads from `current`, writes to `next`.
///
/// Mirrors the other `step_*`: `next` first receives a copy of `current`,
/// then the surplus of lake components is redistributed in place.
pub(crate) fn step_lake_leveling(current: &HexGrid, next: &mut HexGrid, params: &LakeParams) {
    if !params.enabled {
        // Always produce a coherent `next` (the pipeline swaps afterward).
        next.cells_slice_mut()
            .clone_from_slice(current.cells_slice());
        return;
    }

    let cells = current.cells_slice();
    let n = cells.len();
    next.cells_slice_mut().clone_from_slice(cells);

    let is_lake = |i: usize| {
        surplus_mm(cells[i].water_level, cells[i].water_capacity) > Mm(params.min_surplus_mm)
    };

    let mut seen = vec![false; n];
    let mut comp: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..n {
        if seen[start] || !is_lake(start) {
            continue;
        }
        // Connected component (toric neighborhood, consistent with flow).
        comp.clear();
        stack.clear();
        stack.push(start);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            comp.push(i);
            for j in current.neighbor_indices_toric(i) {
                if !seen[j] && is_lake(j) {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if comp.len() < 2 {
            // An isolated puddle is already "flat" (a single surface).
            continue;
        }

        // Total free-water volume of the component, in meters of depth.
        let volume_m: f32 = comp
            .iter()
            .map(|&i| {
                surplus_mm(cells[i].water_level, cells[i].water_capacity)
                    .to_meters()
                    .0
            })
            .sum();

        let level_m = solve_flat_level(&comp, cells, volume_m);

        // Rewrites each cell: trapped water (capacity) + depth up to `H`.
        // A cell whose terrain exceeds `H` falls back to its capacity (it
        // leaves the lake, its surplus has flowed toward the low points).
        let out = next.cells_slice_mut();
        for &i in &comp {
            let depth_m = Meters((level_m - cells[i].elevation).max(0.0));
            out[i].water_level = (Mm(cells[i].water_capacity) + depth_m.to_mm()).0;
        }
    }
}

/// Solves the flat level `H` (m) such that `Σ max(0, H − elev_i) = volume_m`
/// over the component's cells. Incremental fill from the low point: each
/// step covers one more cell. Cells whose terrain exceeds `H` stay dry
/// (their surplus flows downstream).
fn solve_flat_level(comp: &[usize], cells: &[crate::cell::CellProperties], volume_m: f32) -> f32 {
    let mut elevs: Vec<f32> = comp.iter().map(|&i| cells[i].elevation).collect();
    elevs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut level = elevs[0];
    let mut remaining = volume_m;
    // Number of cells already flooded, held as f32 (exact count for any
    // realistic grid size), avoids a usize -> f32 cast.
    let mut covered = 1.0_f32;
    for &e in elevs.iter().skip(1) {
        let step_up = e - level;
        let capacity_here = step_up * covered;
        if remaining <= capacity_here {
            return level + remaining / covered;
        }
        remaining -= capacity_here;
        level = e;
        covered += 1.0;
    }
    // All cells are covered: the remainder spreads out uniformly.
    level + remaining / covered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;

    /// r1 grid (center + 6 neighbors), all filled with deep free water over
    /// a bowl-shaped terrain: after leveling, the surface
    /// (`effective_elevation`) must be flat, and the water conserved.
    fn bowl_grid() -> HexGrid {
        let mut grid = HexGrid::from_radius(1);
        let center = HexCoord::new(0, 0);
        // Bowl: low center, higher ring. All flooded under 2 m of water.
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                let d = coord.distance(center);
                c.elevation = f32::from(i16::try_from(100 + 3 * d).expect("fits")); // 100 or 103
                c.water_capacity = 10.0;
                // Lots of free water: 3000 mm = 3 m of surplus everywhere.
                c.water_level = c.water_capacity + 3000.0;
            }
        }
        grid
    }

    fn total_surplus(grid: &HexGrid) -> f32 {
        grid.cells_slice()
            .iter()
            .map(|c| surplus_mm(c.water_level, c.water_capacity).0)
            .sum()
    }

    #[test]
    fn multi_hex_lake_becomes_flat() {
        let current = bowl_grid();
        let mut next = current.clone();
        step_lake_leveling(&current, &mut next, &LakeParams::default());

        // Surface = effective_elevation. Max/min gap near zero.
        let effs: Vec<f32> = next
            .cells_slice()
            .iter()
            .filter(|c| surplus_mm(c.water_level, c.water_capacity) > Mm(0.0))
            .map(|c| {
                (Meters(c.elevation) + surplus_mm(c.water_level, c.water_capacity).to_meters()).0
            })
            .collect();
        let max = effs.iter().copied().fold(f32::MIN, f32::max);
        let min = effs.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            (max - min) < 1e-3,
            "lake surface must be flat: gap {} m",
            max - min
        );
    }

    #[test]
    fn leveling_conserves_surplus() {
        let current = bowl_grid();
        let before = total_surplus(&current);
        let mut next = current.clone();
        step_lake_leveling(&current, &mut next, &LakeParams::default());
        let after = total_surplus(&next);
        assert!(
            (before - after).abs() < 1e-1,
            "surplus conserved: {before} → {after}"
        );
    }

    #[test]
    fn leveling_moves_water_from_high_to_low() {
        // Bowl: the center (low) must gain water, the ring (high) must lose
        // some, water flows down to the low point up to the flat surface.
        let current = bowl_grid();
        let center = HexCoord::new(0, 0);
        let before_center = current.get(center).unwrap().water_level;
        let mut next = current.clone();
        step_lake_leveling(&current, &mut next, &LakeParams::default());
        let after_center = next.get(center).unwrap().water_level;
        assert!(
            after_center > before_center,
            "the low point must fill: {before_center} → {after_center}"
        );
    }

    #[test]
    fn disabled_is_a_noop() {
        let current = bowl_grid();
        let mut next = current.clone();
        let params = LakeParams {
            enabled: false,
            ..LakeParams::default()
        };
        step_lake_leveling(&current, &mut next, &params);
        for (a, b) in current.cells_slice().iter().zip(next.cells_slice().iter()) {
            assert!((a.water_level - b.water_level).abs() < 1e-6);
        }
    }

    #[test]
    fn shallow_water_below_threshold_untouched() {
        // Thin sheet (< min_surplus): must NOT be leveled (a flowing river,
        // not a lake). Here 20 mm of surplus, threshold 50.
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                let d = coord.distance(HexCoord::new(0, 0));
                c.elevation = f32::from(i16::try_from(100 + 5 * d).expect("fits"));
                c.water_capacity = 10.0;
                c.water_level = c.water_capacity + 20.0; // surplus 20 < threshold 50
            }
        }
        let mut next = grid.clone();
        step_lake_leveling(&grid, &mut next, &LakeParams::default());
        for (a, b) in grid.cells_slice().iter().zip(next.cells_slice().iter()) {
            assert!(
                (a.water_level - b.water_level).abs() < 1e-6,
                "shallow water must not be leveled"
            );
        }
    }
}
