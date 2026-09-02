//! Structural diagnostics: connected components, cluster tracking.
//!
//! Generic primitives to query the simulation's spatio-temporal structure
//! without touching the engine. Read-only: these helpers never mutate
//! the grid, they observe.
//!
//! First consumer tool: `tests/diag_cloud_clusters.rs` (#68, #24,
//! #69). Module designed to be extended (rivers, fronts, effective basins
//! etc.) as work progresses.

use std::collections::{HashMap, HashSet};

use crate::cell::CellProperties;
use crate::grid::HexGrid;

/// Connected component (4/6-neighborhood hex) extracted from a grid at
/// instant T.
///
/// `centroid_hex` is the arithmetic mean of the axial coordinates (q, r)
/// of the cluster's cells. Used by `ClusterTracker` to measure
/// frame-to-frame displacement, hence kept as `f64` (precision sufficient
/// for any practical radius, and the final `f32` conversion at display time
/// is trivial on the consumer side).
#[derive(Debug, Clone)]
pub struct Cluster {
    pub indices: Vec<usize>,
    pub centroid_hex: (f64, f64),
}

/// Hex connected components over a predicate. The predicate receives the
/// dense cell index + the cell itself, useful for combining scalar criteria
/// (`cloud_water > threshold`) and topological criteria (by elevation band
/// for instance).
///
/// Complexity: `O(n_cells)` (each cell visited at most twice: once by the
/// main scan, once as a neighbor of another).
#[must_use]
pub fn connected_components_hex<F>(grid: &HexGrid, predicate: F) -> Vec<Cluster>
where
    F: Fn(usize, &CellProperties) -> bool,
{
    let cells = grid.cells_slice();
    let coords = grid.coords_slice();
    let n = cells.len();
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..n {
        if visited[start] || !predicate(start, &cells[start]) {
            continue;
        }
        stack.clear();
        stack.push(start);
        visited[start] = true;
        let mut indices: Vec<usize> = Vec::new();
        let mut sum_q = 0.0_f64;
        let mut sum_r = 0.0_f64;

        while let Some(idx) = stack.pop() {
            indices.push(idx);
            sum_q += f64::from(coords[idx].q);
            sum_r += f64::from(coords[idx].r);
            for n_opt in grid.neighbor_indices(idx) {
                if let Some(j) = n_opt
                    && !visited[j]
                    && predicate(j, &cells[j])
                {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        let denom = f64::from(u32::try_from(indices.len()).unwrap_or(u32::MAX));
        let centroid_hex = if denom > 0.0 {
            (sum_q / denom, sum_r / denom)
        } else {
            (0.0, 0.0)
        };
        clusters.push(Cluster {
            indices,
            centroid_hex,
        });
    }
    clusters
}

/// Full trajectory of a cluster reconstructed by frame-to-frame matching.
#[derive(Debug, Clone)]
pub struct ClusterTrajectory {
    pub id: u64,
    pub birth_tick: u64,
    /// Last tick at which the cluster was observed. Lifetime = `death_tick -
    /// birth_tick + 1` ticks.
    pub death_tick: u64,
    /// Sequence of (tick, centroid), one point per observed tick.
    pub centroids: Vec<(u64, (f64, f64))>,
    /// `cell_idx -> number of ticks during which the cell belonged to this
    /// cluster`. Allows extracting "attractor cells" (cells systematically
    /// covered by the same lingering cloud).
    pub cell_visits: HashMap<usize, u32>,
    /// Size (number of cells) of the cluster at each observed tick.
    pub sizes: Vec<u32>,
}

impl ClusterTrajectory {
    #[must_use]
    pub fn lifetime_ticks(&self) -> u64 {
        self.death_tick - self.birth_tick + 1
    }

    /// Hex distance traveled between the first and last centroid, expressed
    /// in cells (Euclidean in axial space, an approximation sufficient
    /// for short trajectories; for winding trajectories,
    /// use `path_length`).
    #[must_use]
    pub fn displacement_cells(&self) -> f64 {
        let Some(first) = self.centroids.first() else {
            return 0.0;
        };
        let Some(last) = self.centroids.last() else {
            return 0.0;
        };
        axial_euclid(first.1, last.1)
    }

    /// Cumulative length of the center-to-center path (sum of the jumps
    /// between consecutive frames). More representative than
    /// `displacement_cells` for a zigzagging cluster.
    #[must_use]
    pub fn path_length_cells(&self) -> f64 {
        self.centroids
            .windows(2)
            .map(|w| axial_euclid(w[0].1, w[1].1))
            .sum()
    }
}

/// ASCII render of a scalar field on the hex grid (pointy-top). The extractor
/// receives the dense index of a cell and returns the value to map.
///
/// 6-level quantization: the palette `' '·:░▒█` is calibrated non-uniformly
/// to isolate zeros (first low threshold) from very-low values (second
/// threshold), useful for spotting at a glance the cells that are absolutely
/// dry in a rain/year field.
///
/// Format: one character + one space per cell, lines offset by `r - r_min`
/// spaces to reproduce the pointy-top projection. Multi-line output
/// including the `min=...  max=...  palette: ...` header.
#[must_use]
pub fn heatmap_ascii<F>(grid: &HexGrid, extractor: F) -> String
where
    F: Fn(usize) -> f32,
{
    use std::fmt::Write;

    let coords = grid.coords_slice();
    if coords.is_empty() {
        return String::new();
    }

    let values: Vec<f32> = (0..coords.len()).map(&extractor).collect();
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for v in &values {
        if v.is_finite() {
            min_v = min_v.min(*v);
            max_v = max_v.max(*v);
        }
    }
    if !min_v.is_finite() {
        min_v = 0.0;
        max_v = 1.0;
    }
    let range = (max_v - min_v).max(1e-9);

    let mut q_min = i32::MAX;
    let mut q_max = i32::MIN;
    let mut r_min = i32::MAX;
    let mut r_max = i32::MIN;
    for c in coords {
        q_min = q_min.min(c.q);
        q_max = q_max.max(c.q);
        r_min = r_min.min(c.r);
        r_max = r_max.max(c.r);
    }

    let mut by_coord: HashMap<(i32, i32), usize> = HashMap::new();
    for (i, c) in coords.iter().enumerate() {
        by_coord.insert((c.q, c.r), i);
    }

    let palette: [char; 6] = [' ', '·', ':', '░', '▒', '█'];
    let mut out = String::new();
    let _ = writeln!(
        out,
        "  min={min_v:.3}   max={max_v:.3}   palette (low→high): '{}{}{}{}{}{}'",
        palette[0], palette[1], palette[2], palette[3], palette[4], palette[5]
    );

    for r in r_min..=r_max {
        let pad = usize::try_from(r - r_min).unwrap_or(0);
        for _ in 0..pad {
            out.push(' ');
        }
        for q in q_min..=q_max {
            if let Some(&idx) = by_coord.get(&(q, r)) {
                let v = values[idx];
                let normalized = ((v - min_v) / range).clamp(0.0, 1.0);
                let bucket = if normalized < 0.05 {
                    0
                } else if normalized < 0.20 {
                    1
                } else if normalized < 0.40 {
                    2
                } else if normalized < 0.60 {
                    3
                } else if normalized < 0.80 {
                    4
                } else {
                    5
                };
                out.push(palette[bucket]);
                out.push(' ');
            } else {
                out.push(' ');
                out.push(' ');
            }
        }
        out.push('\n');
    }
    out
}

fn axial_euclid(a: (f64, f64), b: (f64, f64)) -> f64 {
    // Euclidean distance in axial space converted to pointy-top hex
    // Cartesian: x = q + r/2, y = r * sqrt(3)/2.
    let (aq, ar) = a;
    let (bq, br) = b;
    let ax = aq + ar * 0.5;
    let ay = ar * 0.866_025_403_784_438_6;
    let bx = bq + br * 0.5;
    let by = br * 0.866_025_403_784_438_6;
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

/// Streaming tracker: you feed it a frame's clusters at each tick,
/// it maintains the live trajectories and emits the completed trajectories
/// into an internal queue. Call `finalize` at the end to retrieve all of it.
///
/// Matching: for each live trajectory, we look for the new frame's cluster
/// with the best **overlap** = `|A ∩ B| / min(|A|, |B|)`.
/// Above the threshold (0.5 by default), we associate them; otherwise the
/// trajectory dies and a new one spawns for the orphaned cluster.
///
/// The choice of overlap (vs Jaccard) is deliberate: a cloud that grows or
/// dissipates can double/halve its size within a few hours, and Jaccard
/// asymmetrically penalizes these natural variations.
pub struct ClusterTracker {
    next_id: u64,
    overlap_threshold: f64,
    live: HashMap<u64, ActiveTrajectory>,
    completed: Vec<ClusterTrajectory>,
}

struct ActiveTrajectory {
    id: u64,
    birth_tick: u64,
    last_tick: u64,
    last_indices: HashSet<usize>,
    centroids: Vec<(u64, (f64, f64))>,
    sizes: Vec<u32>,
    cell_visits: HashMap<usize, u32>,
}

impl ClusterTracker {
    #[must_use]
    pub fn new(overlap_threshold: f64) -> Self {
        Self {
            next_id: 0,
            overlap_threshold,
            live: HashMap::new(),
            completed: Vec::new(),
        }
    }

    /// Pushes the clusters observed at tick `tick`. Unmatched trajectories
    /// are moved to `completed`.
    ///
    /// # Panics
    /// Panics if the internal invariant (a matched id must exist in `live`)
    /// is violated, defensive, does not trigger in practice.
    pub fn update(&mut self, tick: u64, clusters: Vec<Cluster>) {
        let new_sets: Vec<HashSet<usize>> = clusters
            .iter()
            .map(|c| c.indices.iter().copied().collect())
            .collect();
        let mut matched_new: Vec<Option<u64>> = vec![None; clusters.len()];
        let mut matched_live: HashSet<u64> = HashSet::new();

        // Greedy: for each live trajectory, we take its best
        // still-free candidate. Stable order for reproducibility.
        let mut live_ids: Vec<u64> = self.live.keys().copied().collect();
        live_ids.sort_unstable();

        for id in &live_ids {
            let live = &self.live[id];
            let mut best: Option<(usize, f64)> = None;
            for (j, ns) in new_sets.iter().enumerate() {
                if matched_new[j].is_some() {
                    continue;
                }
                let inter = live.last_indices.intersection(ns).count();
                if inter == 0 {
                    continue;
                }
                let denom = live.last_indices.len().min(ns.len()).max(1);
                let denom_f = f64::from(u32::try_from(denom).unwrap_or(u32::MAX));
                let inter_f = f64::from(u32::try_from(inter).unwrap_or(u32::MAX));
                let score = inter_f / denom_f;
                if score >= self.overlap_threshold && best.is_none_or(|(_, s)| score > s) {
                    best = Some((j, score));
                }
            }
            if let Some((j, _)) = best {
                matched_new[j] = Some(*id);
                matched_live.insert(*id);
            }
        }

        // Update / spawn
        for (j, c) in clusters.into_iter().enumerate() {
            let size = u32::try_from(c.indices.len()).unwrap_or(u32::MAX);
            if let Some(id) = matched_new[j] {
                let live = self
                    .live
                    .get_mut(&id)
                    .expect("matched id was present in live");
                live.last_tick = tick;
                live.last_indices.clone_from(&new_sets[j]);
                live.centroids.push((tick, c.centroid_hex));
                live.sizes.push(size);
                for &idx in &c.indices {
                    *live.cell_visits.entry(idx).or_insert(0) += 1;
                }
            } else {
                let id = self.next_id;
                self.next_id += 1;
                let mut visits: HashMap<usize, u32> = HashMap::new();
                for &idx in &c.indices {
                    visits.insert(idx, 1);
                }
                self.live.insert(
                    id,
                    ActiveTrajectory {
                        id,
                        birth_tick: tick,
                        last_tick: tick,
                        last_indices: new_sets[j].clone(),
                        centroids: vec![(tick, c.centroid_hex)],
                        sizes: vec![size],
                        cell_visits: visits,
                    },
                );
            }
        }

        // Reaper: unmatched trajectories → completed
        for id in live_ids {
            if !matched_live.contains(&id) {
                let live = self.live.remove(&id).expect("live id reaped from snapshot");
                self.completed.push(traj_from_active(live));
            }
        }
    }

    /// Empties the still-live trajectories into `completed` and returns
    /// all of it. The tracker is unusable after this call.
    #[must_use]
    pub fn finalize(mut self) -> Vec<ClusterTrajectory> {
        for (_, live) in self.live.drain() {
            self.completed.push(traj_from_active(live));
        }
        self.completed
    }
}

fn traj_from_active(a: ActiveTrajectory) -> ClusterTrajectory {
    ClusterTrajectory {
        id: a.id,
        birth_tick: a.birth_tick,
        death_tick: a.last_tick,
        centroids: a.centroids,
        cell_visits: a.cell_visits,
        sizes: a.sizes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;

    fn empty_cell() -> CellProperties {
        CellProperties::default()
    }

    #[test]
    fn connected_components_isolated_singletons() {
        let mut grid = HexGrid::from_radius(2);
        let mut a = empty_cell();
        a.cloud_water = 1.0;
        let mut b = empty_cell();
        b.cloud_water = 1.0;
        // Two non-adjacent cells: (2,0) and (-2,0), hex distance = 4.
        grid.get_mut(HexCoord::new(2, 0))
            .map(|c| *c = a)
            .expect("cell (2,0) exists");
        grid.get_mut(HexCoord::new(-2, 0))
            .map(|c| *c = b)
            .expect("cell (-2,0) exists");
        let clusters = connected_components_hex(&grid, |_, c| c.cloud_water > 0.0);
        assert_eq!(clusters.len(), 2);
        assert!(clusters.iter().all(|c| c.indices.len() == 1));
    }

    #[test]
    fn connected_components_merges_adjacent() {
        let mut grid = HexGrid::from_radius(2);
        // (0,0) and its neighbor (1,0) are adjacent → 1 cluster.
        for coord in &[HexCoord::new(0, 0), HexCoord::new(1, 0)] {
            let mut cell = empty_cell();
            cell.cloud_water = 1.0;
            grid.get_mut(*coord)
                .map(|c| *c = cell)
                .expect("cell exists");
        }
        let clusters = connected_components_hex(&grid, |_, c| c.cloud_water > 0.0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].indices.len(), 2);
    }

    #[test]
    fn tracker_continuity_via_overlap() {
        let mut tracker = ClusterTracker::new(0.5);
        let c0 = Cluster {
            indices: vec![0, 1, 2, 3],
            centroid_hex: (0.0, 0.0),
        };
        tracker.update(0, vec![c0]);
        // Frame +1: 3 out of 4 cells shared (overlap = 3/3 = 1.0)
        let c1 = Cluster {
            indices: vec![1, 2, 3, 4],
            centroid_hex: (1.0, 0.0),
        };
        tracker.update(1, vec![c1]);
        // Frame +2: no overlap → the trajectory dies, a new one
        // is born.
        let c2 = Cluster {
            indices: vec![10, 11],
            centroid_hex: (5.0, 0.0),
        };
        tracker.update(2, vec![c2]);
        let trajs = tracker.finalize();
        assert_eq!(trajs.len(), 2);
        let alive = trajs
            .iter()
            .find(|t| t.lifetime_ticks() == 2)
            .expect("expected 2-tick trajectory");
        assert_eq!(alive.birth_tick, 0);
        assert_eq!(alive.death_tick, 1);
    }
}
