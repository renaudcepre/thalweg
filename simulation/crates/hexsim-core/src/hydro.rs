use serde::{Deserialize, Serialize};

use crate::coord::{hex_direction_to_world, opposite_direction};
use crate::dynamics::{CELL_SPACING_M, STEEP_SLOPE_GRADE};
use crate::grid::HexGrid;
use crate::par::{for_each_chunk_mut, for_each_chunk_mut2, for_each_chunk_mut6, sum_dir_out};
use crate::phase_timing::{HydroStepTimings, elapsed_s, mark};

/// Average outgoing flux vector per cell, in world coordinates.
/// Accumulates transfers weighted by direction for each substep.
/// Indexed by `HexGrid::cell_index` (size = `grid.len()`).
pub type FlowVecMap = Vec<(f32, f32)>;

/// Flux map: for each cell, quantity of water sent to its neighbors
/// during a substep (sum over all directions in MFD).
/// Indexed by `HexGrid::cell_index` (size = `grid.len()`).
pub type FluxMap = Vec<f32>;

/// Outgoing flux per edge: for each cell, quantity of water sent to each
/// of its 6 neighbors (order `coord::DIRECTIONS`) during a substep.
/// Since the midpoint of an edge is shared with the neighbor, this export
/// is enough for a consumer to draw continuous flux ribbons from one hex
/// to the next without network detection (#103). `flux_out[i] ==
/// edge_flux[i].sum()` by construction. Indexed by `HexGrid::cell_index`
/// (size = `grid.len()`).
pub type EdgeFluxMap = Vec<[f32; 6]>;

/// The three flux maps produced by the daily hydro slice, grouped into a
/// single handle: they always travel together (same reset/accumulation
/// lifecycle) toward the snapshot. Fields as slices: a `&Vec` (engine
/// maps) as well as a `&[]` (test probes) coerce into it.
pub struct HydroMaps<'a> {
    pub discharge: &'a [f32],
    pub flow_vec: &'a [(f32, f32)],
    pub edge_flux: &'a [[f32; 6]],
}

/// Scratch buffers for the two-phase scatter -> gather MFD routing
/// (r250 perf effort, chunk C1), owned by the caller and reused every
/// call to [`step_hydro_mfd_into`]: content between two calls is
/// undefined, same convention as `atmosphere::AtmoScratch`.
/// `dir_out[d][i]` is the amount cell `i` routes toward its toric
/// neighbor in direction `d` (`coord::DIRECTIONS[d]`) this substep,
/// filled by a parallel per-source outflow pass
/// (`fill_hydro_outflow`) and read back by the following
/// per-destination gather pass via `coord::opposite_direction`.
pub struct HydroScratch {
    pub dir_out: [Vec<f32>; 6],
    /// This substep's `hydro` sub-phase durations, filled by
    /// `step_hydro_mfd_into` and read back by the caller (`Simulation`)
    /// right after the call. See [`HydroStepTimings`] for why this isn't
    /// cumulative.
    pub step_timings: HydroStepTimings,
}

impl HydroScratch {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            dir_out: std::array::from_fn(|_| Vec::with_capacity(n)),
            step_timings: HydroStepTimings::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HydroParams {
    /// Water depth (mm) transferred per substep and per meter of
    /// effective elevation difference: `transfer ≈ flow_rate × Σ delta`
    /// with `delta` in m since #104. Dimensionally an mm/m inherited from
    /// the hybrid unit space, not a pure ratio: a transition coefficient
    /// **assumed as such** (project SI convention, "unit mix marked
    /// explicitly"). Its conversion to SI is not covered by #105: that one
    /// delivered the erosion stream power, a phenomenon distinct from the
    /// hydro slice's transport. A future conversion of `flow_rate` will
    /// therefore have to derive its own flux, without inheriting anything
    /// from #105.
    /// The historical CFL bound ~1/7 ≈ 0.14 dated from the hybrid where
    /// water-against-water leveling re-transferred this fraction of the
    /// imbalance in mm; in SI this same leveling transfers 1000× less
    /// (the A→B→A overshoot is structurally impossible there) and it is
    /// the cap on the mobile stock that bounds terrain-driven transfers.
    pub flow_rate: f32,
    /// Slope (m, raw elevation delta with the lowest neighbor) above which
    /// all sub-capacity water becomes mobile. Below it: proportional
    /// fraction. Derived from `CELL_SPACING_M` (see
    /// `dynamics::STEEP_SLOPE_GRADE`) to stay the same physical slope
    /// regardless of the engine's resolution → water no longer stagnates
    /// as a "trapped puddle" beyond this threshold.
    pub slope_full_mobility: f32,
    /// Concentration exponent for the MFD split (Tarboton D-inf).
    /// `raw_flow_i ∝ delta_i^flow_concentration`:
    ///   1.0 = uniform MFD (pure dispersion).
    ///   2.0-4.0 = weighted MFD, rivers concentrate toward the steepest slope.
    ///   → ∞ = D8 (all to the steepest, no splitting).
    /// 2.0 is the classic compromise in numerical hydrology.
    pub flow_concentration: f32,
}

impl Default for HydroParams {
    fn default() -> Self {
        Self {
            flow_rate: 0.12,
            slope_full_mobility: STEEP_SLOPE_GRADE * CELL_SPACING_M,
            flow_concentration: 6.0,
        }
    }
}

/// Discharge of a cell: total flux out during the tick (accumulated over
/// 8 substeps). In symmetric MFD, no DAG, `discharge = flux_out`, period.
/// Indexed by `HexGrid::cell_index` (size = `grid.len()`).
pub type DischargeMap = Vec<f32>;

// Computes the total water in the grid (useful to check conservation).
#[must_use]
pub fn total_water(grid: &HexGrid) -> f32 {
    grid.iter().map(|(_, cell)| cell.water_level).sum()
}

/// Purely emergent local flow: no precomputed `FlowMap`.
///
/// For each cell, only the surplus above `water_capacity` is mobile
/// (trapped water stays sub-hex). The mobile part is split among all
/// neighbors whose `effective_elevation` is lower, proportionally to the
/// effective slope difference. A CFL cap guarantees we never send more
/// than the available mobile water, a strict stability and conservation
/// condition.
///
/// Returns `(flux_out, flow_vec)` where:
/// - `flux_out[c]` = sum of outgoing transfers from `c` during this step
/// - `flow_vec[c]` = world vector of the average outgoing flux, weighted
///   by transfer
///
/// No `drainage` parameter: the topology emerges dynamically from
/// `effective_elevation` on every call. Lakes appear on their own when
/// the surplus spreads out and equalizes by gradient.
pub fn step_hydro_mfd(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &HydroParams,
) -> (FluxMap, FlowVecMap) {
    let n = current.len();
    let mut flux_out: FluxMap = vec![0.0; n];
    let mut flow_vec: FlowVecMap = vec![(0.0, 0.0); n];
    let mut edge_flux: EdgeFluxMap = vec![[0.0; 6]; n];
    let mut scratch = HydroScratch::new(n);
    step_hydro_mfd_into(
        current,
        next,
        params,
        &mut scratch,
        &mut flux_out,
        &mut flow_vec,
        &mut edge_flux,
    );
    (flux_out, flow_vec)
}

/// Zero-malloc variant: writes into the provided buffers (resize; every
/// element is unconditionally overwritten by the phases below, so no
/// separate reset-to-0 pass is needed).
///
/// Two-phase scatter -> gather (r250 perf effort, chunk C1): the
/// historical single serial loop computed, per source cell, the MFD
/// split toward its downhill neighbors and immediately scattered each
/// share into `next_cells[j].water_level` — a write into ANOTHER cell's
/// output, unsafe to hand to a chunked parallel iterator as-is. Phase 1
/// (`fill_hydro_outflow`) computes that same per-source split into
/// `scratch.dir_out`, reading only `current` (untouched by this whole
/// function): fully independent per source, parallelizable
/// (`par::for_each_chunk_mut6`). `flux_out`/`flow_vec` are pure
/// per-source aggregates of the same split
/// (`fill_hydro_source_aggregates`), and the per-edge history is the
/// same buffer transposed to the `AoS` layout its consumers expect
/// (`fill_hydro_edge_flux`) — no physics recomputed for either.
/// Phase 2 (`gather_hydro_water`) gathers the routed water back per
/// destination cell via `coord::opposite_direction`.
pub fn step_hydro_mfd_into(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &HydroParams,
    scratch: &mut HydroScratch,
    flux_out: &mut FluxMap,
    flow_vec: &mut FlowVecMap,
    edge_flux_out: &mut EdgeFluxMap,
) {
    let n = current.len();
    flux_out.resize(n, 0.0);
    flow_vec.resize(n, (0.0, 0.0));
    edge_flux_out.resize(n, [0.0; 6]);

    // Fresh per-substep sub-phase durations (`HydroStepTimings` carries
    // no meaning between calls, cf. its doc): the caller reads this back
    // right after the call and accumulates it into the cumulative
    // `PhaseTimings::hydro_*` fields.
    scratch.step_timings = HydroStepTimings::default();

    let t0 = mark();
    for dir in &mut scratch.dir_out {
        dir.clear();
        dir.resize(n, 0.0);
    }
    fill_hydro_outflow(current, params, &mut scratch.dir_out);
    scratch.step_timings.outflow += elapsed_s(t0);
    let t0 = mark();
    fill_hydro_source_aggregates(&scratch.dir_out, flux_out, flow_vec);
    scratch.step_timings.aggregates += elapsed_s(t0);
    let t0 = mark();
    fill_hydro_edge_flux(&scratch.dir_out, edge_flux_out);
    scratch.step_timings.edge_flux += elapsed_s(t0);
    let t0 = mark();
    gather_hydro_water(current, &scratch.dir_out, next);
    scratch.step_timings.gather += elapsed_s(t0);
}

/// Phase 1 of [`step_hydro_mfd_into`]: per source cell, the amount
/// routed toward each of its 6 toric neighbors this substep (the
/// symmetric MFD split, Tarboton D-inf), 0 in every direction that
/// isn't downhill or when the source has no mobile water. Reads only
/// `current`, immutable for the whole substep (nothing in this
/// function touches it): fully independent per source, parallelizable
/// (`par::for_each_chunk_mut6`). Same formulas as the historical serial
/// loop, split out so the split doesn't change the physics.
fn fill_hydro_outflow(current: &HexGrid, params: &HydroParams, dir_out: &mut [Vec<f32>; 6]) {
    let cur_cells = current.cells_slice();
    for_each_chunk_mut6(dir_out, |start, chunks| {
        for local in 0..chunks[0].len() {
            let i = start + local;
            for chunk in chunks.iter_mut() {
                chunk[local] = 0.0;
            }
            let cell = &cur_cells[i];
            if cell.water_level <= 0.0 {
                continue;
            }
            let eff = cell.effective_elevation();
            // Toroidal neighborhood: surface water also flows across the
            // seam (periodic terrain → the elevation delta there is
            // physical). A river can exit through one edge and continue
            // through the opposite edge.
            let neighbors = current.neighbor_indices_toric(i);

            // Temporary structure: (weight, dir_idx). We first compute the
            // "desired flow" = flow_rate * sum(delta_i) to preserve the CFL
            // behavior, then the split among neighbors according to
            // weights delta_i^p (Tarboton D-inf).
            let mut targets: [(f32, usize); 6] = [(0.0, 0); 6];
            let mut n_targets = 0_usize;
            let mut total_delta = 0.0_f32;
            let mut total_weight = 0.0_f32;
            let mut max_slope = 0.0_f32;
            for (dir_idx, &j) in neighbors.iter().enumerate() {
                let delta = eff - cur_cells[j].effective_elevation();
                if delta <= 0.0 {
                    continue;
                }
                if delta > max_slope {
                    max_slope = delta;
                }
                let weight = delta.powf(params.flow_concentration);
                targets[n_targets] = (weight, dir_idx);
                n_targets += 1;
                total_delta += delta;
                total_weight += weight;
            }
            if n_targets == 0 || total_weight <= 0.0 {
                continue;
            }

            let total_desired = params.flow_rate * total_delta;

            // Sub-cap water is mobilized proportionally to the local slope:
            // flat (slope=0) → only the surplus flows (stable lake/puddle).
            // Steep slope (>= slope_full_mobility) → the whole water_level
            // can flow.
            let surplus = (cell.water_level - cell.water_capacity).max(0.0);
            let piege = cell.water_level - surplus;
            let slope_factor = (max_slope / params.slope_full_mobility).clamp(0.0, 1.0);
            let mobile = surplus + piege * slope_factor;
            if mobile <= 0.0 {
                continue;
            }

            let scale = if total_desired > 0.0 {
                (mobile / total_desired).min(1.0)
            } else {
                0.0
            };

            for &(weight, dir_idx) in &targets[..n_targets] {
                let raw = total_desired * (weight / total_weight);
                let transfer = raw * scale;
                if transfer > 0.0 {
                    chunks[dir_idx][local] = transfer;
                }
            }
        }
    });
}

/// `flux_out[i]`/`flow_vec[i]` are pure per-source aggregates of what
/// [`fill_hydro_outflow`] just computed — `Σ_d dir_out[d][i]` and that
/// same sum weighted by each direction's world unit vector — so no
/// gather across cells is needed for either, unlike `water_level`
/// itself. A pure per-cell map over already-fully-computed data,
/// parallelizable (`par::for_each_chunk_mut2`).
fn fill_hydro_source_aggregates(
    dir_out: &[Vec<f32>; 6],
    flux_out: &mut FluxMap,
    flow_vec: &mut FlowVecMap,
) {
    for_each_chunk_mut2(flux_out, flow_vec, |start, flux_chunk, vec_chunk| {
        for local in 0..flux_chunk.len() {
            let i = start + local;
            let mut vec_x = 0.0_f32;
            let mut vec_y = 0.0_f32;
            let mut total = 0.0_f32;
            for (d, dir) in dir_out.iter().enumerate() {
                let transfer = dir[i];
                if transfer == 0.0 {
                    continue;
                }
                let (dx, dy) = hex_direction_to_world(d);
                vec_x += dx * transfer;
                vec_y += dy * transfer;
                total += transfer;
            }
            flux_chunk[local] = total;
            vec_chunk[local] = (vec_x, vec_y);
        }
    });
}

/// Per-edge export (#103): `edge_flux_out[i][d] = dir_out[d][i]`, the
/// `SoA` outflow buffer transposed into the `AoS` layout `edge_flux_map`
/// and its consumers (front ribbons, `erosion::update_edge_ema`)
/// expect. The two layouts can't be the same allocation — `dir_out`
/// needs six separate `Vec<f32>` for `for_each_chunk_mut6`, while
/// `EdgeFluxMap` is one `[f32; 6]` per cell — so this is a transpose,
/// not a free reinterpretation; it's cheap (pure data movement, no
/// physics recomputed) and parallelizable (`par::for_each_chunk_mut`).
fn fill_hydro_edge_flux(dir_out: &[Vec<f32>; 6], edge_flux_out: &mut EdgeFluxMap) {
    for_each_chunk_mut(edge_flux_out, |start, chunk| {
        for (local, edges) in chunk.iter_mut().enumerate() {
            let i = start + local;
            for (d, edge) in edges.iter_mut().enumerate() {
                *edge = dir_out[d][i];
            }
        }
    });
}

/// Phase 2 of [`step_hydro_mfd_into`]: per destination cell, applies
/// the net `water_level` change — self-loss (`Σ_d dir_out[d][j]`,
/// exactly what `j` itself routed away in phase 1) and the inflow
/// gathered from its neighbors via `coord::opposite_direction`. Also
/// where the historical `current → next` full-grid copy now lives
/// (r250 perf effort, chunk B2): nothing between the top of
/// [`step_hydro_mfd_into`] and this gather ever touches `next` (phase 1
/// reads only `current` and writes the `dir_out` scratch), so this is
/// the phase's first per-cell pass to write `next[j]` — starting each
/// cell from `*cell = current[j].clone()` before adding the delta is
/// bit-identical to the separate copy, one fewer 88-byte full-grid
/// stream per substep (×8/day).
///
/// `k == j` skips a neighbor slot that
/// [`HexGrid::neighbor_indices_toric`]'s doc calls out as its
/// self-transfer fallback ("wrap unreachable (non-hexagonal grid)"): on
/// a genuine `HexGrid::from_radius` torus this never fires (the tiling
/// is exact, every direction reaches a distinct cell), but on any grid
/// built by hand (proptests included) a filler self-loop at direction
/// `d` makes `opposite_direction(d)` alias one of `j`'s OWN real
/// outgoing directions — gathering it back would double-count `j`'s
/// outflow as its own inflow. The scatter form this replaced was immune
/// by construction (a self-transfer nets `+t; -t` on the same cell);
/// this guard is its gather-form equivalent, load-bearing for
/// `prop_flux_out_never_exceeds_water_level` (2-cell ad hoc grid).
fn gather_hydro_water(current: &HexGrid, dir_out: &[Vec<f32>; 6], next: &mut HexGrid) {
    let cur_cells = current.cells_slice();
    for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            let j = start + local;
            *cell = cur_cells[j].clone();
            let self_loss = sum_dir_out(dir_out, j);
            let neighbors = current.neighbor_indices_toric(j);
            let mut gathered_in = 0.0_f32;
            for (d, &k) in neighbors.iter().enumerate() {
                if k == j {
                    continue;
                }
                gathered_in += dir_out[opposite_direction(d)][k];
            }
            cell.water_level += gathered_in - self_loss;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellProperties;
    use crate::coord::{DIRECTIONS, HexCoord};
    use proptest::prelude::*;

    // --- Symmetric MFD tests ---

    fn mfd_default_params() -> HydroParams {
        HydroParams {
            flow_rate: 0.1,
            ..HydroParams::default()
        }
    }

    #[test]
    fn mfd_conserves_mass() {
        let mut current = HexGrid::from_radius(2);
        if let Some(c) = current.get_mut(HexCoord::new(0, 0)) {
            c.elevation = 100.0;
            c.water_level = 10.0;
            c.water_capacity = 1.0;
        }
        for coord in current.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = current.get_mut(coord)
                && coord != HexCoord::new(0, 0)
            {
                c.elevation = f32::from(
                    i16::try_from(100 - 10 * coord.distance(HexCoord::new(0, 0)))
                        .expect("elevation fits i16"),
                );
                c.water_capacity = 1.0;
            }
        }

        let before = total_water(&current);
        let params = mfd_default_params();
        for _ in 0..50 {
            let mut next = current.clone();
            step_hydro_mfd(&current, &mut next, &params);
            current = next;
        }
        let after = total_water(&current);
        assert!(
            (before - after).abs() < 1e-3,
            "conservation violated: {before} → {after}"
        );
    }

    #[test]
    fn mfd_does_not_drain_below_capacity_on_flat_terrain() {
        // Center has wl=0.8 < cap=1.0, neighbors at the same elevation →
        // slope=0 → slope_factor=0 → sub-cap water stays trapped.
        // Invariant: on a perfectly flat plateau, puddles do not drain.
        let mut current = HexGrid::from_radius(1);
        let center = HexCoord::new(0, 0);
        if let Some(c) = current.get_mut(center) {
            c.elevation = 100.0;
            c.water_level = 0.8;
            c.water_capacity = 1.0;
        }
        for (coord, ()) in current
            .neighbors(center)
            .iter()
            .map(|(c, _)| (*c, ()))
            .collect::<Vec<_>>()
        {
            if let Some(c) = current.get_mut(coord) {
                c.elevation = 100.0;
                c.water_level = 0.0;
                c.water_capacity = 1.0;
            }
        }

        let mut next = current.clone();
        let (flux, vec) = step_hydro_mfd(&current, &mut next, &mfd_default_params());

        assert!(
            flux.iter().all(|&f| f == 0.0),
            "sub-capacity puddle should send nothing"
        );
        assert!(vec.iter().all(|&v| v == (0.0, 0.0)));
        let center_after = next.get(center).unwrap().water_level;
        assert!(
            (center_after - 0.8).abs() < 1e-6,
            "center water_level should stay 0.8, found {center_after}"
        );
    }

    #[test]
    fn mfd_flat_equilibrium() {
        // 7 cells at the same elevation and the same water_level >
        // capacity. effective_elevation identical everywhere → no delta >
        // 0 → no flux.
        let mut current = HexGrid::from_radius(1);
        for coord in current.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = current.get_mut(coord) {
                c.elevation = 50.0;
                c.water_level = 5.0;
                c.water_capacity = 1.0;
            }
        }

        let mut next = current.clone();
        let (flux, _) = step_hydro_mfd(&current, &mut next, &mfd_default_params());

        assert!(
            flux.iter().all(|&f| f == 0.0),
            "saturated flat grid should not produce flux"
        );
        for (coord, cell) in next.iter() {
            let orig = current.get(*coord).unwrap();
            assert!(
                (cell.water_level - orig.water_level).abs() < 1e-6,
                "water_level should stay unchanged for {coord:?}"
            );
        }
    }

    /// The per-edge export is an exact decomposition of the aggregate
    /// (#103): the sum of a cell's 6 edge fluxes must fall back onto its
    /// `flux_out`. If this test breaks, the per-direction accumulation has
    /// diverged from the aggregate, a consumer (front ribbons,
    /// `diag_water_flows`) would see a flux different from what the MFD
    /// actually transferred.
    #[test]
    fn edge_flux_sums_to_flux_out() {
        // Cone: high center + water, everything flows toward the outer ring.
        let mut current = HexGrid::from_radius(2);
        for coord in current.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = current.get_mut(coord) {
                let d = coord.distance(HexCoord::new(0, 0));
                c.elevation = f32::from(i16::try_from(100 - 30 * d).expect("fits i16"));
                c.water_level = if d == 0 { 20.0 } else { 2.0 };
                c.water_capacity = 1.0;
            }
        }
        let mut next = current.clone();
        let n = current.len();
        let mut flux_out = vec![0.0; n];
        let mut flow_vec = vec![(0.0, 0.0); n];
        let mut edge_flux = vec![[0.0_f32; 6]; n];
        let mut scratch = HydroScratch::new(n);
        step_hydro_mfd_into(
            &current,
            &mut next,
            &mfd_default_params(),
            &mut scratch,
            &mut flux_out,
            &mut flow_vec,
            &mut edge_flux,
        );

        assert!(
            flux_out.iter().any(|&f| f > 0.0),
            "the cone should produce flux (otherwise the test setup is broken)"
        );
        for i in 0..n {
            let edge_sum: f32 = edge_flux[i].iter().sum();
            assert!(
                (edge_sum - flux_out[i]).abs() < 1e-5,
                "cell {i}: edge sum {edge_sum} != flux_out {}",
                flux_out[i]
            );
        }
    }

    /// The edge flux targets the right neighbor: on a pure east→west
    /// slope, all the water exits through direction 3 (west) and no
    /// other. Pins the `dir_idx` ↔ `coord::DIRECTIONS` mapping that the
    /// front uses to anchor the ribbons at the midpoint of edges (#103).
    #[test]
    fn edge_flux_targets_downhill_direction() {
        let mut current = HexGrid::from_radius(2);
        for coord in current.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = current.get_mut(coord) {
                // Ramp along the world x axis (∝ q + r/2, here 2q+r to
                // stay integer): higher to the east, lower to the west. A
                // ramp on q alone would give the same delta to the west
                // and southwest (same q) and the water would split 50/50.
                c.elevation =
                    f32::from(i16::try_from(500 + 60 * (2 * coord.q + coord.r)).expect("fits i16"));
                c.water_level = 0.0;
                c.water_capacity = 1.0;
            }
        }
        let center = HexCoord::new(0, 0);
        if let Some(c) = current.get_mut(center) {
            c.water_level = 10.0;
        }
        let mut next = current.clone();
        let n = current.len();
        let mut flux_out = vec![0.0; n];
        let mut flow_vec = vec![(0.0, 0.0); n];
        let mut edge_flux = vec![[0.0_f32; 6]; n];
        let mut scratch = HydroScratch::new(n);
        step_hydro_mfd_into(
            &current,
            &mut next,
            &mfd_default_params(),
            &mut scratch,
            &mut flux_out,
            &mut flow_vec,
            &mut edge_flux,
        );

        let ci = current.cell_index(center).expect("center present");
        assert!(flux_out[ci] > 0.0, "the center should flow west");
        // DIRECTIONS[3] = (-1, 0) = west: the only dominant downhill
        // direction; in concentrated MFD (p=6) the downhill diagonals
        // (SW/NW, half the delta) receive a negligible but nonzero share,
        // we check dominance, not exclusivity.
        let west = edge_flux[ci][3];
        assert!(
            west > 0.9 * flux_out[ci],
            "west should dominate: west={west}, total={}",
            flux_out[ci]
        );
        assert!(
            edge_flux[ci][0] == 0.0 && edge_flux[ci][1] == 0.0,
            "no flux upstream (east/northeast): {:?}",
            edge_flux[ci]
        );
    }

    /// r250 perf effort, chunk C1 gate: one loaded cell with two
    /// downhill neighbors at different elevations lands EXACTLY the
    /// per-neighbor split the MFD formula predicts by hand, not merely
    /// "conserves in aggregate" — pins that the two-phase scatter ->
    /// gather split didn't change which neighbor gets how much.
    /// `flow_concentration=1.0` (linear weight, `weight = delta`) and
    /// `water_capacity == water_level` (surplus = 0, so
    /// `effective_elevation` == raw `elevation`, no mm/m offset) keep
    /// the arithmetic exact:
    ///   `total_delta = 10 + 5 = 15`, `total_weight = 15` (weight=delta)
    ///   `total_desired = flow_rate * total_delta = 0.1 * 15 = 1.5`
    ///   `slope_full_mobility = 1.0 <= max_slope(10)` -> `slope_factor = 1`
    ///   `mobile = surplus(0) + piege(10) * 1 = 10` -> `scale = min(10/1.5, 1) = 1`
    ///   `transfer_A = 1.5 * (10/15) = 1.0`, `transfer_B = 1.5 * (5/15) = 0.5`
    #[test]
    fn hydro_two_phase_split_matches_hand_computed_transfers() {
        let params = HydroParams {
            flow_rate: 0.1,
            flow_concentration: 1.0,
            slope_full_mobility: 1.0,
        };
        let mut current = HexGrid::from_radius(2);
        let center = HexCoord::new(0, 0);
        for coord in current.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = current.get_mut(coord) {
                c.elevation = 110.0;
                c.water_level = 0.0;
                c.water_capacity = 10.0;
            }
        }
        if let Some(c) = current.get_mut(center) {
            c.elevation = 100.0;
            c.water_level = 10.0;
            c.water_capacity = 10.0; // surplus = 0: eff == elevation exactly
        }
        let down_a = center + DIRECTIONS[0];
        let down_b = center + DIRECTIONS[1];
        current.get_mut(down_a).unwrap().elevation = 90.0; // delta 10
        current.get_mut(down_b).unwrap().elevation = 95.0; // delta 5

        let mut next = current.clone();
        let (flux, _) = step_hydro_mfd(&current, &mut next, &params);

        let center_after = next.get(center).unwrap().water_level;
        let a_after = next.get(down_a).unwrap().water_level;
        let b_after = next.get(down_b).unwrap().water_level;
        assert!(
            (a_after - 1.0).abs() < 1e-4,
            "neighbor A (Δz=10) should receive 1.0, got {a_after}"
        );
        assert!(
            (b_after - 0.5).abs() < 1e-4,
            "neighbor B (Δz=5) should receive 0.5, got {b_after}"
        );
        assert!(
            (center_after - 8.5).abs() < 1e-4,
            "center should lose exactly 1.5, got {center_after}"
        );
        let ci = current.cell_index(center).expect("center present");
        assert!(
            (flux[ci] - 1.5).abs() < 1e-4,
            "flux_out should be 1.5, got {}",
            flux[ci]
        );
        for &dir in &DIRECTIONS[2..] {
            let lvl = next.get(center + dir).unwrap().water_level;
            assert!(
                lvl == 0.0,
                "uphill neighbor at dir {dir:?} should receive nothing, got {lvl}"
            );
        }
    }

    /// r250 perf effort, chunk C1 gate: mass conservation of the
    /// two-phase scatter -> gather split across exactly ONE routing
    /// sub-step (not accumulated over many, unlike `mfd_conserves_mass`),
    /// on a radius-2 grid with several downhill directions active at
    /// once (cone shape, same terrain as `edge_flux_sums_to_flux_out`).
    #[test]
    fn hydro_two_phase_conserves_water_within_one_substep() {
        let mut current = HexGrid::from_radius(2);
        for coord in current.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = current.get_mut(coord) {
                let d = coord.distance(HexCoord::new(0, 0));
                c.elevation = f32::from(i16::try_from(100 - 30 * d).expect("fits i16"));
                c.water_level = if d == 0 { 20.0 } else { 2.0 };
                c.water_capacity = 1.0;
            }
        }
        let before = total_water(&current);
        let mut next = current.clone();
        step_hydro_mfd(&current, &mut next, &mfd_default_params());
        let after = total_water(&next);
        let drift = (after - before).abs() / before.max(1.0);
        assert!(
            drift < 1e-4,
            "not conservative within one substep: before={before} after={after} drift={drift}"
        );
    }

    /// r250 perf effort, chunk C1: regression pin for the self-loop
    /// aliasing bug the gather introduced on non-hexagonal ad hoc grids.
    /// `HexGrid::neighbor_indices_toric`'s documented "wrap unreachable"
    /// fallback fills a missing direction with the cell itself; without
    /// the `k == j` guard in `gather_hydro_water`, that self-loop's
    /// `opposite_direction` can alias one of the SAME cell's own real
    /// outgoing directions, so the cell gathered back its own outflow as
    /// inflow — first caught by `prop_flux_out_never_exceeds_water_level`
    /// shrinking to this exact 2-cell shape; this is its deterministic
    /// pin.
    #[test]
    fn hydro_two_phase_conserves_mass_on_non_hexagonal_ad_hoc_grid() {
        let src = HexCoord::new(0, 0);
        let sink = HexCoord::new(1, 0);
        let mut current = HexGrid::new();
        current.insert(
            src,
            CellProperties {
                elevation: 500.0,
                water_level: 20.0,
                water_capacity: 1.0,
                ..Default::default()
            },
        );
        current.insert(
            sink,
            CellProperties {
                elevation: 0.0,
                water_level: 0.0,
                water_capacity: 1.0,
                ..Default::default()
            },
        );
        let before = total_water(&current);
        let mut next = current.clone();
        step_hydro_mfd(&current, &mut next, &mfd_default_params());
        let after = total_water(&next);
        assert!(
            (before - after).abs() < 1e-4,
            "self-loop aliasing on a non-hexagonal grid broke conservation: {before} -> {after}"
        );
        let src_after = next.get(src).unwrap().water_level;
        assert!(
            src_after <= 20.0 + 1e-4,
            "src must not gain back its own outflow via the self-loop alias: {src_after}"
        );
    }

    proptest! {
        /// Structural invariant: a cell cannot send more water than it
        /// has. The MFD computes the transfer as `scale * raw` with
        /// `scale = min(mobile/total_raw, 1.0)`, so
        /// `total_transfer <= mobile <= water_level`.
        ///
        /// If this proptest fails, `step_hydro_mfd` is sending more than
        /// the available stock (a local conservation bug that creates
        /// water out of nothing).
        #[test]
        fn prop_flux_out_never_exceeds_water_level(
            water_source in 0.0_f32..100.0,
            water_cap in 0.0_f32..5.0,
            elev_source in 0.0_f32..1000.0,
            elev_sink in 0.0_f32..1000.0,
            flow_rate in 0.01_f32..0.5,
        ) {
            let src = HexCoord::new(0, 0);
            let sink = HexCoord::new(1, 0);
            let mut current = HexGrid::new();
            current.insert(
                src,
                CellProperties {
                    elevation: elev_source,
                    water_level: water_source,
                    water_capacity: water_cap,
                    ..Default::default()
                },
            );
            current.insert(
                sink,
                CellProperties {
                    elevation: elev_sink,
                    water_level: 0.0,
                    water_capacity: water_cap,
                    ..Default::default()
                },
            );
            let mut next = current.clone();
            let params = HydroParams {
                flow_rate,
                ..HydroParams::default()
            };
            let (flux, _) = step_hydro_mfd(&current, &mut next, &params);

            let flux_src = current
                .cell_index(src)
                .map_or(0.0, |i| flux[i]);
            prop_assert!(
                flux_src <= water_source + 1e-4,
                "flux_out ({flux_src}) > water_level ({water_source})"
            );
            prop_assert!(
                flux_src >= 0.0,
                "negative flux_out: {flux_src}"
            );

            // Local conservation: water[src] + water[sink] conserves
            // (up to floating-point epsilon) the initial water[src].
            let src_after = next.get(src).unwrap().water_level;
            let sink_after = next.get(sink).unwrap().water_level;
            prop_assert!(
                ((src_after + sink_after) - water_source).abs() < 1e-3,
                "conservation broken: {src_after} + {sink_after} != {water_source}"
            );
        }
    }
}
