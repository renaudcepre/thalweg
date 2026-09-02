use serde::{Deserialize, Serialize};

use crate::coord::hex_direction_to_world;
use crate::dynamics::{CELL_SPACING_M, STEEP_SLOPE_GRADE};
use crate::grid::HexGrid;

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
    step_hydro_mfd_into(
        current,
        next,
        params,
        &mut flux_out,
        &mut flow_vec,
        &mut edge_flux,
    );
    (flux_out, flow_vec)
}

/// Zero-malloc variant: writes into the provided buffers (resize + reset
/// to 0).
pub fn step_hydro_mfd_into(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &HydroParams,
    flux_out: &mut FluxMap,
    flow_vec: &mut FlowVecMap,
    edge_flux_out: &mut EdgeFluxMap,
) {
    let n = current.len();
    flux_out.resize(n, 0.0);
    flux_out.fill(0.0);
    flow_vec.resize(n, (0.0, 0.0));
    flow_vec.fill((0.0, 0.0));
    edge_flux_out.resize(n, [0.0; 6]);
    edge_flux_out.fill([0.0; 6]);

    let cur_cells = current.cells_slice();
    next.cells_slice_mut().clone_from_slice(cur_cells);

    let next_cells = next.cells_slice_mut();
    for i in 0..n {
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

        // Temporary structure: (neighbor_idx, weight, dir_idx). We first
        // compute the "desired flow" = flow_rate * sum(delta_i) to
        // preserve the CFL behavior, then the split among neighbors
        // according to weights delta_i^p (Tarboton D-inf).
        let mut targets: [(usize, f32, usize); 6] = [(0, 0.0, 0); 6];
        let mut n_targets = 0usize;
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
            targets[n_targets] = (j, weight, dir_idx);
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

        let mut vec_x = 0.0_f32;
        let mut vec_y = 0.0_f32;
        let mut total_transfer = 0.0_f32;
        for &(j, weight, dir_idx) in &targets[..n_targets] {
            let raw = total_desired * (weight / total_weight);
            let transfer = raw * scale;
            if transfer <= 0.0 {
                continue;
            }
            let (dx, dy) = hex_direction_to_world(dir_idx);
            vec_x += dx * transfer;
            vec_y += dy * transfer;
            total_transfer += transfer;
            edge_flux_out[i][dir_idx] += transfer;
            next_cells[j].water_level += transfer;
        }
        if total_transfer > 0.0 {
            next_cells[i].water_level -= total_transfer;
            flux_out[i] += total_transfer;
            flow_vec[i].0 += vec_x;
            flow_vec[i].1 += vec_y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellProperties;
    use crate::coord::HexCoord;
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
        step_hydro_mfd_into(
            &current,
            &mut next,
            &mfd_default_params(),
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
        step_hydro_mfd_into(
            &current,
            &mut next,
            &mfd_default_params(),
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
