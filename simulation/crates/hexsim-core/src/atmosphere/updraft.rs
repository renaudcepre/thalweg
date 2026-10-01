use crate::coord::hex_direction_to_world;
use crate::grid::HexGrid;
use crate::wind::{WindField, wind_magnitude_to_meters_per_second};

/// Total ascent `w = H·(−∇·v) + v·∇z` (m/s) per cell (Phase 3 synoptic
/// ascent trigger, ex-design C #69).
/// - Column convergence: consistent hexagonal estimator
///   `−∇·v ≈ −Σ_neighbors (v_j·û_ij) / (3d)` (the central term cancels,
///   Σ û = 0), multiplied by the thickness `h_column` of the humid layer.
/// - Orographic uplift: `v·∇z`, altitude gradient via the same
///   estimator, the wind pushing against the slope rises (barrier lift,
///   Smith 1979: `w = U·∇h` with `U` the **incident** wind).
///
/// `ambient_wind` is the wind the estimator derives, and which wind that
/// is decides whether the result is weather or grid noise (#110). It has
/// to be the **ambient** wind, the interpolated synoptic base
/// (`AtmoForcing::synoptic_wind`, smooth at the solver's `L_d`), not the
/// composite 130 m field: measured on 2026-07-15 with the composite,
/// `w` spanned −225 to +314 m/s (std 47.6) where the design reasons
/// about ±1 m/s, because `apply_terrain_deflection` and
/// `propagate_upstream` manufacture divergences of ~1e-2 s⁻¹ that are
/// slope noise, × `h_column` = 1500 m; and its altitude signature was
/// inverted (+10 m/s over the plains, −83 over the summits), because the
/// barrier term on an already deflected wind double-counts the relief
/// with a downslope bias. The composite stays a *transport* modifier; it
/// is not an input of this estimator. The caller falls back to it only
/// when no synoptic base exists (scripted wind, micro-tests).
///
/// The wind is converted from the `WindVec` unit to m/s via
/// `wind::wind_magnitude_to_meters_per_second` (convention #33, single
/// source of truth lives on the type in `wind.rs`). Positive = air that
/// rises (front OR windward flank).
pub(crate) fn fill_updraft_into(
    current: &HexGrid,
    ambient_wind: &WindField,
    h_column: f32,
    out: &mut Vec<f32>,
) {
    let inv3d = 1.0 / (3.0 * crate::dynamics::CELL_SPACING_M);
    let n = current.len();
    let cells = current.cells_slice();
    out.clear();
    out.resize(n, 0.0);
    for (i, w) in out.iter_mut().enumerate() {
        let neighbors = current.neighbor_indices_toric(i);
        let z_c = cells[i].elevation;
        let mut acc = 0.0_f32;
        let (mut gx, mut gy) = (0.0_f32, 0.0_f32);
        for (dir, &j) in neighbors.iter().enumerate() {
            let (dx, dy) = hex_direction_to_world(dir);
            let wj = ambient_wind[j];
            acc -= wj.x * dx + wj.y * dy;
            let dz = cells[j].elevation - z_c;
            gx += dz * dx;
            gy += dz * dy;
        }
        let conv_si = wind_magnitude_to_meters_per_second(acc).0 * inv3d;
        let wc = ambient_wind[i];
        let w_oro = wind_magnitude_to_meters_per_second(wc.x * gx + wc.y * gy).0 * inv3d;
        *w = conv_si * h_column + w_oro;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::wind::WindVec;

    /// Uniform west wind over an isolated hill: the convergence of a
    /// uniform field is exactly zero, only the barrier lift `v·∇z` is
    /// left — positive windward (west flank), negative leeward, of
    /// analytic amplitude `v · Δz / (3d)`. Pins the altitude signature
    /// the right way up and the m/s orders of magnitude (#110): an
    /// estimator fed a deflected wind cannot pass this, its `v·∇z` term
    /// reads the slope twice.
    #[test]
    fn uniform_wind_on_a_hill_lifts_windward_sinks_leeward() {
        let mut grid = HexGrid::from_radius(3);
        let peak_m = 130.0;
        grid.get_mut(HexCoord::new(0, 0))
            .expect("center exists")
            .elevation = peak_m;
        // 3 m/s eastward (+x), in WindVec units (×10 = m/s).
        let east_3ms = wind_magnitude_to_meters_per_second(0.3).0;
        let wind = vec![WindVec { x: 0.3, y: 0.0 }; grid.len()];
        let mut w = Vec::new();
        fill_updraft_into(&grid, &wind, 1500.0, &mut w);

        let windward = grid
            .cell_index(HexCoord::new(-1, 0))
            .expect("windward flank exists");
        let leeward = grid
            .cell_index(HexCoord::new(1, 0))
            .expect("leeward flank exists");
        // v·∇z = 3 m/s × 130 m / (3 × 130 m) = 1 m/s exactly.
        let expected = east_3ms * peak_m / (3.0 * crate::dynamics::CELL_SPACING_M);
        assert!(
            (w[windward] - expected).abs() < 1e-3,
            "windward flank: w = {} expected {expected}",
            w[windward]
        );
        assert!(
            (w[leeward] + expected).abs() < 1e-3,
            "leeward flank: w = {} expected {}",
            w[leeward],
            -expected
        );
        // Away from the hill, uniform wind over flat ground: w = 0.
        let far = grid
            .cell_index(HexCoord::new(3, 0))
            .expect("far cell exists");
        assert!(w[far].abs() < 1e-6, "flat: w = {}", w[far]);
    }
}
