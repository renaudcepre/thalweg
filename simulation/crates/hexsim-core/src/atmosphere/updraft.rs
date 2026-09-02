use crate::coord::hex_direction_to_world;
use crate::grid::HexGrid;
use crate::wind::WindField;

/// Total ascent `w = H·(−∇·v) + v·∇z` (m/s) per cell (Phase 3 synoptic
/// ascent trigger, ex-design C #69).
/// - Column convergence: consistent hexagonal estimator
///   `−∇·v ≈ −Σ_neighbors (v_j·û_ij) / (3d)` (the central term cancels,
///   Σ û = 0), multiplied by the thickness `h_column` of the humid layer.
/// - Orographic uplift: `v·∇z`, altitude gradient via the same
///   estimator, the wind pushing against the slope rises.
///
/// The wind is converted from the `WindVec` unit (magnitude × 10 = m/s, #33) to
/// m/s. Positive = air that rises (front OR windward flank).
pub(crate) fn fill_updraft_into(
    current: &HexGrid,
    wind_field: &WindField,
    h_column: f32,
    out: &mut Vec<f32>,
) {
    const WINDVEC_TO_MS: f32 = 10.0;
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
            let wj = wind_field[j];
            acc -= wj.x * dx + wj.y * dy;
            let dz = cells[j].elevation - z_c;
            gx += dz * dx;
            gy += dz * dy;
        }
        let conv_si = acc * WINDVEC_TO_MS * inv3d;
        let wc = wind_field[i];
        let w_oro = (wc.x * gx + wc.y * gy) * WINDVEC_TO_MS * inv3d;
        *w = conv_si * h_column + w_oro;
    }
}
