//! Dedicated synoptic mesh: the shallow-water solver (`dynamics`) integrates
//! on a **coarse** hexagonal torus at the field's natural scale, not on
//! the fine terrain grid.
//!
//! Why: the synoptic field physically has no content below ~`L_d`
//! (≈ 10.7 km; viscosity, rescaled as `Δx²`, actively smooths out
//! anything that gets close to it), but the explicit solver's cost
//! scales as `N_cells × CFL sub-steps`, and the CFL is in `1/Δx`. On the
//! fine grid at 130 m: 163 sub-steps/h over every cell, **82% of the
//! tick** measured (A/B synoptic on/off via `synoptic.enabled`, r45,
//! work item #88). On a grid at the calibration spacing (~1 km): ~20
//! sub-steps/h over ~64x fewer cells, same physics, and this is the
//! spacing at which ALL solver parameters were calibrated and validated
//! (Phase 0 spike, Phase 4 calibration).
//!
//! Coupling, both ways:
//! - fine -> coarse: **average** temperature per coarse cell (the
//!   thermal forcing `Q(T)` responds to contrast at km scale; averaging
//!   is more physical than sampling noise at 130 m);
//! - coarse -> fine: **barycentric** interpolation (3 neighboring
//!   coarse centers, exact-linear, weights ≥ 0 summing to 1) of the
//!   base wind and the `h`/`u`/`v` fields exported to the front end.
//!
//! Geometry: both grids are exact hexagonal tori
//! (`torus_lattice_vectors`). The coarse one is a domain of radius
//! `Rc ≈ R·Δx_fine/Δx_target`, with spacing `Δx_c = (R/Rc)·Δx_fine`, so
//! that the physical extents coincide. The two tori's translation
//! lattices don't coincide exactly (`s·(2Rc+1) ≠ 2R+1` in general, a
//! gap of ~`s` fine cells over `2R+1`, ~3% at r120): each grid stays an
//! exact torus for ITS solver, the mismatch only shows up at the seam
//! of the fine<->coarse mapping, negligible for a field smooth at the
//! scale `L_d ≫ Δx_c`, and validated globally by the climate ablation
//! (table in `simulation.rs`).
//!
//! `Rc = R` (forced by `HEXSIM_SYNOPTIC_COARSE=0`, or tiny grids)
//! degenerates into an identity mapping: same coordinates, weights
//! `(1, 0, 0)`: the solver reproduces the historical fine-grid behavior
//! bit-for-bit.

use crate::coord::{HexCoord, hex_direction_to_world};
use crate::dynamics::{CELL_SPACING_M, SYNOPTIC_REFERENCE_SPACING_M};
use crate::grid::HexGrid;
use crate::wind::{WindField, WindVec};

/// Floor coarse radius: below 2 (19 cells) the torus degenerates and
/// the solver no longer has enough to represent a system. If the fine
/// domain is itself smaller, we fall back to the identity (`Rc = R`).
const MIN_COARSE_RADIUS: i32 = 2;

/// float->i32 rounding of a hex coordinate bounded by the grid radius
/// (|q| ≤ ~250 even at France scale). `as` is the only std path to
/// round a bounded float to an integer: isolated here, documented (same
/// justification as `temperature.rs::cells_to_radius`).
#[allow(clippy::cast_possible_truncation)]
fn round_coord(x: f32) -> i32 {
    x.round() as i32
}

/// Fine axial coordinate -> f32 (radii ≤ ~250: exact in f32).
fn axial_f(c: HexCoord) -> (f32, f32) {
    (
        f32::from(i16::try_from(c.q).unwrap_or(0)),
        f32::from(i16::try_from(c.r).unwrap_or(0)),
    )
}

/// World position (in spacing units) of a fractional axial coordinate:
/// `x = q + r/2`, `y = (√3/2)·r`, consistent with
/// `hex_direction_to_world` (E = (1,0) → (1.0, 0.0), SE = (0,1) → (0.5, √3/2)).
fn axial_to_world(q: f32, r: f32) -> (f32, f32) {
    const SQRT_3_2: f32 = 0.866_025_4;
    (q + 0.5 * r, SQRT_3_2 * r)
}

/// Hex containing the fractional axial point (standard cube-round): the
/// returned center is at world distance ≤ 1/√3 ≈ 0.577 from the point
/// (Voronoi cell circumradius), which guarantees the point falls inside
/// one of the 6 triangles (center, neighbor k, neighbor k+1); cf
/// `barycentric_weights`.
fn hex_round(qf: f32, rf: f32) -> HexCoord {
    let sf = -qf - rf;
    let (mut q, mut r) = (qf.round(), rf.round());
    let s = sf.round();
    let (dq, dr, ds) = ((q - qf).abs(), (r - rf).abs(), (s - sf).abs());
    if dq > dr && dq > ds {
        q = -r - s;
    } else if dr > ds {
        r = -q - s;
    }
    HexCoord::new(round_coord(q), round_coord(r))
}

/// Barycentric weights of point `v` (world position relative to the
/// center) in the triangle (center, direction k, direction k+1):
/// exact-linear, weights ≥ 0 by construction for any point in the
/// center's Voronoi cell (the chord between two adjacent neighbors
/// passes at cos 30° = 0.866 from the center, beyond the 0.577
/// circumradius). Panics if the geometry is violated: this is a
/// construction invariant, not a runtime case to absorb.
fn barycentric_weights(v: (f32, f32)) -> (usize, f32, f32, f32) {
    const EPS: f32 = 1e-4;
    for k in 0..6 {
        let dk = hex_direction_to_world(k);
        let dk1 = hex_direction_to_world((k + 1) % 6);
        let det = dk.0 * dk1.1 - dk1.0 * dk.1;
        let w1 = (v.0 * dk1.1 - v.1 * dk1.0) / det;
        let w2 = (dk.0 * v.1 - dk.1 * v.0) / det;
        if w1 >= -EPS && w2 >= -EPS && w1 + w2 <= 1.0 + EPS {
            let w1 = w1.max(0.0);
            let w2 = w2.max(0.0);
            let w0 = (1.0 - w1 - w2).max(0.0);
            let sum = w0 + w1 + w2;
            return (k, w0 / sum, w1 / sum, w2 / sum);
        }
    }
    unreachable!("point outside the 6 triangles of the hex-round center: |v| > 1/√3 ?");
}

/// Coupling mesh between the fine terrain grid and the coarse synoptic
/// torus. Entirely deterministic from the fine grid: rebuilt on
/// checkpoint load, never serialized.
pub struct SynopticMesh {
    /// Coarse torus on which the solver integrates. Only the
    /// temperature of its cells is maintained (`aggregate_temperature`).
    grid: HexGrid,
    /// Physical spacing (m) of the coarse torus: to pass to
    /// `SynopticParams::for_spacing`.
    spacing_m: f32,
    /// Coarse cell containing each fine cell (exact partition).
    fine_to_coarse: Vec<usize>,
    /// Coarse -> fine interpolation: 3 barycentric (index, weight)
    /// pairs per fine cell.
    interp: Vec<[(usize, f32); 3]>,
    /// 1/(assigned fine cells) per coarse cell (forcing average).
    inv_count: Vec<f32>,
    /// Aggregation scratch (zero-malloc per call).
    scratch_sum: Vec<f32>,
}

impl SynopticMesh {
    /// Mesh at the natural coarse radius: `Rc ≈ R·CELL_SPACING_M /
    /// SYNOPTIC_REFERENCE_SPACING_M`, bounded to `[MIN_COARSE_RADIUS, R]`.
    #[must_use]
    pub fn build(fine: &HexGrid) -> Self {
        let r = fine.radius();
        let target = f32::from(i16::try_from(r).unwrap_or(0)) * CELL_SPACING_M
            / SYNOPTIC_REFERENCE_SPACING_M;
        let rc = round_coord(target).clamp(MIN_COARSE_RADIUS.min(r), r);
        Self::with_coarse_radius(fine, rc)
    }

    /// Identity mesh (`Rc = R`): the solver integrates on the fine
    /// grid, bit-for-bit historical behavior. Ablation kill switch
    /// (`HEXSIM_SYNOPTIC_COARSE=0`).
    #[must_use]
    pub fn identity(fine: &HexGrid) -> Self {
        Self::with_coarse_radius(fine, fine.radius())
    }

    /// Generic construction: coarse torus of radius `rc`, mapping by
    /// hex-round of fine coordinates scaled down by factor `s = R/Rc`,
    /// barycentric weights per fine cell. `pub(crate)`: checkpoint
    /// loading rebuilds the mesh at the persisted radius, independent
    /// of the current environment.
    pub(crate) fn with_coarse_radius(fine: &HexGrid, rc: i32) -> Self {
        let radius_fine = fine.radius();
        let grid = HexGrid::from_radius(rc);
        let n_coarse = grid.len();
        // s = R/Rc: exactly 1 when rc == radius_fine (identity, no f32
        // rounding).
        let scale = f32::from(i16::try_from(radius_fine).unwrap_or(1)).max(1.0)
            / f32::from(i16::try_from(rc).unwrap_or(1)).max(1.0);
        let spacing_m = CELL_SPACING_M * scale;

        let mut fine_to_coarse = Vec::with_capacity(fine.len());
        let mut interp = Vec::with_capacity(fine.len());
        let mut count = vec![0.0_f32; n_coarse];
        for &coord in fine.coords_slice() {
            let (qf, rf) = axial_f(coord);
            let (qs, rs) = (qf / scale, rf / scale);
            let c0 = hex_round(qs, rs);
            let idx0 = grid
                .index_of(c0)
                .or_else(|| grid.wrap_target(c0).and_then(|w| grid.index_of(w)))
                .expect("hex-round of a fine coordinate outside the coarse torus");
            let (px, py) = axial_to_world(qs, rs);
            // Local geometry from c0 PRE-wrap (the wrap is a lattice
            // translation: the deltas are invariant), index POST-wrap
            // (idx0 and its toric neighbors: `wrap(c0)+d ≡ wrap(c0+d)`
            // on the torus).
            let (c0q, c0r) = axial_f(c0);
            let (cx, cy) = axial_to_world(c0q, c0r);
            let (k, w0, w1, w2) = barycentric_weights((px - cx, py - cy));
            let nbr = grid.neighbor_indices_toric(idx0);
            interp.push([(idx0, w0), (nbr[k], w1), (nbr[(k + 1) % 6], w2)]);
            fine_to_coarse.push(idx0);
            count[idx0] += 1.0;
        }
        // Exact partition: every coarse cell must receive at least one
        // fine cell, otherwise its thermal forcing would be undefined.
        // True by construction as soon as s ≥ 1 (every coarse center
        // has a fine cell at ≤ s/2 from it): we assert it rather than
        // masking it.
        let inv_count = count
            .iter()
            .map(|&c| {
                assert!(c > 0.0, "coarse cell with no fine antecedent");
                1.0 / c
            })
            .collect();
        Self {
            grid,
            spacing_m,
            fine_to_coarse,
            interp,
            inv_count,
            scratch_sum: vec![0.0; n_coarse],
        }
    }

    /// Coarse torus (to pass to `SynopticState::step_hour`).
    #[must_use]
    pub fn grid(&self) -> &HexGrid {
        &self.grid
    }

    /// Physical spacing (m) of the coarse torus.
    #[must_use]
    pub fn spacing_m(&self) -> f32 {
        self.spacing_m
    }

    /// Averages fine temperature into each coarse cell: the input to
    /// the solver's thermal forcing `Q(T)`. To be called before every
    /// `step_hour` (the coarse grid only exists for this).
    pub fn aggregate_temperature(&mut self, fine: &HexGrid) {
        debug_assert_eq!(fine.len(), self.fine_to_coarse.len());
        self.scratch_sum.fill(0.0);
        for (cell, &ci) in fine.cells_slice().iter().zip(&self.fine_to_coarse) {
            self.scratch_sum[ci] += cell.temperature;
        }
        for ((c, &sum), &inv) in self
            .grid
            .cells_slice_mut()
            .iter_mut()
            .zip(&self.scratch_sum)
            .zip(&self.inv_count)
        {
            c.temperature = sum * inv;
        }
    }

    /// Interpolates the coarse base wind onto the fine grid
    /// (barycentric, exact-linear). `out` is resized to the fine size.
    pub fn interpolate_wind(&self, coarse: &WindField, out: &mut WindField) {
        out.resize(self.interp.len(), WindVec::default());
        for (o, tri) in out.iter_mut().zip(&self.interp) {
            let (mut x, mut y) = (0.0, 0.0);
            for &(ci, w) in tri {
                x += coarse[ci].x * w;
                y += coarse[ci].y * w;
            }
            *o = WindVec { x, y };
        }
    }

    /// Samples a coarse scalar field at the center of fine cell
    /// `fine_idx` (same weights as `interpolate_wind`): for snapshot
    /// export of the `h`/`u`/`v` fields per fine cell.
    #[must_use]
    pub fn sample_scalar(&self, field: &[f32], fine_idx: usize) -> f32 {
        self.interp[fine_idx]
            .iter()
            .map(|&(ci, w)| field[ci] * w)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::DIRECTIONS;

    fn fine_grid(radius: i32) -> HexGrid {
        HexGrid::from_radius(radius)
    }

    #[test]
    fn coarse_radius_follows_reference_spacing() {
        // r120 -> 15 (≈1040 m/hex), r45 -> 5, r30 -> 4: the coarse
        // spacing stays within ±15% of the calibration spacing.
        for (r, rc_expected) in [(120, 15), (45, 5), (30, 4)] {
            let mesh = SynopticMesh::build(&fine_grid(r));
            let rc = mesh.grid().radius();
            assert_eq!(rc, rc_expected, "r={r}");
            let ratio = mesh.spacing_m() / SYNOPTIC_REFERENCE_SPACING_M;
            assert!(
                (0.85..=1.15).contains(&ratio),
                "r={r} : espacement grossier {} m",
                mesh.spacing_m()
            );
        }
    }

    #[test]
    fn tiny_grids_degenerate_to_identity() {
        for r in [0, 1, 2, 3] {
            let fine = fine_grid(r);
            let mesh = SynopticMesh::build(&fine);
            assert!(mesh.grid().len() <= fine.len(), "r={r}");
            assert!(mesh.grid().radius() >= MIN_COARSE_RADIUS.min(r), "r={r}");
        }
    }

    #[test]
    fn identity_mesh_maps_each_cell_to_itself_with_weight_one() {
        let fine = fine_grid(6);
        let mesh = SynopticMesh::identity(&fine);
        assert_eq!(mesh.grid().len(), fine.len());
        for (i, tri) in mesh.interp.iter().enumerate() {
            // from_radius(r) regenerates the coords in the same order:
            // the coarse index of cell i is i.
            assert_eq!(mesh.fine_to_coarse[i], i);
            assert!((tri[0].1 - 1.0).abs() < 1e-6, "w0 = {}", tri[0].1);
            assert_eq!(tri[0].0, i);
            assert!(tri[1].1.abs() < 1e-6 && tri[2].1.abs() < 1e-6);
        }
    }

    #[test]
    fn mapping_is_a_partition_and_weights_are_convex() {
        let fine = fine_grid(30);
        let mesh = SynopticMesh::build(&fine);
        let n_coarse = mesh.grid().len();
        let mut seen = vec![0_u32; n_coarse];
        for &ci in &mesh.fine_to_coarse {
            assert!(ci < n_coarse);
            seen[ci] += 1;
        }
        assert!(
            seen.iter().all(|&c| c > 0),
            "coarse cell with no antecedent"
        );
        for tri in &mesh.interp {
            let sum: f32 = tri.iter().map(|&(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum of weights = {sum}");
            assert!(tri.iter().all(|&(ci, w)| w >= 0.0 && ci < n_coarse));
        }
    }

    #[test]
    fn aggregation_averages_and_interpolation_reproduces_a_constant() {
        let mut fine = fine_grid(20);
        for coord in fine.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = fine.get_mut(coord) {
                c.temperature = 7.5;
            }
        }
        let mut mesh = SynopticMesh::build(&fine);
        mesh.aggregate_temperature(&fine);
        for c in mesh.grid().cells_slice() {
            assert!((c.temperature - 7.5).abs() < 1e-5);
        }
        // A constant field is reproduced exactly (convex weights).
        let field = vec![3.25_f32; mesh.grid().len()];
        for i in 0..fine.len() {
            assert!((mesh.sample_scalar(&field, i) - 3.25).abs() < 1e-5);
        }
    }

    #[test]
    fn interpolation_is_linear_exact_away_from_the_seam() {
        // Linear field h = a·x + b·y set on the coarse centers (in
        // coarse world units): the barycentric interpolation must
        // reproduce it exactly at the center of every fine cell WHOSE
        // triangle doesn't cross the toric seam (a linear field is not
        // torus-periodic; the seam is out of scope here).
        let fine = fine_grid(24);
        let mesh = SynopticMesh::build(&fine);
        let coarse = mesh.grid();
        let (slope_x, slope_y) = (0.7_f32, -1.3_f32);
        let field: Vec<f32> = coarse
            .coords_slice()
            .iter()
            .map(|&c| {
                let (cq, cr) = axial_f(c);
                let (wx, wy) = axial_to_world(cq, cr);
                slope_x * wx + slope_y * wy
            })
            .collect();
        let scale = f32::from(i16::try_from(fine.radius()).unwrap_or(1))
            / f32::from(i16::try_from(coarse.radius()).unwrap_or(1));
        let mut checked = 0;
        for (i, &coord) in fine.coords_slice().iter().enumerate() {
            // Seamless triangle: the 3 returned vertices are the
            // direct, NON-wrapped neighbors of the center (direct
            // index_of on c + dir).
            let tri = mesh.interp[i];
            let c0 = coarse.coords_slice()[tri[0].0];
            let all_unwrapped = (0..6).all(|k| {
                coarse
                    .index_of(c0 + DIRECTIONS[k])
                    .is_none_or(|idx| idx == coarse.neighbor_indices_toric(tri[0].0)[k])
            });
            let members_inside = tri
                .iter()
                .all(|&(ci, _)| coarse.index_of(coarse.coords_slice()[ci]).is_some());
            let (qf, rf) = axial_f(coord);
            let (wx, wy) = axial_to_world(qf / scale, rf / scale);
            let hex_dist_ok = c0.distance(HexCoord::new(0, 0)) + 1 < coarse.radius();
            if !(all_unwrapped && members_inside && hex_dist_ok) {
                continue;
            }
            let expected = slope_x * wx + slope_y * wy;
            let got = mesh.sample_scalar(&field, i);
            assert!(
                (got - expected).abs() < 1e-3,
                "fine cell {i}: interpolated {got} vs linear {expected}"
            );
            checked += 1;
        }
        assert!(checked > 100, "too few interior cells: {checked}");
    }
}
