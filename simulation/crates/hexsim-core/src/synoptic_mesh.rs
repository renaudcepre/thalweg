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
use crate::par::{for_each_chunk_mut, for_each_chunk_mut_over};
use crate::wind::{WindField, WindVec};

/// Floor coarse radius: below 2 (19 cells) the torus degenerates and
/// the solver no longer has enough to represent a system. If the fine
/// domain is itself smaller, we fall back to the identity (`Rc = R`).
///
/// `pub(crate)`: the moist-layer coarse mesh (`atmosphere::coarse`) uses
/// this exact floor for its own radius formula, at its own target
/// spacing — same degenerate-torus rationale, not a coincidence to
/// duplicate as a second magic number.
pub(crate) const MIN_COARSE_RADIUS: i32 = 2;

/// float->i32 rounding of a hex coordinate bounded by the grid radius
/// (|q| ≤ ~250 even at France scale). `as` is the only std path to
/// round a bounded float to an integer: isolated here, documented (same
/// justification as `temperature.rs::cells_to_radius`).
///
/// `pub(crate)`: reused by `atmosphere::coarse::moist_coarse_radius`,
/// same rounding contract, a different target spacing.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn round_coord(x: f32) -> i32 {
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
    /// The inverse map of `fine_to_coarse` in CSR form: the fine cells
    /// of coarse cell `ci` are `coarse_to_fine[coarse_offsets[ci]..
    /// coarse_offsets[ci + 1]]`, in ascending fine index. Lets
    /// `aggregate_temperature` gather per coarse cell (one worker per
    /// region of coarse cells, r250 perf effort) instead of scattering
    /// serially over the fine grid; the ascending order is what keeps
    /// the gather's f32 sums bit-identical to that scatter's.
    coarse_to_fine: Vec<usize>,
    coarse_offsets: Vec<usize>,
    /// Coarse -> fine interpolation: 3 barycentric (index, weight)
    /// pairs per fine cell.
    interp: Vec<[(usize, f32); 3]>,
    /// 1/(assigned fine cells) per coarse cell (forcing average).
    inv_count: Vec<f32>,
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
        let (coarse_to_fine, coarse_offsets) = invert_partition(&fine_to_coarse, n_coarse);
        Self {
            grid,
            spacing_m,
            fine_to_coarse,
            coarse_to_fine,
            coarse_offsets,
            interp,
            inv_count,
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
    ///
    /// A gather per coarse cell over its CSR list of fine cells
    /// (`coarse_to_fine`), one worker per region of coarse cells
    /// (`par::for_each_chunk_mut_over`, sized on the fine grid it
    /// streams): the historical serial scatter (`sum[f2c[i]] += T_i` for
    /// `i` ascending) added each coarse cell's terms in ascending fine
    /// index from a zero, exactly the sequence this gather replays, so
    /// the coarse temperatures are bit-identical to it.
    pub fn aggregate_temperature(&mut self, fine: &HexGrid) {
        debug_assert_eq!(fine.len(), self.fine_to_coarse.len());
        let cells = fine.cells_slice();
        let coarse_to_fine = &self.coarse_to_fine;
        let coarse_offsets = &self.coarse_offsets;
        let inv_count = &self.inv_count;
        for_each_chunk_mut_over(self.grid.cells_slice_mut(), cells.len(), |start, chunk| {
            for (local, c) in chunk.iter_mut().enumerate() {
                let ci = start + local;
                let mut sum = 0.0_f32;
                for &fi in &coarse_to_fine[coarse_offsets[ci]..coarse_offsets[ci + 1]] {
                    sum += cells[fi].temperature;
                }
                c.temperature = sum * inv_count[ci];
            }
        });
    }

    /// Interpolates the coarse base wind onto the fine grid
    /// (barycentric, exact-linear). `out` is resized to the fine size.
    /// A pure per-fine-cell map (`out[i]` reads its own three coarse
    /// samples), parallel per region (`par::for_each_chunk_mut`).
    pub fn interpolate_wind(&self, coarse: &WindField, out: &mut WindField) {
        out.resize(self.interp.len(), WindVec::default());
        let interp = &self.interp;
        for_each_chunk_mut(out, |start, chunk| {
            for (local, o) in chunk.iter_mut().enumerate() {
                let (mut x, mut y) = (0.0, 0.0);
                for &(ci, w) in &interp[start + local] {
                    x += coarse[ci].x * w;
                    y += coarse[ci].y * w;
                }
                *o = WindVec { x, y };
            }
        });
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

    /// Number of coarse cells (`grid().len()`), for a caller sizing a
    /// coarse-side buffer without pulling in `grid()` (e.g.
    /// `atmosphere::coarse::MoistCoarseState::new`).
    #[must_use]
    pub fn coarse_len(&self) -> usize {
        self.coarse_offsets.len() - 1
    }

    /// Number of fine cells this mesh maps.
    #[must_use]
    pub(crate) fn fine_len(&self) -> usize {
        self.fine_to_coarse.len()
    }

    /// Fine cells assigned to coarse cell `ci`, as an f32 exact for any
    /// count a hex torus can produce (≤ 65 535 by a wide margin: ≈56 at
    /// r250). The weight a coarse cell carries whenever an intensive
    /// coarse quantity (mm per fine cell) has to be turned back into mass
    /// or into a map mean — `atmosphere::coarse` is the caller, see its
    /// doc on the accounting.
    #[must_use]
    pub(crate) fn fine_count_f32(&self, ci: usize) -> f32 {
        let count = self.coarse_offsets[ci + 1] - self.coarse_offsets[ci];
        f32::from(u16::try_from(count).expect("fine cells per coarse cell fits u16"))
    }

    /// The coarse cell containing each fine cell: the exact partition,
    /// and with it the coarse → fine **broadcast** `out_i =
    /// coarse[fine_to_coarse[i]]`.
    ///
    /// That broadcast is the exact adjoint of [`Self::aggregate_mean`],
    /// and the reason `atmosphere::coarse` uses the pair rather than
    /// [`Self::interpolate_scalar`] for a stock: mean-then-broadcast is an
    /// exactly conservative round trip — `Σ_{i∈c} out_i = N_c ×
    /// coarse_c`, and `out_i ≥ 0` whenever `coarse_c ≥ 0`. Those two
    /// properties are what make the fine fields a *partition* of the
    /// coarse stock, so a fine pass can at worst empty its own cell and
    /// can never take a coarse cell below zero. Barycentric interpolation
    /// has neither property; see [`Self::interpolate_scalar`]'s doc for
    /// what that cost, measured.
    #[must_use]
    pub(crate) fn fine_to_coarse(&self) -> &[usize] {
        &self.fine_to_coarse
    }

    /// Generic version of [`Self::aggregate_temperature`]: CSR mean of an
    /// arbitrary fine scalar field (`fine`, fine-sized) into `out`
    /// (coarse-sized, pre-sized by the caller — not resized here, same
    /// contract as `aggregate_temperature` writing into
    /// `self.grid.cells_slice_mut()`). Same gather, same worker split,
    /// same ascending-fine-index summation order, so a caller that reads
    /// per-cell fields directly (like `aggregate_temperature` does for
    /// `temperature`) and one that goes through this generic slice-based
    /// path produce bit-identical sums for the same values.
    ///
    /// # Panics
    /// Debug builds only: `fine.len()` or `out.len()` doesn't match this
    /// mesh's fine/coarse sizes.
    pub(crate) fn aggregate_mean(&self, fine: &[f32], out: &mut [f32]) {
        debug_assert_eq!(fine.len(), self.fine_to_coarse.len());
        debug_assert_eq!(out.len(), self.coarse_len());
        let coarse_to_fine = &self.coarse_to_fine;
        let coarse_offsets = &self.coarse_offsets;
        let inv_count = &self.inv_count;
        for_each_chunk_mut_over(out, fine.len(), |start, chunk| {
            for (local, o) in chunk.iter_mut().enumerate() {
                let ci = start + local;
                let mut sum = 0.0_f32;
                for &fi in &coarse_to_fine[coarse_offsets[ci]..coarse_offsets[ci + 1]] {
                    sum += fine[fi];
                }
                *o = sum * inv_count[ci];
            }
        });
    }

    /// **Mass-weighted content** of each coarse cell: `Σ_{i∈c} x_i² /
    /// Σ_{i∈c} x_i`, the mean of `x` weighted by `x` itself — what a unit
    /// of the quantity sits in on average, where [`Self::aggregate_mean`]
    /// gives what a column holds on average. Same gather, same worker
    /// split and same ascending-fine-index summation order as
    /// `aggregate_mean`.
    ///
    /// This is the **in-cloud water content** `q_in` of
    /// `atmosphere::coarse`, the number KK2000's super-linear
    /// autoconversion is evaluated on. It replaced (2026-09-07, #158) a
    /// counting cloud fraction `f_c = #{x_i > 0} / N_c` with `q_in =
    /// q_c / f_c`, which weighed a column holding 0.003 mm of diffusion
    /// skirt exactly as much as one holding 1.0 mm of cumulonimbus: at
    /// r8, one loaded column ringed by six skirt columns gave `f_c` =
    /// 7/11 and `q_in` = 0.143 mm, under the 0.15 mm floor, so a 1 mm
    /// cloud rained nothing. Weighting by mass, the same distribution
    /// gives `q_in` = 0.982 mm.
    ///
    /// Properties the caller's conservation argument rests on, for any
    /// non-negative field (`atmosphere::coarse::step_moist_precip` and
    /// the tests below):
    /// - `q_in ≥ q_c` by Cauchy-Schwarz (`(Σx)² ≤ N·Σx²`), with equality
    ///   exactly when `x` is constant over the cell — so the implied
    ///   fraction `q_c / q_in = (Σx)² / (N·Σx²)` lands in `(0, 1]` with
    ///   no clamp. It is the participation ratio of the distribution,
    ///   and on a **two-valued** field (`0` or one constant `a`) it is
    ///   exactly the counting fraction `k / N` this replaced — the old
    ///   closure is the special case where the old closure was right;
    /// - `q_in ≤ max_{i∈c} x_i`, a weighted mean of the `x_i`;
    /// - `0.0` when `Σx ≤ 0`: an empty cell has no content to report and
    ///   nothing to divide by. Not a fallback value and not a floor —
    ///   the caller skips such a cell entirely, there being no mass to
    ///   drain out of it.
    ///
    /// # Panics
    /// Debug builds only: `fine.len()` or `out.len()` doesn't match this
    /// mesh's fine/coarse sizes.
    pub(crate) fn aggregate_mass_weighted_content(&self, fine: &[f32], out: &mut [f32]) {
        debug_assert_eq!(fine.len(), self.fine_to_coarse.len());
        debug_assert_eq!(out.len(), self.coarse_len());
        let coarse_to_fine = &self.coarse_to_fine;
        let coarse_offsets = &self.coarse_offsets;
        for_each_chunk_mut_over(out, fine.len(), |start, chunk| {
            for (local, o) in chunk.iter_mut().enumerate() {
                let ci = start + local;
                let mut sum = 0.0_f32;
                let mut sum_sq = 0.0_f32;
                for &fi in &coarse_to_fine[coarse_offsets[ci]..coarse_offsets[ci + 1]] {
                    let x = fine[fi];
                    sum += x;
                    sum_sq += x * x;
                }
                *o = if sum > 0.0 { sum_sq / sum } else { 0.0 };
            }
        });
    }

    /// Generic version of [`Self::interpolate_wind`]: barycentric
    /// interpolation of an arbitrary coarse scalar field (`coarse`,
    /// coarse-sized) onto the fine grid, same 3-weight-per-cell scheme as
    /// [`Self::sample_scalar`]/`interpolate_wind`. `out` (fine-sized) must
    /// already be sized by the caller — not resized here, same contract
    /// as [`Self::aggregate_mean`].
    ///
    /// # Panics
    /// Debug builds only: `out.len()` doesn't match this mesh's fine size.
    ///
    /// **Not usable for a stock**, and the coarse upper layer measured it
    /// rather than assuming it: the moist layer's step 2 first rebuilt its
    /// fine views with this operator, and the water budget grew by 98 % in
    /// ten years at r3 (1 749 → 3 479). Mechanism: the weights sum to 1
    /// only up to the toric seam's rounding, so `Σ_{i∈c} out_i ≠ N_c ×
    /// coarse_c` and a fine cell can be handed more than its coarse cell
    /// owns; the conservative fine passes then take a coarse cell below
    /// zero, the view goes negative (measured −1e-2 to −3e-2 mm), and every
    /// `.max(0.0)` f32 rounding guard downstream (`uplift`'s apply,
    /// `advection`'s apply, the cloud diffusion) turns that into water out
    /// of nothing, hour after hour, always positive.
    /// [`Self::broadcast_scalar`] is the operator with the right
    /// properties for a stock. This one stays for the display and wind
    /// paths, where the smooth field is the point and nothing is being
    /// conserved.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no production caller since the coarse moist layer \
                      settled on broadcast_scalar for its views, see above; \
                      the wind path uses interpolate_wind/sample_scalar"
        )
    )]
    pub(crate) fn interpolate_scalar(&self, coarse: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.interp.len());
        let interp = &self.interp;
        for_each_chunk_mut(out, |start, chunk| {
            for (local, o) in chunk.iter_mut().enumerate() {
                *o = interp[start + local]
                    .iter()
                    .map(|&(ci, w)| coarse[ci] * w)
                    .sum();
            }
        });
    }
}

/// CSR inverse of the fine → coarse partition: `(values, offsets)` with
/// the fine indices of coarse cell `ci` at `values[offsets[ci]..
/// offsets[ci + 1]]`, ascending (a counting sort over the fine indices
/// in order, so each list is emitted in ascending fine index).
fn invert_partition(fine_to_coarse: &[usize], n_coarse: usize) -> (Vec<usize>, Vec<usize>) {
    let mut offsets = vec![0_usize; n_coarse + 1];
    for &ci in fine_to_coarse {
        offsets[ci + 1] += 1;
    }
    for ci in 0..n_coarse {
        offsets[ci + 1] += offsets[ci];
    }
    let mut cursor = offsets[..n_coarse].to_vec();
    let mut values = vec![0_usize; fine_to_coarse.len()];
    for (fi, &ci) in fine_to_coarse.iter().enumerate() {
        values[cursor[ci]] = fi;
        cursor[ci] += 1;
    }
    (values, offsets)
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

    /// The CSR lists partition the fine grid and each one is ascending:
    /// the order the gather relies on to replay the serial scatter's
    /// f32 sums bit for bit.
    #[test]
    fn coarse_to_fine_lists_are_ascending_and_partition_the_fine_grid() {
        let fine = fine_grid(30);
        let mesh = SynopticMesh::build(&fine);
        let n_coarse = mesh.grid().len();
        assert_eq!(mesh.coarse_offsets.len(), n_coarse + 1);
        assert_eq!(mesh.coarse_offsets[n_coarse], fine.len());
        let mut seen = vec![false; fine.len()];
        for ci in 0..n_coarse {
            let list = &mesh.coarse_to_fine[mesh.coarse_offsets[ci]..mesh.coarse_offsets[ci + 1]];
            assert!(!list.is_empty(), "coarse cell {ci} has no fine cell");
            assert!(
                list.windows(2).all(|w| w[0] < w[1]),
                "coarse cell {ci} not ascending"
            );
            for &fi in list {
                assert_eq!(mesh.fine_to_coarse[fi], ci);
                assert!(!seen[fi], "fine cell {fi} listed twice");
                seen[fi] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "a fine cell is in no list");
    }

    /// The gather reproduces the historical serial scatter bit for bit
    /// on a noisy temperature field.
    #[test]
    fn aggregation_matches_the_serial_scatter_bit_for_bit() {
        let mut fine = fine_grid(30);
        let mut state = 0x1234_5678_u32;
        for c in fine.cells_slice_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            c.temperature = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 40.0 - 20.0;
        }
        let mut mesh = SynopticMesh::build(&fine);
        let mut scatter = vec![0.0_f32; mesh.grid().len()];
        for (cell, &ci) in fine.cells_slice().iter().zip(&mesh.fine_to_coarse) {
            scatter[ci] += cell.temperature;
        }
        mesh.aggregate_temperature(&fine);
        for (ci, c) in mesh.grid().cells_slice().iter().enumerate() {
            let expected = scatter[ci] * mesh.inv_count[ci];
            assert_eq!(
                c.temperature.to_bits(),
                expected.to_bits(),
                "coarse cell {ci}"
            );
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

    /// Generic [`SynopticMesh::aggregate_mean`]/[`SynopticMesh::interpolate_scalar`],
    /// tested independently of `aggregate_temperature`/`interpolate_wind`
    /// (moist-layer coarse mirror, `atmosphere::coarse`, step 1).
    ///
    /// At the identity mesh (`Rc = R`): each fine cell maps to itself with
    /// weight 1 and no other term (`identity_mesh_maps_each_cell_to_itself_with_weight_one`),
    /// so both operators must return the input verbatim, bit for bit — a
    /// single-term sum times `inv_count = 1.0` for the gather, a single
    /// `w0 = 1.0` term for the interpolation, neither introduces rounding.
    #[test]
    fn generic_operators_are_bit_exact_at_identity() {
        let mut fine = fine_grid(6);
        let mut state = 0xabcd_1234_u32;
        for c in fine.cells_slice_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            c.temperature = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 40.0 - 20.0;
        }
        let mesh = SynopticMesh::identity(&fine);
        let field: Vec<f32> = fine.cells_slice().iter().map(|c| c.temperature).collect();

        let mut aggregated = vec![0.0_f32; mesh.coarse_len()];
        mesh.aggregate_mean(&field, &mut aggregated);
        assert_eq!(
            aggregated, field,
            "aggregate_mean must be the identity bit for bit"
        );

        let mut interpolated = vec![0.0_f32; fine.len()];
        mesh.interpolate_scalar(&field, &mut interpolated);
        assert_eq!(
            interpolated, field,
            "interpolate_scalar must be the identity bit for bit"
        );
    }

    /// The three bounds `atmosphere::coarse`'s conservation argument
    /// rests on, on a real mesh (r30, `Rc = 4`, ≈45.8 fine cells per
    /// coarse) and a spiky non-negative field (a fifth of the columns
    /// exactly zero, a fifth loaded, the rest a thin skirt — the
    /// distribution the lever was written for):
    /// `q_c ≤ q_in ≤ max_{i∈c} x_i`, and the implied fraction
    /// `q_c / q_in` inside `(0, 1]` with nothing clamping it.
    ///
    /// The lower bound is Cauchy-Schwarz, exact in ℝ, and it is asserted
    /// here with **no tolerance** — measured, not assumed. In f32 it is
    /// only exact away from a uniform cell: `q_in = Σx²/Σx` and
    /// `q_c = Σx · inv_count` are two different roundings of the same
    /// number when every column of the cell holds the same value, so a
    /// flat field could put `q_c` one ULP above `q_in`. Nothing downstream
    /// depends on it (the conservation argument needs `d ≤ q_in` alone,
    /// see `atmosphere::coarse::drain_fine_cloud_by_coarse_cell`); the
    /// assertion is here to catch a closure that stops being a weighted
    /// mean, not to guard a rounding.
    #[test]
    fn mass_weighted_content_is_between_the_box_mean_and_the_peak() {
        let fine = fine_grid(30);
        let mut state = 0xfeed_1234_u32;
        let field: Vec<f32> = (0..fine.len())
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let u = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0;
                match state % 5 {
                    0 => 0.0,
                    1 => u * 4.0,
                    _ => u * 0.004,
                }
            })
            .collect();
        let mesh = SynopticMesh::build(&fine);
        assert_eq!(mesh.grid().radius(), 4, "r30 must give Rc = 4");

        let mut mean = vec![0.0_f32; mesh.coarse_len()];
        mesh.aggregate_mean(&field, &mut mean);
        let mut in_cloud = vec![0.0_f32; mesh.coarse_len()];
        mesh.aggregate_mass_weighted_content(&field, &mut in_cloud);

        let mut concentrated = 0_usize;
        for ci in 0..mesh.coarse_len() {
            let members =
                &mesh.coarse_to_fine[mesh.coarse_offsets[ci]..mesh.coarse_offsets[ci + 1]];
            let peak = members.iter().map(|&fi| field[fi]).fold(0.0_f32, f32::max);
            let (q_c, q_in) = (mean[ci], in_cloud[ci]);
            assert!(
                q_in >= q_c,
                "coarse cell {ci}: in-cloud {q_in} below the box mean {q_c}"
            );
            assert!(
                q_in <= peak,
                "coarse cell {ci}: in-cloud {q_in} above the peak column {peak}"
            );
            let implied_fraction = q_c / q_in;
            assert!(
                implied_fraction > 0.0 && implied_fraction <= 1.0,
                "coarse cell {ci}: implied fraction {implied_fraction} outside (0, 1]"
            );
            if implied_fraction < 0.5 {
                concentrated += 1;
            }
        }
        // The fixture is only meaningful if the field really is spiky:
        // a flat one would make every bound above an equality.
        assert!(
            concentrated > mesh.coarse_len() / 2,
            "the fixture must be concentrated: {concentrated} of {} cells under \
             an implied fraction of 0.5",
            mesh.coarse_len()
        );
    }

    /// On a **two-valued** field (`0` or one constant `a`) the
    /// mass-weighted content is exactly `a`, and the fraction it implies
    /// (`q_c / q_in`) is exactly the counting fraction `k / N_c` it
    /// replaced. The old closure is the special case where the old
    /// closure was right, which is what makes this lever a
    /// generalisation and not a re-tune.
    ///
    /// `a = 0.75` and the identity mesh's exclusion: r6, `Rc = 2`, 127
    /// fine cells over 19 coarse.
    #[test]
    fn mass_weighted_content_reproduces_the_counting_fraction_on_a_two_valued_field() {
        const A: f32 = 0.75;
        let fine = fine_grid(6);
        let mesh = SynopticMesh::with_coarse_radius(&fine, 2);
        assert!(mesh.coarse_len() < fine.len(), "not the identity mesh");
        let mut state = 0x1234_5678_u32;
        let field: Vec<f32> = (0..fine.len())
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if state.is_multiple_of(3) { A } else { 0.0 }
            })
            .collect();

        let mut mean = vec![0.0_f32; mesh.coarse_len()];
        mesh.aggregate_mean(&field, &mut mean);
        let mut in_cloud = vec![0.0_f32; mesh.coarse_len()];
        mesh.aggregate_mass_weighted_content(&field, &mut in_cloud);

        let mut mixed = 0_usize;
        for ci in 0..mesh.coarse_len() {
            let members =
                &mesh.coarse_to_fine[mesh.coarse_offsets[ci]..mesh.coarse_offsets[ci + 1]];
            let k = members.iter().filter(|&&fi| field[fi] > 0.0).count();
            if k == 0 {
                assert_eq!(in_cloud[ci].to_bits(), 0.0_f32.to_bits(), "empty cell {ci}");
                continue;
            }
            if k < members.len() {
                mixed += 1;
            }
            assert_eq!(
                in_cloud[ci].to_bits(),
                A.to_bits(),
                "coarse cell {ci}: {} instead of the one loaded value {A}",
                in_cloud[ci]
            );
            let counting = f32::from(u16::try_from(k).unwrap()) * mesh.inv_count[ci];
            let implied = mean[ci] / in_cloud[ci];
            assert!(
                (implied - counting).abs() <= 1e-6,
                "coarse cell {ci}: implied fraction {implied} vs counting {counting} \
                 ({k} of {})",
                members.len()
            );
        }
        assert!(
            mixed > 0,
            "the fixture must hold coarse cells that are only partly loaded"
        );
    }

    /// The degenerate case, stated rather than absorbed: a coarse cell
    /// whose fine values sum to zero reports `0.0`, and so does one whose
    /// values are all zero. The caller
    /// (`atmosphere::coarse::drain_fine_cloud_by_coarse_cell`) skips such
    /// a cell — there is no mass to drain out of it — instead of dividing
    /// by zero and instead of substituting a fallback content.
    #[test]
    fn mass_weighted_content_is_zero_where_there_is_nothing() {
        let fine = fine_grid(6);
        let mesh = SynopticMesh::with_coarse_radius(&fine, 2);
        let field = vec![0.0_f32; fine.len()];
        let mut out = vec![f32::NAN; mesh.coarse_len()];
        mesh.aggregate_mass_weighted_content(&field, &mut out);
        for (ci, &v) in out.iter().enumerate() {
            assert_eq!(v.to_bits(), 0.0_f32.to_bits(), "coarse cell {ci}: {v}");
        }
    }

    /// The measurement that named the lever (#158, JOURNAL 2026-09-07),
    /// as arithmetic: one column loaded with 1.0 mm, six neighbours
    /// carrying the 0.003 mm skirt the cloud diffusion leaves, four dry —
    /// the r8 distribution under `phys_rain_footprint_is_a_disc`. Counting
    /// columns gives `f_c` = 7/11 and an in-cloud content of 0.143 mm,
    /// under the 0.15 mm `precip_crit_mm` floor. Weighting by mass gives
    /// 0.982 mm, and the floor does not bite.
    #[test]
    fn a_diffusion_skirt_no_longer_swallows_the_cloud_it_surrounds() {
        let cloudy: Vec<f32> = std::iter::once(1.0_f32)
            .chain(std::iter::repeat_n(0.003_f32, 6))
            .chain(std::iter::repeat_n(0.0_f32, 4))
            .collect();
        assert_eq!(cloudy.len(), 11, "the r8 coarse cell of the fixture");

        let n = 11.0_f32;
        let sum: f32 = cloudy.iter().sum();
        let sum_sq: f32 = cloudy.iter().map(|x| x * x).sum();
        let q_c = sum / n;
        let counting_fraction = 7.0 / n;
        let counting_content = q_c / counting_fraction;
        let mass_weighted = sum_sq / sum;

        assert!(
            counting_content < 0.15,
            "the counting closure must land under the 0.15 mm floor, got {counting_content}"
        );
        assert!(
            (counting_content - 0.1454).abs() < 1e-3,
            "counting content {counting_content}"
        );
        assert!(
            (mass_weighted - 0.9824).abs() < 1e-3,
            "mass-weighted content {mass_weighted}"
        );
        assert!(
            mass_weighted > 0.15,
            "the mass-weighted closure must clear the floor, got {mass_weighted}"
        );
    }

    /// `aggregate_mean` reproduces the historical serial scatter bit for
    /// bit, the same property `aggregation_matches_the_serial_scatter_bit_for_bit`
    /// proves for `aggregate_temperature`, but on an arbitrary scalar
    /// field with no tie to `CellProperties` (the moist layer's
    /// `humidity_upper`/`cloud_water` are the intended callers,
    /// `atmosphere::coarse::MoistCoarseState::gather_from_fine`).
    #[test]
    fn aggregate_mean_matches_the_serial_scatter_bit_for_bit() {
        let fine = fine_grid(30);
        let mut state = 0x0bad_f00d_u32;
        let field: Vec<f32> = (0..fine.len())
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 10.0
            })
            .collect();
        let mesh = SynopticMesh::build(&fine);
        let mut scatter = vec![0.0_f32; mesh.coarse_len()];
        for (&x, &ci) in field.iter().zip(&mesh.fine_to_coarse) {
            scatter[ci] += x;
        }
        let mut out = vec![0.0_f32; mesh.coarse_len()];
        mesh.aggregate_mean(&field, &mut out);
        for (ci, &got) in out.iter().enumerate() {
            let expected = scatter[ci] * mesh.inv_count[ci];
            assert_eq!(got.to_bits(), expected.to_bits(), "coarse cell {ci}");
        }
    }

    /// Conservation and constant-field behavior at a real `Rc < R` ratio:
    /// r30 gives `Rc = 4` (45.8 fine cells per coarse cell on average, the
    /// exact configuration `physics_mountain_precipitation` will exercise
    /// in step 4, cf. design note §9 risk 2).
    ///
    /// Two properties measured, not assumed:
    /// - **Sum conservation** is exact up to float rounding: `Σ_c mean_c ×
    ///   count_c` must equal `Σ_i x_i` to a small *relative* tolerance
    ///   (1e-6), not bit for bit (a mean-then-remultiply round trip through
    ///   `inv_count = 1/count` is not generally exact, see the next point).
    /// - **A constant field is NOT reproduced bit for bit by the mean** at
    ///   `Rc < R` in general: `sum = Σ(count times) v` then `mean = sum ×
    ///   (1/count)` is an exact round trip only for the (value, count)
    ///   pairs where `1/count` happens to invert cleanly in f32 (measured
    ///   directly: `v=3.25` round-trips exactly for `count` in
    ///   {4,5,45,46,67} but drifts by 1 ULP at {60,61}; `v=12.345` (not a
    ///   dyadic fraction) drifts by up to ~1e-6 relative at every count
    ///   tested ≥ 45). So the identity-only bit-exactness above is the
    ///   real invariant; here we assert the actual measured bound instead
    ///   of a false "bit for bit at any Rc" claim.
    #[test]
    fn aggregate_mean_conserves_the_sum_at_rc_lt_r() {
        let fine = fine_grid(30);
        let mesh = SynopticMesh::build(&fine);
        assert_eq!(
            mesh.grid().radius(),
            4,
            "r30 must give Rc = 4 (design note §6/§9)"
        );
        let n_coarse = mesh.coarse_len();

        let mut state = 0x9e37_79b9_u32;
        let field: Vec<f32> = (0..fine.len())
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 10.0
            })
            .collect();
        let mut mean = vec![0.0_f32; n_coarse];
        mesh.aggregate_mean(&field, &mut mean);

        let mut count = vec![0.0_f64; n_coarse];
        for &ci in &mesh.fine_to_coarse {
            count[ci] += 1.0;
        }
        let reconstructed: f64 = mean
            .iter()
            .zip(&count)
            .map(|(&m, &c)| f64::from(m) * c)
            .sum();
        let total: f64 = field.iter().map(|&x| f64::from(x)).sum();
        let rel_err = (reconstructed - total).abs() / total.abs();
        assert!(rel_err < 1e-6, "relative conservation error {rel_err}");

        // Constant field: measured bound, not bit-for-bit (see doc above).
        let constant = 12.345_f32;
        let field_c = vec![constant; fine.len()];
        let mut mean_c = vec![0.0_f32; n_coarse];
        mesh.aggregate_mean(&field_c, &mut mean_c);
        let max_gap = mean_c
            .iter()
            .map(|&m| (m - constant).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_gap < 1e-4,
            "aggregate_mean of a constant field, max gap {max_gap} (measured)"
        );

        // Interpolated back to the fine grid: partition-of-unity residual
        // (cf. `mapping_is_a_partition_and_weights_are_convex`'s own 1e-5
        // bound on Σw), same order of magnitude as the aggregate step
        // above.
        let mut fine_out = vec![0.0_f32; fine.len()];
        mesh.interpolate_scalar(&mean_c, &mut fine_out);
        let max_gap_fine = fine_out
            .iter()
            .map(|&v| (v - constant).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_gap_fine < 1e-3,
            "interpolate_scalar of a constant field, max gap {max_gap_fine} (measured)"
        );
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
