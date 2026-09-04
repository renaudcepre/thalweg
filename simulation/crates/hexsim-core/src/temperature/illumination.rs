//! Per-cell illumination: shadow raymarch (relief occlusion + cloud
//! shadow) that turns a [`super::solar::SolarBeam`] into a `flux_factor`
//! per cell, plus [`IllumCache`], the terrain precomputation that makes
//! the production path (`compute_illumination_cached`) cheap.
//!
//! Split out of the former single `temperature.rs` (#52 pattern): this is
//! the geometry/occlusion half of the radiative pipeline, computed once per
//! tick and consumed by [`super::balance::step_temperature`] via the
//! `flux_factor` it writes — kept separate from the astronomy
//! ([`super::solar`], cell-independent) and from the energy balance itself
//! ([`super::balance`], local per cell, no raymarch).

use crate::ablation::Ablation;
use crate::coord::hex_direction_to_world;
use crate::dynamics::CELL_SPACING_M;
use crate::grid::HexGrid;

use super::{SolarBeam, TemperatureParams, solar_beam_at_tick};

/// Diffuse fraction of the clear-sky global irradiance (dimensionless,
/// ~15-25 % under clear sky, Duffie & Beckman 2013 §2.10). The direct
/// beam depends on the incidence angle and is blocked by upstream
/// relief; the diffuse component comes from the whole sky dome and
/// reaches a slope turned away from the sun or a cell in a relief
/// shadow alike, weighted by its sky-view factor (isotropic sky model,
/// Liu & Jordan 1960). Before 2026-09-02 the diffuse part was only
/// applied under relief occlusion, and a slope facing away from the
/// sun received nothing at all: at 130 m spacing (relief 8x steeper
/// than at the 1074 m calibration) that zero on every ubac and every
/// occluded valley collapsed the map-wide lapse rate to ~1 °C/km and
/// turned the coldest cells into permanent condensers (bisected to
/// e3594f9 + d6be105, JOURNAL 2026-09-02).
pub const DIFFUSE_SKY_FRACTION: f32 = 0.2;

/// Flux factor of a tilted surface under a clear sky partially hidden
/// by relief: `(1 − D)·direct + D·diffuse`, with the direct beam
/// `cos_inc·(1 − t)` (fully blocked at `t = 1`) and the isotropic
/// diffuse component `s_u·(1 + n_u)/2` (sky-view factor of a plane of
/// upward normal component `n_u`). Written as `direct + D·(diffuse −
/// direct)` so that a flat unoccluded cell (`cos_inc = s_u`, `n_u = 1`,
/// `t = 0`) yields exactly `s_u`, the historical horizontal beam,
/// bit-identical.
#[inline]
#[must_use]
fn tilted_flux_factor(s_u: f32, n_u: f32, cos_inc: f32, occlusion_t: f32) -> f32 {
    let direct = cos_inc * (1.0 - occlusion_t);
    let diffuse = s_u * 0.5 * (1.0 + n_u);
    direct + DIFFUSE_SKY_FRACTION * (diffuse - direct)
}

/// Max number of steps in the shadow march (beyond this, potential
/// obstruction is negligible and the cost climbs). Step = 1 cell.
const ILLUM_MAX_STEPS: usize = 64;

/// Compiled-in default for the raymarch ablation switch: OFF. Perf
/// measurement only, never active by default.
pub(crate) const ILLUM_KO_DEFAULT: bool = false;

/// Raymarch ablation switch (`HEXSIM_ILLUM_KO=1`). Delegates to
/// [`Ablation::effective`], which reads the environment once for the
/// whole process.
fn illum_ko() -> bool {
    Ablation::effective().illum_ko
}
/// Elevation gain (m) of an upstream relief above the solar ray that
/// gives full occlusion; soft penumbra below that.
const ILLUM_FULL_M: f32 = 30.0;

/// Hex direction most aligned with the horizontal sun + ray slope
/// (`tan(elevation) = s_u / ‖s_horiz‖`). `None` if the sun is too
/// close to the zenith to define an azimuth (`‖s_horiz‖ < 1e-6`), no
/// march then, local cloud. Shared by the reference march and the
/// cached path so the 6-direction quantization stays identical in
/// both.
fn sun_march_geometry(beam: &SolarBeam) -> Option<(usize, f32)> {
    let horiz = (beam.s_e * beam.s_e + beam.s_n * beam.s_n).sqrt();
    if horiz < 1e-6 {
        return None;
    }
    let ray_slope = beam.s_u / horiz; // tan(elevation)
    // World: x=East, y=South = -North. Argmax of the dot product →
    // direction index (usize), no float→int cast.
    let (sx, sy) = (beam.s_e, -beam.s_n);
    let mut sun_dir = 0_usize;
    let mut best = f32::NEG_INFINITY;
    for k in 0..6 {
        let (dx, dy) = hex_direction_to_world(k);
        let dot = dx * sx + dy * sy;
        if dot > best {
            best = dot;
            sun_dir = k;
        }
    }
    Some((sun_dir, ray_slope))
}

/// Illumination pass (#102, final): for each cell, computes
/// `flux_factor = max(0, S⃗·N⃗) · occlusion · cloud_transmission`
/// (physics: absorbed flux = `beam · flux_factor`) and its display
/// counterpart `illumination` ∈ `[0,1]` (fraction of full sun for a
/// flat, clear, cloudless cell).
///
/// - **aspect**: `max(0, S⃗·N⃗)` against the local normal (sunny
///   slope/shaded slope).
/// - **relief occlusion**: march toward the sun on the elevation grid;
///   an upstream relief that exceeds the ray darkens it (toward the
///   diffuse floor).
/// - **cloud shadow**: samples `cloud_water` at the layer crossing
///   (distance `d = H/tan(elevation)`, shifted farther out when the
///   sun is low).
///
/// Marches in **integer hex coordinates** via `neighbor_indices_toric`:
/// native toric wrap (the world has no edge), zero float→cell
/// conversion. The solar azimuth is quantized to the nearest hex
/// direction (coarse v1, 6 orientations, to refine into a 2-direction
/// DDA). Night (`s_u ≤ 0`) → `flux_factor = 0`, `illumination = 1`
/// (darkness is handled by scene lighting, not albedo).
///
/// **Role since #65**: executable specification. Production goes
/// through [`compute_illumination_cached`] (same outputs, proven
/// bit-identical by `tests/phys_illum_cache_equiv.rs`); this naive
/// march remains the readable reference and the arbiter for the
/// equivalence micro-test. Any evolution of the illumination physics
/// happens HERE first, the cached path follows.
pub fn compute_illumination(
    grid: &HexGrid,
    beam: &SolarBeam,
    cloud_albedo_coef: f32,
    cloud_altitude_m: f32,
    flux_factor: &mut Vec<f32>,
    illumination: &mut Vec<f32>,
) {
    let cells = grid.cells_slice();
    let n = cells.len();
    flux_factor.clear();
    flux_factor.resize(n, 0.0);
    illumination.clear();
    illumination.resize(n, 1.0);
    if beam.s_u <= 0.0 {
        return; // sun below the horizon
    }
    let march = sun_march_geometry(beam);
    let has_azimuth = march.is_some();
    let (sun_dir, ray_slope) = march.unwrap_or((0, 0.0));

    let max_elev = cells
        .iter()
        .map(|c| c.elevation)
        .fold(f32::NEG_INFINITY, f32::max);
    let d_cloud = if has_azimuth {
        cloud_altitude_m / ray_slope
    } else {
        0.0
    };
    // Ablation switch (perf measurement): `HEXSIM_ILLUM_KO=1`
    // short-circuits the raymarch (relief occlusion + shifted cloud
    // shadow), illumination becomes aspect × LOCAL cloud. Temporary,
    // for the visual A/B.
    let ko_raymarch = illum_ko();
    for i in 0..n {
        let (ne, nn) = (cells[i].normal_east, cells[i].normal_north);
        let n_u = (1.0 - ne * ne - nn * nn).max(0.0).sqrt();
        let cos_inc = (beam.s_e * ne + beam.s_n * nn + beam.s_u * n_u).max(0.0);
        let cy = cells[i].elevation;
        let mut over = 0.0_f32; // max exceedance of the ray by an upstream relief
        let mut eff_cloud = cells[i].cloud_water; // default: local (zenith) cloud
        // Slope facing away from the sun (`cos_inc = 0`): no direct beam
        // to occlude, no march; it still receives the diffuse sky below.
        if cos_inc > 0.0 && has_azimuth && !ko_raymarch {
            let mut idx = i;
            let mut dist = 0.0_f32;
            // Sub-cell cloud shadow (offset < 1 cell, sun high) → keep
            // the LOCAL cloud (the cloud overhead darkens its own
            // cell); lateral projection only matters when the sun is
            // low (offset ≥ 1 cell).
            let mut cloud_sampled = d_cloud < CELL_SPACING_M;
            for _ in 0..ILLUM_MAX_STEPS {
                idx = grid.neighbor_indices_toric(idx)[sun_dir];
                dist += CELL_SPACING_M;
                let ray_h = cy + dist * ray_slope;
                over = over.max(cells[idx].elevation - ray_h);
                // Cell whose footprint contains the layer crossing
                // (the one closest to d_cloud, not the first exceeded).
                if !cloud_sampled && dist + 0.5 * CELL_SPACING_M >= d_cloud {
                    eff_cloud = cells[idx].cloud_water;
                    cloud_sampled = true;
                }
                if ray_h >= max_elev && cloud_sampled {
                    break; // nothing else can occlude, cloud sampled
                }
            }
        }
        let t = (over / ILLUM_FULL_M).min(1.0);
        let cover = eff_cloud.clamp(0.0, 1.0); // cloud_water normalized to 1 mm PW
        let transm = 1.0 - (cover * cloud_albedo_coef).min(0.95);
        let ff = tilted_flux_factor(beam.s_u, n_u, cos_inc, t) * transm;
        flux_factor[i] = ff;
        illumination[i] = (ff / beam.s_u).clamp(0.0, 1.0);
    }
}

/// Levels of the doubling jump tables: `2^0 … 2^6` steps, enough to
/// compose any offset `≤ ILLUM_MAX_STEPS` (guaranteed by the const
/// assert).
const ILLUM_SHIFT_LEVELS: usize = 7;
const _: () = assert!(ILLUM_MAX_STEPS == 1 << (ILLUM_SHIFT_LEVELS - 1));

/// Upper bound (m) on the f32 rounding error of the reference march on
/// `elev − (cy + dist·slope)`: magnitudes ≤ ~25,000 m ⇒ error ≤ ~5 mm;
/// 5 cm = 10x margin. The precomputed tangents are biased by this
/// margin so that "no occlusion" / "full occlusion" decided in f64
/// imply the same result as the f32 march, cells within the margin
/// simply fall into the penumbra band and march as before.
const ILLUM_EXACT_MARGIN_M: f64 = 0.05;

/// Terrain precomputations for `compute_illumination_cached` (#65).
/// The shadow raymarch mixes two dynamics: **relief** occlusion
/// (function of terrain alone, immutable outside erosion) and
/// **cloud** shadow shifting (fresh every tick). This cache freezes
/// everything that depends only on terrain, per cell and per hex
/// direction:
///
/// - `s_clear`: solar tangent above which NONE of the 64 upstream
///   steps occludes (`over = 0`, march unnecessary);
/// - `s_full`: tangent below which occlusion is FULL
///   (`over ≥ ILLUM_FULL_M`, `t = 1` without marching);
/// - `dir_max`: max elevation of the 64 upstream steps, a tight stop
///   bound for the residual march (penumbra band between the two
///   tangents);
/// - `shift`: doubling jump tables (`2^j` toric steps) to sample the
///   cloud at the layer crossing in O(popcount) instead of marching
///   to it.
///
/// Measured (`tests/perf_illum_march_stats.rs`, r45 seed 42): 92 to
/// 99.9% of lit cells exit via one of the two tangents depending on
/// the hour, the march only remains for the penumbra band.
///
/// **Invalidation**: terrain only moves through erosion,
/// [`crate::simulation::Simulation`] calls [`IllumCache::mark_dirty`]
/// at the same spot as the surface normal recompute. `ensure` then
/// rebuilds on the next tick. The cache is tied to the last grid seen
/// by `ensure`.
#[derive(Debug, Clone)]
pub struct IllumCache {
    /// Number of grid cells at the last `rebuild`.
    len: usize,
    /// Terrain changed since the last `rebuild` (or never built).
    dirty: bool,
    /// `shift[d][j][i]` = cell `2^j` toric steps from `i` in direction `d`.
    shift: [[Vec<usize>; ILLUM_SHIFT_LEVELS]; 6],
    /// Tangent (f64, `+ILLUM_EXACT_MARGIN_M` bias) of "clear sky" by (dir, cell).
    s_clear: [Vec<f64>; 6],
    /// Tangent (f64, `−ILLUM_EXACT_MARGIN_M` bias) of "full shadow" by (dir, cell).
    s_full: [Vec<f64>; 6],
    /// Max elevation of the `ILLUM_MAX_STEPS` upstream steps by (dir, cell).
    dir_max: [Vec<f32>; 6],
    /// Flat elevations (contiguous copy, march without touching the bulky `CellProperties`).
    elev: Vec<f32>,
}

impl Default for IllumCache {
    fn default() -> Self {
        Self {
            len: 0,
            dirty: true,
            shift: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            s_clear: std::array::from_fn(|_| Vec::new()),
            s_full: std::array::from_fn(|_| Vec::new()),
            dir_max: std::array::from_fn(|_| Vec::new()),
            elev: Vec::new(),
        }
    }
}

impl IllumCache {
    /// Empty cache, rebuilt on the first [`IllumCache::ensure`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Signals that elevation changed (erosion): the next `ensure`
    /// rebuilds. O(1).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Rebuilds if needed (cache never built, terrain modified via
    /// [`IllumCache::mark_dirty`], or grid size changed). Rebuild
    /// cost: `6 dirs × ILLUM_MAX_STEPS × n` flat reads, on the order
    /// of a single tick of the old march, paid once per relief change
    /// (never in steady state without erosion).
    pub fn ensure(&mut self, grid: &HexGrid) {
        if !self.dirty && self.len == grid.len() {
            return;
        }
        self.rebuild(grid);
    }

    fn rebuild(&mut self, grid: &HexGrid) {
        let n = grid.len();
        self.len = n;
        self.dirty = false;
        self.elev.clear();
        self.elev
            .extend(grid.cells_slice().iter().map(|c| c.elevation));
        let spacing = f64::from(CELL_SPACING_M);
        for d in 0..6 {
            // Level 0: 1 toric step. Following levels by composition.
            let lvl0: Vec<usize> = (0..n).map(|i| grid.neighbor_indices_toric(i)[d]).collect();
            self.shift[d][0] = lvl0;
            for j in 1..ILLUM_SHIFT_LEVELS {
                let (built, rest) = self.shift[d].split_at_mut(j);
                let prev = &built[j - 1];
                let cur = &mut rest[0];
                cur.clear();
                cur.extend((0..n).map(|i| prev[prev[i]]));
            }
            // Tangents + upstream max: single march of ILLUM_MAX_STEPS steps.
            let step1 = &self.shift[d][0];
            let s_clear = &mut self.s_clear[d];
            let s_full = &mut self.s_full[d];
            let dir_max = &mut self.dir_max[d];
            s_clear.clear();
            s_full.clear();
            dir_max.clear();
            for i in 0..n {
                let cy = f64::from(self.elev[i]);
                let mut idx = i;
                let mut dist = 0.0_f64;
                let mut best_clear = f64::NEG_INFINITY;
                let mut best_full = f64::NEG_INFINITY;
                let mut dmax = f32::NEG_INFINITY;
                for _ in 0..ILLUM_MAX_STEPS {
                    idx = step1[idx];
                    dist += spacing;
                    let e = self.elev[idx];
                    dmax = dmax.max(e);
                    let b = f64::from(e) - cy;
                    best_clear = best_clear.max((b + ILLUM_EXACT_MARGIN_M) / dist);
                    best_full =
                        best_full.max((b - f64::from(ILLUM_FULL_M) - ILLUM_EXACT_MARGIN_M) / dist);
                }
                s_clear.push(best_clear);
                s_full.push(best_full);
                dir_max.push(dmax);
            }
        }
    }
}

/// Step (1..=`ILLUM_MAX_STEPS`) whose footprint contains the cloud
/// layer crossing, exactly replicates the f32 discovery of the
/// reference march (`dist + 0.5·step ≥ d_cloud`, `dist` accumulated in
/// f32). `0` = local cloud: sub-cell crossing (sun high) or layer
/// never reached within `ILLUM_MAX_STEPS` steps (sun very low).
fn cloud_sample_step(d_cloud: f32) -> usize {
    if d_cloud < CELL_SPACING_M {
        return 0;
    }
    let mut dist = 0.0_f32;
    for step in 1..=ILLUM_MAX_STEPS {
        dist += CELL_SPACING_M;
        if dist + 0.5 * CELL_SPACING_M >= d_cloud {
            return step;
        }
    }
    0
}

/// Production illumination pass (#65): same outputs as
/// [`compute_illumination`] (the equivalence micro-test compares them
/// bit for bit), separating the two dynamics the reference march
/// conflates:
///
/// - **relief** (immutable outside erosion): decided by the
///   precomputed tangents of the [`IllumCache`], the march only
///   remains for the penumbra band (`s_full < slope < s_clear`, a few
///   % of cells), on flat arrays with a stop at `dir_max`;
/// - **cloud** (fresh every tick): sampled in O(popcount) via the
///   jump tables, at the same step as the reference march.
///
/// # Panics
/// If the cache doesn't match the grid (`ensure` forgotten after a
/// relief change), a stale cache would silently produce wrong
/// shadows, we prefer to fail loudly.
pub fn compute_illumination_cached(
    grid: &HexGrid,
    beam: &SolarBeam,
    cloud_albedo_coef: f32,
    cloud_altitude_m: f32,
    cache: &IllumCache,
    flux_factor: &mut Vec<f32>,
    illumination: &mut Vec<f32>,
) {
    let cells = grid.cells_slice();
    let n = cells.len();
    assert!(
        !cache.dirty && cache.len == n,
        "IllumCache stale (dirty={}, len={} vs grid {n}): missing ensure() call",
        cache.dirty,
        cache.len
    );
    flux_factor.clear();
    flux_factor.resize(n, 0.0);
    illumination.clear();
    illumination.resize(n, 1.0);
    if beam.s_u <= 0.0 {
        return; // sun below the horizon
    }
    let march = sun_march_geometry(beam).filter(|_| !illum_ko());
    let kstar = match march {
        Some((_, ray_slope)) => cloud_sample_step(cloud_altitude_m / ray_slope),
        None => 0,
    };
    for i in 0..n {
        let (ne, nn) = (cells[i].normal_east, cells[i].normal_north);
        let n_u = (1.0 - ne * ne - nn * nn).max(0.0).sqrt();
        let cos_inc = (beam.s_e * ne + beam.s_n * nn + beam.s_u * n_u).max(0.0);
        let mut t = 0.0_f32; // normalized occlusion = (over / ILLUM_FULL_M).min(1)
        let mut eff_cloud = cells[i].cloud_water; // default: local (zenith) cloud
        // Same gate as the reference march: a slope facing away from
        // the sun has no direct beam to occlude, only the diffuse sky.
        if let Some((sun_dir, ray_slope)) = march.filter(|_| cos_inc > 0.0) {
            // Cloud shadow: cell at the layer crossing, via 2^j jumps.
            if kstar > 0 {
                let mut idx = i;
                let mut bits = kstar;
                let mut level = 0;
                while bits != 0 {
                    if bits & 1 == 1 {
                        idx = cache.shift[sun_dir][level][idx];
                    }
                    bits >>= 1;
                    level += 1;
                }
                eff_cloud = cells[idx].cloud_water;
            }
            // Relief occlusion: precomputed tangents, march only in penumbra.
            let slope_w = f64::from(ray_slope);
            if slope_w >= cache.s_clear[sun_dir][i] {
                // no upstream step occludes: over = 0, t = 0 (majority path)
            } else if slope_w <= cache.s_full[sun_dir][i] {
                t = 1.0; // full occlusion guaranteed: the march would give min(1)
            } else {
                let cy = cache.elev[i];
                let dmax = cache.dir_max[sun_dir][i];
                let step1 = &cache.shift[sun_dir][0];
                let mut over = 0.0_f32;
                let mut idx = i;
                let mut dist = 0.0_f32;
                for _ in 0..ILLUM_MAX_STEPS {
                    idx = step1[idx];
                    dist += CELL_SPACING_M;
                    let ray_h = cy + dist * ray_slope;
                    over = over.max(cache.elev[idx] - ray_h);
                    if ray_h >= dmax {
                        break; // nothing tall enough left in this direction
                    }
                }
                t = (over / ILLUM_FULL_M).min(1.0);
            }
        }
        let cover = eff_cloud.clamp(0.0, 1.0); // cloud_water normalized to 1 mm PW
        let transm = 1.0 - (cover * cloud_albedo_coef).min(0.95);
        let ff = tilted_flux_factor(beam.s_u, n_u, cos_inc, t) * transm;
        flux_factor[i] = ff;
        illumination[i] = (ff / beam.s_u).clamp(0.0, 1.0);
    }
}

/// Day-of-year sampling stride for
/// [`terrain_annual_mean_insolation_factor`]: every 7th day, 24h each
/// (~53 days × 24h instead of the full 365 × 24h annual sweep, ~7x
/// cheaper: ~0.03 s instead of ~0.2 s at r30 release). The annual
/// insolation cycle is a smooth trig curve (`daily_insolation_factor`),
/// so weekly sampling barely moves the mean (checked against the full
/// sweep by the unit test below). Mirrors the sweep in
/// `tests/diag_illumination_budget.rs`.
const TERRAIN_INSOLATION_SAMPLE_STRIDE_DAYS: u16 = 7;

/// Annual mean ratio `flux_factor / s_u` (dimensionless) the REAL
/// terrain lets through, against the flat horizontal beam `s_u`: the
/// full illumination pass ([`compute_illumination_cached`], relief
/// occlusion + diffuse sky, clouds ignored via `cloud_albedo_coef =
/// 0.0`) rather than [`super::aspect_insolation_correction`]'s pure-tilt
/// approximation (no shadow march, no diffuse-sky mixing). This is the
/// number [`TemperatureParams::terrain_insolation_factor`] carries into
/// `calibration_offset` so `base_temp` is reached on the actual relief
/// instead of an assumed flat one (JOURNAL 2026-09-02/03).
///
/// A flat, unoccluded world returns EXACTLY 1.0: there,
/// `tilted_flux_factor` reduces to `s_u` bit-for-bit (`n_u = 1`,
/// `cos_inc = s_u`, `t = 0`), so `flux_factor == s_u` for every cell and
/// every daylight hour.
///
/// Sampled every `TERRAIN_INSOLATION_SAMPLE_STRIDE_DAYS` days (see its
/// doc for the cost/accuracy tradeoff). `cache` must already be
/// [`IllumCache::ensure`]d against `grid` (same contract as
/// [`compute_illumination_cached`]).
///
/// # Panics
/// Same as [`compute_illumination_cached`]: a stale `cache` (missing
/// `ensure()` after a relief change) panics rather than silently
/// mismeasuring the terrain.
#[must_use]
pub fn terrain_annual_mean_insolation_factor(
    grid: &HexGrid,
    cache: &IllumCache,
    params: &TemperatureParams,
) -> f32 {
    let n = grid.len();
    if n == 0 {
        return 1.0;
    }
    let cells_per_hour = f64::from(u32::try_from(n).expect("grid size fits u32"));
    let mut flux_factor = Vec::with_capacity(n);
    let mut illumination = Vec::with_capacity(n);
    let mut flux_sum = 0.0_f64;
    let mut beam_sum = 0.0_f64;
    let mut day = 0_u16;
    while day < 365 {
        for hour in 0..24_u64 {
            let hour_tick = u64::from(day) * 24 + hour;
            let beam = solar_beam_at_tick(params, hour_tick);
            if beam.s_u <= 0.0 {
                continue; // night: flux = 0 everywhere, ratio unaffected
            }
            // Clouds ignored (`cloud_albedo_coef = 0.0`): this is a
            // terrain property (relief + sky geometry), not a weather
            // one, and mirrors `diag_illumination_budget.rs`.
            compute_illumination_cached(
                grid,
                &beam,
                0.0,
                1500.0, // cloud altitude: irrelevant at cloud_albedo_coef = 0
                cache,
                &mut flux_factor,
                &mut illumination,
            );
            let cell_flux: f64 = flux_factor.iter().map(|&f| f64::from(f)).sum();
            flux_sum += cell_flux;
            beam_sum += f64::from(beam.s_u) * cells_per_hour;
        }
        day += TERRAIN_INSOLATION_SAMPLE_STRIDE_DAYS;
    }
    if beam_sum <= 0.0 {
        return 1.0; // polar night at every sampled hour: degenerate, keep flat behavior
    }
    let ratio = flux_sum / beam_sum;
    // Ratio of two positive sums of physically bounded terms
    // (flux_factor ≤ beam, both in [0, ~1.25]): `ratio` itself sits in
    // that same narrow band, nowhere near f32's ~7-digit precision
    // limit. The narrowing is the field's own SI unit (dimensionless
    // factor stored as f32 across the codebase, see
    // `TemperatureParams::terrain_insolation_factor`), not a precision
    // bug.
    #[expect(clippy::cast_possible_truncation)]
    let ratio_f32 = ratio as f32;
    ratio_f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::temperature::compute_surface_normals;

    /// A slope turned away from the sun (or a cell in a relief shadow)
    /// still receives the diffuse sky: the flux factor never drops to
    /// zero in daylight, and a flat unoccluded cell keeps the exact
    /// horizontal beam (JOURNAL 2026-09-02, lapse-rate collapse at
    /// 130 m spacing).
    #[test]
    fn tilted_flux_keeps_the_diffuse_sky() {
        let s_u = 0.5_f32; // sun at 30° elevation
        // Flat, unoccluded: exactly the horizontal beam, bit-identical.
        assert_eq!(
            tilted_flux_factor(s_u, 1.0, s_u, 0.0).to_bits(),
            s_u.to_bits()
        );
        // Flat, fully occluded: the diffuse fraction survives.
        let flat_shaded = tilted_flux_factor(s_u, 1.0, s_u, 1.0);
        assert!((flat_shaded - DIFFUSE_SKY_FRACTION * s_u).abs() < 1e-6);
        // 30° slope facing away from the sun (`cos_inc = 0`): diffuse
        // only, weighted by its sky-view factor, strictly positive.
        let n_u = 30.0_f32.to_radians().cos();
        let away = tilted_flux_factor(s_u, n_u, 0.0, 0.0);
        let expected = DIFFUSE_SKY_FRACTION * s_u * 0.5 * (1.0 + n_u);
        assert!(
            away > 0.0 && (away - expected).abs() < 1e-6,
            "{away} vs {expected}"
        );
        // The same slope facing the sun receives more than flat, and
        // more than when turned away.
        let toward = tilted_flux_factor(s_u, n_u, 0.9, 0.0);
        assert!(toward > s_u && toward > away);
        // Occlusion removes the direct beam only: the shaded slope
        // converges to the diffuse floor, never below it.
        let shaded = tilted_flux_factor(s_u, n_u, 0.9, 1.0);
        assert!((shaded - expected).abs() < 1e-6, "{shaded} vs {expected}");
    }

    /// `terrain_annual_mean_insolation_factor` on a FLAT world: every
    /// cell has a vertical normal and nothing occludes anything, so
    /// `flux_factor == s_u` every daylight hour (bit-for-bit, see
    /// `tilted_flux_keeps_the_diffuse_sky`) and the ratio must be
    /// EXACTLY 1.0, not merely close. Pins the multiplicative identity
    /// `calibration_offset` relies on: default params
    /// (`terrain_insolation_factor = 1.0`) must reproduce the
    /// historical flat-terrain offset bit-for-bit.
    #[test]
    fn flat_world_terrain_factor_is_exactly_one() {
        let mut grid = HexGrid::from_radius(2);
        compute_surface_normals(&mut grid); // all-flat -> vertical normals
        let mut cache = IllumCache::new();
        cache.ensure(&grid);
        let params = TemperatureParams::default();
        let factor = terrain_annual_mean_insolation_factor(&grid, &cache, &params);
        assert_eq!(factor.to_bits(), 1.0_f32.to_bits(), "flat world: {factor}");
    }

    /// A built ridge (steep enough to self-shadow, mirrors
    /// `tests/phys_aspect_insolation.rs`) must let LESS of the flat beam
    /// through than a flat world: the factor is strictly below 1.0,
    /// never above (`aspect_insolation_correction`'s pure-tilt term can
    /// be positive OR negative, but occlusion + the diffuse-sky floor
    /// only ever remove flux). This is the deficit
    /// `terrain_insolation_factor` exists to carry into
    /// `calibration_offset` (JOURNAL 2026-09-02/03).
    #[test]
    fn steep_ridge_terrain_factor_is_below_one() {
        let radius = 3;
        let mut grid = HexGrid::from_radius(radius);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            let r = f32::from(i16::try_from(coord.r.abs()).unwrap_or(0));
            // Steep enough to self-shadow at CELL_SPACING_M (130 m):
            // slope = 120/130 rad ≈ 43°, well above the ~29° map-mean
            // measured at r30 (JOURNAL 2026-09-03).
            grid.get_mut(coord).unwrap().elevation = 500.0 - 120.0 * r;
        }
        compute_surface_normals(&mut grid);
        let mut cache = IllumCache::new();
        cache.ensure(&grid);
        let params = TemperatureParams::default();
        let factor = terrain_annual_mean_insolation_factor(&grid, &cache, &params);
        assert!(
            factor < 1.0 && factor > 0.0,
            "steep ridge must let through strictly less than the flat beam, got {factor}"
        );
    }

    #[test]
    fn illumination_relief_occludes_toward_sun() {
        // Sun in the east, 45° elevation. A cell with a tall wall to the EAST
        // (toward the sun) receives less than the same cell without a wall
        // (occlusion), but keeps the diffuse floor (never absolute zero).
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let beam = SolarBeam {
            s_e: s,
            s_n: 0.0,
            s_u: s,
            beam: 1000.0,
        };
        let center = HexCoord::new(0, 0);
        let east = HexCoord::new(1, 0);

        let mut grid = HexGrid::from_radius(2);
        if let Some(c) = grid.get_mut(east) {
            c.elevation = 4000.0;
        }
        let ci = grid.index_of(center).unwrap();
        let (mut ff, mut il) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, 1500.0, &mut ff, &mut il);
        let occluded = ff[ci];

        if let Some(c) = grid.get_mut(east) {
            c.elevation = 0.0; // wall removed, same world
        }
        let (mut ff2, mut il2) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, 1500.0, &mut ff2, &mut il2);

        assert!(
            occluded < ff2[ci],
            "occluded {occluded} should be < clear {}",
            ff2[ci]
        );
        assert!(occluded > 0.0, "diffuse floor preserved, got {occluded}");
    }

    #[test]
    fn illumination_cloud_dims_along_ray() {
        // Sun east/45°: d_cloud = cloud_altitude_m/tan45 = cloud_altitude_m,
        // so taking cloud_altitude_m = CELL_SPACING_M, the nearest cell is
        // the 1st east neighbor regardless of the engine's resolution. A
        // cloud THERE (not above the center) dims the cell: lateral cloud
        // shadow.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let beam = SolarBeam {
            s_e: s,
            s_n: 0.0,
            s_u: s,
            beam: 1000.0,
        };
        let center = HexCoord::new(0, 0);
        let cloud_cell = HexCoord::new(1, 0);

        let mut grid = HexGrid::from_radius(3);
        if let Some(c) = grid.get_mut(cloud_cell) {
            c.cloud_water = 1.0;
        }
        let ci = grid.index_of(center).unwrap();
        let (mut ff, mut il) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, CELL_SPACING_M, &mut ff, &mut il);
        let shaded = ff[ci];

        if let Some(c) = grid.get_mut(cloud_cell) {
            c.cloud_water = 0.0;
        }
        let (mut ff2, mut il2) = (Vec::new(), Vec::new());
        compute_illumination(&grid, &beam, 0.5, CELL_SPACING_M, &mut ff2, &mut il2);

        assert!(
            shaded < ff2[ci],
            "lateral cloud should dim: {shaded} vs {}",
            ff2[ci]
        );
    }
}
