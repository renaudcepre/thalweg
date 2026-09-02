use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};
use serde::Deserialize;

use crate::cell::CellProperties;
use crate::coord::{HexCoord, torus_lattice_vectors};
use crate::erosion::{ErosionParams, erode_terrain};
use crate::grid::HexGrid;
use crate::lithology::{self, LithologyId};

// Terrain generation parameters.
//
// Terrain is generated from 3 combined layers:
// 1. Continental base: low-frequency Fbm + domain warping (organic shapes)
// 2. Ridges: RidgedMulti (sharp ridge lines)
// 3. Erosive detail: Swiss noise (elongated valleys, ravines)
//
// The final formula blends the layers by elevation:
// low areas → smooth base, high areas → ridges + erosive detail.
//
// All layers are sampled through the 4D toroidal embedding
// (`TorusNoiseMapping`): the field is exactly periodic under the tiling
// lattice, so terrain stitches together with no seam across the wrap
// (`HexGrid::wrap_target`), a requirement for the full torus (the
// translation wrap joins edges whose relief must stay continuous).
#[derive(Clone)]
pub struct TerrainParams {
    pub seed: u32,
    pub elevation_scale: f32,
    pub base_temperature: f32,
    pub lapse_rate: f32,
    pub initial_water: f32,
    pub initial_humidity: f32,
    pub initial_groundwater: f32,
    pub permeability_seed_offset: u32,
    pub permeability_frequency: f64,
    pub permeability_altitude_bias: f32,

    // Layer 1: continental base (Fbm + domain warping)
    pub continent_frequency: f64,
    pub continent_octaves: usize,
    pub continent_persistence: f64,
    pub continent_warp: f64,

    // Layer 2: ridges (RidgedMulti)
    pub ridge_frequency: f64,
    pub ridge_octaves: usize,
    pub ridge_blend_min: f32,
    pub ridge_blend_max: f32,

    // Layer 3: erosive detail (Swiss noise)
    pub swiss_frequency: f64,
    pub swiss_octaves: usize,
    pub swiss_lacunarity: f64,
    pub swiss_gain: f64,
    pub swiss_warp: f64,
    pub detail_scale: f32,

    // Post-process: plateaus and plains
    pub plateau_factor: f32,
    pub plain_smoothing: f32,

    // Elevation offset applied after shape_elevation. Since the Fbm noise is
    // centered around 0, without an offset the median is negative and the
    // apparent lapse rate collapses. We recenter the grid around a
    // "continental plain" elevation.
    pub base_elevation_offset: f32,

    // water_capacity derived from local topology. Formula:
    //   cap = max(min, base + α * max(0, mean_delta) - β * rugosity)
    // where mean_delta = mean(n.elev - self.elev) (>0 = basin) and
    // rugosity = std_dev(deltas).
    pub water_capacity_base: f32,
    pub water_capacity_min: f32,
    pub water_capacity_cuvette_bonus: f32,
    pub water_capacity_slope_penalty: f32,

    /// Number of **one-shot** fluvial erosion iterations applied to the
    /// bare terrain at generation time (#105): each pass routes the flow
    /// (drained area) and incises the channels (SI stream power),
    /// rerouting each time → a dendritic network digs itself in through
    /// feedback, then the elevation is FROZEN (no live erosion by
    /// default, no drift toward flat). 0 = raw generator terrain. Network
    /// maturity grows with this number (each iteration incises ≤ 25% of
    /// the local drop, CFL cap). Hot-reloadable server-side
    /// (`terrain.erosion_iterations`), applied on the next `reset`.
    pub erosion_iterations: u32,
    /// Carving strength of the one-shot "capture" erosion: divided by
    /// 100, this is the number of METERS removed per iteration at the
    /// strongest channel (the others in proportion to their discharge).
    /// Default 2000 → 20 m/iteration. Higher = more aggressive capture
    /// (sharper channels) but a more lowered terrain. Hot-reloadable
    /// (`terrain.erosion_accel`), applied on the next `reset`.
    /// (Historical name `_accel_years` kept so as not to break the
    /// server key; the semantics are now "carving strength".)
    pub erosion_accel_years: f32,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            seed: 42,
            // Recalibrated 2000 → 2400 with the move to toroidal (4D)
            // terrain. The embedding samples the continent's fundamental
            // octave on a small circle (diameter 2ρf ≈ 0.23 noise cell at
            // R=30) → variance compressed vs. flat 2D sampling,
            // hypsometry tails flattened (max 1574 m, no more permanent
            // snow). 2400 restores the hypsometry measured on the 2D
            // world at R=30 (scale-test standard): max 1789 m (2D:
            // 1807), std 417 (2D: 402), >1500 m: 30 cells (2D: 13).
            elevation_scale: 2400.0,
            base_temperature: 20.0,
            lapse_rate: 6.5,
            initial_water: 1.0,
            initial_humidity: 0.1,
            initial_groundwater: 1.5,
            permeability_seed_offset: 1000,
            permeability_frequency: 0.08,
            permeability_altitude_bias: 0.8,

            continent_frequency: 0.008,
            continent_octaves: 3,
            continent_persistence: 0.5,
            continent_warp: 0.4,

            ridge_frequency: 0.015,
            ridge_octaves: 3,
            ridge_blend_min: 0.4,
            ridge_blend_max: 0.75,

            swiss_frequency: 0.025,
            swiss_octaves: 3,
            swiss_lacunarity: 2.0,
            swiss_gain: 0.5,
            swiss_warp: 0.35,
            detail_scale: 120.0,

            plateau_factor: 0.4,
            plain_smoothing: 0.6,

            base_elevation_offset: 500.0,

            // Phase 3 (#32) rescaled water_level ×200 but forgot
            // water_capacity accordingly. Average soil capacity of
            // 0.5 mm is physically absurd (real soil absorbs 50-300 mm
            // before free water appears). With the open-water fix (Meyer
            // restricted to water_level > water_capacity), an under-
            // calibrated soil means every trace of moisture is instantly
            // treated as a lake → massive evaporation everywhere. Rescale
            // ×200 (memory project_atmo_physical_units: "flagged debt"):
            // typical soil 100 mm (loamy soil), 10 mm minimum (rock).
            water_capacity_base: 100.0,
            water_capacity_min: 10.0,
            water_capacity_cuvette_bonus: 4.0,
            water_capacity_slope_penalty: 3.0,

            // One-shot "capture" erosion: OPT-IN (0 = off by default). The
            // setting the author eyeball-validated (20 passes, force 3000
            // → 30 m/pass at the max channel) carves nice canyons BUT
            // shaves off the peaks → less orographic uplift → dries out
            // the world (rain r10 3874 < 5000, breaks
            // `world_does_not_become_a_desert`). So it isn't imposed on
            // every world by default: enable with
            // `terrain.erosion_iterations 20` + `terrain.erosion_accel 3000`
            // then `reset` when the carved relief is wanted. Default =
            // calibrated/humid world (JOURNAL 2026-07-11).
            erosion_iterations: 0,
            erosion_accel_years: 3000.0,
        }
    }
}

/// Toroidal embedding of the hexagonal domain for noise: lattice
/// coordinates (a, b) ∈ [0, 1)² such that `p = a·v1 + b·v2` (tiling
/// vectors, `torus_lattice_vectors`), then a 4D embedding onto two
/// circles, one per torus axis. Any noise sampled on this embedding is
/// **exactly periodic** under the lattice: the terrain on either side of
/// the seam is the same field, not a patched join.
///
/// Circle radius ρ = |V1|/2π with |V1| = |V2| = √(3N) (N = 3R²+3R+1): the
/// arc length along a circle equals the world distance along the
/// corresponding axis, so the layers' frequencies keep the same feature
/// size as the flat 2D sampling they replace. Residual distortion: the
/// embedding treats (V1, V2) as orthogonal even though they are 60°
/// apart → bounded anisotropy (~±22% along the diagonals), with no seam
/// or discontinuity.
struct TorusNoiseMapping {
    /// N = 3R²+3R+1: lattice determinant = number of cells in the domain.
    cell_count: i64,
    radius: i64,
    /// Δworld → Δ(a, b): inverse of the matrix [V1 V2] (world columns).
    inv: [[f64; 2]; 2],
    /// ρ = √(3N)/(2π).
    circle_radius: f64,
}

impl TorusNoiseMapping {
    fn new(radius: i32) -> Self {
        let [v1, v2, _] = torus_lattice_vectors(radius);
        let (w1x, w1y) = hex_to_world(v1);
        let (w2x, w2y) = hex_to_world(v2);
        let det = w1x * w2y - w2x * w1y;
        let inv = [[w2y / det, -w2x / det], [-w1y / det, w1x / det]];
        let r = i64::from(radius);
        let cell_count = 3 * r * r + 3 * r + 1;
        #[expect(clippy::cast_precision_loss)] // N ≤ ~1e10 ≪ 2^52: exact
        let circle_radius = (3.0 * cell_count as f64).sqrt() / std::f64::consts::TAU;
        Self {
            cell_count,
            radius: r,
            inv,
            circle_radius,
        }
    }

    /// Lattice coordinates (a, b) ∈ [0, 1)² of a cell: exact solution of
    /// `[q; r] = a·v1 + b·v2` via `M⁻¹ = (1/N)·[[R+1, -R], [R, 2R+1]]`,
    /// with numerators **reduced modulo N as integers** before the
    /// floating-point division. Two coordinates congruent modulo the
    /// lattice (on either side of the seam) thus give bit-for-bit
    /// identical (a, b) → terrain rigorously identical across the wrap,
    /// not just "close within epsilon".
    fn lattice_of(&self, coord: HexCoord) -> (f64, f64) {
        let q = i64::from(coord.q);
        let r = i64::from(coord.r);
        let a_num = ((self.radius + 1) * q - self.radius * r).rem_euclid(self.cell_count);
        let b_num = (self.radius * q + (2 * self.radius + 1) * r).rem_euclid(self.cell_count);
        #[expect(clippy::cast_precision_loss)] // numerators < N: exact in f64
        let (a, b) = (
            a_num as f64 / self.cell_count as f64,
            b_num as f64 / self.cell_count as f64,
        );
        (a, b)
    }

    /// 4D embedding of a cell plus a continuous displacement in world
    /// coordinates (domain warping, finite differences). The
    /// displacement shifts (a, b) linearly → periodicity is preserved as
    /// long as the displacement is itself a periodic function of the
    /// world (guaranteed by sampling the warp on this same embedding).
    fn embed(&self, coord: HexCoord, world_offset: [f64; 2]) -> [f64; 4] {
        // Torus center translated far from the noise's origin: Perlin
        // vanishes at ALL integer points of its lattice, origin included.
        // A low-frequency torus (radius ρ·f ≪ 1 cell) centered at
        // (0,0,0,0) would sample a tiny ball around a zero of the field
        // → crushed amplitude. The components are large (to stay generic
        // after multiplying by each octave's frequency) and
        // irrational-ish (to avoid landing back on a lattice corner).
        // Constant translation → periodicity intact.
        const CENTER: [f64; 4] = [
            1_618.033_988_7,
            2_718.281_828_4,
            3_141.592_653_5,
            4_669.201_609_1,
        ];
        let (a0, b0) = self.lattice_of(coord);
        let a = a0 + self.inv[0][0] * world_offset[0] + self.inv[0][1] * world_offset[1];
        let b = b0 + self.inv[1][0] * world_offset[0] + self.inv[1][1] * world_offset[1];
        let ta = std::f64::consts::TAU * (a - a.floor());
        let tb = std::f64::consts::TAU * (b - b.floor());
        [
            CENTER[0] + self.circle_radius * ta.cos(),
            CENTER[1] + self.circle_radius * ta.sin(),
            CENTER[2] + self.circle_radius * tb.cos(),
            CENTER[3] + self.circle_radius * tb.sin(),
        ]
    }
}

/// Multiplies a 4D point by a spatial frequency. Scaling the embedding
/// (rather than (a, b)) preserves periodicity for any frequency: this is
/// the toroidal equivalent of `p * freq` in flat 2D.
fn scale4(p: [f64; 4], freq: f64) -> [f64; 4] {
    [p[0] * freq, p[1] * freq, p[2] * freq, p[3] * freq]
}

/// Terrain sample for a coordinate (pure function of the seed + lattice).
struct TerrainSample {
    elevation: f32,
    temperature: f32,
    lithology: LithologyId,
    permeability: f32,
}

/// All the noise layers plus the toroidal embedding. Kept separate from
/// `generate_terrain` so tests can evaluate the field at any coordinate
/// (including outside the domain) and check the exact periodicity across
/// the seam.
struct TerrainSampler<'a> {
    params: &'a TerrainParams,
    mapping: TorusNoiseMapping,
    fbm_continent: Fbm<Perlin>,
    fbm_warp: Fbm<Perlin>,
    ridged: RidgedMulti<Perlin>,
    swiss_perlin: Perlin,
    fbm_perm: Fbm<Perlin>,
}

impl<'a> TerrainSampler<'a> {
    fn new(radius: i32, params: &'a TerrainParams) -> Self {
        // Layer 1: continental base (low-frequency Fbm + domain warping)
        let fbm_continent = Fbm::<Perlin>::new(params.seed)
            .set_octaves(params.continent_octaves)
            .set_frequency(params.continent_frequency)
            .set_persistence(params.continent_persistence);

        // Separate noise for the continental layer's domain warping
        let fbm_warp = Fbm::<Perlin>::new(params.seed + 500)
            .set_octaves(2)
            .set_frequency(params.continent_frequency * 1.5)
            .set_persistence(0.5);

        // Layer 2: ridges (RidgedMulti)
        let ridged = RidgedMulti::<Perlin>::new(params.seed + 100)
            .set_octaves(params.ridge_octaves)
            .set_frequency(params.ridge_frequency);

        // Layer 3: Swiss noise (base Perlin, the loop is manual)
        let swiss_perlin = Perlin::new(params.seed + 200);

        // Permeability (unchanged)
        let fbm_perm = Fbm::<Perlin>::new(params.seed + params.permeability_seed_offset)
            .set_octaves(4)
            .set_frequency(params.permeability_frequency)
            .set_persistence(0.5);

        Self {
            params,
            mapping: TorusNoiseMapping::new(radius),
            fbm_continent,
            fbm_warp,
            ridged,
            swiss_perlin,
            fbm_perm,
        }
    }

    fn sample(&self, coord: HexCoord) -> TerrainSample {
        let params = self.params;

        // --- Layer 1: continental base with domain warping ---
        // The warp is itself periodic (sampled on the torus), so the
        // base∘warp composition is too. The (100, 100) offset on the
        // second axis decorrelates warp_y from warp_x, as in flat 2D.
        let warp_x =
            self.fbm_warp.get(self.mapping.embed(coord, [0.0, 0.0])) * params.continent_warp;
        let warp_y =
            self.fbm_warp.get(self.mapping.embed(coord, [100.0, 100.0])) * params.continent_warp;
        let base = self
            .fbm_continent
            .get(self.mapping.embed(coord, [warp_x, warp_y]));

        // --- Layer 2: ridges ---
        // RidgedMulti returns [-1, 1] with a negative bias (only the
        // ridges are high). We shift it up so the mean is ~0 like the base.
        let ridge_raw = self.ridged.get(self.mapping.embed(coord, [0.0, 0.0]));
        let ridge = ridge_raw * 0.5 + 0.5; // remap to [0, 1], ridges on top

        // Blend: the ridges fade in gradually in high areas.
        // base is in [-1, 1], normalized to [0, 1] for the blend.
        #[expect(clippy::cast_possible_truncation)]
        let base_norm = ((base * 0.5 + 0.5) as f32).clamp(0.0, 1.0);
        let blend = smoothstep(
            (base_norm - params.ridge_blend_min)
                / (params.ridge_blend_max - params.ridge_blend_min),
        );

        // Combining base + ridges. Base in [-1,1], ridge in [0,1].
        // We normalize base to [0,1] for the lerp, then remap the result
        // to [-1,1] for the elevation scale.
        #[expect(clippy::cast_possible_truncation)]
        let combined = {
            let base_01 = (base * 0.5 + 0.5) as f32;
            let ridge_01 = ridge as f32;
            let lerped = base_01 * (1.0 - blend) + ridge_01 * blend;
            lerped * 2.0 - 1.0 // back to [-1, 1]
        };

        // --- Layer 3: erosive detail (Swiss noise) ---
        #[expect(clippy::cast_possible_truncation)]
        let detail = swiss_noise(&self.swiss_perlin, &self.mapping, coord, params) as f32;

        // Detail normalization: Swiss noise returns ~[0, octaves],
        // we recenter it around 0 and scale it.
        #[expect(clippy::cast_precision_loss)]
        let detail_centered = (detail / params.swiss_octaves as f32) - 0.5;
        let raw_elevation =
            combined * params.elevation_scale + detail_centered * params.detail_scale;

        // --- Post-process: plateaus and plains ---
        let shaped_elevation = shape_elevation(raw_elevation, params);
        // Offset: recenters the grid so the median doesn't fall below 0.
        let elevation = shaped_elevation + params.base_elevation_offset;

        // Temperature (adiabatic gradient): no clamp, the lapse rate applies
        // at any altitude (100% terrestrial world, no ocean at fixed temperature).
        let temperature = params.base_temperature - (elevation / 1000.0) * params.lapse_rate;

        // --- Mineral substrate (#136, L0) ---
        // The noise no longer carries `permeability` directly: it
        // designates a **rock class**, and it's the `lithology::LITHOLOGY`
        // table that gives the hydric aptitude. Identity refactor: the
        // table is calibrated on the conditional means of this same
        // noise, so the field keeps its mean and spatial pattern (cf.
        // `lithology.rs`).
        let substrate_noise = self.fbm_perm.get(self.mapping.embed(coord, [0.0, 0.0]));
        #[expect(clippy::cast_possible_truncation)]
        let substrate_noise = (substrate_noise * 0.5 + 0.5) as f32;
        // *Relative* altitude (non-offset): preserves the logic "plains =
        // max perm, peaks = less perm" independent of the offset. Also
        // used as a second classification axis (bedrock vs fill).
        let relative_altitude = (shaped_elevation / params.elevation_scale).clamp(0.0, 1.0);
        let lithology = lithology::classify(substrate_noise, relative_altitude);

        // Topographic attenuation, unchanged: soil thins out at altitude,
        // less loose material to infiltrate through. This is a *relief*
        // effect, not rock, so it stays outside the table.
        let soil_factor = 1.0 - relative_altitude * params.permeability_altitude_bias;
        let permeability = (lithology.permeability() * soil_factor).clamp(0.05, 1.0);

        TerrainSample {
            elevation,
            temperature,
            lithology,
            permeability,
        }
    }
}

// Swiss noise: multifractal noise that mimics glacial erosion.
// Accumulates the noise gradients to warp the following octaves
// toward the nearest ridge → elongated valleys, ravines.
//
// Toroidal version: each octave samples the 4D Perlin on the embedding
// scaled by `freq` (periodicity preserved). `dsum` is homogeneous to the
// frequency space of the original 2D version: the accumulated warp is
// converted back to a world displacement (`dsum·warp/freq`), and the
// finite-difference derivatives are taken in the world domain then
// brought back to the frequency scale (step h = EPS/freq, division by
// 2·EPS) to keep the same magnitudes as the original.
fn swiss_noise(
    noise: &Perlin,
    mapping: &TorusNoiseMapping,
    coord: HexCoord,
    params: &TerrainParams,
) -> f64 {
    const EPS: f64 = 0.01;
    let (octaves, lacunarity, gain, warp) = (
        params.swiss_octaves,
        params.swiss_lacunarity,
        params.swiss_gain,
        params.swiss_warp,
    );
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = params.swiss_frequency;
    let mut dsum = [0.0_f64; 2];

    for _ in 0..octaves {
        let off = [dsum[0] * warp / freq, dsum[1] * warp / freq];
        let sample = |ox: f64, oy: f64| -> f64 {
            noise.get(scale4(
                mapping.embed(coord, [off[0] + ox, off[1] + oy]),
                freq,
            ))
        };
        let n = sample(0.0, 0.0);

        // Finite-difference derivatives
        let h = EPS / freq;
        let dx = (sample(h, 0.0) - sample(-h, 0.0)) / (2.0 * EPS);
        let dy = (sample(0.0, h) - sample(0.0, -h)) / (2.0 * EPS);

        let ridge = 1.0 - n.abs();
        sum += amp * ridge;
        dsum[0] += amp * dx * -n;
        dsum[1] += amp * dy * -n;

        amp *= gain * sum.clamp(0.0, 1.0);
        freq *= lacunarity;
    }
    sum
}

fn smoothstep(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn generate_terrain(grid: &mut HexGrid, params: &TerrainParams) {
    let sampler = TerrainSampler::new(grid.radius(), params);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();

    for coord in coords {
        let sample = sampler.sample(coord);
        if let Some(cell) = grid.get_mut(coord) {
            *cell = CellProperties {
                elevation: sample.elevation,
                temperature: sample.temperature,
                water_level: params.initial_water,
                humidity_upper: params.initial_humidity,
                groundwater: params.initial_groundwater,
                lithology: sample.lithology,
                permeability: sample.permeability,
                ..Default::default()
            };
        }
    }

    // One-shot fluvial erosion (#105): sculpts the drainage network into
    // the bare relief, THEN freezes. Before `assign_water_capacity`
    // (depends on final topology: basins carved here gain their capacity
    // bonus), followed by a temperature recompute (elevation changed).
    erode_terrain(
        grid,
        &ErosionParams {
            accel_years_per_day: params.erosion_accel_years,
            ..ErosionParams::for_worldgen()
        },
        params.erosion_iterations,
    );
    if params.erosion_iterations > 0 {
        recompute_temperature(grid, params);
    }

    assign_water_capacity(grid, params);
}

/// Recalibrates temperature to the current elevation (lapse rate), after
/// erosion has changed the relief. The live tick recomputes it anyway via
/// the radiative balance; this is just a consistent initial state.
fn recompute_temperature(grid: &mut HexGrid, params: &TerrainParams) {
    for cell in grid.cells_slice_mut() {
        cell.temperature = params.base_temperature - (cell.elevation / 1000.0) * params.lapse_rate;
    }
}

/// An external elevation override cell (real DEM), indexed by axial
/// coordinate. Format consumed by [`apply_dem_override`].
#[derive(Debug, Clone, Deserialize)]
pub struct DemCellOverride {
    pub q: i32,
    pub r: i32,
    pub elevation: f32,
}

/// Provenance of a DEM survey, written by `scripts/dem_import/dem_to_hexgrid.py`.
/// Without it, a file doesn't say which portion of the globe it describes or
/// at what scale: retroactively re-registering an orphaned
/// `drome_fine_r120.json` cost an entire session — its center wasn't recorded
/// anywhere, and recovering it required correlating the imported heightmap
/// against independent real-world SRTM data.
#[derive(Debug, Clone, Deserialize)]
pub struct DemMeta {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius: i32,
    pub cell_spacing_m: f32,
    pub samples_per_cell: u32,
    pub dataset: String,
}

/// Content of a DEM override file. Two accepted forms:
/// - `{"meta": {...}, "cells": [...]}`: current format, with provenance;
/// - `[{q, r, elevation}, ...]`: legacy format, without provenance.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DemOverrideFile {
    WithMeta {
        meta: DemMeta,
        cells: Vec<DemCellOverride>,
    },
    Legacy(Vec<DemCellOverride>),
}

impl DemOverrideFile {
    #[must_use]
    pub fn cells(&self) -> &[DemCellOverride] {
        match self {
            Self::WithMeta { cells, .. } | Self::Legacy(cells) => cells,
        }
    }

    #[must_use]
    pub fn meta(&self) -> Option<&DemMeta> {
        match self {
            Self::WithMeta { meta, .. } => Some(meta),
            Self::Legacy(_) => None,
        }
    }
}

/// Result of an [`apply_dem_override`]: how many cells from the survey
/// landed in the grid, how many fell outside the domain.
///
/// `skipped > 0` means the survey describes a domain larger than the
/// grid: the simulated world is a **crop** of the survey, not the survey.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemApplyReport {
    pub applied: usize,
    pub skipped: usize,
}

/// Replaces the noise-generated elevation with an external survey (real
/// DEM), recomputes temperature (lapse rate) and `water_capacity`
/// accordingly. Domain cells absent from the override: noise-derived
/// elevation is kept (no hole).
///
/// Override cells outside the domain are ignored but **counted** in the
/// [`DemApplyReport`]: loading a radius-120 survey into a radius-80 grid
/// discards 55%, including, in the real case that motivated this
/// counter, the survey's only mountain.
///
/// **Validation only**: the world stays topologically a torus (`HexGrid`
/// doesn't change), but a real survey isn't periodic: edge cells undergo
/// an elevation discontinuity across the seam (the toroidal neighborhood
/// will see a large delta there). Use it to compare climate/vegetation
/// against real relief, not for a playable world.
pub fn apply_dem_override(
    grid: &mut HexGrid,
    overrides: &[DemCellOverride],
    params: &TerrainParams,
) -> DemApplyReport {
    let mut applied = 0;
    for entry in overrides {
        let coord = HexCoord::new(entry.q, entry.r);
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = entry.elevation;
            cell.temperature =
                params.base_temperature - (entry.elevation / 1000.0) * params.lapse_rate;
            applied += 1;
        }
    }
    assign_water_capacity(grid, params);
    DemApplyReport {
        applied,
        skipped: overrides.len() - applied,
    }
}

// Second pass: water_capacity derived from the neighbors' topology.
// Basin (mean_delta > 0) → bonus, slope (high rugosity) → penalty.
// Toroidal neighborhood: since the terrain is periodic, the elevation
// delta across the seam is just as physical as inside it, so edge
// cells are no longer biased by a neighborhood truncated to 3-4 cells.
fn assign_water_capacity(grid: &mut HexGrid, params: &TerrainParams) {
    let n = grid.len();
    let elevations: Vec<f32> = grid.cells_slice().iter().map(|c| c.elevation).collect();

    let mut caps = vec![params.water_capacity_base; n];
    for (i, cap) in caps.iter_mut().enumerate() {
        let deltas = grid
            .neighbor_indices_toric(i)
            .map(|j| elevations[j] - elevations[i]);
        let mean_delta = deltas.iter().sum::<f32>() / 6.0;
        let variance = deltas.iter().map(|d| (d - mean_delta).powi(2)).sum::<f32>() / 6.0;
        let rugosity = variance.sqrt();

        let cuvette_bonus = mean_delta.max(0.0) * params.water_capacity_cuvette_bonus;
        let slope_penalty = rugosity * params.water_capacity_slope_penalty;
        *cap = (params.water_capacity_base + cuvette_bonus - slope_penalty)
            .max(params.water_capacity_min);
    }

    for (cell, cap) in grid.cells_slice_mut().iter_mut().zip(caps) {
        cell.water_capacity = cap;
    }
}

// Flattens the peaks (plateaus) and the depressions (plains).
fn shape_elevation(raw: f32, params: &TerrainParams) -> f32 {
    let scale = params.elevation_scale;
    let norm = raw / scale; // ~[-1, 1]

    if norm > 0.5 {
        // Plateaus: flattens the peaks toward a ceiling
        let excess = norm - 0.5;
        let damped = excess * (1.0 - params.plateau_factor);
        (0.5 + damped) * scale
    } else if norm < -0.2 {
        // Low plains: softens the hollows
        let depth = -0.2 - norm;
        let damped = depth * (1.0 - params.plain_smoothing);
        (-0.2 - damped) * scale
    } else {
        raw
    }
}

// Converts axial hex coordinates to world coordinates. Used to
// build the toroidal embedding (world lattice vectors, warp
// displacement conversion); the noise itself is no longer sampled
// in flat 2D.
fn hex_to_world(coord: HexCoord) -> (f64, f64) {
    let q = f64::from(coord.q);
    let r = f64::from(coord.r);
    let x = 3_f64.sqrt() * q + (3_f64.sqrt() / 2.0) * r;
    let y = 1.5 * r;
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_terrain_sets_elevation() {
        let mut grid = HexGrid::from_radius(3);
        generate_terrain(&mut grid, &TerrainParams::default());

        // All cells must have a nonzero elevation
        // (statistically impossible for the noise to give exactly 0 everywhere)
        let has_nonzero = grid
            .iter()
            .any(|(_, cell)| cell.elevation.abs() > f32::EPSILON);
        assert!(has_nonzero);
    }

    #[test]
    fn generate_terrain_is_deterministic() {
        let params = TerrainParams::default();

        let mut grid1 = HexGrid::from_radius(3);
        generate_terrain(&mut grid1, &params);

        let mut grid2 = HexGrid::from_radius(3);
        generate_terrain(&mut grid2, &params);

        // Same seed → same terrain
        for (coord, cell1) in grid1.iter() {
            let cell2 = grid2.get(*coord).unwrap();
            assert!(
                (cell1.elevation - cell2.elevation).abs() < f32::EPSILON,
                "Non-deterministic for {coord:?}"
            );
        }
    }

    #[test]
    fn terrain_is_exactly_periodic_across_the_seam() {
        // Heart of the torus (journal 2026-07-03): for each edge cell,
        // every neighbor outside the grid must carry EXACTLY the same
        // terrain sample as the cell it wraps onto. The equality is
        // bit-for-bit: `lattice_of` reduces (a, b) modulo the lattice
        // in integer arithmetic before any floating-point op.
        let radius = 6;
        let grid = HexGrid::from_radius(radius);
        let params = TerrainParams::default();
        let sampler = TerrainSampler::new(radius, &params);
        let center = HexCoord::new(0, 0);

        let mut seam_pairs = 0;
        for &coord in grid.coords_slice() {
            if coord.distance(center) < radius {
                continue;
            }
            for outside in coord.neighbors() {
                if grid.contains(outside) {
                    continue;
                }
                let wrapped = grid
                    .wrap_target(outside)
                    .expect("exact tiling: every edge neighbor must wrap");
                let s_out = sampler.sample(outside);
                let s_in = sampler.sample(wrapped);
                assert_eq!(
                    s_out.elevation.to_bits(),
                    s_in.elevation.to_bits(),
                    "elevation discontinuous at the seam: {outside:?} vs {wrapped:?}"
                );
                assert_eq!(
                    s_out.permeability.to_bits(),
                    s_in.permeability.to_bits(),
                    "permeability discontinuous at the seam: {outside:?} vs {wrapped:?}"
                );
                assert_eq!(
                    s_out.temperature.to_bits(),
                    s_in.temperature.to_bits(),
                    "temperature discontinuous at the seam: {outside:?} vs {wrapped:?}"
                );
                seam_pairs += 1;
            }
        }
        // Sanity check: the edge ring indeed exposes 6R outer edges.
        assert!(
            seam_pairs >= 6 * radius,
            "seam under-tested: {seam_pairs} pairs"
        );
    }

    #[test]
    fn terrain_is_periodic_under_full_lattice_translations() {
        // Periodicity beyond the first ring: translating any cell by
        // any lattice vector (and their combinations) gives back the
        // same sample.
        let radius = 4;
        let params = TerrainParams::default();
        let sampler = TerrainSampler::new(radius, &params);
        let [v1, v2, v3] = torus_lattice_vectors(radius);

        let grid = HexGrid::from_radius(radius);
        for &coord in grid.coords_slice() {
            for v in [v1, v2, v3] {
                for translated in [coord + v, coord - v, coord + v + v] {
                    let a = sampler.sample(coord);
                    let b = sampler.sample(translated);
                    assert_eq!(
                        a.elevation.to_bits(),
                        b.elevation.to_bits(),
                        "not periodic: {coord:?} vs {translated:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn temperature_decreases_with_altitude() {
        let mut grid = HexGrid::from_radius(5);
        let params = TerrainParams {
            elevation_scale: 1000.0,
            ..TerrainParams::default()
        };
        generate_terrain(&mut grid, &params);

        // Find a high cell and a low cell
        let (_, highest) = grid
            .iter()
            .max_by(|a, b| a.1.elevation.partial_cmp(&b.1.elevation).unwrap())
            .unwrap();
        let (_, lowest) = grid
            .iter()
            .min_by(|a, b| a.1.elevation.partial_cmp(&b.1.elevation).unwrap())
            .unwrap();

        // If the highest has a positive elevation, it should be colder
        if highest.elevation > 0.0 && lowest.elevation < highest.elevation {
            assert!(highest.temperature <= lowest.temperature);
        }
    }

    #[test]
    fn shape_elevation_flattens_plateaus() {
        // Raw > 0.5 × scale: the plateau must be brought below raw.
        let params = TerrainParams {
            plateau_factor: 0.4,
            plain_smoothing: 0.6,
            elevation_scale: 2000.0,
            ..TerrainParams::default()
        };
        let raw = 0.9 * params.elevation_scale; // 1800, deep in the plateau
        let shaped = shape_elevation(raw, &params);
        assert!(
            shaped < raw,
            "plateau must be flattened: raw={raw}, shaped={shaped}"
        );
        assert!(
            shaped > 0.5 * params.elevation_scale,
            "plateau must stay above the threshold: shaped={shaped}"
        );
    }

    #[test]
    fn shape_elevation_flattens_plains() {
        // Raw < -0.2 × scale: the plain must be raised toward -0.2*scale.
        let params = TerrainParams {
            plain_smoothing: 0.6,
            elevation_scale: 2000.0,
            ..TerrainParams::default()
        };
        let raw = -0.8 * params.elevation_scale; // -1600, deep in the hollow
        let shaped = shape_elevation(raw, &params);
        assert!(
            shaped > raw,
            "plain must be raised: raw={raw}, shaped={shaped}"
        );
        assert!(
            shaped < -0.2 * params.elevation_scale,
            "plain must stay below the threshold: shaped={shaped}"
        );
    }

    #[test]
    fn shape_elevation_identity_in_middle_range() {
        // Raw between -0.2 and +0.5 × scale: no modification.
        let params = TerrainParams {
            elevation_scale: 2000.0,
            ..TerrainParams::default()
        };
        for &raw_norm in &[-0.1_f32, 0.0, 0.2, 0.4] {
            let raw = raw_norm * params.elevation_scale;
            let shaped = shape_elevation(raw, &params);
            assert!(
                (shaped - raw).abs() < 1e-4,
                "mid-range expected unchanged: raw={raw}, shaped={shaped}"
            );
        }
    }

    /// Permeability formula **from before #136**: the substrate noise
    /// directly carried the hydric aptitude. Kept here, and nowhere
    /// else, as a reference for the identity refactor.
    fn legacy_permeability(sampler: &TerrainSampler, coord: HexCoord) -> f32 {
        let params = sampler.params;
        let noise = sampler
            .fbm_perm
            .get(sampler.mapping.embed(coord, [0.0, 0.0]));
        #[expect(clippy::cast_possible_truncation)]
        let raw = (noise * 0.5 + 0.5) as f32;
        let sample = sampler.sample(coord);
        let shaped = sample.elevation - params.base_elevation_offset;
        let alt = (shaped / params.elevation_scale).clamp(0.0, 1.0);
        (raw * (1.0 - alt * params.permeability_altitude_bias)).clamp(0.05, 1.0)
    }

    /// **The identity refactor test (#136, L0).** Rerouting `permeability`
    /// from anonymous noise to the lithology table must not shift the
    /// physics: the field's mean is preserved by construction (the table
    /// carries the conditional means of the noise it replaces), and the
    /// spatial pattern is too (same bands, from the same noise).
    ///
    /// What does change, and what we accept, is the **quantization**:
    /// three porosity levels instead of a continuum. The bounds below
    /// are the **measured** cost, not a wish, so that a future drift is
    /// visible instead of silent.
    ///
    /// Two regimes, and the difference isn't cosmetic: the table is
    /// calibrated on the **population** of worlds, so a world big enough
    /// to be a faithful sample of it (r45 = 6211 cells, the production
    /// radius) recovers the mean within 0.05%. A 721-cell test world
    /// whose noise runs low (seed 7: mean 0.45 vs. 0.50 in the
    /// population) strays from it by up to ~2.5%: that's sampling, not
    /// a calibration defect. The pin says so explicitly rather than
    /// taking the loosest of the two bounds.
    #[test]
    fn lithology_refactor_preserves_permeability_field() {
        // (radius, max mean drift, min correlation, min retained variance)
        let regimes = [
            (45_i32, 0.005_f32, 0.90_f32, 0.80_f32),
            (15, 0.030, 0.85, 0.65),
        ];

        for (radius, max_drift, min_corr, min_var_ratio) in regimes {
            for seed in [42_u32, 7, 1234] {
                let params = TerrainParams {
                    seed,
                    ..TerrainParams::default()
                };
                let sampler = TerrainSampler::new(radius, &params);
                let grid = HexGrid::from_radius(radius);

                let pairs: Vec<(f32, f32)> = grid
                    .coords()
                    .copied()
                    .map(|c| {
                        (
                            legacy_permeability(&sampler, c),
                            sampler.sample(c).permeability,
                        )
                    })
                    .collect();

                #[expect(clippy::cast_precision_loss)]
                let n = pairs.len() as f32;
                let mean_old = pairs.iter().map(|p| p.0).sum::<f32>() / n;
                let mean_new = pairs.iter().map(|p| p.1).sum::<f32>() / n;

                // 1. Mean preserved: the strong guarantee of calibration.
                let drift = (mean_new - mean_old).abs() / mean_old;
                assert!(
                    drift < max_drift,
                    "r={radius} seed={seed}: permeability mean drifted by {:.2} % \
                     (old {mean_old:.4}, new {mean_new:.4})",
                    drift * 100.0
                );

                let v_old = pairs.iter().map(|p| (p.0 - mean_old).powi(2)).sum::<f32>() / n;
                let v_new = pairs.iter().map(|p| (p.1 - mean_new).powi(2)).sum::<f32>() / n;
                let cov = pairs
                    .iter()
                    .map(|p| (p.0 - mean_old) * (p.1 - mean_new))
                    .sum::<f32>()
                    / n;

                // 2. Spatial pattern preserved: the field stays the same
                //    hydric landscape, just quantized.
                let corr = cov / (v_old.sqrt() * v_new.sqrt());
                assert!(
                    corr > min_corr,
                    "r={radius} seed={seed}: spatial pattern lost, correlation {corr:.3}"
                );

                // 3. Quantization only ever removes variance, never adds
                //    any: adding some would signal a table offset from the noise.
                let ratio = v_new / v_old;
                assert!(
                    (min_var_ratio..=1.05).contains(&ratio),
                    "r={radius} seed={seed}: retained variance {ratio:.3} outside the measured \
                     quantization cost"
                );
            }
        }
    }

    /// The four classes must actually populate the world: a class that
    /// never appears is a dead class, and the threshold that makes it
    /// disappear is a calibration bug (not a cosmetic detail: L2/L3 will
    /// depend on it).
    #[test]
    fn every_lithology_class_is_present_in_a_world() {
        use crate::lithology::{LITHOLOGY_COUNT, LithologyId};

        let mut grid = HexGrid::from_radius(45);
        generate_terrain(&mut grid, &TerrainParams::default());

        let mut counts = [0_usize; LITHOLOGY_COUNT];
        for (_, cell) in grid.iter() {
            counts[cell.lithology.index()] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 0, "index class {i} absent from the world: {counts:?}");
        }
        // Granite is the rarest class (bedrock at altitude) but must
        // remain real terrain, not a handful of cells.
        let total: usize = counts.iter().sum();
        let granite = counts[LithologyId::Granite.index()];
        #[expect(clippy::cast_precision_loss)]
        let share = granite as f32 / total as f32;
        assert!(
            share > 0.02,
            "granite nearly absent ({:.1} %): bedrock would no longer support the ridges",
            share * 100.0
        );
    }

    /// Lithology is **static**: set at tick 0, never moved by generation
    /// erosion (exhumation is deferred, cf. design). Deterministic by
    /// seed, like everything else in the world.
    #[test]
    fn lithology_is_deterministic_for_a_seed() {
        let mut a = HexGrid::from_radius(10);
        let mut b = HexGrid::from_radius(10);
        generate_terrain(&mut a, &TerrainParams::default());
        generate_terrain(&mut b, &TerrainParams::default());
        for (coord, cell) in a.iter() {
            assert_eq!(
                cell.lithology,
                b.get(*coord).unwrap().lithology,
                "lithology non-deterministic at {coord:?}"
            );
        }
    }

    #[test]
    fn permeability_is_in_valid_bounds() {
        // On a generated grid, every cell must have permeability
        // in [0.05, 1.0] (final clamp of generate_terrain).
        let mut grid = HexGrid::from_radius(5);
        generate_terrain(&mut grid, &TerrainParams::default());
        for (coord, cell) in grid.iter() {
            assert!(
                (0.05..=1.0).contains(&cell.permeability),
                "permeability out of bounds at {coord:?}: {}",
                cell.permeability
            );
        }
    }

    #[test]
    fn elevation_offset_shifts_median() {
        // Two grids, same seed, different offset: the median difference
        // must match the offset delta (within epsilon).
        let mut grid_lo = HexGrid::from_radius(5);
        let mut grid_hi = HexGrid::from_radius(5);
        let params_lo = TerrainParams {
            base_elevation_offset: 0.0,
            ..TerrainParams::default()
        };
        let params_hi = TerrainParams {
            base_elevation_offset: 500.0,
            ..TerrainParams::default()
        };
        generate_terrain(&mut grid_lo, &params_lo);
        generate_terrain(&mut grid_hi, &params_hi);

        // For each cell, the diff must be exactly +500
        // (the offset is additive and doesn't depend on the noise).
        for (coord, cell_lo) in grid_lo.iter() {
            let cell_hi = grid_hi.get(*coord).unwrap();
            let diff = cell_hi.elevation - cell_lo.elevation;
            assert!(
                (diff - 500.0).abs() < 1e-2,
                "offset not uniform at {coord:?}: diff={diff}"
            );
        }
    }

    #[test]
    fn apply_dem_override_replaces_elevation_and_temperature() {
        let params = TerrainParams::default();
        let mut grid = HexGrid::from_radius(3);
        generate_terrain(&mut grid, &params);

        let overrides = vec![
            DemCellOverride {
                q: 0,
                r: 0,
                elevation: 1500.0,
            },
            DemCellOverride {
                q: 1,
                r: 0,
                elevation: 200.0,
            },
        ];
        let report = apply_dem_override(&mut grid, &overrides, &params);
        assert_eq!(
            report,
            DemApplyReport {
                applied: 2,
                skipped: 0
            }
        );

        let center = grid.get(HexCoord::new(0, 0)).unwrap();
        assert!((center.elevation - 1500.0).abs() < f32::EPSILON);
        let expected_temp = params.base_temperature - (1500.0 / 1000.0) * params.lapse_rate;
        assert!((center.temperature - expected_temp).abs() < 1e-4);

        let neighbor = grid.get(HexCoord::new(1, 0)).unwrap();
        assert!((neighbor.elevation - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_dem_override_ignores_out_of_domain_coords() {
        let params = TerrainParams::default();
        let mut grid = HexGrid::from_radius(2);
        generate_terrain(&mut grid, &params);
        let before: Vec<f32> = grid.iter().map(|(_, c)| c.elevation).collect();

        // (100, 100) is outside the radius-2 domain: should affect nothing.
        let overrides = vec![DemCellOverride {
            q: 100,
            r: 100,
            elevation: 9999.0,
        }];
        let report = apply_dem_override(&mut grid, &overrides, &params);

        let after: Vec<f32> = grid.iter().map(|(_, c)| c.elevation).collect();
        assert_eq!(before, after, "out-of-domain override must be ignored");
        assert_eq!(
            report,
            DemApplyReport {
                applied: 0,
                skipped: 1
            }
        );
    }

    /// The case that bit us: a radius-R survey loaded into a grid of
    /// radius r < R. The old silence made it look like a complete world.
    #[test]
    fn apply_dem_override_reports_cells_dropped_by_a_smaller_grid() {
        let params = TerrainParams::default();
        let wide = HexGrid::from_radius(4);
        let overrides: Vec<DemCellOverride> = wide
            .iter()
            .map(|(coord, _)| DemCellOverride {
                q: coord.q,
                r: coord.r,
                elevation: 500.0,
            })
            .collect();

        let mut narrow = HexGrid::from_radius(2);
        generate_terrain(&mut narrow, &params);
        let report = apply_dem_override(&mut narrow, &overrides, &params);

        assert_eq!(report.applied, narrow.len(), "the whole domain is covered");
        assert_eq!(report.skipped, overrides.len() - narrow.len());
        assert!(report.skipped > 0, "the survey overflows the grid");
    }

    #[test]
    fn dem_override_file_accepts_both_shapes() {
        let legacy: DemOverrideFile =
            serde_json::from_str(r#"[{"q":0,"r":0,"elevation":42.0}]"#).unwrap();
        assert_eq!(legacy.cells().len(), 1);
        assert!(legacy.meta().is_none(), "legacy format: no provenance");

        let with_meta: DemOverrideFile = serde_json::from_str(
            r#"{"meta":{"center_lat":44.62,"center_lon":5.09,"radius":120,
                 "cell_spacing_m":130.0,"samples_per_cell":1,"dataset":"srtm30m"},
                "cells":[{"q":0,"r":0,"elevation":42.0}]}"#,
        )
        .unwrap();
        assert_eq!(with_meta.cells().len(), 1);
        let meta = with_meta.meta().expect("provenance present");
        assert_eq!(meta.radius, 120);
        assert_eq!(meta.dataset, "srtm30m");
        assert!((meta.cell_spacing_m - 130.0).abs() < f32::EPSILON);
    }
}
