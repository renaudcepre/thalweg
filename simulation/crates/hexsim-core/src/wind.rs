use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

use crate::coord::hex_direction_to_world;
use crate::dynamics::{CELL_SPACING_M, STEEP_SLOPE_GRADE};
use crate::grid::HexGrid;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WindVec {
    pub x: f32,
    pub y: f32,
}

impl WindVec {
    #[must_use]
    pub fn magnitude(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Local direction of this wind vector, in degrees, wrapped into
    /// [0, 360°). Per-vector formula (not an aggregate): `atan2(y, x)` then
    /// normalization. Historically duplicated at 3 call sites (#53); this
    /// helper is the single source.
    #[must_use]
    pub fn direction_deg(self) -> f32 {
        (self.y.atan2(self.x).to_degrees() + 360.0) % 360.0
    }
}

/// Wind field indexed by `HexGrid::cell_index` (size = `grid.len()`).
pub type WindField = Vec<WindVec>;

/// Wind field parameters.
///
/// **Unit convention (Phase 4, #33)**: `magnitude` values in the
/// `WindField` are unitless but interpreted via `magnitude × 10 = m/s` by
/// consumers (Meyer in `step_evaporation`). Provisional convention set in
/// Phase 1; Phase 6 will rescale the values so that `magnitude` is directly
/// in m/s.
#[derive(Clone, Serialize, Deserialize)]
pub struct WindParams {
    pub seed: u32,
    /// Perlin noise amplitude on the background wind direction (radians).
    /// Only source of "physically uncaused" wind after the switch to a
    /// closed terrarium; everything else (thermal, orographic) emerges from
    /// internal gradients.
    pub noise_direction_amplitude: f32,
    /// Perlin noise amplitude on the background wind magnitude (0..1).
    pub noise_strength_amplitude: f32,
    /// Temporal rate of change of the Perlin noise (1/tick). 0.08 ≈ a wind
    /// regime that changes every ~2 weeks.
    pub noise_time_scale: f32,
    pub thermal_strength: f32,
    pub terrain_deflection: f32,
    pub terrain_speed_factor: f32,
    pub smoothing_passes: u8,
    pub humidity_advection_rate: f32,
    /// Directional diffusion rate of temperature by the surface wind.
    /// Separate from `humidity_advection_rate` because temperature has its
    /// own inertia (effective heat capacity) vs. vapor which follows the
    /// flow.
    ///
    /// The April 2026 analysis considered this mechanism neutralized by
    /// thermal relaxation (`relax_rate=0.1` from the pre-#43 regime, ~10x
    /// per tick on anomalies; since the SI energy balance #43, the coupling
    /// is driven by `thermal_coupling=1.0` × surface heat capacity).
    /// `scale_knockout_temperature` shows that in the current 2-layer model
    /// the net effect is visible: mean T >1500m drops from 0.79 C to 0, and
    /// the intra-band spatial variance increases by 10-40%; advection
    /// transfers heat from the warm plain to the cold peaks and homogenizes
    /// the bands. Relaxation dominates the temporal dynamics but not the
    /// spatial gradient.
    pub temperature_advection_rate: f32,
    /// Rotation of the upper-level wind relative to the surface wind
    /// (degrees). Inspired by the Ekman spiral: upper-level winds are
    /// deflected by 20-40° relative to the ground in the northern
    /// hemisphere. This rotation decouples the transport of
    /// `humidity_upper` from surface transport; the origin of rain shadows
    /// behind relief features.
    pub wind_upper_rotation_deg: f32,
    /// Magnitude ratio between upper-level wind and surface wind (>1 =
    /// stronger aloft, ground friction slows the flow near the surface).
    pub wind_upper_speed_ratio: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        // Calibrated in #176 (bench-search N=500, run_20260422_201140),
        // chosen at the time for a dominant West wind (ex-`west_bias`=0.32,
        // removed since the synoptic model provides the dominant west
        // wind) and a mild Mediterranean climate (summer/winter precip
        // ratio 0.68). noise_strength, noise_time_scale,
        // terrain_speed_factor, wind_upper_* not explored by the optimizer
        // → kept as-is.
        Self {
            seed: 42,
            noise_direction_amplitude: 1.69,
            noise_strength_amplitude: 0.25,
            noise_time_scale: 0.08,
            thermal_strength: 0.36,
            terrain_deflection: 0.59,
            terrain_speed_factor: 0.8,
            smoothing_passes: 3,
            humidity_advection_rate: 3.0,
            temperature_advection_rate: 0.07,
            wind_upper_rotation_deg: 10.0,
            wind_upper_speed_ratio: 1.4,
        }
    }
}

/// Derives the upper-level wind field from the surface wind via a rotation
/// plus a speed factor. The rotation decouples the transport of
/// `humidity_upper` from the surface wind (origin of rain shadows).
/// API `Vec<WindVec>` indexed by `cell_index` (same layout as `surface`).
#[must_use]
pub fn compute_upper_wind_field(surface: &WindField, params: &WindParams) -> WindField {
    let angle_rad = params.wind_upper_rotation_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let scale = params.wind_upper_speed_ratio;
    surface
        .iter()
        .map(|w| WindVec {
            x: (w.x * cos_a - w.y * sin_a) * scale,
            y: (w.x * sin_a + w.y * cos_a) * scale,
        })
        .collect()
}

/// Zero-malloc variant of `compute_upper_wind_field`: writes into `out`.
pub(crate) fn compute_upper_wind_field_into(
    surface: &WindField,
    params: &WindParams,
    out: &mut WindField,
) {
    let angle_rad = params.wind_upper_rotation_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let scale = params.wind_upper_speed_ratio;
    out.resize(surface.len(), WindVec::default());
    for (i, w) in surface.iter().enumerate() {
        out[i] = WindVec {
            x: (w.x * cos_a - w.y * sin_a) * scale,
            y: (w.x * sin_a + w.y * cos_a) * scale,
        };
    }
}

/// Computes the wind field for the current tick (Perlin noise base, no
/// synoptic forcing; see `compute_wind_field_into` for the synoptic base
/// option).
///
/// Phases: stochastic Perlin noise, local thermal, relief deflection,
/// upwind propagation. No more external forcing; wind emerges from
/// internal gradients.
#[must_use]
pub fn compute_wind_field(grid: &HexGrid, params: &WindParams, tick: u64) -> WindField {
    let mut field: WindField = vec![WindVec::default(); grid.len()];
    let mut scratch: WindField = vec![WindVec::default(); grid.len()];
    compute_wind_field_into(grid, params, tick, &mut field, &mut scratch, None);
    field
}

/// Zero-malloc variant: writes into `field` (resizing if needed), using
/// `scratch` as the snapshot for `propagate_upstream`.
///
/// `synoptic_base` (Phase 1 of the synoptic-dynamics design: prognostic
/// pressure-driven wind):
/// - `None` → base = Perlin noise.
/// - `Some(base)` → the wind base IS the prognostic synoptic wind; the
///   decorative noise is removed (the background wind and its dominant
///   direction emerge from the pressure gradient). Thermal breeze and
///   orographic deflection remain local modifiers on top.
pub fn compute_wind_field_into(
    grid: &HexGrid,
    params: &WindParams,
    tick: u64,
    field: &mut WindField,
    scratch: &mut WindField,
    synoptic_base: Option<&WindField>,
) {
    let n = grid.len();
    field.resize(n, WindVec::default());
    scratch.resize(n, WindVec::default());

    match synoptic_base {
        Some(base) if base.len() == n => field.copy_from_slice(base),
        _ => field.fill(compute_noise_wind(params, tick)),
    }

    add_thermal_component(grid, params, field);
    apply_terrain_deflection(grid, params, field);
    propagate_upstream(grid, params, field, scratch);
}

/// Wind field magnitudes, in a reusable buffer. Recompute at the same
/// cadence as the field itself (subsample #89): evaporation used to
/// consume one `sqrt` per cell per hour while the field only changes once
/// every N hours (perf project #88).
pub fn compute_wind_magnitudes_into(field: &WindField, out: &mut Vec<f32>) {
    out.clear();
    out.extend(field.iter().map(|w| w.magnitude()));
}

/// Phase 1: stochastic background wind (temporally seeded Perlin noise).
/// The native noise ∈ [-1, 1] is remapped to [0, 1] for the magnitude,
/// which guarantees a small background wind that is always positive;
/// otherwise on a tick where `str_noise < 0` the wind would be strictly
/// zero, which makes the sim too static and breaks the propagation of
/// thermal perturbations.
///
/// v0.3.0 PR2 (#38): `hour_tick` is in hours; we convert to cumulative
/// days before indexing the noise so that its rhythm (`noise_time_scale`)
/// stays identical to v0.2.x. Hourly wind variability is carried by the
/// thermal term, not by the noise.
fn compute_noise_wind(params: &WindParams, hour_tick: u64) -> WindVec {
    let noise = Perlin::new(params.seed);
    let day_tick = crate::time::ticks_to_days(hour_tick);
    let t = f64::from(u32::try_from(day_tick % 100_000).unwrap_or(0))
        * f64::from(params.noise_time_scale);
    #[allow(clippy::cast_possible_truncation)] // f64→f32: noise ∈ [-1,1]
    let dir_noise = noise.get([t, 0.0]) as f32;
    #[allow(clippy::cast_possible_truncation)]
    let str_noise = noise.get([t, 50.0]) as f32;

    let direction = dir_noise * params.noise_direction_amplitude;
    // str_noise ∈ [-1,1] → [0,1] via (x+1)/2, then scaled by amplitude.
    let strength = params.noise_strength_amplitude * (0.5 + 0.5 * str_noise);

    WindVec {
        x: direction.cos() * strength,
        y: direction.sin() * strength,
    }
}

/// Phase 1b: local thermal wind (temperature gradient → breeze).
/// Warm air rises → low pressure → surface air converges toward the warmth.
fn add_thermal_component(grid: &HexGrid, params: &WindParams, field: &mut WindField) {
    if params.thermal_strength < 1e-6 {
        return;
    }
    let cells = grid.cells_slice();
    for (i, cell) in cells.iter().enumerate() {
        // Toric neighborhood: the thermal gradient is computed over the 6
        // neighbors, seam included (temperature is continuous across the
        // torus). The truncated edge neighborhood used to systematically
        // bias the breeze along the outer ring. j == i (degenerate grid) →
        // diff 0.
        let neighbors = grid.neighbor_indices_toric(i);

        let mut tg_x = 0.0_f32;
        let mut tg_y = 0.0_f32;
        for (di, &j) in neighbors.iter().enumerate() {
            let (dx, dy) = hex_direction_to_world(di);
            let temp_diff = cells[j].temperature - cell.temperature;
            tg_x += dx * temp_diff;
            tg_y += dy * temp_diff;
        }
        tg_x /= 6.0;
        tg_y /= 6.0;

        let w = &mut field[i];
        w.x += tg_x * params.thermal_strength;
        w.y += tg_y * params.thermal_strength;
    }
}

/// Phase 2: deflection by relief.
/// Parallel/perpendicular decomposition relative to the elevation gradient.
/// Uphill → blocking + circumvention. Downhill → catabatic acceleration.
fn apply_terrain_deflection(grid: &HexGrid, params: &WindParams, field: &mut WindField) {
    let cells = grid.cells_slice();
    for (i, cell) in cells.iter().enumerate() {
        // Toric neighborhood: relief is periodic, its gradient across the
        // seam is physical (same reason as the thermal component).
        let neighbors = grid.neighbor_indices_toric(i);

        let mut grad_x = 0.0_f32;
        let mut grad_y = 0.0_f32;
        for (di, &j) in neighbors.iter().enumerate() {
            let (dx, dy) = hex_direction_to_world(di);
            let elev_diff = cells[j].elevation - cell.elevation;
            grad_x += dx * elev_diff;
            grad_y += dy * elev_diff;
        }
        grad_x /= 6.0;
        grad_y /= 6.0;

        let wind = field[i];
        let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt();

        if grad_mag > 1e-6 {
            let gn_x = grad_x / grad_mag;
            let gn_y = grad_y / grad_mag;
            // Threshold derived from CELL_SPACING_M (see
            // dynamics::STEEP_SLOPE_GRADE): without this, a change in the
            // engine's resolution would silently skew this threshold (cf.
            // feat/dem-terrain-validation, where this threshold used to be
            // hardcoded).
            let slope = (grad_mag / (STEEP_SLOPE_GRADE * CELL_SPACING_M)).min(1.0);

            let dot = wind.x * gn_x + wind.y * gn_y;
            let par_x = gn_x * dot;
            let par_y = gn_y * dot;
            let perp_x = wind.x - par_x;
            let perp_y = wind.y - par_y;

            field[i] = if dot > 0.0 {
                let block = (1.0 - params.terrain_deflection * slope).max(0.0);
                let block_sq = block * block;
                let deflect = 1.0 + params.terrain_deflection * slope * dot.abs().min(1.0);
                WindVec {
                    x: par_x * block_sq + perp_x * deflect,
                    y: par_y * block_sq + perp_y * deflect,
                }
            } else {
                let boost = 1.0 + params.terrain_speed_factor * slope * 0.5;
                WindVec {
                    x: wind.x * boost,
                    y: wind.y * boost,
                }
            };
        }
    }
}

/// Phase 3: asymmetric propagation (wind shadow).
/// Only upwind neighbors have influence; downwind does not dilute
/// perturbations.
fn propagate_upstream(
    grid: &HexGrid,
    params: &WindParams,
    field: &mut WindField,
    snapshot: &mut WindField,
) {
    for _ in 0..params.smoothing_passes {
        snapshot.clone_from(field);
        for i in 0..grid.len() {
            let wind = snapshot[i];

            let mut sum_x = wind.x * 4.0;
            let mut sum_y = wind.y * 4.0;
            let mut weight = 4.0_f32;

            let neighbors = grid.neighbor_indices_toric(i);
            for (dir_idx, &n_idx) in neighbors.iter().enumerate() {
                let nw = snapshot[n_idx];
                let (dx, dy) = hex_direction_to_world(dir_idx);
                let upstream = -(wind.x * dx + wind.y * dy);
                if upstream > 0.0 {
                    sum_x += nw.x * upstream;
                    sum_y += nw.y * upstream;
                    weight += upstream;
                }
            }

            field[i] = WindVec {
                x: sum_x / weight,
                y: sum_y / weight,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;

    #[test]
    fn wind_field_is_computed() {
        // Flat terrain, uniform grid: the field must compute without NaN
        // and have the correct size. The magnitude can be zero depending
        // on the Perlin sample for the tick (noise ∈ [-1,1], clamped to 0
        // if negative).
        let grid = HexGrid::from_radius(5);
        let params = WindParams::default();
        let field = compute_wind_field(&grid, &params, 0);

        assert_eq!(field.len(), grid.len());
        for w in &field {
            assert!(w.x.is_finite() && w.y.is_finite());
        }
    }

    #[test]
    fn noise_drives_temporal_variation() {
        // Without seasonal forcing, the only driver of variation over time
        // is the Perlin noise. Over a sample of varied ticks, at least two
        // fields must differ (guard against a Perlin that would always
        // return 0, or a broken time_scale).
        let grid = HexGrid::from_radius(3);
        let params = WindParams::default();
        let i = grid.cell_index(HexCoord::new(0, 0)).unwrap();
        let samples: Vec<WindVec> = [0_u64, 37, 137, 413, 907]
            .iter()
            .map(|&t| compute_wind_field(&grid, &params, t)[i])
            .collect();

        let first = samples[0];
        let any_different = samples
            .iter()
            .skip(1)
            .any(|w| (w.x - first.x).abs() > 1e-3 || (w.y - first.y).abs() > 1e-3);
        assert!(
            any_different,
            "Perlin noise should vary over time: {samples:?}"
        );
    }

    #[test]
    fn mountain_perturbs_wind() {
        let mut grid = HexGrid::from_radius(3);
        if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
            cell.elevation = 1000.0;
        }

        let params = WindParams::default();
        let field = compute_wind_field(&grid, &params, 0);

        let flat_grid = HexGrid::from_radius(3);
        let flat_field = compute_wind_field(&flat_grid, &params, 0);

        let i = grid.cell_index(HexCoord::new(0, 0)).unwrap();
        let mountain_wind = &field[i];
        let j = flat_grid.cell_index(HexCoord::new(0, 0)).unwrap();
        let flat_wind = &flat_field[j];

        assert!(
            (mountain_wind.x - flat_wind.x).abs() > 1e-6
                || (mountain_wind.y - flat_wind.y).abs() > 1e-6,
            "The mountain should perturb the wind"
        );
    }

    /// Builds a thermal gradient: warm cells to the east, cold cells to the
    /// west. The thermal breeze blows from cold to warm (convergence
    /// toward the warm zone) → dominant wind toward the east.
    fn grid_with_east_thermal_gradient(radius: i32) -> HexGrid {
        let mut grid = HexGrid::from_radius(radius);
        let coords: Vec<HexCoord> = grid.coords().copied().collect();
        for coord in coords {
            if let Some(cell) = grid.get_mut(coord) {
                // T linear in q: warm to the east (positive q), cold to the west.
                cell.temperature = f32::from(i8::try_from(coord.q).unwrap_or(0)) * 10.0;
            }
        }
        grid
    }

    #[test]
    fn wind_deflects_around_mountain() {
        // North-south ridge at the center + west->east thermal gradient: the
        // thermal wind toward the east meets the ridge and should be
        // deflected laterally.
        let mut grid = grid_with_east_thermal_gradient(5);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord)
                && coord.q == 0
            {
                cell.elevation = 800.0;
            }
        }

        let params = WindParams {
            thermal_strength: 0.3,
            smoothing_passes: 2,
            ..WindParams::default()
        };
        let field = compute_wind_field(&grid, &params, 0);

        let upwind = field[grid.cell_index(HexCoord::new(-1, 0)).unwrap()];
        assert!(
            upwind.y.abs() > 0.01,
            "The wind should be deflected laterally near the ridge, wind_y = {}",
            upwind.y
        );
    }

    #[test]
    fn upslope_slows_wind() {
        // Ablation `terrain_deflection` on/off, same upslope cell.
        //
        // The old version compared the magnitude at `(-1,0)` (upslope) to
        // the one at `(-4,0)` (far from the mountain), relying on the
        // planetary background wind `west_bias` (removed #108) to
        // guarantee that the observed difference really came from the
        // relief. Without that background wind, the local regime (thermal
        // + deflection) dominates and the order between the two cells
        // flips; this is not a model bug, it was a poorly isolated test.
        //
        // Here we isolate the `apply_terrain_deflection` piece directly by
        // ablation: same grid, same seed, same tick, only
        // `terrain_deflection` changes (0.59 default vs 0.0). If the
        // orographic blocking piece works, the magnitude at `(-1,0)` must
        // be lower with deflection active than without.
        let mut grid = grid_with_east_thermal_gradient(5);
        if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
            cell.elevation = 1000.0;
        }
        for n in HexCoord::new(0, 0).neighbors() {
            if let Some(cell) = grid.get_mut(n) {
                cell.elevation = 600.0;
            }
        }

        let params_with_deflection = WindParams {
            thermal_strength: 0.3,
            smoothing_passes: 2,
            ..WindParams::default()
        };
        let params_without_deflection = WindParams {
            terrain_deflection: 0.0,
            ..params_with_deflection.clone()
        };

        let field_with = compute_wind_field(&grid, &params_with_deflection, 0);
        let field_without = compute_wind_field(&grid, &params_without_deflection, 0);

        let idx = grid.cell_index(HexCoord::new(-1, 0)).unwrap();
        let upslope_with = field_with[idx].magnitude();
        let upslope_without = field_without[idx].magnitude();

        assert!(
            upslope_with < upslope_without,
            "Upslope wind with deflection ({upslope_with:.3}) should be weaker than without deflection ({upslope_without:.3})"
        );
    }

    #[test]
    fn wind_field_covers_all_cells() {
        let grid = HexGrid::from_radius(10);
        let field = compute_wind_field(&grid, &WindParams::default(), 50);
        assert_eq!(
            field.len(),
            grid.len(),
            "The wind field must cover all cells"
        );
    }

    #[test]
    fn thermal_component_blows_towards_warm_zone() {
        // Linear temperature gradient: warm central cell, cold cell to the
        // east. With thermal_strength > 0 and no background wind, the
        // thermal breeze must have a component toward the warmth (so from
        // east to west: wind.x < 0 on the east cell).
        let mut grid = HexGrid::from_radius(2);
        // Warm cell at the center
        if let Some(c) = grid.get_mut(HexCoord::new(0, 0)) {
            c.temperature = 30.0;
        }
        // Cold cell to the east
        if let Some(c) = grid.get_mut(HexCoord::new(1, 0)) {
            c.temperature = 0.0;
        }

        let params = WindParams {
            noise_direction_amplitude: 0.0,
            noise_strength_amplitude: 0.0,
            thermal_strength: 0.5, // strong, to isolate the signal
            terrain_deflection: 0.0,
            smoothing_passes: 0,
            ..WindParams::default()
        };
        let field = compute_wind_field(&grid, &params, 0);

        let east_wind = field[grid.cell_index(HexCoord::new(1, 0)).unwrap()];
        // The east cell sees its west neighbor (the center) as warmer.
        // add_thermal_component computes tg = sum(dir_world × temp_diff).
        // The neighbor toward the west is at direction (-1, 0) = dx=-1.
        // temp_diff = 30 - 0 = 30 for that neighbor.
        // Contribution: dx * temp_diff = -1 * 30 = -30 (before averaging).
        // So wind.x should be negative (blows toward the west = warm).
        assert!(
            east_wind.x < -0.01,
            "thermal breeze toward the warmth: east.wind.x = {} (expected < 0)",
            east_wind.x
        );
    }
}
