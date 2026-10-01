use crate::cell::CellProperties;
use crate::grid::HexGrid;
use crate::par::{for_each_chunk_mut, reduce_blocks};
use crate::physics::tetens_saturation_vapor_pressure;
use crate::temperature::{SECONDS_PER_HOUR, TemperatureParams};

use super::{AtmosphereParams, R_VAP, SURFACE_LAYER_M};

/// Precipitable water (mm) at saturation for the `upper` layer at
/// temperature `t_upper`. Phase 6 (#29): replaces the phenomenological
/// curve with the physical Clausius-Clapeyron law (Tetens), converted to
/// mm of precipitable water integrated over the layer height.
///
/// Conversion chain:
/// 1. `e_s(T)` = saturation vapor pressure via Tetens (hPa).
/// 2. `rho_vap(T) = e_s / (R_vap × T_K)`, saturation vapor density
///    (kg/m³), ideal gas law applied to vapor alone.
/// 3. `PW = rho_vap × H`, simplified vertical integration (uniform
///    density over the layer), result in mm of precipitable water
///    (1 kg/m² ≡ 1 mm).
///
/// Reference values with H = 1500 m:
/// - T = 0°C: `e_s` ≈ 6.11 hPa → `PW_sat` ≈ 7.3 mm
/// - T = 15°C: `e_s` ≈ 17.0 hPa → `PW_sat` ≈ 19.3 mm
/// - T = 20°C: `e_s` ≈ 23.4 hPa → `PW_sat` ≈ 25.9 mm
#[must_use]
pub fn saturation_upper(t_upper: f32, params: &AtmosphereParams) -> f32 {
    saturation_upper_pw(t_upper, params.upper_layer_altitude_m)
}

/// Variant of `saturation_upper` taking the layer height directly. Useful
/// for callers (metrics, diagnostics) that don't need the full
/// `AtmosphereParams`.
#[must_use]
pub fn saturation_upper_pw(t_upper: f32, altitude_m: f32) -> f32 {
    let e_s_pa = tetens_saturation_vapor_pressure(t_upper).0 * 100.0;
    let t_kelvin = (t_upper + 273.15).max(1.0);
    let rho_vap = e_s_pa / (R_VAP * t_kelvin);
    rho_vap * altitude_m
}

/// Saturation precipitable water (mm) for the boundary layer at
/// `t_surface`. Issue #45: analogous to `saturation_upper` but integrated
/// over the 50 m near the ground, the layer where radiative fog forms.
///
/// Reference values:
/// - T = -5 °C: ≈ 0.16 mm
/// - T = 0 °C : ≈ 0.24 mm
/// - T = 15 °C: ≈ 0.64 mm
/// - T = 25 °C: ≈ 1.15 mm
#[must_use]
pub fn saturation_surface(t_surface: f32) -> f32 {
    saturation_upper_pw(t_surface, SURFACE_LAYER_M)
}

/// Horizontal means of the surface state that anchor the upper-air
/// temperature: `(mean surface temperature °C, mean elevation m)`.
/// Empty grid: `(0, 0)`.
///
/// A deterministic block reduction (`par::reduce_blocks`, r250 perf
/// effort): per-block f32 sums computed in parallel over fixed blocks of
/// `REDUCE_BLOCK_CELLS` cells, folded in block order, divided by the
/// exact cell count. Same bits whatever the thread count; not the bits
/// of the serial running mean (Welford) it replaces, which streamed the
/// whole grid on one core twice per tick (a third time before
/// `AtmoForcing::mean_elevation`) at r250, ~1 ms each at 4 threads on the
/// 4-vCPU VM. The block sums are also the tighter estimate: 4 096 terms
/// per f32 accumulator and 46 partials, against 188 251 sequential
/// `mean += (x − mean) / count` updates.
#[must_use]
pub fn surface_means(grid: &HexGrid) -> (f32, f32) {
    let cells = grid.cells_slice();
    if cells.is_empty() {
        return (0.0, 0.0);
    }
    let mut partials: Vec<(f32, f32)> = Vec::new();
    reduce_blocks(cells.len(), &mut partials, |range| {
        let mut sum_t = 0.0_f32;
        let mut sum_z = 0.0_f32;
        for c in &cells[range] {
            sum_t += c.temperature;
            sum_z += c.elevation;
        }
        (sum_t, sum_z)
    });
    let (sum_t, sum_z) = partials
        .iter()
        .fold((0.0_f32, 0.0_f32), |(t, z), &(pt, pz)| (t + pt, z + pz));
    let count = exact_cell_count(cells.len());
    (sum_t / count, sum_z / count)
}

/// `n` as an f32, exact below 2^24 cells (16.7 M, two orders of magnitude
/// above r250): assembled from the two 16-bit halves through lossless
/// `From` conversions, so no `as` cast and no precision-loss lint to
/// silence.
fn exact_cell_count(n: usize) -> f32 {
    let n = u32::try_from(n).expect("cell count fits in u32");
    debug_assert!(n < 1 << 24, "cell count beyond exact f32 integers");
    let high = u16::try_from(n >> 16).expect("high half of a u32 fits in u16");
    let low = u16::try_from(n & 0xFFFF).expect("low half of a u32 fits in u16");
    f32::from(high) * 65_536.0 + f32::from(low)
}

/// Time constant (s) of the diurnal smoothing applied to the map-mean
/// surface temperature that anchors the upper layer
/// (`Simulation::upper_air_mean_t`, stepped by [`smooth_upper_air_mean_t`]).
///
/// The diurnal cycle lives in the surface boundary layer: the daily
/// amplitude of the air temperature is ~8-10 K at screen level and
/// decays to ~1 K in the free troposphere above the boundary-layer top
/// (Stull 1988, *An Introduction to Boundary Layer Meteorology*, §1.6
/// "diurnal variation" and ch. 11; radiosonde climatologies put the
/// 850 hPa diurnal range at ~1 K). The layer this engine models sits
/// 1500 m above the mean ground (`upper_layer_altitude_m`), above the
/// daytime mixed layer most of the year. Anchored to the *instantaneous*
/// mean (2026-09-02), the whole layer followed the ~8 K nightly cooling
/// of the surface and condensed on the highest cells every night: the
/// crest of `phys_ubac_not_a_rain_attractor` was wet 365 days a year
/// and the procedural world had 0 rain-free days (#63).
///
/// τ = 24 h is the shortest first-order smoothing that removes the
/// diurnal harmonic: a 24 h sinusoid is attenuated to
/// `1/√(1+(2π)²) ≈ 0.157` of its amplitude (8 K → 1.3 K, the observed
/// order of magnitude), while the seasonal signal (period 365 d) passes
/// at 0.9999 of its amplitude with a one-day lag. A shorter τ lets the
/// night through, a longer one only buys lag on the seasons.
pub const UPPER_AIR_SMOOTHING_TAU_S: f32 = 24.0 * SECONDS_PER_HOUR;

/// One hourly step of the first-order (exponential) smoothing of the
/// map-mean surface temperature: `m += (T̄ − m)·(1 − exp(−Δt/τ))`, with
/// Δt = one tick and τ = [`UPPER_AIR_SMOOTHING_TAU_S`]. Exact
/// discretisation of `dm/dt = (T̄ − m)/τ` for a `T̄` held over the tick.
/// Called once per tick by `Simulation::step_hour` before the
/// atmosphere step; the result travels to the atmosphere as
/// `AtmoForcing::upper_air_mean_t`.
#[must_use]
pub fn smooth_upper_air_mean_t(previous: f32, instantaneous: f32) -> f32 {
    let gain = 1.0 - (-SECONDS_PER_HOUR / UPPER_AIR_SMOOTHING_TAU_S).exp();
    previous + (instantaneous - previous) * gain
}

/// Temperature (°C) of the upper layer above a cell at elevation
/// `elevation_m`: the (diurnally smoothed) map-mean surface temperature,
/// corrected by the standard lapse rate for the height of the layer
/// above the map-mean ground, `T̄ − Γ·(z − z̄ + H)/1000`.
///
/// The free atmosphere is horizontally mixed at the scale of the
/// terrarium (a few km): the air 1500 m above a north-facing slope is
/// the same air as above the south-facing slope next to it. The layer
/// therefore only follows the *elevation* of the ground (orographic
/// cooling, the mechanism that makes summits precipitate), never the
/// local surface anomaly (aspect, occlusion, lake cooling, snow albedo).
/// Before 2026-09-02 it was `T_surface − Γ·H`: any cell persistently
/// colder than its neighbours at the surface became a permanent
/// condenser aloft, raining 365 days a year while the rest of the map
/// stayed dry (JOURNAL 2026-09-02, bisected to the aspect insolation
/// e3594f9).
///
/// `mean_surface_t` is the smoothed mean kept by the simulation
/// (`Simulation::upper_air_mean_t`, see [`UPPER_AIR_SMOOTHING_TAU_S`]),
/// not the instantaneous one: the free atmosphere keeps the seasons and
/// the lapse with elevation, not the day/night swing of the surface.
#[must_use]
pub fn upper_air_temperature(
    mean_surface_t: f32,
    mean_elevation_m: f32,
    elevation_m: f32,
    params: &AtmosphereParams,
    temp_params: &TemperatureParams,
) -> f32 {
    let height_above_mean_ground = elevation_m - mean_elevation_m + params.upper_layer_altitude_m;
    mean_surface_t - temp_params.lapse_rate * height_above_mean_ground / 1000.0
}

/// Cloud dynamics for one cell: vapor ↔ droplets.
///
/// - Condensation (#63 Phase 4 Step 3): anchored to Clausius-Clapeyron via
///   Tetens. When `humidity_upper > saturation_upper(T)`, the
///   thermodynamic surplus drains into droplets at `condensation_rate`
///   (pre-clamped to `[0, 1]` by the caller, a per-tick constant —
///   hoisted out so a full-grid sweep doesn't redo the `.min(1.0)` per
///   cell). Natural asymptote at RH=1 (saturation), not an arbitrary
///   dimensionless RH threshold.
/// - Cloud evaporation (saturation adjustment, #63 L2b): below
///   saturation the droplets go straight back to vapour, bounded by the
///   layer's own deficit — `min(cloud_water, sat − humidity_upper)`. No
///   rate, no RH threshold, no dead zone. See
///   [`saturation_adjustment_transfer`] for the physics.
///
/// `t_up` = this cell's upper-air temperature (`AtmoScratch::t_upper[i]`,
/// filled by `fill_upper_air` from [`upper_air_temperature`]).
///
/// Extracted to a per-cell function, no full-grid sweep of its own (r250
/// perf effort, chunk B2): the only caller left is
/// `atmosphere::apply_temperature_advection_then_cloud_and_condensation`,
/// which fuses it with temperature advection's apply and
/// `surface_condensation_for_cell` into one sweep — see that function's
/// doc for the dependency argument. Also exercised directly by this
/// module's own unit tests, on a single cell, no grid needed.
///
/// Returns what [`cloud_dynamics`] returns: the vapour → droplet transfer
/// of this cell for this hour (mm, negative when the layer is
/// subsaturated and droplets go back to vapour). The fused sweep uses its
/// sign to count the cells that condensed, which is the saturated cloud
/// fraction the coarse torus needs (`atmosphere::coarse`).
pub(crate) fn cloud_dynamics_for_cell(
    nc: &mut CellProperties,
    t_up: f32,
    params: &AtmosphereParams,
    condensation_rate: f32,
) -> f32 {
    cloud_dynamics(
        &mut nc.humidity_upper,
        &mut nc.cloud_water,
        saturation_upper(t_up, params),
        condensation_rate,
    )
}

/// The vapour ↔ droplet transition itself, on the two stocks of one
/// column and the saturation that column sees — no `CellProperties`, no
/// grid, no `AtmosphereParams`.
///
/// Split out of [`cloud_dynamics_for_cell`] (coarse upper layer, step 2)
/// so the fine grid and the ~1 km coarse torus run **the same code** on
/// their own stocks rather than two hand-kept twins (one system, not
/// case by case). The seam is `sat` because that is
/// exactly where the two paths differ: a fine cell derives it from its own
/// `T_upper` (`saturation_upper(t_up, params)`, just above), a coarse cell
/// takes the **mean of its fine cells' `sat_upper`** — not `sat_upper` of
/// the mean, Tetens being convex (design note §9 risk 1: the mean is the
/// one that is right for a mass budget).
///
/// Returns the **signed vapour → droplet transfer** (mm): `> 0` when the
/// column was supersaturated and condensed, `< 0` when it was
/// subsaturated and gave droplets back to vapour, `0` when nothing moved.
/// A return value rather than a second out-parameter because the caller
/// that needs it (the fused sweep, for the coarse torus's cloud fraction)
/// only needs its sign, and because a function that moves mass between
/// two stocks should be able to say how much.
pub(crate) fn cloud_dynamics(
    humidity_upper: &mut f32,
    cloud_water: &mut f32,
    sat: f32,
    condensation_rate: f32,
) -> f32 {
    let surplus_mm = *humidity_upper - sat;
    if surplus_mm > 0.0 {
        // CC drain: (humidity_upper - sat) × rate. At steady state
        // with input X mm/tick, hu_eq = sat + X/rate → RH = 1 +
        // X/(rate·sat). With rate=1.0/h and input bounded by the LCL
        // bound on the orographic pump (cf
        // step_orographic_convection), RH plateaus at ~1 + ε.
        let transfer = surplus_mm * condensation_rate;
        *humidity_upper -= transfer;
        *cloud_water += transfer;
        transfer
    } else if *cloud_water > 0.0 {
        // Saturation adjustment, the exact mirror of the branch above:
        // the deficit is `-surplus_mm`, and it is the only bound.
        let transfer = saturation_adjustment_transfer(-surplus_mm, *cloud_water);
        *cloud_water -= transfer;
        *humidity_upper += transfer;
        -transfer
    } else {
        0.0
    }
}

/// Droplets returned to vapour in one hour by a subsaturated layer:
/// `min(cloud_water, deficit_mm)`, with `deficit_mm = sat −
/// humidity_upper ≥ 0`. Saturation adjustment (Sundqvist 1978, *Mon.
/// Wea. Rev.* 106, §2; Tiedtke 1993, *Mon. Wea. Rev.* 121, the
/// large-scale evaporation term of the prognostic cloud scheme).
///
/// The physics is a timescale argument, not a coefficient. A droplet in
/// subsaturated air evaporates with the phase relaxation time
/// `τ = 1/(4π·D_v·N·r̄)` (Rogers & Yau 1989, *A Short Course in Cloud
/// Physics*, 3rd ed., ch. 7): with `D_v ≈ 2.5e-5 m²/s`, `N = 1e8 m⁻³`
/// (100 cm⁻³) and `r̄ = 10 µm`, τ ≈ 3 s. Against a 3600 s tick,
/// `1 − exp(−Δt/τ)` is 1.0 to every digit an f32 carries: over one hour
/// in a well-mixed 1500 m column, a cloud in subsaturated air is gone —
/// unless it saturates the layer first, which is what the `min` says.
/// The two outcomes are exhaustive and both are physical: the cloud is
/// wholly evaporated, or the layer is back at RH 1 with droplets left.
///
/// Same asymmetry as the condensation branch above: the latent heat of
/// the phase change is not fed back into `T_upper`, so `sat` is held
/// fixed over the adjustment. A real saturation adjustment iterates on
/// `(T, q)` together; here the upper-air temperature is diagnosed from
/// the map mean (`upper_air_temperature`), not prognostic, so there is
/// no reservoir to release the heat into. The bound is therefore an
/// upper bound on the true one (evaporative cooling would lower `sat`,
/// hence the deficit): it errs toward evaporating slightly *more* than
/// a fully coupled scheme, never less.
///
/// Replaces `cloud_evap_rate` (0.10/day) and `cloud_evap_hr_threshold`
/// (0.4), two dimensionless coefficients from the April 2026 random
/// search. That pair gave a cloud a ~7-day half-life against dry
/// episodes of ~4.5 days, so a drizzle always survived the episode and
/// `fully_rain_free_days_total` was 0 by construction (JOURNAL
/// 2026-09-06).
#[must_use]
fn saturation_adjustment_transfer(deficit_mm: f32, cloud_water: f32) -> f32 {
    cloud_water.min(deficit_mm)
}

/// Isotropic diffusion of `cloud_water` to neighbors: each cell exports a
/// `cloud_diffusion_rate` fraction of its droplets, distributed equally
/// among its topological neighbors. The flux is conservative by
/// construction. Rationale: without this smoothing, two neighboring cells
/// at `cloud_water = 0.119` and `0.121` have radically different
/// behaviors (precipitates vs. nothing), hence the checkerboard pattern.
/// With diffusion, mass is shared locally and clouds become continuous
/// regions.
///
/// Two-phase scatter -> gather (r250 perf effort), simpler than the
/// wind-driven advections: the share is the SAME toward all 6 neighbors
/// (`outgoing / 6`), so no per-direction storage is needed — a single
/// flat `share` buffer suffices, and the gather doesn't need
/// `coord::opposite_direction` either: "k is my neighbor" is symmetric
/// on the toric lattice, so `Σ_{k ∈ neighbors(j)} share[k]` already sums
/// exactly what `j`'s neighbors sent it, self-loss included in the same
/// pass (`share[j]` itself, since `j` is also its own neighbors'
/// neighbor).
pub(crate) fn step_cloud_diffusion(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    share: &mut Vec<f32>,
    deltas: &mut Vec<f32>,
) {
    let rate = params.cloud_diffusion_rate;
    if rate <= 0.0 {
        return;
    }
    let next_cells = next.cells_slice_mut();
    let n = next_cells.len();
    share.resize(n, 0.0);
    for (i, c) in next_cells.iter().enumerate() {
        // Distribution over 6 toric neighbors (self-fallback via wrap
        // impossible → conservative: the share returns to itself,
        // equivalent to zero loss).
        share[i] = c.cloud_water.max(0.0) * rate / 6.0;
    }
    deltas.resize(n, 0.0);
    {
        let share_ref: &Vec<f32> = share;
        for_each_chunk_mut(deltas, |start, chunk| {
            for (local, d) in chunk.iter_mut().enumerate() {
                let j = start + local;
                let outgoing = share_ref[j] * 6.0;
                let neighbors = current.neighbor_indices_toric(j);
                let mut gathered = 0.0_f32;
                for &k in &neighbors {
                    gathered += share_ref[k];
                }
                *d = gathered - outgoing;
            }
        });
    }
    // Apply pass: reads only `deltas[i]`, writes only `next_cells[i]` — a
    // pure per-cell map, parallelizable (`par::for_each_chunk_mut`).
    for_each_chunk_mut(next_cells, |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            let d = deltas[start + local];
            if d != 0.0 {
                cell.cloud_water = (cell.cloud_water + d).max(0.0);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::scaling::scale_atmosphere_for_hourly_tick;
    use crate::atmosphere::test_support::default_temp_params;
    use crate::coord::HexCoord;
    use crate::grid::HexGrid;
    use proptest::prelude::*;

    /// **The sub-grid condensation trigger (coarse upper layer, step 2b).**
    /// Two fine columns of one coarse cell, one saturated and one dry, the
    /// same `humidity_upper` (the broadcast view makes it uniform by
    /// construction): condensing per column and pooling the result is NOT
    /// the same as condensing on the column means, and Jensen says which
    /// way — `x ↦ (hu − x)⁺` is convex, so the pooled sum is the larger.
    ///
    /// This is the term step 2 lost by running the transition on the
    /// coarse torus, and step 2b gave back by running it on the views.
    /// Pinned numbers: `sat = 4` and `14` mm, `hu = 12` mm, rate 1/h.
    /// Pooled `8 + 0 = 8`; coarse-mean trigger `2 × (12 − 9)⁺ = 6`; ratio
    /// 4/3. It is a lower bound on the real gap, not a typical one — push
    /// `hu` to 9 and the coarse trigger is exactly 0 while the fine sum is
    /// still 5.
    #[test]
    fn pooling_per_column_condensation_beats_the_column_mean_trigger() {
        let rate = 1.0_f32;
        let sat = [4.0_f32, 14.0];
        let hu0 = 12.0_f32;

        let mut pooled = 0.0_f32;
        for &s in &sat {
            let (mut hu, mut cw) = (hu0, 0.0_f32);
            pooled += cloud_dynamics(&mut hu, &mut cw, s, rate);
            assert!((cw - pooled_step(hu0, s, rate)).abs() < 1e-6);
        }
        assert!((pooled - 8.0).abs() < 1e-5, "pooled condensation {pooled}");

        // The coarse cell as step 2 saw it: one column carrying the means.
        let mean_sat = (sat[0] + sat[1]) / 2.0;
        let (mut hu, mut cw) = (hu0, 0.0_f32);
        cloud_dynamics(&mut hu, &mut cw, mean_sat, rate);
        let on_the_means = cw * 2.0; // per fine cell -> the cell's mass
        assert!(
            (on_the_means - 6.0).abs() < 1e-5,
            "column-mean condensation {on_the_means}"
        );

        assert!(
            pooled > on_the_means,
            "Jensen: {pooled} must exceed {on_the_means}"
        );
        let ratio = pooled / on_the_means;
        assert!(
            (ratio - 4.0 / 3.0).abs() < 1e-4,
            "the measured ratio moved: {ratio}"
        );

        // And the saturation adjustment pooled the other way round can
        // never take the coarse stock below zero: what each column gives
        // back is at most its own view, so the sum is at most `N_c × cw_c`.
        let cw_c = 0.7_f32;
        let deficits = [0.1_f32, 5.0, 100.0];
        let mut returned = 0.0_f32;
        for &d in &deficits {
            let (mut hu, mut cw) = (10.0 - d, cw_c);
            returned += -cloud_dynamics(&mut hu, &mut cw, 10.0, rate);
            assert!(cw >= 0.0, "a column cannot go negative: {cw}");
        }
        let n_c = deficits.len();
        assert!(
            returned <= f32::from(u16::try_from(n_c).expect("small")) * cw_c + 1e-6,
            "pooled adjustment {returned} exceeds the coarse stock"
        );
    }

    /// What one column condenses in one step, written out: the reference
    /// [`pooling_per_column_condensation_beats_the_column_mean_trigger`]
    /// checks `cloud_dynamics` against.
    fn pooled_step(hu: f32, sat: f32, rate: f32) -> f32 {
        (hu - sat).max(0.0) * rate
    }

    /// Base law "warm air holds more water" (Clausius-Clapeyron via
    /// Tetens): `saturation_surface` must grow STRICTLY with temperature,
    /// and stick to the documented reference values. This is the physical
    /// foundation of "evaporation accelerates with heat"; it's pinned here
    /// so a refactor of the formula can't break it silently.

    #[test]
    fn saturation_surface_rises_strictly_with_temperature() {
        // Strictly monotonic from −20 to +40 °C.
        let mut prev = saturation_surface(-20.0);
        let mut t = -19.0;
        while t <= 40.0 {
            let s = saturation_surface(t);
            assert!(
                s > prev,
                "saturation must increase with T: {s} at {t} °C ≤ {prev} at the previous step"
            );
            prev = s;
            t += 1.0;
        }
        // Reference values from the doc-comment (±10%: these are anchors
        // rounded to 2 digits, not targets to the last decimal). They fix
        // the order of magnitude; monotonicity and concavity are the
        // strict guards.
        for (temp, expected) in [(-5.0, 0.16), (0.0, 0.24), (15.0, 0.64), (25.0, 1.15)] {
            let got = saturation_surface(temp);
            assert!(
                (got - expected).abs() / expected < 0.10,
                "saturation_surface({temp}) = {got:.3} mm, expected ≈ {expected} mm"
            );
        }
        // Clausius-Clapeyron concavity: +10 °C more than doubles between
        // 15 and 25 °C (quasi-exponential growth, not linear).
        assert!(
            saturation_surface(25.0) > 1.7 * saturation_surface(15.0),
            "the rise must accelerate with T (Clausius-Clapeyron)"
        );
    }

    #[test]
    fn phys_condensation_drain_anchored_to_clausius_clapeyron() {
        // Issue #63 Phase 4 Step 3, CC drain anchored.
        //
        // Setup: humidity_upper in extreme supersaturation + cold T. With
        // drain anchored to absolute mm and rate=1.0/h (Pruppacher & Klett
        // 1997 §13.3.1, τ_phase << 1h → complete drain per tick), the
        // thermodynamic surplus is cleared in one tick.
        //
        // Invariants checked:
        // 1. transfer ≤ surplus_mm × rate (linearity of CC drain, contrast
        //    with the old drain × RH_fraction × hu not bounded by CC)
        // 2. conservation: delta_cloud_water = transfer (no creation)
        // 3. humidity_upper ≥ sat (saturable drain, no artificial
        //    undersaturation)
        // 4. at rate=1.0/h, final RH ≈ 1.0 (complete drain in 1 tick)
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.humidity_upper = 50.0;
            cell.temperature = 0.0;
            cell.cloud_water = 0.0;
        }
        let params = AtmosphereParams::default();
        let params_hourly = scale_atmosphere_for_hourly_tick(&params);
        let temp_params = default_temp_params();

        let initial = grid.get(c0).unwrap().humidity_upper;
        // Single cell: the map means are the cell itself, so the upper
        // air is `T − Γ·H` as in the historical formula.
        let (mean_t, mean_z) = surface_means(&grid);
        let t_upper = upper_air_temperature(mean_t, mean_z, 0.0, &params, &temp_params);
        let sat = saturation_upper(t_upper, &params);
        let surplus_mm = initial - sat;
        assert!(
            surplus_mm > 0.0,
            "invalid setup: initial={initial} must be >> sat={sat}"
        );

        let rate = params_hourly.condensation_rate.min(1.0);
        cloud_dynamics_for_cell(grid.get_mut(c0).unwrap(), t_upper, &params_hourly, rate);

        let after = grid.get(c0).unwrap();
        let transfer = initial - after.humidity_upper;
        let rate_eff = params_hourly.condensation_rate.min(1.0);
        let max_transfer_cc = surplus_mm * rate_eff;
        let tol = 1e-3;

        assert!(
            (after.cloud_water - transfer).abs() < 1e-5,
            "non-conservative transfer: delta_cloud={} != transfer={transfer}",
            after.cloud_water
        );
        assert!(
            transfer <= max_transfer_cc + tol,
            "drain not anchored to CC: transfer={transfer} > surplus_mm × rate={max_transfer_cc}"
        );
        assert!(
            after.humidity_upper >= sat - tol,
            "drain violated CC: humidity_upper={} < sat={sat}",
            after.humidity_upper
        );
        let hr_final = after.humidity_upper / sat;
        assert!(
            (hr_final - 1.0).abs() < 0.01,
            "rate=1.0/h must bring RH to saturation in 1 tick: final RH={hr_final}"
        );
    }

    #[test]
    fn surface_means_are_the_plain_averages() {
        let mut grid = HexGrid::from_radius(2);
        let n = grid.len();
        let coords: Vec<HexCoord> = grid.coords().copied().collect();
        for (k, c) in coords.iter().enumerate() {
            let cell = grid.get_mut(*c).unwrap();
            cell.temperature = f32::from(u8::try_from(k).unwrap()) - 5.0;
            cell.elevation = 100.0 * f32::from(u8::try_from(k).unwrap());
        }
        let (mean_t, mean_z) = surface_means(&grid);
        let k_mean = f32::from(u8::try_from(n - 1).unwrap()) / 2.0;
        assert!((mean_t - (k_mean - 5.0)).abs() < 1e-4, "mean_t={mean_t}");
        assert!((mean_z - 100.0 * k_mean).abs() < 1e-2, "mean_z={mean_z}");
        assert_eq!(surface_means(&HexGrid::new()), (0.0, 0.0));
    }

    #[test]
    fn exact_cell_count_is_lossless_across_the_u16_boundary() {
        for n in [0_usize, 1, 65_535, 65_536, 188_251, (1 << 24) - 1] {
            let got = exact_cell_count(n);
            let expected = f64::from(u32::try_from(n).unwrap());
            assert!((f64::from(got) - expected).abs() < 0.5, "n={n} got {got}");
        }
    }

    /// The block reduction gives the same bits on one worker, on four, and
    /// outside any pool: a grid above `par::PAR_MIN_CELLS` so the pooled
    /// calls really split, temperatures noisy enough that a different
    /// association of the terms would show in f32.
    #[test]
    #[cfg(feature = "parallel")]
    fn surface_means_are_bit_identical_across_thread_counts() {
        let mut grid = HexGrid::from_radius(130);
        let mut state = 0x2545_F491_u32;
        for c in grid.cells_slice_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0;
            c.temperature = unit * 40.0 - 20.0;
            c.elevation = unit * 3000.0;
        }
        let reference = surface_means(&grid);
        for threads in [1, 4] {
            let pooled = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build pool")
                .install(|| surface_means(&grid));
            assert_eq!(
                pooled.0.to_bits(),
                reference.0.to_bits(),
                "{threads} threads: mean_t"
            );
            assert_eq!(
                pooled.1.to_bits(),
                reference.1.to_bits(),
                "{threads} threads: mean_z"
            );
        }
    }

    /// The upper air is horizontally mixed: two cells at the same
    /// elevation share the same upper-air temperature whatever their
    /// surface anomaly (aspect, lake, snow), and a higher cell only
    /// sees the standard lapse for its extra height. A cold surface
    /// therefore cannot become a permanent condenser aloft (JOURNAL
    /// 2026-09-02, bisected to the aspect insolation e3594f9).
    #[test]
    fn upper_air_ignores_surface_anomaly_and_follows_elevation() {
        let params = AtmosphereParams::default();
        let temp_params = default_temp_params();
        let mut grid = HexGrid::from_radius(1);
        let coords: Vec<HexCoord> = grid.coords().copied().collect();
        for (k, c) in coords.iter().enumerate() {
            let cell = grid.get_mut(*c).unwrap();
            cell.elevation = if k == 0 { 1000.0 } else { 0.0 };
            // A 10 °C surface contrast between two flat cells (ubac vs adret).
            cell.temperature = if k == 1 { 5.0 } else { 15.0 };
        }
        let (mean_t, mean_z) = surface_means(&grid);
        let ubac = upper_air_temperature(mean_t, mean_z, 0.0, &params, &temp_params);
        let adret = upper_air_temperature(mean_t, mean_z, 0.0, &params, &temp_params);
        assert_eq!(
            ubac.to_bits(),
            adret.to_bits(),
            "same elevation ⇒ same upper air"
        );
        let summit = upper_air_temperature(mean_t, mean_z, 1000.0, &params, &temp_params);
        let expected_drop = temp_params.lapse_rate;
        assert!(
            ((ubac - summit) - expected_drop).abs() < 1e-4,
            "1000 m higher ⇒ {expected_drop} °C colder aloft, got {}",
            ubac - summit
        );
        // Single cell: reduces to the historical `T − Γ·H`.
        let alone = HexGrid::from_radius(0);
        let (t1, z1) = surface_means(&alone);
        let hist = 0.0 - temp_params.lapse_rate * params.upper_layer_altitude_m / 1000.0;
        assert!((upper_air_temperature(t1, z1, 0.0, &params, &temp_params) - hist).abs() < 1e-5);
    }

    #[test]
    fn cold_upper_air_precipitates_more_easily() {
        // With Clausius-Clapeyron saturation, cold air saturates for less
        // vapor: saturation(-5°C) < saturation(0°C) < saturation(20°C). So
        // a cold cell precipitates at a much lower absolute humidity than
        // a warm cell. This is the inversion of linear reasoning: the
        // sensitivity comes from the exp curve, not from an offset.
        let params = AtmosphereParams::default();
        let sat_cold = saturation_upper(0.0, &params);
        let sat_warm = saturation_upper(20.0, &params);
        assert!(
            sat_cold < sat_warm,
            "saturation 0°C ({sat_cold:.3}) must be < saturation 20°C ({sat_warm:.3})"
        );
    }

    /// Hourly samples of one day, as many as ticks in a day.
    const HOURS_PER_DAY: u8 = 24;

    /// The upper-air anchor is a first-order filter with τ = 24 h: a
    /// 24 h sinusoid (the diurnal cycle of the mean surface, ~8 K) must
    /// come out attenuated to `|H| = 1/√(1+(ωτ)²)` with ωτ = 2π, i.e.
    /// ≈ 0.157 of its amplitude, the ~1 K of the free troposphere. The
    /// discrete EMA (gain `1 − e^{−1/24}` per hour) sits within 0.5 % of
    /// the continuous filter at that frequency; measured on the last 5
    /// of 30 days, once the transient has died out.
    #[test]
    fn upper_air_smoothing_attenuates_the_diurnal_harmonic() {
        let amplitude = 8.0_f32;
        let mean = 10.0_f32;
        let days = 30;
        let mut m = mean;
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for day in 0..days {
            for hour in 0..HOURS_PER_DAY {
                let phase = 2.0 * std::f32::consts::PI * f32::from(hour) / f32::from(HOURS_PER_DAY);
                let surface_t = mean + amplitude * phase.sin();
                m = smooth_upper_air_mean_t(m, surface_t);
                if day >= days - 5 {
                    lo = lo.min(m);
                    hi = hi.max(m);
                }
            }
        }
        let measured_ratio = (hi - lo) / (2.0 * amplitude);
        let omega_tau = 2.0 * std::f32::consts::PI;
        let expected_ratio = 1.0 / (1.0 + omega_tau * omega_tau).sqrt();
        assert!(
            ((measured_ratio - expected_ratio) / expected_ratio).abs() < 0.03,
            "diurnal amplitude ratio {measured_ratio:.4}, first-order filter predicts \
             {expected_ratio:.4} (τ = 24 h)"
        );
        // The mean goes through untouched: the filter removes the
        // harmonic, not the level (the seasons must survive).
        let centre = f32::midpoint(hi, lo);
        assert!(
            (centre - mean).abs() < 0.05,
            "the diurnal mean must pass unchanged: centre {centre:.3} vs {mean}"
        );
    }

    /// Step response: after exactly τ (24 hourly steps) the residual of
    /// a step change is `e^{−1}` of the step, the e-fold time of the
    /// filter (a seasonal change reaches the upper air within days).
    #[test]
    fn upper_air_smoothing_converges_with_e_fold_time_tau() {
        let from = -5.0_f32;
        let to = 15.0_f32;
        let mut m = from;
        for _ in 0..HOURS_PER_DAY {
            m = smooth_upper_air_mean_t(m, to);
        }
        let residual = (to - m) / (to - from);
        let expected = (-1.0_f32).exp();
        assert!(
            (residual - expected).abs() < 1e-3,
            "residual after τ = {residual:.4}, expected e^-1 = {expected:.4}"
        );
        for _ in 0..(10 * HOURS_PER_DAY) {
            m = smooth_upper_air_mean_t(m, to);
        }
        assert!(
            (m - to).abs() < 1e-3,
            "after 11τ the anchor must have converged to the step: {m} vs {to}"
        );
    }

    /// #63 L2b, the shape of the saturation adjustment: after one step in
    /// subsaturated air the cell is in one of exactly two states, no cloud
    /// left or a saturated layer. Swept across the whole range of
    /// cloud/deficit ratios so both outcomes and the exact crossover are
    /// covered.
    #[test]
    fn subsaturated_air_leaves_either_no_cloud_or_a_saturated_layer() {
        let params = AtmosphereParams::default();
        let t_up = 5.0;
        let sat = saturation_upper(t_up, &params);
        // sat ≈ 9.3 mm at 5 °C over 1500 m; RH 0.3 leaves a ~6.5 mm
        // deficit, so clouds on both sides of it are exercised.
        for rh in [0.0_f32, 0.3, 0.7, 0.999] {
            for cloud in [1e-6_f32, 0.2, 5.0, 50.0] {
                let mut cell = CellProperties {
                    humidity_upper: rh * sat,
                    cloud_water: cloud,
                    ..CellProperties::default()
                };
                cloud_dynamics_for_cell(&mut cell, t_up, &params, 1.0);
                let cleared = cell.cloud_water == 0.0;
                let saturated = (cell.humidity_upper - sat).abs() <= 1e-4 * sat.max(1.0);
                assert!(
                    cleared || saturated,
                    "rh={rh} cloud={cloud}: neither cleared nor saturated \
                     (cloud={}, hu={}, sat={sat})",
                    cell.cloud_water,
                    cell.humidity_upper
                );
                assert!(
                    cell.humidity_upper <= sat + 1e-4 * sat.max(1.0),
                    "rh={rh} cloud={cloud}: the adjustment oversaturated the layer, \
                     hu={} > sat={sat}",
                    cell.humidity_upper
                );
            }
        }
    }

    /// A tiny cloud in very dry air reaches a TRUE zero, no residue and no
    /// floor: with `cloud_evap_rate` the stock decayed geometrically and
    /// never got there, which is exactly how a 0.2-0.5 mm drizzle survived
    /// a 4.5-day dry episode (JOURNAL 2026-09-06). One step is enough now.
    #[test]
    fn a_tiny_cloud_in_dry_air_reaches_a_true_zero() {
        let params = AtmosphereParams::default();
        let t_up = -10.0;
        let mut cell = CellProperties {
            humidity_upper: 0.0,
            cloud_water: 1e-7,
            ..CellProperties::default()
        };
        cloud_dynamics_for_cell(&mut cell, t_up, &params, 1.0);
        assert_eq!(
            cell.cloud_water.to_bits(),
            0.0_f32.to_bits(),
            "the cloud must reach exactly 0.0, got {}",
            cell.cloud_water
        );
    }

    /// At or above saturation the reverse branch never fires: droplets are
    /// only ever created there (condensation), never returned. Checked at
    /// exact saturation too, the boundary the `else` sits on.
    #[test]
    fn at_or_above_saturation_the_reverse_branch_is_a_no_op() {
        let params = AtmosphereParams::default();
        let t_up = 12.0;
        let sat = saturation_upper(t_up, &params);
        for hu in [sat, sat * 1.5, sat * 10.0] {
            let mut cell = CellProperties {
                humidity_upper: hu,
                cloud_water: 3.0,
                ..CellProperties::default()
            };
            // condensation_rate = 0 isolates the reverse branch: any
            // change to the stocks here can only come from it.
            cloud_dynamics_for_cell(&mut cell, t_up, &params, 0.0);
            assert_eq!(
                cell.cloud_water.to_bits(),
                3.0_f32.to_bits(),
                "hu={hu} (sat={sat}): droplets moved, got {}",
                cell.cloud_water
            );
            assert_eq!(
                cell.humidity_upper.to_bits(),
                hu.to_bits(),
                "hu={hu} (sat={sat}): vapour moved, got {}",
                cell.humidity_upper
            );
        }
    }

    proptest! {
        /// The transition is a transfer, in both directions and at any
        /// temperature: `humidity_upper + cloud_water` comes out of the
        /// step as it went in, and neither stock goes negative. This is
        /// the terrarium invariant at the scale of one cell.
        #[test]
        fn prop_cloud_dynamics_conserves_upper_water(
            hu in 0.0_f32..80.0,
            cloud in 0.0_f32..40.0,
            t_up in -40.0_f32..40.0,
            rate in 0.0_f32..1.0,
        ) {
            let params = AtmosphereParams::default();
            let mut cell = CellProperties {
                humidity_upper: hu,
                cloud_water: cloud,
                ..CellProperties::default()
            };
            let before = hu + cloud;
            cloud_dynamics_for_cell(&mut cell, t_up, &params, rate);
            let after = cell.humidity_upper + cell.cloud_water;
            // f32 relative tolerance: the transfer is one add and one
            // subtract on stocks up to 120 mm.
            prop_assert!(
                (after - before).abs() <= 1e-5 * before.max(1.0),
                "before={before} after={after} (hu={} cloud={})",
                cell.humidity_upper, cell.cloud_water
            );
            prop_assert!(cell.cloud_water >= 0.0, "negative cloud {}", cell.cloud_water);
            prop_assert!(cell.humidity_upper >= 0.0, "negative vapour {}", cell.humidity_upper);
        }
    }

    /// r250 perf effort: mass conservation of `step_cloud_diffusion`'s
    /// scatter -> gather split, radius-2, and the isotropic split lands
    /// equally on all 6 neighbors (no `coord::opposite_direction` bias:
    /// the uniform-share simplification only holds if every neighbor of
    /// the loaded center gets exactly the same amount back).
    #[test]
    fn cloud_diffusion_conserves_total_and_splits_equally_among_neighbors() {
        let mut grid = HexGrid::from_radius(2);
        grid.get_mut(HexCoord::new(0, 0)).unwrap().cloud_water = 6.0;
        let params = AtmosphereParams {
            cloud_diffusion_rate: 0.3,
            ..AtmosphereParams::default()
        };
        let before: f32 = grid.iter().map(|(_, c)| c.cloud_water).sum();

        let mut next = grid.clone();
        let mut share = Vec::new();
        let mut deltas = Vec::new();
        step_cloud_diffusion(&grid, &mut next, &params, &mut share, &mut deltas);

        let after: f32 = next.iter().map(|(_, c)| c.cloud_water).sum();
        assert!(
            (before - after).abs() < 1e-4,
            "conservation violated: before={before}, after={after}"
        );
        let center = HexCoord::new(0, 0);
        let expected_each = 6.0 * 0.3 / 6.0;
        for n in center.neighbors() {
            let got = next.get(n).unwrap().cloud_water;
            assert!(
                (got - expected_each).abs() < 1e-5,
                "neighbor {n:?}: got {got}, expected {expected_each} (equal split)"
            );
        }
        let center_after = next.get(center).unwrap().cloud_water;
        assert!(
            (center_after - 6.0 * 0.7).abs() < 1e-5,
            "center must keep exactly (1 - rate) of its cloud water, got {center_after}"
        );
    }
}
