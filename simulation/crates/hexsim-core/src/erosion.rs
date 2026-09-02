//! Fluvial erosion (#105): stream power, two modes.
//!
//! Common law: bedrock incision via the stream power law `E = k·(Q/Q_ref)^m·S^n`
//! (Whipple & Tucker 1999), capped by the CFL ceiling on the energy drop.
//! Two paths consume this law:
//!
//! - **One-shot at worldgen** ([`erode_terrain`], the DEFAULT mode): sculpts
//!   the drainage network into the bare terrain at world generation, then
//!   elevation is frozen. Forcing = flow accumulation on slope alone
//!   ([`accumulate_flow`], no climate). **Incision only, non-conservative**
//!   (incised rock leaves the catchment the way a real river's sediment
//!   reaches the sea): conserving mass in the CLOSED torus would only
//!   smooth out the relief. No runtime cost, no drift toward flatness.
//! - **Live** ([`step_erosion`], opt-in `erosion.enabled = 1`): one step per
//!   simulated day, forced by the **EMA** of climate discharge (anti-pattern
//!   #3), with sediment load and deposition, CONSERVATIVE
//!   (`Σ(elevation)+Σ(sediment_load)` invariant), since it's a runtime
//!   terrarium phenomenon. Drifts toward flatness on very long runs (no
//!   tectonics): hence the one-shot default.
//!
//! Strict SI units: elevations and load in m, discharge in m³/s (converted
//! from the engine's mm/day via `CELL_AREA_M2`), slope in m/m, erodability
//! coefficients in m/s. The only non-physical factor is
//! `accel_years_per_day`, the **explicit** compression of geological time.

use serde::{Deserialize, Serialize};

use crate::cell::CellProperties;
use crate::dynamics::{CELL_AREA_M2, CELL_SPACING_M};
use crate::grid::HexGrid;
use crate::hydro::EdgeFluxMap;
use crate::units::MM_PER_M;

/// Seconds per day (SI).
const SECONDS_PER_DAY: f32 = 86_400.0;
/// Seconds per year in the engine calendar (365 days, `time::DAYS_PER_YEAR`).
const SECONDS_PER_YEAR: f32 = 365.0 * SECONDS_PER_DAY;

/// Reference discharge of the power law (m³/s). Normalizing `Q` by this
/// constant keeps `k_incision`/`k_transport` in m/s **regardless of the
/// exponent `m`**: without it, the coefficient's unit would depend on the
/// exponent (a classic dimensional trap in stream power).
pub const Q_REF_M3_S: f32 = 1.0;

/// EMA of daily discharge (mm/day), indexed by `HexGrid::cell_index`.
/// Updated once per day after the hydro slice, even with erosion disabled:
/// rendering (#106) will stabilize the network on these same maps.
pub type DischargeEmaMap = Vec<f32>;

/// EMA of per-edge flux (mm/day, order `coord::DIRECTIONS`), same lifecycle
/// as [`DischargeEmaMap`]. It's the directional memory that routes sediment
/// load (sediment follows water, not a precomputed DAG).
pub type EdgeFluxEmaMap = Vec<[f32; 6]>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErosionParams {
    /// Switch for LIVE mode (erosion on every tick). `false` by default
    /// since the one-shot pivot (#105): the relief is sculpted once at
    /// worldgen ([`erode_terrain`]) then frozen. Setting `true` re-enables
    /// continuous erosion (`step_erosion` in Tier 3): geology that evolves
    /// in-game, at the cost of a slow drift toward flatness on very long
    /// runs.
    pub enabled: bool,
    /// Bedrock erodability `k_incision` (m/s): incision rate at
    /// `Q = Q_REF_M3_S` and slope S = 1. Default 6.3e-10 m/s ≈ 0.02 m/yr;
    /// rescaled to a realistic river (Q = 1 m³/s, S = 0.05) that gives
    /// E ≈ 1 mm/yr, mid-range of the observed bedrock incision range
    /// (0.1-10 mm/yr, Whipple & Tucker 1999).
    pub k_incision: f32,
    /// Transport capacity coefficient (m/s), same form as incision
    /// (`C = k_transport·(Q/Q_ref)^m·S^n`). Default 20 × `k_incision`: a
    /// river transports far more than it locally detaches; the ratio sets
    /// the detachment-limited regime (upstream) vs. transport-limited
    /// regime (piedmont, where load exceeds C and deposits).
    pub k_transport: f32,
    /// Discharge exponent `m` (dimensionless). 0.5 = "unit stream power",
    /// the best-documented (m, n) pair (m/n ≈ 0.5).
    pub m_exponent: f32,
    /// Slope exponent `n` (dimensionless). 1.0, cf. `m_exponent`.
    pub n_exponent: f32,
    /// Time constant of the hydro flux EMA (days). 60 days smooths storms
    /// and slice noise while preserving seasonality; the EMA starts at zero
    /// → erosion ramps up over ~3τ (warm-up assumed, all measurements taken
    /// after ≥ 1 year).
    pub tau_days: f32,
    /// Time acceleration factor: years of **geological** time applied per
    /// **simulated** day. An explicit scale decision (not a catch-all
    /// coefficient): real incision is in mm/yr, a dendritic network takes
    /// 10⁴-10⁷ years to carve, invisible at 1:1. Default 20 yr/day → 10
    /// simulated years ≈ 73,000 years of erosion.
    pub accel_years_per_day: f32,
    /// Stability bound of the explicit scheme: one day's incision cannot
    /// exceed this fraction of the ENERGY drop (delta of
    /// `effective_elevation` toward the lowest neighbour, the same free
    /// surface that carries the stream power). Without this bound, a cell
    /// could reverse the gradient driving the flow in a single step and
    /// oscillate. Exact analogue of the hydro MFD's CFL cap: a documented
    /// discretization condition, not an arbitrary defensive limit.
    ///
    /// Bounding on energy rather than bedrock is a deliberate physical
    /// choice: a powerful flow can carve BELOW its downstream neighbour's
    /// bed (overdeepening, potholes) as long as its water drains away; once
    /// the pit traps water, the free surface flattens, power collapses, and
    /// incision self-stops. This is the "overdeepening → closed basin →
    /// real lake" mechanism from #105: bounding on bedrock would forbid it
    /// by construction. A cell at the bottom of a full pit (flat free
    /// surface) does not incise: it fills in.
    pub cfl_drop_frac: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            // OFF at runtime (#105 one-shot): erosion is computed ONCE at
            // world generation (`erode_terrain`), not every tick. Without
            // tectonics to counterbalance it, endless live erosion drifts
            // toward flatness (peneplanation) on very long runs and costs a
            // tick forever: the world is therefore pre-eroded at worldgen
            // then frozen. The live switch stays available
            // (`erosion.enabled = 1`) for whoever wants geology that
            // evolves in-game.
            enabled: false,
            k_incision: 6.3e-10,
            k_transport: 20.0 * 6.3e-10,
            m_exponent: 0.5,
            n_exponent: 1.0,
            tau_days: 60.0,
            accel_years_per_day: 20.0,
            cfl_drop_frac: 0.25,
        }
    }
}

/// Flow-routing concentration exponent at worldgen (Tarboton D-inf, same
/// role as `HydroParams::flow_concentration`). 4.0 = strong MFD: channels
/// concentrate (dendritic network) without flipping to pure D8.
pub const WORLDGEN_FLOW_CONCENTRATION: f32 = 4.0;

/// Divisor that rescales the **dimensionless** aggressiveness knob of the
/// "capture" model (`ErosionParams::accel_years_per_day`, cf. `erode_terrain`
/// and `TerrainParams::erosion_accel_years`) into "meters removed per
/// iteration" (default 2000 → 20 m/it). NON-SI by design: capture is a
/// deliberately non-physical worldgen sculpting model, there is no SI flux
/// to derive this factor from.
const ACCEL_UNITS_PER_METER: f32 = 100.0;

/// Default time acceleration of the one-shot erosion pass (geological years
/// per iteration). CRUCIAL: too high, incision saturates the CFL ceiling
/// (25% of the drop) on EVERY cell with flow → each cell slides 25% toward
/// its lowest neighbour, which is a SMOOTHING filter (the "melt"). Low
/// enough, incision stays ∝ power (∝√A·S) and CONCENTRATES in channels →
/// valleys deepen, relief increases. 2000 yr/iteration: channels
/// concentrate (measured, seed 7 r30: accumulation gini 0.71 → 0.74 at 25
/// iterations, local relief preserved) without saturating the ceiling.
/// Going higher carves deeper but exports more rock (terrain lowers,
/// climate drifts). Tunable per world via
/// `TerrainParams::erosion_accel_years` (default = this constant).
pub const WORLDGEN_ACCEL_YEARS: f32 = 2000.0;

impl ErosionParams {
    /// Parameters for one-shot erosion at generation (#105). Inherits the
    /// defaults (SI stream power) but forces the default worldgen
    /// acceleration, low enough for incision to concentrate in channels
    /// (not a smoothing pass). For a per-world custom acceleration, build an
    /// `ErosionParams` with an explicit `accel_years_per_day` (what
    /// `generate_terrain` does from `TerrainParams::erosion_accel_years`).
    #[must_use]
    pub fn for_worldgen() -> Self {
        Self {
            accel_years_per_day: WORLDGEN_ACCEL_YEARS,
            ..Self::default()
        }
    }
}

/// MFD flow accumulation on the **bare** terrain (no climate): each cell
/// receives one unit of "rain", routed toward lower neighbours in
/// `effective_elevation`, weighted by `delta^concentration`. Returns
/// `(discharge, edge_flux)`: the cumulative discharge per cell (≈ drained
/// area, in cell units) and its breakdown per edge, exactly the forcing
/// [`step_erosion`] expects for erosion at world creation.
///
/// Processing order is decreasing in elevation: by the time a cell is
/// reached, all of its upstream input is already accumulated (strict
/// "lowest" is a partial order compatible with elevation → a single pass
/// suffices, no convergence iteration). Endorheic: a basin (no lower
/// neighbour) is a sink, flow stops there: it's the future lake.
#[must_use]
pub fn accumulate_flow(grid: &HexGrid, concentration: f32) -> (Vec<f32>, EdgeFluxMap) {
    let cells = grid.cells_slice();
    let n = cells.len();
    let eff: Vec<f32> = cells
        .iter()
        .map(CellProperties::effective_elevation)
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eff[b].total_cmp(&eff[a]));

    let mut accum = vec![1.0_f32; n]; // 1 unit of rain per cell
    let mut edge_flux = vec![[0.0_f32; 6]; n];
    let mut discharge = vec![0.0_f32; n];
    for &i in &order {
        let through = accum[i];
        discharge[i] = through;
        let neighbors = grid.neighbor_indices_toric(i);
        let mut weights = [0.0_f32; 6];
        let mut wsum = 0.0_f32;
        for (d, &j) in neighbors.iter().enumerate() {
            let delta = eff[i] - eff[j];
            if delta > 0.0 {
                let w = delta.powf(concentration);
                weights[d] = w;
                wsum += w;
            }
        }
        if wsum > 0.0 {
            for (d, &j) in neighbors.iter().enumerate() {
                if weights[d] > 0.0 {
                    let f = through * (weights[d] / wsum);
                    edge_flux[i][d] = f;
                    accum[j] += f;
                }
            }
        }
    }
    (discharge, edge_flux)
}

/// **One-shot** erosion at world creation (#105). Sculpts a dendritic
/// drainage network into the bare terrain over `iterations` passes of
/// (flow accumulation → carve ∝ discharge → re-route), then **freezes**
/// elevation.
///
/// **The "capture" model (2026-07-11, author's idea), deliberately
/// simple:** EACH cell is carved proportionally to ITS OWN discharge (the
/// drained area from `accumulate_flow`), **LINEARLY**, no √discharge and no
/// slope term. This is what triggers the capture instability: the dominant
/// channel has more discharge → carves more → becomes lower → when flow is
/// recomputed on the next pass, its neighbours see it as the low point and
/// **pour into it** → it captures even more discharge → it carves even
/// deeper. Parallel rills merge into a tree. The √discharge of classic
/// stream power COMPRESSES this gap (√200/√5 ≈ 6× instead of 40×) and its
/// slope term GRADES the profile (spreads the water out): measured, it does
/// not capture.
///
/// **Non-conservative, and that's a FEATURE at worldgen** (author's note):
/// the carved rock **disappears** (like a real river's sediment reaching
/// the sea). No conservation to respect here: Σrock conservation is an
/// invariant of the **runtime terrarium** (water/energy), not of terrain
/// sculpting. Carving is capped at `cfl × drop to the lowest neighbour` so
/// as not to reverse the slope in a single step; a cell that has become a
/// low point (drop ≤ 0) stops carving (basin → lake).
///
/// Zero runtime cost, frozen relief, deterministic. No-op if
/// `iterations == 0`.
pub fn erode_terrain(grid: &mut HexGrid, params: &ErosionParams, iterations: u32) {
    if iterations == 0 {
        return;
    }
    let n = grid.len();
    // Carving strength: meters removed per iteration for the STRONGEST
    // channel (others pro-rated by their discharge). Derived from the
    // `accel` field so it stays hot-reloadable (`terrain.erosion_accel`)
    // without a new field. The `ACCEL_UNITS_PER_METER` divisor (non-SI, cf.
    // its doc) rescales the dimensionless knob into meters/iteration.
    // `.max(0.0)`: `accel` is a carving magnitude; a value < 0 would mean
    // "deposit", meaningless for one-shot worldgen: this is a genuine
    // physical floor (0 = no sculpting), not a masking safeguard.
    let k_meters = (params.accel_years_per_day / ACCEL_UNITS_PER_METER).max(0.0);
    let cfl = params.cfl_drop_frac.max(0.05);
    let mut carve = vec![0.0_f32; n];
    for _ in 0..iterations {
        let (discharge, _edge) = accumulate_flow(grid, WORLDGEN_FLOW_CONCENTRATION);
        let maxq = discharge.iter().copied().fold(1.0_f32, f32::max);
        let cells = grid.cells_slice();
        for i in 0..n {
            let eff_i = cells[i].effective_elevation();
            let min_nb = grid
                .neighbor_indices_toric(i)
                .iter()
                .map(|&j| cells[j].effective_elevation())
                .fold(f32::INFINITY, f32::min);
            let drop = (eff_i - min_nb).max(0.0);
            carve[i] = (k_meters * discharge[i] / maxq).min(cfl * drop);
        }
        let cells = grid.cells_slice_mut();
        for (i, c) in cells.iter_mut().enumerate() {
            c.elevation -= carve[i];
        }
    }
}

/// Hydro history maps consumed by [`step_erosion`]: produced by the daily
/// hydro slice, smoothed by [`update_ema`]/[`update_edge_ema`].
pub struct ErosionForcing<'a> {
    /// EMA of discharge (mm/day), see [`DischargeEmaMap`].
    pub discharge_ema: &'a [f32],
    /// EMA of per-edge flux (mm/day ×6), see [`EdgeFluxEmaMap`].
    pub edge_flux_ema: &'a [[f32; 6]],
}

/// Balance of one erosion step (m accumulated over the map): instrumentation
/// for diagnostics and the simulation's cumulative counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct ErosionTotals {
    /// Bedrock detached (Σ of the day's incisions, m).
    pub incised_m: f32,
    /// Load turned back into bedrock (Σ of the day's deposits, m).
    pub deposited_m: f32,
}

/// Smoothing step: `ema += (day − ema)/τ`. τ ≤ 1 ⇒ α = 1 (no memory).
fn ema_alpha(tau_days: f32) -> f32 {
    if tau_days <= 1.0 { 1.0 } else { 1.0 / tau_days }
}

/// Updates the discharge EMA with the day's map (resizes as needed:
/// pre-#105 checkpoints load an empty map).
pub fn update_ema(ema: &mut DischargeEmaMap, day: &[f32], tau_days: f32) {
    ema.resize(day.len(), 0.0);
    let alpha = ema_alpha(tau_days);
    for (e, &v) in ema.iter_mut().zip(day) {
        *e += alpha * (v - *e);
    }
}

/// Updates the per-edge flux EMA, same contract as [`update_ema`].
pub fn update_edge_ema(ema: &mut EdgeFluxEmaMap, day: &EdgeFluxMap, tau_days: f32) {
    ema.resize(day.len(), [0.0; 6]);
    let alpha = ema_alpha(tau_days);
    for (row, day_row) in ema.iter_mut().zip(day) {
        for (e, &v) in row.iter_mut().zip(day_row) {
            *e += alpha * (v - *e);
        }
    }
}

/// One daily erosion step: incision → sediment transport → deposition.
///
/// Pure phenomenon: reads `current` (+ the EMA forcing), writes `next`,
/// deterministic. For each cell:
///
/// 1. **Stream power**: `P = (Q/Q_ref)^m · S^n` with `Q` the discharge EMA
///    converted to m³/s and `S` the energy slope (steepest drop of
///    `effective_elevation` toward a toric neighbour, in m/m: a lake's
///    surface is flat ⇒ P ≈ 0 at the bottom of lakes).
/// 2. **Transport**: `mobile = min(load, k_transport·P·dt_geo)` moves
///    toward **downhill** neighbours pro-rated by `edge_flux_ema` (sediment
///    follows water); the excess **deposits** (`elevation += deposit`). No
///    wet downhill direction → everything deposits.
/// 3. **Incision**: `k_incision·P·dt_geo`, bounded by the remaining capacity
///    (cover effect: a saturated flow no longer detaches) and by
///    `cfl_drop_frac × bedrock drop` (stability of the explicit scheme).
///    The detached material joins the outgoing load.
///
/// The load advances one cell per day (the group velocity assumed by the
/// flow registry). Conservation: the rounding residue of the directional
/// split stays in the local load; we only subtract what was actually sent,
/// same as the hydro MFD.
pub(crate) fn step_erosion(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &ErosionParams,
    forcing: &ErosionForcing<'_>,
) -> ErosionTotals {
    let n = current.len();
    let cur_cells = current.cells_slice();
    next.cells_slice_mut().clone_from_slice(cur_cells);
    let next_cells = next.cells_slice_mut();

    // Seconds of geological time applied per simulated day.
    let dt_geo_s = SECONDS_PER_YEAR * params.accel_years_per_day;
    let mut totals = ErosionTotals::default();

    for i in 0..n {
        let cell = &cur_cells[i];
        // Negative rounding dust (≲ 1e-9 m): we don't "correct" it (that
        // would create mass), we ignore it as zero load.
        let load = cell.sediment_load.max(0.0);
        let q_ema_mm_day = forcing.discharge_ema.get(i).copied().unwrap_or(0.0);
        if load <= 0.0 && q_ema_mm_day <= 0.0 {
            continue;
        }

        // One neighbourhood pass: max energy slope (free surface, carries
        // both the stream power and the CFL bound) and routing weights (edge
        // EMA toward neighbours strictly downhill in free surface).
        let neighbors = current.neighbor_indices_toric(i);
        let eff_i = cell.effective_elevation();
        let edge_ema = forcing.edge_flux_ema.get(i);
        let mut s_max = 0.0_f32;
        let mut weights = [0.0_f32; 6];
        let mut weight_sum = 0.0_f32;
        for (d, &j) in neighbors.iter().enumerate() {
            let nb = &cur_cells[j];
            let slope = (eff_i - nb.effective_elevation()) / CELL_SPACING_M;
            if slope > s_max {
                s_max = slope;
            }
            if slope > 0.0
                && let Some(row) = edge_ema
                && row[d] > 0.0
            {
                weights[d] = row[d];
                weight_sum += row[d];
            }
        }

        // Q: mm/day sheet depth over the cell's area → m³/s SI.
        let q_si = q_ema_mm_day * CELL_AREA_M2 / (MM_PER_M * SECONDS_PER_DAY);
        let power = if q_si > 0.0 && s_max > 0.0 {
            (q_si / Q_REF_M3_S).powf(params.m_exponent) * s_max.powf(params.n_exponent)
        } else {
            0.0
        };
        let capacity_m = params.k_transport * power * dt_geo_s;

        // Transport / deposit: with no memory of downhill flow, nothing
        // leaves.
        let (mobile, deposit) = if weight_sum > 0.0 {
            let mobile = load.min(capacity_m);
            (mobile, load - mobile)
        } else {
            (0.0, load)
        };

        // Detachment-limited incision, cover effect, CFL bound on the
        // ENERGY drop (`s_max × d` = delta of effective_elevation toward the
        // lowest neighbour). Bedrock CAN go below the downstream bed
        // (overdeepening → basin → lake); the self-stop comes from water
        // pooling in the pit and flattening the free surface (power → 0).
        let mut incision = 0.0_f32;
        if weight_sum > 0.0 && power > 0.0 {
            let drop_energy = s_max * CELL_SPACING_M;
            incision = (params.k_incision * power * dt_geo_s)
                .min(capacity_m - mobile)
                .min(params.cfl_drop_frac * drop_energy)
                .max(0.0);
        }

        // Downstream routing + exact bookkeeping: we subtract `sent` (what
        // actually left), not the theoretical target: the rounding residue
        // of the split stays in the local load (strict conservation).
        let outgoing = mobile + incision;
        let mut sent = 0.0_f32;
        if outgoing > 0.0 && weight_sum > 0.0 {
            for (d, &j) in neighbors.iter().enumerate() {
                if weights[d] <= 0.0 {
                    continue;
                }
                let transfer = outgoing * (weights[d] / weight_sum);
                if transfer <= 0.0 {
                    continue;
                }
                next_cells[j].sediment_load += transfer;
                sent += transfer;
            }
        }
        let nc = &mut next_cells[i];
        nc.elevation += deposit - incision;
        nc.sediment_load += incision - deposit - sent;
        totals.incised_m += incision;
        totals.deposited_m += deposit;
    }
    totals
}

/// Total "rock" budget: Σ(`elevation`) + Σ(`sediment_load`), in m. Erosion's
/// terrarium invariant, the counterpart of `hydro::total_water`.
#[must_use]
pub fn total_rock(grid: &HexGrid) -> f64 {
    grid.cells_slice()
        .iter()
        .map(|c| f64::from(c.elevation) + f64::from(c.sediment_load))
        .sum()
}

/// Gini index of a discharge map ∈ [0, 1[: 0 = discharge evenly spread
/// (parallel rills), → 1 = all discharge in a few channels (concentrated
/// dendritic network). This is #105's drainage convergence metric: rills
/// merging makes the Gini index RISE. Negative values (rounding dust) count
/// as 0; an empty or dry map ⇒ 0.
#[must_use]
pub fn discharge_gini(discharge: &[f32]) -> f64 {
    let mut values: Vec<f32> = discharge.iter().map(|&x| x.max(0.0)).collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let total: f64 = values.iter().map(|&x| f64::from(x)).sum();
    if total <= 0.0 {
        return 0.0;
    }
    // Gini formula on sorted values: G = 2·Σ(i·xᵢ)/(n·Σx) − (n+1)/n,
    // rank i ∈ [1, n]. f64 counter (no usize→f64 cast).
    let mut rank = 0.0_f64;
    let mut weighted = 0.0_f64;
    for &x in &values {
        rank += 1.0;
        weighted += rank * f64::from(x);
    }
    2.0 * weighted / (rank * total) - (rank + 1.0) / rank
}

/// Indices of closed basins: cells whose **bedrock** is strictly lower than
/// all 6 toric neighbours. Countable before/after a run to measure
/// *emergent* basins (carved or dammed by erosion). A degenerate grid
/// (self-neighbour) has no basin by convention.
#[must_use]
pub fn closed_depression_indices(grid: &HexGrid) -> Vec<usize> {
    let cells = grid.cells_slice();
    (0..cells.len())
        .filter(|&i| {
            let elev = cells[i].elevation;
            grid.neighbor_indices_toric(i)
                .iter()
                .all(|&j| j != i && cells[j].elevation > elev)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use proptest::prelude::*;

    /// Elevation ramp along the world x axis (2q + r, same idiom as hydro's
    /// `edge_flux_targets_downhill_direction` test): west (dir 3) is
    /// consistently downhill. Radius 2: transport requires radius >= 2 (a
    /// radius-0 cell is its own neighbor ×6 on the torus).
    fn build_ramp_grid() -> HexGrid {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                c.elevation =
                    f32::from(i16::try_from(500 + 10 * (2 * coord.q + coord.r)).expect("fits i16"));
                c.water_level = 0.0;
                c.water_capacity = 0.0;
            }
        }
        grid
    }

    /// Uniform forcing: same discharge EMA everywhere, all edge flux toward
    /// the west (dir 3, downhill on the ramp).
    fn uniform_west_forcing(n: usize, q_mm_day: f32) -> (Vec<f32>, Vec<[f32; 6]>) {
        let discharge = vec![q_mm_day; n];
        let mut edge = vec![[0.0; 6]; n];
        for row in &mut edge {
            row[3] = q_mm_day;
        }
        (discharge, edge)
    }

    fn accelerated_params() -> ErosionParams {
        ErosionParams {
            // Test acceleration: make incision visible in 1 step.
            accel_years_per_day: 10_000.0,
            ..ErosionParams::default()
        }
    }

    #[test]
    fn incision_follows_discharge() {
        // Two identical worlds, only the EMA discharges differ (×100):
        // incision at the center must grow with discharge, this is where
        // the "more discharge → more carving" feedback begins.
        let grid = build_ramp_grid();
        let n = grid.len();
        let center = grid.cell_index(HexCoord::new(0, 0)).unwrap();
        let params = accelerated_params();

        let mut carved = [0.0_f32; 2];
        for (k, q) in [1.0_f32, 100.0].into_iter().enumerate() {
            let (discharge, edge) = uniform_west_forcing(n, q);
            let mut next = grid.clone();
            step_erosion(
                &grid,
                &mut next,
                &params,
                &ErosionForcing {
                    discharge_ema: &discharge,
                    edge_flux_ema: &edge,
                },
            );
            carved[k] = grid.cells_slice()[center].elevation - next.cells_slice()[center].elevation;
        }
        assert!(
            carved[0] > 0.0,
            "no incision at low discharge: {}",
            carved[0]
        );
        assert!(
            carved[1] > carved[0] * 5.0,
            "incision doesn't follow discharge: Q×100 → incision ×{:.2} only",
            carved[1] / carved[0]
        );
    }

    #[test]
    fn no_incision_without_water() {
        // Zero EMA discharge everywhere: the relief doesn't move a micron,
        // even with slope. Erosion is fluvial, not spontaneous.
        let grid = build_ramp_grid();
        let n = grid.len();
        let (discharge, edge) = uniform_west_forcing(n, 0.0);
        let mut next = grid.clone();
        let totals = step_erosion(
            &grid,
            &mut next,
            &accelerated_params(),
            &ErosionForcing {
                discharge_ema: &discharge,
                edge_flux_ema: &edge,
            },
        );
        assert!(totals.incised_m == 0.0 && totals.deposited_m == 0.0);
        for (before, after) in grid.cells_slice().iter().zip(next.cells_slice()) {
            assert!((before.elevation - after.elevation).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn deposition_when_capacity_drops() {
        // A cell loaded with sediment but with no discharge (zero capacity):
        // the whole load turns back into bedrock, in place. This is the
        // delta/floodplain mechanism, a lake's inlet has P ≈ 0.
        let mut grid = build_ramp_grid();
        let center = HexCoord::new(0, 0);
        let loaded = 0.5_f32;
        grid.get_mut(center).unwrap().sediment_load = loaded;
        let ci = grid.cell_index(center).unwrap();
        let elev_before = grid.cells_slice()[ci].elevation;

        let n = grid.len();
        let (discharge, edge) = uniform_west_forcing(n, 0.0);
        let mut next = grid.clone();
        let totals = step_erosion(
            &grid,
            &mut next,
            &accelerated_params(),
            &ErosionForcing {
                discharge_ema: &discharge,
                edge_flux_ema: &edge,
            },
        );

        let after = &next.cells_slice()[ci];
        assert!(
            (after.elevation - (elev_before + loaded)).abs() < 1e-6,
            "the deposit didn't rejoin bedrock: {} → {}",
            elev_before,
            after.elevation
        );
        assert!(
            after.sediment_load.abs() < 1e-6,
            "the load should have fully sedimented: {}",
            after.sediment_load
        );
        assert!((totals.deposited_m - loaded).abs() < 1e-6);
    }

    #[test]
    fn incision_bounded_by_cfl_energy_drop() {
        // Absurd erodability: one step's incision stays ≤ cfl_drop_frac ×
        // energy drop (on the dry ramp, free surface = bedrock): the
        // explicit scheme cannot reverse the gradient driving the flow in a
        // single day.
        let grid = build_ramp_grid();
        let n = grid.len();
        let params = ErosionParams {
            k_incision: 1.0,   // 1 m/s: absurd on purpose
            k_transport: 20.0, // keeps the cover ratio
            ..accelerated_params()
        };
        let (discharge, edge) = uniform_west_forcing(n, 1000.0);
        let mut next = grid.clone();
        step_erosion(
            &grid,
            &mut next,
            &params,
            &ErosionForcing {
                discharge_ema: &discharge,
                edge_flux_ema: &edge,
            },
        );
        for i in 0..n {
            let before = &grid.cells_slice()[i];
            let min_eff = grid
                .neighbor_indices_toric(i)
                .iter()
                .map(|&j| grid.cells_slice()[j].effective_elevation())
                .fold(f32::INFINITY, f32::min);
            let drop = (before.effective_elevation() - min_eff).max(0.0);
            let carved = before.elevation - next.cells_slice()[i].elevation;
            assert!(
                carved <= params.cfl_drop_frac * drop + 1e-5,
                "cell {i}: incision {carved} > {} × drop {drop}",
                params.cfl_drop_frac
            );
        }
    }

    /// Overdeepening (#105): a pothole full of DEEP water whose free
    /// surface dominates the downstream bed keeps carving BELOW that bed:
    /// this is the "overdeepening → closed basin → real lake" mechanism. A
    /// bound on bedrock (rejected version) would forbid it: 10 years of the
    /// instrument had produced ONLY dry basins.
    #[test]
    fn overdeepening_digs_below_downstream_bed() {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                // Uniform ring at 102 m, dry.
                c.elevation = 102.0;
                c.water_level = 0.0;
                c.water_capacity = 0.0;
            }
        }
        let center = HexCoord::new(0, 0);
        if let Some(c) = grid.get_mut(center) {
            // Bed at 100 m, ALREADY below the neighbours (102), but 4 m of
            // water: free surface at 104 m, energy still drops downstream.
            c.elevation = 100.0;
            c.water_level = 4000.0;
        }
        let ci = grid.cell_index(center).unwrap();
        let n = grid.len();
        let (discharge, edge) = uniform_west_forcing(n, 50.0);
        let mut next = grid.clone();
        step_erosion(
            &grid,
            &mut next,
            &accelerated_params(),
            &ErosionForcing {
                discharge_ema: &discharge,
                edge_flux_ema: &edge,
            },
        );
        assert!(
            next.cells_slice()[ci].elevation < 100.0,
            "no overdeepening below the downstream bed: {} (expected < 100)",
            next.cells_slice()[ci].elevation
        );
    }

    /// Overdeepening's self-stop: same geometry but TRAPPED water (the free
    /// surface no longer dominates downstream) → flat energy → power = 0 →
    /// no more incision. The pit holds its lake instead of digging forever.
    #[test]
    fn overdeepening_self_arrests_when_water_pools() {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                c.elevation = 102.0;
                c.water_level = 0.0;
                c.water_capacity = 0.0;
            }
        }
        let center = HexCoord::new(0, 0);
        if let Some(c) = grid.get_mut(center) {
            // Only 1 m of water: free surface at 101 m < downstream 102 m.
            c.elevation = 100.0;
            c.water_level = 1000.0;
        }
        let ci = grid.cell_index(center).unwrap();
        let n = grid.len();
        let (discharge, edge) = uniform_west_forcing(n, 50.0);
        let mut next = grid.clone();
        step_erosion(
            &grid,
            &mut next,
            &accelerated_params(),
            &ErosionForcing {
                discharge_ema: &discharge,
                edge_flux_ema: &edge,
            },
        );
        // Only the PIT stops (its energy no longer dominates downstream);
        // the upstream neighbour incising TOWARD the waterbody is
        // legitimate physics (headward erosion): we don't forbid it here.
        assert!(
            (next.cells_slice()[ci].elevation - 100.0).abs() < 1e-6,
            "the full pit kept incising: {}",
            next.cells_slice()[ci].elevation
        );
    }

    #[test]
    fn pit_cell_fills_but_never_incises() {
        // Basin: the center is lower than all its neighbours. Drop ≤ 0 → no
        // incision possible; load arriving there deposits (no downhill
        // direction) → the pit fills in, it never digs deeper.
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                let d = coord.distance(HexCoord::new(0, 0));
                c.elevation = f32::from(i16::try_from(10 * d).expect("fits i16"));
                c.water_level = 0.0;
                c.water_capacity = 0.0;
            }
        }
        let ci = grid.cell_index(HexCoord::new(0, 0)).unwrap();
        let n = grid.len();
        // Discharge everywhere (even at the bottom: an overflowing lake
        // could have some).
        let (discharge, edge) = uniform_west_forcing(n, 50.0);
        let mut next = grid.clone();
        step_erosion(
            &grid,
            &mut next,
            &accelerated_params(),
            &ErosionForcing {
                discharge_ema: &discharge,
                edge_flux_ema: &edge,
            },
        );
        assert!(
            next.cells_slice()[ci].elevation >= grid.cells_slice()[ci].elevation,
            "the basin floor incised: {} → {}",
            grid.cells_slice()[ci].elevation,
            next.cells_slice()[ci].elevation
        );
    }

    #[test]
    fn ema_warms_up_towards_signal() {
        // τ = 10 days: after 1 step the EMA is 10% of the signal, after
        // ~3τ ≈ 95%.
        let mut ema = vec![0.0_f32];
        update_ema(&mut ema, &[100.0], 10.0);
        assert!((ema[0] - 10.0).abs() < 1e-4);
        for _ in 0..29 {
            update_ema(&mut ema, &[100.0], 10.0);
        }
        assert!(ema[0] > 90.0, "EMA not converged after 3τ: {}", ema[0]);
        // Signal cut off: the EMA decays (memory, not a latch).
        update_ema(&mut ema, &[0.0], 10.0);
        assert!(ema[0] < 90.0);
    }

    #[test]
    fn gini_uniform_is_zero_concentrated_is_high() {
        let uniform = vec![5.0_f32; 100];
        assert!(discharge_gini(&uniform).abs() < 1e-9);
        let mut concentrated = vec![0.0_f32; 100];
        concentrated[0] = 42.0;
        assert!(discharge_gini(&concentrated) > 0.95);
        assert!(discharge_gini(&[]).abs() < 1e-9);
        assert!(discharge_gini(&[0.0, 0.0]).abs() < 1e-9);
    }

    #[test]
    fn closed_depressions_found_on_crafted_pit() {
        let mut grid = HexGrid::from_radius(2);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                c.elevation = 100.0;
            }
        }
        grid.get_mut(HexCoord::new(1, 0)).unwrap().elevation = 50.0;
        let pits = closed_depression_indices(&grid);
        assert_eq!(pits.len(), 1);
        assert_eq!(pits[0], grid.cell_index(HexCoord::new(1, 0)).unwrap());
    }

    // --- One-shot worldgen (#105): flow accumulation + erode_terrain ---

    #[test]
    fn accumulate_flow_grows_downstream() {
        // East→west ramp: flow accumulates downstream (west). The lowest
        // cell drains the whole world → maximal discharge; a ridge cell
        // (upstream) only has its own rain.
        let grid = build_ramp_grid();
        let (discharge, edge) = accumulate_flow(&grid, WORLDGEN_FLOW_CONCENTRATION);
        // Max discharge is far greater than the unit rain (1.0): the
        // accumulation has converged to an outlet.
        let max_d = discharge.iter().copied().fold(0.0_f32, f32::max);
        assert!(max_d > 5.0, "accumulation didn't converge: max {max_d}");
        // Local conservation: every cell forwards all of its input
        // (discharge) except sinks, so Σ(edge_flux[i]) == discharge[i] for
        // any cell with a downstream, and ≤ for a sink (nothing leaves).
        for i in 0..grid.len() {
            let out: f32 = edge[i].iter().sum();
            assert!(
                out <= discharge[i] + 1e-3,
                "cell {i} exports more than its discharge: {out} > {}",
                discharge[i]
            );
        }
    }

    #[test]
    fn erode_terrain_incises_channels_not_smooths() {
        // Terrain with relief: one-shot erosion must CONCENTRATE drainage
        // (accumulation gini ↑), not smooth it. Noisy radius-3 cone.
        let mut grid = HexGrid::from_radius(3);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                let d = f32::from(u8::try_from(coord.distance(HexCoord::new(0, 0))).unwrap());
                // Cone + deterministic ripple to create channels.
                let k = f32::from(i16::try_from(2 * coord.q + coord.r).unwrap());
                c.elevation = 300.0 - 40.0 * d + 12.0 * (k * 0.9).sin();
                c.water_capacity = 0.0;
            }
        }
        let gini_before = discharge_gini(&accumulate_flow(&grid, WORLDGEN_FLOW_CONCENTRATION).0);
        let params = ErosionParams::for_worldgen();
        erode_terrain(&mut grid, &params, 40);
        let gini_after = discharge_gini(&accumulate_flow(&grid, WORLDGEN_FLOW_CONCENTRATION).0);
        assert!(
            gini_after >= gini_before,
            "one-shot erosion de-concentrated drainage (smoothing): {gini_before:.4} → {gini_after:.4}"
        );
        // Incision only = rock never rises back up (no deposit); a cell
        // can only go down or stay put.
        assert!(grid.cells_slice().iter().all(|c| c.sediment_load == 0.0));
    }

    #[test]
    fn erode_terrain_zero_iterations_is_noop() {
        let mut a = build_ramp_grid();
        let b = a.clone();
        erode_terrain(&mut a, &ErosionParams::for_worldgen(), 0);
        for (x, y) in a.cells_slice().iter().zip(b.cells_slice()) {
            assert_eq!(x.elevation.to_bits(), y.elevation.to_bits());
        }
    }

    #[test]
    fn erode_terrain_is_deterministic() {
        let params = ErosionParams::for_worldgen();
        let mut a = build_ramp_grid();
        let mut b = build_ramp_grid();
        erode_terrain(&mut a, &params, 15);
        erode_terrain(&mut b, &params, 15);
        for (x, y) in a.cells_slice().iter().zip(b.cells_slice()) {
            assert_eq!(x.elevation.to_bits(), y.elevation.to_bits());
        }
    }

    proptest! {
        /// Terrarium invariant: Σ(elevation) + Σ(sediment_load) is conserved
        /// by an erosion step, regardless of relief, load, and forcing. If
        /// this proptest fails, erosion is creating or destroying rock.
        #[test]
        fn prop_rock_mass_is_conserved(
            seed_elev in 0.0_f32..500.0,
            load in 0.0_f32..10.0,
            q in 0.0_f32..200.0,
            accel in 1.0_f32..50_000.0,
        ) {
            let mut grid = HexGrid::from_radius(2);
            for (k, coord) in grid.coords().copied().collect::<Vec<_>>().into_iter().enumerate() {
                if let Some(c) = grid.get_mut(coord) {
                    // Deterministic pseudo-random relief, bounded.
                    let k_f = f32::from(u8::try_from(k % 97).expect("k%97 < 256"));
                    c.elevation = seed_elev + 37.0 * (k_f * 0.61).sin();
                    c.sediment_load = load * (0.5 + 0.5 * (k_f * 1.7).cos());
                    c.water_level = 0.0;
                    c.water_capacity = 0.0;
                }
            }
            let n = grid.len();
            let (discharge, edge) = {
                let mut d = vec![0.0_f32; n];
                let mut e = vec![[0.0_f32; 6]; n];
                for i in 0..n {
                    let i_f = f32::from(u8::try_from(i % 89).expect("i%89 < 256"));
                    d[i] = q * (0.5 + 0.5 * (i_f * 0.83).sin());
                    // Edge flux split over 2 arbitrary directions.
                    e[i][usize::from(i % 6 == 0)] = d[i] * 0.7;
                    e[i][3] = d[i] * 0.3;
                }
                (d, e)
            };
            let params = ErosionParams { accel_years_per_day: accel, ..ErosionParams::default() };
            let before = total_rock(&grid);
            let mut next = grid.clone();
            let totals = step_erosion(
                &grid,
                &mut next,
                &params,
                &ErosionForcing { discharge_ema: &discharge, edge_flux_ema: &edge },
            );
            let after = total_rock(&next);
            let scale = before.abs().max(1.0);
            prop_assert!(
                ((after - before) / scale).abs() < 1e-6,
                "rock conservation violated: {before} → {after} (incised {}, deposited {})",
                totals.incised_m, totals.deposited_m
            );
            // And never a significantly negative load.
            for c in next.cells_slice() {
                prop_assert!(c.sediment_load > -1e-4, "negative load: {}", c.sediment_load);
            }
        }
    }
}
