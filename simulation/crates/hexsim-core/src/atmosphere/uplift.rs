use serde::Serialize;

use crate::coord::opposite_direction;
use crate::grid::HexGrid;
use crate::par::{for_each_chunk_mut, for_each_chunk_mut2, for_each_chunk_mut6, sum_dir_out};
use crate::physics::meyer_evaporation;
use crate::temperature::{TemperatureParams, local_t_ref};
use crate::time::TICKS_PER_DAY_F32;
use crate::vegetation;
use crate::wind::wind_magnitude_to_meters_per_second;

use super::{AtmoScratch, AtmosphereParams, SOIL_GW_REFERENCE_MM, saturation_upper};

/// Open-water evaporation stats for the tick, aggregated by `step_evaporation`
/// over the exact cells and formula it uses to move water: no separate
/// recomputation exists anywhere else (a diagnostics-side clone of this
/// formula used to drift from it silently, see #29/#89 history). `mean_mm_day`
/// etc. are the physical rate (Dalton/Meyer ET₀), comparable to observed
/// open-water evaporation (temperate ≈ 2-4 mm/day, warm windy > 8 mm/day),
/// not the per-tick mass actually transferred (which is this rate divided by
/// `TICKS_PER_DAY` and capped to the available surplus).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct EvapStats {
    /// Average over the cells taken into account (true open water: surplus
    /// above `water_capacity`, thawed).
    pub mean_mm_day: f32,
    pub min_mm_day: f32,
    pub max_mm_day: f32,
    /// Number of cells taken into account. 0 when no cell has open water,
    /// in which case every field above is 0 (never a NaN from an empty mean).
    pub cell_count: usize,
}

/// Vapour one cell put into `humidity_surface` during one tick, by source
/// (mm of water). The exact masses `step_evaporation` moved, read by the
/// water-cycle instruments instead of being recomputed there (anti-pattern
/// #2): where the box's water leaves the ground, lakes against land.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct VaporSources {
    /// Free-water evaporation from the open surplus of a water body.
    pub open_water: f32,
    /// Plant transpiration, drawn from the root zone.
    pub transpiration: f32,
    /// Sublimation of the snowpack and lake ice.
    pub sublimation: f32,
}

impl VaporSources {
    #[must_use]
    pub fn total(self) -> f32 {
        self.open_water + self.transpiration + self.sublimation
    }

    pub fn add(&mut self, other: Self) {
        self.open_water += other.open_water;
        self.transpiration += other.transpiration;
        self.sublimation += other.sublimation;
    }
}

/// Per-cell output of the parallel pass of [`step_evaporation`]: the
/// open-water evaporative demand that `EvapStats` reduces (mm/day, a
/// negative sentinel for a cell not counted as open water), and the
/// vapour actually moved this tick by source.
#[derive(Debug, Clone, Copy)]
pub struct EvapCell {
    pub demand_mm_day: f32,
    pub vapor: VaporSources,
}

impl Default for EvapCell {
    fn default() -> Self {
        Self {
            demand_mm_day: -1.0,
            vapor: VaporSources::default(),
        }
    }
}

/// Evaporation of liquid water + plant transpiration + snow sublimation.
/// Feeds `humidity_surface`: freshly evaporated vapor must be lifted by
/// uplift before it can precipitate. This temporal separation avoids the
/// captive lake → rain → lake cycle.
///
/// Phase 2 (#31): free-water evaporation goes through Dalton's law
/// (Meyer 1915) via `meyer_evaporation`. Phase 3 (#32): `humidity_surface`
/// is now directly in mm, so Meyer's mm/day output feeds the stock after
/// division by `TICKS_PER_DAY` (Tier 1, v0.3.0). #77: plant transpiration
/// (FAO-56, cf `transpiration_coef`) reuses this same Meyer demand as ET₀
/// and closes the vegetation → atmosphere loop.
/// Snow sublimation remains phenomenological.
///
/// `transpiration_cover` is the per-cell light-weighted cover
/// (`vegetation::transpiration_cover`) memoized by the caller at the
/// cadence of the vegetation (daily); `None` computes it per cell here.
///
/// `out` reports open-water evaporation stats (`EvapStats`) for whoever
/// needs to display them (diagnostics): filled from the same cells, same
/// gate (`open_water > 0`, i.e. a real lake, not any puddle under
/// capacity) and same memoized wind (`wind_mag`) that drive the flux
/// actually applied below. This is now the sole computation of open-water
/// evaporation; nothing else may recompute it (anti-pattern #2).
///
/// Split in two (r250 perf effort): a parallel per-cell pass (no neighbor
/// reads, purely `current[i] -> next[i]`) writes each open-water cell's
/// evaporative demand into `demand` (a negative sentinel for a cell not
/// counted — dry, under capacity, or frozen), then a serial reduction
/// over `demand` in index order folds it into `out`, in the exact order
/// the historical single loop did: bit-identical, not merely within
/// tolerance.
pub fn step_evaporation(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    wind_mag: &[f32],
    transpiration_cover: Option<&[f32]>,
    cells_out: &mut Vec<EvapCell>,
    out: &mut EvapStats,
) {
    let n = current.len();
    cells_out.resize(n, EvapCell::default());
    let cur = current.cells_slice();

    // Folds the historical `current → next` full-grid copy (formerly done
    // by the caller, `step_atmosphere_into`) into this same sweep (r250
    // perf effort, chunk B2): this is the phase's first per-cell pass,
    // it reads only `current[i]` and read-only tick inputs, and writes
    // only `next[i]` — starting each cell from `*next_cell = cell.clone()`
    // before applying evaporation/transpiration/sublimation is
    // bit-identical to the separate copy, one fewer 88-byte full-grid
    // stream per tick.
    for_each_chunk_mut2(
        next.cells_slice_mut(),
        cells_out,
        |start, next_chunk, out_chunk| {
            for (local, next_cell) in next_chunk.iter_mut().enumerate() {
                let i = start + local;
                let cell = &cur[i];
                *next_cell = cell.clone();
                // Reference evaporative demand ET₀ (Dalton/Meyer), in mm/day.
                // Shared by free-water evaporation and plant transpiration: same
                // vapor transfer physics, modulated differently downstream.
                // Magnitude precomputed at the cadence of the wind field (#89):
                // the field only changes one hour out of N, the per-cell-hour
                // sqrt was pure recomputation.
                let wind_ms =
                    wind_magnitude_to_meters_per_second(wind_mag.get(i).copied().unwrap_or(0.0));
                let cap = saturation_upper(cell.temperature, params).max(1e-6);
                let rh = (cell.humidity_surface / cap).clamp(0.0, 1.0);
                let evap_demand_per_day = if cell.temperature >= 0.0 {
                    meyer_evaporation(cell.temperature, cell.temperature, rh, wind_ms).0
                } else {
                    0.0
                };

                // Free-water evaporation (lakes): the demand applies to the
                // open surface, drawn from `water_level`.
                let open_water = (cell.water_level - cell.water_capacity).max(0.0);
                let mut record = EvapCell::default();
                if open_water > 0.0 && cell.temperature >= 0.0 {
                    let evap = (evap_demand_per_day / TICKS_PER_DAY_F32).min(open_water);
                    next_cell.water_level -= evap;
                    next_cell.humidity_surface += evap;
                    record.demand_mm_day = evap_demand_per_day;
                    record.vapor.open_water = evap;
                }

                // Plant transpiration (#77, replaces the `ground_evap_rate`
                // proxy). FAO-56: `ET = Kc × ET₀ × water_stress`, with
                // Kc = Kc_max × biomass (biomass [0,1] proxies LAI / cover
                // fraction) and water stress = groundwater saturation. Water
                // is drawn *from* `groundwater` and returned to
                // `humidity_surface`: strict conservation (uptake =
                // transpiration), no double counting.
                // Cover weighted by each species' crop coefficient (FAO-56,
                // #83) and by the light its stratum receives (#161):
                // Σ crop_coef_i × biomass_i × transmittance(stratum_i). A
                // forest transpires more than a lawn at equal cover. The
                // light factor comes from Penman-Monteith: transpiration is
                // driven by the radiation the leaves absorb, so a layer
                // under a canopy that intercepts ~90 % of the shortwave
                // transpires ~10 % of what it would in the open. That keeps
                // Kc bounded by the light budget of the column, even with
                // three full strata (Σ biomass up to 3), without any clamp
                // (anti-pattern #4): each layer below only works with what
                // the layers above let through (Beer-Lambert, Monsi & Saeki
                // 1953). The sum lives in `vegetation::transpiration_cover`;
                // the caller hands it memoized once a day (`Simulation`,
                // vegetation only changes in the daily tail) or `None` to
                // have it computed here, identically.
                let weighted_cover = transpiration_cover
                    .map_or_else(|| vegetation::transpiration_cover(cell), |memo| memo[i]);
                if weighted_cover > 0.0
                    && params.transpiration_coef > 0.0
                    && cell.temperature >= 0.0
                {
                    let gw_capacity = (cell.permeability * SOIL_GW_REFERENCE_MM).max(1e-6);
                    let water_stress = (cell.groundwater / gw_capacity).clamp(0.0, 1.0);
                    let kc = params.transpiration_coef * weighted_cover;
                    let transp_per_day = evap_demand_per_day * kc * water_stress;
                    let transp = (transp_per_day / TICKS_PER_DAY_F32).min(cell.groundwater);
                    next_cell.groundwater -= transp;
                    next_cell.humidity_surface += transp;
                    record.vapor.transpiration = transp;
                }
                if cell.frozen_surface() > 0.0 && cell.temperature < 0.0 {
                    let cold_factor = (-cell.temperature / 10.0).clamp(0.0, 1.0);
                    let sublim = (params.sublimation_rate * cold_factor).min(cell.frozen_surface());
                    // Exact transfer (cf. `snow::step_snow`): credit humidity
                    // with what ACTUALLY left the frozen stocks (snowpack
                    // first, then lake ice). On a glacier several meters
                    // deep, f32 ULP (~0.5 mm at 4 m) means the rounded
                    // decrement differs from the computed `sublim`; the gap,
                    // constant and sign-biased, leaked into the strict
                    // conservation test (#60 Phase 3, diagnosed via knockout).
                    let departed = next_cell.take_frozen(sublim);
                    next_cell.humidity_surface += departed;
                    record.vapor.sublimation = departed;
                }
                out_chunk[local] = record;
            }
        },
    );

    let mut evap_sum = 0.0_f32;
    let mut evap_min = f32::INFINITY;
    let mut evap_max = 0.0_f32;
    let mut evap_count = 0_usize;
    for d in cells_out.iter().map(|c| c.demand_mm_day) {
        if d >= 0.0 {
            evap_sum += d;
            evap_min = evap_min.min(d);
            evap_max = evap_max.max(d);
            evap_count += 1;
        }
    }

    *out = if evap_count == 0 {
        EvapStats::default()
    } else {
        let nf = f32::from(u16::try_from(evap_count).unwrap_or(u16::MAX));
        EvapStats {
            mean_mm_day: evap_sum / nf,
            min_mm_day: evap_min,
            max_mm_day: evap_max,
            cell_count: evap_count,
        }
    };
}

/// Intra-cell vertical uplift: transfers a fraction of `humidity_surface`
/// to `humidity_upper`, modulated by thermal convection, orographic
/// uplift, and (issue #46) diurnal convective drive. Vertical transport
/// stays within the same column.
///
/// Diurnal drive (#46): `(T - t_ref).max(0) × sin_elev_pos × coef`. Strong
/// in the afternoon over dry summer plains, zero at night (`sin_elev` =
/// 0). Creates the diurnal cumulus that struggled to emerge under the
/// static regime.
pub(crate) fn step_uplift(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    temp_params: &TemperatureParams,
    sin_elev_pos: f32,
) {
    let lat_rad = temp_params.latitude_deg.to_radians();
    // Purely local transform of each cell (no neighbor read, no other
    // input slice): a pure per-cell map, parallelizable
    // (`par::for_each_chunk_mut`).
    for_each_chunk_mut(next.cells_slice_mut(), |_start, chunk| {
        for cell in chunk {
            if cell.humidity_surface <= 0.0 {
                continue;
            }
            let temp_boost = cell.temperature.max(0.0) * params.uplift_thermal_coef;
            // Diurnal drive: active only when the sun is above the horizon. At
            // T_excess=25 K and sin_elev=0.9, contribution ≈ 25 × 0.9 ×
            // convective_diurnal_coef. No extra cap here; the final clamp to
            // 0.9 on `rate` guarantees stability.
            let diurnal_drive = if sin_elev_pos > 0.0 && params.convective_diurnal_coef > 0.0 {
                let t_ref = local_t_ref(cell.elevation, cell.water_level, temp_params, lat_rad);
                let t_excess = (cell.temperature - t_ref).max(0.0);
                t_excess * sin_elev_pos * params.convective_diurnal_coef
            } else {
                0.0
            };
            let rate = (params.uplift_rate + temp_boost + diurnal_drive).clamp(0.0, 0.9);
            let transfer = cell.humidity_surface * rate;
            cell.humidity_surface -= transfer;
            cell.humidity_upper += transfer;
        }
    });
}

/// Orographic convection: independent of wind, each cell sends a
/// fraction of `humidity_surface` to the `humidity_upper` of its
/// higher-altitude neighbors. Physically: vapor-laden air near the
/// ground is unstable close to relief (sharp vertical thermal
/// contrasts) → turbulent rise along the slope. Necessary in a closed
/// terrarium where thermal breezes would otherwise flow down relief
/// instead of up it. The split between neighbors is proportional to
/// the positive elevation difference.
///
/// Two-phase scatter → gather (r250 perf effort): phase 1 computes, per
/// SOURCE cell and independent of every other source (the LCL bound only
/// ever reads the pre-tick snapshots `src_upper`/`sat_upper_offset`,
/// never another source's contribution), the amount pumped toward each
/// of its 6 neighbors into `scratch.dir_out` (`for_each_chunk_mut6`,
/// parallel). Phase 2 gathers it back per DESTINATION cell via
/// `coord::opposite_direction`, deriving both self-loss terms
/// (`delta_surface`, `delta_upper`) from `Σ_d dir_out[d][i]` — the total
/// this cell itself pumped out — split between the surface/upper pools
/// in the same ratio the pre-scale flux used
/// (`p_surf = surf / (surf + upper·0.25)`, independent of `scale`, so
/// exact): no separate self-term buffer is needed. The apply pass is
/// unchanged.
///
pub(crate) fn step_orographic_convection(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    scratch: &mut AtmoScratch,
) {
    if params.orographic_lift_coef <= 0.0 {
        return;
    }
    let n = current.len();
    let next_cells = next.cells_slice_mut();
    let cur_cells = current.cells_slice();

    // Snapshot of the fields before deltas. Deltas are computed in pure
    // read mode and applied in pass 2: no write contention possible.
    // Scratch buffers reused (#88/#65): clear() keeps the capacity, so
    // extend() never reallocates after the first tick.
    let src_surface = &mut scratch.oro_src_surface;
    let src_upper = &mut scratch.oro_src_upper;
    let elev_snap = &mut scratch.oro_elev;
    src_surface.clear();
    src_upper.clear();
    elev_snap.clear();
    for i in 0..n {
        src_surface.push(next_cells[i].humidity_surface);
        src_upper.push(next_cells[i].humidity_upper);
        elev_snap.push(cur_cells[i].elevation);
    }

    // Phase 1 (outflow): per source cell, the amount pumped toward each
    // of its 6 neighbors, 0 for a non-higher direction. Reads only the
    // pre-tick snapshots above and `sat_upper_offset` (#97): fully
    // independent per source.
    for dir in &mut scratch.dir_out {
        dir.clear();
        dir.resize(n, 0.0);
    }
    fill_oro_outflow(
        &OroForcing {
            grid: current,
            src_surface,
            src_upper,
            elev_snap,
            sat_upper_offset: &scratch.sat_upper_offset,
            coef: params.orographic_lift_coef,
        },
        &mut scratch.dir_out,
    );

    // Phase 2 (gather): per destination cell, the surface/upper deltas.
    scratch.oro_delta_surface.clear();
    scratch.oro_delta_surface.resize(n, 0.0);
    scratch.oro_delta_upper.clear();
    scratch.oro_delta_upper.resize(n, 0.0);
    gather_oro_deltas(
        current,
        &scratch.dir_out,
        &scratch.oro_src_surface,
        &scratch.oro_src_upper,
        &mut scratch.oro_delta_surface,
        &mut scratch.oro_delta_upper,
    );

    // Apply pass: reads only `oro_delta_*[i]` (fully computed by the
    // gather phase above), writes only `next_cells[i]` — a pure per-cell
    // map, parallelizable (`par::for_each_chunk_mut`).
    let delta_surface = &scratch.oro_delta_surface;
    let delta_upper = &scratch.oro_delta_upper;
    for_each_chunk_mut(next_cells, |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            let i = start + local;
            let ds = delta_surface[i];
            let du = delta_upper[i];
            if ds == 0.0 && du == 0.0 {
                continue;
            }
            cell.humidity_surface = (cell.humidity_surface + ds).max(0.0);
            cell.humidity_upper = (cell.humidity_upper + du).max(0.0);
        }
    });
}

/// Fraction of a source cell's moisture the orographic pump exports in one
/// tick, given the relief that surrounds it.
///
/// # The law (#156)
///
/// Orographic ascent velocity, Smith (1979) linear mountain-wave theory,
/// reviewed by Roe (2005), *Orographic precipitation*, Annu. Rev. Earth
/// Planet. Sci. **33**, §2.1:
///
/// ```text
/// w_oro = U_slope · s⁺                         [m/s]
/// ```
///
/// with `s⁺ = Σ_j max(z_j − z_i, 0) / L` the dimensionless convergent
/// upslope gradient around the source (`L` = `dynamics::CELL_SPACING_M`)
/// and `U_slope` the horizontal speed feeding the slope.
///
/// Venting a moist layer of depth `H` at that velocity is a first-order
/// drain, `dq/dt = −q · w_oro / H`, whose closed form over a tick of `dt`
/// is
///
/// ```text
/// rate = 1 − exp(−dt/τ),   τ = H / w_oro       [s]
/// x = dt/τ = U_slope·dt/(H·L) · Σ Δz⁺          [-]
/// ```
///
/// `orographic_lift_coef`, already divided by `TICKS_PER_DAY` by
/// `scale_atmosphere_for_hourly_tick` when it gets here, IS that group
/// `U_slope·dt/(H·L)`, in **per metre of relief and per tick**. At the
/// shipped default (0.05 /day/m, `H` = `upper_layer_altitude_m` = 1500 m,
/// `L` = 130 m, `dt` = 3600 s) the implied `U_slope ≈ 0.11 m/s`, inside
/// the 1-10 cm/s range measured for thermally driven export of
/// boundary-layer air to the free troposphere over the Alps (Henne et al.
/// 2004, *Atmos. Chem. Phys.* **4**, 497-509, "Quantification of
/// topographic venting of boundary layer air to the free troposphere").
/// Pinned by `orographic_coef_implies_a_measured_venting_speed`.
///
/// # Why an exponential and not a cap
///
/// `1 − exp(−x)` is in `[0, 1]` **by construction**, so the only bound
/// left on the pump is the conservation one — a cell cannot export more
/// than the stock it holds — plus the LCL bound at the destination
/// (see [`fill_oro_outflow`]). No `clamp` is reachable in normal regime.
/// In `f32` the value rounds to exactly `1.0` past `x ≈ 16.6` (`exp(−x)`
/// drops below the ulp of 1); that is still the conservation bound and not
/// an overflow — [`fill_oro_outflow`]'s split gives a surface loss of
/// `total_out × surf/(surf + 0.25·upper) ≤ surf`, so the source empties
/// and never goes negative.
/// It is also the exact integral of the drain above, the same treatment
/// `condensation`/`regime` already give their relaxations, and it keeps
/// `d(rate)/d(coef) ≈ dt/τ` at small `x`: the coefficient still means
/// something on a gentle slope.
///
/// The law it replaces, `(coef · Σ Δz⁺).clamp(0.0, 0.30)`, was a stability
/// guard that had become the physics. Since `CELL_SPACING_M` went to 130 m
/// (d6be105) the cap binds at `Σ Δz⁺ = 144 m`, about two ordinary 30°
/// neighbours, so 64-78 % of the pumping cell-passes above 300 m ran at
/// exactly 0.30 whatever `orographic_lift_coef` was — anti-pattern 4,
/// measured JOURNAL 2026-09-05, quantified by `diag_oro_pump_saturation`,
/// tracked as issue #156. It was kept reachable bit for bit as an A/B
/// lever (`HEXSIM_ORO_LEGACY_CLAMP`) until this law had been green and
/// merged long enough (since 2026-09-05) to retire it, 2026-09-07.
#[must_use]
fn oro_pump_rate(coef: f32, total_positive_delta_m: f32) -> f32 {
    let x = coef * total_positive_delta_m;
    1.0 - (-x).exp()
}

/// Read-only tick inputs of [`fill_oro_outflow`], grouped per convention
/// #61: the pass already sat at the 7-argument ceiling.
struct OroForcing<'a> {
    grid: &'a HexGrid,
    src_surface: &'a [f32],
    src_upper: &'a [f32],
    elev_snap: &'a [f32],
    sat_upper_offset: &'a [f32],
    /// `orographic_lift_coef`, already scaled to the tick.
    coef: f32,
}

/// Phase 1 of [`step_orographic_convection`]: per source cell, the
/// amount pumped toward each of its 6 neighbors (0 for a non-higher
/// direction). Reads only the pre-tick snapshots and `sat_upper_offset`
/// (#97): fully independent per source, parallelizable
/// (`par::for_each_chunk_mut6`).
fn fill_oro_outflow(forcing: &OroForcing, dir_out: &mut [Vec<f32>; 6]) {
    let &OroForcing {
        grid,
        src_surface,
        src_upper,
        elev_snap,
        sat_upper_offset,
        coef,
    } = forcing;
    for_each_chunk_mut6(dir_out, |start, chunks| {
        for local in 0..chunks[0].len() {
            let i = start + local;
            for chunk in chunks.iter_mut() {
                chunk[local] = 0.0;
            }
            let src_elev = elev_snap[i];
            let surf = src_surface[i];
            let upper = src_upper[i];
            if surf <= 0.0 && upper <= 0.0 {
                continue;
            }
            // Toric neighborhood: orographic uplift also sees the relief
            // on the other side of the seam (periodic terrain). Without
            // this, edge cells had a truncated neighborhood → ring bias.
            let neighbors = grid.neighbor_indices_toric(i);
            let mut total_positive_delta = 0.0_f32;
            for j in neighbors {
                let n_elev = elev_snap[j];
                if n_elev > src_elev {
                    total_positive_delta += n_elev - src_elev;
                }
            }
            if total_positive_delta < 1e-6 {
                continue;
            }
            // Exported fraction of the surface layer, see
            // [`oro_pump_rate`]: `1 − exp(−coef · Σ Δz⁺)`, in [0, 1] by
            // construction. The only bounds on this pump are conservation
            // (here) and the LCL deficit at the destination (below).
            let rate = oro_pump_rate(coef, total_positive_delta);
            // Lift surface -> upper(higher neighbor): main flux.
            let lift_surface_brut = surf * rate;
            // Pump upper -> upper(higher neighbor): 0.25x, secondary
            // suction, much weaker so as not to dry out `humidity_upper`
            // in the lowlands.
            let lift_upper_brut = upper * rate * 0.25;
            let total_in_brut = lift_surface_brut + lift_upper_brut;
            if total_in_brut < 1e-9 {
                continue;
            }

            // LCL bound (#63 Phase 4 Step 3): orographic transport
            // toward a higher neighbor cannot exceed the saturation
            // deficit of `humidity_upper` at the destination. An air
            // parcel rising adiabatically precipitates locally at the
            // LCL: it cannot carry more than `sat_upper - hu_dest`.
            // Without this bound the pump injected HR_upper p99 =
            // 16-288 onto the peaks (cf JOURNAL pivot #63 Phase 4
            // Step 3).
            //
            // Conservative algorithm: find the attenuation factor
            // `scale ≤ 1` such that for EACH higher neighbor j,
            // `total_in × share_j ≤ deficit_j`. The surplus stays at the
            // source (the physical mechanism = compensating downdraft).
            let mut max_total_in_allowed = f32::INFINITY;
            for j in neighbors {
                let n_elev = elev_snap[j];
                if n_elev <= src_elev {
                    continue;
                }
                let share_j = (n_elev - src_elev) / total_positive_delta;
                if share_j < 1e-9 {
                    continue;
                }
                let sat_j = sat_upper_offset[j];
                let deficit_j = (sat_j - src_upper[j]).max(0.0);
                let limit = deficit_j / share_j;
                if limit < max_total_in_allowed {
                    max_total_in_allowed = limit;
                }
            }
            let scale = (max_total_in_allowed / total_in_brut).clamp(0.0, 1.0);
            let total_in = total_in_brut * scale;

            for (d, &j) in neighbors.iter().enumerate() {
                let n_elev = elev_snap[j];
                if n_elev > src_elev {
                    let share = (n_elev - src_elev) / total_positive_delta;
                    chunks[d][local] = total_in * share;
                }
            }
        }
    });
}

/// Phase 2 of [`step_orographic_convection`]: per destination cell, the
/// surface/upper deltas. `Σ_d dir_out[d][j]` is exactly what `j` itself
/// pumped out in phase 1 (whatever it lost to distribute among ITS
/// higher neighbors); splitting it back into a surface share and an
/// upper share uses the same ratio the pre-scale flux used
/// (`lift_surface_brut : lift_upper_brut`, independent of `scale` since
/// both are scaled identically), so this is exact, not an approximation.
fn gather_oro_deltas(
    current: &HexGrid,
    dir_out: &[Vec<f32>; 6],
    src_surface: &[f32],
    src_upper: &[f32],
    delta_surface: &mut [f32],
    delta_upper: &mut [f32],
) {
    for_each_chunk_mut2(delta_surface, delta_upper, |start, ds_chunk, du_chunk| {
        for local in 0..ds_chunk.len() {
            let j = start + local;
            let total_out = sum_dir_out(dir_out, j);
            let denom = src_surface[j] + src_upper[j] * 0.25;
            let p_surf = if denom > 0.0 {
                src_surface[j] / denom
            } else {
                0.0
            };
            let self_surface_loss = total_out * p_surf;
            let self_upper_loss = total_out - self_surface_loss;

            let neighbors = current.neighbor_indices_toric(j);
            let mut gathered_in = 0.0_f32;
            for (d, &k) in neighbors.iter().enumerate() {
                gathered_in += dir_out[opposite_direction(d)][k];
            }

            ds_chunk[local] = -self_surface_loss;
            du_chunk[local] = gathered_in - self_upper_loss;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::scaling::{oro_boosted_params, scale_atmosphere_for_hourly_tick};
    use crate::atmosphere::test_support::{assert_lcl_slack, default_temp_params, oro_pump_world};
    use crate::atmosphere::{surface_means, total_humidity, upper_air_temperature};
    use crate::coord::DIRECTIONS;
    use crate::coord::HexCoord;
    use crate::dynamics::CELL_SPACING_M;
    use crate::temperature::SECONDS_PER_HOUR;
    use crate::units::MetersPerSecond;

    #[test]
    fn step_evaporation_stats_count_only_true_open_water() {
        // Pins the bug fix: the old diagnostics-side observer gated on
        // `water_level > 0` (any surface water, including a puddle under
        // capacity), while the engine only ever evaporates the surplus
        // above `water_capacity` (a real lake). `EvapStats` must reflect
        // exactly the cells and formula the engine itself uses to move
        // water, no separate recomputation. Evaporation is cell-local
        // (no neighbor reads), so radius 0-transport caveats don't apply.
        let mut current = HexGrid::from_radius(1);
        let coords: Vec<HexCoord> = current.coords().copied().collect();

        // Lake: surplus above capacity, thawed -> counted.
        if let Some(c) = current.get_mut(coords[0]) {
            c.water_capacity = 1.0;
            c.water_level = 5.0;
            c.temperature = 20.0;
            c.humidity_surface = 5.0;
        }
        // Puddle: water_level > 0 but under capacity -> NOT open water,
        // excluded (this is exactly what the old buggy gate got wrong).
        if let Some(c) = current.get_mut(coords[1]) {
            c.water_capacity = 10.0;
            c.water_level = 3.0;
            c.temperature = 20.0;
        }
        // Frozen lake: surplus above capacity but T < 0 -> excluded.
        if let Some(c) = current.get_mut(coords[2]) {
            c.water_capacity = 1.0;
            c.water_level = 5.0;
            c.temperature = -5.0;
        }
        // Remaining cells stay at the default (water_level=0 < capacity=1):
        // dry, excluded.

        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let wind_mag = vec![0.0_f32; current.len()];
        let mut demand = Vec::new();
        let mut stats = EvapStats::default();

        step_evaporation(
            &current,
            &mut next,
            &params,
            &wind_mag,
            None,
            &mut demand,
            &mut stats,
        );

        assert_eq!(
            stats.cell_count, 1,
            "only the true lake cell (surplus above capacity, thawed) counts as open water"
        );
        // Ground truth computed independently from the same physical
        // formula step_evaporation applies (Dalton/Meyer), not by calling
        // back into the engine: this is what "the flux the engine applied"
        // means for that single cell.
        let cap = saturation_upper(20.0, &params).max(1e-6);
        let rh = (5.0_f32 / cap).clamp(0.0, 1.0);
        let expected = meyer_evaporation(20.0, 20.0, rh, MetersPerSecond(0.0)).0;
        assert!(
            (stats.mean_mm_day - expected).abs() < 1e-4,
            "mean={} expected={expected}",
            stats.mean_mm_day
        );
        assert!((stats.min_mm_day - expected).abs() < 1e-4);
        assert!((stats.max_mm_day - expected).abs() < 1e-4);
    }

    #[test]
    fn step_evaporation_stats_empty_when_no_open_water() {
        // No cell has open water: EvapStats must fall back to all-zero
        // fields (same convention the removed diagnostics-side observer
        // used), never a NaN from dividing by zero cells.
        let current = HexGrid::from_radius(0);
        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let wind_mag = vec![0.0_f32; current.len()];
        let mut demand = Vec::new();
        let mut stats = EvapStats::default();

        step_evaporation(
            &current,
            &mut next,
            &params,
            &wind_mag,
            None,
            &mut demand,
            &mut stats,
        );

        assert_eq!(stats.cell_count, 0);
        assert!(stats.mean_mm_day.abs() < 1e-6);
        assert!(stats.min_mm_day.abs() < 1e-6);
        assert!(stats.max_mm_day.abs() < 1e-6);
    }

    /// Hourly transpiration (mm drawn from the water table) of each cell of
    /// a radius-1 grid holding `stands` (by id), under the same weather:
    /// 20 °C, dry air, calm, water table at twice its capacity (no water
    /// stress, no availability cap). Evaporation is cell-local, so each
    /// cell is an independent sample.
    fn transpiration_of(stands: &[&[(crate::species::SpeciesId, f32)]]) -> Vec<f32> {
        let mut current = HexGrid::from_radius(1);
        let coords: Vec<HexCoord> = current.coords().copied().collect();
        assert!(stands.len() <= coords.len());
        for (coord, stand) in coords.iter().zip(stands.iter()) {
            let c = current.get_mut(*coord).unwrap();
            c.water_capacity = 10.0;
            c.water_level = 0.0;
            c.temperature = 20.0;
            c.humidity_surface = 2.0;
            c.permeability = 0.5;
            c.groundwater = 2.0 * 0.5 * SOIL_GW_REFERENCE_MM;
            for &(id, v) in *stand {
                c.vegetation[crate::species::species_index(id)] = v;
            }
        }
        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let wind_mag = vec![0.0_f32; current.len()];
        let mut demand = Vec::new();
        let mut stats = EvapStats::default();
        step_evaporation(
            &current,
            &mut next,
            &params,
            &wind_mag,
            None,
            &mut demand,
            &mut stats,
        );
        coords
            .iter()
            .take(stands.len())
            .map(|c| current.get(*c).unwrap().groundwater - next.get(*c).unwrap().groundwater)
            .collect()
    }

    /// The vapour reported per source is the exact mass each stock lost:
    /// the lake's surplus for open water, the root zone for transpiration,
    /// the snowpack for sublimation, and nothing for a dry bare cell.
    /// Cell-local, no transport (radius 1 only to hold four samples).
    #[test]
    fn vapor_sources_are_the_masses_moved() {
        let mut current = HexGrid::from_radius(1);
        let coords: Vec<HexCoord> = current.coords().copied().collect();
        let (lake, forest, snowfield, bare) = (coords[0], coords[1], coords[2], coords[3]);
        for c in current.cells_slice_mut() {
            c.water_capacity = 1.0;
            c.temperature = 20.0;
            c.humidity_surface = 2.0;
            c.permeability = 0.5;
        }
        current.get_mut(lake).unwrap().water_level = 50.0;
        let f = current.get_mut(forest).unwrap();
        f.groundwater = 30.0;
        f.vegetation[crate::species::species_index(crate::species::SpeciesId::Beech)] = 0.9;
        let s = current.get_mut(snowfield).unwrap();
        s.temperature = -8.0;
        s.snow_level = 40.0;

        let mut next = current.clone();
        let params = AtmosphereParams::default();
        let wind_mag = vec![3.0_f32; current.len()];
        let mut cells = Vec::new();
        let mut stats = EvapStats::default();
        step_evaporation(
            &current, &mut next, &params, &wind_mag, None, &mut cells, &mut stats,
        );

        let index = |c: HexCoord| current.cell_index(c).unwrap();
        let before = |c: HexCoord| current.get(c).unwrap().clone();
        let after = |c: HexCoord| next.get(c).unwrap().clone();

        let v = cells[index(lake)].vapor;
        assert!(v.open_water > 0.0, "a warm lake evaporates");
        assert!((v.open_water - (before(lake).water_level - after(lake).water_level)).abs() < 1e-6);

        let v = cells[index(forest)].vapor;
        assert!(v.transpiration > 0.0, "a watered forest transpires");
        assert!(
            (v.transpiration - (before(forest).groundwater - after(forest).groundwater)).abs()
                < 1e-6
        );

        let v = cells[index(snowfield)].vapor;
        assert!(v.sublimation > 0.0, "a cold snowpack sublimates");
        assert!(
            (v.sublimation - (before(snowfield).snow_level - after(snowfield).snow_level)).abs()
                < 1e-6
        );

        assert!(
            cells[index(bare)].vapor.total().abs() < 1e-12,
            "a dry bare cell emits nothing"
        );
        for i in [lake, forest, snowfield, bare] {
            let gained = after(i).humidity_surface - before(i).humidity_surface;
            assert!(
                (gained - cells[index(i)].vapor.total()).abs() < 1e-6,
                "the sources add up to the vapour the cell gained"
            );
        }
    }

    #[test]
    fn phys_shaded_understory_transpires_less_than_in_the_open() {
        // #161: at equal herb cover, the meadow's own transpiration under a
        // closed beech canopy (0.95, LAI 4.75) is the Beer-Lambert share of
        // the light it receives, exp(−0.5 × 4.75) ≈ 9 %, of what it
        // transpires in the open (Penman-Monteith: transpiration follows
        // absorbed radiation). The herb contribution is read as a
        // difference, canopy + meadow − canopy alone.
        use crate::species::{SpeciesId, Stratum};
        use crate::vegetation::light_transmittance_below;
        let meadow = (SpeciesId::Meadow, 0.8);
        let beech = (SpeciesId::Beech, 0.95);
        let t = transpiration_of(&[&[], &[meadow], &[beech], &[beech, meadow]]);
        let (bare, open_meadow, canopy, canopy_meadow) = (t[0], t[1], t[2], t[3]);
        assert!(bare.abs() < 1e-9, "bare soil transpires nothing: {bare}");
        assert!(open_meadow > 0.0 && canopy > 0.0);
        let herb_open = open_meadow - bare;
        let herb_shaded = canopy_meadow - canopy;
        assert!(
            herb_shaded < herb_open,
            "shaded meadow {herb_shaded} mm should transpire less than in the open {herb_open} mm"
        );
        // The ratio is the light transmitted to the herb stratum.
        let mut shaded_cell = crate::cell::CellProperties::default();
        shaded_cell.vegetation[crate::species::species_index(beech.0)] = beech.1;
        let expected = light_transmittance_below(&shaded_cell, Stratum::Herb);
        let ratio = herb_shaded / herb_open;
        assert!(
            (ratio - expected).abs() < 0.01,
            "shaded/open = {ratio:.4}, Beer-Lambert transmittance {expected:.4}"
        );
    }

    #[test]
    fn diurnal_convection_pushes_more_humidity_at_noon_than_at_night() {
        // Issue #46: with sin_elev_pos > 0 and T > t_ref, the diurnal
        // convective drive adds a boost to `step_uplift`. Compares a call
        // at sin_elev_pos = 0.0 (night) vs 0.95 (summer noon, 44.5°N
        // plain) and checks that humidity_surface decreased more under
        // the diurnal regime.
        fn run_with_sin_elev(sin_elev: f32) -> f32 {
            let mut grid = HexGrid::from_radius(0);
            let c0 = HexCoord::new(0, 0);
            if let Some(cell) = grid.get_mut(c0) {
                cell.humidity_surface = 100.0;
                // T = 30 °C >> t_ref(plain 44.5°N) ≈ 2 °C ⇒ t_excess ≈ 28 K
                cell.temperature = 30.0;
                cell.elevation = 0.0;
                cell.water_level = 0.0;
            }
            let params = AtmosphereParams::default();
            // Convert to "hourly" regime to test the real value of a
            // single tick (otherwise convective_diurnal_coef = 0.0005 per
            // day gives almost nothing on a single call).
            let params_hourly = scale_atmosphere_for_hourly_tick(&params);
            let tp = default_temp_params();
            step_uplift(&mut grid, &params_hourly, &tp, sin_elev);
            grid.get(c0).unwrap().humidity_surface
        }

        let after_night = run_with_sin_elev(0.0);
        let after_noon = run_with_sin_elev(0.95);
        let drop_night = 100.0 - after_night;
        let drop_noon = 100.0 - after_noon;
        assert!(
            drop_noon > drop_night * 1.05,
            "drive diurne devait pousser plus d'humidite en haut a midi qu'a minuit : night drop={drop_night:.4} noon drop={drop_noon:.4}"
        );
    }

    #[test]
    fn phys_oro_lift_bounded_by_saturation() {
        // Issue #63 Phase 4 Step 3: LCL bound on the orographic pump.
        //
        // Setup: 1 low cell (radius 0) surrounded by 6 high neighbors
        // already at HR_upper ≈ 0.99. Source full of humidity_surface.
        // Without the LCL bound, `step_orographic_convection` injects 30%
        // of surf into the neighbor's upper on every tick → HR_upper >> 1
        // immediately. With the bound, transport is capped by the
        // downstream saturation deficit, so the neighbor's HR_upper stays
        // ≤ 1 + epsilon.
        let mut current = HexGrid::from_radius(1);
        let center = HexCoord::new(0, 0);
        let coords: Vec<HexCoord> = current.coords().copied().collect();
        let neighbors: Vec<HexCoord> = coords.iter().copied().filter(|&c| c != center).collect();
        let temp_params = default_temp_params();
        let params = AtmosphereParams::default();

        // Low source, saturated with surface humidity
        if let Some(c) = current.get_mut(center) {
            c.elevation = 0.0;
            c.humidity_surface = 100.0;
            c.humidity_upper = 0.0;
            c.temperature = 20.0;
        }
        // High neighbors, already close to upper saturation. The upper
        // air is homogeneous (map means + lapse), so its saturation over
        // a 200 m neighbor is read from the same helper the engine uses.
        for &nc in &neighbors {
            if let Some(c) = current.get_mut(nc) {
                c.elevation = 200.0;
                c.temperature = 18.0; // slightly colder T
                c.humidity_surface = 0.0;
            }
        }
        let (mean_t, mean_z) = surface_means(&current);
        let sat_high = saturation_upper(
            upper_air_temperature(mean_t, mean_z, 200.0, &params, &temp_params),
            &params,
        );
        for &nc in &neighbors {
            current.get_mut(nc).unwrap().humidity_upper = sat_high * 0.99;
        }
        let mut next = current.clone();

        let mut scratch = AtmoScratch::new(current.len());
        // Orographic convection consumes the precomputed `sat_upper_offset`
        // (#97); on a direct call outside `step_atmosphere_into`, fill it
        // here.
        scratch.fill_upper_air(&current, mean_t, mean_z, &params, &temp_params);
        step_orographic_convection(&current, &mut next, &params, &mut scratch);

        // LCL invariant: no high neighbor should be oversaturated.
        for &nc in &neighbors {
            let cell = next.get(nc).unwrap();
            let sat = sat_high;
            let hr = cell.humidity_upper / sat;
            assert!(
                hr <= 1.05,
                "high neighbor oversaturated: hr={hr}, hu={}, sat={sat}",
                cell.humidity_upper
            );
        }

        // Conservation invariant: the source lost exactly what the
        // neighbors gained (conservative pump + surplus falls back to the
        // source).
        let total_before = 100.0_f32
            + neighbors.iter().fold(0.0_f32, |acc, &nc| {
                acc + current.get(nc).unwrap().humidity_upper
            });
        let total_after: f32 = coords
            .iter()
            .map(|&c| {
                let cell = next.get(c).unwrap();
                cell.humidity_surface + cell.humidity_upper
            })
            .sum();
        assert!(
            (total_after - total_before).abs() < 1e-3,
            "conservation violated: before={total_before}, after={total_after}"
        );
    }

    /// Harness for the orographic pump micro-tests: builds the world via
    /// `oro_pump_world` (`test_support`) then runs
    /// `step_orographic_convection` in isolation, with
    /// `orographic_lift_coef` multiplied by `coef_factor` (1.0 = defaults).
    fn run_oro_pump_with(current: &HexGrid, coef_factor: f32) -> HexGrid {
        let base = AtmosphereParams::default();
        let params = AtmosphereParams {
            orographic_lift_coef: base.orographic_lift_coef * coef_factor,
            ..base
        };
        run_oro_pump_params(current, &params)
    }

    fn run_oro_pump_params(current: &HexGrid, params: &AtmosphereParams) -> HexGrid {
        let temp_params = default_temp_params();
        let mut next = current.clone();
        let mut scratch = AtmoScratch::new(current.len());
        let (mean_t, mean_z) = surface_means(current);
        scratch.fill_upper_air(current, mean_t, mean_z, params, &temp_params);
        step_orographic_convection(current, &mut next, params, &mut scratch);
        next
    }

    /// Shorthand for the shipped law at the shipped coefficient.
    fn run_oro_pump(current: &HexGrid) -> HexGrid {
        run_oro_pump_with(current, 1.0)
    }

    /// The orographic pump is an elevator, not a diffuser: surface vapor
    /// rises into `humidity_upper` of the HIGHER neighbor, and exactly
    /// nothing goes to the lower neighbor or to neighbors at the same
    /// elevation. Directional complement to
    /// `phys_oro_lift_bounded_by_saturation` (which pins the LCL bound and
    /// conservation, not the direction of transport).
    #[test]
    fn orographic_pump_lifts_only_toward_higher_neighbors() {
        // 3 mm, not the 10 mm this test used to hold: on a +400 m step the
        // #156 law exports essentially the whole stock in one tick
        // (`1 − exp(−0.05 × 400) ≈ 1`) instead of the old capped 30 %, and
        // 10 mm would sit above the upper layer's own saturation
        // (~10.4 mm at 15 °C / 1500 m) — the LCL bound would then bite and
        // this test would stop measuring the direction of transport. Setup
        // adjusted, assertions untouched, cf `assert_lcl_slack`'s doc.
        const CENTER_HUMIDITY_MM: f32 = 3.0;
        let (mut grid, coords) = oro_pump_world(CENTER_HUMIDITY_MM);
        let center = HexCoord::new(0, 0);
        let uphill = center + DIRECTIONS[0];
        let downhill = center + DIRECTIONS[3];
        grid.get_mut(uphill).unwrap().elevation = 500.0;
        grid.get_mut(downhill).unwrap().elevation = 0.0;
        assert_lcl_slack(CENTER_HUMIDITY_MM);

        let next = run_oro_pump(&grid);

        let up = next.get(uphill).unwrap();
        assert!(
            up.humidity_upper > 0.0,
            "the high neighbor must receive vapor in the upper layer"
        );
        assert!(
            up.humidity_surface == 0.0,
            "the pump delivers aloft, not at the surface: surf={}",
            up.humidity_surface
        );
        for &c in &coords {
            if c == center || c == uphill {
                continue;
            }
            let cell = next.get(c).unwrap();
            assert!(
                cell.humidity_surface == 0.0 && cell.humidity_upper == 0.0,
                "only the HIGH neighbor receives, leak toward {c:?} \
                 (surf={}, upper={})",
                cell.humidity_surface,
                cell.humidity_upper
            );
        }
        let lost = CENTER_HUMIDITY_MM - next.get(center).unwrap().humidity_surface;
        assert!(
            (lost - up.humidity_upper).abs() < 1e-4,
            "the center must lose exactly what the peak gains: \
             lost={lost}, gained={}",
            up.humidity_upper
        );
    }

    /// The split between higher neighbors is proportional to the
    /// positive elevation difference: a neighbor at +400 m receives 4x
    /// what a neighbor at +100 m receives (`share_j = Δz_j / ΣΔz⁺`). Same
    /// temperatures and empty upper → the LCL bound doesn't bite, the
    /// ratio is exact.
    #[test]
    fn orographic_pump_share_scales_with_elevation_gap() {
        // 3 mm for the same reason as
        // `orographic_pump_lifts_only_toward_higher_neighbors`: under the
        // #156 law a 500 m total gap exports the whole stock, and the LCL
        // bound would clip the tall neighbour first, destroying the very
        // ratio this test measures.
        const CENTER_HUMIDITY_MM: f32 = 3.0;
        let (mut grid, _coords) = oro_pump_world(CENTER_HUMIDITY_MM);
        let center = HexCoord::new(0, 0);
        let tall = center + DIRECTIONS[0];
        let short = center + DIRECTIONS[2];
        grid.get_mut(tall).unwrap().elevation = 500.0; // +400 m
        grid.get_mut(short).unwrap().elevation = 200.0; // +100 m
        assert_lcl_slack(CENTER_HUMIDITY_MM);

        let next = run_oro_pump(&grid);

        let g_tall = next.get(tall).unwrap().humidity_upper;
        let g_short = next.get(short).unwrap().humidity_upper;
        assert!(
            g_tall > 0.0 && g_short > 0.0,
            "both high neighbors must receive: tall={g_tall}, short={g_short}"
        );
        let ratio = g_tall / g_short;
        assert!(
            (ratio - 4.0).abs() < 1e-3,
            "share ∝ elevation gap: ratio measured {ratio}, expected 4.0 (400 m / 100 m)"
        );
    }

    /// r250 perf effort: mass conservation of the scatter -> gather split
    /// (`step_orographic_convection`'s phase 1 writes `dir_out`, phase 2
    /// derives both self-loss terms from it and gathers the inflow via
    /// `coord::opposite_direction`). Several neighbors at different
    /// elevations on a radius-2 grid, so more than one direction of
    /// `dir_out` is non-zero per source: total humidity (surface + upper)
    /// over the whole grid must be unchanged within f32 rounding.
    #[test]
    fn orographic_pump_gather_conserves_total_humidity() {
        let (mut grid, coords) = oro_pump_world(10.0);
        for (k, &c) in coords.iter().enumerate() {
            if c != HexCoord::new(0, 0) {
                let bucket = f32::from(u16::try_from(k % 6).unwrap());
                grid.get_mut(c).unwrap().elevation = 100.0 + 20.0 * bucket;
            }
        }
        let before: f32 = coords
            .iter()
            .map(|&c| {
                let cell = grid.get(c).unwrap();
                cell.humidity_surface + cell.humidity_upper
            })
            .sum();
        let next = run_oro_pump(&grid);
        let after: f32 = coords
            .iter()
            .map(|&c| {
                let cell = next.get(c).unwrap();
                cell.humidity_surface + cell.humidity_upper
            })
            .sum();
        let drift = (after - before).abs() / before.max(1.0);
        assert!(
            drift < 1e-4,
            "gather not conservative: before={before} after={after} drift={drift}"
        );
    }

    /// Ablation #oro (subsampled cadence): a boosted pass (`sub = 3`,
    /// `orographic_lift_coef` tripled via `oro_boosted_params`) must stay
    /// exactly as conservative as the plain hourly pass above — the
    /// gather step derives both self-loss terms from `dir_out`,
    /// independent of the coefficient's magnitude. Same world/elevation
    /// layout as `orographic_pump_gather_conserves_total_humidity`.
    #[test]
    fn orographic_pump_conserves_total_humidity_boosted_x3() {
        let (mut grid, coords) = oro_pump_world(10.0);
        for (k, &c) in coords.iter().enumerate() {
            if c != HexCoord::new(0, 0) {
                let bucket = f32::from(u16::try_from(k % 6).unwrap());
                grid.get_mut(c).unwrap().elevation = 100.0 + 20.0 * bucket;
            }
        }
        let before: f32 = coords
            .iter()
            .map(|&c| {
                let cell = grid.get(c).unwrap();
                cell.humidity_surface + cell.humidity_upper
            })
            .sum();

        let params = oro_boosted_params(&AtmosphereParams::default(), 3);
        let temp_params = default_temp_params();
        let mut next = grid.clone();
        let mut scratch = AtmoScratch::new(grid.len());
        let (mean_t, mean_z) = surface_means(&grid);
        scratch.fill_upper_air(&grid, mean_t, mean_z, &params, &temp_params);
        step_orographic_convection(&grid, &mut next, &params, &mut scratch);

        let after: f32 = coords
            .iter()
            .map(|&c| {
                let cell = next.get(c).unwrap();
                cell.humidity_surface + cell.humidity_upper
            })
            .sum();
        let drift = (after - before).abs() / before.max(1.0);
        assert!(
            drift < 1e-4,
            "gather not conservative at sub=3: before={before} after={after} drift={drift}"
        );
    }

    /// Ablation: without relief, the pump is inert. It really is the
    /// ELEVATION DIFFERENCE that causes the transport in the two previous
    /// tests, not the mere presence of humidity: a flat world comes out
    /// bit-identical, no diffuse leak.
    #[test]
    fn orographic_pump_is_inert_on_flat_terrain() {
        let (grid, coords) = oro_pump_world(50.0);
        let next = run_oro_pump(&grid);
        for &c in &coords {
            let before = grid.get(c).unwrap();
            let after = next.get(c).unwrap();
            // Bit comparison (precedent time.rs): keeps the strict
            // "bit-identical" invariant without clippy's float_cmp.
            assert!(
                before.humidity_surface.to_bits() == after.humidity_surface.to_bits()
                    && before.humidity_upper.to_bits() == after.humidity_upper.to_bits(),
                "flat terrain: {c:?} moved (surf {} → {}, upper {} → {})",
                before.humidity_surface,
                after.humidity_surface,
                before.humidity_upper,
                after.humidity_upper
            );
        }
    }

    // ================================================================
    // #156: the pump rate law
    // ================================================================

    /// The rate law itself, on the numbers rather than through the grid.
    /// Monotone in the relief, never above 1 whatever the relief (the
    /// conservation bound, held by construction and without a clamp), and
    /// tangent to its own small-`x` linear approximation `x = coef ×
    /// Σ Δz⁺` — the pre-#156 law before its cap ever bound, `x.clamp(0.0,
    /// 0.30)`, retired 2026-09-07 once this law had been green and merged
    /// long enough (since 2026-09-05).
    ///
    /// The bound is `≤ 1`, not `< 1`: in `f32`, `1 − exp(−x)` rounds to
    /// exactly `1.0` past `x ≈ 16.6`. That is still "the cell exports its
    /// whole surface stock and nothing more", cf [`oro_pump_rate`]'s doc.
    #[test]
    fn oro_pump_rate_is_monotone_bounded_and_linear_at_small_x() {
        let coef = AtmosphereParams::default().orographic_lift_coef / TICKS_PER_DAY_F32;
        let mut previous = 0.0_f32;
        for rise_m in [1.0_f32, 10.0, 72.0, 216.0, 500.0, 5_000.0, 100_000.0] {
            let rate = oro_pump_rate(coef, rise_m);
            assert!(
                rate >= previous,
                "rate must grow with relief: {rise_m} m gave {rate} after {previous}"
            );
            assert!(
                (0.0..=1.0).contains(&rate),
                "conservation bound: a cell cannot export more than it \
                 holds, {rise_m} m gave {rate}"
            );
            previous = rate;
        }
        // Strictly increasing over the band the engine actually visits
        // (relief up to the map's own maximum, a few hundred metres of
        // summed positive gap): no plateau where the coefficient is
        // supposed to be readable.
        for pair in [(10.0_f32, 72.0), (72.0, 216.0), (216.0, 500.0)] {
            let (lo, hi) = pair;
            assert!(
                oro_pump_rate(coef, hi) > oro_pump_rate(coef, lo) * 1.05,
                "no plateau expected between {lo} m and {hi} m of relief"
            );
        }
        // Gentle slope: the exponential agrees with its own Taylor
        // expansion `1 − exp(−x) ≈ x` to better than 1 % — the law is not
        // a recalibration, it only changes what happens at large `x`,
        // where the pre-#156 cap used to bind instead.
        let x = coef * 5.0;
        let gentle = oro_pump_rate(coef, 5.0);
        assert!(
            (gentle - x).abs() / x < 0.01,
            "small-x tangency: exponential={gentle}, linear approximation={x}"
        );
    }

    /// Units check (SI rule): `orographic_lift_coef` is the
    /// group `U_slope · dt / (H · L)` of [`oro_pump_rate`], so the shipped
    /// default has to correspond to a physically real upslope venting
    /// speed. 1-10 cm/s is what Henne et al. (2004) measure for
    /// thermally driven export of Alpine boundary-layer air to the free
    /// troposphere; anything outside ~0.03-0.5 m/s would mean the
    /// coefficient has stopped being a velocity in disguise.
    #[test]
    fn orographic_coef_implies_a_measured_venting_speed() {
        let params = AtmosphereParams::default();
        // The engine consumes the per-tick value, cf
        // `scale_atmosphere_for_hourly_tick`.
        let coef_per_tick = params.orographic_lift_coef / TICKS_PER_DAY_F32;
        let u_slope_m_per_s =
            coef_per_tick * params.upper_layer_altitude_m * CELL_SPACING_M / SECONDS_PER_HOUR;
        assert!(
            (0.03..0.5).contains(&u_slope_m_per_s),
            "implied upslope venting speed {u_slope_m_per_s} m/s is outside \
             the 1-10 cm/s band measured for topographic venting"
        );
    }

    #[test]
    fn uplift_conserves_total_humidity() {
        let mut grid = HexGrid::from_radius(3);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.humidity_surface = 60.0;
                cell.temperature = 15.0;
            }
        }
        let before = total_humidity(&grid);
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        step_uplift(&mut grid, &params, &tp, 0.5);
        let after = total_humidity(&grid);
        // Relative tolerance (≤ 1e-5) vs absolute: the Phase 3 ×200
        // rescale amplifies numerical noise in absolute value while
        // preserving f32 relative precision.
        let drift = (before - after).abs() / before.max(1.0);
        assert!(drift < 1e-5, "Uplift non conservatif : {before} -> {after}");
    }

    #[test]
    fn uplift_moves_surface_to_upper() {
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.humidity_surface = 100.0;
            cell.temperature = 20.0;
        }
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        step_uplift(&mut grid, &params, &tp, 0.5);
        let after = grid.get(c0).unwrap();
        assert!(after.humidity_surface < 100.0);
        assert!(after.humidity_upper > 0.0);
    }
}
