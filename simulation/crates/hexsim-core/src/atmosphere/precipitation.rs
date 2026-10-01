use crate::cell::CellProperties;
use crate::grid::HexGrid;
use crate::par::{for_each_chunk_mut, for_each_chunk_mut2};

use super::{AtmoScratch, AtmosphereParams, PrecipitationMap};

/// Rounds `precip_spread_radius` (hexes) to the integer number of
/// diffusion passes [`step_precipitation_into`] runs: the nearest ring
/// count, floored at 1 (a footprint narrower than one ring isn't
/// meaningful — `precip_neighbor_share`'s leaving fraction still has to
/// land somewhere). `1` is today's historical single-ring footprint,
/// reproduced bit for bit (no extra pass runs at all, see
/// `tests::radius_one_is_bit_identical_to_legacy`).
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded and floored at 1.0 on the line above: never negative, \
              and precip_spread_radius is a human-configured knob (default \
              3), nowhere near u32::MAX"
)]
pub(crate) fn precip_spread_passes(radius_hexes: f32) -> u32 {
    radius_hexes.round().max(1.0) as u32
}

/// Precipitation: consumes `cloud_water` above the collision/coalescence
/// threshold. Surplus droplets fall as rain (T>=0) or snow (T<0).
///
/// Global gate with hysteresis: opens if `mean(cloud_water)` > gate,
/// closes when mean < gate × 0.75. Except snow (T < 0°C), always allowed
/// to preserve winter snowpack accumulation.
///
/// `mean_cloud` is supplied by the caller, not read off a grid (coarse
/// upper layer, step 2): the fine path passes the plain map mean it
/// always did, the coarse torus passes `Σ_c N_c·cw_c / N` — the mean
/// weighted by each coarse cell's fine-cell count, because coarse cells
/// do not all hold the same number of them and an unweighted mean would
/// move this threshold silently (design note §9 risk 4). `None` = an
/// empty map, the gate keeps its state, same as before.
pub(crate) fn update_precip_gate(
    gate_open: &mut bool,
    params: &AtmosphereParams,
    mean_cloud: Option<f32>,
) {
    if params.global_precip_gate > 0.0 {
        if let Some(mean_cloud) = mean_cloud {
            if *gate_open {
                if mean_cloud < params.global_precip_gate * 0.75 {
                    *gate_open = false;
                }
            } else if mean_cloud > params.global_precip_gate {
                *gate_open = true;
            }
        }
    } else {
        *gate_open = true;
    }
}

/// Plain map mean of `cloud_water` over a fine grid, the number
/// [`update_precip_gate`] has always tested. `None` on an empty grid.
///
/// # Panics
/// Above 65 535 cells, as it always has (the gate is off by default —
/// `global_precip_gate = 0.0` — so this is only reachable on a map that
/// explicitly turned it on; pre-existing, untouched here).
fn fine_mean_cloud_water(grid: &HexGrid) -> Option<f32> {
    let cells = grid.cells_slice();
    let n = cells.len();
    if n == 0 {
        return None;
    }
    let n_f = f32::from(u16::try_from(n).expect("cell count fits u16"));
    Some(cells.iter().map(|c| c.cloud_water).sum::<f32>() / n_f)
}

// v0.5.x: physical autoconversion, Khairoutdinov & Kogan 2000.
//
// Reference: Khairoutdinov M. & Kogan Y. (2000), "A New Cloud Physics
// Parameterization in a Large-Eddy Simulation Model of Marine Stratocumulus",
// Mon. Weather Rev. 128, 229–243. See also Wood (2005) for the review.
//
// Formula: `P_auto = K × q_c^a × N_c^b` (kg/kg/s) where
//   - q_c is the cloud water mixing ratio (kg water / kg dry air)
//   - N_c is the droplet concentration (cm^-3)
//   - K = 1350 s^-1, a = 2.47, b = -1.79 (original KK2000 values).
//
// Local conversion: our `cloud_water` stock is in mm of LWP (Liquid Water
// Path) integrated over the average cloud layer (~1500 m). To feed
// KK2000 we derive q_c via density: water_density = LWP / L,
// q_c = water_density / air_density ≈ cloud_water_mm × 1e-3 / L_m.
//
// The super-linear character (exponent 2.47) is *the whole point* of the
// change: a small cloud (q_c ~ 1e-5) produces 10^4 times less rain than a
// large one (q_c ~ 1e-3). Emergent consequence: clouds grow and drift
// with the wind before raining, the old linear drain used to empty them
// as soon as they existed.
//
// `CLOUD_MIN_PRECIP` is a numerical floor to avoid the near-zero root in
// the KK2000 power (q_c^2.47 → 0).
const CLOUD_MIN_PRECIP: f32 = 0.05;

/// `K` coefficient of KK2000 (s^-1): Khairoutdinov & Kogan 2000.
const KK2000_K: f32 = 1350.0;
/// Exponent on `q_c` in KK2000.
const KK2000_QC_EXP: f32 = 2.47;
/// Exponent on `N_c` (droplet concentration) in KK2000.
const KK2000_NC_EXP: f32 = -1.79;
/// Assumed air density (kg/m^3) at the average cloud layer level.
/// 1.0 is a simplification; the real value at ~1500 m is ~1.05.
const AIR_DENSITY_KG_M3: f32 = 1.0;
/// Seconds per hour, to go from the SI rate to the hourly time step.
const SECONDS_PER_HOUR_KK: f32 = 3600.0;

/// Converts `cloud_water` (mm of LWP integrated over
/// `layer_thickness_m`) to a mixing ratio `q_c` (kg/kg). See the module
/// comment block above for the derivation.
///
/// Key identity: 1 mm of LWP = 1 kg/m^2 (volume × water density = 1 m^2
/// × 1e-3 m × 1000 kg/m^3 = 1 kg). Over an air column of mass
/// `L × air_density` kg/m^2, we get `q_c` = LWP / (L × `air_density`).
#[must_use]
pub fn cloud_water_to_qc(cloud_water_mm: f32, layer_thickness_m: f32) -> f32 {
    if layer_thickness_m <= 0.0 {
        return 0.0;
    }
    cloud_water_mm / (layer_thickness_m * AIR_DENSITY_KG_M3)
}

/// KK2000 microphysical drain rate, in mm of `cloud_water` lost per hour
/// (= mm of rain produced per hour, mass conservation).
///
/// `droplet_count_pow` = `N_c^KK2000_NC_EXP` precomputed by the caller:
/// `N_c` is a constant parameter during the tick, the per-cell `powf` was
/// pure recomputation (perf project #88, half of the 27M powf/year).
/// Equals 0.0 if `N_c <= 0` (drain disabled, same guard as before).
#[must_use]
fn kk2000_drain_mm_per_hour(
    cloud_water_mm: f32,
    layer_thickness_m: f32,
    droplet_count_pow: f32,
) -> f32 {
    if cloud_water_mm <= CLOUD_MIN_PRECIP || droplet_count_pow <= 0.0 {
        return 0.0;
    }
    let qc = cloud_water_to_qc(cloud_water_mm, layer_thickness_m);
    // P_auto in kg/kg/s. Same multiplication order as before the precompute
    // (bit-identical): (K × qc^a) × nc_pow.
    let p_auto = KK2000_K * qc.powf(KK2000_QC_EXP) * droplet_count_pow;
    // Convert back to mm/s of cloud_water (q_c × L × air_density = mm).
    let drain_mm_per_s = p_auto * layer_thickness_m * AIR_DENSITY_KG_M3;
    drain_mm_per_s * SECONDS_PER_HOUR_KK
}

/// `cloud_water` (mm) converted to rain over `dt_hours` by KK2000
/// autoconversion, via **analytical integration** of `dq/dt = -C·q^α`
/// (α = 2.47, #50).
///
/// KK2000 is super-linear: at high `q` the instantaneous rate `C·q^α`
/// naively integrated by Euler (`rate × dt`) exceeds the available
/// stock, hence the old corrective `.min(cloud_water)` (anti-pattern #4:
/// production silently bounded by availability, which would mask a
/// `cloud_water` drift coming from elsewhere). The exact solution of
/// `dq/dt = -C·q^α` (α ≠ 1) is monotonically decreasing toward 0:
///
/// ```text
///   q(t) = q₀ · (1 + (α−1)·(rate₀/q₀)·t)^(−1/(α−1)),   rate₀ = C·q₀^α
/// ```
///
/// so the drain `q₀ − q(dt)` is **≤ q₀ by construction**, with no clamp,
/// for any `dt`. At small drains (`x → 0`) it converges to Euler (`≈
/// rate₀ · dt`): the drizzle regime is not disturbed, only heavy showers
/// are smoothed (exponential tail instead of a truncated purge).
///
/// `dt_hours` is the cadence of the precipitation pass
/// (`PRECIP_SUBSAMPLE_HOURS`): integrating one pass over `N` hours is
/// **not** "hourly rate × N", which would overshoot exactly where the
/// super-linear rate is largest — it is the same closed form with `x`
/// carrying the longer `dt`, so `N` chained hourly passes and one
/// `dt = N` pass drain the same stock to rounding. `dt_hours = 1` is the
/// historical hourly path bit for bit: `x × 1.0` is exact in IEEE 754
/// (pinned by `autoconv_dt1_is_bit_identical_to_the_historical_form`).
/// The multiplication is unconditional rather than branched on
/// `dt_hours == 1.0` because that float equality trips
/// `clippy::float_cmp` (pedantic, denied at commit) and the project bans
/// `#[allow]`.
#[must_use]
fn kk2000_autoconv_over_hours(
    cloud_water_mm: f32,
    layer_thickness_m: f32,
    droplet_count_pow: f32,
    dt_hours: f32,
) -> f32 {
    // Instantaneous rate C·q₀^α (mm/h). 0 below the floor or N_c disabled.
    let rate0 = kk2000_drain_mm_per_hour(cloud_water_mm, layer_thickness_m, droplet_count_pow);
    if rate0 <= 0.0 {
        return 0.0;
    }
    let exp_m1 = KK2000_QC_EXP - 1.0;
    // x = (α−1)·rate₀/q₀ · dt, dimensionless.
    // rate₀ = C·q₀^α ⇒ (α−1)·C·q₀^(α−1) = (α−1)·rate₀/q₀: this avoids
    // reconstructing C.
    let x = exp_m1 * rate0 / cloud_water_mm * dt_hours;
    let q_end = cloud_water_mm * (1.0 + x).powf(-1.0 / exp_m1);
    // q_end ∈ (0, q₀] ⇒ drain ∈ [0, q₀). The max(0) only covers f32
    // rounding, it is not a physical safeguard (the analytical solution
    // already guarantees ≤ stock).
    (cloud_water_mm - q_end).max(0.0)
}

/// Per-source outflow of one precipitation tick (r250 perf effort),
/// packed as a single struct so the heavy per-cell KK2000 math runs once
/// in a single parallel pass (`par::for_each_chunk_mut`) instead of one
/// pass per field. `self_rain`/`self_snow` are pure self-terms (the
/// fraction the source keeps, `1 - precip_neighbor_share`): no gather
/// needed, carried straight through to the apply pass. `share_rain`/
/// `share_snow` are the uniform amount sent to EACH of the source's 6
/// toric neighbors (`precip_neighbor_share`, split by 6): like
/// `step_cloud_diffusion`, the split is direction-agnostic, so the
/// gather sums `Σ_{k ∈ neighbors(j)} share[k]` without needing
/// `coord::opposite_direction`.
#[derive(Clone, Copy, Default)]
pub struct PrecipOutflow {
    cloud_delta: f32,
    self_rain: f32,
    self_snow: f32,
    share_rain: f32,
    share_snow: f32,
}

/// Per-tick rates and thresholds the per-column drain
/// [`precip_amount_mm`] needs, grouped so its callers stay under the
/// argument ceiling: all read-only, derived once per pass from
/// `AtmosphereParams` by [`PrecipRates::new`].
pub(crate) struct PrecipRates {
    gate_closed: bool,
    precip_floor: f32,
    altitude_m: f32,
    max_precip_per_tick: f32,
    w_ref: f32,
    w_floor: f32,
    droplet_count_pow: f32,
    dt_hours: f32,
}

impl PrecipRates {
    /// Derives the pass's rates from the (already hourly-scaled, already
    /// cadence-boosted) params. `gate_closed` is the state of the global
    /// hysteresis gate, stepped by the caller just before.
    pub(crate) fn new(params: &AtmosphereParams, dt_hours: f32, gate_closed: bool) -> Self {
        Self {
            gate_closed,
            // Convective inhibition (ex-design A, #69): below the critical
            // mass the cloud builds up and travels, it does not
            // precipitate. Default 0.0 → only CLOUD_MIN_PRECIP applies.
            precip_floor: CLOUD_MIN_PRECIP.max(params.precip_crit_mm),
            altitude_m: params.upper_layer_altitude_m,
            max_precip_per_tick: params.max_precip_per_tick,
            // Updraft trigger (synoptic Phase 3): precip factor ∝ vertical
            // velocity (convergence + orographic). Default w_ref=0 →
            // factor=1 everywhere.
            w_ref: params.updraft_ref_ms,
            w_floor: params.updraft_floor,
            // N_c^b precomputed once per pass: constant parameter, the
            // per-cell powf was pure recomputation (#88). 0.0 = drain
            // disabled (N_c <= 0), same guard as in
            // `kk2000_drain_mm_per_hour` before the precompute.
            droplet_count_pow: if params.kk2000_droplet_count > 0.0 {
                params.kk2000_droplet_count.powf(KK2000_NC_EXP)
            } else {
                0.0
            },
            dt_hours,
        }
    }

    /// Does this pass read the updraft field at all? `false` at the
    /// shipped default (`updraft_ref_ms = 0`), where `scratch.convergence`
    /// is never even filled — so callers must not index it.
    pub(crate) fn uses_updraft(&self) -> bool {
        self.w_ref > 0.0
    }
}

/// Cloud water (mm) one column turns into falling precipitation over this
/// pass: KK2000 autoconversion integrated over `dt_hours`, capped by the
/// microphysical per-pass ceiling, modulated by the ascent trigger. `0.0`
/// for a column under the floor or held back by the global gate.
///
/// Split out of [`fill_precip_outflow`] (coarse upper layer, step 2) so
/// the fine grid and the ~1 km coarse torus (`atmosphere::coarse`) drain
/// by the same code. What the two paths do with the result differs and is
/// NOT here: the fine one splits it into a rain/snow phase at the source
/// and shares a fraction with the neighbours, the coarse one drops it
/// uniformly on its own fine cells, each deciding its own phase (design
/// note §4, options (a) and (d)).
///
/// # `cloud_fraction`: the saturated part of the column's footprint
///
/// `cloud_water` is the **grid-box mean** `q_c`; `cloud_fraction` is the
/// share `f_c ∈ (0, 1]` of that box the droplets actually occupy, so the
/// **in-cloud** content is `q_in = q_c / f_c`. KK2000 is evaluated on
/// `q_in` and the resulting drain is weighted back by `f_c`: this is the
/// standard sub-grid cloud closure (Sundqvist 1978, *Mon. Wea. Rev.* 106;
/// Smith 1990, *QJRMS* 116; Tiedtke 1993, *Mon. Wea. Rev.* 121), and it
/// matters because the autoconversion is super-linear (`q^2.47`): the same
/// condensate concentrated on a sixth of the box drains
/// `f_c^(1−2.47) ≈ 15×` more than spread flat over it. On a hex torus a
/// coarse cell holds ~46 to 56 fine cells, so ignoring `f_c` is a real
/// factor, not a rounding.
///
/// Conservation needs no clamp: `kk2000_autoconv_over_hours` returns at
/// most its own argument, so `f_c × drain(q_in) ≤ f_c × q_in = q_c`
/// whatever `f_c` is. And the per-pass cap is applied to the **grid-box
/// sheet**, after the weighting, which is where it means "a cloud cannot
/// dump more than this per pass over its footprint".
///
/// `cloud_fraction = 1.0` is the no-sub-grid-information case and the fine
/// grid's permanent value: `q_c / 1.0` and `drain × 1.0` are exact in IEEE
/// 754, so the fine path is bit-identical to before this parameter existed
/// (pinned by `unit_cloud_fraction_is_bit_identical_to_the_grid_box_drain`).
///
/// `temperature` only feeds the gate's snow exception (snow is always
/// allowed, so a closed gate cannot stop a winter snowpack from building).
/// `updraft` must be `0.0` when [`PrecipRates::uses_updraft`] is false:
/// the field is not filled then.
#[must_use]
pub(crate) fn precip_amount_mm(
    cloud_water: f32,
    cloud_fraction: f32,
    temperature: f32,
    updraft: f32,
    rates: &PrecipRates,
) -> f32 {
    // In-cloud content: what the microphysics sees inside the saturated
    // part of the box, `q_c / f_c`. The floor is a threshold on the cloud
    // itself, so it is tested there too — a thin veil spread over the
    // whole box and a real cloud over a sixth of it are not the same
    // cloud, which is the entire point of the closure.
    let in_cloud = cloud_water / cloud_fraction;
    if in_cloud <= rates.precip_floor {
        return 0.0;
    }
    let precip_allowed = !rates.gate_closed || temperature < 0.0;
    if !precip_allowed {
        return 0.0;
    }

    // KK2000 autoconversion: super-linear drain in cloud_water^2.47.
    // Small clouds: near-zero drain (they have time to travel). Large
    // clouds: fast drain (natural purge of cumulonimbus). Analytically
    // integrated over the pass's `dt_hours` (#50): the drain is ≤ stock
    // by construction, no more `.min(cloud_water)` conservation clamp.
    // Evaluated in-cloud, then weighted back to the grid box by `f_c`.
    let drained = kk2000_autoconv_over_hours(
        in_cloud,
        rates.altitude_m,
        rates.droplet_count_pow,
        rates.dt_hours,
    ) * cloud_fraction;
    // Microphysical cap: even very heavily loaded, a cloud cannot dump
    // more than a certain volume per pass (bounded fall speed). Spreads
    // heavy showers over several passes. Already scaled by the cadence
    // upstream (see `step_precipitation_into`'s doc). This is a physical
    // cap distinct from conservation; `drained ≤ cloud_water` is already
    // guaranteed, this `.min` no longer masks anything.
    let mut amount = if rates.max_precip_per_tick > 0.0 {
        drained.min(rates.max_precip_per_tick)
    } else {
        drained
    };
    // Modulation by updraft: rain where air rises (front, windward
    // slope), dry under subsidence. Unprecipitated water stays in
    // cloud_water, so it accumulates and travels until an updraft.
    if rates.w_ref > 0.0 {
        let f = (rates.w_floor + updraft / rates.w_ref).clamp(0.0, 1.0);
        amount *= f;
    }
    amount
}

/// [`PrecipRates`] plus the fine grid's own neighbour share: the coarse
/// torus has no use for it (the coarse cell IS the footprint, design note
/// §4), so it does not travel in the shared struct.
struct PrecipOutflowRates {
    column: PrecipRates,
    share: f32,
}

/// One precipitation pass covering `dt_hours` of simulated time.
///
/// `dt_hours` is the cadence of the pass (`PRECIP_SUBSAMPLE_HOURS`, 1 =
/// hourly = historical behavior). It is threaded to the KK2000 closed
/// form, which integrates the autoconversion ODE over that duration
/// instead of scaling an hourly rate. It is deliberately NOT applied to
/// every quantity in the pass, because they are not all rates:
///
/// - `max_precip_per_tick` (mm per pass) IS scaled by the caller
///   (`scaling::precip_boosted_params`): it is a bounded fall speed over
///   the pass, so a pass three times longer may drop three times as much
///   before the cap bites. Scaling it in the caller rather than here
///   keeps the cap a param, visible in the checkpointed config.
/// - `precip_floor`, `updraft_ref_ms`/`updraft_floor`,
///   `precip_neighbor_share`, `precip_spread_radius` and the 0 °C
///   rain/snow split do NOT scale: a threshold on a stock, a
///   dimensionless modulation factor, a dimensionless dispersion share, a
///   footprint size in hexes and a phase test are per-event quantities,
///   independent of how long the event integrates over.
/// - `global_precip_gate` is a hysteresis on the map-mean cloud water,
///   evaluated once per pass: at cadence `N` it is simply sampled one
///   hour in `N`, a coarser sampling of the same hysteresis, nothing to
///   rescale.
///
/// The footprint itself: `gather_precip_deltas` spreads one ring
/// (`precip_neighbor_share` split 6 ways), then `spread_precip_further`
/// runs `precip_spread_passes(precip_spread_radius) - 1` more isotropic
/// passes over the result, reaching the configured radius with a
/// gradient decreasing outward (see that function's doc for the
/// mass-conservation argument and its isotropy caveat).
///
/// `events` must be zeroed by the caller BEFORE the cadence gate, so a
/// skipped hour reports zero precipitation. That is a real change of
/// forcing downstream, not just an accounting one: `events` feeds
/// `Simulation::scratch_precip_tick`, which `step_snow` reads next tick
/// as "rain of the previous tick" for rain-on-snow heat advection. At
/// cadence `N` the snow pack sees the same daily rain total arriving in
/// `N`-hour bursts instead of hourly. The daily accumulator
/// (`last_precipitation`) is a sum and is cadence-invariant.
pub(crate) fn step_precipitation_into(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    dt_hours: f32,
    gate_open: &mut bool,
    events: &mut PrecipitationMap,
    scratch: &mut AtmoScratch,
) {
    update_precip_gate(gate_open, params, fine_mean_cloud_water(next));
    let gate_closed = !*gate_open;

    let n = next.len();
    scratch.precip_outflow.clear();
    scratch.precip_outflow.resize(n, PrecipOutflow::default());

    let rates = PrecipOutflowRates {
        column: PrecipRates::new(params, dt_hours, gate_closed),
        share: params.precip_neighbor_share.clamp(0.0, 1.0),
    };
    fill_precip_outflow(
        next.cells_slice(),
        &scratch.convergence,
        &rates,
        &mut scratch.precip_outflow,
    );

    // Phase 2 (gather): per destination cell, self-terms plus the
    // uniform share gathered from its 6 toric neighbors (same
    // direction-agnostic sum as `step_cloud_diffusion`, no
    // `coord::opposite_direction` needed).
    scratch.precip_water_delta.clear();
    scratch.precip_water_delta.resize(n, 0.0);
    scratch.precip_snow_delta.clear();
    scratch.precip_snow_delta.resize(n, 0.0);
    gather_precip_deltas(
        next,
        &scratch.precip_outflow,
        &mut scratch.precip_water_delta,
        &mut scratch.precip_snow_delta,
    );

    // Phase 2b (footprint): `gather_precip_deltas` above is exactly one
    // application of the isotropic diffusion rule (`1 -
    // precip_neighbor_share` stays, `precip_neighbor_share / 6` moves to
    // each toric neighbor) to the per-source `amount` — the historical,
    // `precip_spread_radius = 1` footprint. Applying the SAME rule
    // `extra_passes` more times to the field it just produced spreads the
    // footprint one more ring per pass (see `spread_precip_further`'s
    // doc for why this stays mass-conserving and why a uniform field
    // still caps at `max_precip_per_tick` for any radius). A no-op, no
    // buffer touched, when the radius rounds to 1.
    let extra_passes = precip_spread_passes(params.precip_spread_radius).saturating_sub(1);
    spread_precip_further(
        next,
        rates.share,
        extra_passes,
        &mut scratch.precip_water_delta,
        &mut scratch.precip_snow_delta,
        &mut scratch.precip_water_delta_tmp,
        &mut scratch.precip_snow_delta_tmp,
    );

    // Pass 3: apply. Reads only `precip_outflow[i].cloud_delta` and the
    // gathered `precip_water_delta`/`precip_snow_delta[i]`, writes only
    // `next[i]` and `events[i]` — a pure per-cell map, parallelizable
    // (`par::for_each_chunk_mut2`).
    let cloud_delta: &Vec<PrecipOutflow> = &scratch.precip_outflow;
    let water_delta: &Vec<f32> = &scratch.precip_water_delta;
    let snow_delta: &Vec<f32> = &scratch.precip_snow_delta;
    for_each_chunk_mut2(
        next.cells_slice_mut(),
        events,
        |start, cells_chunk, events_chunk| {
            for (local, nc) in cells_chunk.iter_mut().enumerate() {
                let i = start + local;
                let cd = cloud_delta[i].cloud_delta;
                let rain = water_delta[i];
                let snow = snow_delta[i];
                if cd == 0.0 && rain == 0.0 && snow == 0.0 {
                    continue;
                }
                nc.cloud_water = (nc.cloud_water + cd).max(0.0);
                nc.water_level = (nc.water_level + rain).max(0.0);
                nc.snow_level = (nc.snow_level + snow).max(0.0);
                events_chunk[local].rain += rain;
                events_chunk[local].snow += snow;
            }
        },
    );
}

/// Phase 1 of [`step_precipitation_into`]: per source cell, the heavy
/// KK2000 math and the rain/snow split. Reads only `cells[i]` and
/// `updraft[i]`: fully independent per source, parallelizable
/// (`par::for_each_chunk_mut`).
fn fill_precip_outflow(
    cells: &[CellProperties],
    updraft: &[f32],
    rates: &PrecipOutflowRates,
    outflow: &mut [PrecipOutflow],
) {
    let reads_updraft = rates.column.uses_updraft();
    for_each_chunk_mut(outflow, |start, chunk| {
        for (local, out) in chunk.iter_mut().enumerate() {
            let i = start + local;
            let nc = &cells[i];
            *out = PrecipOutflow::default();

            // `convergence` is only filled when the ascent trigger is
            // active (`fill_updraft_into`'s call site), so the slice must
            // not be indexed otherwise.
            let w = if reads_updraft { updraft[i] } else { 0.0 };
            // `1.0`: the fine grid has no sub-grid distribution to
            // exploit — a 130 m column IS the footprint of its own cloud.
            let amount = precip_amount_mm(nc.cloud_water, 1.0, nc.temperature, w, &rates.column);
            if amount <= 0.0 {
                continue;
            }
            out.cloud_delta = -amount;

            // Spatial dispersion: a fraction `precip_neighbor_share` of
            // the rain falls on the neighbors (mixed air), the rest on
            // the source cell.
            let self_amount = amount * (1.0 - rates.share);
            let share_each = (amount * rates.share) / 6.0;
            let is_snow = nc.temperature < 0.0;
            if is_snow {
                out.self_snow = self_amount;
                out.share_snow = share_each;
            } else {
                out.self_rain = self_amount;
                out.share_rain = share_each;
            }
        }
    });
}

/// Phase 2 of [`step_precipitation_into`]: per destination cell,
/// self-terms plus the uniform share gathered from its 6 toric
/// neighbors (same direction-agnostic sum as `step_cloud_diffusion`, no
/// `coord::opposite_direction` needed).
fn gather_precip_deltas(
    grid: &HexGrid,
    outflow: &[PrecipOutflow],
    rain_delta: &mut [f32],
    snow_delta: &mut [f32],
) {
    for_each_chunk_mut2(rain_delta, snow_delta, |start, rain_chunk, snow_chunk| {
        for local in 0..rain_chunk.len() {
            let j = start + local;
            let neighbors = grid.neighbor_indices_toric(j);
            let mut rain = outflow[j].self_rain;
            let mut snow = outflow[j].self_snow;
            for &k in &neighbors {
                rain += outflow[k].share_rain;
                snow += outflow[k].share_snow;
            }
            rain_chunk[local] = rain;
            snow_chunk[local] = snow;
        }
    });
}

/// Phase 2b of [`step_precipitation_into`]: runs `extra_passes` more
/// applications of [`gather_precip_deltas`]'s isotropic diffusion rule on
/// the rain/snow field it produced, spreading the footprint one more
/// ring per pass (`precip_spread_radius`). A no-op when `extra_passes ==
/// 0` (the `precip_spread_radius = 1` default before this feature, and
/// the value that reproduces it bit for bit).
///
/// Why this conserves mass for any `extra_passes`: write the rule as a
/// linear map `D` on the per-cell field,
///
/// ```text
/// D(f)[j] = f[j] * (1 - share) + sum_{k in neighbors(j)} f[k] * share/6
/// ```
///
/// `D` is symmetric (`j` is a neighbor of `k` iff `k` is a neighbor of
/// `j`, so the `f[k]` coefficient at `j` equals the `f[j]` coefficient at
/// `k`) and every row sums to 1 — `(1-share) + 6*(share/6) = 1`, whatever
/// the neighbor indices are, even the degenerate small-torus case where
/// some of a cell's "6 neighbors" coincide (`HexGrid::neighbor_indices_toric`'s
/// contract). `gather_precip_deltas` is `D` applied once to the
/// per-source `amount` (a value concentrated at the source: `self_rain =
/// amount * (1-share)`, `share_rain = amount * share / 6` at each
/// neighbor, exactly `D` of a point mass); this function applies `D`
/// `extra_passes` more times, so the whole footprint is `D` to the power
/// `1 + extra_passes` of that point mass. Symmetric plus row-sum-1
/// (doubly stochastic) is closed under matrix product, so any power of
/// `D` stays symmetric and doubly stochastic: total mass is conserved
/// (row sums stay 1, nothing leaves the grid), and because it is also
/// symmetric its column sums are 1 too, so a destination surrounded by
/// identically-capped sources still receives exactly the cap whatever
/// the radius is (`phys_snapshot_precip_is_hourly`'s per-hour ceiling
/// holds unchanged).
///
/// Isotropy caveat, found while implementing this, not claimed by the
/// design note that proposed it: two or more passes do NOT spread
/// isotropically cell by cell. Two cells at the same hex distance from
/// the source can get different weights (the hex lattice's 6-neighbor
/// short-range walk isn't rotationally symmetric at low step counts,
/// only in the large-radius limit). Measured for the shipped
/// `share = 0.35`, radius 3: ring 2 (12 cells) spans 0.0078-0.0145, ring
/// 3 (18 cells) spans 0.00020-0.00060, same order of magnitude, not
/// equal. Rings ARE strictly ordered (every ring-3 cell's weight is
/// below every ring-2 cell's, which is below every ring-1 cell's): the
/// gradient the owner asked for is real, just not a perfect disc.
pub(crate) fn spread_precip_further(
    grid: &HexGrid,
    share: f32,
    extra_passes: u32,
    rain: &mut Vec<f32>,
    snow: &mut Vec<f32>,
    rain_tmp: &mut Vec<f32>,
    snow_tmp: &mut Vec<f32>,
) {
    if extra_passes == 0 {
        return;
    }
    let n = rain.len();
    rain_tmp.clear();
    rain_tmp.resize(n, 0.0);
    snow_tmp.clear();
    snow_tmp.resize(n, 0.0);
    for _ in 0..extra_passes {
        diffuse_precip_field(grid, share, rain, snow, rain_tmp, snow_tmp);
        // Swaps the `Vec`s themselves (buffer/len/cap only, no element
        // copy): after this, `rain`/`snow` hold the pass's fresh output
        // and `rain_tmp`/`snow_tmp` hold the now-stale input, ready to be
        // overwritten by the next pass (or left stale if this was the
        // last one — never read again by the caller).
        std::mem::swap(rain, rain_tmp);
        std::mem::swap(snow, snow_tmp);
    }
}

/// One application of the diffusion rule `D` (see
/// [`spread_precip_further`]'s doc) to an already-fallen rain/snow field:
/// the same isotropic self/neighbor split as [`gather_precip_deltas`],
/// but over a plain per-cell amount instead of a [`PrecipOutflow`] —
/// there is no source-side KK2000 term left to recompute past the first
/// pass, only redistribution.
fn diffuse_precip_field(
    grid: &HexGrid,
    share: f32,
    rain_in: &[f32],
    snow_in: &[f32],
    rain_out: &mut [f32],
    snow_out: &mut [f32],
) {
    // `share` is the same for every cell of this pass: the division is
    // hoisted out of the per-cell, per-neighbor loop below (one divide
    // per pass instead of 6 per cell — divides cost noticeably more than
    // multiplies, and this loop is the one that runs `extra_passes`
    // times over the whole grid, see `spread_precip_further`'s perf
    // note).
    let share_over_6 = share / 6.0;
    let self_share = 1.0 - share;
    for_each_chunk_mut2(rain_out, snow_out, |start, rain_chunk, snow_chunk| {
        for local in 0..rain_chunk.len() {
            let j = start + local;
            let neighbors = grid.neighbor_indices_toric(j);
            let mut rain = rain_in[j] * self_share;
            let mut snow = snow_in[j] * self_share;
            for &k in &neighbors {
                rain += rain_in[k] * share_over_6;
                snow += snow_in[k] * share_over_6;
            }
            rain_chunk[local] = rain;
            snow_chunk[local] = snow;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{
        AtmoScratch, AtmosphereParams, CLOUD_MIN_PRECIP, KK2000_NC_EXP, KK2000_QC_EXP, PrecipRates,
        diffuse_precip_field, kk2000_autoconv_over_hours, kk2000_drain_mm_per_hour,
        precip_amount_mm, precip_spread_passes, spread_precip_further, step_precipitation_into,
    };
    use crate::atmosphere::scaling::{precip_boosted_params, precip_runs_this_hour};
    use crate::climate::DayRecord;
    use crate::coord::HexCoord;
    use crate::grid::HexGrid;
    use proptest::prelude::*;

    // N_c = 50 (engine default), 1500 m layer: reproduces the real wiring.
    const LAYER_M: f32 = 1500.0;
    fn nc_pow() -> f32 {
        50.0_f32.powf(KK2000_NC_EXP)
    }

    /// The core of #50: at an absurdly high `cloud_water` the instantaneous
    /// KK2000 (super-linear) rate exceeds the stock over 1 h; the
    /// analytical integration must yield a drain **strictly < stock, with
    /// no clamp**.
    #[test]
    fn autoconv_never_exceeds_stock_even_for_huge_cloud() {
        let pow = nc_pow();
        for &cw in &[1.0_f32, 3.0, 10.0, 100.0, 1000.0] {
            let drained = kk2000_autoconv_over_hours(cw, LAYER_M, pow, 1.0);
            assert!(
                drained < cw,
                "drain {drained} must stay < stock {cw} by construction (#50)"
            );
            assert!(drained >= 0.0, "negative drain {drained} for cw={cw}");
            // The analytical form never drains more than the raw Euler rate.
            let euler = kk2000_drain_mm_per_hour(cw, LAYER_M, pow);
            assert!(
                drained <= euler,
                "cw={cw}: analytic {drained} > Euler {euler}"
            );
        }
    }

    /// Hourly `PrecipRates` at the engine defaults, gate open: what both
    /// paths build before calling [`precip_amount_mm`].
    fn hourly_defaults() -> AtmosphereParams {
        crate::atmosphere::scaling::scale_atmosphere_for_hourly_tick(&AtmosphereParams::default())
    }

    fn open_rates() -> PrecipRates {
        PrecipRates::new(&hourly_defaults(), 1.0, false)
    }

    /// **Sub-grid closure, the mechanism (coarse upper layer, step 2b).**
    /// The same grid-box mean `q_c` drains strictly MORE when the
    /// droplets are known to sit on a fraction of the box: KK2000 goes
    /// like `q^2.47`, so `f_c · drain(q_c / f_c)` grows as `f_c` shrinks.
    /// Ratios pinned as measured at the engine defaults (`N_c` = 50,
    /// 1500 m, dt = 1 h, `q_c` = 0.30 mm), 1/6 being the brief's own
    /// "a cloud on a sixth of the coarse cell".
    #[test]
    fn a_concentrated_cloud_drains_more_than_the_same_water_spread_flat() {
        let rates = open_rates();
        let q_c = 0.30_f32;
        let flat = precip_amount_mm(q_c, 1.0, 15.0, 0.0, &rates);
        let sixth = precip_amount_mm(q_c, 1.0 / 6.0, 15.0, 0.0, &rates);
        assert!(flat > 0.0, "the flat control must rain at all: {flat}");
        assert!(
            sixth > flat,
            "concentrating the same {q_c} mm on a sixth of the box must drain more: \
             {sixth} against {flat}"
        );
        // Measured 2026-09-06: 15.5x. The analytic factor is
        // f^(1-2.47) = 6^1.47 = 14.0 before the per-pass cap and the
        // KK2000 floor bend the curve; pinned loosely so it flags a
        // change of law, not a change of the third digit.
        let ratio = sixth / flat;
        assert!(
            (8.0..25.0).contains(&ratio),
            "the concentration factor moved: {ratio} (was 15.5 on 2026-09-06)"
        );
    }

    /// **Sub-grid closure, conservation without a clamp.** For ANY
    /// fraction down to one fine cell out of a full coarse cell, the sheet
    /// stays at or below the grid-box stock: `drain(q_in) <= q_in` is
    /// guaranteed by the analytic KK2000 form, and `f_c * q_in = q_c`.
    /// Nothing in `precip_amount_mm` floors `f_c`, and this is why nothing
    /// needs to.
    #[test]
    fn any_cloud_fraction_keeps_the_sheet_under_the_grid_box_stock() {
        let rates = open_rates();
        for &q_c in &[0.06_f32, 0.2, 1.0, 5.0, 50.0] {
            for n in [1_u32, 2, 6, 16, 46, 56] {
                let f = 1.0 / f32::from(u16::try_from(n).unwrap());
                let sheet = precip_amount_mm(q_c, f, 15.0, 0.0, &rates);
                assert!(
                    sheet >= 0.0 && sheet <= q_c,
                    "q_c={q_c} f={f}: sheet {sheet} outside [0, q_c]"
                );
            }
        }
    }

    /// **The fine path pays nothing for the closure.** `cloud_fraction =
    /// 1.0` divides and multiplies by one, both exact in IEEE 754, so
    /// every column of the fine grid drains the bits it drained before
    /// step 2b — which is what keeps the golden state identical to `main`.
    #[test]
    fn unit_cloud_fraction_is_bit_identical_to_the_grid_box_drain() {
        let rates = open_rates();
        let pow = nc_pow();
        for &q_c in &[0.06_f32, 0.1, 0.5, 1.0, 4.0, 40.0] {
            let through_the_closure = precip_amount_mm(q_c, 1.0, 15.0, 0.0, &rates);
            // The pre-2b body, written out: floor on the grid-box stock,
            // KK2000 over the hour, per-pass cap.
            let params = hourly_defaults();
            let raw = if q_c <= CLOUD_MIN_PRECIP.max(params.precip_crit_mm) {
                0.0
            } else {
                kk2000_autoconv_over_hours(q_c, LAYER_M, pow, 1.0).min(params.max_precip_per_tick)
            };
            assert_eq!(
                through_the_closure.to_bits(),
                raw.to_bits(),
                "q_c={q_c}: {through_the_closure} against {raw}"
            );
        }
    }

    /// Proves the clamped regime exists: for a large cloud the Euler rate
    /// clearly exceeds the stock (the old `.min(cloud_water)` used to
    /// bite), but the analytical form stays bounded, so the previous test
    /// is testing something real.
    #[test]
    fn euler_overshoots_stock_where_analytic_saves_it() {
        let pow = nc_pow();
        for &cw in &[10.0_f32, 100.0, 1000.0] {
            let euler = kk2000_drain_mm_per_hour(cw, LAYER_M, pow);
            assert!(euler > cw, "cw={cw}: Euler {euler} should exceed the stock");
            assert!(kk2000_autoconv_over_hours(cw, LAYER_M, pow, 1.0) < cw);
        }
    }

    /// At small drains (just above the floor) the analytical form matches
    /// Euler to within a few per mille: the drizzle regime is not
    /// disturbed.
    #[test]
    fn autoconv_matches_euler_for_small_drain() {
        let pow = nc_pow();
        let cw = 0.06_f32;
        let rate = kk2000_drain_mm_per_hour(cw, LAYER_M, pow);
        let drained = kk2000_autoconv_over_hours(cw, LAYER_M, pow, 1.0);
        assert!(
            rate > 0.0 && rate < cw,
            "small drain regime expected (rate={rate})"
        );
        let rel = (drained - rate).abs() / rate;
        assert!(rel < 0.05, "Euler/analytic gap {rel} > 5% in small drain");
    }

    /// Monotonicity: more cloud ⇒ more drain (the integral preserves
    /// KK2000's super-linear sense).
    #[test]
    fn autoconv_monotone_in_cloud_water() {
        let pow = nc_pow();
        let mut prev = 0.0_f32;
        for &cw in &[0.1_f32, 0.3, 1.0, 3.0, 10.0] {
            let drained = kk2000_autoconv_over_hours(cw, LAYER_M, pow, 1.0);
            assert!(
                drained > prev,
                "non-increasing drain at cw={cw} ({drained} <= {prev})"
            );
            prev = drained;
        }
    }

    /// Below the numerical floor: no drain (same guard as before).
    #[test]
    fn autoconv_zero_below_floor() {
        let pow = nc_pow();
        // to_bits: exact zero, not "close to zero" (below the floor the
        // function returns the literal 0.0).
        assert_eq!(
            kk2000_autoconv_over_hours(CLOUD_MIN_PRECIP, LAYER_M, pow, 1.0).to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            kk2000_autoconv_over_hours(0.0, LAYER_M, pow, 1.0).to_bits(),
            0.0f32.to_bits()
        );
    }

    /// r250 perf effort: mass conservation of `step_precipitation_into`'s
    /// scatter -> gather split (`PrecipOutflow`), radius-2. The isotropic
    /// neighbor share (like `step_cloud_diffusion`) lands equally on all
    /// 6 neighbors, and every drop of `cloud_water` lost is accounted
    /// for in `water_level` AND in `events` (same total, since both are
    /// filled from the same gathered `precip_water_delta`).
    #[test]
    fn precipitation_gather_conserves_and_splits_share_equally() {
        let mut grid = HexGrid::from_radius(2);
        let center = HexCoord::new(0, 0);
        {
            let c = grid.get_mut(center).unwrap();
            c.cloud_water = 5.0;
            c.temperature = 15.0;
        }
        let params = AtmosphereParams::default();
        let mut gate_open = true;
        let mut scratch = AtmoScratch::new(grid.len());
        let mut events = vec![DayRecord::default(); grid.len()];
        let before_cloud: f32 = grid.iter().map(|(_, c)| c.cloud_water).sum();

        let mut next = grid.clone();
        step_precipitation_into(
            &mut next,
            &params,
            1.0,
            &mut gate_open,
            &mut events,
            &mut scratch,
        );

        let after_cloud: f32 = next.iter().map(|(_, c)| c.cloud_water).sum();
        let after_rain: f32 = next.iter().map(|(_, c)| c.water_level).sum();
        let cloud_lost = before_cloud - after_cloud;
        assert!(
            cloud_lost > 0.0,
            "setup must actually precipitate something, lost={cloud_lost}"
        );
        assert!(
            (cloud_lost - after_rain).abs() < 1e-4,
            "cloud lost ({cloud_lost}) must equal rain produced ({after_rain})"
        );
        let events_rain: f32 = events.iter().map(|e| e.rain).sum();
        assert!(
            (events_rain - after_rain).abs() < 1e-5,
            "events must carry the same total as water_level: {events_rain} vs {after_rain}"
        );

        let neighbor_rain: Vec<f32> = center
            .neighbors()
            .into_iter()
            .map(|n| next.get(n).unwrap().water_level)
            .collect();
        for &r in &neighbor_rain {
            assert!(r > 0.0, "every neighbor must receive a share");
            assert!(
                (r - neighbor_rain[0]).abs() < 1e-6,
                "the uniform share must land equally on every neighbor: {neighbor_rain:?}"
            );
        }
    }

    /// The historical (pre-cadence-switch) closed form, transcribed
    /// literally from the `dt = 1 h` implementation. The reference for
    /// the bit-identity of the default path: if the general
    /// `dt_hours` form ever stops reproducing it exactly, the ablation
    /// switch has changed the shipped physics.
    fn historical_autoconv_over_one_hour(
        cloud_water_mm: f32,
        layer_thickness_m: f32,
        droplet_count_pow: f32,
    ) -> f32 {
        let rate0 = kk2000_drain_mm_per_hour(cloud_water_mm, layer_thickness_m, droplet_count_pow);
        if rate0 <= 0.0 {
            return 0.0;
        }
        let exp_m1 = KK2000_QC_EXP - 1.0;
        let x = exp_m1 * rate0 / cloud_water_mm;
        let q_end = cloud_water_mm * (1.0 + x).powf(-1.0 / exp_m1);
        (cloud_water_mm - q_end).max(0.0)
    }

    /// Bit-identity of the default cadence: `dt_hours = 1` must return
    /// the exact same bits as the historical form, over the whole useful
    /// range of stocks (`× 1.0` is exact in IEEE 754, this pins it).
    #[test]
    fn autoconv_dt1_is_bit_identical_to_the_historical_form() {
        let pow = nc_pow();
        for &cw in &[
            0.0_f32,
            0.049,
            CLOUD_MIN_PRECIP,
            0.051,
            0.06,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
            100.0,
            1000.0,
        ] {
            let general = kk2000_autoconv_over_hours(cw, LAYER_M, pow, 1.0);
            let historical = historical_autoconv_over_one_hour(cw, LAYER_M, pow);
            assert_eq!(
                general.to_bits(),
                historical.to_bits(),
                "cw={cw}: dt=1 must be bit-identical ({general} vs {historical})"
            );
        }
    }

    /// The whole point of integrating the ODE over `dt = N` instead of
    /// scaling the hourly rate: one `dt = 3` pass drains what three
    /// chained hourly passes drain. The closed form is exact, so the gap
    /// is f32 rounding only (and it stays sub-per-mille even at the
    /// stocks where the drain is a large fraction of the stock).
    #[test]
    fn autoconv_dt3_equals_three_chained_hourly_steps() {
        let pow = nc_pow();
        for &cw in &[0.06_f32, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0] {
            let one_pass = kk2000_autoconv_over_hours(cw, LAYER_M, pow, 3.0);
            let mut q = cw;
            for _ in 0..3 {
                q -= kk2000_autoconv_over_hours(q, LAYER_M, pow, 1.0);
            }
            let chained = cw - q;
            let rel = (one_pass - chained).abs() / chained;
            assert!(
                rel < 1e-3,
                "cw={cw}: dt=3 pass drained {one_pass}, three hourly passes {chained} (rel {rel})"
            );
        }
    }

    /// Naive "hourly rate × N" is the trap this integration avoids: at a
    /// heavy stock it overshoots the chained truth by a wide margin,
    /// while the closed form tracks it. Proves the previous test is
    /// testing something real.
    #[test]
    fn autoconv_dt3_beats_scaling_the_hourly_drain() {
        let pow = nc_pow();
        let cw = 3.0_f32;
        let mut q = cw;
        for _ in 0..3 {
            q -= kk2000_autoconv_over_hours(q, LAYER_M, pow, 1.0);
        }
        let chained = cw - q;
        let naive = kk2000_autoconv_over_hours(cw, LAYER_M, pow, 1.0) * 3.0;
        let integrated = kk2000_autoconv_over_hours(cw, LAYER_M, pow, 3.0);
        assert!(
            naive > chained * 1.2,
            "the naive rate×3 must visibly overshoot: {naive} vs {chained}"
        );
        assert!(
            (integrated - chained).abs() < (naive - chained).abs(),
            "the closed form {integrated} must be closer to {chained} than the naive {naive}"
        );
    }

    /// Monotone in `dt`: a longer pass drains more, and never more than
    /// the stock (the analytical solution is bounded for any `dt`).
    #[test]
    fn autoconv_monotone_in_dt() {
        let pow = nc_pow();
        let cw = 2.0_f32;
        let mut prev = 0.0_f32;
        for &dt in &[0.5_f32, 1.0, 2.0, 3.0, 6.0, 24.0] {
            let drained = kk2000_autoconv_over_hours(cw, LAYER_M, pow, dt);
            assert!(
                drained > prev,
                "non-increasing drain at dt={dt} ({drained} <= {prev})"
            );
            assert!(
                drained < cw,
                "dt={dt}: drain {drained} must stay < stock {cw}"
            );
            prev = drained;
        }
    }

    /// Conservation through one boosted pass at cadence 3, radius 2 (the
    /// dispersion needs real neighbors): every mm of `cloud_water` lost
    /// lands in `water_level` + `snow_level`, and `events` carries the
    /// same total. The cadence changes how much falls, never whether the
    /// terrarium is closed.
    #[test]
    fn boosted_pass_at_sub3_conserves_water() {
        let mut grid = HexGrid::from_radius(2);
        let center = HexCoord::new(0, 0);
        {
            let c = grid.get_mut(center).unwrap();
            c.cloud_water = 5.0;
            c.temperature = 15.0;
        }
        let params = precip_boosted_params(&AtmosphereParams::default(), 3);
        let mut gate_open = true;
        let mut scratch = AtmoScratch::new(grid.len());
        let mut events = vec![DayRecord::default(); grid.len()];
        let before: f32 = grid
            .iter()
            .map(|(_, c)| c.cloud_water + c.water_level + c.snow_level)
            .sum();

        let mut next = grid.clone();
        step_precipitation_into(
            &mut next,
            &params,
            3.0,
            &mut gate_open,
            &mut events,
            &mut scratch,
        );

        let after: f32 = next
            .iter()
            .map(|(_, c)| c.cloud_water + c.water_level + c.snow_level)
            .sum();
        assert!(
            (after - before).abs() < 1e-4,
            "cloud + water + snow must be conserved: {before} -> {after}"
        );
        let events_total: f32 = events.iter().map(|e| e.rain + e.snow).sum();
        let fallen: f32 = next
            .iter()
            .map(|(_, c)| c.water_level + c.snow_level)
            .sum::<f32>();
        assert!(fallen > 0.0, "the setup must actually precipitate");
        assert!(
            (events_total - fallen).abs() < 1e-4,
            "events must carry the same total as what fell: {events_total} vs {fallen}"
        );
    }

    /// A cadence-3 pass drains more than a cadence-1 pass from the same
    /// stock (it covers three hours), but stays below three times it: the
    /// super-linear rate decays as the cloud empties.
    #[test]
    fn boosted_pass_at_sub3_rains_more_than_hourly_but_less_than_triple() {
        let run = |sub: u16, dt: f32| -> f32 {
            let mut grid = HexGrid::from_radius(2);
            {
                let c = grid.get_mut(HexCoord::new(0, 0)).unwrap();
                c.cloud_water = 5.0;
                c.temperature = 15.0;
            }
            let params = precip_boosted_params(&AtmosphereParams::default(), sub);
            let mut gate_open = true;
            let mut scratch = AtmoScratch::new(grid.len());
            let mut events = vec![DayRecord::default(); grid.len()];
            let mut next = grid.clone();
            step_precipitation_into(
                &mut next,
                &params,
                dt,
                &mut gate_open,
                &mut events,
                &mut scratch,
            );
            events.iter().map(|e| e.rain + e.snow).sum()
        };
        let hourly = run(1, 1.0);
        let boosted = run(3, 3.0);
        assert!(hourly > 0.0, "the hourly setup must precipitate");
        assert!(
            boosted > hourly,
            "a 3 h pass must drain more than a 1 h pass: {boosted} vs {hourly}"
        );
        assert!(
            boosted < hourly * 3.0,
            "the drain decays as the cloud empties: {boosted} >= 3 x {hourly}"
        );
    }

    /// The orchestrator's contract, replayed over hours 0..6 at cadence
    /// 3 with the real helpers: `events` is zeroed BEFORE the gate, so a
    /// skipped hour reports zero precipitation and leaves `next`
    /// untouched. `step_snow` reads that map next tick as "rain of the
    /// previous tick": a stale map would advect the same rain heat
    /// three times.
    #[test]
    fn a_skipped_hour_reports_zero_and_touches_nothing() {
        let sub = 3_u16;
        let mut grid = HexGrid::from_radius(2);
        {
            let c = grid.get_mut(HexCoord::new(0, 0)).unwrap();
            c.cloud_water = 5.0;
            c.temperature = 15.0;
        }
        let params = precip_boosted_params(&AtmosphereParams::default(), sub);
        let mut gate_open = true;
        let mut scratch = AtmoScratch::new(grid.len());
        let mut events = vec![DayRecord::default(); grid.len()];
        let mut ran_at_least_once = false;

        for hour in 0u64..6 {
            let before = serde_json::to_value(grid.cells_slice()).expect("cells serialize");
            // Same two lines as `step_atmosphere_into`, in the same
            // order: zero first, gate second.
            events.fill(DayRecord::default());
            let runs = precip_runs_this_hour(hour, sub);
            if runs {
                step_precipitation_into(
                    &mut grid,
                    &params,
                    f32::from(sub),
                    &mut gate_open,
                    &mut events,
                    &mut scratch,
                );
            }
            let total: f32 = events.iter().map(|e| e.rain + e.snow).sum();
            let after = serde_json::to_value(grid.cells_slice()).expect("cells serialize");
            if runs {
                assert!(total > 0.0, "hour {hour} runs, it must precipitate");
                ran_at_least_once = true;
            } else {
                assert_eq!(total.to_bits(), 0.0f32.to_bits(), "hour {hour} is skipped");
                assert!(
                    events.iter().all(|e| e.rain == 0.0 && e.snow == 0.0),
                    "hour {hour} is skipped: every event must be zero"
                );
                assert_eq!(
                    before, after,
                    "hour {hour} is skipped: `next` must not move"
                );
            }
        }
        assert!(ran_at_least_once, "hours 0..6 must contain a run hour");
    }

    // ---- L3: rain footprint spread to radius R (rain-regime work, 2026-09-06) ----

    /// Rounding contract of `precip_spread_radius` -> integer passes:
    /// nearest, floored at 1 (never fewer passes than the historical
    /// single-ring footprint, whatever a hot-reloaded or malformed value
    /// throws at it).
    #[test]
    fn spread_passes_rounds_to_nearest_and_floors_at_one() {
        assert_eq!(precip_spread_passes(0.0), 1);
        assert_eq!(precip_spread_passes(0.4), 1);
        assert_eq!(precip_spread_passes(1.0), 1);
        assert_eq!(precip_spread_passes(1.4), 1);
        assert_eq!(precip_spread_passes(1.6), 2);
        assert_eq!(precip_spread_passes(3.0), 3);
        assert_eq!(
            precip_spread_passes(-5.0),
            1,
            "negative radius still floors at 1"
        );
    }

    /// `precip_spread_radius = 1` (`extra_passes = 0`) must reproduce the
    /// historical footprint bit for bit. The only code this feature adds
    /// to the pipeline is `spread_precip_further`, so proving IT is a
    /// true no-op at `extra_passes = 0` — untouched buffers, same bits in
    /// as out, no allocation — proves the whole pipeline is unchanged at
    /// `R = 1`: everything upstream (`fill_precip_outflow`,
    /// `gather_precip_deltas`) has no dependency on the radius at all.
    #[test]
    fn radius_one_is_bit_identical_to_legacy() {
        assert_eq!(
            precip_spread_passes(1.0).saturating_sub(1),
            0,
            "R=1 must add zero extra passes"
        );
        let grid = HexGrid::from_radius(2);
        let mut rain = vec![1.0_f32, 0.0, 3.25, 0.0, -0.0];
        let mut snow = vec![9.0_f32, 8.5, 0.0, 0.0, 2.0];
        let before_rain: Vec<u32> = rain.iter().map(|v| v.to_bits()).collect();
        let before_snow: Vec<u32> = snow.iter().map(|v| v.to_bits()).collect();
        let mut rain_tmp = Vec::new();
        let mut snow_tmp = Vec::new();

        spread_precip_further(
            &grid,
            0.35,
            0,
            &mut rain,
            &mut snow,
            &mut rain_tmp,
            &mut snow_tmp,
        );

        assert_eq!(
            rain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before_rain
        );
        assert_eq!(
            snow.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before_snow
        );
        assert!(
            rain_tmp.is_empty() && snow_tmp.is_empty(),
            "extra_passes=0 must not even allocate the ping-pong buffers"
        );
    }

    /// One diffusion pass (`share`-weighted self/neighbor split) applied
    /// `passes` times to a point mass wets exactly the cells within hex
    /// distance `passes` of the source and none beyond: `D` (see
    /// `spread_precip_further`'s doc) moves mass at most one hex per
    /// application, so `D^passes` of a point mass is supported on a
    /// radius-`passes` disc. Radius-8 torus (217 cells): its wrap
    /// distance (`coord::torus_lattice_vectors(8)`'s shortest vector) is
    /// 17 hexes, far past the `passes <= 4` tested here, so nothing wraps
    /// back onto the disc from the far side of the torus.
    #[test]
    fn single_source_wets_exactly_the_disc_of_its_pass_count() {
        let grid = HexGrid::from_radius(8);
        let n = grid.len();
        let center = HexCoord::new(0, 0);
        let source = grid.index_of(center).expect("center exists");
        let share = 0.35_f32;

        for &passes in &[1u32, 2, 3, 4] {
            let mut field = vec![0.0_f32; n];
            field[source] = 1.0;
            let zero = vec![0.0_f32; n];
            let mut scratch_zero = vec![0.0_f32; n];
            let mut tmp = vec![0.0_f32; n];
            for _ in 0..passes {
                diffuse_precip_field(&grid, share, &field, &zero, &mut tmp, &mut scratch_zero);
                std::mem::swap(&mut field, &mut tmp);
            }
            let passes_i32 = i32::try_from(passes).expect("passes fits i32");
            for (idx, &coord) in grid.coords_slice().iter().enumerate() {
                let d = center.distance(coord);
                if d > passes_i32 {
                    assert_eq!(
                        field[idx].to_bits(),
                        0.0f32.to_bits(),
                        "distance {d} > passes {passes}: cell {idx} must stay dry, got {}",
                        field[idx]
                    );
                } else {
                    assert!(
                        field[idx] > 0.0,
                        "distance {d} <= passes {passes}: cell {idx} must be wet"
                    );
                }
            }
        }
    }

    /// Delivered amounts are non-increasing with ring distance: every
    /// ring-3 cell's weight is below every ring-2 cell's, itself below
    /// every ring-1 cell's, itself below the source's own ring-0 share —
    /// the gradient the owner asked for. Ring 0 (one cell) and ring 1
    /// (the immediate neighbors) ARE exactly isotropic: a single
    /// diffusion pass sends the identical `share/6` to all six.
    ///
    /// **Rings >= 2 are NOT exactly isotropic — a finding from
    /// implementing this, not a claim of the brief that proposed it.**
    /// The hex lattice's short-range 6-neighbor walk isn't rotationally
    /// symmetric at low step counts (only in the large-radius limit), so
    /// two cells at the same hex distance can carry different weights.
    /// Pinned here with the measured shape (share=0.35, passes=3) instead
    /// of asserted away: same order of magnitude within a ring, ratio
    /// bounded, not equal.
    #[test]
    fn footprint_gradient_decreases_with_ring_distance() {
        let grid = HexGrid::from_radius(8);
        let n = grid.len();
        let center = HexCoord::new(0, 0);
        let source = grid.index_of(center).expect("center exists");
        let share = 0.35_f32;
        let passes = 3u32;

        let mut field = vec![0.0_f32; n];
        field[source] = 1.0;
        let zero = vec![0.0_f32; n];
        let mut scratch_zero = vec![0.0_f32; n];
        let mut tmp = vec![0.0_f32; n];
        for _ in 0..passes {
            diffuse_precip_field(&grid, share, &field, &zero, &mut tmp, &mut scratch_zero);
            std::mem::swap(&mut field, &mut tmp);
        }

        let mut rings: [Vec<f32>; 4] = Default::default();
        for (idx, &coord) in grid.coords_slice().iter().enumerate() {
            let d = center.distance(coord);
            if (0..=3).contains(&d) {
                let ring = usize::try_from(d).expect("d in 0..=3");
                rings[ring].push(field[idx]);
            }
        }

        let min_max = |v: &[f32]| -> (f32, f32) {
            (
                v.iter().copied().fold(f32::INFINITY, f32::min),
                v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            )
        };

        assert_eq!(rings[0].len(), 1, "ring 0 is the source cell alone");
        let (r1_min, r1_max) = min_max(&rings[1]);
        assert_eq!(rings[1].len(), 6, "ring 1 has 6 cells");
        assert!(
            (r1_max - r1_min).abs() < 1e-6,
            "ring 1 must be exactly isotropic (single pass, uniform by \
             construction): {:?}",
            rings[1]
        );

        let (r2_min, r2_max) = min_max(&rings[2]);
        assert_eq!(rings[2].len(), 12, "ring 2 has 12 cells");
        assert!(r2_min > 0.0, "every ring-2 cell must be wet");
        assert!(
            r2_max / r2_min < 3.0,
            "ring 2 anisotropy grew past the measured bound: {r2_min}..{r2_max}"
        );

        let (r3_min, r3_max) = min_max(&rings[3]);
        assert_eq!(rings[3].len(), 18, "ring 3 has 18 cells");
        assert!(r3_min > 0.0, "every ring-3 cell must be wet");
        assert!(
            r3_max / r3_min < 5.0,
            "ring 3 anisotropy grew past the measured bound: {r3_min}..{r3_max}"
        );

        assert!(
            r1_min > r2_max,
            "ring 1 ({r1_min}) must dominate ring 2 ({r2_max})"
        );
        assert!(
            r2_min > r3_max,
            "ring 2 ({r2_min}) must dominate ring 3 ({r3_max})"
        );
    }

    proptest! {
        /// Mass conservation of the diffusion (rain and snow channels
        /// independently), across a swept radius (including torus sizes
        /// small enough that a `passes`-step walk wraps and
        /// self-interacts), share and pass count. `D` (see
        /// `spread_precip_further`'s doc) is doubly stochastic for any
        /// number of applications, on any regular-degree-6 toric graph
        /// (wraparound or not): total mass in must equal total mass out.
        #[test]
        fn diffusion_conserves_mass_on_any_torus_share_and_pass_count(
            radius in 0i32..=6,
            share in 0.0f32..=1.0,
            extra_passes in 0u32..=5,
            seed in 0u32..=10_000u32,
        ) {
            let grid = HexGrid::from_radius(radius);
            let n = grid.len();
            let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            let mut next_unit = move || -> f32 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from(u16::try_from(state >> 16).unwrap_or(0)) / 65_535.0
            };
            let mut rain: Vec<f32> = (0..n).map(|_| next_unit() * 10.0).collect();
            let mut snow: Vec<f32> = (0..n).map(|_| next_unit() * 10.0).collect();
            let before_rain: f64 = rain.iter().map(|&v| f64::from(v)).sum();
            let before_snow: f64 = snow.iter().map(|&v| f64::from(v)).sum();
            let mut rain_tmp = Vec::new();
            let mut snow_tmp = Vec::new();

            spread_precip_further(
                &grid, share, extra_passes, &mut rain, &mut snow, &mut rain_tmp, &mut snow_tmp,
            );

            let after_rain: f64 = rain.iter().map(|&v| f64::from(v)).sum();
            let after_snow: f64 = snow.iter().map(|&v| f64::from(v)).sum();
            let tol_rain = (before_rain.abs() * 1e-4).max(1e-3);
            let tol_snow = (before_snow.abs() * 1e-4).max(1e-3);
            prop_assert!(
                (after_rain - before_rain).abs() < tol_rain,
                "rain mass drifted: {before_rain} -> {after_rain} (radius={radius} \
                 share={share} passes={extra_passes})"
            );
            prop_assert!(
                (after_snow - before_snow).abs() < tol_snow,
                "snow mass drifted: {before_snow} -> {after_snow} (radius={radius} \
                 share={share} passes={extra_passes})"
            );
        }
    }
}
