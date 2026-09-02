use crate::grid::HexGrid;

use super::{AtmoScratch, AtmosphereParams, PrecipitationMap};

/// Precipitation: consumes `cloud_water` above the collision/coalescence
/// threshold. Surplus droplets fall as rain (T>=0) or snow (T<0).
///
/// Global gate with hysteresis: opens if `mean(cloud_water)` > gate,
/// closes when mean < gate × 0.75. Except snow (T < 0°C), always allowed
/// to preserve winter snowpack accumulation.
fn update_precip_gate(gate_open: &mut bool, params: &AtmosphereParams, grid: &HexGrid) {
    if params.global_precip_gate > 0.0 {
        let cells = grid.cells_slice();
        let n = cells.len();
        if n > 0 {
            let n_f = f32::from(u16::try_from(n).expect("cell count fits u16"));
            let mean_cloud: f32 = cells.iter().map(|c| c.cloud_water).sum::<f32>() / n_f;
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

/// `cloud_water` (mm) converted to rain over **one hour** by KK2000
/// autoconversion, via **analytical integration** of `dq/dt = -C·q^α`
/// (α = 2.47, #50).
///
/// KK2000 is super-linear: at high `q` the instantaneous rate `C·q^α`
/// naively integrated by Euler (`rate × 1 h`) exceeds the available
/// stock, hence the old corrective `.min(cloud_water)` (anti-pattern #4:
/// production silently bounded by availability, which would mask a
/// `cloud_water` drift coming from elsewhere). The exact solution of
/// `dq/dt = -C·q^α` (α ≠ 1) is monotonically decreasing toward 0:
///
/// ```text
///   q(t) = q₀ · (1 + (α−1)·(rate₀/q₀)·t)^(−1/(α−1)),   rate₀ = C·q₀^α
/// ```
///
/// so the drain `q₀ − q(1 h)` is **≤ q₀ by construction**, with no
/// clamp. At small drains (`x → 0`) it converges to Euler (`≈ rate₀ ·
/// dt`): the drizzle regime is not disturbed, only heavy showers are
/// smoothed (exponential tail instead of a truncated purge).
#[must_use]
fn kk2000_autoconv_over_hour(
    cloud_water_mm: f32,
    layer_thickness_m: f32,
    droplet_count_pow: f32,
) -> f32 {
    // Instantaneous rate C·q₀^α (mm/h). 0 below the floor or N_c disabled.
    let rate0 = kk2000_drain_mm_per_hour(cloud_water_mm, layer_thickness_m, droplet_count_pow);
    if rate0 <= 0.0 {
        return 0.0;
    }
    let exp_m1 = KK2000_QC_EXP - 1.0;
    // x = (α−1)·rate₀/q₀ · dt (dt = 1 h), dimensionless.
    // rate₀ = C·q₀^α ⇒ (α−1)·C·q₀^(α−1) = (α−1)·rate₀/q₀: this avoids
    // reconstructing C.
    let x = exp_m1 * rate0 / cloud_water_mm;
    let q_end = cloud_water_mm * (1.0 + x).powf(-1.0 / exp_m1);
    // q_end ∈ (0, q₀] ⇒ drain ∈ [0, q₀). The max(0) only covers f32
    // rounding, it is not a physical safeguard (the analytical solution
    // already guarantees ≤ stock).
    (cloud_water_mm - q_end).max(0.0)
}

pub(crate) fn step_precipitation_into(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    gate_open: &mut bool,
    events: &mut PrecipitationMap,
    scratch: &mut AtmoScratch,
) {
    update_precip_gate(gate_open, params, next);
    let gate_closed = !*gate_open;

    // N_c^b precomputed once per tick: constant parameter, the per-cell
    // powf was pure recomputation (#88). 0.0 = drain disabled (N_c <= 0),
    // same guard as in kk2000_drain_mm_per_hour before the precompute.
    let droplet_count_pow = if params.kk2000_droplet_count > 0.0 {
        params.kk2000_droplet_count.powf(KK2000_NC_EXP)
    } else {
        0.0
    };

    let n = next.len();
    let cloud_delta = &mut scratch.precip_cloud_delta;
    let water_delta = &mut scratch.precip_water_delta;
    let snow_delta = &mut scratch.precip_snow_delta;
    cloud_delta.clear();
    cloud_delta.resize(n, 0.0);
    water_delta.clear();
    water_delta.resize(n, 0.0);
    snow_delta.clear();
    snow_delta.resize(n, 0.0);

    // Convective inhibition (ex-design A, #69): below the critical mass
    // the cloud builds up and travels, it does not precipitate. Default
    // 0.0 → only CLOUD_MIN_PRECIP applies.
    let precip_floor = CLOUD_MIN_PRECIP.max(params.precip_crit_mm);
    // Updraft trigger (synoptic Phase 3): precip factor ∝ vertical
    // velocity (convergence + orographic). Default w_ref=0 → factor=1
    // everywhere.
    let w_ref = params.updraft_ref_ms;
    let w_floor = params.updraft_floor;
    let updraft = &scratch.convergence;

    {
        let next_cells = next.cells_slice();
        for i in 0..n {
            let nc = &next_cells[i];

            if nc.cloud_water <= precip_floor {
                continue;
            }
            let precip_allowed = !gate_closed || nc.temperature < 0.0;
            if !precip_allowed {
                continue;
            }

            // KK2000 autoconversion: super-linear drain in cloud_water^2.47.
            // Small clouds: near-zero drain (they have time to travel).
            // Large clouds: fast drain (natural purge of cumulonimbus).
            // Analytically integrated over the hour (#50): the drain is
            // ≤ stock by construction, no more `.min(cloud_water)`
            // conservation clamp.
            let drained = kk2000_autoconv_over_hour(
                nc.cloud_water,
                params.upper_layer_altitude_m,
                droplet_count_pow,
            );
            // Microphysical cap: even very heavily loaded, a cloud cannot
            // dump more than a certain volume per tick (bounded fall
            // speed). Spreads heavy showers over several ticks. This is a
            // physical cap distinct from conservation; `drained ≤
            // cloud_water` is already guaranteed, this `.min` no longer
            // masks anything.
            let mut amount = if params.max_precip_per_tick > 0.0 {
                drained.min(params.max_precip_per_tick)
            } else {
                drained
            };
            // Modulation by updraft: rain where air rises (front, windward
            // slope), dry under subsidence. Unprecipitated water stays in
            // cloud_water, so it accumulates and travels until an updraft.
            if w_ref > 0.0 {
                let f = (w_floor + updraft[i] / w_ref).clamp(0.0, 1.0);
                amount *= f;
            }
            cloud_delta[i] -= amount;

            // Spatial dispersion: a fraction `precip_neighbor_share` of the
            // rain falls on the neighbors (mixed air), the rest on the
            // source cell.
            let share = params.precip_neighbor_share.clamp(0.0, 1.0);
            let self_amount = amount * (1.0 - share);
            let is_snow = nc.temperature < 0.0;
            if is_snow {
                snow_delta[i] += self_amount;
                events[i].snow += self_amount;
            } else {
                water_delta[i] += self_amount;
                events[i].rain += self_amount;
            }

            if share > 0.0 {
                // Dispersion over 6 toric neighbors: the connected hex
                // grid guarantees 6 valid indices (antipode via
                // wrap_target at the edge).
                let neighbor_indices = next.neighbor_indices_toric(i);
                let share_each = (amount * share) / 6.0;
                for &ni in &neighbor_indices {
                    if is_snow {
                        snow_delta[ni] += share_each;
                        events[ni].snow += share_each;
                    } else {
                        water_delta[ni] += share_each;
                        events[ni].rain += share_each;
                    }
                }
            }
        }
    }

    // Pass 2: apply the deltas.
    for (i, nc) in next.cells_slice_mut().iter_mut().enumerate() {
        if cloud_delta[i] == 0.0 && water_delta[i] == 0.0 && snow_delta[i] == 0.0 {
            continue;
        }
        nc.cloud_water = (nc.cloud_water + cloud_delta[i]).max(0.0);
        nc.water_level = (nc.water_level + water_delta[i]).max(0.0);
        nc.snow_level = (nc.snow_level + snow_delta[i]).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CLOUD_MIN_PRECIP, KK2000_NC_EXP, kk2000_autoconv_over_hour, kk2000_drain_mm_per_hour,
    };

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
            let drained = kk2000_autoconv_over_hour(cw, LAYER_M, pow);
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
            assert!(kk2000_autoconv_over_hour(cw, LAYER_M, pow) < cw);
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
        let drained = kk2000_autoconv_over_hour(cw, LAYER_M, pow);
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
            let drained = kk2000_autoconv_over_hour(cw, LAYER_M, pow);
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
            kk2000_autoconv_over_hour(CLOUD_MIN_PRECIP, LAYER_M, pow).to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            kk2000_autoconv_over_hour(0.0, LAYER_M, pow).to_bits(),
            0.0f32.to_bits()
        );
    }
}
