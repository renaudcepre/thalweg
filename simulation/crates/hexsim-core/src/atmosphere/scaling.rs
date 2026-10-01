use crate::ablation::Ablation;
use crate::time::TICKS_PER_DAY_F32;
use crate::wind::WindParams;

use super::AtmosphereParams;

/// Subsampling of horizontal transport passes (humidity surface+upper
/// advection, cloud advection and diffusion). These passes represent slow
/// transport that doesn't need hourly diurnal cycle resolution: run only
/// 1 hour per `N`, with rates scaled x N (daily transport ~ conserved).
///
/// `N=3` validated by A/B at radius 30 over 3 years (ablation #opt):
/// -24% engine cost (76 to 58 ms/day) for negligible climate drift, only
/// hillside rain drops ~5% (112 to 106 j/year medians), plain/mid+high
/// elevations/mountain clouds/lapse/drift unchanged. `N=4` gave -28% but
/// hillside drift grew; `N=3` is the sweet spot.
pub(crate) const TRANSPORT_SUBSAMPLE_HOURS: u16 = 3;

/// Effective value: `TRANSPORT_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_TRANSPORT_SUBSAMPLE` (useful for parametric A/B without
/// recompile). Delegates to [`Ablation::effective`], which reads the
/// environment once for the whole process.
pub(crate) fn transport_subsample() -> u16 {
    Ablation::effective().transport_subsample
}

/// Subsampling of the orographic convection pump (`step_orographic_
/// convection`), same shape as [`TRANSPORT_SUBSAMPLE_HOURS`]: run it only
/// 1 hour per `N`, with `orographic_lift_coef` scaled x`N`. Measured and
/// REFUTED for `N > 1`, twice.
///
/// First refusal (JOURNAL 2026-09-05, 3-seed climate bench, r30): the
/// per-pass cap `rate.clamp(0.0, 0.30)` then in `fill_oro_outflow` already
/// bound 64-78 % of the pumping cell-passes above 300 m at `N=1`, and
/// 93-100 % once the coefficient was tripled, so "daily transport ~
/// conserved" could not hold: the pump moved LESS per day at `N=2` or
/// `3`, rain days above 1500 m collapsed (91 -> 0-29), mountain cloud
/// cover -37 to -68 %, and `phys_ubac_not_a_rain_attractor` went red.
///
/// Re-measured after #156 removed that cap, which is exactly what this
/// switch was kept for. The boost is now analytically exact for the drain
/// — `1 − exp(−N·x)` IS the `N`-hour solution of the same ODE — and the
/// damage drops by half to three quarters, but it does not vanish and
/// keeps the same sign on the three seeds: rain days above 1500 m
/// 107/115/128 -> 87/102/116 (legacy law on the same run: 92/104/118 ->
/// 55/72/95), 800-1500 m -8 to -16 days, mountain cloud cover -1 to
/// -9 % (legacy -22 to -29 %). What the exact boost cannot restore is the
/// two skipped hours of freshly evaporated vapour, and a 3x pulse meeting
/// the same per-destination LCL deficit that three hourly pulses did not
/// saturate. So `1` (hourly, historical behavior) stays the shipped
/// default; the switch stays as the instrument, never as a perf lever on
/// its own (-11 % of the r250 tick at `N=3` on the 4-vCPU VM, for a
/// mountain climate still measurably drier).
pub(crate) const ORO_SUBSAMPLE_HOURS: u16 = 1;

/// Effective value: `ORO_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_ORO_SUBSAMPLE` (parametric A/B without recompile, same pattern
/// as [`transport_subsample`]). Delegates to [`Ablation::effective`].
pub(crate) fn oro_subsample() -> u16 {
    Ablation::effective().oro_subsample
}

/// Pure predicate: does the orographic pump run on this hour, at cadence
/// `sub`? Same shape as the transport gate (`on_transport_tick` in
/// `atmosphere::mod`), factored out here so the cadence itself is testable
/// without going through [`Ablation::effective`] (a process-wide
/// `OnceLock`, cf `ablation.rs`).
#[must_use]
pub(crate) fn oro_runs_this_hour(hour_tick: u64, sub: u16) -> bool {
    hour_tick.is_multiple_of(u64::from(sub))
}

/// Subsampling of the temperature advection GATHER (`fill_temp_deltas`
/// in `atmosphere::advection`), same shape as [`ORO_SUBSAMPLE_HOURS`]:
/// run the gather only 1 hour per `N`, with `temperature_advection_rate`
/// scaled x`N` (daily transport ~ conserved). Unlike the gather, the
/// fused per-cell APPLY (`atmosphere::apply_temperature_advection_then_
/// cloud_and_condensation`) stays hourly regardless of `N`: it also runs
/// `cloud_dynamics_for_cell` and `surface_condensation_for_cell`, hourly
/// physics fused into the same sweep since chunk B2, which must not be
/// gated just because the gather skipped this hour.
///
/// `N=3` validated by A/B at radius 30 over 3 years, 3 seeds, against a
/// null-perturbation control (JOURNAL 2026-09-05): every climate
/// metric inside the control's own noise floor, rain days per elevation
/// band within +-4.6 % with no sign pattern, no physics test turned red;
/// the one same-sign shift is `ratio_precip_summer_winter` at -0.1 to
/// -1.3 % on the three seeds, under that metric's 3 % floor. Cost:
/// `atmo_advection_temperature` -48 % at r250 (the gather is that half of
/// the bucket), -7 % of the tick on the 4-vCPU VM. `N=2` bought half the
/// saving for the same footprint. `1` = historical hourly behavior,
/// overridable via `HEXSIM_TEMP_ADVECTION_SUBSAMPLE` for A/B.
pub(crate) const TEMP_ADVECTION_SUBSAMPLE_HOURS: u16 = 3;

/// Effective value: `TEMP_ADVECTION_SUBSAMPLE_HOURS` by default,
/// overridden by `HEXSIM_TEMP_ADVECTION_SUBSAMPLE` (parametric A/B
/// without recompile, same pattern as [`oro_subsample`]). Delegates to
/// [`Ablation::effective`].
pub(crate) fn temp_advection_subsample() -> u16 {
    Ablation::effective().temp_advection_subsample
}

/// Pure predicate: does the temperature advection gather run on this
/// hour, at cadence `sub`? Same shape and same reason as
/// [`oro_runs_this_hour`] (testable without going through
/// [`Ablation::effective`]).
#[must_use]
pub(crate) fn temp_advection_runs_this_hour(hour_tick: u64, sub: u16) -> bool {
    hour_tick.is_multiple_of(u64::from(sub))
}

/// Subsampling of the precipitation pass (`step_precipitation_into`),
/// same shape as [`TEMP_ADVECTION_SUBSAMPLE_HOURS`]: run it only 1 hour
/// per `N`. The compensation is NOT a rate boost, unlike the transport
/// passes: the KK2000 autoconversion drain is the analytical solution of
/// `dq/dt = -C·q^α`, so a pass covering `N` hours integrates that same
/// closed form over `dt = N` (see `kk2000_autoconv_over_hours`), never
/// "hourly rate × N" — the super-linear rate would overshoot exactly
/// where it is largest. Only the per-pass cap `max_precip_per_tick`
/// scales, via [`precip_boosted_params`]; the floor, the updraft factor,
/// the neighbor share and the rain/snow split are per-event quantities
/// and do not (see `step_precipitation_into`'s doc for each).
///
/// Measured at `N=2` and `N=3` (JOURNAL 2026-09-05, 3-seed climate
/// bench against a null-perturbation control): the amount of rain is
/// conserved at every cadence (the 4 mm cap never binds), but at `N=3`
/// its SHAPE moves with the same sign on the three seeds: rain days in
/// the 800-1500 m band +4 to +10 %, plains daily peaks -8 to -24 %,
/// the cloud drifting three hours further between drains and dropping a
/// bigger, more dispersed event. Wider and thinner rain is the wrong
/// direction for #63 (planetary drizzle) and would confound its fix;
/// `N=2` is inside the noise floor but its tick saving is too small to
/// measure. So `1` (hourly, historical behavior) is the shipped default
/// (-66 % on `atmo_precipitation` at `N=3`, -5 % of the r250 tick on the
/// 4-vCPU VM, left on the table until #63), overridable via
/// `HEXSIM_PRECIP_SUBSAMPLE` for A/B without recompile.
pub(crate) const PRECIP_SUBSAMPLE_HOURS: u16 = 1;

/// Effective value: `PRECIP_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_PRECIP_SUBSAMPLE` (parametric A/B without recompile, same
/// pattern as [`temp_advection_subsample`]). Delegates to
/// [`Ablation::effective`].
pub(crate) fn precip_subsample() -> u16 {
    Ablation::effective().precip_subsample
}

/// Pure predicate: does the precipitation pass run on this hour, at
/// cadence `sub`? Same shape and same reason as [`oro_runs_this_hour`]
/// (testable without going through [`Ablation::effective`]).
#[must_use]
pub(crate) fn precip_runs_this_hour(hour_tick: u64, sub: u16) -> bool {
    hour_tick.is_multiple_of(u64::from(sub))
}

/// Boosted copy of `params` for the gated precipitation pass: only the
/// per-pass microphysical cap `max_precip_per_tick` moves (mm per pass,
/// a bounded fall speed — a pass covering `sub` hours may drop `sub`
/// times as much before the cap bites). The drain itself is rescaled by
/// integrating the KK2000 ODE over the longer `dt`, not here. `sub == 1`
/// copies as-is (historical behavior without subsampling, same branch
/// shape as [`oro_boosted_params`]).
#[must_use]
pub(crate) fn precip_boosted_params(params: &AtmosphereParams, sub: u16) -> AtmosphereParams {
    if sub > 1 {
        AtmosphereParams {
            max_precip_per_tick: params.max_precip_per_tick * f32::from(sub),
            ..params.clone()
        }
    } else {
        params.clone()
    }
}

/// Apply Tier 1 scaling to atmospheric rates.
///
/// v0.3.0 PR2 (#38): rates in `AtmosphereParams` are expressed per day
/// (v0.2.x convention preserved). Since `step_atmosphere_into` now runs each
/// hour (Tier 1), divide rates by `TICKS_PER_DAY` before call so their cumulative
/// effect over 24 ticks equals old daily regime.
///
/// Only *rates* (fractions/absolute per tick) are scaled; *thresholds*,
/// *dimensionless coefficients*, and *initial conditions* stay unchanged.
#[must_use]
pub(crate) fn scale_atmosphere_for_hourly_tick(p: &AtmosphereParams) -> AtmosphereParams {
    let f = 1.0 / TICKS_PER_DAY_F32;
    AtmosphereParams {
        // `transpiration_coef` (Kc_max FAO-56) is dimensionless: NOT scaled here.
        // Transpiration computes demand mm/day (Meyer) then divides by TICKS_PER_DAY
        // inline in `step_evaporation`, like free water evaporation. Falls into
        // `..p.clone()`.
        sublimation_rate: p.sublimation_rate * f,
        uplift_rate: p.uplift_rate * f,
        uplift_thermal_coef: p.uplift_thermal_coef * f,
        condensation_rate: p.condensation_rate * f,
        // #63 L2b: no `cloud_evap_rate` left to scale. The reverse
        // transition is a saturation adjustment, bounded by the layer's
        // deficit in mm: a stock, not a rate, so the tick length does
        // not enter it.
        cloud_diffusion_rate: p.cloud_diffusion_rate * f,
        cloud_advection_rate: p.cloud_advection_rate * f,
        max_precip_per_tick: p.max_precip_per_tick,
        orographic_lift_coef: p.orographic_lift_coef * f,
        // Issue #45: surface condensation rate in hourly regime.
        fog_condensation_rate: p.fog_condensation_rate * f,
        // Issue #46: diurnal convective drive coef in hourly regime.
        convective_diurnal_coef: p.convective_diurnal_coef * f,
        // Thresholds, dimensionless coefs, initial conditions: unchanged.
        ..p.clone()
    }
}

/// Apply Tier 1 scaling to wind advection rates. Only
/// `humidity_advection_rate` and `temperature_advection_rate` figure in
/// `step_atmosphere`, other `WindParams` fields govern instantaneous wind
/// field calculation, not per-tick transfers.
#[must_use]
pub(crate) fn scale_wind_for_hourly_tick(p: &WindParams) -> WindParams {
    let f = 1.0 / TICKS_PER_DAY_F32;
    WindParams {
        humidity_advection_rate: p.humidity_advection_rate * f,
        temperature_advection_rate: p.temperature_advection_rate * f,
        ..p.clone()
    }
}

/// Boosted copies of params for gated transport passes (rates x sub,
/// daily transport ~ conserved). `sub == 1` = copies as-is (historical
/// behavior without subsampling).
pub(crate) fn transport_boosted_params(
    params: &AtmosphereParams,
    wind_params: &WindParams,
    sub: u16,
) -> (AtmosphereParams, WindParams) {
    if sub > 1 {
        let nf = f32::from(sub);
        let pt = AtmosphereParams {
            orographic_lift_coef: params.orographic_lift_coef * nf,
            cloud_advection_rate: params.cloud_advection_rate * nf,
            cloud_diffusion_rate: params.cloud_diffusion_rate * nf,
            ..params.clone()
        };
        let wt = WindParams {
            humidity_advection_rate: wind_params.humidity_advection_rate * nf,
            ..wind_params.clone()
        };
        (pt, wt)
    } else {
        (params.clone(), wind_params.clone())
    }
}

/// Boosted copy of `params` for the gated orographic pump (rate x sub,
/// daily transport ~ conserved): only `orographic_lift_coef` moves.
/// `sub == 1` copies as-is (historical behavior without subsampling, same
/// branch shape as [`transport_boosted_params`] even though `x * 1.0` is
/// exact in IEEE 754 — keeps the two boosted-params helpers symmetric).
#[must_use]
pub(crate) fn oro_boosted_params(params: &AtmosphereParams, sub: u16) -> AtmosphereParams {
    if sub > 1 {
        AtmosphereParams {
            orographic_lift_coef: params.orographic_lift_coef * f32::from(sub),
            ..params.clone()
        }
    } else {
        params.clone()
    }
}

/// Boosted copy of `wind_params` for the gated temperature advection
/// gather (rate x sub, daily transport ~ conserved): only
/// `temperature_advection_rate` moves. `sub == 1` copies as-is
/// (historical behavior without subsampling, same branch shape as
/// [`oro_boosted_params`]).
#[must_use]
pub(crate) fn temp_advection_boosted_wind_params(wind_params: &WindParams, sub: u16) -> WindParams {
    if sub > 1 {
        WindParams {
            temperature_advection_rate: wind_params.temperature_advection_rate * f32::from(sub),
            ..wind_params.clone()
        }
    } else {
        wind_params.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_json(p: &AtmosphereParams) -> serde_json::Value {
        serde_json::to_value(p).expect("AtmosphereParams serializes")
    }

    fn as_json_wind(p: &WindParams) -> serde_json::Value {
        serde_json::to_value(p).expect("WindParams serializes")
    }

    #[test]
    fn oro_boosted_params_sub1_is_bit_identical() {
        let params = AtmosphereParams::default();
        let boosted = oro_boosted_params(&params, 1);
        assert_eq!(
            as_json(&boosted),
            as_json(&params),
            "sub=1 must leave every field untouched"
        );
    }

    #[test]
    fn oro_boosted_params_sub3_triples_only_orographic_lift_coef() {
        let params = AtmosphereParams::default();
        let boosted = oro_boosted_params(&params, 3);
        assert_eq!(
            boosted.orographic_lift_coef.to_bits(),
            (params.orographic_lift_coef * 3.0).to_bits(),
            "orographic_lift_coef must be exactly tripled"
        );
        let mut expected = as_json(&params);
        expected["orographic_lift_coef"] = as_json(&boosted)["orographic_lift_coef"].clone();
        assert_eq!(
            as_json(&boosted),
            expected,
            "no field other than orographic_lift_coef may change"
        );
    }

    #[test]
    fn oro_runs_this_hour_gates_on_the_subsample_cadence() {
        assert!(oro_runs_this_hour(0, 3), "hour 0 is a multiple of 3");
        assert!(!oro_runs_this_hour(1, 3), "hour 1 is not a multiple of 3");
        assert!(!oro_runs_this_hour(2, 3), "hour 2 is not a multiple of 3");
        assert!(oro_runs_this_hour(3, 3), "hour 3 is a multiple of 3");
        assert!(
            oro_runs_this_hour(7, 1),
            "sub=1 must run on every hour (historical behavior)"
        );
    }

    #[test]
    fn temp_advection_boosted_wind_params_sub1_is_bit_identical() {
        let wind_params = WindParams::default();
        let boosted = temp_advection_boosted_wind_params(&wind_params, 1);
        assert_eq!(
            as_json_wind(&boosted),
            as_json_wind(&wind_params),
            "sub=1 must leave every field untouched"
        );
    }

    #[test]
    fn temp_advection_boosted_wind_params_sub3_triples_only_temperature_advection_rate() {
        let wind_params = WindParams::default();
        let boosted = temp_advection_boosted_wind_params(&wind_params, 3);
        assert_eq!(
            boosted.temperature_advection_rate.to_bits(),
            (wind_params.temperature_advection_rate * 3.0).to_bits(),
            "temperature_advection_rate must be exactly tripled"
        );
        let mut expected = as_json_wind(&wind_params);
        expected["temperature_advection_rate"] =
            as_json_wind(&boosted)["temperature_advection_rate"].clone();
        assert_eq!(
            as_json_wind(&boosted),
            expected,
            "no field other than temperature_advection_rate may change"
        );
    }

    #[test]
    fn precip_boosted_params_sub1_is_bit_identical() {
        let params = AtmosphereParams::default();
        let boosted = precip_boosted_params(&params, 1);
        assert_eq!(
            as_json(&boosted),
            as_json(&params),
            "sub=1 must leave every field untouched"
        );
    }

    #[test]
    fn precip_boosted_params_sub3_triples_only_max_precip_per_tick() {
        let params = AtmosphereParams::default();
        let boosted = precip_boosted_params(&params, 3);
        assert_eq!(
            boosted.max_precip_per_tick.to_bits(),
            (params.max_precip_per_tick * 3.0).to_bits(),
            "max_precip_per_tick must be exactly tripled"
        );
        let mut expected = as_json(&params);
        expected["max_precip_per_tick"] = as_json(&boosted)["max_precip_per_tick"].clone();
        assert_eq!(
            as_json(&boosted),
            expected,
            "no field other than max_precip_per_tick may change: the drain \
             is rescaled by integrating the ODE over the longer dt, and the \
             floor/updraft/share/phase quantities are per-event"
        );
    }

    #[test]
    fn precip_runs_this_hour_gates_on_the_subsample_cadence() {
        for hour in 0u64..6 {
            assert!(
                precip_runs_this_hour(hour, 1),
                "sub=1 must run on every hour (historical behavior), hour {hour}"
            );
        }
        let expected_sub3 = [true, false, false, true, false, false];
        for (hour, &expected) in expected_sub3.iter().enumerate() {
            let hour = u64::try_from(hour).expect("hour fits u64");
            assert_eq!(
                precip_runs_this_hour(hour, 3),
                expected,
                "hour {hour} at sub=3"
            );
        }
        let expected_sub2 = [true, false, true, false, true, false];
        for (hour, &expected) in expected_sub2.iter().enumerate() {
            let hour = u64::try_from(hour).expect("hour fits u64");
            assert_eq!(
                precip_runs_this_hour(hour, 2),
                expected,
                "hour {hour} at sub=2"
            );
        }
    }

    #[test]
    fn temp_advection_runs_this_hour_gates_on_the_subsample_cadence() {
        for hour in 0u64..6 {
            assert!(
                temp_advection_runs_this_hour(hour, 1),
                "sub=1 must run on every hour (historical behavior), hour {hour}"
            );
        }
        let expected_sub3 = [true, false, false, true, false, false];
        for (hour, &expected) in expected_sub3.iter().enumerate() {
            let hour = u64::try_from(hour).expect("hour fits u64");
            assert_eq!(
                temp_advection_runs_this_hour(hour, 3),
                expected,
                "hour {hour} at sub=3"
            );
        }
    }
}
