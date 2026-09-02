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
const TRANSPORT_SUBSAMPLE_HOURS: u16 = 3;

/// Effective value: `TRANSPORT_SUBSAMPLE_HOURS` by default, overridden by
/// `HEXSIM_TRANSPORT_SUBSAMPLE` (useful for parametric A/B without recompile).
pub(crate) fn transport_subsample() -> u16 {
    use std::sync::OnceLock;
    static N: OnceLock<u16> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("HEXSIM_TRANSPORT_SUBSAMPLE")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(TRANSPORT_SUBSAMPLE_HOURS)
    })
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
        cloud_evap_rate: p.cloud_evap_rate * f,
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
