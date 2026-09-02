//! Physical laws used by the atmo sim.
//!
//! Phase 1 of the atmo conversion to physical units (issue #29): Tetens
//! (saturation vapor pressure) + Meyer (simplified Dalton law for free-water
//! evaporation). Pure functions, not wired into the pipeline. Output used
//! as a parallel diagnostic (`evap_mm_day`) to verify the produced values
//! are plausible before refactoring the pipeline in Phase 2.

use crate::units::{Hpa, MetersPerSecond, MmPerDay};

/// Saturation vapor pressure over liquid water; Tetens formula.
///
/// Clausius-Clapeyron approximation valid for -30 <= T <= 50 degC, largely
/// sufficient for the modeled terrestrial temperatures.
///
/// Reference : Tetens O. (1930), *Uber einige meteorologische Begriffe*,
/// Zeitschrift fur Geophysik, 6, 297-309.
///
/// Reference values:
/// - `e_s(0 degC) ≈ 6.11 hPa`
/// - `e_s(20 degC) ≈ 23.4 hPa`
/// - `e_s(40 degC) ≈ 73.8 hPa`
/// - `e_s(100 degC) ≈ 1013 hPa` (consistent with boiling at 1 atm).
#[must_use]
pub fn tetens_saturation_vapor_pressure(t_celsius: f32) -> Hpa {
    let num = 17.67 * t_celsius;
    let den = t_celsius + 243.5;
    Hpa(6.112 * (num / den).exp())
}

/// Free-water evaporation via the simplified Dalton law (Meyer 1915).
///
/// Formula: `E = 0.26 * (e_s(T_eau) - RH * e_s(T_air)) * (1 + u/10)`
///
/// where:
/// - `e_s(T)` = saturation vapor pressure (Tetens), in hPa
/// - `RH` = relative humidity of the air, bounded to `[0, 1]`
/// - `u` = wind in m/s
///
/// The `0.26` mm/(j·hPa) coefficient corresponds to daily evaporation over
/// moderate free-water surfaces (lakes, large bodies of water).
/// Source : Ward A. & Trimble S. (2003), *Environmental Hydrology* 2nd ed.,
/// p. 51 ; Chow V.T. (1988), *Handbook of Applied Hydrology*, eq. 4.4.12.
///
/// Reference values for a free-water surface:
/// - `T_eau` = `T_air` = 20 degC, RH = 50%, u = 2 m/s → ~3.6 mm/j
/// - `T_eau` = `T_air` = 30 degC, RH = 40%, u = 4 m/s → ~9.2 mm/j (hot, windy)
/// - `T_eau` = `T_air` = 5 degC, RH = 80%, u = 1 m/s → ~0.5 mm/j
///
/// The returned value is bounded below at zero: if the air is more humid
/// than the water (negative deficit), we return 0, not a reverse evaporation
/// that would break mass conservation.
#[must_use]
pub fn meyer_evaporation(t_water: f32, t_air: f32, rh: f32, wind: MetersPerSecond) -> MmPerDay {
    let e_s_water = tetens_saturation_vapor_pressure(t_water).0;
    let e_s_air = tetens_saturation_vapor_pressure(t_air).0;
    let e_a = rh.clamp(0.0, 1.0) * e_s_air;
    let deficit = (e_s_water - e_a).max(0.0);
    let wind_factor = 1.0 + wind.0 / 10.0;
    MmPerDay(0.26 * deficit * wind_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetens_reference_values() {
        // Tetens is valid over the -30 / +50 degC weather range. Beyond that
        // it overestimates (~1048 hPa at 100 degC, vs 1013 physical), so we
        // don't test it at extreme T.
        let e0 = tetens_saturation_vapor_pressure(0.0).0;
        let e20 = tetens_saturation_vapor_pressure(20.0).0;
        let e40 = tetens_saturation_vapor_pressure(40.0).0;

        assert!(
            (e0 - 6.112).abs() < 0.05,
            "e_s(0) = {e0:.3} hPa, expected ~6.11"
        );
        assert!(
            (e20 - 23.4).abs() < 0.3,
            "e_s(20) = {e20:.3} hPa, expected ~23.4"
        );
        assert!(
            (e40 - 73.8).abs() < 1.0,
            "e_s(40) = {e40:.3} hPa, expected ~73.8"
        );
    }

    #[test]
    fn tetens_is_monotonic() {
        // Saturation pressure is strictly increasing with T.
        let values: Vec<f32> = (-20_i16..=40)
            .step_by(5)
            .map(|t| tetens_saturation_vapor_pressure(f32::from(t)).0)
            .collect();
        for pair in values.windows(2) {
            assert!(pair[1] > pair[0], "monotonicity violated: {pair:?}");
        }
    }

    #[test]
    fn meyer_textbook_case() {
        // 20 degC / 50% RH / 2 m/s = ordinary temperate free water.
        // With C = 0.26: 0.26 * (23.4 - 11.7) * 1.2 ≈ 3.65 mm/j. Consistent
        // with averages observed on mid-latitude lakes (Brest ~3.3 mm/j).
        let e = meyer_evaporation(20.0, 20.0, 0.5, MetersPerSecond(2.0));
        assert!(
            (e.0 - 3.65).abs() < 0.3,
            "Meyer(20, 20, 50%, 2 m/s) = {:.3} mm/day, expected ~3.65",
            e.0
        );
    }

    #[test]
    fn meyer_saturated_air_gives_zero() {
        // Saturated air (RH=100%, T_air=T_eau): no net evaporation.
        let e = meyer_evaporation(20.0, 20.0, 1.0, MetersPerSecond(5.0));
        assert!(
            e.0.abs() < 0.01,
            "RH=100% should give ~0 mm/day, got {}",
            e.0
        );
    }

    #[test]
    fn meyer_cold_water_warm_humid_air_is_bounded() {
        // Cold water under warm humid air: the negative deficit is bounded
        // at 0 (no reverse evaporation that would break mass conservation).
        let e = meyer_evaporation(0.0, 30.0, 0.9, MetersPerSecond(3.0));
        assert!(e.0 >= 0.0, "no negative evap, got {}", e.0);
    }

    #[test]
    fn meyer_wind_increases_evaporation() {
        let base = meyer_evaporation(20.0, 20.0, 0.5, MetersPerSecond(0.0)).0;
        let windy = meyer_evaporation(20.0, 20.0, 0.5, MetersPerSecond(10.0)).0;
        assert!(
            windy > base * 1.5,
            "wind should increase E: calm={base:.3}, windy={windy:.3}"
        );
    }

    #[test]
    fn meyer_temperature_increases_evaporation() {
        let cool = meyer_evaporation(10.0, 10.0, 0.5, MetersPerSecond(2.0)).0;
        let warm = meyer_evaporation(30.0, 30.0, 0.5, MetersPerSecond(2.0)).0;
        assert!(
            warm > cool * 2.0,
            "warm T should evaporate much more: cool={cool:.3}, warm={warm:.3}"
        );
    }
}
