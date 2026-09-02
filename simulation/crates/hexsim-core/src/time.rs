//! Simulator time contract (v0.3.0, issue #38).
//!
//! Since v0.3.0: 1 tick = 1 hour. Helpers to convert between
//! hours (internal counter), days (historical external API) and day
//! of year (astro calendar).
//!
//! PR1 (infrastructure): the hourly counter exists, but the phenomena
//! are still all called once per day. PR2 will promote them to Tier 1/2
//! with rate division and consumption of `solar_elevation_at_hour`.

/// Number of sub-ticks per day. Fixes the diurnal resolution at 1h.
pub const TICKS_PER_DAY: u64 = 24;

/// Ready-to-use f32 version of `TICKS_PER_DAY`, avoids the
/// `TICKS_PER_DAY as f32` casts scattered across the Tier 1 phenomena
/// (scaling daily rates to hourly). 24 < 2^23 f32 mantissa, an
/// exactly representable value.
#[allow(clippy::cast_precision_loss)]
pub const TICKS_PER_DAY_F32: f32 = TICKS_PER_DAY as f32;

/// Number of days in the simulated year. Simplified calendar model
/// (fixed 365 days, no leap years).
pub const DAYS_PER_YEAR: u64 = 365;

/// Number of sub-ticks per simulated year.
pub const TICKS_PER_YEAR: u64 = TICKS_PER_DAY * DAYS_PER_YEAR;

/// Day of year [0, 364] for a given hour counter.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn day_of_year(tick: u64) -> u16 {
    ((tick / TICKS_PER_DAY) % DAYS_PER_YEAR) as u16
}

/// Local hour [0, 23] within the current day (sub-tick index).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn hour_of_day(tick: u64) -> u8 {
    (tick % TICKS_PER_DAY) as u8
}

/// Wall-clock hour [0, 24) for solar calculations, **independent of the
/// number of sub-ticks per day**. At `TICKS_PER_DAY = N`, sub-tick k represents
/// the instant `k · 24/N` hours: the sun's position stays spread over the
/// real 24 h, not over an N-hour day. For N = 24 this is exactly
/// `hour_of_day` in f32 (factor 1.0, bit-identical). For N < 24 the diurnal
/// cycle is sampled more coarsely but without scale distortion.
/// Use everywhere the hour feeds `solar_elevation_at_hour`; reserve
/// `hour_of_day` for day boundaries (== 0) and the diagnostic API.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn clock_hour_of_day(tick: u64) -> f32 {
    (tick % TICKS_PER_DAY) as f32 * (24.0 / TICKS_PER_DAY_F32)
}

/// Cumulative days since tick=0 (integer division, ignores the hourly
/// remainder).
#[must_use]
pub const fn ticks_to_days(tick: u64) -> u64 {
    tick / TICKS_PER_DAY
}

/// Converts a number of days into a number of ticks (hours).
#[must_use]
pub const fn days_to_ticks(days: u64) -> u64 {
    days * TICKS_PER_DAY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_tick_is_day_zero_hour_zero() {
        assert_eq!(day_of_year(0), 0);
        assert_eq!(hour_of_day(0), 0);
    }

    #[test]
    fn one_full_day_advances_day_of_year() {
        assert_eq!(day_of_year(TICKS_PER_DAY), 1);
        assert_eq!(hour_of_day(TICKS_PER_DAY), 0);
    }

    #[test]
    fn sub_day_ticks_keep_day_of_year() {
        for h in 0..TICKS_PER_DAY {
            assert_eq!(day_of_year(h), 0, "day_of_year({h}) should stay at 0");
            assert_eq!(
                hour_of_day(h),
                u8::try_from(h).expect("h < TICKS_PER_DAY = 24")
            );
        }
    }

    #[test]
    fn clock_hour_matches_hour_of_day_at_n24() {
        // At N=24, the clock hour = the sub-tick index (factor ×1.0),
        // bit-identical to the historical `hour_of_day as f32` path.
        for h in 0..48u64 {
            // Bit-identical is intentional (comment above), not a tolerance:
            // comparing the bits avoids clippy's float_cmp without weakening
            // the assertion.
            assert_eq!(
                clock_hour_of_day(h).to_bits(),
                f32::from(hour_of_day(h)).to_bits()
            );
        }
    }

    #[test]
    fn clock_hour_stays_in_24h_range() {
        // Whatever N is, the clock hour stays within [0, 24); the sun
        // covers the real day, not an N-hour day.
        let span = 24.0 / TICKS_PER_DAY_F32;
        for tick in 0..TICKS_PER_DAY {
            let h = clock_hour_of_day(tick);
            assert!(
                (0.0..24.0).contains(&h),
                "clock_hour({tick}) = {h} hors [0,24)"
            );
            #[allow(clippy::cast_precision_loss)]
            let expected = tick as f32 * span;
            assert!((h - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn day_of_year_wraps_at_year_boundary() {
        assert_eq!(day_of_year(TICKS_PER_YEAR), 0);
        assert_eq!(
            day_of_year(TICKS_PER_YEAR - 1),
            u16::try_from(DAYS_PER_YEAR).expect("365 fits u16") - 1
        );
    }

    #[test]
    fn days_to_ticks_is_multiplication() {
        assert_eq!(days_to_ticks(0), 0);
        assert_eq!(days_to_ticks(10), 240);
        assert_eq!(days_to_ticks(365), TICKS_PER_YEAR);
    }

    #[test]
    fn ticks_to_days_is_floor_division() {
        assert_eq!(ticks_to_days(0), 0);
        assert_eq!(ticks_to_days(23), 0);
        assert_eq!(ticks_to_days(24), 1);
        assert_eq!(ticks_to_days(47), 1);
        assert_eq!(ticks_to_days(TICKS_PER_YEAR), DAYS_PER_YEAR);
    }
}
