//! Dimensioned types for physical quantities used across the simulation.
//!
//! This module holds two unrelated efforts and is honest about which is
//! which:
//!
//! - `MmPerDay`, `Hpa`, `MetersPerSecond` are Phase 1 of the atmo-units
//!   conversion (#29): **labels, not enforced units**. They carry no
//!   arithmetic, so every real use unwraps `.0` back to a bare `f32`
//!   immediately (see `physics.rs`). They document a function's return unit
//!   at the call site; nothing stops the caller from mixing quantities once
//!   unwrapped. Phases 2+ of #29, which would have given them real
//!   arithmetic, never happened, and none is planned — treat them as
//!   documentation, not a guarantee.
//! - `Mm` and `Meters` close one specific hole (#104, "lakes on slopes"):
//!   a water depth in millimeters was added straight to a terrain elevation
//!   in meters, because both were bare `f32`. These two types DO carry the
//!   arithmetic they need to stay apart — `Add`/`Sub` within a unit, an
//!   explicit `to_meters()` / `to_mm()` to cross between them, and no
//!   `Add`/`Sub` across the two — so mixing them is rejected at compile
//!   time (see the `compile_fail` doctest on [`crate::units::Meters`]), not caught by a
//!   careful review. They are scoped to the call sites that actually mix a
//!   water depth with an elevation (`cell.rs`, `lake.rs`, `erosion.rs`);
//!   `CellProperties::water_level` and friends stay plain `f32` in mm, only
//!   passing through `Mm`/`Meters` where they meet an elevation.
//!
//! This is a deliberately partial unit system, not a dimensional-analysis
//! library: it does not, for example, stop a caller from multiplying an
//! `Mm` by an area and forgetting the mm -> m factor inside that product
//! (see `Mm::areal_flux_to_m3_per_s`, which exists precisely because that
//! conversion can't be decomposed into `to_meters()` without changing the
//! float rounding of the existing formula). Extending coverage further is
//! future work, done where the next bug shows a real need: a wider type
//! system built before a bug asks for it is a structure nobody can judge.

use serde::Serialize;

/// Millimeters per meter — the sole conversion factor between the engine's
/// two length domains. Private: reached only through [`Mm`]/[`Meters`], so
/// no call site can hand-roll `x / 1000.0` or `x * 1000.0` and get the
/// direction wrong (#104's actual failure mode).
const MM_PER_M: f32 = 1000.0;

/// A height of standing or trapped water, in **millimeters** — the unit the
/// engine's water stocks (`water_level`, `water_capacity`) are natively kept
/// in (1 mm ≡ 1 kg/m² over the column).
///
/// See [`Meters`] for the paired elevation type, the conversion between the
/// two, and the compile-time guarantee this split gives.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Mm(pub f32);

impl Mm {
    /// Converts a water depth from millimeters to meters. The only way to
    /// turn an [`Mm`] into a [`Meters`].
    #[must_use]
    pub fn to_meters(self) -> Meters {
        Meters(self.0 / MM_PER_M)
    }

    /// Clamps a surplus at zero: a cell below `water_capacity` holds no free
    /// water, it isn't "negative water". A clamp needs a physical reason or
    /// it hides an unforeseen case; this one's reason is that a deficit
    /// against capacity has no topological meaning, only the excess does.
    #[must_use]
    pub fn non_negative(self) -> Mm {
        Mm(self.0.max(0.0))
    }

    /// Converts an areal water flux (this value, in mm, accumulated over
    /// `period_s`) into an SI volumetric flow rate (m³/s) through a cell
    /// footprint of `area_m2`.
    ///
    /// Deliberately one fused expression instead of `self.to_meters()`
    /// composed with a separate division: `f32` division does not
    /// distribute over multiplication, so reordering these ops changes the
    /// rounded result (checked against 200k sampled inputs against the
    /// original hand-rolled formula: ~40% differed in the last bit). This
    /// is the one place outside this module allowed to depend on the mm/m
    /// factor, so a future call site converts through here instead of
    /// retyping — and risking dropping — it.
    #[must_use]
    pub fn areal_flux_to_m3_per_s(self, area_m2: f32, period_s: f32) -> f32 {
        self.0 * area_m2 / (MM_PER_M * period_s)
    }
}

impl std::ops::Sub for Mm {
    type Output = Mm;

    fn sub(self, rhs: Mm) -> Mm {
        Mm(self.0 - rhs.0)
    }
}

impl std::ops::Add for Mm {
    type Output = Mm;

    fn add(self, rhs: Mm) -> Mm {
        Mm(self.0 + rhs.0)
    }
}

/// A terrain elevation, or a water-sheet height above it, in **meters** —
/// the unit `CellProperties::elevation` and the lake solver's flat level are
/// kept in.
///
/// Only [`Meters`] + [`Meters`] compiles; adding a raw [`Mm`] does not —
/// that is the entire guarantee this pair of types exists for (#104,
/// "lakes on slopes": 100 mm of surplus water was once added straight to a
/// 100 m elevation, producing a stable "lake" on a slope).
///
/// ```compile_fail
/// use hexsim_core::units::{Meters, Mm};
/// let bad = Meters(100.0) + Mm(50.0); // mismatched units: rejected at compile time
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Meters(pub f32);

impl Meters {
    /// Converts an elevation or a water-sheet height from meters to
    /// millimeters. The only way to turn a [`Meters`] into an [`Mm`].
    #[must_use]
    pub fn to_mm(self) -> Mm {
        Mm(self.0 * MM_PER_M)
    }
}

impl std::ops::Add for Meters {
    type Output = Meters;

    fn add(self, rhs: Meters) -> Meters {
        Meters(self.0 + rhs.0)
    }
}

/// Millimeters per day: evaporation or precipitation flux.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize)]
pub struct MmPerDay(pub f32);

/// Hectopascals: standard meteorological unit for vapor pressures.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize)]
pub struct Hpa(pub f32);

/// Meters per second: wind speed.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize)]
pub struct MetersPerSecond(pub f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtypes_are_constructible_and_default() {
        assert!((MmPerDay(3.2).0 - 3.2).abs() < 1e-6);
        assert!((Hpa(1013.25).0 - 1013.25).abs() < 1e-6);
        assert!((MetersPerSecond(5.0).0 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn mm_and_meters_round_trip() {
        assert!((Mm(1000.0).to_meters().0 - 1.0).abs() < 1e-6);
        assert!((Meters(1.0).to_mm().0 - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn surplus_below_capacity_is_not_negative() {
        // A cell 10 mm short of `water_capacity`: no free water, not "-10 mm".
        let surplus = (Mm(90.0) - Mm(100.0)).non_negative();
        assert!((surplus.0 - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn surplus_above_capacity_is_the_excess() {
        let surplus = (Mm(3000.0) - Mm(1000.0)).non_negative();
        assert!((surplus.0 - 2000.0).abs() < f32::EPSILON);
    }
}
