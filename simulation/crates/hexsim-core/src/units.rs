//! Dimensioned types for the physical atmosphere.
//!
//! Phase 1 of converting atmo to physical units (issue #29): newtype
//! with no cross arithmetic, so the compiler rejects `Mm + Hpa`
//! once we start handling stocks in real units (phases 2+).
//!
//! In Phase 1, these types only serve as labels on the outputs of
//! physical functions (Tetens, Meyer). Arithmetic will arrive as
//! later phases require it.

use serde::Serialize;

/// Millimeters of water depth per meter. The engine's water stocks are in mm
/// (1 mm ≡ 1 kg/m² over the column); any water quantity entering a
/// topographic comparison (elevations in m) must be divided by this
/// constant, never added as-is (#104: 100 mm treated as
/// 100 m produced "sloped lakes").
pub const MM_PER_M: f32 = 1000.0;

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
}
