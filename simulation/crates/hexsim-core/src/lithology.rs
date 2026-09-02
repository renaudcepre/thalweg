//! Lithology: the mineral substrate beneath each cell (L0 stage, epic #136).
//!
//! Before this module, the world had **no composition**: `elevation` was an
//! anonymous bedrock and `permeability` came out of independent noise, with
//! no named phenomenon behind it. A **rock class** per cell gives this
//! noise a physical identity, and opens up later couplings (erodibility,
//! fertility, mineral resources) without inventing a magic coefficient (a
//! hand-calibrated dimensionless parameter bundling several distinct
//! physical phenomena is banned by convention).
//!
//! ## Pattern
//!
//! Modeled on `species.rs`: an `enum LithologyId`, a static table
//! [`LITHOLOGY`] as the **single source of truth** (behavior changes by
//! editing the table, never in code), a stable index per cell.
//!
//! ## "The noise is the big bang"
//!
//! The class is set at tick 0 by [`classify`], a **pure** function of the
//! substrate noise field and the relative altitude; deterministic by seed,
//! like elevation. It is then **static**: neither erosion nor deposition
//! change it for now (exhumation/burial is deferred to later work).
//! **Stratigraphy** (vertical layer
//! columns) is explicitly out of scope: expensive, and it violates the flat
//! scalars of the cell model.
//!
//! ## Two axes, not one
//!
//! The assignment crosses two criteria, and that's what makes it
//! geologically legible rather than a simple thresholding:
//!
//! - **the substrate noise** carries the *porosity* axis (impermeable →
//!   porous);
//! - **the relative altitude** separates, at equal porosity, the
//!   crystalline basement of the heights from the sedimentary infill of the
//!   basins: granite and marl are both impermeable, but one crowns the
//!   ridges and the other blankets the basins.
//!
//! The second axis costs **nothing** in permeability (both classes share
//! the same value, cf. [`LITHOLOGY`]): it only prepares the L2/L3
//! couplings, where the soft rock erodes and where marl feeds the flora.
//!
//! ## Units
//!
//! [`Lithology::permeability`] is the dimensionless aptitude ∈ [0, 1]
//! already consumed by `groundwater` (water table capacity = `permeability
//! × max_capacity`) and `snow`. This module doesn't change its semantics:
//! it changes its **provenance**.

use serde::{Deserialize, Serialize};

/// Identity of a rock class. Serialized in `snake_case` for the front end,
/// like `SpeciesId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LithologyId {
    /// Granite: crystalline basement of the heights. Hard, impermeable,
    /// poor. Carries the ridges and, later, the veins (the "ore" crafting
    /// sheet).
    Granite,
    /// Marl / parent clay: soft infill of the basins. Also impermeable,
    /// but fertile: it's the source of the "clay" sheet.
    Marl,
    /// Sandstone: intermediate sedimentary rock, average porosity and
    /// erodibility.
    Sandstone,
    /// Limestone: karstic. The **porous** class: it's the one that will
    /// store the water table reserve sustaining base flow (#107).
    Limestone,
}

/// Physical parameters of a rock class. Static table = single source of
/// truth (`species::Species` pattern).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Lithology {
    pub id: LithologyId,
    /// Aptitude of the rock to store and let water circulate ∈ [0, 1].
    /// Consumed as-is by `groundwater` (water table capacity) and `snow`.
    ///
    /// **L0 values deliberately not very contrasted**, see [`LITHOLOGY`].
    pub permeability: f32,
}

/// Number of classes in the model. Few classes (4), not 50: each one must
/// earn its place through a distinct behavior in a coupling.
pub const LITHOLOGY_COUNT: usize = 4;

/// The world's rock classes. Order = stable indices (serialization,
/// diagnostics); do not reorder without migrating checkpoints.
///
/// ## Why these permeability values are so close
///
/// They are **not** the geological values one would expect (a sound
/// granite is far more impermeable than a karstic limestone). These are
/// the **conditional averages of the noise field** that lithology
/// replaces, measured across 6 worlds (r15 and r45 × seeds 42/7/1234):
///
/// | Noise band | Average | Share of world | Class |
/// |---|---|---|---|
/// | `< 0.47` | 0.4075 | 37.9 % | granite (high) / marl (low) |
/// | `[0.47, 0.55)` | 0.5097 | 33.2 % | sandstone |
/// | `≥ 0.55` | 0.6040 | 28.9 % | limestone |
///
/// This is **deliberate**: L0 is an *identity refactor* (roadmap § P1); we
/// reroute the field's source without moving the physics, so the table is
/// calibrated to **reproduce** the existing field, global average
/// preserved by construction (law of total expectation).
///
/// Spreading the values apart (porous limestone vs. runoff-prone granite)
/// is the **L1** stage, and it is **gated**: its guard metric is
/// persistence #107, which is only reliable after the rain regime is
/// settled (#63/#110/#48). Tuning these values before that would be tuning
/// against something fake (#120 rule 6).
///
/// Granite and marl share the same value for the same reason: they occupy
/// the same noise band, and nothing at this stage justifies separating
/// them on water. Their difference is already real elsewhere (hardness,
/// fertility) and will be consumed by L2/L3.
pub const LITHOLOGY: [Lithology; LITHOLOGY_COUNT] = [
    Lithology {
        id: LithologyId::Granite,
        permeability: 0.4075,
    },
    Lithology {
        id: LithologyId::Marl,
        permeability: 0.4075,
    },
    Lithology {
        id: LithologyId::Sandstone,
        permeability: 0.5097,
    },
    Lithology {
        id: LithologyId::Limestone,
        permeability: 0.6040,
    },
];

/// Noise threshold below which the substrate is impermeable (granite or
/// marl). Measured: ~38 % of the world. Cf. [`LITHOLOGY`] for the
/// provenance.
const NOISE_IMPERMEABLE_MAX: f32 = 0.47;

/// Noise threshold above which the substrate is karstic (limestone).
/// Measured: ~29 % of the world.
const NOISE_POROUS_MIN: f32 = 0.55;

/// Relative altitude (`shaped_elevation / elevation_scale`, ∈ [0, 1]) above
/// which an impermeable cell is crystalline basement rather than basin
/// infill. Measured: splits the impermeable band into ~14 % granite / ~24 %
/// marl.
///
/// This is the **same** altitude normalization used by the topographic
/// attenuation of infiltration (`terrain.rs`): one single sense of "high
/// relief" in the generator, not two competing definitions.
const ALTITUDE_BASEMENT_MIN: f32 = 0.08;

/// Bands that cross would make sandstone silently disappear. Checked at
/// compile time rather than in a test: this is a property of the
/// constants, not of behavior.
const _: () = assert!(NOISE_IMPERMEABLE_MAX < NOISE_POROUS_MIN);

impl LithologyId {
    /// Stable index in [`LITHOLOGY`]: compact serialization and diagnostic
    /// histograms.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Granite => 0,
            Self::Marl => 1,
            Self::Sandstone => 2,
            Self::Limestone => 3,
        }
    }

    /// Physical parameters of the class.
    #[must_use]
    pub const fn params(self) -> &'static Lithology {
        &LITHOLOGY[self.index()]
    }

    /// Water aptitude of the class: shortcut for `params().permeability`.
    #[must_use]
    pub const fn permeability(self) -> f32 {
        self.params().permeability
    }
}

/// Rock class of a cell, **pure function** of its substrate noise and its
/// relative altitude. Deterministic: same seed ⇒ same world.
///
/// - `substrate_noise` ∈ [0, 1]: substrate noise field (the one that
///   directly drove `permeability` before #136).
/// - `relative_altitude` ∈ [0, 1]: shaped elevation relative to
///   `elevation_scale`, 0 in plains/basins.
#[must_use]
pub fn classify(substrate_noise: f32, relative_altitude: f32) -> LithologyId {
    if substrate_noise < NOISE_IMPERMEABLE_MAX {
        // Impermeable band: relief decides between basement and infill.
        if relative_altitude >= ALTITUDE_BASEMENT_MIN {
            LithologyId::Granite
        } else {
            LithologyId::Marl
        }
    } else if substrate_noise < NOISE_POROUS_MIN {
        LithologyId::Sandstone
    } else {
        LithologyId::Limestone
    }
}

impl Default for LithologyId {
    /// Sandstone, the median class: fallback value for checkpoints prior
    /// to #136 (cf. `CellProperties::lithology`).
    fn default() -> Self {
        Self::Sandstone
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_consistent_with_indices() {
        // The table and the enum cannot diverge silently: that's what would
        // break serialization by index.
        for (i, litho) in LITHOLOGY.iter().enumerate() {
            assert_eq!(litho.id.index(), i, "unstable index for {:?}", litho.id);
        }
    }

    #[test]
    fn permeability_is_bounded() {
        for litho in &LITHOLOGY {
            assert!(
                (0.0..=1.0).contains(&litho.permeability),
                "{:?} permeability out of [0,1]: {}",
                litho.id,
                litho.permeability
            );
        }
    }

    #[test]
    fn porosity_is_ordered_impermeable_to_karst() {
        // The table's physical ordering is what will give the L1 coupling
        // its meaning: karst stores more than basement. Even if not very
        // contrasted, the values must never invert.
        assert!(
            LithologyId::Granite.permeability() < LithologyId::Sandstone.permeability(),
            "granite must stay less porous than sandstone"
        );
        assert!(
            LithologyId::Sandstone.permeability() < LithologyId::Limestone.permeability(),
            "sandstone must stay less porous than limestone"
        );
        assert!(
            (LithologyId::Granite.permeability() - LithologyId::Marl.permeability()).abs() < 1e-6,
            "granite and marl share the impermeable band until L1 splits the values apart"
        );
    }

    #[test]
    fn classify_is_monotone_on_noise() {
        // At fixed altitude, the more porous the noise, the more porous the
        // class.
        let plain = 0.0;
        assert_eq!(classify(0.10, plain), LithologyId::Marl);
        assert_eq!(classify(0.50, plain), LithologyId::Sandstone);
        assert_eq!(classify(0.90, plain), LithologyId::Limestone);
        let a = classify(0.10, plain).permeability();
        let b = classify(0.50, plain).permeability();
        let c = classify(0.90, plain).permeability();
        assert!(a < b && b < c, "non-monotone permeability: {a} {b} {c}");
    }

    #[test]
    fn altitude_splits_basement_from_basin_fill() {
        // Same impermeable noise: basement at altitude, marl in the basin.
        let noise = 0.2;
        assert_eq!(classify(noise, 0.6), LithologyId::Granite);
        assert_eq!(classify(noise, 0.0), LithologyId::Marl);
    }

    #[test]
    fn altitude_split_costs_nothing_hydraulically() {
        // The second axis must change NOTHING in the permeability field:
        // that's the condition that makes the identity refactor honest. If
        // one day L1 splits granite and marl apart, this test must fail
        // knowingly.
        for noise in [0.0_f32, 0.1, 0.3, 0.46] {
            let high = classify(noise, 1.0).permeability();
            let low = classify(noise, 0.0).permeability();
            assert!(
                (high - low).abs() < 1e-6,
                "altitude changed permeability at noise={noise}: {high} vs {low}"
            );
        }
    }

    #[test]
    fn classify_only_ever_sees_porous_above_threshold() {
        // Limestone is the only class above the porous threshold, whatever
        // the altitude: karst is not a matter of relief.
        for alt in [0.0_f32, 0.5, 1.0] {
            assert_eq!(classify(0.8, alt), LithologyId::Limestone);
        }
    }
}
