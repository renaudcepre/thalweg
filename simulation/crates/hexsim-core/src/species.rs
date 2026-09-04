//! Plant species and ecological niche (#80, epic #78).
//!
//! A **species** carries a *niche*: **lethal limits** (frost, heat wave,
//! drought; once exceeded, the species dies) and **optima** (temperature,
//! water, sunlight) that modulate its fitness. `suitability(normals) ∈
//! [0, 1]` combines these responses into a single climate-fit score for the
//! local cell.
//!
//! This is the N-species generalization of the single-species
//! `carrying_capacity` (T gaussian × water Monod) from `vegetation.rs`. Here
//! we do **not** touch biomass dynamics (logistic growth, stock): that's
//! step C (#81), which will consume `suitability` to grow/compete the
//! species. B is limited to the **pure** niche function.
//!
//! ## Lethal on extremes, optimum on means
//!
//! Key ecological distinction: a **single** late frost kills (lethal on the
//! annual `t_min`), whereas **vigor** depends on the mean climate (optimum
//! on `t_mean`). Different fields of `CellClimateNormals` are read
//! depending on whether it's a limit or an optimum.
//!
//! ## SI units
//!
//! Temperatures in °C, water in mm (root-available water = water table +
//! surface), sunlight in W/m² (mean absorbed flux). The starting parameters
//! are "Drôme flavor" estimates to **calibrate via scale test #82** (D);
//! metrics before tuning (no physical balance change without global
//! metrics at scale).

use serde::{Deserialize, Serialize};

use crate::climate_normals::CellClimateNormals;

/// Species identity. Serialized as `snake_case` for the frontend (#84).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeciesId {
    /// Downy oak, warm-dry foothill zone (lowland).
    OakPubescent,
    /// Scots pine / juniper, pioneer, broad tolerance.
    Pine,
    /// Beech, cool moist montane.
    Beech,
    /// Fir / spruce, cold montane-subalpine.
    Fir,
    /// Alpine grassland / grasses, subalpine/alpine, low biomass.
    AlpineGrass,
}

/// Climatic niche of a species. The `*_lethal_*` fields apply to annual
/// **extremes** (isolated frost/heat wave/drought), the optima to **means**.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Species {
    pub id: SpeciesId,
    /// Lethal frost: if the annual `t_min` < this threshold, the species dies (°C).
    pub temp_lethal_min: f32,
    /// Lethal heat wave: if the annual `t_max` > this threshold (°C).
    pub temp_lethal_max: f32,
    /// Thermal optimum on **mean** temperature (°C).
    pub temp_opt: f32,
    /// Width of the thermal window (standard deviation of the gaussian, °C).
    pub temp_width: f32,
    /// Lethal drought: if the **minimum** annual available water < this
    /// threshold (mm), the species dies. 0 = tolerant of extreme drought.
    pub moisture_lethal_min: f32,
    /// Monod half-saturation on **mean** water (mm):
    /// `f_water = moisture_mean / (moisture_mean + moisture_half)`.
    pub moisture_half: f32,
    /// Half-saturation of the response to **mean** sunlight (W/m²):
    /// `f_sun = insolation_mean / (insolation_mean + sun_half)`.
    pub sun_half: f32,
    /// Species-specific crop coefficient `Kc` (FAO-56, dimensionless):
    /// transpiration efficiency per unit of biomass. A dense forest (`≈1`)
    /// transpires more than a grassland (`<1`) at equal canopy cover. Consumed
    /// by transpiration (#83): `Kc_cell = Σ crop_coef_i × v_i`.
    pub crop_coef: f32,
    /// Shade tolerance ∈ [0, 1] (#85): pioneer↔climax axis. Low =
    /// heliophilous pioneer (pine) that needs open ground; high = climax
    /// species (fir, beech) that regenerates under canopy and **displaces**
    /// pioneers through succession (`vegetation::step_vegetation`).
    pub shade_tolerance: f32,
}

impl Species {
    /// Fitness of the species to the local climate, ∈ [0, 1]. `0` if a lethal
    /// limit is crossed; otherwise the product of thermal (gaussian), water
    /// (Monod), and light (saturating) responses over the annual means.
    ///
    /// **Pure** function: no state dependency, deterministic.
    #[must_use]
    pub fn suitability(&self, n: &CellClimateNormals) -> f32 {
        // --- Lethal cutoffs (annual extremes) ---
        if n.t_min < self.temp_lethal_min || n.t_max > self.temp_lethal_max {
            return 0.0;
        }
        // `> 0.0` guard (#151): a species whose lethal threshold is 0
        // has no drought limit, but `moisture_min` goes slightly
        // negative through f32 rounding of the transfers, which killed
        // oak, pine and grass on ~50% of the bare cells.
        if self.moisture_lethal_min > 0.0 && n.moisture_min < self.moisture_lethal_min {
            return 0.0;
        }

        // --- Fitness (annual means) ---
        let z = (n.t_mean - self.temp_opt) / self.temp_width;
        let f_temp = (-(z * z)).exp();
        let f_water = n.moisture_mean / (n.moisture_mean + self.moisture_half).max(1e-6);
        let f_sun = n.insolation_mean / (n.insolation_mean + self.sun_half).max(1e-6);

        (f_temp * f_water * f_sun).clamp(0.0, 1.0)
    }
}

/// Number of species in the model. Sizes `CellProperties.vegetation`
/// (`[f32; SPECIES_COUNT]`, #81). Stable indices = order of `SPECIES`.
pub const SPECIES_COUNT: usize = 5;

/// Starting species set (Drôme flavor). Initial parameters to calibrate via
/// scale test #82. Order = display order / stable indices.
pub const SPECIES: [Species; SPECIES_COUNT] = [
    // Downy oak: warm plain, tolerates drought well (garrigue scrubland).
    Species {
        id: SpeciesId::OakPubescent,
        temp_lethal_min: -20.0,
        temp_lethal_max: 45.0,
        temp_opt: 14.0,
        temp_width: 9.0,
        moisture_lethal_min: 0.0,
        moisture_half: 3.0,
        sun_half: 50.0,
        crop_coef: 1.0,
        shade_tolerance: 0.5,
    },
    // Pine / juniper: pioneer, wide thermal window, very low water needs.
    Species {
        id: SpeciesId::Pine,
        temp_lethal_min: -30.0,
        temp_lethal_max: 42.0,
        temp_opt: 12.0,
        temp_width: 14.0,
        moisture_lethal_min: 0.0,
        moisture_half: 2.0,
        sun_half: 40.0,
        crop_coef: 0.9,
        shade_tolerance: 0.1,
    },
    // Beech: cool montane, water-demanding (dies under marked drought).
    Species {
        id: SpeciesId::Beech,
        temp_lethal_min: -25.0,
        temp_lethal_max: 35.0,
        temp_opt: 10.0,
        temp_width: 7.0,
        moisture_lethal_min: 1.0,
        moisture_half: 5.0,
        sun_half: 60.0,
        crop_coef: 1.05,
        shade_tolerance: 0.85,
    },
    // Fir / spruce: cold, humid, withstands severe cold.
    Species {
        id: SpeciesId::Fir,
        temp_lethal_min: -35.0,
        temp_lethal_max: 30.0,
        temp_opt: 7.0,
        temp_width: 7.0,
        moisture_lethal_min: 1.0,
        moisture_half: 5.0,
        sun_half: 70.0,
        crop_coef: 1.0,
        shade_tolerance: 0.9,
    },
    // Alpine grassland: withstands severe cold, low water needs, heliophilous.
    Species {
        id: SpeciesId::AlpineGrass,
        temp_lethal_min: -40.0,
        temp_lethal_max: 30.0,
        temp_opt: 6.0,
        temp_width: 8.0,
        moisture_lethal_min: 0.0,
        moisture_half: 3.0,
        sun_half: 40.0,
        crop_coef: 0.8,
        shade_tolerance: 0.25,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Warm plain normals (foothill zone): temperate, average water, well
    /// sunlit, no severe frost.
    fn warm_plain() -> CellClimateNormals {
        CellClimateNormals {
            t_mean: 14.0,
            t_min: -5.0,
            t_max: 34.0,
            moisture_mean: 8.0,
            moisture_min: 2.0,
            moisture_max: 30.0,
            insolation_mean: 160.0,
        }
    }

    /// Cold highland normals (subalpine): cold, humid, marked frost.
    fn cold_highland() -> CellClimateNormals {
        CellClimateNormals {
            t_mean: 6.0,
            t_min: -15.0,
            t_max: 18.0,
            moisture_mean: 10.0,
            moisture_min: 3.0,
            moisture_max: 40.0,
            insolation_mean: 150.0,
        }
    }

    fn species(id: SpeciesId) -> &'static Species {
        SPECIES.iter().find(|s| s.id == id).unwrap()
    }

    #[test]
    fn suitability_is_bounded() {
        let envs = [warm_plain(), cold_highland()];
        for env in envs {
            for s in &SPECIES {
                let f = s.suitability(&env);
                assert!((0.0..=1.0).contains(&f), "{:?} suitability={f}", s.id);
                assert!(f.is_finite());
            }
        }
    }

    #[test]
    fn frost_below_lethal_kills() {
        // Oak under a -25 °C frost (< -20 lethal): dies, even with an ok mean climate.
        let mut env = warm_plain();
        env.t_min = -25.0;
        assert!(species(SpeciesId::OakPubescent).suitability(&env) < 1e-6);
    }

    #[test]
    fn heatwave_above_lethal_kills() {
        // Fir under a 32 °C heat wave (> 30 lethal): dies.
        let mut env = cold_highland();
        env.t_max = 32.0;
        assert!(species(SpeciesId::Fir).suitability(&env) < 1e-6);
    }

    #[test]
    fn drought_kills_water_demanding_not_drought_tolerant() {
        // Extreme drought (water min = 0): beech (lethal 1.0) dies, oak
        // (lethal 0.0) survives.
        let mut env = warm_plain();
        env.moisture_min = 0.0;
        assert!(species(SpeciesId::Beech).suitability(&env) < 1e-6);
        assert!(species(SpeciesId::OakPubescent).suitability(&env) > 0.0);
    }

    #[test]
    fn oak_wins_in_warm_plain() {
        let env = warm_plain();
        let oak = species(SpeciesId::OakPubescent).suitability(&env);
        let fir = species(SpeciesId::Fir).suitability(&env);
        assert!(oak > fir, "oak {oak} should beat fir {fir} in warm plain");
    }

    #[test]
    fn fir_wins_in_cold_highland() {
        let env = cold_highland();
        let fir = species(SpeciesId::Fir).suitability(&env);
        let oak = species(SpeciesId::OakPubescent).suitability(&env);
        assert!(
            fir > oak,
            "fir {fir} should beat oak {oak} in cold highland"
        );
    }

    #[test]
    fn nothing_thrives_on_frozen_rock() {
        // Glacial summit (t_min -45 °C, below the lethal frost of ALL
        // species, even alpine grass at -40): sterile rock, no species
        // survives.
        let env = CellClimateNormals {
            t_mean: -2.0,
            t_min: -45.0,
            t_max: 8.0,
            moisture_mean: 5.0,
            moisture_min: 1.0,
            moisture_max: 20.0,
            insolation_mean: 140.0,
        };
        let best = SPECIES
            .iter()
            .map(|s| s.suitability(&env))
            .fold(0.0_f32, f32::max);
        assert!(best < 0.1, "frozen rock should be ~sterile, best={best}");
    }
}
