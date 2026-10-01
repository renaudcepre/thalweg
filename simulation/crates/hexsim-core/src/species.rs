//! Plant species as vectors of **functional traits** (#161 step 1, epic
//! #78 for the niche).
//!
//! A **species** is one row of the static `SPECIES` table: a **stratum**
//! (herb, shrub, tree: the vertical layer it occupies), a **growth form**
//! (what the front draws), and a **niche**: **lethal limits** (frost,
//! heat wave, drought; once exceeded, the species dies) and **optima**
//! (temperature, water, light) that modulate its fitness, plus the traits
//! the dynamics read (leaf area at full cover, crop coefficient, shade
//! tolerance, relative turnover). The mechanics read traits, never the
//! identity: adding a species is one row here, no new branch anywhere
//! (`vegetation::step_vegetation`, `atmosphere::step_evaporation`,
//! `fire::step_fire` are all trait-driven).
//!
//! `suitability(normals, light) ∈ [0, 1]` combines the responses into a
//! single climate-fit score for the local cell **at the light its stratum
//! receives** (#161 step 2): the light term is what couples the strata.
//!
//! ## Lethal on extremes, optimum on means
//!
//! Key ecological distinction: a **single** late frost kills (lethal on the
//! annual `t_min`), whereas **vigor** depends on the mean climate (optimum
//! on `t_mean`). Different fields of `CellClimateNormals` are read
//! depending on whether it's a limit or an optimum.
//!
//! ## Light: compensation point, then saturation
//!
//! The light response is `((I − I_c) / (I − I_c + I_half)).max(0)`: zero
//! net growth at the **light compensation point** `I_c` (Larcher 2003,
//! *Physiological Plant Ecology*: ~1-2 % of full sun for shade plants,
//! 5-10 % for sun plants), saturating above it. Under a closed canopy the
//! understory light drops below the compensation point of a heliophilous
//! grass, its suitability is exactly 0 and it dies off as if outside its
//! niche; it comes back in the gaps. Nothing is clamped to a floor: the
//! zero is the physics.
//!
//! ## SI units
//!
//! Temperatures in °C, water in mm (root-available water = water table +
//! surface), light in W/m² (mean absorbed shortwave flux at the stratum),
//! leaf area index in m²/m². The parameters are "Drôme flavor" estimates
//! to calibrate against `diag_species_distribution` on three seeds;
//! metrics before tuning (no physical balance change without global
//! metrics at scale).

use serde::{Deserialize, Serialize};

use crate::climate_normals::CellClimateNormals;

/// Species identity. Serialized as `snake_case` for the frontend (#84).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeciesId {
    // --- Herb stratum ---
    /// Dry calcareous grassland (brachypode, fescue): warm, dry, full sun.
    DryGrassland,
    /// Mesophilous meadow: temperate, better watered, mown-looking.
    Meadow,
    /// Alpine grassland: subalpine/alpine, cold-hardy, low biomass.
    AlpineGrass,
    // --- Shrub stratum ---
    /// Boxwood: the garrigue and the understory of the downy oak,
    /// evergreen, very shade-tolerant, drought-tolerant.
    Boxwood,
    /// Common juniper: pioneer of abandoned pastures, full sun, dry.
    Juniper,
    /// Broom: short-lived pioneer of fallow land, full sun, warm.
    Broom,
    /// Hazel: hedges and cool understory, semi-shade, water-demanding.
    Hazel,
    /// Heath (heather, bilberry): montane to subalpine, cold-hardy, sun.
    Heath,
    // --- Tree stratum ---
    /// Holm oak: Mediterranean fringe, evergreen, extreme drought tolerance.
    HolmOak,
    /// Downy oak, warm-dry foothill zone (lowland).
    OakPubescent,
    /// Beech, cool moist montane.
    Beech,
    /// Fir / spruce, cold montane-subalpine.
    Fir,
    /// Scots pine, pioneer, broad tolerance.
    Pine,
    /// Larch: subalpine, cold-hardy, deciduous conifer, full sun.
    Larch,
    /// Maple (sycamore, field maple): cool ravines, semi-shade.
    Maple,
    /// Riparian woodland (willow, alder, ash): banks and wet bottoms,
    /// water table high all year round.
    Riparian,
}

/// Vertical layer a species occupies. Each stratum has its own space
/// (`Σv ≤ k_total` per stratum, `vegetation::step_vegetation`); the
/// strata interact through the **light** the upper ones let through
/// (`vegetation::light_transmittance_below`). Order = bottom to top, the
/// index is the layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stratum {
    /// Ground layer: grasses and forbs.
    Herb,
    /// Shrub layer: bushes and dwarf shrubs.
    Shrub,
    /// Canopy: trees.
    Tree,
}

/// Number of strata (`Stratum` variants), bottom to top.
pub const STRATUM_COUNT: usize = 3;

/// Strata bottom to top; `STRATA[i]` is the layer at index `i`.
pub const STRATA: [Stratum; STRATUM_COUNT] = [Stratum::Herb, Stratum::Shrub, Stratum::Tree];

impl Stratum {
    /// Layer index, 0 = ground.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Herb => 0,
            Self::Shrub => 1,
            Self::Tree => 2,
        }
    }
}

/// What the species looks like: the front picks a silhouette from it and
/// never from the identity (a new species draws right on the day it is
/// added to the table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrowthForm {
    /// Ground cover, no trunk: rendered through the terrain tint.
    Grass,
    /// Low woody bush.
    Shrub,
    /// Broadleaf tree (blob crown).
    Broadleaf,
    /// Conifer (cone crown).
    Conifer,
}

/// One species: identity, layer, silhouette, climatic niche and the
/// traits the dynamics read. The `*_lethal_*` fields apply to annual
/// **extremes** (isolated frost/heat wave/drought), the optima to
/// **means**.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Species {
    pub id: SpeciesId,
    /// Vertical layer (space budget and light coupling, #161 step 2).
    pub stratum: Stratum,
    /// Silhouette for the front.
    pub growth_form: GrowthForm,
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
    /// Light compensation point (W/m² of mean absorbed shortwave at the
    /// stratum): net growth is zero at and below it. ~150 W/m² is full sun
    /// on this grid, so 3 W/m² is a shade plant, 15 W/m² a sun plant.
    pub light_compensation: f32,
    /// Half-saturation of the light response above the compensation
    /// point (W/m²): `f_sun = (I − I_c) / (I − I_c + sun_half)`.
    pub sun_half: f32,
    /// Leaf area index at full cover of the stratum (m²/m², Beer-Lambert):
    /// what this species' canopy subtracts from the light of the strata
    /// below it (`vegetation::LIGHT_EXTINCTION × lai_max × v`). Closed
    /// stands: beech 5-8, fir/spruce 6-10, sycamore ~5, Scots pine 2-3
    /// (Leuschner & Ellenberg 2017, *Ecology of Central European
    /// Forests*); a beech or fir floor gets 1-5 % of full sun, which is
    /// what these values give through `exp(−0.5 × LAI)`.
    pub lai_max: f32,
    /// Species-specific crop coefficient `Kc` (FAO-56, dimensionless):
    /// transpiration efficiency per unit of biomass. A dense forest (`≈1`)
    /// transpires more than a grassland (`<1`) at equal canopy cover. Consumed
    /// by transpiration (#83): `Kc_cell = Σ crop_coef_i × v_i × light_i`.
    pub crop_coef: f32,
    /// Shade tolerance ∈ [0, 1] (#85): pioneer↔climax axis **within a
    /// stratum**. Low = heliophilous pioneer (pine) that needs open ground;
    /// high = climax species (fir, beech) that regenerates under canopy and
    /// **displaces** pioneers through succession (`vegetation::step_vegetation`).
    pub shade_tolerance: f32,
    /// Growth and colonization rate relative to
    /// `VegetationParams::growth_rate` / `colonization_rate` (dimensionless
    /// functional-type trait): 1 for a woody species, above 1 for the
    /// fast turnover of an herbaceous one.
    pub growth_rel: f32,
    /// Background mortality relative to `VegetationParams::base_mortality`
    /// (dimensionless): 1 for a woody species, above 1 for an annual or
    /// short-lived one.
    pub mortality_rel: f32,
}

/// Which annual extreme excludes a species from a cell. Exposed so the
/// diagnostics name the limit instead of re-deriving the thresholds
/// (anti-pattern #2: the `> 0.0` drought guard lived only in
/// `suitability` and a diag that compared the raw fields over-reported
/// drought on the drought-tolerant species).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LethalCause {
    /// Sustained cold below `temp_lethal_min`.
    Frost,
    /// Sustained heat above `temp_lethal_max`.
    Heat,
    /// Sustained drought below `moisture_lethal_min`.
    Drought,
}

/// The three factors of `Species::suitability`, each in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NicheResponses {
    /// Gaussian thermal response on the annual mean temperature.
    pub temp: f32,
    /// Monod water response on the annual mean root-available water.
    pub water: f32,
    /// Light response above the compensation point, at the light the
    /// stratum receives.
    pub sun: f32,
}

impl Species {
    /// The first lethal limit the cell's annual extremes cross, checked in
    /// the order frost, heat, drought; `None` when the species can live
    /// there. A species with `moisture_lethal_min == 0` has no drought
    /// limit at all: `moisture_min` goes slightly negative through f32
    /// rounding of the transfers, which killed oak, pine and grass on
    /// ~50% of the bare cells before the guard (#151).
    #[must_use]
    pub fn lethal_cause(&self, n: &CellClimateNormals) -> Option<LethalCause> {
        if n.t_min < self.temp_lethal_min {
            return Some(LethalCause::Frost);
        }
        if n.t_max > self.temp_lethal_max {
            return Some(LethalCause::Heat);
        }
        if self.moisture_lethal_min > 0.0 && n.moisture_min < self.moisture_lethal_min {
            return Some(LethalCause::Drought);
        }
        None
    }

    /// The three fitness responses over the annual means, each in
    /// `[0, 1]`: thermal (gaussian around `temp_opt`), water (Monod on
    /// `moisture_mean`) and light (compensation point then saturation, at
    /// `light` W/m², the mean absorbed flux reaching this species'
    /// stratum). Their product is the suitability when no lethal limit is
    /// crossed; read separately, they say which term starves a species.
    #[must_use]
    pub fn responses(&self, n: &CellClimateNormals, light: f32) -> NicheResponses {
        let z = (n.t_mean - self.temp_opt) / self.temp_width;
        let above_compensation = (light - self.light_compensation).max(0.0);
        NicheResponses {
            temp: (-(z * z)).exp(),
            water: n.moisture_mean / (n.moisture_mean + self.moisture_half).max(1e-6),
            sun: above_compensation / (above_compensation + self.sun_half).max(1e-6),
        }
    }

    /// Fitness of the species to the local climate at the light its
    /// stratum receives, ∈ [0, 1]. `0` if a lethal limit is crossed
    /// (`lethal_cause`) or the light is at or below the compensation
    /// point; otherwise the product of the thermal, water and light
    /// `responses` over the annual means. For a canopy species (nothing
    /// above it) `light` is `n.insolation_mean`.
    ///
    /// **Pure** function: no state dependency, deterministic.
    #[must_use]
    pub fn suitability(&self, n: &CellClimateNormals, light: f32) -> f32 {
        if self.lethal_cause(n).is_some() {
            return 0.0;
        }
        let r = self.responses(n, light);
        (r.temp * r.water * r.sun).clamp(0.0, 1.0)
    }

    /// `suitability` at full light (`n.insolation_mean`), what a species
    /// with nothing above it gets.
    #[must_use]
    pub fn suitability_in_full_light(&self, n: &CellClimateNormals) -> f32 {
        self.suitability(n, n.insolation_mean)
    }
}

/// Number of species in the model. Sizes `CellProperties.vegetation`
/// (`[f32; SPECIES_COUNT]`, #81). Stable indices = order of `SPECIES`.
pub const SPECIES_COUNT: usize = 16;

/// Index of a species in `SPECIES` (= its column in
/// `CellProperties::vegetation`). The one place that knows the order:
/// tests and consumers address a species by id, never by a number.
///
/// # Panics
///
/// Never in practice: every `SpeciesId` has its row in `SPECIES`, which
/// `every_id_has_exactly_one_row_and_the_index_is_its_column` pins.
#[must_use]
pub fn species_index(id: SpeciesId) -> usize {
    SPECIES
        .iter()
        .position(|s| s.id == id)
        .expect("every SpeciesId has a row in SPECIES")
}

/// The species table, row `i` = column `i` of `CellProperties::vegetation`.
/// Order: herbs, shrubs, trees, warm to cold within a stratum. Starting
/// parameters, Drôme flavor, to calibrate via `diag_species_distribution`
/// (three seeds) before any tuning.
pub const SPECIES: [Species; SPECIES_COUNT] = [
    // ------------------------------------------------------------------
    // Herb stratum: fast turnover (growth ×2, mortality ×4 of a woody
    // species), no shading of anything below, sun-demanding.
    // ------------------------------------------------------------------
    // Dry grassland: warm, frugal in water, dies in the shade.
    Species {
        id: SpeciesId::DryGrassland,
        stratum: Stratum::Herb,
        growth_form: GrowthForm::Grass,
        temp_lethal_min: -25.0,
        temp_lethal_max: 45.0,
        temp_opt: 13.0,
        temp_width: 8.0,
        moisture_lethal_min: 0.0,
        moisture_half: 1.5,
        light_compensation: 15.0,
        sun_half: 50.0,
        lai_max: 2.0,
        crop_coef: 0.6,
        shade_tolerance: 0.1,
        growth_rel: 2.0,
        mortality_rel: 4.0,
    },
    // Meadow: temperate, needs more water than the dry grassland.
    Species {
        id: SpeciesId::Meadow,
        stratum: Stratum::Herb,
        growth_form: GrowthForm::Grass,
        temp_lethal_min: -30.0,
        temp_lethal_max: 40.0,
        temp_opt: 10.0,
        temp_width: 7.0,
        moisture_lethal_min: 0.0,
        moisture_half: 4.0,
        light_compensation: 12.0,
        sun_half: 50.0,
        lai_max: 3.0,
        crop_coef: 0.85,
        shade_tolerance: 0.2,
        growth_rel: 2.0,
        mortality_rel: 4.0,
    },
    // Alpine grassland: withstands severe cold, low water needs, heliophilous.
    Species {
        id: SpeciesId::AlpineGrass,
        stratum: Stratum::Herb,
        growth_form: GrowthForm::Grass,
        temp_lethal_min: -40.0,
        temp_lethal_max: 30.0,
        temp_opt: 5.0,
        temp_width: 7.0,
        moisture_lethal_min: 0.0,
        moisture_half: 3.0,
        light_compensation: 12.0,
        sun_half: 40.0,
        lai_max: 1.5,
        crop_coef: 0.8,
        shade_tolerance: 0.25,
        growth_rel: 2.0,
        mortality_rel: 4.0,
    },
    // ------------------------------------------------------------------
    // Shrub stratum: woody turnover, shaded by the trees, shades the herbs.
    // ------------------------------------------------------------------
    // Boxwood: evergreen understory of the downy oak, lives in deep shade.
    Species {
        id: SpeciesId::Boxwood,
        stratum: Stratum::Shrub,
        growth_form: GrowthForm::Shrub,
        temp_lethal_min: -20.0,
        temp_lethal_max: 42.0,
        temp_opt: 12.0,
        temp_width: 8.0,
        moisture_lethal_min: 0.0,
        moisture_half: 2.0,
        light_compensation: 3.0,
        sun_half: 20.0,
        lai_max: 3.0,
        crop_coef: 0.7,
        shade_tolerance: 0.8,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Juniper: pioneer of abandoned pastures, full sun, very dry.
    Species {
        id: SpeciesId::Juniper,
        stratum: Stratum::Shrub,
        growth_form: GrowthForm::Shrub,
        temp_lethal_min: -30.0,
        temp_lethal_max: 42.0,
        temp_opt: 11.0,
        temp_width: 10.0,
        moisture_lethal_min: 0.0,
        moisture_half: 1.5,
        light_compensation: 15.0,
        sun_half: 60.0,
        lai_max: 2.0,
        crop_coef: 0.6,
        shade_tolerance: 0.05,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Broom: short-lived pioneer of fallow land, full sun, warm.
    Species {
        id: SpeciesId::Broom,
        stratum: Stratum::Shrub,
        growth_form: GrowthForm::Shrub,
        temp_lethal_min: -18.0,
        temp_lethal_max: 42.0,
        temp_opt: 13.0,
        temp_width: 7.0,
        moisture_lethal_min: 0.0,
        moisture_half: 2.0,
        light_compensation: 15.0,
        sun_half: 60.0,
        lai_max: 1.5,
        crop_coef: 0.7,
        shade_tolerance: 0.05,
        growth_rel: 1.5,
        mortality_rel: 2.0,
    },
    // Hazel: cool hedges and understory, semi-shade, water-demanding.
    Species {
        id: SpeciesId::Hazel,
        stratum: Stratum::Shrub,
        growth_form: GrowthForm::Shrub,
        temp_lethal_min: -30.0,
        temp_lethal_max: 36.0,
        temp_opt: 9.0,
        temp_width: 6.0,
        moisture_lethal_min: 0.5,
        moisture_half: 5.0,
        light_compensation: 6.0,
        sun_half: 30.0,
        lai_max: 3.0,
        crop_coef: 0.9,
        shade_tolerance: 0.6,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Heath: montane to subalpine dwarf shrubs, cold-hardy, sun.
    Species {
        id: SpeciesId::Heath,
        stratum: Stratum::Shrub,
        growth_form: GrowthForm::Shrub,
        temp_lethal_min: -40.0,
        temp_lethal_max: 30.0,
        temp_opt: 5.0,
        temp_width: 6.0,
        moisture_lethal_min: 0.0,
        moisture_half: 3.0,
        light_compensation: 10.0,
        sun_half: 45.0,
        lai_max: 2.0,
        crop_coef: 0.6,
        shade_tolerance: 0.3,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // ------------------------------------------------------------------
    // Tree stratum: the canopy, nothing above it, shades everything.
    // ------------------------------------------------------------------
    // Holm oak: Mediterranean fringe, evergreen, extreme drought tolerance,
    // frost-sensitive.
    Species {
        id: SpeciesId::HolmOak,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Broadleaf,
        temp_lethal_min: -12.0,
        temp_lethal_max: 46.0,
        temp_opt: 15.0,
        temp_width: 6.0,
        moisture_lethal_min: 0.0,
        moisture_half: 1.5,
        light_compensation: 8.0,
        sun_half: 50.0,
        lai_max: 4.0,
        crop_coef: 0.9,
        shade_tolerance: 0.6,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Downy oak: warm plain, tolerates drought well (garrigue scrubland).
    Species {
        id: SpeciesId::OakPubescent,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Broadleaf,
        temp_lethal_min: -20.0,
        temp_lethal_max: 45.0,
        temp_opt: 14.0,
        temp_width: 9.0,
        moisture_lethal_min: 0.0,
        moisture_half: 3.0,
        light_compensation: 10.0,
        sun_half: 50.0,
        lai_max: 3.5,
        crop_coef: 1.0,
        shade_tolerance: 0.5,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Beech: cool montane, water-demanding (dies under marked drought),
    // the densest canopy.
    Species {
        id: SpeciesId::Beech,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Broadleaf,
        temp_lethal_min: -25.0,
        temp_lethal_max: 35.0,
        temp_opt: 10.0,
        temp_width: 7.0,
        moisture_lethal_min: 1.0,
        moisture_half: 5.0,
        light_compensation: 4.0,
        sun_half: 60.0,
        lai_max: 6.0,
        crop_coef: 1.05,
        shade_tolerance: 0.85,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Fir / spruce: cold, humid, withstands severe cold, dense evergreen canopy.
    Species {
        id: SpeciesId::Fir,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Conifer,
        temp_lethal_min: -35.0,
        temp_lethal_max: 30.0,
        temp_opt: 7.0,
        temp_width: 7.0,
        moisture_lethal_min: 1.0,
        moisture_half: 5.0,
        light_compensation: 3.0,
        sun_half: 70.0,
        lai_max: 6.5,
        crop_coef: 1.0,
        shade_tolerance: 0.9,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Scots pine: pioneer, wide thermal window, very low water needs,
    // light canopy.
    Species {
        id: SpeciesId::Pine,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Conifer,
        temp_lethal_min: -30.0,
        temp_lethal_max: 42.0,
        temp_opt: 12.0,
        temp_width: 14.0,
        moisture_lethal_min: 0.0,
        moisture_half: 2.0,
        light_compensation: 15.0,
        sun_half: 40.0,
        lai_max: 2.5,
        crop_coef: 0.9,
        shade_tolerance: 0.1,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Larch: subalpine, cold-hardy, deciduous conifer, full sun, open crown.
    Species {
        id: SpeciesId::Larch,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Conifer,
        temp_lethal_min: -45.0,
        temp_lethal_max: 28.0,
        temp_opt: 4.0,
        temp_width: 6.0,
        moisture_lethal_min: 0.5,
        moisture_half: 4.0,
        light_compensation: 15.0,
        sun_half: 70.0,
        lai_max: 3.0,
        crop_coef: 0.8,
        shade_tolerance: 0.05,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Maple: cool ravines, semi-shade, water-demanding.
    Species {
        id: SpeciesId::Maple,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Broadleaf,
        temp_lethal_min: -28.0,
        temp_lethal_max: 36.0,
        temp_opt: 9.0,
        temp_width: 7.0,
        moisture_lethal_min: 0.8,
        moisture_half: 5.0,
        light_compensation: 6.0,
        sun_half: 45.0,
        lai_max: 5.0,
        crop_coef: 1.0,
        shade_tolerance: 0.7,
        growth_rel: 1.0,
        mortality_rel: 1.0,
    },
    // Riparian woodland: dies as soon as the sustained root water drops
    // under 4 mm, so it only lives where the water table stays high all
    // year (banks, wet bottoms). Fast-growing, short-lived.
    Species {
        id: SpeciesId::Riparian,
        stratum: Stratum::Tree,
        growth_form: GrowthForm::Broadleaf,
        temp_lethal_min: -30.0,
        temp_lethal_max: 38.0,
        temp_opt: 11.0,
        temp_width: 8.0,
        moisture_lethal_min: 4.0,
        moisture_half: 8.0,
        light_compensation: 10.0,
        sun_half: 55.0,
        lai_max: 4.0,
        crop_coef: 1.1,
        shade_tolerance: 0.4,
        growth_rel: 1.5,
        mortality_rel: 2.0,
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
        &SPECIES[species_index(id)]
    }

    #[test]
    fn every_id_has_exactly_one_row_and_the_index_is_its_column() {
        for (i, s) in SPECIES.iter().enumerate() {
            assert_eq!(species_index(s.id), i, "{:?} is row {i}", s.id);
            assert_eq!(
                SPECIES.iter().filter(|o| o.id == s.id).count(),
                1,
                "{:?} appears once",
                s.id
            );
        }
        assert_eq!(
            STRATA.iter().map(|s| s.index()).collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }

    #[test]
    fn traits_are_within_their_physical_ranges() {
        for s in &SPECIES {
            assert!(
                s.temp_lethal_min < s.temp_opt && s.temp_opt < s.temp_lethal_max,
                "{:?}",
                s.id
            );
            assert!(
                s.temp_width > 0.0 && s.moisture_half > 0.0 && s.sun_half > 0.0,
                "{:?}",
                s.id
            );
            assert!(s.moisture_lethal_min >= 0.0, "{:?}", s.id);
            assert!(
                s.light_compensation >= 0.0 && s.light_compensation < 30.0,
                "{:?}",
                s.id
            );
            assert!(s.lai_max > 0.0 && s.lai_max <= 8.0, "{:?}", s.id);
            assert!((0.0..=1.0).contains(&s.shade_tolerance), "{:?}", s.id);
            assert!(s.growth_rel > 0.0 && s.mortality_rel > 0.0, "{:?}", s.id);
            assert!(s.crop_coef > 0.0, "{:?}", s.id);
        }
    }

    #[test]
    fn suitability_is_bounded() {
        let envs = [warm_plain(), cold_highland()];
        for env in envs {
            for s in &SPECIES {
                for light in [0.0, 5.0, 40.0, env.insolation_mean] {
                    let f = s.suitability(&env, light);
                    assert!((0.0..=1.0).contains(&f), "{:?} suitability={f}", s.id);
                    assert!(f.is_finite());
                }
            }
        }
    }

    #[test]
    fn frost_below_lethal_kills() {
        // Oak under a -25 °C frost (< -20 lethal): dies, even with an ok mean climate.
        let mut env = warm_plain();
        env.t_min = -25.0;
        let oak = species(SpeciesId::OakPubescent);
        assert_eq!(oak.lethal_cause(&env), Some(LethalCause::Frost));
        assert!(oak.suitability_in_full_light(&env) < 1e-6);
    }

    #[test]
    fn heatwave_above_lethal_kills() {
        // Fir under a 32 °C heat wave (> 30 lethal): dies.
        let mut env = cold_highland();
        env.t_max = 32.0;
        let fir = species(SpeciesId::Fir);
        assert_eq!(fir.lethal_cause(&env), Some(LethalCause::Heat));
        assert!(fir.suitability_in_full_light(&env) < 1e-6);
    }

    #[test]
    fn drought_kills_water_demanding_not_drought_tolerant() {
        // Extreme drought (water min = 0): beech (lethal 1.0) dies, oak
        // (lethal 0.0) survives.
        let mut env = warm_plain();
        env.moisture_min = 0.0;
        assert_eq!(
            species(SpeciesId::Beech).lethal_cause(&env),
            Some(LethalCause::Drought)
        );
        assert!(species(SpeciesId::Beech).suitability_in_full_light(&env) < 1e-6);
        assert!(species(SpeciesId::OakPubescent).suitability_in_full_light(&env) > 0.0);
    }

    #[test]
    fn a_slightly_negative_minimum_does_not_kill_a_drought_tolerant_species() {
        // f32 rounding of the transfers (#151): no drought limit at all
        // for a species whose threshold is 0.
        let mut env = warm_plain();
        env.moisture_min = -0.001;
        assert_eq!(species(SpeciesId::OakPubescent).lethal_cause(&env), None);
        assert_eq!(species(SpeciesId::DryGrassland).lethal_cause(&env), None);
    }

    #[test]
    fn oak_wins_in_warm_plain() {
        let env = warm_plain();
        let oak = species(SpeciesId::OakPubescent).suitability_in_full_light(&env);
        let fir = species(SpeciesId::Fir).suitability_in_full_light(&env);
        assert!(oak > fir, "oak {oak} should beat fir {fir} in warm plain");
    }

    #[test]
    fn fir_wins_in_cold_highland() {
        let env = cold_highland();
        let fir = species(SpeciesId::Fir).suitability_in_full_light(&env);
        let oak = species(SpeciesId::OakPubescent).suitability_in_full_light(&env);
        assert!(
            fir > oak,
            "fir {fir} should beat oak {oak} in cold highland"
        );
    }

    #[test]
    fn light_below_the_compensation_point_is_a_zero_not_a_floor() {
        // Dry grassland (compensation 15 W/m²) under a closed canopy that
        // lets 10 W/m² through: exactly 0, it dies like outside its
        // niche. Boxwood (compensation 3 W/m²) still grows there.
        let env = warm_plain();
        let grass = species(SpeciesId::DryGrassland);
        let boxwood = species(SpeciesId::Boxwood);
        assert!(grass.suitability(&env, 10.0) <= 0.0);
        assert!(boxwood.suitability(&env, 10.0) > 0.0);
        // Above it, the response rises with the light and saturates.
        let dim = grass.suitability(&env, 30.0);
        let bright = grass.suitability(&env, 150.0);
        assert!(0.0 < dim && dim < bright && bright < 1.0);
    }

    #[test]
    fn sun_plants_are_more_light_limited_in_the_shade_than_shade_plants() {
        // The light response saturates at 1 for every species, so full
        // sun tells them apart only through the other traits; what the
        // compensation point and the half-saturation pin is the shade:
        // there the shade plant keeps most of its response, the sun
        // plant loses most of it. (First version of this test expected
        // juniper to beat boxwood in the open through this term alone:
        // mis-calibrated, a sun plant's edge in the open is its growth
        // rate, not a light response above 1.)
        let env = warm_plain();
        let boxwood = species(SpeciesId::Boxwood);
        let juniper = species(SpeciesId::Juniper);
        let shade = 20.0;
        let open = env.insolation_mean;
        assert!(boxwood.responses(&env, shade).sun > juniper.responses(&env, shade).sun);
        let boxwood_loss = boxwood.responses(&env, shade).sun / boxwood.responses(&env, open).sun;
        let juniper_loss = juniper.responses(&env, shade).sun / juniper.responses(&env, open).sun;
        assert!(
            juniper_loss < boxwood_loss,
            "juniper keeps {juniper_loss:.2} of its light response in the shade, boxwood {boxwood_loss:.2}"
        );
    }

    #[test]
    fn riparian_only_lives_where_the_water_table_stays_high() {
        let riparian = species(SpeciesId::Riparian);
        let mut wet = warm_plain();
        wet.moisture_min = 6.0;
        assert_eq!(riparian.lethal_cause(&wet), None);
        let mut slope = warm_plain();
        slope.moisture_min = 1.0;
        assert_eq!(riparian.lethal_cause(&slope), Some(LethalCause::Drought));
    }

    #[test]
    fn nothing_thrives_on_frozen_rock() {
        // Glacial summit (t_min -50 °C, below the lethal frost of ALL
        // species, even larch at -45): sterile rock, no species survives.
        let env = CellClimateNormals {
            t_mean: -2.0,
            t_min: -50.0,
            t_max: 8.0,
            moisture_mean: 5.0,
            moisture_min: 1.0,
            moisture_max: 20.0,
            insolation_mean: 140.0,
        };
        let best = SPECIES
            .iter()
            .map(|s| s.suitability_in_full_light(&env))
            .fold(0.0_f32, f32::max);
        assert!(best < 0.1, "frozen rock should be ~sterile, best={best}");
    }
}
