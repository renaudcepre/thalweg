//! Vegetation layer, **multi-species biomass in three strata** (epic #78,
//! step C: #81; strata and light: #161 step 2).
//!
//! Pure phenomenon (double-buffer) that evolves a **biomass per species**
//! `CellProperties.vegetation: [f32; SPECIES_COUNT]`. Each species grows
//! according to its **suitability** to the local climate **at the light
//! its stratum receives** (`species::Species::suitability`, climate
//! normals #79) and the **space left in its stratum**. The landscape
//! (forest, open woodland, grassland, bare soil, dominant species)
//! **emerges from who wins**; nothing is painted (anti-pattern #2:
//! derived, never stored).
//!
//! ## Strata, light, competition, succession
//!
//! Each species lives in one stratum (herb, shrub, tree) and the three
//! strata are three **separate space budgets**: `Σ v_i ≤ k_total` within a
//! stratum, so a closed canopy and a full understory coexist on the same
//! ground. The one coupling between strata is **light**: a stratum
//! receives `insolation_mean × exp(−k × LAI_above)` (Beer-Lambert, Monsi &
//! Saeki 1953), and a species whose stratum light sits at or below its
//! **light compensation point** (Larcher 2003) has zero suitability and
//! dies off as if outside its niche. A heliophilous grass vanishes under a
//! beech canopy and comes back in the gaps; a shade-tolerant boxwood lives
//! under the oaks.
//!
//! Within a stratum, two mechanisms: (1) **shared logistic growth** toward
//! the free space of the stratum,
//! `growth_i ∝ growth_rel_i × suitability_i × v_i × (1 − Σ_S v/k_total)`;
//! (2) **succession** (#85): a shade-tolerant species displaces the less
//! tolerant ones **of its own stratum** (conservative intra-stratum flux).
//! Without (2), pine (a broad-niche generalist) dominated everything
//! (~95%, measured in #82); with it, the lowland-to-mountain gradient
//! emerges (oak at low elevation, beech/fir in the mountains,
//! grassland/rock at altitude).
//!
//! ## Atmosphere coupling (#77/#83, #161)
//!
//! Transpiration (FAO-56, `atmosphere::step_evaporation`) is driven by the
//! cover of each species weighted by its crop coefficient **and by the
//! light its stratum receives** (`light_transmittance_below`): a shaded
//! understory transpires little. Water drawn from the water table is
//! returned to the atmosphere (strict conservation).
//!
//! ## Determinism
//!
//! No RNG: colonization = **deterministic** rate × suitability. Cell-local
//! phenomenon on a double-buffer, independent of iteration order.

use serde::{Deserialize, Serialize};

use crate::cell::CellProperties;
use crate::climate_normals::CellClimateNormals;
use crate::grid::HexGrid;
use crate::species::{SPECIES, SPECIES_COUNT, STRATA, STRATUM_COUNT, SpeciesId, Stratum};

/// Biomass dynamics parameters (rates **per day**, 1/day). Niches (optima,
/// lethal limits) live on the `species::Species` side; here we only keep
/// the common dynamics.
#[derive(Clone, Serialize, Deserialize)]
pub struct VegetationParams {
    /// Logistic growth rate of an established species toward its share of
    /// space.
    pub growth_rate: f32,
    /// Colonization by propagules: lets an absent species (`v = 0`) settle
    /// where its niche is good and space remains.
    pub colonization_rate: f32,
    /// Background mortality (natural turnover of biomass).
    pub base_mortality: f32,
    /// Accelerated mortality **outside the niche** (zero suitability =
    /// lethal stress): die-off of the standing biomass when the climate
    /// becomes unlivable.
    pub lethal_mortality: f32,
    /// **Succession** rate (#85): speed at which a shade-tolerant species
    /// displaces a less tolerant one (conservative intra-cell flux). 0 = no
    /// succession (pure shared-space competition).
    pub succession_rate: f32,
    /// Total occupation capacity of a cell (full cover). The sum of
    /// biomasses is bounded by this value via the logistic limitation.
    pub k_total: f32,
    /// Surplus of free water above `water_capacity` (mm), liquid **or
    /// frozen** (`CellProperties::water_body_surplus`), beyond which the
    /// cell is open water (lake): no terrestrial vegetation.
    pub open_water_excess: f32,
}

impl Default for VegetationParams {
    fn default() -> Self {
        Self {
            growth_rate: 0.20,
            colonization_rate: 0.01,
            // Slow perennial stock: a winter without growth must not wipe
            // it out. Low turnover.
            base_mortality: 0.005,
            // Outside the niche: net die-off (dead within a few weeks).
            lethal_mortality: 0.10,
            succession_rate: 0.20,
            // Normalized total cover: the sum of biomasses is in [0, 1].
            k_total: 1.0,
            open_water_excess: crate::cell::OPEN_WATER_EXCESS_MM,
        }
    }
}

/// Beer-Lambert extinction coefficient of a plant canopy per unit of leaf
/// area index (Monsi & Saeki 1953; Campbell & Norman 1998, spherical leaf
/// angle distribution, `k ≈ 0.5`): the light under a layer of leaf area
/// index `LAI` is `exp(−LIGHT_EXTINCTION × LAI)` of the light above it.
pub const LIGHT_EXTINCTION: f32 = 0.5;

/// Cover of one stratum: the share of the cell's ground held by the
/// species of that layer, `Σ v_i` over them (#161 step 2). Each stratum
/// has its own space budget (`VegetationParams::k_total`), so a cell can
/// carry a full canopy and a full understory at once.
#[must_use]
pub fn stratum_cover(cell: &CellProperties, stratum: Stratum) -> f32 {
    SPECIES
        .iter()
        .zip(cell.vegetation.iter())
        .filter(|(s, _)| s.stratum == stratum)
        .map(|(_, &v)| v)
        .sum()
}

/// Leaf area index of one stratum, `Σ lai_max_i × v_i` (m²/m²): what its
/// species subtract from the light of the layers below.
#[must_use]
pub fn stratum_lai(cell: &CellProperties, stratum: Stratum) -> f32 {
    SPECIES
        .iter()
        .zip(cell.vegetation.iter())
        .filter(|(s, _)| s.stratum == stratum)
        .map(|(s, &v)| s.lai_max * v)
        .sum()
}

/// Fraction of the cell's mean absorbed shortwave that reaches the top
/// of a stratum through every stratum above it, Beer-Lambert on their
/// summed leaf area: `exp(−k × Σ LAI_above)`. 1 for the canopy. The one
/// coupling between strata: an understory species reads
/// `insolation_mean × transmittance` as its light
/// (`Species::suitability`), and transpires in proportion
/// (`atmosphere::step_evaporation`).
#[must_use]
pub fn light_transmittance_below(cell: &CellProperties, stratum: Stratum) -> f32 {
    strata_light_transmittance(cell)[stratum.index()]
}

/// `light_transmittance_below` for every stratum at once, indexed by
/// `Stratum::index()`, in one pass over the species (the hourly
/// transpiration reads all three for every cell). Single home of the
/// Beer-Lambert law (Monsi & Saeki 1953): the leaf area of each stratum is
/// summed in table order, then accumulated from the canopy down.
#[must_use]
pub fn strata_light_transmittance(cell: &CellProperties) -> [f32; STRATUM_COUNT] {
    let mut lai = [0.0_f32; STRATUM_COUNT];
    for (s, &v) in SPECIES.iter().zip(cell.vegetation.iter()) {
        lai[s.stratum.index()] += s.lai_max * v;
    }
    let mut transmittance = [1.0_f32; STRATUM_COUNT];
    let mut lai_above = 0.0_f32;
    for k in (0..STRATUM_COUNT).rev() {
        transmittance[k] = (-LIGHT_EXTINCTION * lai_above).exp();
        lai_above += lai[k];
    }
    transmittance
}

/// Fraction of a stratum visible from the sky, `Π (1 − cover_above)`
/// over the strata above it: 1 for the canopy, the gaps of the canopy for
/// the shrubs, the gaps of both for the herbs. A map coloured by dominant
/// species shows the canopy first, then what its gaps let through. The
/// `.min(1.0)` only absorbs f32 rounding of a sum bounded by `k_total`,
/// it is not a physical cap.
#[must_use]
pub fn stratum_visibility(cell: &CellProperties, stratum: Stratum) -> f32 {
    STRATA
        .iter()
        .filter(|s| s.index() > stratum.index())
        .map(|&s| 1.0 - stratum_cover(cell, s).min(1.0))
        .product()
}

/// Cover of the cell as seen from above, `1 − Π (1 − cover_stratum)`: the
/// share of the ground under at least one layer, in `[0, 1]`. This is the
/// `vegetation` field of the snapshot and the fuel cover of the fire. It
/// replaces the plain sum of biomasses, which exceeds 1 as soon as an
/// understory lives under a canopy.
#[must_use]
pub fn canopy_cover(cell: &CellProperties) -> f32 {
    1.0 - STRATA
        .iter()
        .map(|&s| 1.0 - stratum_cover(cell, s).min(1.0))
        .product::<f32>()
}

/// Dominant species of a cell **as seen from the sky**: the highest
/// `v_i × stratum_visibility(stratum_i)`, or `None` if the soil is bare.
/// A grass at 0.9 under an oak canopy at 0.8 shows as oak (its visible
/// cover is 0.18); in a gap it shows as grass. Derived, never stored;
/// single source of truth for diags / front.
#[must_use]
pub fn dominant_species(cell: &CellProperties) -> Option<SpeciesId> {
    let (idx, max) = SPECIES
        .iter()
        .zip(cell.vegetation.iter())
        .enumerate()
        .map(|(i, (s, &v))| (i, v * stratum_visibility(cell, s.stratum)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    (max > 1e-4).then(|| SPECIES[idx].id)
}

/// Light-weighted transpiring cover of a cell,
/// `Σ crop_coef_i × v_i × transmittance(stratum_i)`: the cover the FAO-56
/// crop coefficient multiplies in `atmosphere::step_evaporation`
/// (`Kc = Kc_max × transpiration_cover`). Each stratum counts in
/// proportion to the light it receives (Penman-Monteith: transpiration
/// follows the absorbed radiation), so a shaded understory adds little
/// and three full strata stay within the column's light budget without
/// any clamp. The one place this sum is written: the hourly evaporation
/// reads it from the daily memo of `Simulation` (vegetation only changes
/// in the daily tail) or computes it here when it has no memo.
#[must_use]
pub fn transpiration_cover(cell: &CellProperties) -> f32 {
    let transmittance = strata_light_transmittance(cell);
    cell.vegetation
        .iter()
        .zip(SPECIES.iter())
        .map(|(&v, s)| v * s.crop_coef * transmittance[s.stratum.index()])
        .sum()
}

/// `transpiration_cover` of every cell of `grid`, into `out` (cleared and
/// refilled, indexed like `grid.cells_slice()`). The daily memo of the
/// hourly transpiration: 16 species and three exponentials per cell once
/// a day instead of once an hour (measured 0.11 → 0.57 ms/h-tick at r45
/// without it, 2026-09-30).
pub fn fill_transpiration_cover_into(grid: &HexGrid, out: &mut Vec<f32>) {
    out.clear();
    out.extend(grid.cells_slice().iter().map(transpiration_cover));
}

/// Vegetation phenomenon (Tier 3, 1x/day). Evolves the per-species biomass
/// of each cell, stratum by stratum (#161 step 2):
///
/// - **space**: each stratum `S` has its own free space
///   `free_S = (1 − Σ_{i∈S} v_i / k_total).max(0)`;
/// - **light**: stratum `S` receives
///   `light_S = insolation_mean × light_transmittance_below(cell, S)`, read
///   from `current` (Beer-Lambert through the strata above);
/// - **growth / colonization / mortality** of species `i` in `S`:
///   `growth_rate × growth_rel_i × v_i × suit_i × free_S`,
///   `colonization_rate × growth_rel_i × suit_i × free_S`,
///   `base_mortality × mortality_rel_i × v_i`, plus
///   `lethal_mortality × v_i` when `suit_i ≤ 0` (outside the climatic niche
///   **or** below the light compensation point), with
///   `suit_i = suitability(normals, light_S)`;
/// - **succession**: shade-tolerance transfer between two species of the
///   same stratum only; it conserves the stratum sum.
///
/// `stand_age` is the mean age of the **tree** stratum: the herbs' fast
/// turnover does not reset the canopy age.
///
/// Pure: reads `current`, writes `next`. `normals` = per-cell climate
/// normals (#79), indexed like `current.cells_slice()`. As long as no year
/// has completed, they hold the default value, so suitability is zero and
/// vegetation stays at 0 (bootstrap after the first year, lag accepted).
pub fn step_vegetation(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &VegetationParams,
    normals: &[CellClimateNormals],
) {
    let cur = current.cells_slice();
    next.cells_slice_mut().clone_from_slice(cur);
    let next_cells = next.cells_slice_mut();
    let k_total = params.k_total.max(1e-6);

    for (i, cell) in cur.iter().enumerate() {
        let veg = cell.vegetation;

        // Open water (lake): no terrestrial vegetation, biomass recedes.
        // Liquid or frozen: a lake that froze over is still a lake, its
        // surplus just moved to `ice_level` (`CellProperties::is_open_water_at`).
        if cell.is_open_water_at(params.open_water_excess) {
            for (nv, &v) in next_cells[i].vegetation.iter_mut().zip(veg.iter()) {
                *nv = (v - params.lethal_mortality * v).max(0.0);
            }
            next_cells[i].stand_age = 0.0; // lake: no terrestrial canopy.
            continue;
        }

        let normals_i = normals.get(i).copied().unwrap_or_default();
        // Per stratum, indexed by `Stratum::index()` (`STRATA[k].index() ==
        // k`). Free space: logistic occupation limitation of the stratum,
        // zero growth once its cover saturates `k_total`. Not a toxic cap
        // (anti-pattern #4), just the physically available room in that
        // layer. Light: the mean absorbed shortwave that reaches the top of
        // the stratum through the leaf area above it, from `current`.
        let free = STRATA.map(|s| (1.0 - stratum_cover(cell, s) / k_total).max(0.0));
        let light = strata_light_transmittance(cell).map(|t| normals_i.insolation_mean * t);

        // 1) Growth / colonization / mortality, per species → `newv`.
        let mut newv = [0.0_f32; SPECIES_COUNT];
        let mut suits = [0.0_f32; SPECIES_COUNT];
        for (s, ((species, &v), suit_slot)) in SPECIES
            .iter()
            .zip(veg.iter())
            .zip(suits.iter_mut())
            .enumerate()
        {
            let k = species.stratum.index();
            let suit = species.suitability(&normals_i, light[k]);
            *suit_slot = suit;
            let growth = params.growth_rate * species.growth_rel * v * suit * free[k];
            let colonization = params.colonization_rate * species.growth_rel * suit * free[k];
            let mut mortality = params.base_mortality * species.mortality_rel * v;
            if suit <= 0.0 {
                mortality += params.lethal_mortality * v;
            }
            newv[s] = (v + growth + colonization - mortality).max(0.0);
        }

        // 2) Succession (#85): conservative intra-stratum flux, a more
        // shade-tolerant species `i` takes biomass from a less tolerant one
        // `j` **of the same stratum** where its niche is good (a beech
        // regenerates under pines, not under a meadow). The stratum sum is
        // unchanged → each stratum stays bounded by `k_total`. Computed on
        // the `pre` snapshot (order-independent).
        let pre = newv;
        for (i_idx, (&pre_i, &suit_i)) in pre.iter().zip(suits.iter()).enumerate() {
            for (j_idx, &pre_j) in pre.iter().enumerate() {
                if SPECIES[i_idx].stratum != SPECIES[j_idx].stratum {
                    continue;
                }
                let adv = SPECIES[i_idx].shade_tolerance - SPECIES[j_idx].shade_tolerance;
                if adv <= 0.0 {
                    continue;
                }
                let transfer = params.succession_rate * pre_i * pre_j * adv * suit_i;
                newv[i_idx] += transfer;
                newv[j_idx] -= transfer;
            }
        }

        for (nv, slot) in newv.iter().zip(next_cells[i].vegetation.iter_mut()) {
            *slot = nv.max(0.0);
        }

        // Mean age of the tree stratum (the canopy that fire and the front
        // read): existing tree biomass ages +1 day, new tree biomass (net
        // growth + colonization) enters at age 0 and dilutes the average;
        // a net loss leaves the survivors' age unchanged. Herbs and shrubs
        // do not enter it, so their fast turnover never resets the canopy
        // age. If the tree stratum collapses (post-fire), age drops to 0.
        let trees_before = stratum_cover(cell, Stratum::Tree);
        let trees_after = stratum_cover(&next_cells[i], Stratum::Tree);
        let aged = cell.stand_age + 1.0 / 365.0;
        next_cells[i].stand_age = if trees_after > 1e-4 {
            aged * (trees_before / trees_after).min(1.0)
        } else {
            0.0
        };
    }
}

/// `true` if the cell is open water (lake): water surplus above
/// `water_capacity` beyond the `open_water_excess` threshold, **liquid or
/// frozen** (`CellProperties::water_body_surplus`), so a lake stays a lake
/// under its winter ice. No terrestrial vegetation. Single source of truth
/// (anti-pattern #2); the front consumes this flag, it does not re-derive
/// the threshold.
#[must_use]
pub fn is_open_water(cell: &CellProperties) -> bool {
    cell.is_open_water()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::species::species_index;
    use proptest::prelude::*;

    /// Warm and DRY lowland normals (warm-dry collinean, Drôme-like): the
    /// minimum water drops below the lethal threshold of beech/fir, so
    /// these humid-mountain species are excluded, oaks (drought-tolerant)
    /// and pine dominate.
    fn warm_plain() -> CellClimateNormals {
        CellClimateNormals {
            t_mean: 16.0,
            t_min: -3.0,
            t_max: 38.0,
            moisture_mean: 4.0,
            moisture_min: 0.5,
            moisture_max: 18.0,
            insolation_mean: 175.0,
        }
    }

    /// Cold highland normals (subalpine).
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

    /// Cold and dry inner-alpine valley (continental steppe-forest): the
    /// winter frost (−26 °C) kills the oaks and the beech, the summer heat
    /// wave (31 °C) the fir and the larch, the summer drought (0.3 mm)
    /// the maple, the hazel and the riparian woodland. Scots pine is the
    /// climax tree there, juniper the only shrub, the meadow the only herb:
    /// an open pine woodland that no shade-tolerant tree comes to close.
    fn dry_inner_alpine() -> CellClimateNormals {
        CellClimateNormals {
            t_mean: 7.0,
            t_min: -26.0,
            t_max: 31.0,
            moisture_mean: 5.0,
            moisture_min: 0.3,
            moisture_max: 20.0,
            insolation_mean: 160.0,
        }
    }

    /// Runs `step_vegetation` on 1 cell for `steps` days with `params`.
    /// Cell-local phenomenon (no transport): radius 0 is legitimate.
    fn run_params(
        cell: CellProperties,
        normals: CellClimateNormals,
        steps: usize,
        params: &VegetationParams,
    ) -> CellProperties {
        let c0 = HexCoord::new(0, 0);
        let mut grid = HexGrid::from_radius(0);
        *grid.get_mut(c0).unwrap() = cell;
        let norm = vec![normals];
        for _ in 0..steps {
            let mut next = grid.clone();
            step_vegetation(&grid, &mut next, params, &norm);
            grid = next;
        }
        grid.get(c0).unwrap().clone()
    }

    /// `run_params` with a given `succession_rate` (everything else =
    /// defaults).
    fn run_with(
        cell: CellProperties,
        normals: CellClimateNormals,
        steps: usize,
        succession_rate: f32,
    ) -> CellProperties {
        let params = VegetationParams {
            succession_rate,
            ..VegetationParams::default()
        };
        run_params(cell, normals, steps, &params)
    }

    /// `run_params` at the default parameters.
    fn run(cell: CellProperties, normals: CellClimateNormals, steps: usize) -> CellProperties {
        run_params(cell, normals, steps, &VegetationParams::default())
    }

    fn favorable_cell() -> CellProperties {
        CellProperties {
            water_capacity: 1.0,
            ..Default::default()
        }
    }

    /// A favorable cell seeded with the given biomasses, by id.
    fn stand(seed: &[(SpeciesId, f32)]) -> CellProperties {
        let mut cell = favorable_cell();
        for &(id, v) in seed {
            cell.vegetation[species_index(id)] = v;
        }
        cell
    }

    /// A three-layer stand within every stratum budget: oak + pine
    /// canopy, boxwood understory, meadow ground layer.
    fn layered_stand() -> CellProperties {
        stand(&[
            (SpeciesId::OakPubescent, 0.6),
            (SpeciesId::Pine, 0.3),
            (SpeciesId::Boxwood, 0.4),
            (SpeciesId::Meadow, 0.5),
        ])
    }

    fn biomass(cell: &CellProperties, id: SpeciesId) -> f32 {
        cell.vegetation[species_index(id)]
    }

    /// Mean absorbed shortwave reaching the top of `stratum` (W/m²).
    fn light_at(cell: &CellProperties, normals: &CellClimateNormals, stratum: Stratum) -> f32 {
        normals.insolation_mean * light_transmittance_below(cell, stratum)
    }

    fn light_compensation(id: SpeciesId) -> f32 {
        SPECIES[species_index(id)].light_compensation
    }

    #[test]
    fn bare_ground_colonizes_under_good_climate() {
        // Bare soil under lowland climate: cover establishes through
        // colonization.
        let cell = run(favorable_cell(), warm_plain(), 400);
        assert!(
            canopy_cover(&cell) > 0.3,
            "cover expected, got {}",
            canopy_cover(&cell)
        );
    }

    #[test]
    fn outside_all_niches_stays_bare() {
        // Glacial summit (t_min −50 °C, below the lethal frost of every
        // species, larch included at −45 °C): none can establish.
        let frozen = CellClimateNormals {
            t_mean: -3.0,
            t_min: -50.0,
            t_max: 6.0,
            moisture_mean: 5.0,
            moisture_min: 1.0,
            moisture_max: 20.0,
            insolation_mean: 140.0,
        };
        let cell = run(favorable_cell(), frozen, 400);
        assert!(
            canopy_cover(&cell) < 0.05,
            "frozen rock should stay bare, got {}",
            canopy_cover(&cell)
        );
        assert!(
            dominant_species(&cell).is_none(),
            "no species should hold on"
        );
    }

    #[test]
    fn lethal_climate_kills_established_biomass() {
        // Established beech, then extreme drought climate (min water 0,
        // below its lethal threshold 1.0): its biomass regresses.
        let before = 0.5;
        let start = stand(&[(SpeciesId::Beech, before)]);
        let dry = CellClimateNormals {
            t_mean: 16.0,
            t_min: -3.0,
            t_max: 33.0,
            moisture_mean: 1.0,
            moisture_min: 0.0,
            moisture_max: 6.0,
            insolation_mean: 170.0,
        };
        let cell = run(start, dry, 60);
        let after = biomass(&cell, SpeciesId::Beech);
        assert!(
            after < before * 0.5,
            "beech should die under drought: {before} → {after}"
        );
    }

    #[test]
    fn warm_plain_favours_warm_species() {
        // In warm lowland, the dominant is a lowland tree (holm oak, downy
        // oak or pine), never a cold species (fir, larch, alpine grass).
        let cell = run(favorable_cell(), warm_plain(), 400);
        let dom = dominant_species(&cell).expect("nonzero cover");
        assert!(
            matches!(
                dom,
                SpeciesId::HolmOak | SpeciesId::OakPubescent | SpeciesId::Pine
            ),
            "warm plain dominated by {dom:?}"
        );
    }

    #[test]
    fn cold_highland_favours_cold_species() {
        // At cold altitude, the dominant is a mountain species, never a
        // lowland oak.
        let cell = run(favorable_cell(), cold_highland(), 400);
        let dom = dominant_species(&cell).expect("nonzero cover");
        assert!(
            matches!(
                dom,
                SpeciesId::Fir | SpeciesId::Larch | SpeciesId::AlpineGrass
            ),
            "cold altitude dominated by {dom:?}"
        );
    }

    #[test]
    fn succession_favours_shade_tolerant() {
        // Pine (pioneer) + fir (tolerant climax) in cold climate.
        // Succession must give more ground to fir than pure "shared space"
        // competition would (succession_rate = 0).
        let start = stand(&[(SpeciesId::Pine, 0.3), (SpeciesId::Fir, 0.1)]);
        let with = run_with(start.clone(), cold_highland(), 300, 0.05);
        let without = run_with(start, cold_highland(), 300, 0.0);
        let (fir_with, fir_without) = (
            biomass(&with, SpeciesId::Fir),
            biomass(&without, SpeciesId::Fir),
        );
        assert!(
            fir_with > fir_without,
            "succession should favor fir (tolerant): with={fir_with} without={fir_without}"
        );
    }

    /// Biomass share of a tree species in the tree stratum (0 without
    /// trees).
    fn tree_share(cell: &CellProperties, id: SpeciesId) -> f32 {
        let trees = stratum_cover(cell, Stratum::Tree);
        if trees < 1e-6 {
            return 0.0;
        }
        biomass(cell, id) / trees
    }

    #[test]
    fn forest_matures_to_shade_tolerant_climax_over_200y() {
        // A SINGLE hex, bare soil, constant cold-humid climate, 200 years.
        // Succession (rate 0.20) must make a shade-tolerant climax emerge
        // in the tree stratum: fir/beech (shade 0.85-0.9) take ground from
        // the pioneers (pine 0.1, larch 0.05, maple 0.7). We compare the
        // climax share of the tree stratum at the pioneer stage (one
        // season) and at maturity (200 years): succession is a transfer
        // that accumulates. With its own space budget the tree stratum
        // closes in ~60 days and its composition settles within ~2 years
        // (fir + beech 0.79 of the trees at 90 days, 0.976 at 1 year,
        // 0.9796 from year 5 on, measured 2026-09-30): a 5-year horizon,
        // the one used before #161, sits on the plateau already.
        let early = run(favorable_cell(), cold_highland(), 90);
        let climax = run(favorable_cell(), cold_highland(), 200 * 365);

        let dom = dominant_species(&climax).expect("nonzero cover at climax");
        assert!(
            matches!(dom, SpeciesId::Fir | SpeciesId::Beech),
            "cold-humid climax expected shade-tolerant (fir/beech), got {dom:?}"
        );
        let climax_share =
            tree_share(&climax, SpeciesId::Fir) + tree_share(&climax, SpeciesId::Beech);
        let early_share = tree_share(&early, SpeciesId::Fir) + tree_share(&early, SpeciesId::Beech);
        assert!(
            climax_share > early_share,
            "shade-tolerant share should grow with succession: 90d={early_share:.3}, 200y={climax_share:.3}"
        );
        // Closed tree stratum and mature canopy (stand_age accumulates
        // over a stable stand).
        let trees = stratum_cover(&climax, Stratum::Tree);
        assert!(trees > 0.5, "mature forest cover expected, got {trees}");
        assert!(
            climax.stand_age > 30.0,
            "mature canopy expected after 200 years, age {}",
            climax.stand_age
        );
    }

    #[test]
    fn warming_flips_dominant_from_cold_to_warm_species() {
        // Mature cold forest, then the climate flips to warm-dry: the
        // cold-humid species fall outside their niche (heatwave > fir's
        // lethal threshold, drought) and die, a lowland species takes
        // over. This is "vary the temperature, the cover changes".
        let matured = run(favorable_cell(), cold_highland(), 60 * 365);
        let cold_dom = dominant_species(&matured).expect("nonzero cold cover");
        assert!(
            matches!(
                cold_dom,
                SpeciesId::Fir | SpeciesId::Beech | SpeciesId::Larch | SpeciesId::AlpineGrass
            ),
            "initial state: cold dominant expected, got {cold_dom:?}"
        );
        // Same cell, continuing under warm-dry climate.
        let warmed = run(matured, warm_plain(), 40 * 365);
        let warm_dom = dominant_species(&warmed).expect("nonzero warm cover");
        assert!(
            matches!(
                warm_dom,
                SpeciesId::HolmOak | SpeciesId::OakPubescent | SpeciesId::Pine
            ),
            "after warming, lowland dominant expected, got {warm_dom:?}"
        );
        assert_ne!(
            cold_dom, warm_dom,
            "dominant should have changed with the climate"
        );
    }

    #[test]
    fn open_water_clears_vegetation() {
        // Vegetated cell that becomes a lake: vegetation recedes to 0 in
        // every stratum.
        let mut start = layered_stand();
        start.water_level = 50.0; // >> capacity + open_water_excess
        let cell = run(start, warm_plain(), 200);
        assert!(
            canopy_cover(&cell) < 1e-3,
            "lake should be free of vegetation, got {}",
            canopy_cover(&cell)
        );
        assert!(cell.stand_age.abs() < f32::EPSILON, "lake: no canopy age");
        assert!(is_open_water(&cell));
    }

    #[test]
    fn frozen_lake_is_still_open_water_and_stays_bare() {
        // The lake's surplus sits entirely in `ice_level` (deep winter):
        // the identity must follow the water, not its phase. Seen
        // 2026-09-05: trees on a frozen lake within one winter.
        let mut start = layered_stand();
        start.water_level = start.water_capacity; // no liquid surplus at all
        start.ice_level = 50.0;
        assert!(is_open_water(&start), "frozen lake must read as open water");
        let cell = run(start, cold_highland(), 200);
        assert!(
            canopy_cover(&cell) < 1e-3,
            "frozen lake should be free of vegetation, got {}",
            canopy_cover(&cell)
        );
    }

    // ------------------------------------------------------------------
    // Strata and light (#161 step 2). Radius 0: cell-local, no transport.
    // ------------------------------------------------------------------

    #[test]
    fn phys_light_is_beer_lambert_through_the_strata_above() {
        // Beech 0.95 over boxwood 0.9 over a meadow: the canopy gets full
        // light, the shrubs what the beech lets through, the herbs what
        // both let through; the herbs' own leaf area shades nothing.
        // Analytic, k = 0.5, leaf areas read from the table so a
        // recalibrated `lai_max` cannot silently turn this red.
        let cell = stand(&[
            (SpeciesId::Beech, 0.95),
            (SpeciesId::Boxwood, 0.9),
            (SpeciesId::Meadow, 0.9),
        ]);
        let lai_beech = SPECIES[species_index(SpeciesId::Beech)].lai_max * 0.95;
        let lai_boxwood = SPECIES[species_index(SpeciesId::Boxwood)].lai_max * 0.9;
        let t = strata_light_transmittance(&cell);
        let expected = [
            (-0.5_f32 * (lai_beech + lai_boxwood)).exp(),
            (-0.5_f32 * lai_beech).exp(),
            1.0,
        ];
        for s in STRATA {
            let k = s.index();
            assert!(
                (t[k] - expected[k]).abs() < 1e-6,
                "{s:?}: {} vs {}",
                t[k],
                expected[k]
            );
            assert_eq!(
                light_transmittance_below(&cell, s).to_bits(),
                t[k].to_bits()
            );
        }
    }

    #[test]
    fn phys_grass_dies_under_a_closed_beech_forest() {
        // Meadow under a closed beech canopy, cool-humid mountain: within a
        // year the ground layer falls below the meadow's light compensation
        // point (12 W/m²) and it vanishes, while the beech holds. The full
        // palette plays: the canopy alone (beech at its logistic
        // equilibrium ~0.93, LAI 4.6) lets 14.8 W/m² through, just above
        // the meadow's compensation point, and it is the shade-tolerant
        // shrub layer that settles under the beech (boxwood, hazel) that
        // brings the ground light to ~4 W/m² (measured 2026-09-30).
        let n = cold_highland();
        let start = stand(&[(SpeciesId::Beech, 0.95), (SpeciesId::Meadow, 0.5)]);
        let cell = run(start, n, 365);
        let meadow = biomass(&cell, SpeciesId::Meadow);
        let beech = biomass(&cell, SpeciesId::Beech);
        let ground_light = light_at(&cell, &n, Stratum::Herb);
        assert!(
            meadow < 1e-3,
            "meadow should die under a closed beech forest, got {meadow} (ground light {ground_light:.1} W/m²)"
        );
        assert!(
            ground_light < light_compensation(SpeciesId::Meadow),
            "the ground light {ground_light:.1} W/m² should sit below the meadow's compensation point"
        );
        assert!(beech > 0.5, "the beech should persist, got {beech}");
    }

    #[test]
    fn phys_grass_and_pines_coexist_in_an_open_woodland() {
        // The same meadow under a light Scots pine canopy (lai_max 2.5),
        // on a site where pine is the climax (`dry_inner_alpine`): the
        // canopy (LAI ~2.4) and the juniper understory it lets grow still
        // let ~20 W/m² reach the ground, above the meadow's compensation
        // point: three layers coexist. In a fir climate (`cold_highland`)
        // the same pine stand is a pioneer stage, fir replaces it and
        // closes the canopy (succession, not open woodland).
        let n = dry_inner_alpine();
        let start = stand(&[(SpeciesId::Pine, 0.8), (SpeciesId::Meadow, 0.5)]);
        let one_year = run(start, n, 365);
        let ten_years = run(one_year.clone(), n, 9 * 365);
        for (label, cell) in [("1 y", &one_year), ("10 y", &ten_years)] {
            let meadow = biomass(cell, SpeciesId::Meadow);
            let pine = biomass(cell, SpeciesId::Pine);
            assert!(
                meadow > 0.2,
                "{label}: meadow should persist under the pines, got {meadow} (ground light {:.1} W/m²)",
                light_at(cell, &n, Stratum::Herb)
            );
            assert!(
                pine > 0.8,
                "{label}: the pine canopy should hold, got {pine}"
            );
        }
    }

    #[test]
    fn phys_boxwood_persists_under_a_dense_oak_canopy() {
        // Boxwood (compensation 3 W/m²) under a dense downy oak canopy,
        // warm-dry plain: the evergreen understory of the oak woods lives
        // in the shade the heliophilous herbs cannot take.
        let n = warm_plain();
        let start = stand(&[(SpeciesId::OakPubescent, 0.95), (SpeciesId::Boxwood, 0.3)]);
        for years in [1, 10] {
            let cell = run(start.clone(), n, years * 365);
            let boxwood = biomass(&cell, SpeciesId::Boxwood);
            let trees = stratum_cover(&cell, Stratum::Tree);
            assert!(
                boxwood > 0.5,
                "{years} y: boxwood should hold under the oaks, got {boxwood} (shrub light {:.1} W/m²)",
                light_at(&cell, &n, Stratum::Shrub)
            );
            assert!(
                trees > 0.9,
                "{years} y: the canopy should stay closed, got {trees}"
            );
        }
    }

    #[test]
    fn phys_visible_dominant_is_the_canopy() {
        // Seen from the sky, a grass at 0.9 under an oak canopy at 0.8
        // shows through the gaps only (0.9 × 0.2 = 0.18 < 0.8): oak. Under
        // a few scattered oaks (0.05), the meadow is what one sees.
        let under = stand(&[(SpeciesId::Meadow, 0.9), (SpeciesId::OakPubescent, 0.8)]);
        assert_eq!(dominant_species(&under), Some(SpeciesId::OakPubescent));
        let scattered = stand(&[(SpeciesId::Meadow, 0.9), (SpeciesId::OakPubescent, 0.05)]);
        assert_eq!(dominant_species(&scattered), Some(SpeciesId::Meadow));
    }

    #[test]
    fn phys_strata_have_separate_space_budgets() {
        // A herb stratum filled to k_total leaves the tree stratum empty
        // room: trees colonize the meadow (under the single shared space
        // before #161, a full meadow blocked every tree for good). The
        // herbs' own budget stays bounded.
        let n = warm_plain();
        let k_total = VegetationParams::default().k_total;
        let start = stand(&[
            (SpeciesId::Meadow, 0.6 * k_total),
            (SpeciesId::DryGrassland, 0.4 * k_total),
        ]);
        let day_one = run(start.clone(), n, 1);
        assert!(
            stratum_cover(&day_one, Stratum::Tree) > 0.0,
            "trees should start colonizing a full meadow on day one"
        );
        let cell = run(start, n, 200);
        let trees = stratum_cover(&cell, Stratum::Tree);
        let herbs = stratum_cover(&cell, Stratum::Herb);
        assert!(
            trees > 0.5,
            "trees should establish over the meadow, got {trees}"
        );
        assert!(
            herbs <= k_total + 1e-3,
            "herb stratum {herbs} over its budget {k_total}"
        );
    }

    #[test]
    fn phys_herb_turnover_does_not_reset_the_canopy_age() {
        // `stand_age` is the age of the tree stratum: an 80-year-old pine
        // canopy over a bare floor and the same canopy over a meadow that
        // colonizes and grows under it age exactly alike (trees read
        // neither the herbs' space nor their leaf area). Under the single
        // shared space before #161, the new herb biomass entered the mean
        // age at 0 and made an old forest look young.
        let n = dry_inner_alpine();
        let mut bare_floor = stand(&[(SpeciesId::Pine, 0.95)]);
        bare_floor.stand_age = 80.0;
        let mut grassy_floor = bare_floor.clone();
        grassy_floor.vegetation[species_index(SpeciesId::Meadow)] = 0.5;
        let a = run(bare_floor, n, 365);
        let b = run(grassy_floor, n, 365);
        // Not vacuous: the meadow colonized the bare floor and grew there,
        // new herb biomass entered the cell all year.
        let herbs = stratum_cover(&a, Stratum::Herb);
        assert!(
            herbs > 0.1,
            "the meadow should grow under the pines, got {herbs}"
        );
        assert_eq!(
            a.stand_age.to_bits(),
            b.stand_age.to_bits(),
            "canopy age depends on the herbs: {} vs {}",
            a.stand_age,
            b.stand_age
        );
        assert_eq!(
            stratum_cover(&a, Stratum::Tree).to_bits(),
            stratum_cover(&b, Stratum::Tree).to_bits()
        );
        assert!(
            a.stand_age > 80.5,
            "the pine canopy should keep aging, got {}",
            a.stand_age
        );
    }

    #[test]
    fn transpiration_cover_is_the_light_weighted_kc_sum() {
        // Beech 0.9 over boxwood 0.5 over meadow 0.4: the canopy counts in
        // full, the shrubs and herbs through the light the layers above
        // let through. Bit-identical to the term-by-term sum in the same
        // order, and a bare cell transpires nothing.
        let cell = stand(&[
            (SpeciesId::Beech, 0.9),
            (SpeciesId::Boxwood, 0.5),
            (SpeciesId::Meadow, 0.4),
        ]);
        let t = strata_light_transmittance(&cell);
        let mut expected = 0.0_f32;
        for (s, &v) in SPECIES.iter().zip(cell.vegetation.iter()) {
            expected += v * s.crop_coef * t[s.stratum.index()];
        }
        assert_eq!(transpiration_cover(&cell).to_bits(), expected.to_bits());
        assert!(transpiration_cover(&cell) < 0.9 * 1.05 + 0.5 * 0.7 + 0.4 * 0.85);
        assert_eq!(
            transpiration_cover(&CellProperties::default()).to_bits(),
            0.0_f32.to_bits()
        );
        let mut grid = HexGrid::from_radius(1);
        *grid.get_mut(HexCoord::new(0, 0)).unwrap() = cell.clone();
        let mut memo = Vec::new();
        fill_transpiration_cover_into(&grid, &mut memo);
        assert_eq!(memo.len(), grid.len());
        assert_eq!(
            memo[0].to_bits(),
            transpiration_cover(&grid.cells_slice()[0]).to_bits()
        );
    }

    proptest! {
        /// Finite biomass, per species ≥ 0, each stratum bounded by
        /// k_total, for any plausible input (no NaN, no overflow). The
        /// input itself fits every stratum budget (8 trees × 0.12 < 1).
        #[test]
        fn prop_biomass_bounded(
            t_mean in -40.0_f32..40.0,
            t_min in -50.0_f32..0.0,
            t_max in 0.0_f32..55.0,
            moisture in 0.0_f32..60.0,
            insol in 0.0_f32..400.0,
            v0 in prop::array::uniform16(0.0_f32..0.12),
        ) {
            let c0 = HexCoord::new(0, 0);
            let mut grid = HexGrid::from_radius(0);
            grid.get_mut(c0).unwrap().vegetation = v0;
            grid.get_mut(c0).unwrap().water_capacity = 1.0;
            let normals = vec![CellClimateNormals {
                t_mean, t_min, t_max,
                moisture_mean: moisture,
                moisture_min: moisture * 0.3,
                moisture_max: moisture * 1.5,
                insolation_mean: insol,
            }];
            let params = VegetationParams::default();
            let mut next = grid.clone();
            step_vegetation(&grid, &mut next, &params, &normals);
            let cell = next.get(c0).unwrap();
            for &v in &cell.vegetation {
                prop_assert!(v.is_finite() && v >= 0.0, "v out of bounds: {v}");
            }
            for s in STRATA {
                let cover = stratum_cover(cell, s);
                prop_assert!(
                    cover <= params.k_total + 1e-3,
                    "{s:?} cover {cover} > k_total"
                );
            }
            prop_assert!(cell.stand_age.is_finite() && cell.stand_age >= 0.0);
        }
    }
}
