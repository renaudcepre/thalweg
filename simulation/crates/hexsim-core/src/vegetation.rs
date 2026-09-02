//! Vegetation layer, **multi-species biomass** (epic #78, step C: #81).
//!
//! Pure phenomenon (double-buffer) that evolves a **biomass per species**
//! `CellProperties.vegetation: [f32; SPECIES_COUNT]`. Each species grows
//! according to its **suitability** to the local climate
//! (`species::Species::suitability`, computed from the climate normals
//! #79) and the **available space**: species share an occupation capacity
//! `k_total` (competition). The landscape (forest/grassland/bare soil,
//! dominant species) **emerges from who wins**; nothing is painted
//! (anti-pattern #2: derived, never stored).
//!
//! ## Competition + succession
//!
//! Two mechanisms: (1) **shared logistic growth** toward the free space
//! `growth_i ∝ suitability_i × v_i × (1 − Σv/k_total)`; (2)
//! **succession** (#85), a shade-tolerant species displaces less tolerant
//! ones (conservative intra-cell flux). Without (2), pine (a broad-niche
//! generalist) dominated everything (~95%, measured in #82); with it, the
//! lowland-to-mountain gradient emerges (oak at low elevation, beech/fir in
//! the mountains, grassland/rock at altitude).
//!
//! ## Atmosphere coupling (#77/#83)
//!
//! Transpiration (FAO-56, `atmosphere::step_evaporation`) is driven by the
//! **total cover** (`cell_total_vegetation`); water drawn from the water
//! table is returned to the atmosphere (strict conservation). #83 will
//! refine this per species.
//!
//! ## Determinism
//!
//! No RNG: colonization = **deterministic** rate × suitability. Cell-local
//! phenomenon on a double-buffer, independent of iteration order.

use serde::{Deserialize, Serialize};

use crate::cell::CellProperties;
use crate::climate_normals::CellClimateNormals;
use crate::grid::HexGrid;
use crate::species::{SPECIES, SPECIES_COUNT, SpeciesId};

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
    /// Surplus of free water above `water_capacity` (mm) beyond which the
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
            open_water_excess: 3.0,
        }
    }
}

/// Total plant cover of a cell (sum of biomasses per species).
#[must_use]
pub fn cell_total_vegetation(cell: &CellProperties) -> f32 {
    cell.vegetation.iter().sum()
}

/// Dominant species (highest biomass) of a cell, or `None` if the soil is
/// bare. Derived, never stored; single source of truth for diags / front.
#[must_use]
pub fn dominant_species(cell: &CellProperties) -> Option<SpeciesId> {
    let (idx, &max) = cell
        .vegetation
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    (max > 1e-4).then(|| SPECIES[idx].id)
}

/// Vegetation phenomenon (Tier 3, 1x/day). Evolves the per-species biomass
/// of each cell: growth/colonization weighted by suitability and free
/// space, mortality (background + lethal die-off outside the niche). Pure:
/// reads `current`, writes `next`.
///
/// `normals` = per-cell climate normals (#79), indexed like
/// `current.cells_slice()`. As long as no year has completed, they hold
/// the default value, so suitability is zero and vegetation stays at 0
/// (bootstrap after the first year, lag accepted).
pub(crate) fn step_vegetation(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &VegetationParams,
    normals: &[CellClimateNormals],
) {
    let cur = current.cells_slice();
    next.cells_slice_mut().clone_from_slice(cur);
    let next_cells = next.cells_slice_mut();

    for (i, cell) in cur.iter().enumerate() {
        let veg = cell.vegetation;

        // Open water (lake): no terrestrial vegetation, biomass recedes.
        if cell.water_level - cell.water_capacity > params.open_water_excess {
            for (nv, &v) in next_cells[i].vegetation.iter_mut().zip(veg.iter()) {
                *nv = (v - params.lethal_mortality * v).max(0.0);
            }
            next_cells[i].stand_age = 0.0; // lake: no terrestrial canopy.
            continue;
        }

        let normals_i = normals.get(i).copied().unwrap_or_default();
        let occupied: f32 = veg.iter().sum();
        // Shared logistic occupation limitation: zero growth once cover
        // saturates `k_total`. Not a toxic cap (anti-pattern #4), just the
        // physically available space.
        let free = (1.0 - occupied / params.k_total.max(1e-6)).max(0.0);

        // 1) Growth / colonization / mortality, per species → `newv`.
        let mut newv = [0.0_f32; SPECIES_COUNT];
        let mut suits = [0.0_f32; SPECIES_COUNT];
        for (s, ((species, &v), suit_slot)) in SPECIES
            .iter()
            .zip(veg.iter())
            .zip(suits.iter_mut())
            .enumerate()
        {
            let suit = species.suitability(&normals_i);
            *suit_slot = suit;
            let growth = params.growth_rate * v * suit * free;
            let colonization = params.colonization_rate * suit * free;
            let mut mortality = params.base_mortality * v;
            if suit <= 0.0 {
                mortality += params.lethal_mortality * v;
            }
            newv[s] = (v + growth + colonization - mortality).max(0.0);
        }

        // 2) Succession (#85): conservative intra-cell flux, a more
        // shade-tolerant species `i` takes biomass from a less tolerant one
        // `j` where its niche is good. `Σ newv` unchanged → cover always
        // bounded. Computed on the `pre` snapshot (order-independent).
        let pre = newv;
        for (i_idx, (&pre_i, &suit_i)) in pre.iter().zip(suits.iter()).enumerate() {
            for (j_idx, &pre_j) in pre.iter().enumerate() {
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

        // Average canopy age: existing biomass ages +1 day, new biomass
        // (net growth + colonization) enters at age 0 and dilutes the
        // average. Succession conserves the sum → b1 = post-growth total.
        // If the cover collapses (post-fire), age drops back to 0.
        let b1: f32 = newv.iter().sum();
        let aged = cell.stand_age + 1.0 / 365.0;
        next_cells[i].stand_age = if b1 > 1e-4 {
            aged * (occupied / b1).min(1.0)
        } else {
            0.0
        };
    }
}

/// `true` if the cell is open water (lake): water surplus above
/// `water_capacity` beyond the `open_water_excess` threshold. No
/// terrestrial vegetation. Single source of truth (anti-pattern #2); the
/// front consumes this flag, it does not re-derive the threshold.
#[must_use]
pub fn is_open_water(cell: &CellProperties) -> bool {
    cell.water_level - cell.water_capacity > VegetationParams::default().open_water_excess
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::species::{SPECIES_COUNT, SpeciesId};
    use proptest::prelude::*;

    /// Warm and DRY lowland normals (warm-dry collinean, Drôme-like): the
    /// minimum water drops below the lethal threshold of beech/fir, so
    /// these humid-mountain species are excluded, oak (drought-tolerant)
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

    /// Runs `step_vegetation` on 1 cell for `steps` days, with a given
    /// `succession_rate` (everything else = defaults).
    fn run_with(
        cell: CellProperties,
        normals: CellClimateNormals,
        steps: usize,
        succession_rate: f32,
    ) -> CellProperties {
        let c0 = HexCoord::new(0, 0);
        let mut grid = HexGrid::from_radius(0);
        *grid.get_mut(c0).unwrap() = cell;
        let params = VegetationParams {
            succession_rate,
            ..VegetationParams::default()
        };
        let norm = vec![normals];
        for _ in 0..steps {
            let mut next = grid.clone();
            step_vegetation(&grid, &mut next, &params, &norm);
            grid = next;
        }
        grid.get(c0).unwrap().clone()
    }

    /// `run_with` at the default succession rate.
    fn run(cell: CellProperties, normals: CellClimateNormals, steps: usize) -> CellProperties {
        run_with(
            cell,
            normals,
            steps,
            VegetationParams::default().succession_rate,
        )
    }

    fn favorable_cell() -> CellProperties {
        CellProperties {
            water_capacity: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn bare_ground_colonizes_under_good_climate() {
        // Bare soil under lowland climate: cover establishes through
        // colonization.
        let cell = run(favorable_cell(), warm_plain(), 400);
        assert!(
            cell_total_vegetation(&cell) > 0.3,
            "cover expected, got {}",
            cell_total_vegetation(&cell)
        );
    }

    #[test]
    fn outside_all_niches_stays_bare() {
        // Glacial summit (t_min -45 °C, below the lethal frost of every
        // species): none can establish.
        let frozen = CellClimateNormals {
            t_mean: -3.0,
            t_min: -45.0,
            t_max: 6.0,
            moisture_mean: 5.0,
            moisture_min: 1.0,
            moisture_max: 20.0,
            insolation_mean: 140.0,
        };
        let cell = run(favorable_cell(), frozen, 400);
        assert!(
            cell_total_vegetation(&cell) < 0.05,
            "frozen rock should stay bare, got {}",
            cell_total_vegetation(&cell)
        );
        assert!(
            dominant_species(&cell).is_none(),
            "no species should hold on"
        );
    }

    #[test]
    fn lethal_climate_kills_established_biomass() {
        // Established beech (index 2), then extreme drought climate
        // (min water 0, below its lethal threshold 1.0): its biomass
        // regresses.
        let mut start = favorable_cell();
        start.vegetation[2] = 0.5; // Beech
        let dry = CellClimateNormals {
            t_mean: 16.0,
            t_min: -3.0,
            t_max: 33.0,
            moisture_mean: 1.0,
            moisture_min: 0.0,
            moisture_max: 6.0,
            insolation_mean: 170.0,
        };
        let before = 0.5;
        let cell = run(start, dry, 60);
        assert!(
            cell.vegetation[2] < before * 0.5,
            "beech should die under drought: {} → {}",
            before,
            cell.vegetation[2]
        );
    }

    #[test]
    fn warm_plain_favours_warm_species() {
        // In warm lowland, the dominant is a lowland species (oak or pine),
        // never a cold species (fir / alpine grass).
        let cell = run(favorable_cell(), warm_plain(), 400);
        let dom = dominant_species(&cell).expect("nonzero cover");
        assert!(
            matches!(dom, SpeciesId::OakPubescent | SpeciesId::Pine),
            "warm plain dominated by {dom:?}"
        );
    }

    #[test]
    fn cold_highland_favours_cold_species() {
        // At cold altitude, the dominant is a cold species (fir or alpine
        // grass), never oak.
        let cell = run(favorable_cell(), cold_highland(), 400);
        let dom = dominant_species(&cell).expect("nonzero cover");
        assert!(
            matches!(dom, SpeciesId::Fir | SpeciesId::AlpineGrass),
            "cold altitude dominated by {dom:?}"
        );
    }

    #[test]
    fn succession_favours_shade_tolerant() {
        // Pine (pioneer, idx 1) + fir (tolerant climax, idx 3) in cold
        // climate. Succession must give more ground to fir than pure
        // "shared space" competition would (succession_rate = 0).
        let mut start = favorable_cell();
        start.vegetation[1] = 0.3; // pine
        start.vegetation[3] = 0.1; // fir (regeneration)
        let with = run_with(start.clone(), cold_highland(), 300, 0.05);
        let without = run_with(start, cold_highland(), 300, 0.0);
        assert!(
            with.vegetation[3] > without.vegetation[3],
            "succession should favor fir (tolerant): with={} without={}",
            with.vegetation[3],
            without.vegetation[3]
        );
    }

    /// Biomass share of a species in the total cover (0 if bare soil).
    fn share(cell: &CellProperties, id: SpeciesId) -> f32 {
        let total = cell_total_vegetation(cell);
        if total < 1e-6 {
            return 0.0;
        }
        cell.vegetation[id as usize] / total
    }

    #[test]
    fn forest_matures_to_shade_tolerant_climax_over_200y() {
        // A SINGLE hex, bare soil, constant cold-humid climate, 200 years.
        // Succession (rate 0.20) must make a shade-tolerant climax emerge:
        // fir/beech (shade 0.85-0.9) take ground from the pioneers (pine
        // 0.1, grassland 0.25). We compare the climax share at an early
        // horizon (5 years) and at maturity (200 years): succession is a
        // transfer that accumulates.
        let early = run(favorable_cell(), cold_highland(), 5 * 365);
        let climax = run(favorable_cell(), cold_highland(), 200 * 365);

        let dom = dominant_species(&climax).expect("nonzero cover at climax");
        assert!(
            matches!(dom, SpeciesId::Fir | SpeciesId::Beech),
            "cold-humid climax expected shade-tolerant (fir/beech), got {dom:?}"
        );
        let climax_share = share(&climax, SpeciesId::Fir) + share(&climax, SpeciesId::Beech);
        let early_share = share(&early, SpeciesId::Fir) + share(&early, SpeciesId::Beech);
        assert!(
            climax_share > early_share,
            "shade-tolerant share should grow with succession: 5y={early_share:.3}, 200y={climax_share:.3}"
        );
        // Established cover and mature canopy (stand_age accumulates over
        // a stable stand).
        assert!(
            cell_total_vegetation(&climax) > 0.5,
            "mature forest cover expected, got {}",
            cell_total_vegetation(&climax)
        );
        assert!(
            climax.stand_age > 30.0,
            "mature canopy expected after 200 years, age {}",
            climax.stand_age
        );
    }

    #[test]
    fn warming_flips_dominant_from_cold_to_warm_species() {
        // Mature cold forest (fir/beech), then the climate flips to
        // warm-dry: the cold-humid species fall outside their niche
        // (heatwave > fir's lethal threshold, drought) and die, a lowland
        // species (oak/pine) takes over. This is "vary the temperature,
        // the cover changes".
        let matured = run(favorable_cell(), cold_highland(), 60 * 365);
        let cold_dom = dominant_species(&matured).expect("nonzero cold cover");
        assert!(
            matches!(
                cold_dom,
                SpeciesId::Fir | SpeciesId::Beech | SpeciesId::AlpineGrass
            ),
            "initial state: cold dominant expected, got {cold_dom:?}"
        );
        // Same cell, continuing under warm-dry climate.
        let warmed = run(matured, warm_plain(), 40 * 365);
        let warm_dom = dominant_species(&warmed).expect("nonzero warm cover");
        assert!(
            matches!(warm_dom, SpeciesId::OakPubescent | SpeciesId::Pine),
            "after warming, lowland dominant expected, got {warm_dom:?}"
        );
        assert_ne!(
            cold_dom, warm_dom,
            "dominant should have changed with the climate"
        );
    }

    #[test]
    fn open_water_clears_vegetation() {
        // Vegetated cell that becomes a lake: vegetation recedes to 0.
        let mut start = favorable_cell();
        start.vegetation = [0.2; SPECIES_COUNT];
        start.water_level = 50.0; // >> capacity + open_water_excess
        let cell = run(start, warm_plain(), 200);
        assert!(
            cell_total_vegetation(&cell) < 1e-3,
            "lake should be free of vegetation, got {}",
            cell_total_vegetation(&cell)
        );
        assert!(is_open_water(&cell));
    }

    proptest! {
        /// Finite biomass, per species ≥ 0, total cover bounded by k_total,
        /// for any plausible input (no NaN, no overflow).
        #[test]
        fn prop_biomass_bounded(
            t_mean in -40.0_f32..40.0,
            t_min in -50.0_f32..0.0,
            t_max in 0.0_f32..55.0,
            moisture in 0.0_f32..60.0,
            insol in 0.0_f32..400.0,
            v0 in prop::array::uniform5(0.0_f32..0.18),
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
            let mut next = grid.clone();
            step_vegetation(&grid, &mut next, &VegetationParams::default(), &normals);
            let veg = next.get(c0).unwrap().vegetation;
            let total: f32 = veg.iter().sum();
            for &v in &veg {
                prop_assert!(v.is_finite() && v >= 0.0, "v hors borne : {v}");
            }
            prop_assert!(total <= 1.0 + 1e-3, "couvert total {total} > k_total");
        }
    }
}
