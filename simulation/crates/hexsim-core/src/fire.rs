//! Emergent forest fires (#wildfire).
//!
//! No scripted events: risk **emerges** from local state. Old dense forest
//! (fuel) + drought + heat (hot/dry day) raises ignition probability, kept
//! **very low** (~1 start / 3 years on map). Fire spreads cell-by-cell at
//! probability **below percolation threshold** (hex grid, z=6) and self-extinguishes:
//! burned cell loses biomass, becomes firebreak for years. Subsequent pioneer
//! recolonization restarts succession (cycle, not plateau).
//!
//! ## Closed world
//!
//! `step_fire` **never** touches water directly. It destroys biomass (non-conserved
//! stock, like mortality) and injects **combustion heat** (legitimate energy source,
//! like sun) into `temperature`. Desiccation then emerges via existing
//! `step_evaporation` (`water_level`/`groundwater` → `humidity_surface`, conserving),
//! then advection. No water-logic duplication (anti-pattern #2).
//!
//! ## Determinism
//!
//! Randomness = hash of `(seed, day, q, r, salt)` → reproducible and
//! order-independent (read from `current`, write to `next`). "One seed = one world"
//! preserved.

use serde::{Deserialize, Serialize};

use crate::grid::HexGrid;
use crate::temperature::local_heat_capacity;
use crate::vegetation::canopy_cover;

/// Combustion enthalpy of dry wood (J/kg). Standard value ~18 MJ/kg.
const COMBUSTION_ENTHALPY_J_PER_KG: f32 = 18.0e6;

/// Fire parameters. `enabled = false` by default: fire is dormant until a
/// consumer activates it (live server does; historical tests unchanged).
/// Probability coefficients are **hazard rates** per day, decomposed into named
/// factors (fuel, drought) rather than opaque coefficient.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FireParams {
    /// Master switch. `false` = no fire (no-op).
    pub enabled: bool,
    /// Base ignition rate per cell per day, before modulation by fuel × drought.
    /// Very small (calibrated ~1 start / 3 years).
    pub ignition_rate: f32,
    /// Spread coefficient per burning neighbor, before modulation by fuel × drought
    /// of target neighbor. Kept sub-critical.
    pub spread_rate: f32,
    /// Root water (`groundwater + water_level`, mm) beyond which drought is zero
    /// (no ignition). Linear below.
    pub moisture_ref_mm: f32,
    /// Temperature (C) below which heat doesn't contribute to ignition (low ramp).
    pub temp_ignite_lo: f32,
    /// Temperature (C) above which heat factor saturates to 1.
    pub temp_ignite_hi: f32,
    /// Age (years) at which flammability reaches half its maximum (saturating):
    /// young regrowth ≈ non-flammable, old forest ≈ 1.
    pub fuel_age_half_years: f32,
    /// Fraction of biomass consumed per day when cell burns.
    pub combustion_fraction_per_day: f32,
    /// Residual cover (`vegetation::canopy_cover`) below which fire
    /// extinguishes (no more fuel).
    pub extinguish_fuel_min: f32,
    /// Dry fuel load per unit of cover seen from above
    /// (`vegetation::canopy_cover`, kg/m²). Forest assumption.
    pub fuel_load_kg_per_m2: f32,
    /// Fraction of combustion energy that heats ground locally (rest goes to
    /// plume/radiation to atmosphere). Small.
    pub combustion_heat_ground_fraction: f32,
}

impl Default for FireParams {
    fn default() -> Self {
        Self {
            enabled: false,
            // Calibrated (seed 42, radius 30): 2e-4 → ~1.5 lightning starts/year for
            // peak_burning=5 (never near map). 4e-5 → ~1 start / 3 years, meets
            // "very low risk" target. Spread 0.12 stays well below hex percolation
            // threshold → small fires, self-extinguish.
            ignition_rate: 4.0e-5,
            spread_rate: 0.12,
            moisture_ref_mm: 60.0,
            temp_ignite_lo: 18.0,
            temp_ignite_hi: 33.0,
            fuel_age_half_years: 15.0,
            // #92: near-total combustion per day. At 0.4 (multiplicative ×0.6),
            // cell took ~5 days to drop below `extinguish_fuel_min`, letting absolute
            // vegetation colonization (~0.026/day, step_vegetation just before, buffer
            // read by fire) recharge it: post-fire cover stabilized at fixed point
            // x* = col·(1−c)/c ABOVE 0.02 → perpetual fire. Critical threshold c ≈ 0.57;
            // at 0.85, x* ≈ 0.005 ≪ 0.02, fire empties cell in ~2-4 days and dies.
            // Fire becomes propagating front, no longer "smolders" in place.
            combustion_fraction_per_day: 0.85,
            extinguish_fuel_min: 0.02,
            fuel_load_kg_per_m2: 3.0,
            combustion_heat_ground_fraction: 0.05,
        }
    }
}

/// Deterministic hash `(seed, day, q, r, salt)` → `[0, 1)`. `salt`
/// separates independent streams (ignition vs spread). The hash itself
/// lives in [`crate::hashing`], shared with the weather regime (#63)
/// rather than written a second time; the word order below is the one
/// this phenomenon has always used, so its stream is unchanged.
#[must_use]
fn hash01(seed: u32, day: u64, q: i32, r: i32, salt: u32) -> f32 {
    crate::hashing::hash01(&[
        u64::from(seed),
        day,
        crate::hashing::coord_word(q),
        crate::hashing::coord_word(r),
        u64::from(salt),
    ])
}

/// Flammability ∈ [0, 1], saturating increase with canopy age (the mean
/// age of the tree stratum, `CellProperties::stand_age`).
#[must_use]
fn flammability(stand_age: f32, params: &FireParams) -> f32 {
    let a = stand_age.max(0.0);
    a / (a + params.fuel_age_half_years.max(1e-3))
}

/// Instantaneous dryness ∈ [0, 1]: heat × lack of root water. Both
/// must be combined (product) → ignition on days that are hot **and** dry.
#[must_use]
fn dryness(temperature: f32, root_water_mm: f32, params: &FireParams) -> f32 {
    let span = (params.temp_ignite_hi - params.temp_ignite_lo).max(1e-3);
    let heat = ((temperature - params.temp_ignite_lo) / span).clamp(0.0, 1.0);
    let water_dry = ((params.moisture_ref_mm - root_water_mm) / params.moisture_ref_mm.max(1e-3))
        .clamp(0.0, 1.0);
    heat * water_dry
}

/// Fuel of a cell = cover × flammability (age). The cover is the one seen
/// from above, `vegetation::canopy_cover` ∈ [0, 1] (#161): an understory
/// adds fuel only through the gaps of the layers above it, it does not
/// stack on the canopy (the plain sum of biomasses reaches 3 under a full
/// three-layer stand and would push the spread probability over its
/// sub-critical calibration).
#[must_use]
fn fuel(cover: f32, stand_age: f32, params: &FireParams) -> f32 {
    cover * flammability(stand_age, params)
}

/// Counters for one fire step, for calibration metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct FireTick {
    /// New starts from **lightning** this day (excluding spread).
    pub ignitions: u32,
    /// Cells on fire at the end of the step (intensity > 0).
    pub burning: u32,
}

/// One fire step (1 day). Reads `current`, writes `next` (cloned here to stay
/// self-contained like the other phenomena). `day` = simulated day for the
/// deterministic randomness. Returns the step's counters.
pub(crate) fn step_fire(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &FireParams,
    seed: u32,
    day: u64,
) -> FireTick {
    // next = copy of current (all fields), then fire is applied.
    next.cells_slice_mut()
        .clone_from_slice(current.cells_slice());
    let mut stats = FireTick::default();
    if !params.enabled {
        return stats;
    }

    let coords = current.coords_slice();
    let cur = current.cells_slice();

    // --- Pass 1: combustion of burning cells + lightning ignition. O(n), no
    // neighbor lookup (spread is handled next, only from the small
    // number of burning cells, fire is rare, this is the hot path).
    let mut sources: Vec<usize> = Vec::new();
    {
        let next_cells = next.cells_slice_mut();
        for (i, cell) in cur.iter().enumerate() {
            let coord = coords[i];
            let cover = canopy_cover(cell);
            let wet = cell.frozen_surface() > 1e-3;

            if cell.fire_intensity > 1e-3 {
                sources.push(i);
                let consume =
                    (params.combustion_fraction_per_day * cell.fire_intensity).clamp(0.0, 1.0);
                // Every stratum burns: the same fraction of every species'
                // biomass goes up in flames.
                for (nv, &v) in next_cells[i]
                    .vegetation
                    .iter_mut()
                    .zip(cell.vegetation.iter())
                {
                    *nv = (v - v * consume).max(0.0);
                }
                let remaining = canopy_cover(&next_cells[i]);
                // Combustion heat (energy source) → ΔT via the local heat
                // capacity shared with step_temperature. Does not alter water.
                // The load is per unit of cover seen from above, so the
                // burned fuel is the cover lost (the calibration of
                // `fuel_load_kg_per_m2` predates the strata, when the sum of
                // biomasses was that cover).
                let burned_cover = cover - remaining;
                let energy_j_per_m2 = burned_cover
                    * params.fuel_load_kg_per_m2
                    * COMBUSTION_ENTHALPY_J_PER_KG
                    * params.combustion_heat_ground_fraction;
                next_cells[i].temperature = cell.temperature
                    + energy_j_per_m2 / local_heat_capacity(cell.water_level, cell.groundwater);

                next_cells[i].fire_intensity = if remaining < params.extinguish_fuel_min || wet {
                    0.0
                } else {
                    1.0
                };
            } else {
                let root_water = cell.groundwater + cell.water_level;
                let dry = dryness(cell.temperature, root_water, params);
                let cell_fuel = fuel(cover, cell.stand_age, params);
                if !wet && cell_fuel > 1e-3 && dry > 0.0 {
                    let p_ign = params.ignition_rate * cell_fuel * dry;
                    if hash01(seed, day, coord.q, coord.r, 1) < p_ign {
                        next_cells[i].fire_intensity = 1.0;
                        stats.ignitions += 1; // a "start" = lightning.
                    }
                }
            }
        }
    }

    // --- Pass 2: spread from burning cells to their intact dry neighbors.
    // Cost O(burning_cells × 6), negligible since fire is rare.
    for &src in &sources {
        // Toric neighborhood: fire also spreads across the seam
        // (the world is a torus, there's no edge firebreak). The
        // draw stays deterministic: hash by (target cell, direction).
        for (dir, &nb) in current.neighbor_indices_toric(src).iter().enumerate() {
            if nb == src {
                continue; // degenerate grid without wrap: no target.
            }
            let nb_coord = coords[nb];
            let ncell = &cur[nb];
            if ncell.fire_intensity > 1e-3 || ncell.frozen_surface() > 1e-3 {
                continue; // already burning (source) or snow-covered.
            }
            if next.cells_slice()[nb].fire_intensity > 1e-3 {
                continue; // already ignited this tick (lightning or another neighbor).
            }
            let nb_fuel = fuel(canopy_cover(ncell), ncell.stand_age, params);
            let dry = dryness(
                ncell.temperature,
                ncell.groundwater + ncell.water_level,
                params,
            );
            if nb_fuel <= 1e-3 || dry <= 0.0 {
                continue;
            }
            let p1 = (params.spread_rate * nb_fuel * dry).clamp(0.0, 1.0);
            // Draw deterministic per (target cell, direction) → independent
            // of source order, OR'd across burning neighbors.
            let salt = 10 + u32::try_from(dir).unwrap_or(0);
            if hash01(seed, day, nb_coord.q, nb_coord.r, salt) < p1 {
                next.cells_slice_mut()[nb].fire_intensity = 1.0;
            }
        }
    }

    stats.burning = next
        .cells_slice()
        .iter()
        .filter(|c| c.fire_intensity > 1e-3)
        .count()
        .try_into()
        .unwrap_or(u32::MAX);
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellProperties;
    use crate::coord::HexCoord;
    use crate::species::{SPECIES_COUNT, SpeciesId, species_index};
    use crate::vegetation::{VegetationParams, stratum_cover};

    /// Old dense forest in three layers, each stratum within its budget
    /// (`k_total`): oak + pine canopy, boxwood understory, meadow.
    fn layered_forest() -> [f32; SPECIES_COUNT] {
        let mut v = [0.0; SPECIES_COUNT];
        for (id, b) in [
            (SpeciesId::OakPubescent, 0.6),
            (SpeciesId::Pine, 0.3),
            (SpeciesId::Boxwood, 0.4),
            (SpeciesId::Meadow, 0.5),
        ] {
            v[species_index(id)] = b;
        }
        v
    }

    fn grid_one_burning() -> (HexGrid, HexGrid) {
        let mut g = HexGrid::from_radius(2);
        // Everyone: old dense forest, dry, hot.
        for coord in g.coords().copied().collect::<Vec<_>>() {
            let c = g.get_mut(coord).unwrap();
            c.vegetation = layered_forest();
            c.stand_age = 100.0;
            c.temperature = 35.0;
            c.groundwater = 0.0;
            c.water_level = 0.0;
        }
        // Center on fire.
        g.get_mut(HexCoord::new(0, 0)).unwrap().fire_intensity = 1.0;
        let n = g.clone();
        (g, n)
    }

    #[test]
    fn disabled_is_noop() {
        let (cur, mut next) = grid_one_burning();
        let p = FireParams::default(); // enabled = false
        step_fire(&cur, &mut next, &p, 42, 100);
        // next identical to current (center stays at 1.0, nothing happens).
        for (a, b) in cur.cells_slice().iter().zip(next.cells_slice()) {
            assert!((a.fire_intensity - b.fire_intensity).abs() < 1e-9);
            assert_eq!(
                a.vegetation.map(f32::to_bits),
                b.vegetation.map(f32::to_bits)
            );
        }
    }

    #[test]
    fn burning_cell_consumes_fuel_and_heats() {
        let (cur, mut next) = grid_one_burning();
        let p = FireParams {
            enabled: true,
            ..Default::default()
        };
        step_fire(&cur, &mut next, &p, 42, 100);
        let c = next.get(HexCoord::new(0, 0)).unwrap();
        let before = cur.get(HexCoord::new(0, 0)).unwrap();
        assert!(
            canopy_cover(c) < canopy_cover(before),
            "fire must consume fuel"
        );
        // Every stratum burns, not only the canopy.
        for (&v0, &v1) in before.vegetation.iter().zip(c.vegetation.iter()) {
            assert!(v0 <= 0.0 || v1 < v0, "every species must burn: {v0} → {v1}");
        }
        assert!(
            c.temperature > before.temperature,
            "combustion must produce heat"
        );
    }

    #[test]
    fn fire_does_not_touch_water() {
        // Closed world: fire alters neither water, groundwater, nor
        // humidity directly.
        let (cur, mut next) = grid_one_burning();
        let p = FireParams {
            enabled: true,
            ..Default::default()
        };
        step_fire(&cur, &mut next, &p, 42, 100);
        for (a, b) in cur.cells_slice().iter().zip(next.cells_slice()) {
            assert!(
                (a.water_level - b.water_level).abs() < 1e-9,
                "water_level intact"
            );
            assert!(
                (a.groundwater - b.groundwater).abs() < 1e-9,
                "groundwater intact"
            );
            assert!(
                (a.humidity_surface - b.humidity_surface).abs() < 1e-9,
                "humidity intact"
            );
        }
    }

    #[test]
    fn deterministic_given_seed() {
        let (cur, mut n1) = grid_one_burning();
        let (_, mut n2) = grid_one_burning();
        let p = FireParams {
            enabled: true,
            ignition_rate: 0.5, // strong, to force distinct draws
            ..Default::default()
        };
        step_fire(&cur, &mut n1, &p, 7, 12);
        step_fire(&cur, &mut n2, &p, 7, 12);
        for (a, b) in n1.cells_slice().iter().zip(n2.cells_slice()) {
            assert!((a.fire_intensity - b.fire_intensity).abs() < 1e-9);
        }
    }

    #[test]
    fn fire_spreads_to_dry_neighbor() {
        // With a strong spread_rate, the center's neighbor catches fire.
        let (cur, mut next) = grid_one_burning();
        let p = FireParams {
            enabled: true,
            spread_rate: 1.0,
            ignition_rate: 0.0,
            ..Default::default()
        };
        step_fire(&cur, &mut next, &p, 1, 1);
        let neighbor_caught = HexCoord::new(0, 0)
            .neighbors()
            .iter()
            .any(|n| next.get(*n).is_some_and(|c| c.fire_intensity > 1e-3));
        assert!(neighbor_caught, "fire must spread to a dry neighbor");
    }

    #[test]
    fn fire_extinguishes_within_days_and_stays_out() {
        // Anti-regression guard for #92 (anti-pattern #4: multiplicative decay
        // that never reaches zero). A single burning hex, BARE neighbors (no
        // fuel → no spread or re-ignition possible), no spontaneous
        // ignition. Fire must burn through its stock then go OUT within a
        // bounded number of days, and stay out.
        let center = HexCoord::new(0, 0);
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            let c = grid.get_mut(coord).unwrap();
            c.temperature = 35.0; // hot and dry: nothing slows combustion
            c.groundwater = 0.0;
            c.water_level = 0.0;
            c.vegetation = [0.0; SPECIES_COUNT]; // bare neighbors by default
        }
        // Only the center carries fuel, and it burns.
        let c = grid.get_mut(center).unwrap();
        c.vegetation = layered_forest();
        c.stand_age = 100.0;
        c.fire_intensity = 1.0;

        let params = FireParams {
            enabled: true,
            ignition_rate: 0.0, // no spontaneous ignition: isolate extinction
            ..Default::default()
        };
        let mut next = grid.clone();
        // 90-day margin: combustion (85 %/day of every stratum) exhausts
        // the ~0.97 canopy cover of the layered forest and drops below
        // extinguish_fuel_min well before that.
        let mut extinguished_on = None;
        for day in 0..90 {
            step_fire(&grid, &mut next, &params, 42, day);
            std::mem::swap(&mut grid, &mut next);
            if grid.get(center).unwrap().fire_intensity < 1e-6 {
                extinguished_on = Some(day);
                break;
            }
        }
        let day = extinguished_on.expect("the fire never went out (regression #92)");
        // It doesn't re-ignite on its own once fuel is below the threshold.
        for d in day..day + 30 {
            step_fire(&grid, &mut next, &params, 42, d);
            std::mem::swap(&mut grid, &mut next);
            assert!(
                grid.get(center).unwrap().fire_intensity < 1e-6,
                "re-ignition without external ignition at day {d} (the stock must reach true zero)"
            );
        }
    }

    #[test]
    fn phys_understory_adds_fuel_only_through_the_canopy_gaps() {
        // #161: the fuel cover is the cover seen from above. A full
        // understory (boxwood 0.9 + meadow 0.95) adds fuel only through the
        // gaps of the canopy: at most the 5 % gap of a closed oak canopy
        // (0.95), most of the 80 % gap of an open one (0.2). The fuel never
        // exceeds the flammability (cover ≤ 1): the plain sum of biomasses
        // (2.8 here) would push the spread probability far past its
        // sub-critical calibration.
        let p = FireParams::default();
        let age = 100.0;
        let with_ids = |seed: &[(SpeciesId, f32)]| {
            let mut c = CellProperties::default();
            for &(id, v) in seed {
                c.vegetation[species_index(id)] = v;
            }
            c
        };
        let (closed_oak, open_oak) = (0.95, 0.2);
        let understory = [(SpeciesId::Boxwood, 0.9), (SpeciesId::Meadow, 0.95)];
        let closed = with_ids(&[(SpeciesId::OakPubescent, closed_oak)]);
        let closed_layered = with_ids(&[
            (SpeciesId::OakPubescent, closed_oak),
            understory[0],
            understory[1],
        ]);
        let open = with_ids(&[(SpeciesId::OakPubescent, open_oak)]);
        let open_layered = with_ids(&[
            (SpeciesId::OakPubescent, open_oak),
            understory[0],
            understory[1],
        ]);
        let k_total = VegetationParams::default().k_total;
        for s in crate::species::STRATA {
            assert!(stratum_cover(&closed_layered, s) <= k_total);
        }
        let fuel_of = |c: &CellProperties| fuel(canopy_cover(c), age, &p);
        let flam = flammability(age, &p);
        let closed_gain = fuel_of(&closed_layered) - fuel_of(&closed);
        let open_gain = fuel_of(&open_layered) - fuel_of(&open);
        assert!(
            fuel_of(&closed_layered) <= flam,
            "fuel {} over the flammability {flam}",
            fuel_of(&closed_layered)
        );
        assert!(
            closed_gain <= (1.0 - closed_oak) * flam + 1e-6,
            "an understory under a closed canopy adds at most its gaps, got {closed_gain}"
        );
        assert!(
            open_gain <= (1.0 - open_oak) * flam + 1e-6 && open_gain > 10.0 * closed_gain,
            "under an open canopy the understory should be most of the fuel: open {open_gain}, closed {closed_gain}"
        );
    }
}
