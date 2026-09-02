use serde::{Deserialize, Serialize};

use crate::lithology::LithologyId;
use crate::species::SPECIES_COUNT;
use crate::units::MM_PER_M;

/// Properties of a hexagonal cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellProperties {
    pub elevation: f32,
    pub temperature: f32,
    pub water_level: f32,
    /// Volume of water trapped locally (sub-hex) before it starts raising the
    /// effective surface. As long as `water_level <= water_capacity`, the water
    /// is a puddle invisible to the topology (does not flow, does not raise
    /// `effective_elevation`). Beyond that, the surplus acts as open water.
    pub water_capacity: f32,
    /// Freshly evaporated vapor, trapped at ground level. Not precipitable
    /// until it has been lifted to `humidity_upper` by uplift
    /// (thermal + orographic).
    pub humidity_surface: f32,
    /// Altitude vapor (invisible), advected by upper-level wind.
    /// Condenses into `cloud_water` when relative humidity approaches
    /// saturation. Does not precipitate directly, must first become a
    /// droplet.
    pub humidity_upper: f32,
    /// Liquid water droplets in suspension (visible clouds).
    /// Produced by condensation of `humidity_upper` when RH > 0.6.
    /// Return to `humidity_upper` by evaporation if RH drops back below 0.4.
    /// Precipitate (rain or snow) once they exceed a critical
    /// collision/coalescence threshold. This is the stock the UI renders as
    /// clouds.
    pub cloud_water: f32,
    pub groundwater: f32,
    pub snow_level: f32,
    /// Rock class of the substrate (#136, tier L0). **Static**: set at
    /// tick 0 by `terrain::generate_terrain` from the substrate noise and
    /// the relief, never modified afterward (exhumation by erosion is
    /// deferred to later work).
    ///
    /// This is now the **source** of `permeability`, the anonymous noise that
    /// used to carry it now sits behind the `lithology::LITHOLOGY` table.
    /// `serde(default)`: checkpoints predating #136 → sandstone (median
    /// class), without which they would refuse to load.
    #[serde(default)]
    pub lithology: LithologyId,
    /// Hydric aptitude of the cell ∈ [0, 1]: groundwater capacity
    /// (`groundwater`) and retention under snow (`snow`). Derived from
    /// [`Self::lithology`] via the table, attenuated by relief (thinner soil
    /// at altitude), see `terrain::TerrainSampler::sample`.
    pub permeability: f32,
    /// Plant biomass **per species** (slow stock, indexed like
    /// `species::SPECIES`). Each component ∈ [0, 1]; total cover
    /// (sum) stays ≤ `VegetationParams::k_total`. Emerges from climate via
    /// `vegetation::step_vegetation` (competition for shared space, #81).
    pub vegetation: [f32; SPECIES_COUNT],
    /// Average canopy age (years). Ages with time, diluted by new
    /// biomass (colonization/growth), reset to ~0 by fire. Proxy
    /// for "old-growth forest" → drives flammability (`fire::step_fire`, #wildfire).
    pub stand_age: f32,
    /// Intensity of the ongoing fire [0, 1]; 0 = no fire. Transient stock
    /// (ignition → spread → extinction) managed by `fire::step_fire`.
    pub fire_intensity: f32,
    /// Sediment load in transit in the water column (m of rock
    /// equivalent over the cell's area, `ρ_rock` ≈ 2650 kg/m³ for the
    /// mass correspondence). Produced by bedrock incision
    /// (`erosion::step_erosion`), routed downstream with the water, becomes
    /// `elevation` again on deposit. Does NOT contribute to topography (neither
    /// `effective_elevation` nor thermal lapse): it is matter in
    /// suspension, not a deposited layer. Terrarium invariant:
    /// Σ(`elevation`) + Σ(`sediment_load`) is conserved.
    /// `serde(default)`: checkpoints predating #105 → 0.
    #[serde(default)]
    pub sediment_load: f32,
    /// East component of the surface normal (ENU, dimensionless). Precomputed
    /// from the elevation gradient over the 6 neighbors, see
    /// `temperature::compute_surface_normals`, recomputed after each effective
    /// erosion step (#105), elevation is no longer frozen. Drives
    /// sunlight exposure depending on slope orientation (sunny/shaded slope) via
    /// `cos(incidence) = S⃗·N⃗`. Flat cell ⇒ (0, 0) = vertical normal.
    pub normal_east: f32,
    /// North component of the surface normal (ENU, dimensionless). WARNING:
    /// astronomical north, not the world axis (where +y = South). A south-facing
    /// slope (sunny slope) has `normal_north < 0`. See `normal_east`.
    pub normal_north: f32,
}

impl Default for CellProperties {
    fn default() -> Self {
        Self {
            elevation: 0.0,
            temperature: 0.0,
            water_level: 0.0,
            water_capacity: 1.0,
            humidity_surface: 0.0,
            humidity_upper: 0.0,
            cloud_water: 0.0,
            groundwater: 0.0,
            snow_level: 0.0,
            lithology: LithologyId::default(),
            permeability: 0.0,
            vegetation: [0.0; SPECIES_COUNT],
            stand_age: 0.0,
            fire_intensity: 0.0,
            sediment_load: 0.0,
            normal_east: 0.0,
            normal_north: 0.0,
        }
    }
}

impl CellProperties {
    /// Effective elevation in **SI meters**: terrain (m) + height of the open
    /// water sheet above capacity (surplus in mm, converted to m).
    /// Water trapped under `water_capacity` does not raise the surface, it is a
    /// sub-hex puddle. Only the excess acts topologically.
    ///
    /// Before #104 the surplus (mm) was added directly to the terrain's
    /// meters: 100 mm of water offset 100 m of relief, hence stable
    /// water bodies on slopes ("lakes on slopes", diag #103). The hydrostatic
    /// equilibrium is now a genuine flat free surface in m.
    #[must_use]
    pub fn effective_elevation(&self) -> f32 {
        self.elevation + (self.water_level - self.water_capacity).max(0.0) / MM_PER_M
    }

    /// Total humidity of the atmospheric column (surface + upper + droplets).
    /// Used for conservation tests.
    #[must_use]
    pub fn humidity_total(&self) -> f32 {
        self.humidity_surface + self.humidity_upper + self.cloud_water
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn default_is_zero() {
        let c = CellProperties::default();
        assert!(c.elevation.abs() < f32::EPSILON);
        assert!(c.temperature.abs() < f32::EPSILON);
        assert!(c.water_level.abs() < f32::EPSILON);
        assert!((c.water_capacity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn effective_elevation_trapped_below_capacity() {
        let c = CellProperties {
            elevation: 100.0,
            water_level: 0.5,
            water_capacity: 1.0,
            ..Default::default()
        };
        assert!((c.effective_elevation() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn effective_elevation_surplus_above_capacity() {
        // 2000 mm of surplus = 2 m of water sheet above the terrain (#104).
        let c = CellProperties {
            elevation: 100.0,
            water_level: 3000.0,
            water_capacity: 1000.0,
            ..Default::default()
        };
        assert!((c.effective_elevation() - 102.0).abs() < f32::EPSILON);
    }

    /// SI pin (#104): the surplus is a water sheet in mm, not meters.
    /// 100 mm of open water only raises the surface by 0.1 m, a 1 m mound
    /// still dominates the water table. In the hybrid space it was the
    /// opposite (100 mm ≡ 100 m), the "flat eff" equilibrium then stacked
    /// ~1 mm of water per meter of elevation change: the "lakes on slopes"
    /// from diag #103.
    #[test]
    fn effective_elevation_hundred_mm_are_not_hundred_meters() {
        let nappe = CellProperties {
            elevation: 0.0,
            water_level: 100.0,
            water_capacity: 0.0,
            ..Default::default()
        };
        let butte = CellProperties {
            elevation: 1.0,
            water_level: 0.0,
            ..Default::default()
        };
        assert!((nappe.effective_elevation() - 0.1).abs() < 1e-6);
        assert!(nappe.effective_elevation() < butte.effective_elevation());
    }

    proptest! {
        #[test]
        fn prop_effective_elevation_is_elev_plus_surplus_in_meters(
            elev in -1000.0_f32..3000.0,
            wl in 0.0_f32..100_000.0,
            wc in 0.0_f32..10.0,
        ) {
            let cell = CellProperties {
                elevation: elev,
                water_level: wl,
                water_capacity: wc,
                ..Default::default()
            };
            let expected = elev + (wl - wc).max(0.0) / MM_PER_M;
            prop_assert!((cell.effective_elevation() - expected).abs() < 1e-3);
        }
    }
}
