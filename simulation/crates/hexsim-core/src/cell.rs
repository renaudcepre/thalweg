use std::fmt;

use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::lithology::LithologyId;
use crate::species::SPECIES_COUNT;
use crate::units::{Meters, Mm};

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
    /// Root-zone soil water (mm): what plants transpire, what the
    /// infiltration fills and what drains laterally above field capacity.
    pub groundwater: f32,
    /// Deep aquifer below the root zone (mm), out of reach of the roots
    /// (#107). Recharged by percolation of the root zone's water above
    /// field capacity, drained to the surface by Maillet's recession: the
    /// slow reservoir that carries a river between two rains.
    /// `serde(default)`: checkpoints predating it load with an empty
    /// aquifer.
    #[serde(default)]
    pub aquifer: f32,
    /// Snowpack (mm of water equivalent): fallen snow plus soil water
    /// frozen in place. The frozen surplus of a water body is NOT here,
    /// see [`Self::ice_level`].
    pub snow_level: f32,
    /// Frozen free surface water of a **water body** (lake or river ice),
    /// mm of water equivalent. Written by `snow::step_snow` when the liquid
    /// surplus above `water_capacity` freezes, melts back into
    /// `water_level`. Kept apart from `snow_level` (the snowpack, which
    /// also settles on top of the ice) so that the open-water identity
    /// follows the water and not its phase: a frozen lake is still a lake
    /// ([`Self::water_body_surplus`]), a snowy slope is not.
    /// `serde(default)`: checkpoints predating the split load with 0, their
    /// lake ice sits in `snow_level` until the next thaw.
    #[serde(default)]
    pub ice_level: f32,
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
    /// `species::SPECIES`). Each component ∈ [0, 1]; the sum over the
    /// species of one **stratum** (herb, shrub, tree) stays ≤
    /// `VegetationParams::k_total`, so a cell can carry a full canopy over
    /// a full understory (#161). Emerges from climate via
    /// `vegetation::step_vegetation` (competition for the space of each
    /// stratum, light through the strata above).
    ///
    /// Deserializes from a row of **any** length up to `SPECIES_COUNT`
    /// (`deserialize_species_columns`): a checkpoint written when the
    /// model had fewer species (5 before #161) must still load. The row is
    /// read raw, in the file's column order; `Checkpoint::decode` then
    /// moves each column to its species' place by id. Serialization is
    /// the plain fixed array.
    #[serde(deserialize_with = "deserialize_species_columns")]
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

/// Reads a per-species row of **any** length up to `SPECIES_COUNT` and
/// zero-pads the missing columns. A shorter row is an older engine with
/// fewer species (the 42-year `frontend/worlds/aged.ckptz` the public
/// embed booted on until v0.14.0 carries 5): its columns land raw, in the file's order,
/// and `Checkpoint::decode` remaps them by species id. A longer row comes
/// from an engine that knows species this one doesn't: refused, since
/// there is no column to put their biomass in and dropping it would
/// silently destroy a stock.
fn deserialize_species_columns<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<[f32; SPECIES_COUNT], D::Error> {
    struct Columns;

    impl<'de> Visitor<'de> for Columns {
        type Value = [f32; SPECIES_COUNT];

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "at most {SPECIES_COUNT} per-species vegetation columns (a longer row was \
                 written by an engine with more species than this one)"
            )
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut columns = [0.0_f32; SPECIES_COUNT];
            let mut len = 0;
            while let Some(v) = seq.next_element::<f32>()? {
                if let Some(slot) = columns.get_mut(len) {
                    *slot = v;
                }
                len += 1;
            }
            if len > SPECIES_COUNT {
                return Err(serde::de::Error::invalid_length(len, &self));
            }
            Ok(columns)
        }
    }

    deserializer.deserialize_seq(Columns)
}

/// Surplus of free water above `water_capacity` (mm), liquid or frozen,
/// beyond which a cell is a water body: a lake or a river reach, no
/// terrestrial vegetation, freezes from its surface. 3 mm: below that a
/// hex holds a drizzle trace, not a pond visible at 1.46 ha. Default of
/// `VegetationParams::open_water_excess`; the live parameter tunes the
/// vegetation branch, this constant is the one every other reader uses.
pub const OPEN_WATER_EXCESS_MM: f32 = 3.0;

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
            aquifer: 0.0,
            snow_level: 0.0,
            ice_level: 0.0,
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
    ///
    /// Routed through `Mm`/`Meters` (units.rs) rather than a bare division:
    /// the compiler, not a re-read of this function, is what now rejects
    /// adding `water_level` (mm) straight to `elevation` (m).
    #[must_use]
    pub fn effective_elevation(&self) -> f32 {
        let surplus = (Mm(self.water_level) - Mm(self.water_capacity)).non_negative();
        (Meters(self.elevation) + surplus.to_meters()).0
    }

    /// Total humidity of the atmospheric column (surface + upper + droplets).
    /// Used for conservation tests.
    #[must_use]
    pub fn humidity_total(&self) -> f32 {
        self.humidity_surface + self.humidity_upper + self.cloud_water
    }

    /// Frozen stock lying on the surface (mm w.e.): snowpack + lake ice.
    /// What the radiative balance sees as white (`temperature::balance`),
    /// what melt and sublimation can draw from, what fire treats as wet.
    #[must_use]
    pub fn frozen_surface(&self) -> f32 {
        self.snow_level + self.ice_level
    }

    /// Free water of a water body above the retention capacity (mm), liquid
    /// **or frozen**: `water_level + ice_level − water_capacity`. The stock
    /// that makes the cell a lake or a river whatever the season. The
    /// open-water identity (`vegetation::is_open_water`) reads this and
    /// never the liquid surplus alone, otherwise a lake stops being a lake
    /// the week it freezes and gets colonized (seen 2026-09-05). Negative
    /// when the cell holds less than its capacity.
    #[must_use]
    pub fn water_body_surplus(&self) -> f32 {
        self.water_level + self.ice_level - self.water_capacity
    }

    /// `true` if the cell is a water body (lake, river reach): its surplus,
    /// liquid or frozen, exceeds [`OPEN_WATER_EXCESS_MM`]. The one predicate
    /// behind `vegetation::is_open_water`, the lake branch of
    /// `step_vegetation` and the freeze cap of `snow::step_snow`.
    #[must_use]
    pub fn is_open_water(&self) -> bool {
        self.is_open_water_at(OPEN_WATER_EXCESS_MM)
    }

    /// [`Self::is_open_water`] with an explicit threshold (mm), for the
    /// live `VegetationParams::open_water_excess`. Two ways to qualify:
    /// the water body's surplus (liquid + ice) above capacity, **or** an
    /// ice slab thicker than the threshold on its own. Ice only forms from
    /// free surplus, so a slab proves a water body froze here; without the
    /// second clause a frozen pond whose soil water seeped into the water
    /// table during the winter (`water_level` → 0 under the ice, the
    /// retention bucket reads empty) flipped to land under its own ice and
    /// got colonized (115 of 43 561 cells, r120 seed 42, January).
    #[must_use]
    pub fn is_open_water_at(&self, excess_mm: f32) -> bool {
        self.water_body_surplus() > excess_mm || self.ice_level > excess_mm
    }

    /// Debits `amount` (mm, ≥ 0) from the frozen surface, snowpack first
    /// (it lies on top), then lake ice. Returns what ACTUALLY left the two
    /// stocks: each debit is measured after the f32 subtraction, so the
    /// caller credits exactly the departed mass (the ULP guard of
    /// `snow::step_snow` and `atmosphere::step_evaporation`, kept in one
    /// place). Neither stock goes negative.
    pub fn take_frozen(&mut self, amount: f32) -> f32 {
        let from_snow = amount.min(self.snow_level);
        let new_snow = self.snow_level - from_snow;
        let departed_snow = self.snow_level - new_snow;
        self.snow_level = new_snow;

        let from_ice = (amount - from_snow).min(self.ice_level);
        let new_ice = self.ice_level - from_ice;
        let departed_ice = self.ice_level - new_ice;
        self.ice_level = new_ice;

        departed_snow + departed_ice
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn take_frozen_debits_snow_before_ice_and_returns_the_departed_mass() {
        let mut c = CellProperties {
            snow_level: 2.0,
            ice_level: 5.0,
            ..CellProperties::default()
        };
        let departed = c.take_frozen(3.0);
        assert!((departed - 3.0).abs() < 1e-6, "departed {departed}");
        assert!(c.snow_level.abs() < 1e-6, "snow first: {}", c.snow_level);
        assert!(
            (c.ice_level - 4.0).abs() < 1e-6,
            "then ice: {}",
            c.ice_level
        );

        // More than the whole frozen surface: everything leaves, no negative.
        let departed = c.take_frozen(100.0);
        assert!((departed - 4.0).abs() < 1e-6, "departed {departed}");
        assert!(c.snow_level >= 0.0 && c.ice_level >= 0.0);
        assert!(c.frozen_surface().abs() < 1e-6);
    }

    #[test]
    fn ice_slab_on_drained_soil_is_still_open_water() {
        // A frozen pond whose soil water seeped away: bucket empty, ice on top.
        let drained = CellProperties {
            water_capacity: 95.0,
            water_level: 0.0,
            ice_level: 38.0,
            ..CellProperties::default()
        };
        assert!(drained.water_body_surplus() < 0.0, "the bucket reads empty");
        assert!(drained.is_open_water(), "yet the ice slab is a water body");
        // A frost film under the threshold does not make land a lake.
        let film = CellProperties {
            water_capacity: 95.0,
            water_level: 40.0,
            ice_level: 2.0,
            ..CellProperties::default()
        };
        assert!(!film.is_open_water());
    }

    #[test]
    fn water_body_surplus_counts_liquid_and_ice() {
        let c = CellProperties {
            water_capacity: 1.0,
            water_level: 1.0,
            ice_level: 50.0,
            ..CellProperties::default()
        };
        assert!((c.water_body_surplus() - 50.0).abs() < 1e-6);
        // A dry cell under a snowpack is not a water body.
        let dry = CellProperties {
            water_capacity: 1.0,
            water_level: 0.2,
            snow_level: 800.0,
            ..CellProperties::default()
        };
        assert!(dry.water_body_surplus() < 0.0);
    }

    /// The `vegetation` row round-trips through both encodings the crate
    /// uses (JSON for params/tests, `MessagePack` for checkpoints), bit
    /// for bit: the custom deserializer changes nothing for a full row.
    #[test]
    fn vegetation_row_round_trips_through_json_and_msgpack() {
        let mut c = CellProperties::default();
        for (i, v) in c.vegetation.iter_mut().enumerate() {
            *v = f32::from(u8::try_from(i).expect("small")) / 17.0;
        }
        let json: CellProperties =
            serde_json::from_str(&serde_json::to_string(&c).expect("json encode"))
                .expect("json decode");
        let named: CellProperties =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&c).expect("msgpack encode"))
                .expect("msgpack decode");
        let positional: CellProperties =
            rmp_serde::from_slice(&rmp_serde::to_vec(&c).expect("msgpack encode"))
                .expect("msgpack decode");
        for back in [json, named, positional] {
            assert_eq!(
                back.vegetation.map(f32::to_bits),
                c.vegetation.map(f32::to_bits)
            );
        }
    }

    /// A shorter row (an engine with fewer species, #161) is zero-padded,
    /// its columns kept raw in place: the remap by id is the checkpoint's
    /// job, not the cell's.
    #[test]
    fn a_shorter_vegetation_row_is_zero_padded() {
        let mut json = serde_json::to_value(CellProperties::default()).expect("json");
        json["vegetation"] = serde_json::json!([0.5, 0.25, 0.125, 0.0625, 0.75]);
        let c: CellProperties = serde_json::from_value(json).expect("a 5-column row loads");
        assert_eq!(c.vegetation[..5], [0.5, 0.25, 0.125, 0.0625, 0.75]);
        assert!(c.vegetation[5..].iter().all(|v| v.abs() < f32::EPSILON));
    }

    /// A longer row (an engine with species this one doesn't know) is
    /// refused, never truncated: its extra biomass has no column here.
    #[test]
    fn a_longer_vegetation_row_is_refused() {
        let mut json = serde_json::to_value(CellProperties::default()).expect("json");
        json["vegetation"] = serde_json::json!(vec![0.1_f32; SPECIES_COUNT + 1]);
        let err = serde_json::from_value::<CellProperties>(json)
            .expect_err("a row longer than SPECIES_COUNT must be refused")
            .to_string();
        assert!(
            err.contains(&format!("invalid length {}", SPECIES_COUNT + 1))
                && err.contains(&format!("at most {SPECIES_COUNT}")),
            "{err}"
        );
    }

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
            let surplus = (Mm(wl) - Mm(wc)).non_negative();
            let expected = (Meters(elev) + surplus.to_meters()).0;
            prop_assert!((cell.effective_elevation() - expected).abs() < 1e-3);
        }
    }
}
