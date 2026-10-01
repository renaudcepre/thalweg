//! JSON export format: [`crate::snapshot::CellSnapshot`]/[`crate::snapshot::GridState`] and the `HexGrid`
//! method that builds them from the live simulation state.
//!
//! Split out of `grid.rs` (was its "JSON export format" section): this DTO
//! layer reads from every phenomenon (`hydro`, `species`, `vegetation`,
//! `wind`) to flatten a tick into something serializable, which used to
//! make `grid.rs` import "upward" from phenomena built on top of it
//! (`hydro.rs` in turn imports `HexGrid`, a real dependency cycle). Moving
//! the DTO here restores the correct stratification: `grid` is the base
//! data structure and no longer depends on any phenomenon, `snapshot`
//! depends on `grid` + the phenomena it flattens, which is the direction
//! the dependency should point.

use serde::{Deserialize, Serialize};

use crate::grid::HexGrid;
use crate::hydro::HydroMaps;
use crate::lithology::LithologyId;
use crate::species::{
    GrowthForm, SPECIES, SPECIES_COUNT, STRATA, STRATUM_COUNT, Species, SpeciesId, Stratum,
};
use crate::vegetation::{canopy_cover, dominant_species, is_open_water, stratum_cover};
use crate::wind::WindField;

/// A flattened cell for JSON serialization: coord + properties + flux.
#[derive(Debug, Serialize, Deserialize)]
pub struct CellSnapshot {
    pub q: i32,
    pub r: i32,
    pub elevation: f32,
    pub temperature: f32,
    pub water_level: f32,
    pub water_capacity: f32,
    /// Low-layer vapor (not directly precipitable).
    pub humidity_surface: f32,
    /// High-altitude vapor (invisible). Reservoir advected by upper winds.
    pub humidity_upper: f32,
    /// Condensed droplets (visible clouds). This is what the renderer paints
    /// as cloud, distinct from the `humidity_upper` vapor.
    pub cloud_water: f32,
    pub groundwater: f32,
    pub snow_level: f32,
    /// Lake / river ice (mm w.e.): the frozen surplus of a water body,
    /// apart from the snowpack. Rendered as part of the water body.
    pub ice_level: f32,
    /// Hydric aptitude ∈ [0, 1] derived from `lithology` and relief: the
    /// water table holds `permeability × 100 mm`, infiltration scales with
    /// it. Dimensionless by construction, shown as a percent by the front.
    pub permeability: f32,
    /// Rock class of the substrate (`snake_case`: granite, marl, sandstone,
    /// limestone), the source of `permeability` (#136). Exported so the
    /// inspector can name the soil instead of showing a bare 0-1 number.
    pub lithology: LithologyId,
    /// Canopy cover [0, 1]: share of the ground under at least one
    /// vegetation layer, `1 − Π(1 − cover_S)` over the strata
    /// (`vegetation::canopy_cover`, #161). No longer the plain sum of the
    /// biomasses, which exceeds 1 as soon as an understory lives under a
    /// canopy.
    pub vegetation: f32,
    /// Cover of each stratum [0, 1], order = `GridState::stratum_order`
    /// (herb, shrub, tree): the sum of `species_mix` over the species of
    /// that stratum (`vegetation::stratum_cover`). Each stratum has its
    /// own space, so the three don't add up to `vegetation`. Lets the
    /// front size a tree scatter by the tree layer without knowing which
    /// columns are trees (anti-pattern #2).
    pub cover_by_stratum: [f32; STRATUM_COUNT],
    /// Dominant species **as seen from the sky** (`vegetation::dominant_species`:
    /// a grass under a closed canopy shows as the canopy), or `null` if
    /// bare ground. Derived by the core, consumed as-is by the front
    /// (anti-pattern #2).
    pub dominant_species: Option<SpeciesId>,
    /// Biomass per species [0, 1], in the order of `species::SPECIES`
    /// (`GridState::species_order`). Lets callers judge the **mix** of
    /// species in a hex (mono vs mixed) without recomputing on the
    /// consumer side. Summed over one stratum's species = that entry of
    /// `cover_by_stratum`.
    pub species_mix: [f32; SPECIES_COUNT],
    /// Average canopy age (years), proxy for "old-growth forest" (#wildfire).
    pub stand_age: f32,
    /// Current fire intensity [0, 1]; 0 = no fire.
    pub fire_intensity: f32,
    /// `true` if open water (lake): the front renders it blue, not as cover.
    pub is_open_water: bool,
    /// `true` if it is raining or snowing this hour-tick (same map as
    /// `rain_amount`/`snow_amount`, not the daily accumulator).
    pub is_raining: bool,
    /// Liquid precipitation fallen this hour-tick (mm, rain/h).
    pub rain_amount: f32,
    /// Solid precipitation fallen this hour-tick (mm w.e., snow/h).
    pub snow_amount: f32,
    /// Outflow flux, sourced from `HydroMaps::discharge`: the 60-day EMA
    /// (#106) in production via `Simulation::snapshot`, not the
    /// instantaneous daily slice, so the displayed network drifts with the
    /// seasons instead of rearranging itself with every rain.
    pub outflow_flux: f32,
    /// World-space vector of average outflow flux (for river trail
    /// rendering). Instantaneous (daily slice), no EMA exists for this field.
    pub flow_vec_x: f32,
    pub flow_vec_y: f32,
    /// Flux per edge (order `coord::DIRECTIONS`), sourced from
    /// `HydroMaps::edge_flux` (same EMA as `outflow_flux` in production,
    /// #106), quantized to u8 on a square-root scale relative to the frame
    /// max (#103):
    /// `b = round(255·√(flux/edge_flux_max))` ⇔ `flux = (b/255)²·edge_flux_max`.
    /// The square root allocates resolution to small flows (a trickle at
    /// 0.1% of the max stays distinguishable from zero); `b/255` is directly
    /// a relative visual intensity. 0 = nothing flows through this edge over
    /// this window.
    pub edge_flux: [u8; 6],
    pub wind_x: f32,
    pub wind_y: f32,
    /// Synoptic geopotential height `h` (m), a pressure proxy, the front's
    /// isobars. Filled by `Simulation::snapshot` (synoptic state lives in
    /// the sim, not the grid); 0 via `grid.snapshot` alone.
    pub synoptic_h: f32,
    /// Total synoptic wind (m/s SI, includes mean zonal flow), the basis of
    /// the wind consumed when `synoptic.enabled`. Filled by `Simulation::snapshot`.
    pub synoptic_u: f32,
    pub synoptic_v: f32,
    /// Display illumination ∈ `[0,1]` (#102): fraction of sunlight received vs
    /// a flat, clear, cloudless cell (aspect × occlusion × cloud shadow).
    /// Filled by `Simulation::snapshot` (like the synoptic fields); the
    /// front multiplies albedo by this value. 1.0 via `grid.snapshot` alone.
    pub illumination: f32,
}

/// What a consumer needs to know about a species to draw it: its layer
/// and its silhouette. One entry per column of `CellSnapshot::species_mix`
/// (`GridState::species_catalog`), built from `species::SPECIES`, so the
/// front never hardcodes which id is a conifer or a grass: a species added
/// to the table draws right on the day it is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesInfo {
    pub id: SpeciesId,
    /// Vertical layer, the index of `CellSnapshot::cover_by_stratum` it
    /// counts in (through `GridState::stratum_order`).
    pub stratum: Stratum,
    /// Silhouette (grass = ground tint, shrub, broadleaf, conifer).
    pub growth_form: GrowthForm,
}

impl From<&Species> for SpeciesInfo {
    fn from(s: &Species) -> Self {
        Self {
            id: s.id,
            stratum: s.stratum,
            growth_form: s.growth_form,
        }
    }
}

/// Complete grid state, ready to serialize to JSON.
#[derive(Debug, Serialize, Deserialize)]
pub struct GridState {
    /// Tick in simulated days (v0.2.x compat, front-end consumers use it
    /// for `tickToDate`, season label, etc.).
    pub tick: u64,
    /// Tick in simulated hours (issue #47 / #42 v0.3.0 project). Lets the
    /// front compute the instantaneous solar cycle: `tick` (in days) stays
    /// constant for 24 consecutive ticks, which would give a day/night
    /// cycle lasting 24 simulated days. Source: `Simulation::hour_tick()`.
    pub hour_tick: u64,
    pub cell_count: usize,
    pub total_surface_water: f32,
    pub total_humidity: f32,
    /// Stock of condensed droplets (visible clouds). Subset of
    /// `total_humidity`, exported separately for the UI.
    pub total_cloud_water: f32,
    /// Rain + snow fallen during this hour-tick only (flux, not stock),
    /// summed over the grid. The front derives an mm/day-ish rate from it
    /// by multiplying the per-cell mean by 24, it is not itself a daily
    /// total (see `CellSnapshot::rain_amount`/`snow_amount`).
    pub total_precip_this_tick: f32,
    pub total_groundwater: f32,
    /// Deep aquifer over the grid (mm, #107), apart from the root zone's
    /// `total_groundwater`. Header only, like `total_sky_water`.
    pub total_aquifer: f32,
    pub total_snow: f32,
    /// Lake / river ice over the grid (mm w.e.), apart from `total_snow`.
    pub total_ice: f32,
    /// Water held outside the box by the imposed weather regime (#63),
    /// mm, map total. Header only, no per-cell field: it is one scalar
    /// for the whole map, and the wire already costs 140 B/cell. Filled
    /// by `Simulation::snapshot` (the reservoir lives in the simulation);
    /// 0 through `HexGrid::snapshot` alone, and 0 whenever the regime is
    /// off. ON is the shipped default since 2026-09-06 (#63/#146), so a
    /// consumer that sums the totals to check the terrarium must include
    /// it.
    pub total_sky_water: f32,
    /// `true` while the imposed weather regime is in its wet phase.
    /// Same provenance as `total_sky_water`; always `false` when the
    /// regime is off.
    pub weather_regime_wet: bool,
    /// Species order matching the indices of `CellSnapshot::species_mix`
    /// (= order of `species::SPECIES`). Makes the mix self-describing on
    /// the consumer side: `species_mix[i]` ↔ `species_order[i]`.
    pub species_order: [SpeciesId; SPECIES_COUNT],
    /// Layer and silhouette of each species, same order as
    /// `species_order` (`species_catalog[i]` describes `species_mix[i]`).
    /// Header only: static for the engine's lifetime, paid once per frame.
    pub species_catalog: [SpeciesInfo; SPECIES_COUNT],
    /// Strata order matching the indices of `CellSnapshot::cover_by_stratum`
    /// (= `species::STRATA`, bottom to top): the stratum cover is
    /// self-describing the same way `species_order` makes the mix.
    pub stratum_order: [Stratum; STRATUM_COUNT],
    /// Quantization scale for `CellSnapshot::edge_flux`: largest edge flux
    /// (mm) observed this frame. 0 if nothing flows anywhere.
    pub edge_flux_max: f32,
    pub cells: Vec<CellSnapshot>,
}

/// Quantizes a cell's 6 edge flows to u8 on a square-root scale (see the
/// docs for `CellSnapshot::edge_flux`). `flux ≤ max` by construction (`max`
/// is the frame's global max) so `255·√(flux/max) ∈ [0, 255]`: the cast is
/// bounded, isolated, and documented here (same precedent as
/// `synoptic_mesh::round_coord`).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantize_edge_flux(dirs: Option<&[f32; 6]>, max: f32) -> [u8; 6] {
    let Some(dirs) = dirs else { return [0; 6] };
    if max <= 0.0 {
        return [0; 6];
    }
    dirs.map(|flux| {
        if flux <= 0.0 {
            0
        } else {
            (255.0 * (flux / max).min(1.0).sqrt()).round() as u8
        }
    })
}

impl HexGrid {
    #[must_use]
    pub fn snapshot(
        &self,
        tick: u64,
        hour_tick: u64,
        hydro: &HydroMaps<'_>,
        wind_field: &WindField,
        precipitation: &crate::atmosphere::PrecipitationMap,
    ) -> GridState {
        // Global max first: it's the quantization scale for the u8
        // `edge_flux` of every cell in the frame.
        let edge_flux_max = hydro
            .edge_flux
            .iter()
            .flat_map(|dirs| dirs.iter().copied())
            .fold(0.0_f32, f32::max);
        let cells = self
            .coords_slice()
            .iter()
            .zip(self.cells_slice().iter())
            .enumerate()
            .map(|(i, (coord, props))| {
                let outflow_flux = hydro.discharge.get(i).copied().unwrap_or(0.0);
                let (flow_vec_x, flow_vec_y) = hydro.flow_vec.get(i).copied().unwrap_or((0.0, 0.0));
                let wind = wind_field.get(i).copied().unwrap_or_default();
                let precip = precipitation.get(i);
                let rain_amount = precip.map_or(0.0, |d| d.rain);
                let snow_amount = precip.map_or(0.0, |d| d.snow);
                CellSnapshot {
                    q: coord.q,
                    r: coord.r,
                    elevation: props.elevation,
                    temperature: props.temperature,
                    water_level: props.water_level,
                    water_capacity: props.water_capacity,
                    humidity_surface: props.humidity_surface,
                    humidity_upper: props.humidity_upper,
                    cloud_water: props.cloud_water,
                    groundwater: props.groundwater,
                    snow_level: props.snow_level,
                    ice_level: props.ice_level,
                    permeability: props.permeability,
                    lithology: props.lithology,
                    vegetation: canopy_cover(props),
                    cover_by_stratum: STRATA.map(|stratum| stratum_cover(props, stratum)),
                    dominant_species: dominant_species(props),
                    species_mix: props.vegetation,
                    stand_age: props.stand_age,
                    fire_intensity: props.fire_intensity,
                    is_open_water: is_open_water(props),
                    is_raining: rain_amount > 1e-4 || snow_amount > 1e-4,
                    rain_amount,
                    snow_amount,
                    outflow_flux,
                    flow_vec_x,
                    flow_vec_y,
                    edge_flux: quantize_edge_flux(hydro.edge_flux.get(i), edge_flux_max),
                    wind_x: wind.x,
                    wind_y: wind.y,
                    synoptic_h: 0.0,
                    synoptic_u: 0.0,
                    synoptic_v: 0.0,
                    illumination: 1.0,
                }
            })
            .collect();

        let total_surface_water: f32 = self.cells_slice().iter().map(|c| c.water_level).sum();
        let total_humidity: f32 = self
            .cells_slice()
            .iter()
            .map(crate::cell::CellProperties::humidity_total)
            .sum();
        let total_cloud_water: f32 = self.cells_slice().iter().map(|c| c.cloud_water).sum();
        let total_precip_this_tick: f32 = precipitation.iter().map(|p| p.rain + p.snow).sum();
        let total_groundwater: f32 = self.cells_slice().iter().map(|c| c.groundwater).sum();
        let total_aquifer: f32 = self.cells_slice().iter().map(|c| c.aquifer).sum();
        let total_snow: f32 = self.cells_slice().iter().map(|c| c.snow_level).sum();
        let total_ice: f32 = self.cells_slice().iter().map(|c| c.ice_level).sum();

        GridState {
            tick,
            hour_tick,
            cell_count: self.cells_slice().len(),
            total_surface_water,
            total_humidity,
            total_cloud_water,
            total_precip_this_tick,
            total_groundwater,
            total_aquifer,
            total_snow,
            total_ice,
            // Filled by `Simulation::snapshot`: the regime's state lives
            // in the simulation, the grid knows nothing about it (same
            // pattern as the synoptic and illumination fields above).
            total_sky_water: 0.0,
            weather_regime_wet: false,
            species_order: SPECIES.map(|s| s.id),
            species_catalog: SPECIES.each_ref().map(SpeciesInfo::from),
            stratum_order: STRATA,
            edge_flux_max,
            cells,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::species::species_index;

    fn empty_hydro_maps() -> HydroMaps<'static> {
        HydroMaps {
            discharge: &[],
            flow_vec: &[],
            edge_flux: &[],
        }
    }

    /// One cell with a meadow under a boxwood under a downy oak: each
    /// stratum's cover is the sum of its species' columns, exported in the
    /// header's `stratum_order`, and `vegetation` is the canopy cover
    /// (`1 − Π(1 − cover_S)`), not the 1.4 plain sum.
    #[test]
    fn snapshot_exports_the_cover_of_each_stratum() {
        let mut grid = HexGrid::from_radius(0);
        let cell = grid.get_mut(HexCoord::new(0, 0)).expect("center cell");
        cell.vegetation[species_index(SpeciesId::Meadow)] = 0.5;
        cell.vegetation[species_index(SpeciesId::Boxwood)] = 0.25;
        cell.vegetation[species_index(SpeciesId::OakPubescent)] = 0.5;
        cell.vegetation[species_index(SpeciesId::Beech)] = 0.25;

        let state = grid.snapshot(0, 0, &empty_hydro_maps(), &Vec::new(), &Vec::new());
        let c = &state.cells[0];
        let cover = |s: Stratum| {
            let i = state
                .stratum_order
                .iter()
                .position(|&o| o == s)
                .expect("every stratum is in the order");
            c.cover_by_stratum[i]
        };
        assert!((cover(Stratum::Herb) - 0.5).abs() < 1e-6);
        assert!((cover(Stratum::Shrub) - 0.25).abs() < 1e-6);
        assert!((cover(Stratum::Tree) - 0.75).abs() < 1e-6);
        let canopy = 1.0 - 0.5 * 0.75 * 0.25;
        assert!(
            (c.vegetation - canopy).abs() < 1e-6,
            "canopy cover {} vs {canopy}",
            c.vegetation
        );
        assert_eq!(
            c.species_mix.map(f32::to_bits),
            grid.cells_slice()[0].vegetation.map(f32::to_bits)
        );
    }

    /// The catalog is the species table's own layer and silhouette, in the
    /// `species_order` columns: the front reads the conifers from it,
    /// never from a hardcoded id list.
    #[test]
    fn species_catalog_mirrors_the_species_table() {
        let grid = HexGrid::from_radius(0);
        let state = grid.snapshot(0, 0, &empty_hydro_maps(), &Vec::new(), &Vec::new());
        for ((info, &id), s) in state
            .species_catalog
            .iter()
            .zip(&state.species_order)
            .zip(&SPECIES)
        {
            assert_eq!(info.id, id, "catalog and order share the columns");
            assert_eq!((info.stratum, info.growth_form), (s.stratum, s.growth_form));
        }
        assert_eq!(state.stratum_order, STRATA);
        let fir = state.species_catalog[species_index(SpeciesId::Fir)];
        assert_eq!(
            (fir.stratum, fir.growth_form),
            (Stratum::Tree, GrowthForm::Conifer)
        );
        let json = serde_json::to_value(fir).expect("json");
        assert_eq!(
            json,
            serde_json::json!({"id": "fir", "stratum": "tree", "growth_form": "conifer"}),
            "snake_case on the wire, the front's keys"
        );
    }
}
