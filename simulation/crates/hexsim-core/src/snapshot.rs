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
use crate::species::{SPECIES, SPECIES_COUNT, SpeciesId};
use crate::vegetation::{cell_total_vegetation, dominant_species, is_open_water};
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
    pub permeability: f32,
    /// Total vegetation cover [0, 1] (sum of per-species biomass).
    pub vegetation: f32,
    /// Dominant species (highest biomass), or `null` if bare ground. Derived
    /// by the core, consumed as-is by the front (anti-pattern #2).
    pub dominant_species: Option<SpeciesId>,
    /// Biomass per species [0, 1], in the order of `species::SPECIES`. Lets
    /// callers judge the **mix** of species in a hex (mono vs mixed) without
    /// recomputing on the consumer side. Sum = `vegetation`.
    pub species_mix: [f32; SPECIES_COUNT],
    /// Average canopy age (years), proxy for "old-growth forest" (#wildfire).
    pub stand_age: f32,
    /// Current fire intensity [0, 1]; 0 = no fire.
    pub fire_intensity: f32,
    /// `true` if open water (lake): the front renders it blue, not as cover.
    pub is_open_water: bool,
    pub is_raining: bool,
    /// Liquid precipitation fallen this tick (rain/tick).
    pub rain_amount: f32,
    /// Solid precipitation fallen this tick (snow/tick).
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
    /// Rain + snow fallen during this tick only (flux, not stock).
    /// Used to compute average rainfall in mm/day for the UI.
    pub total_precip_this_tick: f32,
    pub total_groundwater: f32,
    pub total_snow: f32,
    /// Species order matching the indices of `CellSnapshot::species_mix`
    /// (= order of `species::SPECIES`). Makes the mix self-describing on
    /// the consumer side: `species_mix[i]` ↔ `species_order[i]`.
    pub species_order: [SpeciesId; SPECIES_COUNT],
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
                    permeability: props.permeability,
                    vegetation: cell_total_vegetation(props),
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
        let total_snow: f32 = self.cells_slice().iter().map(|c| c.snow_level).sum();

        GridState {
            tick,
            hour_tick,
            cell_count: self.cells_slice().len(),
            total_surface_water,
            total_humidity,
            total_cloud_water,
            total_precip_this_tick,
            total_groundwater,
            total_snow,
            species_order: SPECIES.map(|s| s.id),
            edge_flux_max,
            cells,
        }
    }
}
