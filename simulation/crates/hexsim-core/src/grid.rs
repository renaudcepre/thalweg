use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::cell::CellProperties;
use crate::coord::{DIRECTIONS, HexCoord, torus_lattice_vectors};

/// Hexagonal grid: `Vec` indexed in parallel with a `HashMap<HexCoord, usize>`
/// for lookup. Iteration order is deterministic (insertion order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HexGrid {
    cells: Vec<CellProperties>,
    coords: Vec<HexCoord>,
    coord_index: HashMap<HexCoord, usize>,
    /// Hexagonal radius of the domain: max distance to the center among
    /// inserted coordinates (exactly R for a grid built via `from_radius(R)`).
    /// Defines the torus translation lattice (`torus_lattice_vectors`); the
    /// tiling is only exact for a full hexagonal domain. On an arbitrarily
    /// shaped grid, `wrap_target` can return `None` and flows fall back to a
    /// conservative self-transfer.
    /// `serde(default)`: JSON without this field yields 0 → wrap degrades to
    /// a no-op, never wrong (no consumer deserializes a grid today; grids
    /// are always rebuilt via `from_radius`).
    #[serde(default)]
    radius: i32,
    /// Precomputed neighbor table (parallel to `cells`): for each cell, the
    /// index of its 6 neighbors, or `-1` at the border. Topology is frozen
    /// after construction → avoids 6 `HashMap` lookups (`SipHash` + random
    /// access) per cell per phenomenon on every tick. Rebuilt by
    /// `build_neighbor_cache`, cleared by `insert` of a new coordinate.
    /// Outside (de)serialization: rebuilt on demand otherwise.
    #[serde(skip)]
    neighbor_cache: Vec<[i32; 6]>,
    /// Precomputed toric neighbor table (parallel to `cells`): same indices
    /// as `neighbor_cache` on the interior, wrapped by translation at the
    /// border. Same lifecycle as `neighbor_cache`. Avoids recomputing the
    /// wrap in the 6+ transport passes per tick (hot path).
    #[serde(skip)]
    neighbor_cache_toric: Vec<[usize; 6]>,
}

impl HexGrid {
    /// Creates an empty grid.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            coords: Vec::new(),
            coord_index: HashMap::new(),
            radius: 0,
            neighbor_cache: Vec::new(),
            neighbor_cache_toric: Vec::new(),
        }
    }

    /// Generates a hexagonal grid of the given radius.
    /// Radius 0 = 1 cell (center), radius 1 = 7 cells, etc.
    /// Formula: 3r² + 3r + 1 cells.
    #[must_use]
    pub fn from_radius(radius: i32) -> Self {
        let r_usize = usize::try_from(radius).unwrap_or(0);
        let cap = 3 * r_usize * r_usize + 3 * r_usize + 1;
        let mut grid = Self {
            cells: Vec::with_capacity(cap),
            coords: Vec::with_capacity(cap),
            coord_index: HashMap::with_capacity(cap),
            radius: 0,
            neighbor_cache: Vec::new(),
            neighbor_cache_toric: Vec::new(),
        };
        for q in -radius..=radius {
            let r_min = (-radius).max(-q - radius);
            let r_max = radius.min(-q + radius);
            for r in r_min..=r_max {
                grid.insert(HexCoord::new(q, r), CellProperties::default());
            }
        }
        grid.build_neighbor_cache();
        grid
    }

    /// (Re)builds the precomputed neighbor tables (flat and toric).
    /// Call once the topology is complete (done by `from_radius`).
    /// Idempotent.
    pub fn build_neighbor_cache(&mut self) {
        let mut cache = vec![[-1_i32; 6]; self.cells.len()];
        for (idx, row) in cache.iter_mut().enumerate() {
            let coord = self.coords[idx];
            for (i, dir) in DIRECTIONS.iter().enumerate() {
                if let Some(&ni) = self.coord_index.get(&(coord + *dir)) {
                    row[i] = i32::try_from(ni).unwrap_or(-1);
                }
            }
        }
        self.neighbor_cache = cache;
        self.neighbor_cache_toric = (0..self.cells.len()).map(|i| self.toric_row(i)).collect();
    }

    #[must_use]
    pub fn get(&self, coord: HexCoord) -> Option<&CellProperties> {
        self.coord_index.get(&coord).map(|&i| &self.cells[i])
    }

    /// Internal index of a coordinate (parallel to `cells_slice`/`coords_slice`).
    #[must_use]
    pub fn index_of(&self, coord: HexCoord) -> Option<usize> {
        self.coord_index.get(&coord).copied()
    }

    pub fn get_mut(&mut self, coord: HexCoord) -> Option<&mut CellProperties> {
        let idx = *self.coord_index.get(&coord)?;
        Some(&mut self.cells[idx])
    }

    /// Inserts a new cell or replaces the existing one. Preserves insertion
    /// order for new coordinates.
    pub fn insert(&mut self, coord: HexCoord, cell: CellProperties) {
        if let Some(&idx) = self.coord_index.get(&coord) {
            self.cells[idx] = cell;
        } else {
            let idx = self.cells.len();
            self.cells.push(cell);
            self.coords.push(coord);
            self.coord_index.insert(coord, idx);
            self.radius = self.radius.max(coord.distance(HexCoord::new(0, 0)));
            // New topology → stale caches (rebuilt by build_neighbor_cache).
            self.neighbor_cache.clear();
            self.neighbor_cache_toric.clear();
        }
    }

    #[must_use]
    pub fn contains(&self, coord: HexCoord) -> bool {
        self.coord_index.contains_key(&coord)
    }

    /// Returns the existing neighbors of a coordinate (handles terrarium edges).
    #[must_use]
    pub fn neighbors(&self, coord: HexCoord) -> Vec<(HexCoord, &CellProperties)> {
        coord
            .neighbors()
            .into_iter()
            .filter_map(|n| self.coord_index.get(&n).map(|&i| (n, &self.cells[i])))
            .collect()
    }

    /// Iterates over all grid coordinates (insertion order).
    pub fn coords(&self) -> impl Iterator<Item = &HexCoord> {
        self.coords.iter()
    }

    /// Iterates over all (coord, cell) pairs (insertion order).
    pub fn iter(&self) -> impl Iterator<Item = (&HexCoord, &CellProperties)> {
        self.coords.iter().zip(self.cells.iter())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Toric wrap: for a coordinate outside the grid, translate by the
    /// tiling lattice (`torus_lattice_vectors`) until it falls back inside
    /// the domain. Pure translation → orientation is preserved (a true
    /// torus): a flow exiting east re-enters from the west while still
    /// heading east. The old mirror wrap `(-q, -r)` glued the edges back
    /// together by flipping them (RP² projective-plane topology, not a
    /// torus): a non-periodic seam, hot edge ↔ cold edge → permanent
    /// condensation on the seam (edge hotspots, journal 2026-07-03).
    ///
    /// For a full hexagonal domain of radius R, the tiling is exact: every
    /// coordinate one step past the edge falls back onto exactly one cell.
    /// `None` only if the grid isn't a full hexagon (hand-built test
    /// grids), in which case the caller falls back to a conservative
    /// self-transfer.
    #[must_use]
    pub fn wrap_target(&self, outside_coord: HexCoord) -> Option<HexCoord> {
        for v in torus_lattice_vectors(self.radius) {
            let back = outside_coord - v;
            if self.contains(back) {
                return Some(back);
            }
            let forward = outside_coord + v;
            if self.contains(forward) {
                return Some(forward);
            }
        }
        None
    }

    /// Hexagonal radius of the domain (max distance to center inserted).
    #[must_use]
    pub fn radius(&self) -> i32 {
        self.radius
    }

    // --- Indexed API (pub(crate), for simulation hot paths) ---

    /// Dense index of a coordinate (None if outside the grid).
    #[must_use]
    pub fn cell_index(&self, coord: HexCoord) -> Option<usize> {
        self.coord_index.get(&coord).copied()
    }

    /// Ordered slice of coordinates: avoids repeated
    /// `coords().copied().collect()` on every tick.
    #[must_use]
    pub fn coords_slice(&self) -> &[HexCoord] {
        &self.coords
    }

    /// Indices of the 6 neighbors of cell `idx`. `None` for neighbors outside
    /// the grid. Zero allocation, zero hash: reads the precomputed table
    /// (`build_neighbor_cache`). Falls back to `HashMap` if the cache is
    /// missing (deserialized grid, or still under construction).
    #[must_use]
    pub fn neighbor_indices(&self, idx: usize) -> [Option<usize>; 6] {
        if self.neighbor_cache.len() == self.cells.len() {
            let row = &self.neighbor_cache[idx];
            let mut out = [None; 6];
            for (o, &ni) in out.iter_mut().zip(row.iter()) {
                *o = usize::try_from(ni).ok();
            }
            return out;
        }
        let coord = self.coords[idx];
        let mut out = [None; 6];
        for (i, dir) in DIRECTIONS.iter().enumerate() {
            out[i] = self.coord_index.get(&(coord + *dir)).copied();
        }
        out
    }

    /// Full slice of `CellProperties`, indexed by `cell_index`. Hot path for
    /// phenomena, avoids a `HashMap` lookup per cell.
    #[must_use]
    pub fn cells_slice(&self) -> &[CellProperties] {
        &self.cells
    }

    /// Mutable variant of `cells_slice`. Allows writing into `next` by index
    /// without going back through `next.get_mut(coord)`.
    #[must_use]
    pub fn cells_slice_mut(&mut self) -> &mut [CellProperties] {
        &mut self.cells
    }

    /// Indices of the 6 toric neighbors of `idx`. For each direction outside
    /// the grid, returns the cell at the opposite edge via `wrap_target`
    /// (translation by the tiling lattice → bijective: each cell is the
    /// toric neighbor in direction `d` of exactly one cell, and the round
    /// trip `d` then `opposite(d)` returns to the start). Falls back to
    /// `idx` itself if the wrap is unreachable (non-hexagonal grid):
    /// self-transfer = no-op, hence conservative.
    ///
    /// This is the neighborhood used by ALL phenomena of the tick (mass
    /// transport, thermal/relief gradients, hydro, groundwater, fire): since
    /// the terrain is periodic, the seam is a region like any other, and a
    /// neighborhood truncated at the border used to create a systematic
    /// ring bias (edge rain hotspots, journal 2026-07-03). `neighbor_indices`
    /// (non-toric) remains legitimate only for tools that reason about the
    /// domain's geometry (structural diagnostics, exports).
    #[must_use]
    pub fn neighbor_indices_toric(&self, idx: usize) -> [usize; 6] {
        if self.neighbor_cache_toric.len() == self.cells.len() {
            return self.neighbor_cache_toric[idx];
        }
        self.toric_row(idx)
    }

    /// Computes the toric row for `idx` (slow path, no cache): neighbors
    /// within the grid, otherwise wrap by translation, otherwise self-transfer.
    fn toric_row(&self, idx: usize) -> [usize; 6] {
        let coord = self.coords[idx];
        let mut out = [idx; 6];
        for (i, dir) in DIRECTIONS.iter().enumerate() {
            let target = coord + *dir;
            if let Some(&ni) = self.coord_index.get(&target) {
                out[i] = ni;
            } else if let Some(wrap) = self.wrap_target(target)
                && let Some(&ni) = self.coord_index.get(&wrap)
            {
                out[i] = ni;
            }
        }
        out
    }
}

impl Default for HexGrid {
    fn default() -> Self {
        Self::new()
    }
}

// --- JSON export format ---

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
    /// Display illumination ∈ [0,1] (#102): fraction of sunlight received vs
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
            .coords
            .iter()
            .zip(self.cells.iter())
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

        let total_surface_water: f32 = self.cells.iter().map(|c| c.water_level).sum();
        let total_humidity: f32 = self
            .cells
            .iter()
            .map(super::cell::CellProperties::humidity_total)
            .sum();
        let total_cloud_water: f32 = self.cells.iter().map(|c| c.cloud_water).sum();
        let total_precip_this_tick: f32 = precipitation.iter().map(|p| p.rain + p.snow).sum();
        let total_groundwater: f32 = self.cells.iter().map(|c| c.groundwater).sum();
        let total_snow: f32 = self.cells.iter().map(|c| c.snow_level).sum();

        GridState {
            tick,
            hour_tick,
            cell_count: self.cells.len(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn expected_cell_count(radius: usize) -> usize {
        3 * radius * radius + 3 * radius + 1
    }

    #[test]
    fn from_radius_0_has_one_cell() {
        let grid = HexGrid::from_radius(0);
        assert_eq!(grid.len(), 1);
        assert!(grid.contains(HexCoord::new(0, 0)));
    }

    #[test]
    fn from_radius_1_has_seven_cells() {
        let grid = HexGrid::from_radius(1);
        assert_eq!(grid.len(), 7);
    }

    #[test]
    fn from_radius_formula() {
        for r in 0..=10_u32 {
            let grid = HexGrid::from_radius(i32::try_from(r).unwrap());
            assert_eq!(grid.len(), expected_cell_count(r as usize), "radius = {r}");
        }
    }

    #[test]
    fn center_has_six_neighbors() {
        let grid = HexGrid::from_radius(2);
        let neighbors = grid.neighbors(HexCoord::new(0, 0));
        assert_eq!(neighbors.len(), 6);
    }

    #[test]
    fn edge_has_fewer_neighbors() {
        let grid = HexGrid::from_radius(1);
        // (1, 0) is on the edge, has 3 neighbors in the grid, not 6
        let neighbors = grid.neighbors(HexCoord::new(1, 0));
        assert!(neighbors.len() < 6);
    }

    #[test]
    fn neighbors_outside_grid_returns_empty() {
        let grid = HexGrid::from_radius(1);
        let neighbors = grid.neighbors(HexCoord::new(100, 100));
        assert!(neighbors.is_empty());
    }

    // --- Toric topology (wrap by translation) ---
    //
    // Load-bearing piece of the torus (journal 2026-07-03): if the toric
    // neighborhood isn't a bijection per direction, transport flows
    // concentrate or dilute mass on the seam, which was the mechanism
    // behind the edge rain hotspots under the mirror wrap.

    fn opposite(dir: usize) -> usize {
        (dir + 3) % 6
    }

    /// Each direction is a permutation of the cells: everyone has exactly
    /// one upstream and one downstream toric neighbor per direction.
    fn assert_toric_bijective(grid: &HexGrid) {
        for dir in 0..6 {
            let mut seen = vec![false; grid.len()];
            for i in 0..grid.len() {
                let j = grid.neighbor_indices_toric(i)[dir];
                assert!(
                    !seen[j],
                    "dir {dir}: cell {j} is the toric neighbor of two cells"
                );
                seen[j] = true;
            }
            assert!(
                seen.iter().all(|&s| s),
                "dir {dir}: the toric neighborhood does not cover all cells"
            );
        }
    }

    /// The round trip `d` then `opposite(d)` returns to the starting cell:
    /// translation preserves orientation (a true torus; the mirror violated
    /// this symmetry on the seam).
    fn assert_toric_symmetric(grid: &HexGrid) {
        for i in 0..grid.len() {
            let row = grid.neighbor_indices_toric(i);
            for (dir, &j) in row.iter().enumerate() {
                assert_eq!(
                    grid.neighbor_indices_toric(j)[opposite(dir)],
                    i,
                    "round trip broken: cell {i}, dir {dir}"
                );
            }
        }
    }

    #[test]
    fn toric_neighbors_bijective_and_symmetric() {
        for radius in [1, 2, 3, 5, 8, 30] {
            let grid = HexGrid::from_radius(radius);
            assert_toric_bijective(&grid);
            assert_toric_symmetric(&grid);
        }
    }

    #[test]
    fn toric_neighbors_never_self_above_radius_zero() {
        // R ≥ 1: the lattice vectors have length 2R+1 ≥ 3 while a neighbor
        // step is worth 1 → a wrap can never fall back onto the source
        // cell (self-transfer only serves degenerate non-hexagonal
        // grids).
        for radius in [1, 2, 5, 30] {
            let grid = HexGrid::from_radius(radius);
            for i in 0..grid.len() {
                for &j in &grid.neighbor_indices_toric(i) {
                    assert_ne!(j, i, "unexpected self-transfer at R={radius}, cell {i}");
                }
            }
        }
    }

    #[test]
    fn toric_wrap_of_single_cell_is_itself() {
        // R = 0: single-cell degenerate torus, all 6 directions loop back
        // onto themselves (v1 = (1,0) brings every neighbor back to the origin).
        let grid = HexGrid::from_radius(0);
        assert_eq!(grid.neighbor_indices_toric(0), [0; 6]);
    }

    #[test]
    fn wrap_is_a_translation_not_a_mirror() {
        // Journal trace 2026-07-03, radius 30: the east neighbor of (30, -1)
        // is outside the grid at (31, -1). Translation by -v1 = (-61, 30) →
        // (-30, 29): re-entry through the west edge, orientation preserved.
        // The antipodal mirror gave (-31, 1) ≈ (-30, 1): edge flipped.
        let grid = HexGrid::from_radius(30);
        assert_eq!(
            grid.wrap_target(HexCoord::new(31, -1)),
            Some(HexCoord::new(-30, 29))
        );
        // And the round trip: the west neighbor of (-30, 29) outside the
        // grid at (-31, 29) must translate back to (30, -1).
        assert_eq!(
            grid.wrap_target(HexCoord::new(-31, 29)),
            Some(HexCoord::new(30, -1))
        );
    }

    #[test]
    fn toric_cache_matches_slow_path() {
        // The precomputed cache (from_radius) and the slow path (grid
        // whose cache was invalidated) must give exactly the same
        // neighborhood.
        let cached = HexGrid::from_radius(6);
        // Same topology rebuilt via inserts without build_neighbor_cache:
        // neighbor_indices_toric goes through the slow path there.
        let mut rebuilt = HexGrid::new();
        for (coord, cell) in cached.iter() {
            rebuilt.insert(*coord, cell.clone());
        }
        for i in 0..cached.len() {
            assert_eq!(
                cached.neighbor_indices_toric(i),
                rebuilt.neighbor_indices_toric(i),
                "cache ≠ slow path for cell {i}"
            );
        }
    }

    proptest! {
        #[test]
        fn prop_from_radius_cell_count(radius in 0..=50i32) {
            let grid = HexGrid::from_radius(radius);
            prop_assert_eq!(grid.len(), expected_cell_count(usize::try_from(radius).unwrap()));
        }

        /// Fundamental torus invariant: toric neighborhood bijective per
        /// direction and symmetric under round trip, for any radius.
        #[test]
        fn prop_toric_neighbors_bijective_and_symmetric(radius in 1..=20i32) {
            let grid = HexGrid::from_radius(radius);
            for dir in 0..6 {
                let mut seen = vec![false; grid.len()];
                for i in 0..grid.len() {
                    let j = grid.neighbor_indices_toric(i)[dir];
                    prop_assert!(!seen[j]);
                    seen[j] = true;
                    prop_assert_eq!(grid.neighbor_indices_toric(j)[(dir + 3) % 6], i);
                }
            }
        }
    }
}
