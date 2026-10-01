//! Climatological initial state deduced from the terrain (#152).
//!
//! "The noise is the big bang" already held for the static properties
//! (relief, lithology, permeability, retention capacity, the drainage
//! network carved by worldgen erosion). The **stocks** started uniform: one
//! millimetre of surface water everywhere, no lake, no snow, a flat vapour
//! floor, no vegetation at all for a blind first year while the climate
//! normals accumulated. A fresh world was a bare plateau, and the embed
//! shipped a 42-year checkpoint to get out of it.
//!
//! This module poses, once, at generation, the state a world of this relief
//! would have on the morning of January 1st under the engine's own climate.
//! Nothing here is a new stock and nothing is painted by the tick: it is an
//! initial condition, deterministic by seed, that the phenomena then take
//! over. Every rule is the engine's own physics evaluated at its
//! climatological mean, never a number read off a picture:
//!
//! - **water table**: the field-capacity endowment of #151, modulated by the
//!   topographic wetness index (Beven & Kirkby 1979): valleys above field
//!   capacity, crests below, the TOPMODEL steady state;
//! - **surface water**: a runoff sheet routed down the relief into its closed
//!   depressions and leveled there, cascading over a spillway when a basin
//!   is full (every depression of a noise relief is tens of metres deep, so
//!   "filled to the spillway" is not a budget any closed box can hold: the
//!   sheet is the budget, the relief decides where it rests);
//! - **climate normals**: the engine's linearised radiative balance solved
//!   on the real relief (aspect, occlusion, diffuse sky from the
//!   illumination cache, the mixed-air sensible exchange) over a sampled
//!   year: the annual mean under the calibration cloud cover, the sustained
//!   extremes under a clear sky, which is what a cold snap and a heat wave
//!   are. Measured on r30 × 3 seeds (2026-10-01): the residual of the
//!   engine's own normals against insolation is 0.0886 K per W/m², which is
//!   `1/(LIN + SENS)` of the balance to three digits, and the sustained
//!   extremes sit 14-17 K from the mean, where the clear-sky balance puts
//!   them;
//! - **snowpack**: the snowfall of the frost weeks before January 1st;
//! - **vegetation**: each stratum seeded at the logistic equilibrium cover
//!   its best species reaches under those normals, shared in proportion to
//!   suitability, the light coupling evaluated top-down; canopy ages drawn
//!   from the fire-return distribution so no two neighbours start in step.
//!
//! The t0 is then an **instrument**: a stock the dynamics do not sustain
//! drains anyway (#151's lesson), and the year-by-year drift of each stock
//! from this state (`tests/diag_initial_state.rs`) names the leak instead
//! of hiding it in a checkpoint.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::cell::CellProperties;
use crate::climate_normals::CellClimateNormals;
use crate::dynamics::CELL_SPACING_M;
use crate::erosion::{WORLDGEN_FLOW_CONCENTRATION, accumulate_flow};
use crate::grid::HexGrid;
use crate::groundwater::DEFAULT_MAX_CAPACITY_MM;
use crate::hashing::{coord_word, hash01};
use crate::lake::solve_flat_level;
use crate::species::{SPECIES, SPECIES_COUNT, STRATA, Stratum};
use crate::temperature::{
    ATMO_IR_BACK_CLEAR, ATMO_IR_BACK_CLOUDY_BOOST, IllumCache, LIN_RADIATIVE_COEF,
    SENSIBLE_EXCHANGE_COEF, STEFAN_BOLTZMANN_AT_T0, TemperatureParams,
    aspect_insolation_correction, calibration_offset, compute_illumination_cached,
    compute_surface_normals, solar_beam_at_tick, terrain_insolation_sample_stride_days,
};
use crate::vegetation::{VegetationParams, strata_light_transmittance, stratum_cover};

// ====================================================================
// Relief: depressions, spill levels, catchments
// ====================================================================

/// Sentinel for "no flood parent": the seed of the priority flood.
const NO_PARENT: usize = usize::MAX;

/// The closed depressions of the bedrock and how they drain into one
/// another, read once from the relief.
///
/// `spill_level[i]` is the lowest level at which water standing at `i`
/// reaches the global minimum of the map: a priority flood (Barnes,
/// Lehman & Mulla 2014, *Priority-flood: an optimal depression-filling
/// algorithm*) seeded at that minimum, on the torus, so every closed
/// depression but the one holding the global minimum reads its own
/// spillway height, and the terminal basin stays at its bedrock. `pit_of[i]`
/// is the local minimum the steepest descent from `i` ends in, so each
/// pit's catchment is the set of cells that map to it.
pub struct ReliefBasins {
    /// Priority-flood level (m) of every cell, `≥ elevation`.
    pub spill_level: Vec<f32>,
    /// The cell each one was flooded from, [`NO_PARENT`] for the seed:
    /// walking it from a pit leaves its depression through the spillway.
    flood_parent: Vec<usize>,
    /// Terminal pit of the steepest descent from every cell (itself for a
    /// pit).
    pub pit_of: Vec<usize>,
}

/// Min-heap entry of the priority flood: the level, then the index to keep
/// the pop order total (and so the flood deterministic).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Flood {
    level_bits: u32,
    idx: usize,
}

impl Flood {
    fn level(self) -> f32 {
        f32::from_bits(self.level_bits)
    }
}

impl Ord for Flood {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .level()
            .total_cmp(&self.level())
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

impl PartialOrd for Flood {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ReliefBasins {
    /// Reads the depressions of `grid`'s bedrock (`elevation`, not the
    /// free surface: this runs before any water is placed).
    #[must_use]
    pub fn of(grid: &HexGrid) -> Self {
        let cells = grid.cells_slice();
        let n = cells.len();
        let spill_level = priority_flood_levels(grid);
        let flood_parent = priority_flood_parents(grid);
        let pit_of = steepest_descent_pits(grid);
        debug_assert_eq!(spill_level.len(), n);
        Self {
            spill_level,
            flood_parent,
            pit_of,
        }
    }

    /// The pit the excess of a full depression pours into: walk the flood
    /// parents out of the depression until the level drops below the
    /// pit's spill level, then descend from there. Always strictly lower
    /// in spill level than `pit`, so a cascade processed in decreasing
    /// spill order never pours into a basin already settled.
    fn downstream_pit(&self, pit: usize) -> usize {
        let spill = self.spill_level[pit];
        let mut cursor = pit;
        loop {
            let parent = self.flood_parent[cursor];
            if parent == NO_PARENT {
                return self.pit_of[cursor];
            }
            if self.spill_level[parent] < spill {
                return self.pit_of[parent];
            }
            cursor = parent;
        }
    }

    /// Whether `pit` is the terminal basin: the one holding the global
    /// minimum, with no spillway (its level is its bedrock).
    fn is_terminal(&self, pit: usize, cells: &[CellProperties]) -> bool {
        self.spill_level[pit] <= cells[pit].elevation
    }
}

/// Priority flood from the global minimum of the bedrock: `level[j] =
/// max(elevation[j], level[i])` on first discovery from a popped `i`.
/// Pops come in increasing level, so the first offer is the lowest one and
/// a single assignment per cell is exact.
fn priority_flood(grid: &HexGrid) -> (Vec<f32>, Vec<usize>) {
    let cells = grid.cells_slice();
    let n = cells.len();
    let mut level = vec![f32::NAN; n];
    let mut parent = vec![NO_PARENT; n];
    if n == 0 {
        return (level, parent);
    }
    let seed = (0..n)
        .min_by(|&a, &b| {
            cells[a]
                .elevation
                .total_cmp(&cells[b].elevation)
                .then_with(|| a.cmp(&b))
        })
        .expect("non-empty grid");
    let mut heap = BinaryHeap::new();
    level[seed] = cells[seed].elevation;
    heap.push(Flood {
        level_bits: level[seed].to_bits(),
        idx: seed,
    });
    let mut done = vec![false; n];
    while let Some(entry) = heap.pop() {
        let i = entry.idx;
        if done[i] {
            continue;
        }
        done[i] = true;
        let here = entry.level();
        for j in grid.neighbor_indices_toric(i) {
            if done[j] || !level[j].is_nan() {
                continue;
            }
            level[j] = cells[j].elevation.max(here);
            parent[j] = i;
            heap.push(Flood {
                level_bits: level[j].to_bits(),
                idx: j,
            });
        }
    }
    (level, parent)
}

fn priority_flood_levels(grid: &HexGrid) -> Vec<f32> {
    priority_flood(grid).0
}

fn priority_flood_parents(grid: &HexGrid) -> Vec<usize> {
    priority_flood(grid).1
}

/// Steepest-descent (D8) terminal pit of every cell on the bedrock. A cell
/// with no strictly lower neighbour is a pit; the descent is strictly
/// decreasing, so it ends.
fn steepest_descent_pits(grid: &HexGrid) -> Vec<usize> {
    let cells = grid.cells_slice();
    let n = cells.len();
    let successor: Vec<usize> = (0..n)
        .map(|i| {
            let mut best = i;
            let mut best_elev = cells[i].elevation;
            for j in grid.neighbor_indices_toric(i) {
                if cells[j].elevation < best_elev {
                    best = j;
                    best_elev = cells[j].elevation;
                }
            }
            best
        })
        .collect();
    let mut pit_of = vec![NO_PARENT; n];
    let mut path = Vec::new();
    for start in 0..n {
        if pit_of[start] != NO_PARENT {
            continue;
        }
        path.clear();
        let mut cursor = start;
        while pit_of[cursor] == NO_PARENT && successor[cursor] != cursor {
            path.push(cursor);
            cursor = successor[cursor];
        }
        let pit = if pit_of[cursor] == NO_PARENT {
            cursor
        } else {
            pit_of[cursor]
        };
        pit_of[cursor] = pit;
        for &c in &path {
            pit_of[c] = pit;
        }
    }
    pit_of
}

/// Topographic wetness index `ln(a / tan β)` of every cell (Beven &
/// Kirkby 1979, *A physically based, variable contributing area model of
/// basin hydrology*): `a` the area drained through the cell, in cells,
/// routed on the bare relief with the worldgen flow concentration
/// (`erosion::accumulate_flow`); `tan β` the steepest descent to a
/// neighbour over the cell spacing, floored at one metre of drop so a
/// flat cell reads as a wet one rather than an infinite one.
#[must_use]
pub fn wetness_index(grid: &HexGrid) -> Vec<f32> {
    let (drained, _) = accumulate_flow(grid, WORLDGEN_FLOW_CONCENTRATION);
    let cells = grid.cells_slice();
    (0..cells.len())
        .map(|i| {
            let drop = grid
                .neighbor_indices_toric(i)
                .iter()
                .map(|&j| cells[i].elevation - cells[j].elevation)
                .fold(0.0_f32, f32::max)
                .max(1.0);
            (drained[i] / (drop / CELL_SPACING_M)).ln()
        })
        .collect()
}

/// `n` as an f32, exact below 2^24 cells (same construction as
/// `atmosphere::condensation::exact_cell_count`).
fn exact_count(n: usize) -> f32 {
    let n = u32::try_from(n).expect("cell count fits in u32");
    let high = u16::try_from(n >> 16).expect("high half fits u16");
    let low = u16::try_from(n & 0xFFFF).expect("low half fits u16");
    f32::from(high) * 65_536.0 + f32::from(low)
}

// ====================================================================
// Stocks: water table, lakes
// ====================================================================

/// Water table at worldgen: the field-capacity endowment of #151
/// (`mean_frac × permeability × DEFAULT_MAX_CAPACITY_MM`) plus the TOPMODEL
/// steady-state deviation, `slope_mm × (λ_i − λ̄)` with `λ` the
/// [`wetness_index`]: the local saturation deficit is the catchment's mean
/// deficit minus `m` times the index anomaly (Beven & Kirkby 1979, eq. 10),
/// so valley floors sit above field capacity, where the daily piezometric
/// flow drains them into the rivers, and crests below it. Bounded by the
/// cell's pore storage on both sides: no negative water, no water past the
/// capacity of the rock (physical bounds, not a mask).
pub fn seed_groundwater_by_wetness(grid: &mut HexGrid, mean_frac: f32, slope_mm: f32) {
    let twi = wetness_index(grid);
    let n = twi.len();
    if n == 0 {
        return;
    }
    let twi_mean = twi.iter().map(|&v| f64::from(v)).sum::<f64>() / f64::from(exact_count(n));
    for (cell, &index) in grid.cells_slice_mut().iter_mut().zip(twi.iter()) {
        let capacity = cell.permeability * DEFAULT_MAX_CAPACITY_MM;
        #[expect(clippy::cast_possible_truncation)] // a mean of f32 values, back to f32
        let anomaly = (f64::from(index) - twi_mean) as f32;
        cell.groundwater = (mean_frac * capacity + slope_mm * anomaly).clamp(0.0, capacity);
    }
}

/// Routes a uniform runoff sheet of `sheet_mm` per cell down the steepest
/// descent into the closed depressions of the bedrock and levels it there:
/// each pit receives the sheet its catchment sheds; a depression holding
/// more than its volume to the spillway fills to the spillway and pours the
/// excess into the basin its spillway drains to; the terminal basin takes
/// whatever reaches it. Flooded cells carry their retention bucket plus the
/// free depth (`water_level = water_capacity + depth`), as a lake cell does
/// in the engine; every other cell starts dry.
///
/// Returns the surface water placed (mm summed over the cells).
pub fn fill_basins_with_runoff(grid: &mut HexGrid, basins: &ReliefBasins, sheet_mm: f32) -> f32 {
    let n = grid.len();
    if n == 0 || sheet_mm <= 0.0 {
        for cell in grid.cells_slice_mut() {
            cell.water_level = 0.0;
        }
        return 0.0;
    }
    let cells = grid.cells_slice();
    let sheet_m = f64::from(sheet_mm) / 1000.0;
    let mut volume_m = vec![0.0_f64; n];
    for &pit in &basins.pit_of {
        volume_m[pit] += sheet_m;
    }
    // Catchment members, grouped by pit and sorted by elevation.
    let mut by_pit: Vec<usize> = (0..n).collect();
    by_pit.sort_by(|&a, &b| {
        basins.pit_of[a]
            .cmp(&basins.pit_of[b])
            .then_with(|| cells[a].elevation.total_cmp(&cells[b].elevation))
            .then_with(|| a.cmp(&b))
    });
    let mut span = vec![(0_usize, 0_usize); n];
    let mut start = 0;
    while start < n {
        let pit = basins.pit_of[by_pit[start]];
        let mut end = start;
        while end < n && basins.pit_of[by_pit[end]] == pit {
            end += 1;
        }
        span[pit] = (start, end);
        start = end;
    }
    // Upstream first: decreasing spill level, index as the tie-break.
    let mut pits: Vec<usize> = (0..n).filter(|&i| basins.pit_of[i] == i).collect();
    pits.sort_by(|&a, &b| {
        basins.spill_level[b]
            .total_cmp(&basins.spill_level[a])
            .then_with(|| a.cmp(&b))
    });

    let mut depth_mm = vec![0.0_f32; n];
    for &pit in &pits {
        let volume = volume_m[pit];
        if volume <= 0.0 {
            continue;
        }
        let (s, e) = span[pit];
        let comp = &by_pit[s..e];
        let spill = basins.spill_level[pit];
        let to_spill: f64 = comp
            .iter()
            .map(|&c| f64::from((spill - cells[c].elevation).max(0.0)))
            .sum();
        let level = if basins.is_terminal(pit, cells) || volume <= to_spill {
            #[expect(clippy::cast_possible_truncation)] // metres of depth, f32 is the engine's unit
            let volume_f32 = volume as f32;
            solve_flat_level(comp, cells, volume_f32)
        } else {
            volume_m[basins.downstream_pit(pit)] += volume - to_spill;
            spill
        };
        for &c in comp {
            depth_mm[c] = (level - cells[c].elevation).max(0.0) * 1000.0;
        }
    }

    let mut placed = 0.0_f32;
    for (cell, &depth) in grid.cells_slice_mut().iter_mut().zip(depth_mm.iter()) {
        cell.water_level = if depth > 0.0 {
            cell.water_capacity + depth
        } else {
            0.0
        };
        placed += cell.water_level;
    }
    placed
}

// ====================================================================
// Climate: the sampled-year balance on the real relief
// ====================================================================

/// Number of sampled days kept per cell for the frost count: the second
/// half of the year, from day 182 to the last sample before January 1st.
fn autumn_sample_count() -> usize {
    let stride = usize::from(terrain_insolation_sample_stride_days());
    let first_kept = 182_usize.div_ceil(stride) * stride;
    (365 - first_kept).div_ceil(stride)
}

/// What the climate sweep hands back: the calibrated temperature
/// parameters, the illumination cache it was computed with (ready for the
/// first tick), the per-cell normals, and two seasonal readings of the
/// same balance.
pub struct TerrainClimate {
    /// `params` with `aspect_correction` and `terrain_insolation_factor`
    /// measured on this relief, exactly what `Simulation::new` used to
    /// compute in two separate calls.
    pub params: TemperatureParams,
    /// Built and `ensure`d against the grid: adopted by the simulation.
    pub cache: IllumCache,
    /// Analytic climate normals, indexed like `cells_slice()`.
    pub normals: Vec<CellClimateNormals>,
    /// Expected daily-mean temperature (°C) on January 1st, the t0 field.
    pub temperature_day0: Vec<f32>,
    /// Number of consecutive sampled days before January 1st whose
    /// expected temperature sits below 0 °C: the frost run the snowpack
    /// accumulated through.
    pub frost_samples_before_day0: Vec<u16>,
}

/// Per-cell running statistics of the sampled year.
struct SeasonAccum {
    t_cloud_sum: f64,
    t_clear_min: f32,
    t_clear_max: f32,
    insolation_sum: f64,
}

/// The constants of the daily balance, read once from the parameters and
/// the relief (see [`terrain_climate`] for the equations).
struct DayBalance {
    /// `1 − cloud_albedo_coef × c̄`: the calibration cloud's shortwave factor.
    solar_cloud_factor: f32,
    /// `B(c̄) − σT0⁴` and `B(0) − σT0⁴`: the net infrared under the
    /// calibration cloud and under a clear sky (W/m²).
    longwave_cloud: f32,
    longwave_clear: f32,
    /// Lapse rate per metre (°C/m).
    gamma_per_m: f32,
    /// Map-mean elevation (m), the mixed air's reference.
    mean_elevation: f32,
    /// `t_ref` without the calibration offset: lapse and open-water
    /// cooling, per cell, and its map mean.
    t_ref_rel: Vec<f32>,
    mean_t_ref_rel: f32,
}

impl DayBalance {
    fn new(cells: &[CellProperties], params: &TemperatureParams) -> Self {
        let n = cells.len();
        let count = f64::from(exact_count(n));
        let cloud = params.mean_cloud_cover_for_calibration.clamp(0.0, 1.0);
        let gamma_per_m = params.lapse_rate / 1000.0;
        let t_ref_rel: Vec<f32> = cells
            .iter()
            .map(|c| {
                -gamma_per_m * c.elevation
                    - params.water_cooling * (1.0 + c.water_level / 1000.0).ln()
            })
            .collect();
        let mean = |values: &[f32]| -> f32 {
            if n == 0 {
                return 0.0;
            }
            #[expect(clippy::cast_possible_truncation)] // a mean of f32 values
            let m = (values.iter().map(|&v| f64::from(v)).sum::<f64>() / count) as f32;
            m
        };
        let elevations: Vec<f32> = cells.iter().map(|c| c.elevation).collect();
        Self {
            solar_cloud_factor: 1.0 - params.cloud_albedo_coef * cloud,
            longwave_cloud: ATMO_IR_BACK_CLEAR + cloud * ATMO_IR_BACK_CLOUDY_BOOST
                - STEFAN_BOLTZMANN_AT_T0,
            longwave_clear: ATMO_IR_BACK_CLEAR - STEFAN_BOLTZMANN_AT_T0,
            gamma_per_m,
            mean_elevation: mean(&elevations),
            mean_t_ref_rel: mean(&t_ref_rel),
            t_ref_rel,
        }
    }

    /// Map-mean daily temperature (offset excluded) for a map-mean
    /// absorbed shortwave `mean_flux` and a net infrared `longwave`.
    fn map_mean(&self, mean_flux: f32, longwave: f32) -> f32 {
        (mean_flux + longwave) / LIN_RADIATIVE_COEF + self.mean_t_ref_rel
    }

    /// One cell's daily temperature (offset excluded) given its absorbed
    /// shortwave, the net infrared and the map mean of the same sky.
    fn cell(&self, i: usize, elevation: f32, flux: f32, longwave: f32, t_bar: f32) -> f32 {
        let air = t_bar + self.gamma_per_m * (self.mean_elevation - elevation);
        (flux + longwave + LIN_RADIATIVE_COEF * self.t_ref_rel[i] + SENSIBLE_EXCHANGE_COEF * air)
            / (LIN_RADIATIVE_COEF + SENSIBLE_EXCHANGE_COEF)
    }
}

/// The sampled-year sweep: one illumination pass per daylight hour of
/// every sampled day, reduced per cell into [`SeasonAccum`], the autumn
/// temperatures (frost run) and the January 1st field, plus the sums the
/// terrain insolation factor is the ratio of.
struct ClimateSweep<'a> {
    grid: &'a HexGrid,
    cache: &'a IllumCache,
    params: &'a TemperatureParams,
    balance: DayBalance,
    accum: Vec<SeasonAccum>,
    autumn_t: Vec<f32>,
    autumn_slot: usize,
    temperature_day0: Vec<f32>,
    daily: Vec<f32>,
    flux_factor: Vec<f32>,
    illumination: Vec<f32>,
    flux_sum: f64,
    beam_sum: f64,
    sampled_days: u32,
}

impl<'a> ClimateSweep<'a> {
    fn new(grid: &'a HexGrid, cache: &'a IllumCache, params: &'a TemperatureParams) -> Self {
        let cells = grid.cells_slice();
        let n = cells.len();
        Self {
            grid,
            cache,
            params,
            balance: DayBalance::new(cells, params),
            accum: (0..n)
                .map(|_| SeasonAccum {
                    t_cloud_sum: 0.0,
                    t_clear_min: f32::INFINITY,
                    t_clear_max: f32::NEG_INFINITY,
                    insolation_sum: 0.0,
                })
                .collect(),
            autumn_t: vec![0.0; n * autumn_sample_count()],
            autumn_slot: 0,
            temperature_day0: vec![0.0; n],
            daily: vec![0.0; n],
            flux_factor: Vec::with_capacity(n),
            illumination: Vec::with_capacity(n),
            flux_sum: 0.0,
            beam_sum: 0.0,
            sampled_days: 0,
        }
    }

    /// The 24 hours of one sampled day: the clear-sky absorbed flux per
    /// cell (`daily`, W/m² daily mean), accumulated exactly as
    /// `terrain_annual_mean_insolation_factor` does for the factor.
    fn illuminate_day(&mut self, day: u16) {
        let n = self.grid.len();
        let cells_per_hour = f64::from(exact_count(n));
        self.daily.fill(0.0);
        for hour in 0..24_u64 {
            let hour_tick = u64::from(day) * 24 + hour;
            let beam = solar_beam_at_tick(self.params, hour_tick);
            if beam.s_u <= 0.0 {
                continue;
            }
            compute_illumination_cached(
                self.grid,
                &beam,
                0.0,
                1500.0,
                self.cache,
                &mut self.flux_factor,
                &mut self.illumination,
            );
            let cell_flux: f64 = self.flux_factor.iter().map(|&f| f64::from(f)).sum();
            self.flux_sum += cell_flux;
            self.beam_sum += f64::from(beam.s_u) * cells_per_hour;
            for (d, &ff) in self.daily.iter_mut().zip(self.flux_factor.iter()) {
                *d += beam.beam * ff;
            }
        }
        for d in &mut self.daily {
            *d /= 24.0;
        }
    }

    /// Solves the two skies of one sampled day and folds them into the
    /// per-cell statistics.
    fn sample_day(&mut self, day: u16) {
        self.illuminate_day(day);
        self.sampled_days += 1;
        let cells = self.grid.cells_slice();
        let n = cells.len();
        if n == 0 {
            return;
        }
        #[expect(clippy::cast_possible_truncation)] // a mean of f32 fluxes
        let mean_flux = (self.daily.iter().map(|&v| f64::from(v)).sum::<f64>()
            / f64::from(exact_count(n))) as f32;
        let b = &self.balance;
        let t_bar_cloud = b.map_mean(mean_flux * b.solar_cloud_factor, b.longwave_cloud);
        let t_bar_clear = b.map_mean(mean_flux, b.longwave_clear);
        let autumn = day >= 182;
        for (i, acc) in self.accum.iter_mut().enumerate() {
            let z = cells[i].elevation;
            let shortwave_cloud = self.daily[i] * b.solar_cloud_factor;
            let t_cloud = b.cell(i, z, shortwave_cloud, b.longwave_cloud, t_bar_cloud);
            let t_clear = b.cell(i, z, self.daily[i], b.longwave_clear, t_bar_clear);
            acc.t_cloud_sum += f64::from(t_cloud);
            acc.t_clear_min = acc.t_clear_min.min(t_clear);
            acc.t_clear_max = acc.t_clear_max.max(t_clear);
            acc.insolation_sum += f64::from(shortwave_cloud);
            if day == 0 {
                self.temperature_day0[i] = t_cloud;
            }
            if autumn {
                self.autumn_t[self.autumn_slot * n + i] = t_cloud;
            }
        }
        if autumn {
            self.autumn_slot += 1;
        }
    }

    /// Reduces the sweep: the terrain factor, the offset, the normals.
    fn finish(self, mut params: TemperatureParams, cache: IllumCache) -> TerrainClimate {
        debug_assert_eq!(self.autumn_slot, autumn_sample_count());
        let cells = self.grid.cells_slice();
        let n = cells.len();
        #[expect(clippy::cast_possible_truncation)] // a ratio of bounded sums, see the factor's doc
        let factor = if self.beam_sum <= 0.0 {
            1.0
        } else {
            (self.flux_sum / self.beam_sum) as f32
        };
        params.terrain_insolation_factor = factor;
        let offset = calibration_offset(&params, params.latitude_deg.to_radians());
        let days = f64::from(self.sampled_days.max(1));
        let normals: Vec<CellClimateNormals> = self
            .accum
            .iter()
            .zip(cells.iter())
            .map(|(acc, cell)| {
                #[expect(clippy::cast_possible_truncation)] // means of f32 values
                let (t_mean, insolation_mean) = (
                    (acc.t_cloud_sum / days) as f32 + offset,
                    (acc.insolation_sum / days) as f32,
                );
                let moisture_mean = cell.groundwater + cell.water_level;
                CellClimateNormals {
                    t_mean,
                    t_min: acc.t_clear_min + offset,
                    t_max: acc.t_clear_max + offset,
                    moisture_mean,
                    moisture_min: 0.0,
                    moisture_max: moisture_mean,
                    insolation_mean,
                }
            })
            .collect();
        let temperature_day0 = self.temperature_day0.iter().map(|t| t + offset).collect();
        let autumn_len = autumn_sample_count();
        let frost_samples_before_day0 = (0..n)
            .map(|i| {
                let mut frost = 0_u16;
                for slot in (0..autumn_len).rev() {
                    if self.autumn_t[slot * n + i] + offset < 0.0 {
                        frost += 1;
                    } else {
                        break;
                    }
                }
                frost
            })
            .collect();
        TerrainClimate {
            params,
            cache,
            normals,
            temperature_day0,
            frost_samples_before_day0,
        }
    }
}

/// The engine's linearised radiative balance, solved at the daily mean on
/// the real relief for every sampled day of the year, then reduced to the
/// annual normals. One illumination sweep of the year, the same one
/// `terrain_annual_mean_insolation_factor` runs (same days, same hours,
/// same accumulation, so the factor comes out bit for bit the same): the
/// terrain's own insolation deficit, the aspect tilt and the per-cell
/// daily flux are three readings of one sweep.
///
/// Daily mean, steady (soil responds in hours, a year is slow):
///
/// ```text
/// (LIN + SENS) × T_i = S_i + B − σT0⁴ + LIN × t_ref_i + SENS × T_air_i
/// T_air_i = T̄ + Γ (z̄ − z_i) / 1000,   T̄ = (S̄ + B − σT0⁴) / LIN + mean(t_ref)
/// ```
///
/// with `S_i` the day's mean absorbed shortwave of the cell and `B` the
/// downward infrared. Two skies: the **calibration cloud cover**
/// (`mean_cloud_cover_for_calibration`, on the shortwave through the cloud
/// albedo and on the infrared through the cloudy boost) for the annual
/// mean and the mean insolation, the **clear sky** for the sustained
/// extremes, a cold snap being the clear winter week and a heat wave the
/// clear summer one. The calibration offset is the engine's
/// (`calibration_offset`), applied after the sweep. Note what that
/// implies and the engine confirms: the offset leaves the cloud albedo
/// out of its solar term, so the map's annual mean lands
/// `cloud_albedo_coef × c̄ × S̄ / LIN` (≈ 7 K at the defaults) below
/// `base_temp`, in the engine's measured normals as in these (r30 × 3
/// seeds, 2026-10-01: 7.8-8.2 °C at the mean elevation for `base_temp`
/// 15).
///
/// Moisture normals read the stocks the grid holds when this runs:
/// `moisture_mean = groundwater + water_level`, the t0 endowment being the
/// climatological mean by construction; `moisture_min = 0`, because this
/// engine's water table has no wilting point and the summer
/// evapotranspiration empties an unfed reservoir (measured 2026-10-01, r30
/// × 3 seeds: the year's minimum is 2-8 % of the mean in every band, below
/// every positive drought threshold of the species table).
pub fn terrain_climate(grid: &mut HexGrid, params: TemperatureParams) -> TerrainClimate {
    compute_surface_normals(grid);
    let mut params = params;
    params.aspect_correction = aspect_insolation_correction(grid, &params);
    let mut cache = IllumCache::new();
    cache.ensure(grid);
    let stride = terrain_insolation_sample_stride_days();
    let mut sweep = ClimateSweep::new(grid, &cache, &params);
    let mut day = 0_u16;
    while day < 365 {
        sweep.sample_day(day);
        day += stride;
    }
    let sweep = sweep;
    let params_out = params.clone();
    let cache_out = cache.clone();
    sweep.finish(params_out, cache_out)
}

// ====================================================================
// Seeding: snow, vegetation, temperature
// ====================================================================

/// Snowpack on January 1st: the snowfall of the consecutive frost samples
/// before it, `frost_samples × stride_days × snowfall_mm_per_day`. Nothing
/// melts while the expected temperature stays below zero, and nothing
/// falls as snow before the first frost sample. A cell whose January is
/// expected above freezing starts bare.
pub fn seed_snowpack(grid: &mut HexGrid, climate: &TerrainClimate, snowfall_mm_per_day: f32) {
    let stride = f32::from(terrain_insolation_sample_stride_days());
    for (cell, &frost) in grid
        .cells_slice_mut()
        .iter_mut()
        .zip(climate.frost_samples_before_day0.iter())
    {
        cell.snow_level = f32::from(frost) * stride * snowfall_mm_per_day.max(0.0);
        cell.ice_level = 0.0;
    }
}

/// Salt of the canopy-age draw, separating its stream from the fire's and
/// the regime's.
const STAND_AGE_SALT: u64 = 0x5EED_A6E5;

/// Vegetation on January 1st: each stratum, canopy first, is seeded at the
/// cover the shared logistic equilibrium gives its best species under the
/// cell's normals and the light the strata above let through,
///
/// ```text
/// cover_S = k_total × max_i (1 − base_mortality × mortality_rel_i / (growth_rate × growth_rel_i × suit_i))
/// ```
///
/// (the steady state of `v' = g v suit (1 − v/k) − m v` for the species
/// whose growth-to-mortality ratio is best, zero when none grows), shared
/// among the stratum's species in proportion to their suitability. The
/// light each lower stratum reads is Beer-Lambert through the canopy just
/// seeded above it, so a closed canopy starts with the understory it can
/// carry and the ground layer it cannot. Open water carries nothing.
///
/// The canopy age is drawn per cell from the exponential distribution of
/// times since the last stand-replacing disturbance, mean
/// `stand_age_mean_years` (a fire rotation of a few decades in
/// Mediterranean and pre-alpine woodlands; Pausas 2004, *Climatic Change*
/// 63), hashed from the world seed and the cell so no two neighbours start
/// in step and a checkpoint restart replays the same draw.
pub fn seed_vegetation(
    grid: &mut HexGrid,
    normals: &[CellClimateNormals],
    veg: &VegetationParams,
    seed: u32,
    stand_age_mean_years: f32,
) {
    let coords: Vec<_> = grid.coords_slice().to_vec();
    for (i, cell) in grid.cells_slice_mut().iter_mut().enumerate() {
        cell.vegetation = [0.0; SPECIES_COUNT];
        cell.stand_age = 0.0;
        cell.fire_intensity = 0.0;
        let Some(n) = normals.get(i) else {
            continue;
        };
        if cell.is_open_water_at(veg.open_water_excess) {
            continue;
        }
        for stratum in STRATA.iter().rev() {
            seed_stratum(cell, n, veg, *stratum);
        }
        if stratum_cover(cell, Stratum::Tree) > 1e-4 {
            let draw = hash01(&[
                u64::from(seed),
                coord_word(coords[i].q),
                coord_word(coords[i].r),
                STAND_AGE_SALT,
            ]);
            cell.stand_age = -stand_age_mean_years.max(0.0) * (1.0 - draw).ln();
        }
    }
}

/// Seeds one stratum of `cell` (see [`seed_vegetation`]), reading the
/// light through the strata already seeded above it.
fn seed_stratum(
    cell: &mut CellProperties,
    n: &CellClimateNormals,
    veg: &VegetationParams,
    stratum: Stratum,
) {
    let light = n.insolation_mean * strata_light_transmittance(cell)[stratum.index()];
    let mut suits = [0.0_f32; SPECIES_COUNT];
    let mut best_equilibrium = 0.0_f32;
    let mut suit_sum = 0.0_f32;
    for (s, (species, suit)) in SPECIES.iter().zip(suits.iter_mut()).enumerate() {
        if species.stratum != stratum {
            continue;
        }
        *suit = species.suitability(n, light);
        let _ = s;
        if *suit <= 0.0 {
            continue;
        }
        let growth = veg.growth_rate * species.growth_rel * *suit;
        let mortality = veg.base_mortality * species.mortality_rel;
        if growth > 0.0 {
            best_equilibrium = best_equilibrium.max(1.0 - mortality / growth);
        }
        suit_sum += *suit;
    }
    if best_equilibrium <= 0.0 || suit_sum <= 0.0 {
        return;
    }
    let cover = veg.k_total * best_equilibrium;
    for (v, &suit) in cell.vegetation.iter_mut().zip(suits.iter()) {
        if suit > 0.0 {
            *v = cover * suit / suit_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::species::{SpeciesId, species_index};
    use crate::temperature::terrain_annual_mean_insolation_factor;
    use crate::terrain::{TerrainParams, generate_terrain};

    /// Radius-2 bowl: the centre 20 m below a rim, the rim 10 m below the
    /// outer ring except one gap cell at the rim's height minus 5 m, the
    /// spillway. Transport needs radius ≥ 2 (a radius-0 cell is its own
    /// neighbour on the torus).
    fn bowl() -> HexGrid {
        let mut grid = HexGrid::from_radius(2);
        let centre = HexCoord::new(0, 0);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            let d = coord.distance(centre);
            let elevation = match d {
                0 => 100.0,
                1 => 120.0,
                _ => 130.0,
            };
            let cell = grid.get_mut(coord).unwrap();
            cell.elevation = elevation;
            cell.water_capacity = 10.0;
            cell.permeability = 0.5;
        }
        grid
    }

    #[test]
    fn spill_level_of_a_bowl_is_its_lowest_rim() {
        let mut grid = bowl();
        // Lower one rim cell to 115 m: the spillway.
        let gap = HexCoord::new(1, 0);
        grid.get_mut(gap).unwrap().elevation = 115.0;
        // And a second, deeper pit outside the bowl so the bowl is not
        // the terminal basin.
        let far = HexCoord::new(2, 0);
        grid.get_mut(far).unwrap().elevation = 50.0;
        let basins = ReliefBasins::of(&grid);
        let centre = grid.index_of(HexCoord::new(0, 0)).unwrap();
        assert!(
            (basins.spill_level[centre] - 115.0).abs() < 1e-6,
            "spill {}",
            basins.spill_level[centre]
        );
        let far_i = grid.index_of(far).unwrap();
        assert!((basins.spill_level[far_i] - 50.0).abs() < 1e-6);
        assert_eq!(basins.pit_of[centre], centre);
        assert_eq!(basins.downstream_pit(centre), far_i);
    }

    #[test]
    fn runoff_sheet_fills_the_pit_and_conserves_its_catchment() {
        let grid = bowl();
        let basins = ReliefBasins::of(&grid);
        let n = grid.len();
        // Every cell drains to the centre: one pit, the terminal basin.
        assert!(basins.pit_of.iter().all(|&p| p == basins.pit_of[0]));
        let mut grid = grid;
        let sheet = 2.0_f32;
        let placed = fill_basins_with_runoff(&mut grid, &basins, sheet);
        let centre = grid.get(HexCoord::new(0, 0)).unwrap();
        // 19 cells × 2 mm = 38 mm of free depth in the single pit, over
        // its 10 mm bucket; 20 m to the rim, nothing else floods.
        // 0.01 mm: the f32 rounding of a level stated in metres over a
        // 100 m bedrock (ulp 7.6 µm), the same gap `lake::step_lake_leveling`
        // credits back; immaterial for a t0 endowment.
        let expected = 10.0 + sheet * exact_count(n);
        assert!(
            (centre.water_level - expected).abs() < 1e-2,
            "centre {} vs {expected}",
            centre.water_level
        );
        assert!((placed - expected).abs() < 1e-2);
        let dry = grid
            .cells_slice()
            .iter()
            .filter(|c| c.water_level == 0.0)
            .count();
        assert_eq!(dry, n - 1);
    }

    #[test]
    fn a_full_depression_pours_its_excess_downstream() {
        let mut grid = bowl();
        let gap = HexCoord::new(1, 0);
        grid.get_mut(gap).unwrap().elevation = 100.5; // spillway 0.5 m above the pit
        let far = HexCoord::new(2, 0);
        grid.get_mut(far).unwrap().elevation = 50.0;
        let basins = ReliefBasins::of(&grid);
        let centre = grid.index_of(HexCoord::new(0, 0)).unwrap();
        let far_i = grid.index_of(far).unwrap();
        let catchment: usize = basins.pit_of.iter().filter(|&&p| p == centre).count();
        assert!(catchment >= 2, "the bowl keeps a catchment");
        // 1000 mm per cell over the catchment: far more than the 500 mm
        // the bowl holds below its spillway.
        fill_basins_with_runoff(&mut grid, &basins, 1000.0);
        let c = grid.get(HexCoord::new(0, 0)).unwrap();
        assert!(
            (c.water_level - (10.0 + 500.0)).abs() < 1e-2,
            "the bowl stops at its spillway: {}",
            c.water_level
        );
        let f = grid.cells_slice()[far_i].water_level;
        assert!(f > 10.0 + 500.0, "the excess reached the lower basin: {f}");
        let total: f32 = grid.cells_slice().iter().map(|c| c.water_level).sum();
        let flooded = grid
            .cells_slice()
            .iter()
            .filter(|c| c.water_level > 0.0)
            .count();
        let buckets = 10.0 * exact_count(flooded);
        assert!(
            (total - buckets - 1000.0 * exact_count(grid.len())).abs() < 1.0,
            "sheet conserved: {total}"
        );
    }

    #[test]
    fn wetness_index_is_higher_in_the_valley_than_on_the_rim() {
        let grid = bowl();
        let twi = wetness_index(&grid);
        let centre = grid.index_of(HexCoord::new(0, 0)).unwrap();
        let rim = grid.index_of(HexCoord::new(1, 0)).unwrap();
        assert!(twi[centre] > twi[rim], "{} vs {}", twi[centre], twi[rim]);
    }

    #[test]
    fn groundwater_by_wetness_keeps_the_mean_endowment_and_stays_in_the_rock() {
        let mut grid = HexGrid::from_radius(4);
        generate_terrain(
            &mut grid,
            &TerrainParams {
                seed: 7,
                ..TerrainParams::default()
            },
        );
        let frac = 0.65;
        seed_groundwater_by_wetness(&mut grid, frac, 5.0);
        let twi = wetness_index(&grid);
        let (mut wet_sum, mut wet_n, mut dry_sum, mut dry_n) = (0.0, 0.0, 0.0, 0.0);
        let median = {
            let mut s = twi.clone();
            s.sort_by(f32::total_cmp);
            s[s.len() / 2]
        };
        for (cell, &index) in grid.cells_slice().iter().zip(twi.iter()) {
            let capacity = cell.permeability * DEFAULT_MAX_CAPACITY_MM;
            assert!(cell.groundwater >= 0.0 && cell.groundwater <= capacity + 1e-4);
            let f = cell.groundwater / capacity;
            if index > median {
                wet_sum += f;
                wet_n += 1.0;
            } else {
                dry_sum += f;
                dry_n += 1.0;
            }
        }
        assert!(wet_sum / wet_n > frac && dry_sum / dry_n < frac);
    }

    #[test]
    fn climate_sweep_reproduces_the_terrain_insolation_factor_bit_for_bit() {
        let mut grid = HexGrid::from_radius(6);
        generate_terrain(
            &mut grid,
            &TerrainParams {
                seed: 42,
                ..TerrainParams::default()
            },
        );
        let base = TemperatureParams::default();
        let climate = terrain_climate(&mut grid, base.clone());
        let mut reference = base.clone();
        reference.aspect_correction = aspect_insolation_correction(&grid, &reference);
        let mut cache = IllumCache::new();
        cache.ensure(&grid);
        let factor = terrain_annual_mean_insolation_factor(&grid, &cache, &reference);
        assert_eq!(
            climate.params.terrain_insolation_factor.to_bits(),
            factor.to_bits()
        );
        assert_eq!(
            climate.params.aspect_correction.to_bits(),
            reference.aspect_correction.to_bits()
        );
    }

    /// A flat, dry world, every cell alike: the normals are identical
    /// across cells, the clear-sky extremes bracket the mean by the
    /// seasonal swing of a 44.5° latitude, January is colder than the
    /// year, and the annual mean lands where the engine's calibration
    /// puts it: `base_temp` minus the calibration cloud's shortwave
    /// deficit, `cloud_albedo_coef × c̄ × S̄ / LIN` (the offset leaves the
    /// cloud albedo out of its solar term, see `terrain_climate`).
    #[test]
    fn analytic_normals_sit_on_the_calibrated_mean_and_bracket_it() {
        let mut grid = HexGrid::from_radius(2);
        let params = TemperatureParams::default();
        let climate = terrain_climate(&mut grid, params.clone());
        let plain = climate.normals[0];
        for n in &climate.normals {
            assert_eq!(n.t_mean.to_bits(), plain.t_mean.to_bits());
            assert_eq!(n.insolation_mean.to_bits(), plain.insolation_mean.to_bits());
        }
        assert!(
            plain.t_min < plain.t_mean - 8.0 && plain.t_max > plain.t_mean + 8.0,
            "{plain:?}"
        );
        let mean_flat = crate::temperature::SOLAR_CONSTANT
            * params.atmospheric_transmittance
            * (1.0 - params.ground_albedo)
            * crate::temperature::annual_mean_insolation_factor(params.latitude_deg.to_radians());
        let cloud = params.mean_cloud_cover_for_calibration;
        let expected =
            params.base_temp - params.cloud_albedo_coef * cloud * mean_flat / LIN_RADIATIVE_COEF;
        assert!(
            (plain.t_mean - expected).abs() < 0.5,
            "annual mean {} vs the calibration's {expected}",
            plain.t_mean
        );
        assert!(
            (plain.insolation_mean - mean_flat * (1.0 - params.cloud_albedo_coef * cloud)).abs()
                < 2.0,
            "insolation {} vs {}",
            plain.insolation_mean,
            mean_flat * (1.0 - params.cloud_albedo_coef * cloud)
        );
        assert!(
            climate.temperature_day0[0] < plain.t_mean - 5.0,
            "January is colder than the year"
        );
        assert_eq!(climate.frost_samples_before_day0.len(), grid.len());
    }

    /// Lapse rate: a flat world raised by 1000 m as a whole runs the lapse
    /// rate colder in every normal (its mean elevation rises with it, so
    /// the mixed air follows), with no shadow to muddle the comparison.
    #[test]
    fn a_world_raised_by_a_kilometre_runs_the_lapse_rate_colder() {
        let params = TemperatureParams::default();
        let mut low = HexGrid::from_radius(2);
        let mut high = HexGrid::from_radius(2);
        for c in high.cells_slice_mut() {
            c.elevation = 1000.0;
        }
        let low_climate = terrain_climate(&mut low, params.clone());
        let high_climate = terrain_climate(&mut high, params.clone());
        for (l, h) in low_climate.normals.iter().zip(high_climate.normals.iter()) {
            for (a, b) in [(l.t_mean, h.t_mean), (l.t_min, h.t_min), (l.t_max, h.t_max)] {
                assert!((a - b - params.lapse_rate).abs() < 1e-3, "{a} vs {b}");
            }
        }
        assert!(
            high_climate.frost_samples_before_day0[0] >= low_climate.frost_samples_before_day0[0],
            "the higher world freezes for at least as long"
        );
    }

    #[test]
    fn seeded_canopy_leaves_no_room_for_a_meadow_under_an_oak_forest() {
        let n = CellClimateNormals {
            t_mean: 14.0,
            t_min: -5.0,
            t_max: 34.0,
            moisture_mean: 30.0,
            moisture_min: 0.0,
            moisture_max: 30.0,
            insolation_mean: 120.0,
        };
        let mut grid = HexGrid::from_radius(0);
        let cell = grid.get_mut(HexCoord::new(0, 0)).unwrap();
        cell.water_capacity = 10.0;
        seed_vegetation(&mut grid, &[n], &VegetationParams::default(), 42, 30.0);
        let cell = grid.get(HexCoord::new(0, 0)).unwrap();
        let trees = stratum_cover(cell, Stratum::Tree);
        let herbs = stratum_cover(cell, Stratum::Herb);
        assert!(trees > 0.8, "closed canopy expected, got {trees}");
        assert!(herbs < 0.05, "no meadow under it, got {herbs}");
        assert!(
            cell.vegetation[species_index(SpeciesId::Beech)] < 1e-6,
            "beech needs 1 mm of sustained water, the dry season gives none"
        );
        assert!(cell.stand_age > 0.0);
    }

    #[test]
    fn canopy_ages_differ_between_neighbours_and_follow_the_seed() {
        let n = CellClimateNormals {
            t_mean: 14.0,
            t_min: -5.0,
            t_max: 34.0,
            moisture_mean: 30.0,
            moisture_min: 0.0,
            moisture_max: 30.0,
            insolation_mean: 120.0,
        };
        let normals = vec![n; 7];
        let mut a = HexGrid::from_radius(1);
        let mut b = HexGrid::from_radius(1);
        for g in [&mut a, &mut b] {
            for c in g.cells_slice_mut() {
                c.water_capacity = 10.0;
            }
        }
        seed_vegetation(&mut a, &normals, &VegetationParams::default(), 42, 30.0);
        seed_vegetation(&mut b, &normals, &VegetationParams::default(), 42, 30.0);
        let ages: Vec<f32> = a.cells_slice().iter().map(|c| c.stand_age).collect();
        assert!(ages.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-3));
        for (x, y) in a.cells_slice().iter().zip(b.cells_slice()) {
            assert_eq!(x.stand_age.to_bits(), y.stand_age.to_bits());
        }
        let mean = ages.iter().sum::<f32>() / 7.0;
        assert!(mean > 1.0 && mean < 120.0, "mean age {mean}");
    }

    #[test]
    fn open_water_is_never_seeded() {
        let n = CellClimateNormals {
            t_mean: 14.0,
            t_min: -5.0,
            t_max: 34.0,
            moisture_mean: 30.0,
            moisture_min: 0.0,
            moisture_max: 30.0,
            insolation_mean: 120.0,
        };
        let mut grid = HexGrid::from_radius(0);
        let cell = grid.get_mut(HexCoord::new(0, 0)).unwrap();
        cell.water_capacity = 10.0;
        cell.water_level = 500.0;
        seed_vegetation(&mut grid, &[n], &VegetationParams::default(), 42, 30.0);
        let cell = grid.get(HexCoord::new(0, 0)).unwrap();
        assert!(cell.vegetation.iter().all(|&v| v.abs() < f32::EPSILON));
        assert!(cell.stand_age.abs() < f32::EPSILON);
    }

    #[test]
    fn snowpack_follows_the_frost_run() {
        let mut grid = HexGrid::from_radius(1);
        let n = grid.len();
        let climate = TerrainClimate {
            params: TemperatureParams::default(),
            cache: IllumCache::new(),
            normals: vec![CellClimateNormals::default(); n],
            temperature_day0: vec![0.0; n],
            frost_samples_before_day0: (0..n).map(|i| u16::try_from(i).unwrap()).collect(),
        };
        seed_snowpack(&mut grid, &climate, 0.1);
        let stride = f32::from(terrain_insolation_sample_stride_days());
        for (i, c) in grid.cells_slice().iter().enumerate() {
            let expected = exact_count(i) * stride * 0.1;
            assert!((c.snow_level - expected).abs() < 1e-6);
        }
    }
}
