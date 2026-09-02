use serde::Serialize;

pub mod structural;

use crate::atmosphere::EvapStats;
use crate::coord::HexCoord;
use crate::dynamics::SynopticStats;
use crate::grid::HexGrid;
use crate::hydro::DischargeMap;
use crate::lithology::{LITHOLOGY, LITHOLOGY_COUNT, LithologyId};
use crate::wind::{WindField, WindVec};

/// Compact summary of the water cycle, designed to be readable by a human
/// or pasted into a chat without overwhelming the reader.
#[derive(Debug, Serialize)]
pub struct Diagnostics {
    pub tick: u64,
    pub cell_count: usize,
    pub water_budget: WaterBudget,
    pub surface: PropertyStats,
    pub humidity: PropertyStats,
    pub groundwater: PropertyStats,
    pub snow: PropertyStats,
    pub temperature: PropertyStats,
    pub elevation: PropertyStats,
    pub hydrology: HydrologyStats,
    pub wind: WindStats,
    pub altitude: AltitudeStats,
    /// Open-water evaporation (Dalton/Meyer ET₀, mm/day): filled in by
    /// `Simulation::diagnostics` from `AtmoScratch::evap`, itself written
    /// by `step_evaporation`'s `out` parameter during the tick that just
    /// ran. Same computation the engine applies, not a recomputation
    /// (there used to be a diagnostics-side clone of this formula that
    /// silently drifted from the engine's gate and wind source; removed).
    /// Zero (`EvapStats::default()`) via the low-level `compute_diagnostics`
    /// call, which only `Simulation::diagnostics` ever invokes.
    pub evap_observer: EvapStats,
    /// Phase 2 (#31) → Phase 3 (#32): `humidity_surface` is directly
    /// in mm of equivalent PW (×200 rescale of stocks in Phase 3). Real
    /// observational range: 5-75 mm depending on climate.
    pub humidity_surface_mm: PropertyStats,
    /// Synoptic dynamics (Phase 2 of the synoptic-dynamics design: system
    /// drift via mean-flow advection): filled in by `Simulation::diagnostics`
    /// (the state lives in the sim, not the grid); `None` only via the
    /// low-level `compute_diagnostics` call.
    pub synoptic: Option<SynopticStats>,
    /// Fluvial erosion (#105): filled in by `Simulation::diagnostics` (the
    /// EMA and the cumulative counters live in the sim); `None` via the
    /// low-level `compute_diagnostics` call.
    pub erosion: Option<ErosionStats>,
    /// Partition of the world by rock class (#136, L0).
    pub lithology: LithologyStats,
}

/// Breakdown of the mineral substrate (#136, L0), the metric that makes the
/// following couplings calibratable.
///
/// Without it, "does limestone carry more groundwater than granite?" has no
/// numeric answer, and L1/L2/L3 would be tuned blind (no physical balance
/// change without global metrics at scale). It is **observational**: no
/// phenomenon reads it, it only makes the substrate measurable.
#[derive(Debug, Serialize)]
pub struct LithologyStats {
    /// One entry per class, in the stable order of `lithology::LITHOLOGY`.
    pub classes: Vec<LithologyClassStats>,
}

/// Statistics for a rock class over the current world.
#[derive(Debug, Serialize)]
pub struct LithologyClassStats {
    pub id: LithologyId,
    pub cells: usize,
    /// Share of the world ∈ [0, 1]. A class at 0 is a dead class, the
    /// thresholds in `lithology::classify` need revisiting.
    pub share: f32,
    /// **Effective** average permeability of the class's cells: the table
    /// value attenuated by relief. So it drifts from the table, the
    /// more so the higher up the class sits (granite).
    pub mean_permeability: f32,
    /// Average elevation (m), reads directly off the bedrock/fill axis:
    /// granite should dominate marl.
    pub mean_elevation: f32,
    /// Average groundwater (mm) carried by the class. This is the **guard
    /// metric for L1**: when the table separates limestone and granite
    /// (gated by #107), this is where the gap should show up.
    pub mean_groundwater: f32,
}

/// Fluvial erosion metrics (#105), drainage convergence and rock budget,
/// the measurable "done-when" criteria for the issue.
#[derive(Debug, Serialize)]
pub struct ErosionStats {
    pub enabled: bool,
    /// Gini of the discharge EMA ∈ [0, 1[: rises as drainage converges
    /// (parallel rills → dendritic network). See `erosion::discharge_gini`.
    pub gini_discharge_ema: f64,
    /// Sediment load in transit, Σ over the map (m rock-equivalent).
    pub sediment_in_transit_m: f64,
    /// Cumulative bedrock incision since creation (m Σ over the map).
    pub incised_total_m: f64,
    /// Cumulative redeposited load since creation (m Σ over the map).
    pub deposited_total_m: f64,
    /// Closed depressions in the bedrock (strict local minima, toroidal
    /// neighbors).
    pub closed_depressions: usize,
}

/// Wind field statistics.
#[derive(Debug, Serialize)]
pub struct WindStats {
    pub mean_magnitude: f32,
    pub max_magnitude: f32,
    pub mean_direction_deg: f32,
}

/// Altitude-weather correlation: compares high vs low cells.
#[derive(Debug, Serialize)]
pub struct AltitudeStats {
    pub median_elevation: f32,
    pub humidity_high: f32,
    pub humidity_low: f32,
    pub snow_high: f32,
    pub snow_low: f32,
    pub raining_high: usize,
    pub raining_low: usize,
    pub temp_high: f32,
    pub temp_low: f32,
    /// Temperature-vs-elevation slope measured by linear regression over
    /// all cells, in °C/km. `params.lapse_rate` convention:
    /// positive = colder at altitude (expected physical behavior).
    /// Negative = thermal inversion (likely bug: wind advection too
    /// strong, or convective coupling flattening the adiabatic gradient).
    ///
    /// Significant divergence from `params.lapse_rate` signals that the
    /// dynamic terms (advection, `water_cooling`, cloud albedo) have
    /// dominated the configured gradient, see test `scale_climate_lapse_rate`.
    pub effective_lapse_rate_c_per_km: f32,
}

/// Total water cycle budget (conservation = surface + humidity + groundwater).
#[derive(Debug, Serialize)]
pub struct WaterBudget {
    pub surface: f32,
    pub humidity: f32,
    pub groundwater: f32,
    pub snow: f32,
    pub total: f32,
}

/// Coordinates of a notable cell.
#[derive(Debug, Serialize)]
pub struct CellRef {
    pub q: i32,
    pub r: i32,
    pub value: f32,
}

/// Distribution statistics for a property + extreme cells.
#[derive(Debug, Serialize)]
pub struct PropertyStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub stddev: f32,
    pub total: f32,
    pub p25: f32,
    pub median: f32,
    pub p75: f32,
    pub nonzero_count: usize,
    /// The 6 cells with the highest values.
    pub top: Vec<CellRef>,
    /// The 6 cells with the lowest values (non-zero for water/humidity/gw).
    pub bottom: Vec<CellRef>,
}

/// Hydrology metrics: rivers, puddles, overflows.
#[derive(Debug, Serialize)]
pub struct HydrologyStats {
    /// Cells with outflow above the river display threshold.
    pub river_cells: usize,
    pub max_discharge: f32,
    /// Cells with water trapped within their capacity (puddle/pool sub-hex).
    pub puddle_cells: usize,
    /// Cells whose water surplus exceeds capacity (lake or overflow).
    pub overflow_cells: usize,
    pub raining_cells: usize,
}

/// Computes the diagnostics from the simulation's current state.
#[must_use]
pub(crate) fn compute_diagnostics(
    grid: &HexGrid,
    tick: u64,
    discharge_map: &DischargeMap,
    wind_field: &WindField,
    precipitation: &crate::atmosphere::PrecipitationMap,
) -> Diagnostics {
    let cell_count = grid.len();

    let mut water_entries: Vec<(HexCoord, f32)> = Vec::with_capacity(cell_count);
    let mut humidity_entries: Vec<(HexCoord, f32)> = Vec::with_capacity(cell_count);
    let mut humidity_surface_mm_entries: Vec<(HexCoord, f32)> = Vec::with_capacity(cell_count);
    let mut gw_entries: Vec<(HexCoord, f32)> = Vec::with_capacity(cell_count);
    let mut snow_entries: Vec<(HexCoord, f32)> = Vec::with_capacity(cell_count);
    let mut temp_entries: Vec<(HexCoord, f32)> = Vec::with_capacity(cell_count);
    let mut elev_entries: Vec<(HexCoord, f32)> = Vec::with_capacity(cell_count);

    let mut total_surface = 0.0_f32;
    let mut total_humidity = 0.0_f32;
    let mut total_groundwater = 0.0_f32;
    let mut total_snow = 0.0_f32;
    let mut raining_cells = 0_usize;
    let mut puddle_cells = 0_usize;
    let mut overflow_cells = 0_usize;

    for (i, (&coord, cell)) in grid.iter().enumerate() {
        water_entries.push((coord, cell.water_level));
        humidity_entries.push((coord, cell.humidity_total()));
        humidity_surface_mm_entries.push((coord, cell.humidity_surface));
        gw_entries.push((coord, cell.groundwater));
        snow_entries.push((coord, cell.snow_level));
        temp_entries.push((coord, cell.temperature));
        elev_entries.push((coord, cell.elevation));

        total_surface += cell.water_level;
        total_humidity += cell.humidity_total();
        total_groundwater += cell.groundwater;
        total_snow += cell.snow_level;

        if cell.water_level > cell.water_capacity {
            overflow_cells += 1;
        } else if cell.water_level > 1e-4 {
            puddle_cells += 1;
        }

        if precipitation
            .get(i)
            .is_some_and(|d| d.rain > 1e-4 || d.snow > 1e-4)
        {
            raining_cells += 1;
        }
    }

    let river_threshold = 0.5;
    let mut river_cells = 0_usize;
    let mut max_discharge = 0.0_f32;
    for &d in discharge_map {
        if d > river_threshold {
            river_cells += 1;
        }
        if d > max_discharge {
            max_discharge = d;
        }
    }

    let wind_stats = compute_wind_stats(wind_field);
    let lithology_stats = compute_lithology_stats(grid);

    elev_entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let median_elev = if elev_entries.is_empty() {
        0.0
    } else {
        elev_entries[elev_entries.len() / 2].1
    };
    let altitude_stats = compute_altitude_stats(grid, median_elev, precipitation);

    Diagnostics {
        tick,
        cell_count,
        synoptic: None,
        erosion: None,
        water_budget: WaterBudget {
            surface: total_surface,
            humidity: total_humidity,
            groundwater: total_groundwater,
            snow: total_snow,
            total: total_surface + total_humidity + total_groundwater + total_snow,
        },
        surface: property_stats(&mut water_entries),
        humidity: property_stats(&mut humidity_entries),
        groundwater: property_stats(&mut gw_entries),
        snow: property_stats(&mut snow_entries),
        temperature: property_stats(&mut temp_entries),
        elevation: property_stats(&mut elev_entries),
        hydrology: HydrologyStats {
            river_cells,
            max_discharge,
            puddle_cells,
            overflow_cells,
            raining_cells,
        },
        wind: wind_stats,
        altitude: altitude_stats,
        // Filled in by `Simulation::diagnostics` from `AtmoScratch::evap`,
        // see the field doc-comment: this low-level call has no tick
        // context to draw a real value from.
        evap_observer: EvapStats::default(),
        humidity_surface_mm: property_stats(&mut humidity_surface_mm_entries),
        lithology: lithology_stats,
    }
}

/// Partition of the world by rock class (#136, L0). A dedicated pass: the
/// diagnostics are not on the hot path (called on demand), and mixing these
/// counters into the water cycle loop made it unreadable.
/// Aggregates accumulate in `f64` (thousands of terms) and come back down to
/// `f32` for transport: the truncation is the intended output format, not
/// an accident, same usage as the other observers in this module.
#[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn compute_lithology_stats(grid: &HexGrid) -> LithologyStats {
    let mut cells = [0_usize; LITHOLOGY_COUNT];
    let mut perm = [0.0_f64; LITHOLOGY_COUNT];
    let mut elev = [0.0_f64; LITHOLOGY_COUNT];
    let mut gw = [0.0_f64; LITHOLOGY_COUNT];

    for (_, cell) in grid.iter() {
        let i = cell.lithology.index();
        cells[i] += 1;
        perm[i] += f64::from(cell.permeability);
        elev[i] += f64::from(cell.elevation);
        gw[i] += f64::from(cell.groundwater);
    }

    let total = grid.len();
    LithologyStats {
        classes: LITHOLOGY
            .iter()
            .enumerate()
            .map(|(i, litho)| {
                let n = cells[i];
                // A missing class reports 0, never a NaN: a diagnostic
                // must not poison whatever reads it.
                let mean = |sum: f64| if n == 0 { 0.0 } else { sum / n as f64 };
                LithologyClassStats {
                    id: litho.id,
                    cells: n,
                    share: if total == 0 {
                        0.0
                    } else {
                        (n as f64 / total as f64) as f32
                    },
                    mean_permeability: mean(perm[i]) as f32,
                    mean_elevation: mean(elev[i]) as f32,
                    mean_groundwater: mean(gw[i]) as f32,
                }
            })
            .collect(),
    }
}

/// Effective temp-vs-elevation slope over the whole grid, in °C/km.
///
/// Linear regression `temp = a * elev + b`, then converts `a` (°C/m) to
/// °C/km with an inverted sign to match the `params.lapse_rate` convention
/// (positive = cooling with altitude).
///
/// Uses f64 for the accumulation to stay stable when the sim
/// contains >10000 cells (sums over elevations can exceed 1e8).
#[must_use]
pub fn effective_lapse_rate_c_per_km(grid: &HexGrid) -> f32 {
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_product = 0.0_f64;
    let mut sum_x_squared = 0.0_f64;
    let mut n = 0_u32;
    for (_, cell) in grid.iter() {
        let x = f64::from(cell.elevation);
        let y = f64::from(cell.temperature);
        sum_x += x;
        sum_y += y;
        sum_product += x * y;
        sum_x_squared += x * x;
        n += 1;
    }
    if n < 2 {
        return 0.0;
    }
    let nf = f64::from(n);
    let mean_x = sum_x / nf;
    let mean_y = sum_y / nf;
    let num = sum_product - nf * mean_x * mean_y;
    let den = sum_x_squared - nf * mean_x * mean_x;
    if den < 1e-3 {
        // No elevation variance: slope undefined, return 0.
        return 0.0;
    }
    let slope = num / den; // °C/m
    #[allow(clippy::cast_possible_truncation)]
    let result = (-slope * 1000.0) as f32; // °C/km, positive = colder at altitude
    result
}

fn compute_altitude_stats(
    grid: &HexGrid,
    median_elev: f32,
    precipitation: &crate::atmosphere::PrecipitationMap,
) -> AltitudeStats {
    let mut hum_high = 0.0_f32;
    let mut hum_low = 0.0_f32;
    let mut snow_high = 0.0_f32;
    let mut snow_low = 0.0_f32;
    let mut temp_high = 0.0_f32;
    let mut temp_low = 0.0_f32;
    let mut rain_high = 0_usize;
    let mut rain_low = 0_usize;
    let mut n_high = 0_u16;
    let mut n_low = 0_u16;

    for (i, (_coord, cell)) in grid.iter().enumerate() {
        let raining = precipitation
            .get(i)
            .is_some_and(|d| d.rain > 1e-4 || d.snow > 1e-4);

        if cell.elevation >= median_elev {
            hum_high += cell.humidity_total();
            snow_high += cell.snow_level;
            temp_high += cell.temperature;
            if raining {
                rain_high += 1;
            }
            n_high += 1;
        } else {
            hum_low += cell.humidity_total();
            snow_low += cell.snow_level;
            temp_low += cell.temperature;
            if raining {
                rain_low += 1;
            }
            n_low += 1;
        }
    }

    let fh = f32::from(n_high.max(1));
    let fl = f32::from(n_low.max(1));
    AltitudeStats {
        median_elevation: median_elev,
        humidity_high: hum_high / fh,
        humidity_low: hum_low / fl,
        snow_high: snow_high / fh,
        snow_low: snow_low / fl,
        raining_high: rain_high,
        raining_low: rain_low,
        temp_high: temp_high / fh,
        temp_low: temp_low / fl,
        effective_lapse_rate_c_per_km: effective_lapse_rate_c_per_km(grid),
    }
}

fn compute_wind_stats(wind_field: &WindField) -> WindStats {
    if wind_field.is_empty() {
        return WindStats {
            mean_magnitude: 0.0,
            max_magnitude: 0.0,
            mean_direction_deg: 0.0,
        };
    }
    let mut sum_mag = 0.0_f32;
    let mut max_mag = 0.0_f32;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;

    for &WindVec { x, y } in wind_field {
        let mag = (x * x + y * y).sqrt();
        sum_mag += mag;
        if mag > max_mag {
            max_mag = mag;
        }
        sum_x += x;
        sum_y += y;
    }

    let n = f32::from(u16::try_from(wind_field.len()).unwrap_or(u16::MAX));
    let mean_dir = sum_y.atan2(sum_x).to_degrees();

    WindStats {
        mean_magnitude: sum_mag / n,
        max_magnitude: max_mag,
        mean_direction_deg: (mean_dir + 360.0) % 360.0,
    }
}

const TOP_N: usize = 6;

fn property_stats(entries: &mut [(HexCoord, f32)]) -> PropertyStats {
    if entries.is_empty() {
        return PropertyStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            stddev: 0.0,
            total: 0.0,
            p25: 0.0,
            median: 0.0,
            p75: 0.0,
            nonzero_count: 0,
            top: Vec::new(),
            bottom: Vec::new(),
        };
    }

    entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let n = entries.len();
    let nf = f32::from(u16::try_from(n).unwrap_or(u16::MAX));
    let sum: f32 = entries.iter().map(|(_, v)| v).sum();
    let mean = sum / nf;
    let variance: f32 = entries.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f32>() / nf;
    let stddev = variance.sqrt();
    let nonzero_count = entries.iter().filter(|(_, v)| *v > 1e-6).count();

    // Top 6 (highest values, from max downward)
    let top: Vec<CellRef> = entries
        .iter()
        .rev()
        .take(TOP_N)
        .map(|(c, v)| CellRef {
            q: c.q,
            r: c.r,
            value: *v,
        })
        .collect();

    // Bottom 6 (lowest values)
    let bottom: Vec<CellRef> = entries
        .iter()
        .take(TOP_N)
        .map(|(c, v)| CellRef {
            q: c.q,
            r: c.r,
            value: *v,
        })
        .collect();

    PropertyStats {
        min: entries[0].1,
        max: entries[n - 1].1,
        mean,
        stddev,
        total: sum,
        p25: entries[n / 4].1,
        median: entries[n / 2].1,
        p75: entries[3 * n / 4].1,
        nonzero_count,
        top,
        bottom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::PrecipitationMap;

    #[test]
    fn property_stats_on_known_distribution() {
        // 10 entries with values 1..=10 in arbitrary order: after sorting, the
        // percentiles fall at known indices.
        let mut entries: Vec<(HexCoord, f32)> = (1..=10_i32)
            .rev()
            .map(|i| (HexCoord::new(i, 0), f32::from(i16::try_from(i).unwrap())))
            .collect();
        let stats = property_stats(&mut entries);
        assert!((stats.min - 1.0).abs() < 1e-5, "min={}", stats.min);
        assert!((stats.max - 10.0).abs() < 1e-5, "max={}", stats.max);
        assert!((stats.mean - 5.5).abs() < 1e-5, "mean={}", stats.mean);
        // n=10: median = entries[5].1 = 6.0 (index n/2 after sorting)
        assert!((stats.median - 6.0).abs() < 1e-5, "median={}", stats.median);
        // p25 = entries[2] = 3.0; p75 = entries[7] = 8.0
        assert!((stats.p25 - 3.0).abs() < 1e-5, "p25={}", stats.p25);
        assert!((stats.p75 - 8.0).abs() < 1e-5, "p75={}", stats.p75);
        // total = 1+2+...+10 = 55
        assert!((stats.total - 55.0).abs() < 1e-4, "total={}", stats.total);
        assert_eq!(stats.nonzero_count, 10);
        assert_eq!(stats.top.len(), TOP_N);
        assert_eq!(stats.bottom.len(), TOP_N);
    }

    #[test]
    fn property_stats_empty_returns_zero() {
        let mut entries: Vec<(HexCoord, f32)> = Vec::new();
        let stats = property_stats(&mut entries);
        assert!(stats.min.abs() < 1e-6);
        assert!(stats.max.abs() < 1e-6);
        assert!(stats.mean.abs() < 1e-6);
        assert!(stats.total.abs() < 1e-6);
        assert_eq!(stats.nonzero_count, 0);
        assert!(stats.top.is_empty());
        assert!(stats.bottom.is_empty());
    }

    #[test]
    fn property_stats_top_and_bottom_are_6() {
        // 20 entries: top[0]=max=20, top[5]=15; bottom[0]=min=1, bottom[5]=6
        let mut entries: Vec<(HexCoord, f32)> = (1..=20_i32)
            .map(|i| (HexCoord::new(i, 0), f32::from(i16::try_from(i).unwrap())))
            .collect();
        let stats = property_stats(&mut entries);
        assert_eq!(stats.top.len(), TOP_N);
        assert_eq!(stats.bottom.len(), TOP_N);
        assert!((stats.top[0].value - 20.0).abs() < 1e-5);
        assert!((stats.top[5].value - 15.0).abs() < 1e-5);
        assert!((stats.bottom[0].value - 1.0).abs() < 1e-5);
        assert!((stats.bottom[5].value - 6.0).abs() < 1e-5);
    }

    #[test]
    fn wind_stats_uniform_field() {
        // 7 cells all with wind (1, 0): direction = 0 deg, magnitude = 1
        let grid = HexGrid::from_radius(1);
        let field: WindField = vec![WindVec { x: 1.0, y: 0.0 }; grid.len()];
        let stats = compute_wind_stats(&field);
        assert!(
            (stats.mean_magnitude - 1.0).abs() < 1e-5,
            "mean_mag={}",
            stats.mean_magnitude
        );
        assert!(
            (stats.max_magnitude - 1.0).abs() < 1e-5,
            "max_mag={}",
            stats.max_magnitude
        );
        // atan2(0, 7) = 0 rad; modulo 360 can give 0 or ~360 depending on floats.
        assert!(
            stats.mean_direction_deg < 1.0 || stats.mean_direction_deg > 359.0,
            "expected direction ~0 deg, got {}",
            stats.mean_direction_deg
        );
    }

    #[test]
    fn wind_stats_empty_field_returns_zero() {
        let field: WindField = WindField::new();
        let stats = compute_wind_stats(&field);
        assert!(stats.mean_magnitude.abs() < 1e-6);
        assert!(stats.max_magnitude.abs() < 1e-6);
    }

    #[test]
    fn altitude_stats_partitions_by_median() {
        // Radius 1 = 7 cells. Cleanly tiered: 3 high (elev=1000,
        // humidity 0.5, cold), 3 low (elev=0, humidity 0.1, warm), 1 in the
        // middle (elev=500). Median=500: the partitioner uses >= median.
        let mut grid = HexGrid::from_radius(1);
        let coords: Vec<HexCoord> = grid.coords().copied().collect();
        for (i, coord) in coords.iter().enumerate() {
            if let Some(cell) = grid.get_mut(*coord) {
                if i < 3 {
                    cell.elevation = 1000.0;
                    cell.humidity_upper = 0.5;
                    cell.temperature = 0.0;
                } else if i < 6 {
                    cell.elevation = 0.0;
                    cell.humidity_upper = 0.1;
                    cell.temperature = 20.0;
                } else {
                    cell.elevation = 500.0;
                    cell.humidity_upper = 0.25;
                    cell.temperature = 10.0;
                }
            }
        }
        let stats = compute_altitude_stats(&grid, 500.0, &PrecipitationMap::new());
        assert!(
            (stats.median_elevation - 500.0).abs() < 1e-6,
            "median failed to be passed through: {}",
            stats.median_elevation
        );
        assert!(
            stats.humidity_high > stats.humidity_low,
            "humidity_high ({}) must exceed humidity_low ({})",
            stats.humidity_high,
            stats.humidity_low
        );
        assert!(
            stats.temp_high < stats.temp_low,
            "temp_high ({}) must be lower than temp_low ({})",
            stats.temp_high,
            stats.temp_low
        );
    }
}
