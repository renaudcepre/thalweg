//! Per-cell climate history: 365-day ring buffer of precipitation
//! events (rain + snow). Used to characterize the **temporal
//! distribution** of rain (how many days it rained, what total
//! accumulation, what percentage of arid cells per elevation band)
//! rather than instantaneous values that hide the real geography.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::atmosphere::PrecipitationMap;
use crate::coord::HexCoord;
use crate::grid::HexGrid;

/// Precipitation event for one tick, for one cell.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DayRecord {
    pub rain: f32,
    pub snow: f32,
}

impl DayRecord {
    #[must_use]
    pub fn wet(self) -> bool {
        self.rain > 1e-4 || self.snow > 1e-4
    }
    #[must_use]
    pub fn rained(self) -> bool {
        self.rain > 1e-4
    }
    #[must_use]
    pub fn snowed(self) -> bool {
        self.snow > 1e-4
    }
}

/// Time window for queries.
#[derive(Clone, Copy, Debug, Serialize)]
pub enum Window {
    Last30,
    Last180,
    Last365,
}

impl Window {
    #[must_use]
    pub fn days(self) -> usize {
        match self {
            Window::Last30 => 30,
            Window::Last180 => 180,
            Window::Last365 => 365,
        }
    }
}

/// Elevation band for aggregates.
#[derive(Debug, Clone)]
pub struct AltBand {
    pub name: String,
    pub min_m: f32,
    pub max_m: f32,
}

impl AltBand {
    #[must_use]
    pub fn contains(&self, elev: f32) -> bool {
        elev >= self.min_m && elev < self.max_m
    }
    #[must_use]
    pub fn range(&self) -> String {
        let fmt = |v: f32| -> String {
            if v.is_infinite() {
                if v > 0.0 {
                    "+inf".to_string()
                } else {
                    "-inf".to_string()
                }
            } else {
                format!("{v:.0}m")
            }
        };
        format!("{}..{}", fmt(self.min_m), fmt(self.max_m))
    }
}

/// Default bands: low < 100m, mid 100-800m, high >= 800m.
#[must_use]
pub fn default_bands() -> Vec<AltBand> {
    vec![
        AltBand {
            name: "low".into(),
            min_m: f32::NEG_INFINITY,
            max_m: 100.0,
        },
        AltBand {
            name: "mid".into(),
            min_m: 100.0,
            max_m: 800.0,
        },
        AltBand {
            name: "high".into(),
            min_m: 800.0,
            max_m: f32::INFINITY,
        },
    ]
}

/// Per-cell ring buffer over `capacity` days.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateHistory {
    history: HashMap<HexCoord, VecDeque<DayRecord>>,
    capacity: usize,
}

impl Default for ClimateHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl ClimateHistory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
            capacity: 365,
        }
    }

    /// Records a tick for all known cells. `precip` is indexed by
    /// `HexGrid::cell_index` (size = `grid.len()`). Cells without rain
    /// receive `DayRecord::default()`.
    pub fn record_tick(&mut self, precip: &PrecipitationMap, grid: &HexGrid) {
        for (i, &coord) in grid.coords_slice().iter().enumerate() {
            let record = precip.get(i).copied().unwrap_or_default();
            let entry = self
                .history
                .entry(coord)
                .or_insert_with(|| VecDeque::with_capacity(self.capacity));
            if entry.len() >= self.capacity {
                entry.pop_front();
            }
            entry.push_back(record);
        }
    }

    /// Number of rain days for a cell within the window.
    #[must_use]
    pub fn rain_days(&self, coord: HexCoord, window: Window) -> u16 {
        self.count_days(coord, window, DayRecord::rained)
    }

    /// Number of snow days for a cell within the window.
    #[must_use]
    pub fn snow_days(&self, coord: HexCoord, window: Window) -> u16 {
        self.count_days(coord, window, DayRecord::snowed)
    }

    /// Total rain over the window.
    #[must_use]
    pub fn total_rain(&self, coord: HexCoord, window: Window) -> f32 {
        self.sum_days(coord, window, |d| d.rain)
    }

    /// Total snow over the window.
    #[must_use]
    pub fn total_snow(&self, coord: HexCoord, window: Window) -> f32 {
        self.sum_days(coord, window, |d| d.snow)
    }

    /// Number of days recorded for the cell (capped at capacity).
    #[must_use]
    pub fn days_recorded(&self, coord: HexCoord) -> usize {
        self.history.get(&coord).map_or(0, VecDeque::len)
    }

    fn count_days<F: Fn(DayRecord) -> bool>(
        &self,
        coord: HexCoord,
        window: Window,
        pred: F,
    ) -> u16 {
        let Some(hist) = self.history.get(&coord) else {
            return 0;
        };
        let n = window.days().min(hist.len());
        u16::try_from(hist.iter().rev().take(n).filter(|d| pred(**d)).count()).unwrap_or(u16::MAX)
    }

    fn sum_days<F: Fn(&DayRecord) -> f32>(
        &self,
        coord: HexCoord,
        window: Window,
        getter: F,
    ) -> f32 {
        let Some(hist) = self.history.get(&coord) else {
            return 0.0;
        };
        let n = window.days().min(hist.len());
        hist.iter().rev().take(n).map(getter).sum()
    }

    /// Aggregate per elevation band for a given window.
    #[must_use]
    pub fn aggregate(&self, grid: &HexGrid, bands: &[AltBand], window: Window) -> Vec<BandStats> {
        bands
            .iter()
            .map(|band| self.stats_for_band(grid, band, window))
            .collect()
    }

    fn stats_for_band(&self, grid: &HexGrid, band: &AltBand, window: Window) -> BandStats {
        let mut cells = 0_u32;
        let mut arid = 0_u32;
        let mut wet = 0_u32;
        let mut total_rain_days = 0_u32;
        let mut total_snow_days = 0_u32;
        let mut total_rain = 0.0_f32;
        let mut total_snow = 0.0_f32;

        for (&coord, cell) in grid.iter() {
            if !band.contains(cell.elevation) {
                continue;
            }
            cells += 1;
            let rd = u32::from(self.rain_days(coord, window));
            let sd = u32::from(self.snow_days(coord, window));
            total_rain_days += rd;
            total_snow_days += sd;
            total_rain += self.total_rain(coord, window);
            total_snow += self.total_snow(coord, window);
            if rd == 0 && sd == 0 {
                arid += 1;
            } else if rd + sd >= 10 {
                wet += 1;
            }
        }

        let f_cells = f32::from(u16::try_from(cells).unwrap_or(u16::MAX).max(1));
        BandStats {
            name: band.name.clone(),
            range: band.range(),
            cells,
            arid_cells: arid,
            wet_cells: wet,
            #[allow(clippy::cast_precision_loss)]
            avg_rain_days: total_rain_days as f32 / f_cells,
            #[allow(clippy::cast_precision_loss)]
            avg_snow_days: total_snow_days as f32 / f_cells,
            avg_total_rain: total_rain / f_cells,
            avg_total_snow: total_snow / f_cells,
        }
    }
}

/// Aggregated stats for an elevation band.
#[derive(Debug, Clone, Serialize)]
pub struct BandStats {
    pub name: String,
    pub range: String,
    pub cells: u32,
    /// Cells that received neither rain nor snow within the window.
    pub arid_cells: u32,
    /// Cells with at least 10 days of precipitation (rain+snow).
    pub wet_cells: u32,
    pub avg_rain_days: f32,
    pub avg_snow_days: f32,
    pub avg_total_rain: f32,
    pub avg_total_snow: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_precip(grid: &HexGrid) -> PrecipitationMap {
        vec![DayRecord::default(); grid.len()]
    }

    fn precip_at(grid: &HexGrid, entries: &[(HexCoord, DayRecord)]) -> PrecipitationMap {
        let mut p = empty_precip(grid);
        for &(coord, rec) in entries {
            if let Some(i) = grid.cell_index(coord) {
                p[i] = rec;
            }
        }
        p
    }

    fn grid_with_altitudes(altitudes: &[(HexCoord, f32)]) -> HexGrid {
        let max_dist = altitudes
            .iter()
            .map(|(c, _)| c.distance(HexCoord::new(0, 0)))
            .max()
            .unwrap_or(0);
        let mut grid = HexGrid::from_radius(max_dist);
        for &(coord, elev) in altitudes {
            if let Some(cell) = grid.get_mut(coord) {
                cell.elevation = elev;
            }
        }
        grid
    }

    #[test]
    fn ring_buffer_caps_at_capacity() {
        let grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        let mut history = ClimateHistory::new();

        let precip = precip_at(
            &grid,
            &[(
                coord,
                DayRecord {
                    rain: 1.0,
                    snow: 0.0,
                },
            )],
        );
        for _ in 0..400 {
            history.record_tick(&precip, &grid);
        }

        assert_eq!(history.days_recorded(coord), 365);
    }

    #[test]
    fn rain_days_count_last_n() {
        let grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        let mut history = ClimateHistory::new();

        // 10 dry days, then 5 days of rain
        let dry = empty_precip(&grid);
        let wet = precip_at(
            &grid,
            &[(
                coord,
                DayRecord {
                    rain: 0.5,
                    snow: 0.0,
                },
            )],
        );
        for _ in 0..10 {
            history.record_tick(&dry, &grid);
        }
        for _ in 0..5 {
            history.record_tick(&wet, &grid);
        }

        assert_eq!(history.rain_days(coord, Window::Last30), 5);
        assert_eq!(history.rain_days(coord, Window::Last180), 5);
    }

    #[test]
    fn snow_days_and_rain_days_are_independent() {
        let grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        let mut history = ClimateHistory::new();

        let p = precip_at(
            &grid,
            &[(
                coord,
                DayRecord {
                    rain: 0.0,
                    snow: 0.8,
                },
            )],
        );
        for _ in 0..3 {
            history.record_tick(&p, &grid);
        }

        assert_eq!(history.rain_days(coord, Window::Last30), 0);
        assert_eq!(history.snow_days(coord, Window::Last30), 3);
    }

    #[test]
    fn total_rain_sums_over_window() {
        let grid = HexGrid::from_radius(0);
        let coord = HexCoord::new(0, 0);
        let mut history = ClimateHistory::new();

        let p = precip_at(
            &grid,
            &[(
                coord,
                DayRecord {
                    rain: 0.25,
                    snow: 0.0,
                },
            )],
        );
        for _ in 0..4 {
            history.record_tick(&p, &grid);
        }

        assert!((history.total_rain(coord, Window::Last30) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn aggregate_separates_bands_and_detects_arid() {
        let cells = [
            (HexCoord::new(0, 0), 50.0),    // low, receives rain
            (HexCoord::new(1, 0), 50.0),    // low, arid
            (HexCoord::new(0, 1), 500.0),   // mid, arid
            (HexCoord::new(-1, 0), 1000.0), // high, receives
        ];
        let grid = grid_with_altitudes(&cells);
        let mut history = ClimateHistory::new();

        let p = precip_at(
            &grid,
            &[
                (
                    HexCoord::new(0, 0),
                    DayRecord {
                        rain: 0.1,
                        snow: 0.0,
                    },
                ),
                (
                    HexCoord::new(-1, 0),
                    DayRecord {
                        rain: 0.0,
                        snow: 0.2,
                    },
                ),
            ],
        );
        for _ in 0..30 {
            history.record_tick(&p, &grid);
        }

        let bands = default_bands();
        let stats = history.aggregate(&grid, &bands, Window::Last30);

        let low = stats.iter().find(|s| s.name == "low").unwrap();
        let mid = stats.iter().find(|s| s.name == "mid").unwrap();
        let high = stats.iter().find(|s| s.name == "high").unwrap();

        // Radius-1 grid = 7 cells. Explicitly defined: (0,0)=50, (1,0)=50,
        // (0,1)=500, (-1,0)=1000. The 3 remaining have elev=0 (default) => low.
        // low band: 5 cells (2 at 50m + 3 at 0m), 4 arid, 1 wet ((0,0))
        assert_eq!(low.cells, 5);
        assert_eq!(low.arid_cells, 4);
        assert_eq!(low.wet_cells, 1);

        assert_eq!(mid.cells, 1);
        assert_eq!(mid.arid_cells, 1);

        assert_eq!(high.cells, 1);
        assert_eq!(high.arid_cells, 0);
        assert!(high.avg_snow_days > 0.0);
    }

    #[test]
    fn empty_coord_returns_zero() {
        let history = ClimateHistory::new();
        let coord = HexCoord::new(42, 42);

        assert_eq!(history.rain_days(coord, Window::Last30), 0);
        assert!((history.total_rain(coord, Window::Last365) - 0.0).abs() < 1e-9);
    }
}
