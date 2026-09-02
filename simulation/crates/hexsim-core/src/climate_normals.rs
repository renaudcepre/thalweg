//! Climate normals per cell (#79), foundation of species ecological niche
//! model (epic #78).
//!
//! Aggregates over a **calendar year** climate descriptors per cell: temperature
//! (mean/min/max), available root water (mean/min/max), and mean absorbed
//! insolation. Entry to species *suitability* calculation (#80).
//!
//! **Read-only.** These are aggregates: no mutation of water or grid energy,
//! terrarium conservation unchanged.
//!
//! **Window.** Calendar year, frozen at year boundary (`day_of_year` resets to 0).
//! Year N-1 normals serve year N (assumed 1-year lag: vegetation responds to
//! past climate, not instantaneous). Before first complete year, `normals()`
//! returns defaults.

use serde::{Deserialize, Serialize};

use crate::grid::HexGrid;
use crate::time::TICKS_PER_DAY_F32;

/// Lethal extreme smoothing window (days). Species death shouldn't come from
/// instantaneous spike (afternoon at 46 C, moment of dry soil) but from stress
/// sustained over days (more realistic). `suitability` cutoffs read extremes of
/// moving average (EMA) over this window, not raw extreme. Applies to temperature
/// AND humidity, all species.
const LETHAL_SMOOTHING_DAYS: f32 = 3.0;
/// EMA coefficient per hourly tick: `α = 1 / (days × ticks/day)`.
const LETHAL_EMA_ALPHA: f32 = 1.0 / (LETHAL_SMOOTHING_DAYS * TICKS_PER_DAY_F32);

/// Aggregated climate descriptors per cell over a year (SI units).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CellClimateNormals {
    /// Annual mean temperature (°C).
    pub t_mean: f32,
    /// SUSTAINED cold: minimum of the smoothed temperature (EMA ~a few days)
    /// over the year (°C), lethal frost trigger. A single cold night doesn't
    /// trip it, only a persistent cold snap.
    pub t_min: f32,
    /// SUSTAINED heat: maximum of the smoothed temperature (EMA) over the
    /// year (°C). A single scorching afternoon doesn't count, a heat wave
    /// does.
    pub t_max: f32,
    /// Mean root-available water (mm) = `groundwater + water_level`.
    pub moisture_mean: f32,
    /// SUSTAINED drought: minimum of the smoothed available water (EMA) over
    /// the year (mm). A one-day dip doesn't kill, a drought lasting several
    /// days does.
    pub moisture_min: f32,
    /// Maximum available water (mm).
    pub moisture_max: f32,
    /// Mean absorbed shortwave flux (W/m²), modulated by clouds.
    pub insolation_mean: f32,
}

impl Default for CellClimateNormals {
    fn default() -> Self {
        Self {
            t_mean: 0.0,
            t_min: 0.0,
            t_max: 0.0,
            moisture_mean: 0.0,
            moisture_min: 0.0,
            moisture_max: 0.0,
            insolation_mean: 0.0,
        }
    }
}

/// Per-cell accumulator for the current year. Sums in `f64` to limit
/// rounding drift over ~8760 ticks/year.
///
/// The lethal extremes (`t_ema_min/max`, `moist_ema_min`) are the extremes
/// of a moving average (EMA over `LETHAL_SMOOTHING_DAYS`), not raw extremes:
/// an isolated spike (a single scorching afternoon) barely moves them, only
/// sustained stress over several days trips them.
#[derive(Clone, Serialize, Deserialize)]
struct CellAccum {
    t_sum: f64,
    t_ema: f32,
    t_ema_min: f32,
    t_ema_max: f32,
    moisture_sum: f64,
    moist_ema: f32,
    moist_ema_min: f32,
    moisture_max: f32,
    insolation_sum: f64,
    count: u32,
    initialized: bool,
}

impl CellAccum {
    fn new() -> Self {
        Self {
            t_sum: 0.0,
            t_ema: 0.0,
            t_ema_min: f32::INFINITY,
            t_ema_max: f32::NEG_INFINITY,
            moisture_sum: 0.0,
            moist_ema: 0.0,
            moist_ema_min: f32::INFINITY,
            moisture_max: f32::NEG_INFINITY,
            insolation_sum: 0.0,
            count: 0,
            initialized: false,
        }
    }

    fn record(&mut self, t: f32, moisture: f32, insolation: f32) {
        self.t_sum += f64::from(t);
        self.moisture_sum += f64::from(moisture);
        self.insolation_sum += f64::from(insolation);
        self.moisture_max = self.moisture_max.max(moisture);

        if self.initialized {
            // EMA: smooths hourly spikes; only trips on sustained stress.
            self.t_ema += LETHAL_EMA_ALPHA * (t - self.t_ema);
            self.moist_ema += LETHAL_EMA_ALPHA * (moisture - self.moist_ema);
        } else {
            // Bootstrap on the 1st value (no transient artifact).
            self.t_ema = t;
            self.moist_ema = moisture;
            self.initialized = true;
        }
        self.t_ema_min = self.t_ema_min.min(self.t_ema);
        self.t_ema_max = self.t_ema_max.max(self.t_ema);
        self.moist_ema_min = self.moist_ema_min.min(self.moist_ema);
        self.count += 1;
    }

    #[allow(clippy::cast_possible_truncation)]
    fn finalize(&self) -> CellClimateNormals {
        if self.count == 0 {
            return CellClimateNormals::default();
        }
        let n = f64::from(self.count);
        CellClimateNormals {
            t_mean: (self.t_sum / n) as f32,
            // SMOOTHED extremes (EMA): death on sustained stress, not on a spike.
            t_min: self.t_ema_min,
            t_max: self.t_ema_max,
            moisture_mean: (self.moisture_sum / n) as f32,
            moisture_min: self.moist_ema_min,
            moisture_max: self.moisture_max,
            insolation_mean: (self.insolation_sum / n) as f32,
        }
    }
}

/// Accumulates climate normals per cell, year by year.
///
/// Usage cycle (driven by `Simulation`): `record_tick` every hour,
/// `finalize_year` at the year rollover. `normals()` exposes the last
/// complete year.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClimateNormalsAccumulator {
    acc: Vec<CellAccum>,
    normals: Vec<CellClimateNormals>,
    finalized_once: bool,
}

impl ClimateNormalsAccumulator {
    #[must_use]
    pub fn new(cell_count: usize) -> Self {
        Self {
            acc: vec![CellAccum::new(); cell_count],
            normals: vec![CellClimateNormals::default(); cell_count],
            finalized_once: false,
        }
    }

    /// Accumulates one tick. `flux_factor[i]` = per-cell illumination factor
    /// (aspect × relief occlusion × cloud shadow, cf
    /// `temperature::compute_illumination`); absorbed insolation =
    /// `beam_mag · flux_factor[i]`, single source of truth shared with the
    /// thermal balance.
    pub fn record_tick(&mut self, grid: &HexGrid, flux_factor: &[f32], beam_mag: f32) {
        for ((a, cell), &ff) in self.acc.iter_mut().zip(grid.cells_slice()).zip(flux_factor) {
            let moisture = cell.groundwater + cell.water_level;
            a.record(cell.temperature, moisture, beam_mag * ff);
        }
    }

    /// Freezes the elapsed year's normals and resets the accumulators to zero.
    /// Call at the year rollover (after a full year of `record_tick`).
    pub fn finalize_year(&mut self) {
        for (n, a) in self.normals.iter_mut().zip(self.acc.iter_mut()) {
            *n = a.finalize();
            *a = CellAccum::new();
        }
        self.finalized_once = true;
    }

    /// Normals for the last complete year (indexed like `cells_slice`).
    /// Default values until at least one year has completed.
    #[must_use]
    pub fn normals(&self) -> &[CellClimateNormals] {
        &self.normals
    }

    /// `true` as soon as at least one year has been finalized.
    #[must_use]
    pub fn has_normals(&self) -> bool {
        self.finalized_once
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellProperties;
    use crate::coord::HexCoord;

    /// 1-cell grid fixed to a given state, to drive the accumulators.
    fn one_cell(temperature: f32, groundwater: f32, water_level: f32, cloud_water: f32) -> HexGrid {
        let mut grid = HexGrid::from_radius(0);
        *grid.get_mut(HexCoord::new(0, 0)).unwrap() = CellProperties {
            temperature,
            water_level,
            cloud_water,
            groundwater,
            ..Default::default()
        };
        grid
    }

    #[test]
    fn no_normals_before_first_year() {
        let acc = ClimateNormalsAccumulator::new(1);
        assert!(!acc.has_normals());
        assert_eq!(acc.normals()[0], CellClimateNormals::default());
    }

    #[test]
    fn constant_climate_gives_constant_normals() {
        // Constant climate: mean = min = max = the constant, no NaN.
        let grid = one_cell(12.0, 30.0, 5.0, 0.0);
        let mut acc = ClimateNormalsAccumulator::new(1);
        // Night same as day: clear sky is null at night, so we force a
        // constant flux to test the insolation aggregate separately.
        for _ in 0..100 {
            acc.record_tick(&grid, &[1.0_f32], 400.0);
        }
        acc.finalize_year();

        let n = acc.normals()[0];
        assert!((n.t_mean - 12.0).abs() < 1e-4, "t_mean={}", n.t_mean);
        assert!((n.t_min - 12.0).abs() < 1e-4);
        assert!((n.t_max - 12.0).abs() < 1e-4);
        // moisture = groundwater + water_level = 35.
        assert!(
            (n.moisture_mean - 35.0).abs() < 1e-4,
            "moist={}",
            n.moisture_mean
        );
        assert!((n.moisture_min - 35.0).abs() < 1e-4);
        assert!((n.moisture_max - 35.0).abs() < 1e-4);
        // Clear sky 400, no cloud, absorbed = 400.
        assert!(
            (n.insolation_mean - 400.0).abs() < 1e-3,
            "insol={}",
            n.insolation_mean
        );
    }

    #[test]
    fn smoothed_min_max_bracket_mean() {
        // Long cyclical series (the EMA has time to follow): the SMOOTHED
        // extremes bracket the mean, but stay INSIDE the raw amplitude
        // (smoothing shaves off the spikes).
        let mut acc = ClimateNormalsAccumulator::new(1);
        for i in 0..2000 {
            #[allow(clippy::cast_precision_loss)]
            let t = 10.0 + 15.0 * ((i as f32) * 0.1).sin(); // oscillates 10 ± 15
            acc.record_tick(&one_cell(t, 0.0, 0.0, 0.0), &[1.0_f32], 0.0);
        }
        acc.finalize_year();

        let n = acc.normals()[0];
        assert!((n.t_mean - 10.0).abs() < 1.0, "mean≈10, got {}", n.t_mean);
        assert!(
            n.t_min <= n.t_mean && n.t_mean <= n.t_max,
            "brackets the mean"
        );
        // Smoothing: the extremes are strictly inside [-5, 25].
        assert!(
            n.t_min > -5.0 && n.t_max < 25.0,
            "EMA smoothed to: [{}, {}]",
            n.t_min,
            n.t_max
        );
    }

    #[test]
    fn brief_spike_does_not_trigger_lethal_extreme() {
        // Key property (#87): an isolated spike (one afternoon at 50°C) must
        // NOT trip t_max; only sustained stress would. Same for an isolated
        // drought dip on moisture_min.
        let mut acc = ClimateNormalsAccumulator::new(1);
        for _ in 0..500 {
            acc.record_tick(&one_cell(10.0, 5.0, 0.0, 0.0), &[1.0_f32], 0.0); // stable
        }
        acc.record_tick(&one_cell(50.0, 0.0, 0.0, 0.0), &[1.0_f32], 0.0); // 1 hot + dry spike
        for _ in 0..500 {
            acc.record_tick(&one_cell(10.0, 5.0, 0.0, 0.0), &[1.0_f32], 0.0);
        }
        acc.finalize_year();

        let n = acc.normals()[0];
        // Without smoothing t_max=50 and moisture_min=0; with EMA they barely move.
        assert!(
            n.t_max < 12.0,
            "an isolated spike must not yield t_max={}",
            n.t_max
        );
        assert!(
            n.moisture_min > 4.0,
            "an isolated dip must not yield moisture_min={}",
            n.moisture_min
        );
    }

    #[test]
    fn finalize_resets_for_next_year() {
        // Year 1 warm, year 2 cold: the normals must reflect the elapsed
        // year, not the accumulation of both.
        let mut acc = ClimateNormalsAccumulator::new(1);
        for _ in 0..10 {
            acc.record_tick(&one_cell(20.0, 0.0, 0.0, 0.0), &[1.0_f32], 0.0);
        }
        acc.finalize_year();
        for _ in 0..10 {
            acc.record_tick(&one_cell(0.0, 0.0, 0.0, 0.0), &[1.0_f32], 0.0);
        }
        acc.finalize_year();

        assert!((acc.normals()[0].t_mean - 0.0).abs() < 1e-4, "year 2 cold");
    }
}
