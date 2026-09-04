//! Micro e2e (2026-09-03): the upper layer keeps the seasons and the
//! lapse with elevation, not the day/night swing of the surface.
//!
//! The upper-air temperature is anchored to `Simulation::upper_air_mean_t`,
//! a first-order smoothing (τ = 24 h, `UPPER_AIR_SMOOTHING_TAU_S`) of the
//! instantaneous map-mean surface temperature. In the free troposphere
//! the diurnal range is ~1 K against ~8-10 K at the surface (Stull 1988);
//! before this anchor the whole 1500 m layer cooled by the surface's
//! nightly drop and re-saturated over the highest cells every night
//! (JOURNAL 2026-09-02). Full pipeline on the built east-west ridge of
//! `phys_ubac_not_a_rain_attractor` (127 cells, uniform wind, a real
//! diurnal cycle at the surface: 19 K of range in the map mean in
//! summer, measured 2026-09-03 with the smoothed range at 2.9 K).
//!
//! Two properties, both measured before being asserted:
//! - in situ attenuation: the daily range of the anchor is at most 0.2
//!   of the daily range of the instantaneous map mean (first-order
//!   filter at 24 h: 0.157; measured 0.153 on days 150-180);
//! - the seasons pass: the anchor in high summer sits well above the
//!   anchor in the first winter month (measured +17 °C vs −8 °C).

use hexsim_core::atmosphere::{AtmosphereParams, surface_means};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::{WindParams, WindVec};

const RADIUS: i32 = 6;
const PEAK_M: f32 = 500.0;
const SLOPE_M_PER_R: f32 = 60.0;
const HOURS_PER_DAY: u32 = 24;
/// Days 150-180: high summer, the largest surface diurnal range.
const SUMMER_FROM: u32 = 150;
const SUMMER_TO: u32 = 180;
/// Days 30-60: the anchor's winter level, once the EMA transient of the
/// first days (3τ) is long gone.
const WINTER_FROM: u32 = 30;
const WINTER_TO: u32 = 60;
/// Upper bound on the in-situ daily range ratio anchor / instantaneous.
/// A first-order filter at τ = 24 h gives 0.157 for a pure 24 h sinusoid;
/// the surface cycle is not a pure sinusoid (its harmonics are damped
/// even more), 0.2 leaves room for the residual while rejecting any
/// anchor that lets the night through.
const MAX_DIURNAL_RATIO: f32 = 0.2;
/// The seasonal contrast that must survive the smoothing (measured
/// 25 °C between the two windows on this ridge).
const MIN_SEASONAL_CONTRAST_C: f32 = 10.0;

fn ridge_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        let z = PEAK_M - SLOPE_M_PER_R * f32::from(i16::try_from(coord.r.abs()).unwrap_or(0));
        grid.get_mut(coord).unwrap().elevation = z;
        grid.get_mut(coord).unwrap().water_level = 2.0;
    }
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );
    sim.set_uniform_wind(WindVec { x: 0.1, y: 0.0 });
    sim
}

/// Running mean without integer→float casts (the count is an exact f32).
struct Mean {
    count: f32,
    value: f32,
}

impl Mean {
    const fn new() -> Self {
        Self {
            count: 0.0,
            value: 0.0,
        }
    }

    fn push(&mut self, x: f32) {
        self.count += 1.0;
        self.value += (x - self.value) / self.count;
    }
}

#[test]
fn upper_air_anchor_keeps_the_seasons_not_the_nights() {
    let mut sim = ridge_sim();
    let mut summer_ratio = Mean::new();
    let mut summer_anchor = Mean::new();
    let mut winter_anchor = Mean::new();
    let mut summer_surface_range = Mean::new();
    let mut summer_anchor_range = Mean::new();
    for day in 0..SUMMER_TO {
        let mut inst_lo = f32::INFINITY;
        let mut inst_hi = f32::NEG_INFINITY;
        let mut anchor_lo = f32::INFINITY;
        let mut anchor_hi = f32::NEG_INFINITY;
        for _ in 0..HOURS_PER_DAY {
            sim.step_hour();
            let (inst, _) = surface_means(sim.grid());
            let anchor = sim.upper_air_mean_t();
            inst_lo = inst_lo.min(inst);
            inst_hi = inst_hi.max(inst);
            anchor_lo = anchor_lo.min(anchor);
            anchor_hi = anchor_hi.max(anchor);
        }
        let anchor_mid = f32::midpoint(anchor_lo, anchor_hi);
        if (WINTER_FROM..WINTER_TO).contains(&day) {
            winter_anchor.push(anchor_mid);
        }
        if (SUMMER_FROM..SUMMER_TO).contains(&day) {
            summer_ratio.push((anchor_hi - anchor_lo) / (inst_hi - inst_lo));
            summer_anchor.push(anchor_mid);
            summer_surface_range.push(inst_hi - inst_lo);
            summer_anchor_range.push(anchor_hi - anchor_lo);
        }
    }
    eprintln!(
        "phys_upper_air_diurnal: summer daily range surface={:.1} K anchor={:.1} K \
         ratio={:.3}; anchor winter={:.1} °C summer={:.1} °C",
        summer_surface_range.value,
        summer_anchor_range.value,
        summer_ratio.value,
        winter_anchor.value,
        summer_anchor.value
    );
    assert!(
        summer_ratio.value <= MAX_DIURNAL_RATIO,
        "the upper-air anchor follows the surface's day/night swing: daily range ratio \
         {:.3} > {MAX_DIURNAL_RATIO}",
        summer_ratio.value
    );
    assert!(
        summer_anchor.value - winter_anchor.value >= MIN_SEASONAL_CONTRAST_C,
        "the seasons must pass through the anchor: summer {:.1} °C vs winter {:.1} °C \
         (< {MIN_SEASONAL_CONTRAST_C} K apart)",
        summer_anchor.value,
        winter_anchor.value
    );
}
