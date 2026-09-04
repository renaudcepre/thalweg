//! Micro e2e (2026-09-02): a north-facing slope (ubac) is colder than the
//! south-facing slope (adret) at the same altitude (`phys_aspect_insolation`),
//! but that surface anomaly must NOT turn it into a permanent condenser
//! aloft. The upper air is horizontally mixed at the scale of the
//! terrarium: the two flanks share the same upper-air temperature, so
//! neither can rain every day while the other stays dry.
//!
//! This is the property whose loss was bisected to the aspect insolation
//! (e3594f9): with `T_upper = T_surface − Γ·H`, the coldest cells of the
//! map rained 365 days a year and the whole water cycle collapsed onto
//! them (0 rain-free day per year, JOURNAL 2026-09-02). Same built
//! east-west ridge as `phys_aspect_insolation`: twin flanks at equal
//! altitude, opposite normals, full pipeline over one year. Wet days
//! count rain AND snow: the flanks differ in phase (the ubac snows where
//! the adret rains), not in precipitation, and phase is not the property.
//!
//! Out of scope, printed only: the crest itself is wet 365 days a year.
//! 2026-09-02 read it as the nightly cooling of the upper air (the layer
//! followed the surface's diurnal swing). 2026-09-03 smoothed that swing
//! away (`Simulation::upper_air_mean_t`, `phys_upper_air_diurnal`) and
//! the crest is still wet 365 days: its upper air sits at RH = 1.000
//! around the clock in summer and drains 8-9 mm/day (1280 mm/yr, was
//! 1600 with the nightly cycle), a steady-state condenser fed hour after
//! hour by the orographic pump (LCL bound = saturation of the highest
//! neighbour) from a fixture where every cell is a 2 m lake. That is the
//! pump/drain balance of #63, not a diurnal artefact; tracked there, not
//! asserted here.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
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
const FLANK: i32 = 3;
const DAYS: u64 = 365;
/// No flank cell may be wet more than this many days a year: a real
/// temperate slope has well under 250 precipitation days, 300 leaves a
/// wide margin while still rejecting the 365-day condenser.
const MAX_WET_DAYS: u32 = 300;
/// Ubac / adret ratio of wet days: the cold flank must not precipitate
/// more than twice as often as its warm twin. One-sided on purpose: the
/// warm adret evaporates its own surface water faster (Tetens) and gets
/// it back as local precipitation (ratio 0.28 measured on 2026-09-02),
/// the local recycling pattern of #24, a different property.
const MAX_RATIO: f32 = 2.0;

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
    // Uniform light westerly (#108 seam): the property under test is the
    // aspect, not a convergence spot of the noise wind on a 127-cell torus.
    sim.set_uniform_wind(WindVec { x: 0.1, y: 0.0 });
    sim
}

/// Wet days over the run for every cell of flank `r` (all `q`).
fn flank_wet_days(wet_days: &[u32], sim: &Simulation, r: i32) -> Vec<u32> {
    sim.grid()
        .coords_slice()
        .iter()
        .enumerate()
        .filter(|(_, c)| c.r == r)
        .map(|(i, _)| wet_days[i])
        .collect()
}

/// Running mean without integer→float casts (the count is an exact f32).
fn mean(v: &[u32]) -> f32 {
    let mut count = 0.0_f32;
    let mut m = 0.0_f32;
    for &d in v {
        count += 1.0;
        m += (f32::from(u16::try_from(d).expect("fits u16")) - m) / count;
    }
    m
}

#[test]
fn ubac_does_not_become_a_permanent_condenser() {
    let mut sim = ridge_sim();
    let n = sim.grid().len();
    let peak = sim.grid().index_of(HexCoord::new(0, 0)).unwrap();
    let mut wet_days = vec![0_u32; n];
    let mut crest_rain_days = 0_u32;
    let mut crest_snow_days = 0_u32;
    let mut crest_mm = 0.0_f32;
    for _ in 0..DAYS {
        sim.step();
        for (i, rec) in sim.last_precipitation().iter().enumerate() {
            if rec.wet() {
                wet_days[i] += 1;
            }
        }
        let crest = sim.last_precipitation()[peak];
        crest_rain_days += u32::from(crest.rained());
        crest_snow_days += u32::from(crest.snowed());
        crest_mm += crest.rain + crest.snow;
    }
    let adret = flank_wet_days(&wet_days, &sim, FLANK);
    let ubac = flank_wet_days(&wet_days, &sim, -FLANK);
    let (adret_mean, ubac_mean) = (mean(&adret), mean(&ubac));
    let flank_max = adret.iter().chain(ubac.iter()).copied().max().unwrap_or(0);
    eprintln!(
        "phys_ubac_not_a_rain_attractor: wet days/yr adret={adret_mean:.1} ubac={ubac_mean:.1} \
         flank max={flank_max} crest={} (rain {crest_rain_days} d, snow {crest_snow_days} d, \
         {crest_mm:.0} mm/yr; crest chimney: #63, not asserted)",
        wet_days[peak]
    );
    assert!(
        flank_max <= MAX_WET_DAYS,
        "a flank cell is wet {flank_max} days a year (> {MAX_WET_DAYS}): permanent condenser"
    );
    let ratio = ubac_mean / adret_mean.max(1.0);
    assert!(
        ratio <= MAX_RATIO,
        "ubac / adret wet-day ratio {ratio:.2} > {MAX_RATIO}: the cold flank became a condenser"
    );
}
