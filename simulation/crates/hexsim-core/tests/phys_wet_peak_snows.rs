//! **Dumb, blunt physics test**: pre-saturate the sky and put a cold
//! peak underneath, it must snow on the peak.
//!
//! Minimal setup (radius 2, 19 cells):
//! - Plain at 500m everywhere
//! - A mountain at the center (q=0, r=0): 2000m (very cold via lapse rate)
//! - `humidity_upper` = 0.5 pre-filled everywhere (saturated sky)
//!
//! Expected property: over 200 ticks, the peak must **reach** a
//! `snow_level` >= 0.5 at some point (winter accumulation). We track the
//! maximum reached, not the final value, because `season_amplitude=18`
//! fully melts the snow in mid-summer (tick 160+). The physics under
//! test is *accumulation*, not year-round persistence.
//!
//! Isolates one link of the full cycle: checks that at minimum, when
//! humidity is ALREADY there in the right place, snow falls. Independent
//! of transport / evaporation / uplift concerns.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const TICKS: u64 = 200;
// Phase 3 : rescale ×200 (0.5 → 100.0 mm). Pre-remplissage humidite d'altitude.
const PRE_SATURATED_UPPER: f32 = 100.0;
// Phase 3 (#32) : rescale ×200 (0.4 → 80 mm). Snow_level maintenant en mm.
const MIN_SNOW_ON_PEAK: f32 = 80.0;

fn build_scene() -> Simulation {
    let mut grid = HexGrid::from_radius(2);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 500.0;
            cell.temperature = 5.0;
            cell.humidity_upper = PRE_SATURATED_UPPER;
        }
    }
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.elevation = 2000.0;
        cell.temperature = -10.0;
    }
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    )
}

#[test]
fn cold_peak_under_saturated_sky_accumulates_snow() {
    let mut sim = build_scene();
    let mut max_snow = 0.0_f32;
    for _ in 0..TICKS {
        sim.step();
        let peak = sim.grid().get(HexCoord::new(0, 0)).expect("peak exists");
        if peak.snow_level > max_snow {
            max_snow = peak.snow_level;
        }
    }
    let peak = sim.grid().get(HexCoord::new(0, 0)).expect("peak exists");
    eprintln!(
        "phys_wet_peak_snows: over {TICKS} ticks, max_snow = {max_snow:.4}, \
         final snow = {:.4}, final T = {:.1}°C",
        peak.snow_level, peak.temperature
    );
    assert!(
        max_snow >= MIN_SNOW_ON_PEAK,
        "Cold peak under saturated sky: no snow accumulated \
         (max_snow = {max_snow:.4}, expected >= {MIN_SNOW_ON_PEAK}). \
         The condensation → solid precipitation cycle is broken."
    );
}
