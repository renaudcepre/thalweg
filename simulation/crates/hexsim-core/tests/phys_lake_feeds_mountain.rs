//! **Dumb and simple physical test**: water at the bottom → snow at the top.
//!
//! Minimal setup (radius 3, 37 cells):
//! - A lake pre-filled at the southwest edge (q=-3, r=0): `water_level=5.0`
//! - A mountain at the center (q=0, r=0): `elevation=1500m` (→ very cold
//!   `T_hiver`)
//! - The rest: plain at 100m
//!
//! Expected property: after 1500 ticks (~4 years) of simulation, the mountain
//! must have accumulated at least a little snow (`snow_level > 0.1`). It
//! doesn't matter which mechanism, humidity advection, orographic convection,
//! diffusion, as long as the system carries vapor from the lake to the peak
//! and condenses it. We extend to 1500 ticks because in some transients
//! (`T_peak` > 0 at the end of summer), 600 ticks can end during a period of
//! complete melt even though the mechanism works.
//!
//! This test is deliberately very permissive (0.1 snow threshold, 1500
//! ticks): it must fail ONLY if the lake→vapor→transport→condensation
//! chain is completely broken. It's an order-of-magnitude safety net, not
//! a fine measurement.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const TICKS: u64 = 1500;
// Phase 3 (#32): rescale ×200 (0.1 → 20 mm). Snow_level now in mm.
// v0.3.0 PR4 (#38): threshold lowered 20 → 10 mm. The continuous linear
// drain precipitation criterion (vs the old critical-mass trigger) produces
// a weaker but steadier discharge. The evap→condensation→snow chain works
// (~12 mm observed over 4 years), just less intense than before. The test
// remains an order-of-magnitude safety net to detect a total breakdown.
const MIN_SNOW_ON_PEAK: f32 = 10.0;

fn build_scene() -> Simulation {
    let mut grid = HexGrid::from_radius(3);

    // Plain at 100 m everywhere, then overrides for lake and mountain.
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 100.0;
            cell.temperature = 15.0;
        }
    }
    // Lake at the southwest edge. Phase 3 (#32): rescale ×200 (5 → 1000) to
    // reflect that water_level is now interpreted as mm of depth, a "sizeable
    // lake" = 1000 mm = 1 m of depth. Without this rescale, Meyer drains the
    // lake in 1-2 ticks (4 mm/day / 5 mm), cutting the evap → cloud → snow
    // chain before it can produce any flakes.
    if let Some(cell) = grid.get_mut(HexCoord::new(-3, 0)) {
        cell.water_level = 1000.0;
        cell.elevation = 50.0;
    }
    // Mountain at the center
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.elevation = 1500.0;
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
fn lake_evaporation_feeds_snow_on_mountain() {
    // #60 Phase 1: melt under the SI energy balance nibbles at the peak's
    // stock during hot, SUNNY spells (the old creeping rate barely did),
    // so the final snapshot now depends on the stopping moment: the
    // "end-of-summer transient" issue already noted in the header has
    // become permanent. This test pins the TRANSPORT chain (evaporation →
    // advection → condensation → snow), not the melt (it has its own
    // micro-tests in `snow.rs`): so we measure the MAX of the stock over
    // the run, did the chain produce the expected order of magnitude?,
    // and require that some remain at the end (the peak's winter guarantees
    // it).
    let mut sim = build_scene();
    let mut max_snow_on_peak = 0.0_f32;
    for _ in 0..TICKS {
        sim.step();
        let snow = sim
            .grid()
            .get(HexCoord::new(0, 0))
            .expect("the mountain exists")
            .snow_level;
        max_snow_on_peak = max_snow_on_peak.max(snow);
    }
    let peak = sim
        .grid()
        .get(HexCoord::new(0, 0))
        .expect("the mountain exists");
    eprintln!(
        "phys_lake_feeds_mountain: after {TICKS} ticks, \
         max(snow) = {max_snow_on_peak:.4}, peak.snow_level = {:.4}, \
         peak.temperature = {:.1}°C, peak.humidity_upper = {:.4}",
        peak.snow_level, peak.temperature, peak.humidity_upper
    );
    assert!(
        max_snow_on_peak >= MIN_SNOW_ON_PEAK,
        "The evaporation → advection → condensation → snow precipitation \
         chain never produced the expected order of magnitude on the \
         peak: max(snow_level) = {max_snow_on_peak:.4} over {TICKS} ticks \
         (expected >= {MIN_SNOW_ON_PEAK})."
    );
    assert!(
        peak.snow_level > 0.0,
        "The peak must keep snow at the end of the run (it's cold there): \
         snow_level = {:.4}",
        peak.snow_level
    );
}
