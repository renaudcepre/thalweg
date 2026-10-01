//! Sentinel for the fix/rain-regime defect: `Simulation::snapshot()` must
//! put this hour-tick's precipitation flux on the wire
//! (`CellSnapshot::rain_amount`/`snow_amount`/`is_raining`,
//! `GridState::total_precip_this_tick`), NOT the midnight-to-now daily
//! accumulator (`last_precipitation`). Before the fix the front's overlay
//! painted a stock as if it were a flux (anti-pattern #3): the
//! painted fraction of the map grew monotonically across the day.
//!
//! Verified red before the fix: reverting `snapshot()` in
//! `simulation/accessors.rs` to pass `&self.last_precipitation` (the old
//! line) turns this test red starting at hour 2 of the run below — the
//! daily accumulator has summed 2 hours of rain by then while
//! `precip_this_tick()` (this test's oracle) still holds only the latest
//! hour, so `rain_amount + snow_amount` on the snapshot stops matching it.
//!
//! Setup: flat radius-2 world (19 cells, transport-safe per the
//! "no radius-0 transport" rule), `cloud_water = 1.0` everywhere so KK2000
//! keeps producing measurable rain/snow hour after hour, split half
//! warm (rain) / half sub-zero (snow) so both branches of
//! `rain_amount`/`snow_amount` get exercised. `cloud_advection_rate = 0`
//! (same freeze as `phys_kk2000_heavy_cloud_rains.rs`) keeps the cloud
//! camped on its cell so precipitation stays driven by KK2000 alone, not
//! by wind sweeping clouds off the map between hours.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

mod common;

fn wet_world(radius: i32) -> Simulation {
    let mut grid = HexGrid::from_radius(radius);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            // Half the map sub-zero (snow branch), half warm (rain
            // branch): exercises both `rain_amount` and `snow_amount`,
            // not just one of them.
            cell.temperature = if coord.q < 0 { -5.0 } else { 15.0 };
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
            cell.humidity_surface = 0.0;
            cell.humidity_upper = 0.0;
            cell.cloud_water = 1.0;
        }
    }

    // #63 L2b: the fixture used to leave `humidity_upper = 0` and rely on
    // the old `cloud_evap_rate` to keep the 1 mm cloud alive for hours.
    // The saturation adjustment that replaced it evaporates that cloud in
    // the first hour, and the run reached the "no cell ever rained" guard
    // below. `freeze_phase_transition` pins both directions of the
    // transition, which is what "precipitation driven by KK2000 alone"
    // already meant.
    let mut atmo = AtmosphereParams {
        cloud_advection_rate: 0.0,
        ..AtmosphereParams::default()
    };
    common::freeze_phase_transition(&mut atmo);
    Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    )
}

/// For each of the first 4 simulated hours (well past the first hour of
/// the day, where the daily accumulator and the tick flux still
/// coincide), the snapshot's per-cell precipitation must equal this
/// hour's flux exactly, stay within the physical per-hour cap, and the
/// grid-level total must equal the sum of the per-cell values.
#[test]
fn snapshot_precip_is_this_hour_not_the_daily_stock() {
    let mut sim = wet_world(2);
    let max_precip_per_tick = sim.atmosphere_params().max_precip_per_tick;
    assert!(
        max_precip_per_tick > 0.0,
        "fixture assumption: a real per-hour cap must be configured"
    );

    let mut any_rain = false;
    let mut any_snow = false;

    for hour in 0..4 {
        sim.step_hour();

        let tick_precip = sim.precip_this_tick().clone();
        let snapshot = sim.snapshot();

        assert_eq!(
            snapshot.cells.len(),
            tick_precip.len(),
            "hour {hour}: snapshot cell count must match the precip map"
        );

        let mut sum_rain_plus_snow = 0.0_f64;
        for (i, cell) in snapshot.cells.iter().enumerate() {
            let tick = tick_precip[i];

            // The whole point of the fix: the wire's rain/snow fields are
            // THIS hour's flux, not the daily accumulator. Before the fix
            // `snapshot()` sourced them from `last_precipitation` (the
            // midnight-to-now sum), which diverges from `tick` starting
            // hour 2 (see module doc).
            assert!(
                (cell.rain_amount - tick.rain).abs() < 1e-6,
                "hour {hour} cell {i}: snapshot.rain_amount={} != this hour's rain={} \
                 (snapshot is reading the daily accumulator, not the tick flux)",
                cell.rain_amount,
                tick.rain
            );
            assert!(
                (cell.snow_amount - tick.snow).abs() < 1e-6,
                "hour {hour} cell {i}: snapshot.snow_amount={} != this hour's snow={} \
                 (snapshot is reading the daily accumulator, not the tick flux)",
                cell.snow_amount,
                tick.snow
            );

            // Physical ceiling: a cell cannot receive more than
            // `max_precip_per_tick` mm of combined rain + snow in one
            // hour-tick (source cap `.min(max_precip_per_tick)` in
            // `fill_precip_outflow`, preserved through the neighbor
            // redistribution: self-share + 6 neighbor-shares of a
            // capped source sum back to at most the cap).
            let combined = cell.rain_amount + cell.snow_amount;
            assert!(
                combined <= max_precip_per_tick + 1e-4,
                "hour {hour} cell {i}: rain+snow={combined} exceeds the per-hour cap \
                 {max_precip_per_tick} (stock/flux confusion: a daily total would blow \
                 through this hourly ceiling, which is exactly the bug this test pins)"
            );

            any_rain |= cell.rain_amount > 1e-4;
            any_snow |= cell.snow_amount > 1e-4;
            sum_rain_plus_snow += f64::from(cell.rain_amount) + f64::from(cell.snow_amount);
        }

        assert!(
            (f64::from(snapshot.total_precip_this_tick) - sum_rain_plus_snow).abs() < 1e-3,
            "hour {hour}: total_precip_this_tick={} != sum of per-cell rain+snow={}",
            snapshot.total_precip_this_tick,
            sum_rain_plus_snow
        );
    }

    assert!(any_rain, "fixture not meaningful: no cell ever rained");
    assert!(any_snow, "fixture not meaningful: no cell ever snowed");
}
