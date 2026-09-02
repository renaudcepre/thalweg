//! **3-stock atmospheric model**: vapor (invisible) / droplets (visible
//! clouds) / precipitation. Separates what the physics used to conflate
//! when `humidity_upper` served both as "there is a cloud" and "it's going
//! to rain". The concrete expected consequence: clouds persist without
//! precipitating immediately, and a cell doesn't rain continuously for 3
//! months because `cloud_water` is an intermediate reservoir that rebuilds
//! after each shower.
//!
//! Expected transitions:
//! - `humidity_upper → cloud_water`: condensation when RH > 0.6
//! - `cloud_water → humidity_upper`: cloud evaporation if RH < 0.4
//! - `cloud_water → water_level / snow_level`: precipitation when
//!   `cloud_water` exceeds a critical threshold (collision/coalescence)

use hexsim_core::atmosphere::{AtmosphereParams, saturation_upper};
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

/// Minimal setup: a single cell (radius 0), preloaded atmosphere.
fn build_single_cell(humidity_upper: f32, temperature: f32) -> Simulation {
    let mut grid = HexGrid::from_radius(0);
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.elevation = 500.0;
        cell.temperature = temperature;
        cell.humidity_upper = humidity_upper;
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

/// **Test 1**: a supersaturated cell (RH > 1) must form a cloud
/// (`cloud_water > 0`) after a few ticks, without torrential rain.
/// The cloud exists as an intermediate reservoir fed by CC.
///
/// Calibration (#63 Phase 4 Step 3): modest surplus (`humidity_upper` just
/// above saturation) so as not to saturate KK2000, latency now comes from
/// KK2000's non-linearity (`P_auto` ∝ `q_c`^2.47) rather than progressive
/// draining. With immediate CC draining (rate=1.0/h, cf `condensation_rate`
/// doc-comment), supersaturation resolves in 1 tick per Pruppacher & Klett;
/// the cloud→rain latency is still carried by KK2000's non-linearity.
#[test]
fn cloud_forms_before_rain() {
    let atmo = AtmosphereParams::default();
    let temp = TemperatureParams::default();
    // T_surface = 5 °C → t_upper ≈ -4.75 °C → saturation ≈ 5 mm (Tetens).
    // humidity_upper = 7.0 mm → modest 2 mm surplus → cloud_water ≈ 2 mm
    // at tick 1, KK2000 produces moderate drizzle (q_c^2.47 sublinear for
    // q_c < 1) rather than an intense downpour.
    let mut sim = build_single_cell(7.0, 5.0);
    let water_before = sim
        .grid()
        .get(HexCoord::new(0, 0))
        .map_or(0.0, |c| c.water_level);

    // 5 ticks: enough to trigger condensation, not enough to accumulate
    // cloud_water up to the critical precipitation threshold.
    for _ in 0..5 {
        sim.step();
    }

    let cell = sim
        .grid()
        .get(HexCoord::new(0, 0))
        .expect("the cell exists");
    let t_upper = cell.temperature - temp.lapse_rate * 1.5;
    let sat = saturation_upper(t_upper, &atmo);
    let hr = if sat > 0.0 {
        cell.humidity_upper / sat
    } else {
        0.0
    };

    eprintln!(
        "cloud_forms_before_rain after 5 ticks: \
         humidity_upper={:.3}, cloud_water={:.4}, sat={:.3}, HR={:.2}, \
         water_level delta={:.4}",
        cell.humidity_upper,
        cell.cloud_water,
        sat,
        hr,
        cell.water_level - water_before
    );
    assert!(
        cell.cloud_water > 0.0,
        "Cloud must have formed (cloud_water > 0). \
         Condensation vapor → cloud is not working."
    );
    // Precipitation must have latency: no intense downpour in 5 ticks.
    let rain_delta = cell.water_level - water_before;
    // Phase 3: threshold rescaled ×200 (0.05 → 10.0) for consistency with
    // the new rain fluxes in mm/tick.
    assert!(
        rain_delta < 10.0,
        "Too much rain in 5 ticks (delta={rain_delta:.3}). \
         cloud_water must be an intermediate reservoir, not a direct pass-through."
    );
}

/// **Test 2**: an existing cloud must be able to dissipate through
/// evaporation if the atmosphere around it dries out, without necessarily
/// raining. It's the symmetric counterpart of condensation: `cloud_water`
/// → vapor.
///
/// Surface evaporation is disabled to isolate the mechanism: otherwise the
/// rain that fell (even very little) evaporates → recharges
/// `humidity_upper` → re-condenses, masking the dissipation we're after.
#[test]
fn cloud_dissipates_when_air_dries() {
    let atmo = AtmosphereParams {
        uplift_rate: 0.0,
        uplift_thermal_coef: 0.0,
        // Disables the vapor floor at startup: otherwise humidity_upper
        // is forced to 0.15 → RH > 1 → condenses instead of dissipating.
        initial_humidity_floor: 0.0,
        ..AtmosphereParams::default()
    };
    let temp = TemperatureParams::default();

    let mut grid = HexGrid::from_radius(0);
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.elevation = 500.0;
        cell.temperature = 15.0; // Warm T_upper → high saturation ~43 mm
        cell.humidity_upper = 4.0; // RH ~ 0.09, dry air (Phase 3 rescale)
        cell.cloud_water = 16.0; // Existing cloud (Phase 3 rescale)
    }
    // Disables the seasonal cycle and relaxation: otherwise the cell's T
    // converges to average base_temp within a few ticks and overrides
    // the scenario's initial 15 °C.
    let temp_params = TemperatureParams {
        latitude_deg: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        atmo.clone(),
        GroundwaterParams::default(),
        SnowParams::default(),
        temp_params,
        WindParams::default(),
    );

    let water_before = sim
        .grid()
        .get(HexCoord::new(0, 0))
        .map_or(0.0, |c| c.water_level);
    let cloud_before = 16.0_f32;

    // 30 ticks to give the cloud time to evaporate back to humidity_upper.
    for _ in 0..30 {
        sim.step();
    }

    let cell = sim
        .grid()
        .get(HexCoord::new(0, 0))
        .expect("the cell exists");
    let t_upper = cell.temperature - temp.lapse_rate * 1.5;
    let sat = saturation_upper(t_upper, &atmo);
    let hr = if sat > 0.0 {
        cell.humidity_upper / sat
    } else {
        0.0
    };
    let rain_delta = cell.water_level - water_before;
    eprintln!(
        "cloud_dissipates_when_air_dries after 30 ticks: \
         cloud_water {:.4} → {:.4}, humidity_upper={:.3}, HR={:.2}, \
         water_level delta={:.4}",
        cloud_before, cell.cloud_water, cell.humidity_upper, hr, rain_delta
    );
    assert!(
        cell.cloud_water < cloud_before,
        "Cloud must have decreased via evaporation: {:.3} → {:.3} (expected decreasing). \
         Cloud_water isn't returning to humidity_upper when RH < 0.4.",
        cloud_before,
        cell.cloud_water
    );
}

/// **Test 3**: OBSOLETE since #63 Phase 4 Step 3 (physical CC drain).
///
/// Before: latency came from the progressive vapor→cloud drain (rate=0.04
/// in an RH-fractional formula). The test checked that a supersaturated
/// cell took at least 2 day-ticks before the first rain.
///
/// After: draining is immediate per Pruppacher & Klett (`τ_phase` << 1h).
/// A `Simulation::step()` tick aggregates 24 hourly ticks: even with a
/// sub-millimetric initial surplus, the 24 hours of immediate drain are
/// enough to fill `cloud_water` above the KK2000 threshold. More
/// fundamentally, latency is no longer an emergent property of the 3-stock
/// model, it's a KK2000 attribute (non-linearity + minimum threshold), too
/// dependent on other mechanisms (uplift, orographic, evap) to be testable
/// in single-cell isolation.
///
/// Marked `#[ignore]` rather than deleted: the property "a sub-tick
/// dynamic is needed to go cloud→rain" remains relevant, but will need to
/// be re-tested at another level (targeted knockout on KK2000 scale
/// tests).
#[ignore = "obsolete since physical CC drain #63 Phase 4 Step 3"]
#[test]
fn precipitation_has_latency_after_cloud_forms() {
    // Historical setup kept for reference: the 10 mm supersaturation at
    // 10 °C now produces the first rain at tick 1 (24h of immediate CC
    // drain >> the 0.05 mm "real rain" threshold).
    let mut grid = HexGrid::from_radius(0);
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.elevation = 500.0;
        cell.temperature = 10.0;
        cell.humidity_upper = 10.0;
    }
    // Locks the temperature: without this, the seasonal cycle pushes T
    // negative in winter and precipitation turns to snow.
    let temp_params = TemperatureParams {
        latitude_deg: 0.0,
        thermal_coupling: 0.0,
        ..TemperatureParams::default()
    };
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        temp_params,
        WindParams::default(),
    );

    // We're looking for the tick at which the first significant drop
    // appears. v0.5.x KK2000: the super-linear formula produces
    // infinitesimal drizzle as soon as the cloud appears (no binary
    // threshold). The detection threshold is therefore raised 0.001 →
    // 0.05 mm to capture "real rain" rather than "trace above the noise".
    // The latency is still relevant: the cloud must grow for P_auto ∝
    // q_c^2.47 to produce noticeable drops.
    let mut first_rain_tick: Option<u64> = None;
    for tick in 1..=30 {
        sim.step();
        let cell = sim
            .grid()
            .get(HexCoord::new(0, 0))
            .expect("the cell exists");
        if cell.water_level > 0.05 {
            first_rain_tick = Some(tick);
            break;
        }
    }

    eprintln!("precipitation_has_latency: first drop at tick {first_rain_tick:?}");
    let first_rain = first_rain_tick.expect(
        "No rain in 30 ticks, either cloud_water never crosses the critical \
         threshold, or the vapor is drained another way.",
    );
    assert!(
        first_rain >= 2,
        "Precipitation too fast: rain at tick {first_rain} (expected >= 2). \
         If == 1, it means the vapor → cloud → rain cascade is too direct \
         and cloud_water doesn't have time to act as a reservoir."
    );
}
