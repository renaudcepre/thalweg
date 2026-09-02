//! Test d'integration : source karstique via descente piezometrique.
//!
//! Mechanism: a mountain with surface water infiltrates into the
//! water table, the water table flows underground toward a plain below
//! (piezometric = elev + gw), and when the plain's water table exceeds
//! its local capacity, the excess rises back to the surface (resurgence).
//!
//! To isolate the underground path from any direct runoff, we
//! disable `step_hydro_mfd` (`flow_rate = 0`) and cut off all
//! atmospheric inputs (closed terrarium, evap/diffusion/precip=0).
//! What ends up on the plain can therefore only come from the water table.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;

const MOUNTAIN: HexCoord = HexCoord { q: 0, r: 0 };
const PLAIN: HexCoord = HexCoord { q: 1, r: 0 };

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(2);
    for coord in grid.coords().copied().collect::<Vec<_>>() {
        if let Some(cell) = grid.get_mut(coord) {
            if coord == MOUNTAIN {
                cell.elevation = 500.0;
                // Phase 3 (#32) : rescale ×200 (10 → 2000 mm = 2 m).
                cell.water_level = 2000.0;
                cell.permeability = 0.5;
            } else if coord == PLAIN {
                cell.elevation = 0.0;
                cell.water_level = 0.0;
                // Low permeability: capacity = 0.2 * 5 = 1.0. The water
                // table saturates quickly and overflows to the surface.
                cell.permeability = 0.2;
            } else {
                cell.elevation = 1000.0; // walls: no transfer
                cell.water_level = 0.0;
                cell.permeability = 0.0;
            }
            cell.water_capacity = 0.0;
            cell.humidity_upper = 0.0;
            cell.groundwater = 0.0;
            cell.snow_level = 0.0;
            cell.temperature = 10.0;
        }
    }

    // `flow_rate = 0`: no surface MFD transfer. The only way for
    // the plain to receive water is via the water table.
    let hydro = HydroParams {
        flow_rate: 0.0,
        ..HydroParams::default()
    };
    // diffusion=0 + sublim=0 + humidity floor=0: the atmosphere doesn't
    // circulate (the terrarium is closed by default). Meyer evap can fire
    // on cells with water_level > 0, but uplift=0 blocks it from rising,
    // so the vapor stays in humidity_surface without affecting the
    // tested mechanic. transpiration_coef=0: vegetation grows on the
    // plain (rising gw) and its transpiration (#77) drains the water
    // table; a confound for this test, which isolates the ONLY karstic
    // path water table -> resurgence. We neutralize it like
    // flow_rate/uplift/diffusion.
    let atmo = AtmosphereParams {
        sublimation_rate: 0.0,
        uplift_rate: 0.0,
        initial_humidity_floor: 0.0,
        transpiration_coef: 0.0,
        ..AtmosphereParams::default()
    };
    let wind = WindParams {
        thermal_strength: 0.0,
        noise_direction_amplitude: 0.0,
        noise_strength_amplitude: 0.0,
        ..WindParams::default()
    };
    let temp = TemperatureParams {
        latitude_deg: 0.0,
        thermal_coupling: 0.0,
        water_cooling: 0.0,
        ..TemperatureParams::default()
    };

    Simulation::new(
        grid,
        hydro,
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        temp,
        wind,
    )
}

#[test]
fn plain_receives_water_via_groundwater_spring() {
    let mut sim = build_sim();
    let plain_before = sim.grid().get(PLAIN).unwrap().water_level;
    assert!(plain_before < 1e-6, "plain must be dry at the start");

    // 300 ticks: enough to infiltrate the mountain, transfer
    // piezometrically, saturate the plain and make it overflow.
    for _ in 0..300 {
        sim.step();
    }

    let plain = sim.grid().get(PLAIN).unwrap();
    assert!(
        plain.water_level > 0.1,
        "the plain must receive water via the water table: water={:.4} gw={:.4}",
        plain.water_level,
        plain.groundwater
    );
    assert!(
        plain.groundwater > 0.5,
        "the plain's water table must be significantly filled: gw={:.4}",
        plain.groundwater
    );
}

#[test]
fn mountain_aquifer_drains_into_plain() {
    // Transfer invariant: the mountain must LOSE water (surface
    // and gw) to the plain's benefit. Checks the flow direction.
    let mut sim = build_sim();
    let mountain_before = {
        let c = sim.grid().get(MOUNTAIN).unwrap();
        c.water_level + c.groundwater
    };
    let plain_before = {
        let c = sim.grid().get(PLAIN).unwrap();
        c.water_level + c.groundwater
    };

    for _ in 0..300 {
        sim.step();
    }

    let mountain_after = {
        let c = sim.grid().get(MOUNTAIN).unwrap();
        c.water_level + c.groundwater
    };
    let plain_after = {
        let c = sim.grid().get(PLAIN).unwrap();
        c.water_level + c.groundwater
    };

    assert!(
        mountain_after < mountain_before,
        "mountain must lose water: before={mountain_before:.2}, after={mountain_after:.2}"
    );
    assert!(
        plain_after > plain_before,
        "plain must gain water: before={plain_before:.2}, after={plain_after:.2}"
    );
}

#[test]
fn karstic_cycle_conserves_mass() {
    // Closed terrarium + flow_rate=0 + atmosphere disabled: total mass
    // conserved to within epsilon.
    let mut sim = build_sim();
    let total_before: f32 = sim
        .grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum();

    for _ in 0..300 {
        sim.step();
    }

    let total_after: f32 = sim
        .grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum();

    let drift = (total_after - total_before).abs() / total_before.max(1.0);
    assert!(
        drift < 0.01,
        "conservation over karst cycle: {total_before:.2} -> {total_after:.2} \
         (drift {:.3} %)",
        drift * 100.0
    );
}
