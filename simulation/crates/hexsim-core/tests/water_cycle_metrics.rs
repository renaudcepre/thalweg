use hexsim_core::atmosphere::{
    AtmoForcing, AtmosphereParams, smooth_upper_air_mean_t, step_atmosphere, surface_means,
    total_humidity,
};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::{GroundwaterParams, step_groundwater, total_groundwater};
use hexsim_core::hydro::{HydroParams, step_hydro_mfd, total_water};
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::{WindParams, compute_wind_field, compute_wind_magnitudes_into};

#[test]
fn water_cycle_conservation() {
    let mut current = HexGrid::from_radius(12);
    generate_terrain(&mut current, &TerrainParams::default());
    let mut next = current.clone();

    let atmo = AtmosphereParams::default();
    let temp = TemperatureParams::default();
    let mut precip_gate_open = false;
    let gw = GroundwaterParams::default();
    // Defaults : terrarium strictement ferme (conservation stricte garantie).
    let hydro = HydroParams::default();
    let wp = WindParams::default();

    let initial = total_water(&current) + total_humidity(&current) + total_groundwater(&current);

    println!("tick,surface,humidity,groundwater,total,drift_pct");
    println!(
        "0,{:.2},{:.2},{:.2},{:.2},0.0000",
        total_water(&current),
        total_humidity(&current),
        total_groundwater(&current),
        initial
    );

    let mut wind_mag = Vec::new();
    // Upper-air anchor (EMA τ = 24 h), same bookkeeping as `Simulation`.
    let mut upper_air_mean_t = surface_means(&current).0;
    for tick in 1..=500_u64 {
        let wf = compute_wind_field(&current, &wp, tick);
        compute_wind_magnitudes_into(&wf, &mut wind_mag);
        upper_air_mean_t = smooth_upper_air_mean_t(upper_air_mean_t, surface_means(&current).0);
        step_atmosphere(
            &current,
            &mut next,
            &atmo,
            &AtmoForcing {
                temp_params: &temp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wind_mag,
                hour_tick: tick,
                upper_air_mean_t,
            },
            &mut precip_gate_open,
        );
        std::mem::swap(&mut current, &mut next);

        step_groundwater(&current, &mut next, &gw);
        std::mem::swap(&mut current, &mut next);

        for _ in 0..8 {
            step_hydro_mfd(&current, &mut next, &hydro);
            std::mem::swap(&mut current, &mut next);
        }

        let surface = total_water(&current);
        let humidity = total_humidity(&current);
        let groundwater = total_groundwater(&current);
        let total = surface + humidity + groundwater;
        let drift_pct = ((total - initial) / initial) * 100.0;

        if tick % 10 == 0 {
            println!(
                "{tick},{surface:.2},{humidity:.2},{groundwater:.2},{total:.2},{drift_pct:.4}"
            );
        }
    }

    let final_total =
        total_water(&current) + total_humidity(&current) + total_groundwater(&current);
    let drift = ((final_total - initial) / initial).abs();
    assert!(
        drift < 0.01,
        "Drift > 1% after 500 ticks: {drift:.2}% ({initial:.1} → {final_total:.1})"
    );
}
