//! Responses to read commands (`diagnostics`, `cell`, `params`…).
//!
//! All functions in this module are **pure**: `&Simulation` in,
//! [`serde_json::Value`] out, no side effects. That's what makes them
//! usable both from the WebSocket server and from the WASM module, where
//! there is neither a socket nor an async task.
//!
//! The formats produced here are consumed by the frontend (`main.js`),
//! `hexsim-ctl` and `hexsim-mcp`: adding a field is safe, renaming one
//! breaks all three.

use hexsim_core::climate::{Window, default_bands};
use hexsim_core::coord::HexCoord;
use hexsim_core::dynamics::CELL_SPACING_M;
use hexsim_core::simulation::Simulation;
use serde_json::{Value, json};

use crate::command::Query;

/// Identity of the binary serving the simulation, as reported by the
/// `meta` query. The frontend displays it (`v0.7.3 · a1b2c3d`) to spot a
/// forgotten `just rebuild`.
///
/// The shell provides it because it comes from *its own* `build.rs`: the
/// native server and the WASM module are two distinct builds, with two
/// distinct hashes.
#[derive(Debug, Clone)]
pub struct BuildInfo {
    pub version: &'static str,
    pub hash: &'static str,
    pub unix: u64,
}

/// Builds the response to a query. Never mutates the simulation.
#[must_use]
pub fn answer(sim: &Simulation, query: &Query, build: &BuildInfo) -> Value {
    match query {
        Query::Diagnostics => diagnostics(sim),
        Query::Climate => climate(sim),
        Query::Cell { coord } => cell_detail(sim, *coord),
        Query::Region { center, radius } => region_detail(sim, *center, *radius),
        Query::Params => params(sim),
        Query::Meta => meta(sim, build),
    }
}

/// Compact diagnostics + fire metrics (calibration via `just diag`).
#[must_use]
pub fn diagnostics(sim: &Simulation) -> Value {
    let mut val = serde_json::to_value(sim.diagnostics()).unwrap_or_default();
    val["type"] = Value::String("diagnostics".into());
    if let Ok(fire) = serde_json::to_value(sim.fire_stats()) {
        val["fire"] = fire;
    }
    val
}

/// Climate aggregates per elevation band, over three windows.
#[must_use]
pub fn climate(sim: &Simulation) -> Value {
    let bands = default_bands();
    let grid = sim.grid();
    let history = sim.climate_history();
    json!({
        "type": "climate",
        "tick": sim.tick(),
        "last_30": history.aggregate(grid, &bands, Window::Last30),
        "last_180": history.aggregate(grid, &bands, Window::Last180),
        "last_365": history.aggregate(grid, &bands, Window::Last365),
    })
}

/// Build identity (version, git hash, date) + grid constants.
#[must_use]
pub fn meta(sim: &Simulation, build: &BuildInfo) -> Value {
    json!({
        "type": "meta",
        "tick": sim.tick(),
        "version": build.version,
        "build_hash": build.hash,
        "build_unix": build.unix,
        "cell_spacing_m": CELL_SPACING_M,
    })
}

/// Full state of the runtime parameters, one object per module (the 11
/// modules accepted by `update_param`, #2). Response to the `params`
/// query AND payload broadcast after every successful `set_param`: the
/// frontend realigns its sliders on this (single source of truth), never
/// on its static HTML values.
#[must_use]
pub fn params(sim: &Simulation) -> Value {
    // `synoptic.enabled` is a runtime flag outside `SynopticParams`: merged
    // into the object so `synoptic.enabled` reads like the other keys.
    let mut synoptic = serde_json::to_value(sim.synoptic_params()).unwrap_or_default();
    if let Some(obj) = synoptic.as_object_mut() {
        obj.insert("enabled".to_owned(), Value::Bool(sim.synoptic_enabled()));
    }
    json!({
        "type": "params",
        "tick": sim.tick(),
        "atmosphere": sim.atmosphere_params(),
        "hydro": sim.hydro_params(),
        "groundwater": sim.groundwater_params(),
        "snow": sim.snow_params(),
        "temperature": sim.temperature_params(),
        "wind": sim.wind_params(),
        "vegetation": sim.vegetation_params(),
        "erosion": sim.erosion_params(),
        "lake": sim.lake_params(),
        "fire": sim.fire_params(),
        "synoptic": synoptic,
    })
}

/// Detailed record for a cell: its properties, the flux maps at its
/// index, and its 6 neighbors.
#[must_use]
pub fn cell_detail(sim: &Simulation, coord: HexCoord) -> Value {
    let grid = sim.grid();
    let Some(cell) = grid.get(coord) else {
        return json!({
            "type": "cell",
            "tick": sim.tick(),
            "found": false,
            "q": coord.q,
            "r": coord.r,
        });
    };

    let idx = grid.cell_index(coord);
    let outflow_flux = idx
        .and_then(|i| sim.discharge_map().get(i).copied())
        .unwrap_or(0.0);
    let (flow_vec_x, flow_vec_y) = idx
        .and_then(|i| sim.flow_vec_map().get(i).copied())
        .unwrap_or((0.0, 0.0));
    let wind = idx
        .and_then(|i| sim.wind_field().get(i).copied())
        .unwrap_or_default();
    let precip = idx.and_then(|i| sim.last_precipitation().get(i));
    let (rain, snow) = precip.map_or((0.0, 0.0), |p| (p.rain, p.snow));
    let is_raining = rain > 1e-4 || snow > 1e-4;

    let wind_mag = wind.magnitude();
    let wind_dir_deg = wind.direction_deg();

    let neighbors: Vec<Value> = grid
        .neighbors(coord)
        .into_iter()
        .map(|(ncoord, ncell)| {
            json!({
                "q": ncoord.q,
                "r": ncoord.r,
                "elevation": ncell.elevation,
                "water_level": ncell.water_level,
                "water_capacity": ncell.water_capacity,
                "temperature": ncell.temperature,
                "humidity": ncell.humidity_total(),
                "cloud_water": ncell.cloud_water,
            })
        })
        .collect();

    json!({
        "type": "cell",
        "tick": sim.tick(),
        "found": true,
        "q": coord.q,
        "r": coord.r,
        "elevation": cell.elevation,
        "temperature": cell.temperature,
        "water_level": cell.water_level,
        "water_capacity": cell.water_capacity,
        "humidity": cell.humidity_total(),
        "cloud_water": cell.cloud_water,
        "humidity_upper": cell.humidity_upper,
        "humidity_surface": cell.humidity_surface,
        "groundwater": cell.groundwater,
        "snow_level": cell.snow_level,
        "permeability": cell.permeability,
        "outflow_flux": outflow_flux,
        "flow_vec": {"x": flow_vec_x, "y": flow_vec_y},
        "is_raining": is_raining,
        "rain": rain,
        "snow": snow,
        "wind": {
            "x": wind.x,
            "y": wind.y,
            "magnitude": wind_mag,
            "direction_deg": wind_dir_deg,
        },
        "neighbors": neighbors,
    })
}

/// Grid slice centered on `center`, sorted by `(r, q)` for stable
/// reading on the command line.
#[must_use]
pub fn region_detail(sim: &Simulation, center: HexCoord, radius: i32) -> Value {
    let grid = sim.grid();
    let discharge_map = sim.discharge_map();
    let wind_field = sim.wind_field();
    let precip_map = sim.last_precipitation();

    let mut cells: Vec<Value> = grid
        .iter()
        .enumerate()
        .filter(|(_, (coord, _))| center.distance(**coord) <= radius)
        .map(|(i, (coord, props))| {
            let wind = wind_field.get(i).copied().unwrap_or_default();
            let discharge = discharge_map.get(i).copied().unwrap_or(0.0);
            let precip = precip_map.get(i);
            let is_raining = precip.is_some_and(|p| p.rain > 1e-4 || p.snow > 1e-4);
            json!({
                "q": coord.q,
                "r": coord.r,
                "elevation": props.elevation,
                "temperature": props.temperature,
                "water_level": props.water_level,
                "humidity": props.humidity_total(),
                "cloud_water": props.cloud_water,
                "humidity_upper": props.humidity_upper,
                "groundwater": props.groundwater,
                "snow_level": props.snow_level,
                "discharge": discharge,
                "is_raining": is_raining,
                "wind_x": wind.x,
                "wind_y": wind.y,
            })
        })
        .collect();
    cells.sort_by_key(|v| {
        let q = v.get("q").and_then(Value::as_i64).unwrap_or(0);
        let r = v.get("r").and_then(Value::as_i64).unwrap_or(0);
        (r, q)
    });

    json!({
        "type": "region",
        "tick": sim.tick(),
        "center": {"q": center.q, "r": center.r},
        "radius": radius,
        "cell_count": cells.len(),
        "cells": cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::tiny_sim;

    /// #2: the `params` response covers the 11 modules accepted by
    /// `update_param`; a module added to one without the other breaks this
    /// test.
    #[test]
    fn params_json_exposes_the_11_modules() {
        let val = params(&tiny_sim());
        for module in [
            "atmosphere",
            "hydro",
            "groundwater",
            "snow",
            "temperature",
            "wind",
            "vegetation",
            "erosion",
            "lake",
            "fire",
            "synoptic",
        ] {
            assert!(
                val.get(module).is_some_and(Value::is_object),
                "module `{module}` missing from params response"
            );
        }
        assert_eq!(val["type"], "params");
        // Runtime flag merged into the synoptic object (lives outside the struct).
        assert!(val["synoptic"]["enabled"].is_boolean());
    }

    /// #2: an `update_param` is visible in the JSON that follows; that's
    /// the payload the frontend uses to realign its sliders.
    #[test]
    fn params_json_reflete_update_param() {
        let mut sim = tiny_sim();
        assert!(sim.update_param("wind.humidity_advection_rate", 1.25));
        assert!(sim.update_param("fire.enabled", 1.0));
        let val = params(&sim);
        let rate = val["wind"]["humidity_advection_rate"]
            .as_f64()
            .expect("champ wind.humidity_advection_rate");
        assert!((rate - 1.25).abs() < 1e-6);
        assert_eq!(val["fire"]["enabled"], Value::Bool(true));
    }

    /// A coordinate outside the grid responds `found: false` instead of
    /// panicking: the frontend sometimes clicks just outside the domain.
    #[test]
    fn cell_hors_grille_repond_not_found() {
        let val = cell_detail(&tiny_sim(), HexCoord::new(9999, -9999));
        assert_eq!(val["found"], Value::Bool(false));
        assert_eq!(val["type"], "cell");
    }

    /// Radius 0 = the center cell alone. Guard on the distance filter,
    /// which is the only non-trivial logic in `region_detail`.
    #[test]
    fn region_radius_zero_returns_only_the_center() {
        let val = region_detail(&tiny_sim(), HexCoord::new(0, 0), 0);
        assert_eq!(val["cell_count"], 1);
        assert_eq!(val["cells"][0]["q"], 0);
        assert_eq!(val["cells"][0]["r"], 0);
    }
}
