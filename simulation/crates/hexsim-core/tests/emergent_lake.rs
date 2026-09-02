//! Critical test for the MFD refactor: a multi-cell lake must emerge
//! **without** global orchestration (no more `identify_basins` or
//! `equalize_basins`).
//!
//! Setup: a closed basin, water injected at the center, no rain and no
//! evaporation. The local MFD physics must spread the water out, make it
//! overflow onto neighbors up to the saddle, and stabilize a flat surface.

use hexsim_core::cell::CellProperties;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::hydro::{HydroParams, step_hydro_mfd};

#[test]
fn lake_emerges_across_several_cells_without_orchestration() {
    // Radius 2: 19 cells. Center in a basin, middle ring at elev=5, outer
    // ring at elev=10 => closed basin.
    let mut grid = HexGrid::from_radius(2);
    let center = HexCoord::new(0, 0);

    for coord in grid.coords().copied().collect::<Vec<_>>() {
        let d = coord.distance(center);
        if let Some(c) = grid.get_mut(coord) {
            *c = CellProperties {
                elevation: match d {
                    0 => 0.0,
                    1 => 5.0,
                    _ => 10.0,
                },
                water_capacity: 1.0,
                ..Default::default()
            };
        }
    }

    // #104 (SI): water is a sheet in mm, the rim of the basin is at 5 m; it
    // takes more than 5000 mm at the center to overflow (the pre-#104 4000 mm
    // only cleared the rim because it counted as "4000 m"). 20,000 mm =>
    // hydrostatic equilibrium surface S = (20 + 6×5)/7 ≈ 7.14 m: center +
    // ring 1 flooded (7 cells).
    grid.get_mut(center).unwrap().water_level = 20_000.0;

    let params = HydroParams {
        flow_rate: 0.15,
        ..HydroParams::default()
    };

    let total_initial: f32 = grid.iter().map(|(_, c)| c.water_level).sum();

    // 5000 steps: the water-against-water leveling transfers `flow_rate ×
    // delta_m` mm per step; the time constant of the center/ring gradient is
    // ~950 steps, and the initial 15 m gap must drop below the flatness
    // tolerance. Still instantaneous at 19 cells.
    let mut current = grid;
    for _ in 0..5000 {
        let mut next = current.clone();
        step_hydro_mfd(&current, &mut next, &params);
        current = next;
    }

    // --- Assertion 1: the lake covers at least 3 cells (center + neighbors) ---
    // A "lake cell" = water that has exceeded the local capacity.
    let lake_cells: Vec<(HexCoord, f32)> = current
        .iter()
        .filter(|(_, c)| c.water_level > c.water_capacity)
        .map(|(coord, c)| (*coord, c.water_level))
        .collect();
    assert!(
        lake_cells.len() >= 3,
        "expected ≥3 lake cells, got {}: {:?}",
        lake_cells.len(),
        lake_cells
    );

    // --- Assertion 2: water surface roughly flat over the lake cells ---
    let surfaces: Vec<f32> = lake_cells
        .iter()
        .map(|(coord, _)| current.get(*coord).unwrap().effective_elevation())
        .collect();
    let smax = surfaces.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let smin = surfaces.iter().copied().fold(f32::INFINITY, f32::min);
    // 0.25 m of spread over a lake ~2 m deep: flat in REAL meters (#104);
    // the old hybrid equilibrium was "flat" at 1 mm of water per meter of
    // elevation change, a surface that hugged the slope.
    assert!(
        smax - smin < 0.25,
        "water surface not flat: spread {:.3} between {smin:.3} and {smax:.3}",
        smax - smin
    );

    // --- Assertion 3: strict conservation ---
    let total_final: f32 = current.iter().map(|(_, c)| c.water_level).sum();
    assert!(
        (total_final - total_initial).abs() < 0.2,
        "conservation violated: {total_initial:.3} → {total_final:.3}"
    );
}
