//! Micro-test (e2e-unit, radius 0, ms): **a frozen lake is still a lake**.
//!
//! Seen in the front on 2026-09-05: trees growing on a lake within a single
//! winter. Freezing moves the lake's liquid surplus into the frozen stock
//! (`snow::step_snow`), so the liquid surplus alone no longer says "open
//! water" and `vegetation::step_vegetation` starts colonizing the ice at
//! `colonization_rate` (0.01/day): 0.05 of cover in ~10 days, one tree in
//! the front.
//!
//! The identity of a water body must follow the water, not its phase. Two
//! cell-local phenomena only (freeze/melt, growth), so radius 0 is
//! legitimate (no transport involved, cf. the micro-test rule "no
//! transport at radius 0").

use hexsim_core::cell::CellProperties;
use hexsim_core::climate_normals::CellClimateNormals;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::snow::{SnowForcing, SnowParams, step_snow};
use hexsim_core::time::TICKS_PER_DAY;
use hexsim_core::vegetation::{VegetationParams, canopy_cover, is_open_water, step_vegetation};

const WINTER_DAYS: u64 = 90;
const WINTER_AIR_C: f32 = -10.0;
const CENTER: HexCoord = HexCoord { q: 0, r: 0 };

/// Subalpine normals where fir and alpine grass thrive: the niche is
/// good, so the ONLY thing that can keep vegetation at zero is the
/// open-water identity of the cell.
fn favorable_normals() -> CellClimateNormals {
    CellClimateNormals {
        t_mean: 6.0,
        t_min: -15.0,
        t_max: 18.0,
        moisture_mean: 10.0,
        moisture_min: 3.0,
        moisture_max: 40.0,
        insolation_mean: 150.0,
    }
}

/// One-cell grid holding `surplus_mm` of free water above capacity, bare,
/// in winter air, no soil exchange (permeability 0).
fn water_cell(surplus_mm: f32) -> HexGrid {
    let mut grid = HexGrid::from_radius(0);
    let cell = grid.get_mut(CENTER).unwrap();
    cell.water_capacity = 1.0;
    cell.water_level = cell.water_capacity + surplus_mm;
    cell.temperature = WINTER_AIR_C;
    cell.permeability = 0.0;
    cell.groundwater = 0.0;
    cell.snow_level = 0.0;
    grid
}

/// Every stock a freeze/melt/growth loop can move water into.
fn frozen_plus_liquid(c: &CellProperties) -> f32 {
    c.water_level + c.frozen_surface() + c.groundwater
}

/// `days` of winter: 24 hourly freeze steps under a calm clear night,
/// then the daily vegetation step, air pinned at `WINTER_AIR_C`.
fn winter(mut grid: HexGrid, days: u64) -> HexGrid {
    let snow_params = SnowParams::default();
    let veg_params = VegetationParams::default();
    let normals = vec![favorable_normals()];
    let mut next = grid.clone();
    for _ in 0..days {
        for _ in 0..TICKS_PER_DAY {
            grid.get_mut(CENTER).unwrap().temperature = WINTER_AIR_C;
            step_snow(&grid, &mut next, &snow_params, &SnowForcing::night_calm());
            std::mem::swap(&mut grid, &mut next);
        }
        step_vegetation(&grid, &mut next, &veg_params, &normals);
        std::mem::swap(&mut grid, &mut next);
    }
    grid
}

#[test]
fn frozen_lake_grows_no_vegetation() {
    let excess = VegetationParams::default().open_water_excess;
    let start = water_cell(60.0);
    let budget_before = frozen_plus_liquid(start.get(CENTER).unwrap());
    assert!(
        is_open_water(start.get(CENTER).unwrap()),
        "setup: a 60 mm surplus is a lake"
    );

    let end = winter(start, WINTER_DAYS);
    let c = end.get(CENTER).unwrap();

    // The discriminating condition: the LIQUID surplus did fall below the
    // open-water threshold, i.e. the lake really froze. Without this the
    // test would pass for the wrong reason (a lake that never froze).
    let liquid_surplus = c.water_level - c.water_capacity;
    assert!(
        liquid_surplus < excess,
        "setup: the lake must freeze within {WINTER_DAYS} days at {WINTER_AIR_C} °C, \
         liquid surplus still {liquid_surplus:.2} mm (threshold {excess} mm)"
    );

    assert!(
        is_open_water(c),
        "a frozen lake is still open water: water={:.2} ice={:.2} snow={:.2} capacity={:.2}",
        c.water_level,
        c.ice_level,
        c.snow_level,
        c.water_capacity
    );
    assert!(
        c.ice_level > 50.0,
        "the lake's surplus is lake ice, not snowpack: ice={:.2} snow={:.2}",
        c.ice_level,
        c.snow_level
    );
    // Canopy cover is exactly 0 only when every species' biomass is 0
    // (non-negative biomasses, `1 − Π(1 − cover_S)`), so this is "nothing
    // grew in any stratum", not a sum that could hide a layer.
    let veg = canopy_cover(c);
    assert!(
        veg == 0.0,
        "no colonization on a frozen lake, got canopy cover {veg:.4} after {WINTER_DAYS} days"
    );

    let budget_after = frozen_plus_liquid(c);
    assert!(
        (budget_after - budget_before).abs() < 1e-3,
        "freeze must conserve water: {budget_before:.4} → {budget_after:.4}"
    );
}

/// Control: the same winter on a cell that is NOT a lake (a 1 mm film,
/// under the open-water threshold) must colonize. Proves the protocol
/// itself grows vegetation when the identity allows it, so the zero
/// above is the identity at work, not a dead loop.
#[test]
fn frozen_film_on_land_gets_colonized() {
    let start = water_cell(1.0);
    assert!(
        !is_open_water(start.get(CENTER).unwrap()),
        "setup: a 1 mm film is land"
    );

    let end = winter(start, WINTER_DAYS);
    let c = end.get(CENTER).unwrap();
    let veg = canopy_cover(c);
    assert!(
        veg > 0.05,
        "land under a favorable niche must colonize within {WINTER_DAYS} days, got {veg:.4}"
    );
}
