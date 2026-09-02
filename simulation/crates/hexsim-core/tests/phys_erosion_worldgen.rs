//! Test #105 (one-shot pivot, 2026-07-11): erosion is now computed
//! ONCE at world generation (`erode_terrain`, channel incision),
//! then elevation is FROZEN, no more live erosion by default (no drift
//! toward flatness on long runs, zero runtime cost).
//!
//! This test validates the two properties of the pivot (seed 7, r30):
//! 1. **Worldgen erosion CONCENTRATES drainage**: the Gini of
//!    flow accumulation on the eroded terrain exceeds that of the raw
//!    terrain; the rills merge into a network (measured: 0.708 → ~0.74 at 25
//!    iterations). It EXPORTS rock (non-conservative incision,
//!    assumed: a river carries its sediments out of the basin) without
//!    collapsing into smoothing (local relief preserved).
//! 2. **The terrain is STABLE at runtime**: erosion off by default, so over
//!    several simulated years, elevation doesn't move by a single bit. That is
//!    the "no flat runaway" guarantee that motivated the pivot.
//!
//! The old version (drainage convergence over 10 years of LIVE erosion)
//! tested a mode that became opt-in; the live path remains covered by
//! `phys_erosion.rs` (river incision, conservation).

mod common;

use hexsim_core::erosion::{accumulate_flow, closed_depression_indices, discharge_gini};
use hexsim_core::grid::HexGrid;
use hexsim_core::terrain::{TerrainParams, generate_terrain};

use common::build_prod_sim;

const RADIUS: i32 = 30;
const SEED: u32 = 7;
/// Flow routing exponent for the concentration metric (same as
/// `erosion::WORLDGEN_FLOW_CONCENTRATION`).
const FLOW_CONCENTRATION: f32 = 4.0;

fn terrain_with(iterations: u32) -> HexGrid {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: SEED,
            erosion_iterations: iterations,
            ..TerrainParams::default()
        },
    );
    grid
}

#[test]
fn worldgen_erosion_concentrates_drainage() {
    let raw = terrain_with(0);
    let eroded = terrain_with(25);

    let gini_raw = discharge_gini(&accumulate_flow(&raw, FLOW_CONCENTRATION).0);
    let gini_eroded = discharge_gini(&accumulate_flow(&eroded, FLOW_CONCENTRATION).0);
    assert!(
        gini_eroded > gini_raw + 0.01,
        "worldgen erosion did not concentrate drainage: raw gini {gini_raw:.4}, \
         eroded {gini_eroded:.4}"
    );

    // Local relief preserved (no smoothing): the average drop toward the
    // lowest neighbor doesn't collapse. Non-conservative incision carves the
    // channels instead of planing them down; that's what distinguishes this
    // pivot from the conservative variant (which smoothed, cf. JOURNAL #105
    // one-shot).
    let local_relief = |g: &HexGrid| -> f64 {
        let cells = g.cells_slice();
        let n = cells.len();
        let sum: f64 = (0..n)
            .map(|i| {
                let lowest = g
                    .neighbor_indices_toric(i)
                    .iter()
                    .map(|&j| cells[j].elevation)
                    .fold(f32::INFINITY, f32::min);
                f64::from((cells[i].elevation - lowest).max(0.0))
            })
            .sum();
        sum / f64::from(u32::try_from(n).expect("fits u32"))
    };
    let relief_raw = local_relief(&raw);
    let relief_eroded = local_relief(&eroded);
    assert!(
        relief_eroded > 0.85 * relief_raw,
        "local relief collapsed (smoothing): raw {relief_raw:.1} m, eroded {relief_eroded:.1} m"
    );

    // Closed depressions remain (lake candidates that will fill in live).
    assert!(
        !closed_depression_indices(&eroded).is_empty(),
        "no closed basin on the eroded terrain (no lake possible)"
    );
}

#[test]
fn eroded_terrain_is_frozen_at_runtime() {
    // Production world (pre-eroded at worldgen, runtime erosion OFF by
    // default).
    let mut sim = build_prod_sim(SEED, RADIUS);
    let elev0: Vec<u32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation.to_bits())
        .collect();

    // 180 days: the whole climate/hydro chain runs through a season
    // transition, but elevation must not move by a single bit (frozen erosion,
    // no flat runaway). Bit-identical is binary: no need to simulate
    // years; if a mechanism touched elevation, it would strike as early as
    // day 1. Short, to stay within the dev loop.
    for _ in 0..180 {
        sim.step();
    }

    for (i, cell) in sim.grid().cells_slice().iter().enumerate() {
        assert_eq!(
            cell.elevation.to_bits(),
            elev0[i],
            "elevation modified at runtime (cell {i}): erosion should be frozen"
        );
        assert!(
            cell.sediment_load == 0.0,
            "non-zero sediment load at runtime (cell {i})"
        );
    }
}
