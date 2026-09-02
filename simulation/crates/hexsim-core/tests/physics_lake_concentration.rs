//! Test with two assertions on the hydrological state after a 1-year
//! simulation:
//!
//! 1. **Topology**: at least 1 true visible lake = connected component
//!    of at least 3 contiguous cells with `water_level > 50 mm`. 19
//!    scattered deep cells = 19 puddles, whereas 19 cells
//!    in 3 clusters = 3 visible lakes; only clustering tells them apart.
//!
//! 2. **Dynamics**: there must be at least one day in the year
//!    where **no cell** rains. Otherwise: continuous planetary drizzle,
//!    no high/low pressure cycle. This criterion is carried over from
//!    `scale_universal_invariants`, which observes a `dry streak = 0` over 6
//!    years, a post-PR4 v0.3.0 pathology: `precip_rate` not scaled by /24 and
//!    `CLOUD_MIN_PRECIP=0.05` (a near-zero stop threshold) keep
//!    precipitation going perpetually as soon as a cloud forms.
//!
//! Without criterion 2, the test passes without saying anything about the
//! quality of the overall hydrological cycle: the user sees "it rains all
//! the time", the sim accumulates water in the atmosphere without ever
//! depleting it.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;
use std::collections::{HashSet, VecDeque};

/// Minimum depth for a cell to count as "lake water"
/// (vs puddle/residual humidity).
const LAKE_DEPTH_MM: f32 = 50.0;
/// Minimum size of a connected component for it to be recognized
/// as a true visible lake (vs an isolated cell).
const MIN_CLUSTER_SIZE: usize = 3;
/// Target: at least 1 true lake on the map. Modest, a procedural
/// radius-30 terrain can reasonably have 1 to 4 visible lakes
/// depending on the topography.
const MIN_LAKES: usize = 1;
/// Floor target: at least 1 day in the year where no cell
/// receives **rain** (global high-pressure phase). Measured via
/// `rained()` and not `wet()`: alpine snow falls ~all year on the
/// cold high-altitude band and has nothing to do with a dry lowland
/// phase; counting it masked the ~40 days/year actually without rain that
/// the engine already produces (confounder identified via `bench_metrics`).
/// Guard against a true planetary drizzle (0 rain-free days/year).
const MIN_RAIN_FREE_DAYS: usize = 1;

const RADIUS: i32 = 30;
const YEARS: u64 = 1;
const SEED: u32 = 42;

fn build_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed: SEED,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    let wind = WindParams {
        seed: SEED,
        ..WindParams::default()
    };
    Simulation::new(
        grid,
        HydroParams::default(),
        AtmosphereParams::default(),
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind,
    )
}

/// Connected components of cells with `water_level > threshold`,
/// sorted by decreasing size.
fn find_lake_clusters(grid: &HexGrid, threshold: f32) -> Vec<Vec<HexCoord>> {
    let deep: HashSet<HexCoord> = grid
        .iter()
        .filter_map(|(c, cell)| {
            if cell.water_level > threshold {
                Some(*c)
            } else {
                None
            }
        })
        .collect();

    let mut visited: HashSet<HexCoord> = HashSet::new();
    let mut clusters: Vec<Vec<HexCoord>> = Vec::new();

    for &start in &deep {
        if visited.contains(&start) {
            continue;
        }
        let mut cluster = Vec::new();
        let mut queue = VecDeque::from([start]);
        while let Some(c) = queue.pop_front() {
            if !visited.insert(c) {
                continue;
            }
            cluster.push(c);
            for (n, _) in grid.neighbors(c) {
                if deep.contains(&n) && !visited.contains(&n) {
                    queue.push_back(n);
                }
            }
        }
        clusters.push(cluster);
    }
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    clusters
}

#[test]
fn surface_water_forms_lakes_and_dry_days_exist() {
    let mut sim = build_sim();

    let mut rain_free_days: usize = 0;
    for _ in 0..(YEARS * 365) {
        sim.step();
        // `rained()` (rain only) and not `wet()`: alpine snow doesn't
        // constitute lowland precipitation and would mask the
        // high-pressure phase we're trying to detect.
        let has_any_rain = sim.last_precipitation().iter().any(|r| r.rained());
        if !has_any_rain {
            rain_free_days += 1;
        }
    }

    // ----- Topology: at least one true connected lake -----
    let clusters = find_lake_clusters(sim.grid(), LAKE_DEPTH_MM);
    let lakes: Vec<&Vec<HexCoord>> = clusters
        .iter()
        .filter(|c| c.len() >= MIN_CLUSTER_SIZE)
        .collect();
    let total_deep_cells: usize = clusters.iter().map(Vec::len).sum();
    let cluster_sizes: Vec<usize> = clusters.iter().take(10).map(Vec::len).collect();

    eprintln!(
        "lake_concentration: {} cells > {LAKE_DEPTH_MM} mm, spread over {} components \
         (top 10 sizes: {:?}). {} components >= {MIN_CLUSTER_SIZE} cells = real visible lakes. \
         Global rain-free days = {rain_free_days}/{}.",
        total_deep_cells,
        clusters.len(),
        cluster_sizes,
        lakes.len(),
        YEARS * 365,
    );

    let mut failures: Vec<String> = Vec::new();
    if lakes.len() < MIN_LAKES {
        failures.push(format!(
            "topology: {} lakes (cluster >= {MIN_CLUSTER_SIZE} contiguous cells) found, \
             target >= {MIN_LAKES}. {total_deep_cells} deep cells scattered with no connectivity",
            lakes.len()
        ));
    }
    if rain_free_days < MIN_RAIN_FREE_DAYS {
        failures.push(format!(
            "dynamics: {rain_free_days} rain-free days (target >= {MIN_RAIN_FREE_DAYS}), \
             rain falls somewhere every day of the year, no high-pressure phase"
        ));
    }
    assert!(
        failures.is_empty(),
        "physics_lake_concentration: {} failure(s):\n  - {}",
        failures.len(),
        failures.join("\n  - ")
    );
}
