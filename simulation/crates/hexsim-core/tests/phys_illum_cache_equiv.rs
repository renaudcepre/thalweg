//! Sentinel #65: cached illumination path (`compute_illumination_cached`)
//! reproduces **bit-for-bit** reference march (`compute_illumination`) on
//! real terrain, all hours and seasons, including clouds.
//!
//! This is the contract that enables optimization: "we see shadow very clearly"
//! so shadow must not change by one ulp. If test breaks after illumination
//! physics evolution, reference evolved without its twin: port change to
//! `compute_illumination_cached`.
//!
//! Radius 10 intentional: 64-step march wraps torus multiple times,
//! toric wrap of jump tables stressed, not just flat case.

use hexsim_core::grid::HexGrid;
use hexsim_core::temperature::{
    IllumCache, TemperatureParams, compute_illumination, compute_illumination_cached,
    compute_surface_normals, solar_beam_at_tick,
};
use hexsim_core::terrain::{TerrainParams, generate_terrain};

const CLOUD_ALTITUDE_M: f32 = 1500.0;

fn test_grid(seed: u32) -> HexGrid {
    let mut grid = HexGrid::from_radius(10);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed,
            ..TerrainParams::default()
        },
    );
    compute_surface_normals(&mut grid);
    // Varied synthetic cloud field (deterministic, no cast): "moved cloud shadow"
    // path must read SAME cell as reference.
    let mut x = 0.0_f32;
    for cell in grid.cells_slice_mut() {
        cell.cloud_water = x;
        x = (x + 0.37) % 2.0;
    }
    grid
}

/// Compare two paths over 24h of three days (solstices + equinox).
fn assert_equiv_all_hours(grid: &HexGrid, cache: &IllumCache, params: &TemperatureParams) {
    let (mut ff_ref, mut il_ref) = (Vec::new(), Vec::new());
    let (mut ff_new, mut il_new) = (Vec::new(), Vec::new());
    for day in [354_u64, 80, 172] {
        for hour in 0..24_u64 {
            let tick = day * 24 + hour;
            let beam = solar_beam_at_tick(params, tick);
            compute_illumination(
                grid,
                &beam,
                params.cloud_albedo_coef,
                CLOUD_ALTITUDE_M,
                &mut ff_ref,
                &mut il_ref,
            );
            compute_illumination_cached(
                grid,
                &beam,
                params.cloud_albedo_coef,
                CLOUD_ALTITUDE_M,
                cache,
                &mut ff_new,
                &mut il_new,
            );
            for i in 0..grid.len() {
                assert_eq!(
                    ff_ref[i].to_bits(),
                    ff_new[i].to_bits(),
                    "flux_factor divergent (day {day}, hour {hour}, cell {i}): \
                     ref {} vs cache {}",
                    ff_ref[i],
                    ff_new[i]
                );
                assert_eq!(
                    il_ref[i].to_bits(),
                    il_new[i].to_bits(),
                    "illumination divergent (day {day}, hour {hour}, cell {i}): \
                     ref {} vs cache {}",
                    il_ref[i],
                    il_new[i]
                );
            }
        }
    }
}

#[test]
fn cached_illumination_is_bit_identical_to_reference() {
    let grid = test_grid(42);
    let params = TemperatureParams::default();
    let mut cache = IllumCache::new();
    cache.ensure(&grid);
    assert_equiv_all_hours(&grid, &cache, &params);
}

#[test]
fn cached_illumination_survives_elevation_change_after_mark_dirty() {
    let mut grid = test_grid(7);
    let params = TemperatureParams::default();
    let mut cache = IllumCache::new();
    cache.ensure(&grid);

    // Simulated erosion: carves and deposits, normals recalculated - same
    // sequence as `Simulation` (compute_surface_normals + mark_dirty).
    let mut sign = 1.0_f32;
    for cell in grid.cells_slice_mut() {
        cell.elevation += 40.0 * sign;
        sign = -sign;
    }
    compute_surface_normals(&mut grid);
    cache.mark_dirty();
    cache.ensure(&grid);
    assert_equiv_all_hours(&grid, &cache, &params);
}

#[test]
#[should_panic(expected = "IllumCache stale")]
fn stale_cache_panics_instead_of_shading_wrong() {
    let grid = test_grid(42);
    let params = TemperatureParams::default();
    let mut cache = IllumCache::new();
    cache.ensure(&grid);
    cache.mark_dirty(); // relief "modified", ensure forgotten
    let beam = solar_beam_at_tick(&params, 354 * 24 + 12);
    let (mut ff, mut il) = (Vec::new(), Vec::new());
    compute_illumination_cached(
        &grid,
        &beam,
        params.cloud_albedo_coef,
        CLOUD_ALTITUDE_M,
        &cache,
        &mut ff,
        &mut il,
    );
}
