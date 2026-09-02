//! Structural diagnostic: heatmap of rain attractors / rainfall deserts.
//!
//! Tool 2 of the `feat/structural-diag-toolkit` toolkit. Targets:
//! - **#24** cell-lake cycle: spot cells that are abnormally waterlogged
//! - **#69** drizzle: visualize the *spatial distribution* of drizzle (which
//!   cells rain all the time? which never rain?)
//!
//! For each cell, over N years at daily resolution:
//! - `rain_per_year` (days/year with rain > 0)
//! - `wet_streak_max` (longest consecutive run of rainy days)
//! - `dry_streak_max` (longest consecutive run of dry days)
//! - `time_in_cluster_pct` (% of days with `cloud_water > threshold`)
//!
//! Output: aggregates by elevation band + E/W x N/S quadrant, plus top/bottom
//! 10 per metric with coordinates and elevation. No ASCII heatmap: unreadable
//! for a human at radius 30 and useless for an LLM, which reasons better over
//! numeric tables.
//!
//! **Eval style** (`scale_tests_eval_style`): `#[ignore]`, `eprintln!`, no
//! assert. Read, form hypotheses, act.
//!
//! Run with:
//! ```text
//! cargo test --release -p hexsim-core --test diag_rain_attractors \
//!     -- --ignored --nocapture
//! ```

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

use std::fmt::Write as _;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const WARMUP_DAYS: u64 = 90;
const YEARS: u64 = 2;
const TOTAL_DAYS: u64 = YEARS * 365;

const CLOUD_THRESHOLD_MM: f32 = 0.05;
const TOP_N: usize = 10;

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

struct CellMetrics {
    rain_days: Vec<u32>,
    max_wet_streak: Vec<u32>,
    max_dry_streak: Vec<u32>,
    in_cluster_days: Vec<u32>,
}

fn collect_metrics(sim: &mut Simulation) -> CellMetrics {
    let n = sim.grid().len();
    let mut rain_days = vec![0_u32; n];
    let mut wet_streak = vec![0_u32; n];
    let mut max_wet_streak = vec![0_u32; n];
    let mut dry_streak = vec![0_u32; n];
    let mut max_dry_streak = vec![0_u32; n];
    let mut in_cluster_days = vec![0_u32; n];

    for _ in 0..TOTAL_DAYS {
        sim.step();
        let precip = sim.last_precipitation();
        let cells = sim.grid().cells_slice();
        for i in 0..n {
            if precip[i].rain > 0.0 {
                rain_days[i] += 1;
                wet_streak[i] += 1;
                dry_streak[i] = 0;
                if wet_streak[i] > max_wet_streak[i] {
                    max_wet_streak[i] = wet_streak[i];
                }
            } else {
                dry_streak[i] += 1;
                wet_streak[i] = 0;
                if dry_streak[i] > max_dry_streak[i] {
                    max_dry_streak[i] = dry_streak[i];
                }
            }
            if cells[i].cloud_water > CLOUD_THRESHOLD_MM {
                in_cluster_days[i] += 1;
            }
        }
    }

    CellMetrics {
        rain_days,
        max_wet_streak,
        max_dry_streak,
        in_cluster_days,
    }
}

fn rain_per_year(rain_days: u32) -> f32 {
    let r = f32::from(u16::try_from(rain_days).unwrap_or(u16::MAX));
    let y = f32::from(u16::try_from(YEARS).unwrap_or(u16::MAX));
    if y > 0.0 { r / y } else { 0.0 }
}

fn pct_of_total(count: u32) -> f32 {
    let c = f32::from(u16::try_from(count).unwrap_or(u16::MAX));
    let t = f32::from(u16::try_from(TOTAL_DAYS).unwrap_or(u16::MAX));
    if t > 0.0 { 100.0 * c / t } else { 0.0 }
}

fn print_top_bottom(
    label: &str,
    values: &[f32],
    coords: &[HexCoord],
    elevations: &[f32],
    unit: &str,
) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("\n  -- Top {TOP_N} ({label}) --");
    eprintln!(
        "    {:<5} {:<14} {:>8} {:>10}",
        "rank", "coord(q,r)", "elev_m", unit
    );
    for (rank, &idx) in order.iter().take(TOP_N).enumerate() {
        eprintln!(
            "    {:<5} ({:>4},{:>4})    {:>8.0} {:>10.2}",
            rank + 1,
            coords[idx].q,
            coords[idx].r,
            elevations[idx],
            values[idx]
        );
    }
    eprintln!("\n  -- Bottom {TOP_N} ({label}) --");
    eprintln!(
        "    {:<5} {:<14} {:>8} {:>10}",
        "rank", "coord(q,r)", "elev_m", unit
    );
    for (rank, &idx) in order.iter().rev().take(TOP_N).enumerate() {
        eprintln!(
            "    {:<5} ({:>4},{:>4})    {:>8.0} {:>10.2}",
            rank + 1,
            coords[idx].q,
            coords[idx].r,
            elevations[idx],
            values[idx]
        );
    }
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_rain_attractors() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    let n_cells = sim.grid().len();
    let coords: Vec<HexCoord> = sim.grid().coords_slice().to_vec();
    let elevations: Vec<f32> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.elevation)
        .collect();

    eprintln!(
        "\n=== Rain attractor diag (#24/#69), radius {RADIUS}, seed {SEED}, {YEARS} years ({TOTAL_DAYS} days) ==="
    );
    eprintln!("  warmup = {WARMUP_DAYS} days, n_cells = {n_cells}");
    eprintln!("  cloud_water threshold = {CLOUD_THRESHOLD_MM:.3} mm");

    let m = collect_metrics(&mut sim);

    let rain_per_year_vec: Vec<f32> = m.rain_days.iter().map(|&d| rain_per_year(d)).collect();
    let dry_streak_vec: Vec<f32> = m
        .max_dry_streak
        .iter()
        .map(|&d| f32::from(u16::try_from(d).unwrap_or(u16::MAX)))
        .collect();
    let wet_streak_vec: Vec<f32> = m
        .max_wet_streak
        .iter()
        .map(|&d| f32::from(u16::try_from(d).unwrap_or(u16::MAX)))
        .collect();
    let cluster_pct_vec: Vec<f32> = m.in_cluster_days.iter().map(|&d| pct_of_total(d)).collect();

    let metrics: [(&str, &[f32], &str); 4] = [
        ("rain/an", &rain_per_year_vec, "j/an"),
        ("dry_streak_max", &dry_streak_vec, "days"),
        ("wet_streak_max", &wet_streak_vec, "days"),
        ("time_in_cluster_pct", &cluster_pct_vec, "%"),
    ];

    print_band_table(&metrics, &coords, &elevations);
    print_quadrant_table(&metrics, &coords);
    print_ring_table(&metrics, &coords);
    print_ring_profile(&rain_per_year_vec, &coords, HexCoord::new(0, 0));
    print_ring_profile(&rain_per_year_vec, &coords, HexCoord::new(20, -10));

    for (label, values, unit) in &metrics {
        print_top_bottom(label, values, &coords, &elevations, unit);
    }
}

/// Elevation bands aligned with `scale_drizzle_*` so they can be compared.
const BANDS: &[(&str, f32, f32)] = &[
    ("<0m", f32::NEG_INFINITY, 0.0),
    ("0-300m", 0.0, 300.0),
    ("300-800m", 300.0, 800.0),
    ("800-1500m", 800.0, 1500.0),
    (">1500m", 1500.0, f32::INFINITY),
];

fn band_index(elev: f32) -> Option<usize> {
    BANDS
        .iter()
        .position(|(_, lo, hi)| elev >= *lo && elev < *hi)
}

fn mean_p50_p99(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let p50 = sorted[(n - 1) / 2];
    let p99 = sorted[((n - 1) * 99) / 100];
    let n_f = f32::from(u16::try_from(n).unwrap_or(u16::MAX));
    let mean: f32 = sorted.iter().sum::<f32>() / n_f;
    (mean, p50, p99)
}

fn print_band_table(metrics: &[(&str, &[f32], &str)], coords: &[HexCoord], elevations: &[f32]) {
    eprintln!("\n== Distribution by elevation band (mean / p50 / p99) ==");
    eprintln!(
        "  {:<24} {:>20} {:>20} {:>20} {:>20} {:>20}",
        "metric", BANDS[0].0, BANDS[1].0, BANDS[2].0, BANDS[3].0, BANDS[4].0
    );
    let _ = coords; // marqueur d'indexation
    let mut band_indices: [Vec<usize>; 5] = Default::default();
    for (i, &elev) in elevations.iter().enumerate() {
        if let Some(b) = band_index(elev) {
            band_indices[b].push(i);
        }
    }
    eprintln!(
        "  {:<24} {:>20} {:>20} {:>20} {:>20} {:>20}",
        "n cells",
        band_indices[0].len(),
        band_indices[1].len(),
        band_indices[2].len(),
        band_indices[3].len(),
        band_indices[4].len(),
    );
    for (label, values, unit) in metrics {
        let mut row = format!("  {:<24}", format!("{label} ({unit})"));
        for indices in &band_indices {
            let sub: Vec<f32> = indices.iter().map(|&i| values[i]).collect();
            let (m, p50, p99) = mean_p50_p99(&sub);
            let _ = write!(row, " {m:>5.1}/{p50:>5.1}/{p99:>6.1}    ");
        }
        eprintln!("{row}");
    }
}

/// Axial quadrant: `q < 0` ⇒ west (W), `q > 0` ⇒ east (E); `r < 0` ⇒ north
/// (N), `r > 0` ⇒ south (S). Cells on the axes (q == 0 or r == 0) discarded.
/// Lets us spot asymmetries induced by the dominant westerly wind and by
/// latitude (solar temperature).
fn quadrant(coord: HexCoord) -> Option<usize> {
    match (coord.q.signum(), coord.r.signum()) {
        (-1, -1) => Some(0), // NW
        (1, -1) => Some(1),  // NE
        (-1, 1) => Some(2),  // SW
        (1, 1) => Some(3),   // SE
        _ => None,
    }
}

/// Distance-from-center bands, by hex ring: grouped interior vs. the last 3
/// rings, each isolated. Target: the edge artifact (journal, 2026-07-03),
/// permanent rain pinned on ring d = R by the non-toric wrap. If columns
/// R-1/R diverge from the interior, it's the seam that's raining, not the
/// terrain.
fn print_ring_table(metrics: &[(&str, &[f32], &str)], coords: &[HexCoord]) {
    let labels = [
        format!("d<={} (int.)", RADIUS - 3),
        format!("d={}", RADIUS - 2),
        format!("d={}", RADIUS - 1),
        format!("d={RADIUS} (edge)"),
    ];
    eprintln!("\n== Distribution by ring distance from center (mean / p50 / p99) ==");
    eprintln!(
        "  {:<24} {:>20} {:>16} {:>16} {:>16}",
        "metric", labels[0], labels[1], labels[2], labels[3]
    );
    let center = HexCoord::new(0, 0);
    let mut ring_indices: [Vec<usize>; 4] = Default::default();
    for (i, c) in coords.iter().enumerate() {
        let d = c.distance(center);
        let slot = match RADIUS - d {
            0 => 3,
            1 => 2,
            2 => 1,
            _ => 0,
        };
        ring_indices[slot].push(i);
    }
    eprintln!(
        "  {:<24} {:>20} {:>16} {:>16} {:>16}",
        "n cells",
        ring_indices[0].len(),
        ring_indices[1].len(),
        ring_indices[2].len(),
        ring_indices[3].len(),
    );
    for (label, values, unit) in metrics {
        let mut row = format!("  {:<24}", format!("{label} ({unit})"));
        for indices in &ring_indices {
            let sub: Vec<f32> = indices.iter().map(|&i| values[i]).collect();
            let (m, p50, p99) = mean_p50_p99(&sub);
            let _ = write!(row, " {m:>4.1}/{p50:>4.1}/{p99:>5.1} ");
        }
        eprintln!("{row}");
    }
}

/// Toric distance: min of the hexagonal distance over the 7 representatives
/// (identity + the 6 lattice translations). On the torus, the "center" of
/// the map is an arbitrary chart choice; comparing profiles from two
/// different centers distinguishes an edge artifact (the profile follows the
/// map) from a real weather structure (the profile follows the geography).
fn toric_distance(c: HexCoord, center: HexCoord, radius: i32) -> i32 {
    let d = c - center;
    let mut best = d.distance(HexCoord::new(0, 0));
    for v in hexsim_core::coord::torus_lattice_vectors(radius) {
        best = best.min((d - v).distance(HexCoord::new(0, 0)));
        best = best.min((d + v).distance(HexCoord::new(0, 0)));
    }
    best
}

/// Compact profile: average rain by toric distance from the given center.
fn print_ring_profile(rain_per_year: &[f32], coords: &[HexCoord], center: HexCoord) {
    let mut sums: Vec<(f32, u32)> = vec![(0.0, 0); usize::try_from(RADIUS).unwrap() + 1];
    for (i, c) in coords.iter().enumerate() {
        let d = usize::try_from(toric_distance(*c, center, RADIUS)).unwrap_or(0);
        if let Some(slot) = sums.get_mut(d) {
            slot.0 += rain_per_year[i];
            slot.1 += 1;
        }
    }
    eprintln!(
        "\n== Rain/year profile by toric distance from center ({}, {}) ==",
        center.q, center.r
    );
    let mut row = String::new();
    for (d, (sum, n)) in sums.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        #[expect(clippy::cast_precision_loss)]
        let mean = sum / *n as f32;
        let _ = write!(row, "d{d}:{mean:.0} ");
    }
    eprintln!("  {row}");
}

fn print_quadrant_table(metrics: &[(&str, &[f32], &str)], coords: &[HexCoord]) {
    eprintln!("\n== Distribution by axial quadrant (mean / p50 / p99) ==");
    eprintln!("  W = q<0 (leeward to the east? west? cf wind), E = q>0  |  N = r<0, S = r>0");
    eprintln!(
        "  {:<24} {:>16} {:>16} {:>16} {:>16}",
        "metric", "NW", "NE", "SW", "SE"
    );
    let mut quad_indices: [Vec<usize>; 4] = Default::default();
    for (i, c) in coords.iter().enumerate() {
        if let Some(q) = quadrant(*c) {
            quad_indices[q].push(i);
        }
    }
    eprintln!(
        "  {:<24} {:>16} {:>16} {:>16} {:>16}",
        "n cells",
        quad_indices[0].len(),
        quad_indices[1].len(),
        quad_indices[2].len(),
        quad_indices[3].len(),
    );
    for (label, values, unit) in metrics {
        let mut row = format!("  {:<24}", format!("{label} ({unit})"));
        for indices in &quad_indices {
            let sub: Vec<f32> = indices.iter().map(|&i| values[i]).collect();
            let (m, p50, p99) = mean_p50_p99(&sub);
            let _ = write!(row, " {m:>4.1}/{p50:>4.1}/{p99:>5.1} ");
        }
        eprintln!("{row}");
    }
}
