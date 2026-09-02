//! Scale tests, universal invariants (seed-agnostic).
//!
//! Philosophy: these tests verify physical properties that must hold
//! *regardless of the generation seed*. They run on 3 seeds in
//! parallel and assert on each one. A failure signals a bug in the
//! simulator, not a particular geography.
//!
//! The assertions are designed like an eval: some can fail, we let them
//! fail visibly and iterate.

mod common;

use std::collections::HashMap;

use common::{PerfTimer, pct};
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

const RADIUS: i32 = 20;
const SEEDS: [u32; 3] = [42, 1337, 7];
const WARMUP_TICKS: u64 = 365;
const MEASURE_TICKS: u64 = 365;
const TOTAL_TICKS: u64 = WARMUP_TICKS + MEASURE_TICKS;
/// Tolerance for negative values due to floating-point rounding in step functions.
const NEG_EPS: f32 = -1e-4;
/// Lake permanence threshold : 80 % of `MEASURE_TICKS` (= 365 * 80 / 100 = 292).
const LAKE_PERM_TICKS: u32 = 292;

/// Per-seed accumulators: aggregate everything during the sim to assert only at the end.
struct SeedStats {
    seed: u32,
    cell_count: usize,
    water_initial: f32,
    water_final: f32,
    nan_inf_events: Vec<String>,
    bounds_events: Vec<String>,
    /// Rain days per cell (rain > 1e-4), over the measurement window.
    rain_days: HashMap<HexCoord, u32>,
    /// Number of ticks where the cell is in a lake basin, measurement window.
    lake_ticks: HashMap<HexCoord, u32>,
    /// Frozen elevations (terrain doesn't move).
    elevations: HashMap<HexCoord, f32>,
    /// Min/max temperature observed per cell over the measurement window
    /// (to validate the effective seasonal amplitude).
    temp_min: HashMap<HexCoord, f32>,
    temp_max: HashMap<HexCoord, f32>,
    /// Largest discharge observed over the window (a sim with climate
    /// must produce at least one visible river).
    max_discharge_seen: f32,
    /// Longest run of consecutive ticks with no global rain (measurement
    /// window). Global rain = at least one cell with rain > 1e-4.
    dry_streak_max: u32,
    dry_streak_cur: u32,
    /// Global wind rose: number of observations (cell × tick) per
    /// 60° sector. Index 0 = [0, 60°), 1 = [60, 120°), etc. Angle computed
    /// via atan2(y, x), wrapped into [0, 360°).
    wind_sector_counts: [u64; 6],
    /// Sum of |wind| and the max observed, to verify there is indeed
    /// circulation. Counted separately since it accumulates (cell × tick).
    wind_magnitude_sum: f64,
    wind_magnitude_samples: u64,
    wind_magnitude_max: f32,
    /// Total snowpack (sum over the grid) min and max over the measurement
    /// window. A real annual cycle must vary this total by a factor > 2.
    snow_total_min: f32,
    snow_total_max: f32,
}

fn build_sim(seed: u32) -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let terrain = TerrainParams {
        seed,
        ..TerrainParams::default()
    };
    generate_terrain(&mut grid, &terrain);
    let wind = WindParams {
        seed,
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

fn total_water(sim: &Simulation) -> f32 {
    sim.grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum()
}

fn accumulate_measuring_tick(sim: &Simulation, stats: &mut SeedStats) {
    let coords = sim.grid().coords_slice();
    let mut any_rain = false;
    for (i, rec) in sim.last_precipitation().iter().enumerate() {
        if rec.rain > 1e-4 {
            *stats.rain_days.entry(coords[i]).or_default() += 1;
            any_rain = true;
        }
    }
    if any_rain {
        stats.dry_streak_cur = 0;
    } else {
        stats.dry_streak_cur += 1;
        if stats.dry_streak_cur > stats.dry_streak_max {
            stats.dry_streak_max = stats.dry_streak_cur;
        }
    }
    for w in sim.wind_field() {
        let mag = w.magnitude();
        stats.wind_magnitude_sum += f64::from(mag);
        stats.wind_magnitude_samples += 1;
        if mag > stats.wind_magnitude_max {
            stats.wind_magnitude_max = mag;
        }
        if mag > 1e-4 {
            let angle_deg = w.direction_deg();
            let sector: usize = if angle_deg < 60.0 {
                0
            } else if angle_deg < 120.0 {
                1
            } else if angle_deg < 180.0 {
                2
            } else if angle_deg < 240.0 {
                3
            } else if angle_deg < 300.0 {
                4
            } else {
                5
            };
            stats.wind_sector_counts[sector] += 1;
        }
    }
    let snow_total: f32 = sim.grid().iter().map(|(_, c)| c.snow_level).sum();
    if snow_total < stats.snow_total_min {
        stats.snow_total_min = snow_total;
    }
    if snow_total > stats.snow_total_max {
        stats.snow_total_max = snow_total;
    }
    for (coord, cell) in sim.grid().iter() {
        if cell.water_level > 0.1 * cell.water_capacity {
            *stats.lake_ticks.entry(*coord).or_default() += 1;
        }
        stats
            .temp_min
            .entry(*coord)
            .and_modify(|t| *t = t.min(cell.temperature))
            .or_insert(cell.temperature);
        stats
            .temp_max
            .entry(*coord)
            .and_modify(|t| *t = t.max(cell.temperature))
            .or_insert(cell.temperature);
    }
    for &d in sim.discharge_map() {
        if d > stats.max_discharge_seen {
            stats.max_discharge_seen = d;
        }
    }
}

fn run_one_seed(seed: u32, timer: &mut PerfTimer) -> SeedStats {
    let mut sim = build_sim(seed);
    timer.lap(&format!("setup seed {seed}"));

    let cell_count = sim.grid().len();
    let elevations: HashMap<HexCoord, f32> = sim
        .grid()
        .iter()
        .map(|(c, cell)| (*c, cell.elevation))
        .collect();
    let water_initial = total_water(&sim);

    let mut stats = SeedStats {
        seed,
        cell_count,
        water_initial,
        water_final: 0.0,
        nan_inf_events: Vec::new(),
        bounds_events: Vec::new(),
        rain_days: HashMap::new(),
        lake_ticks: HashMap::new(),
        elevations,
        temp_min: HashMap::new(),
        temp_max: HashMap::new(),
        max_discharge_seen: 0.0,
        dry_streak_max: 0,
        dry_streak_cur: 0,
        wind_sector_counts: [0; 6],
        wind_magnitude_sum: 0.0,
        wind_magnitude_samples: 0,
        wind_magnitude_max: 0.0,
        snow_total_min: f32::INFINITY,
        snow_total_max: 0.0,
    };

    for tick in 0..TOTAL_TICKS {
        sim.step();
        let grid = sim.grid();

        for (coord, cell) in grid.iter() {
            let props: [(&str, f32); 6] = [
                ("elevation", cell.elevation),
                ("temperature", cell.temperature),
                ("water_level", cell.water_level),
                ("humidity", cell.humidity_total()),
                ("groundwater", cell.groundwater),
                ("snow_level", cell.snow_level),
            ];
            for (name, v) in props {
                if !v.is_finite() && stats.nan_inf_events.len() < 5 {
                    stats.nan_inf_events.push(format!(
                        "tick {tick}, ({},{}) {name} = {v}",
                        coord.q, coord.r
                    ));
                }
            }
            // NEG_EPS: tolerates the residual floating-point epsilon from hydro/atmo ops.
            if cell.humidity_total() < NEG_EPS && stats.bounds_events.len() < 5 {
                stats.bounds_events.push(format!(
                    "humidity < 0 : {:.5} tick {tick}",
                    cell.humidity_total()
                ));
            }
            if cell.water_level < NEG_EPS && stats.bounds_events.len() < 5 {
                stats.bounds_events.push(format!(
                    "water_level < 0 : {:.5} tick {tick}",
                    cell.water_level
                ));
            }
            if cell.snow_level < NEG_EPS && stats.bounds_events.len() < 5 {
                stats.bounds_events.push(format!(
                    "snow_level < 0 : {:.5} tick {tick}",
                    cell.snow_level
                ));
            }
            if (cell.temperature < -80.0 || cell.temperature > 80.0)
                && stats.bounds_events.len() < 5
            {
                stats.bounds_events.push(format!(
                    "temperature hors [-80,80] : {:.1} tick {tick}",
                    cell.temperature
                ));
            }
        }

        if tick >= WARMUP_TICKS {
            accumulate_measuring_tick(&sim, &mut stats);
        }
    }

    stats.water_final = total_water(&sim);
    timer.lap(&format!("simulation seed {seed}"));
    stats
}

fn check_seed(stats: &SeedStats, must_pass: &mut Vec<String>, climatology: &mut Vec<String>) {
    eprintln!("--- Seed {} (cells = {}) ---", stats.seed, stats.cell_count);

    // Check 1 : conservation
    let drift = (stats.water_final - stats.water_initial).abs() / stats.water_initial.max(1e-6);
    eprintln!(
        "  water   init={:.2} final={:.2} drift={}",
        stats.water_initial,
        stats.water_final,
        pct(drift)
    );
    if drift > 0.01 {
        climatology.push(format!(
            "[seed {}] water conservation: drift {} > 1%",
            stats.seed,
            pct(drift)
        ));
    }

    // Check 2+3 : NaN/Inf et bornes physiques
    if !stats.nan_inf_events.is_empty() {
        must_pass.push(format!(
            "[seed {}] NaN/Inf detected: {:?}",
            stats.seed, stats.nan_inf_events
        ));
    }
    if !stats.bounds_events.is_empty() {
        must_pass.push(format!(
            "[seed {}] physical bounds violated: {:?}",
            stats.seed, stats.bounds_events
        ));
    }

    check_precipitation_gradient(stats, climatology);
    check_lake_halo(stats, climatology);
    check_rain_ceiling(stats, must_pass);
    check_seasonal_amplitude(stats, climatology);

    // Check 9: rivers
    eprintln!(
        "  max_discharge observed over the window = {:.3}",
        stats.max_discharge_seen
    );
    if stats.max_discharge_seen <= 1e-3 {
        must_pass.push(format!(
            "[seed {}] max_discharge = {:.3e}: no river",
            stats.seed, stats.max_discharge_seen
        ));
    }

    check_permanent_lakes(stats, climatology);
    check_dry_streak(stats, climatology);
    check_rose_vents(stats, climatology);
    check_snow_cycle(stats, climatology);
}

fn check_precipitation_gradient(stats: &SeedStats, climatology: &mut Vec<String>) {
    let bands: [(&str, f32, f32); 4] = [
        ("0-200", 0.0, 200.0),
        ("200-500", 200.0, 500.0),
        ("500-1000", 500.0, 1000.0),
        ("1000+", 1000.0, f32::INFINITY),
    ];
    let mut band_means: Vec<f32> = Vec::new();
    for (name, lo, hi) in bands {
        let cells: Vec<HexCoord> = stats
            .elevations
            .iter()
            .filter(|(_, e)| **e >= lo && **e < hi)
            .map(|(c, _)| *c)
            .collect();
        if cells.is_empty() {
            eprintln!("  band {name:<9}: (empty)");
            band_means.push(f32::NAN);
            continue;
        }
        let n_f = f32::from(u16::try_from(cells.len()).expect("fits u16"));
        let mean: f32 = cells
            .iter()
            .map(|c| {
                f32::from(
                    u16::try_from(stats.rain_days.get(c).copied().unwrap_or(0)).expect("fits u16"),
                )
            })
            .sum::<f32>()
            / n_f;
        eprintln!(
            "  band {name:<9} cells={:<4} avg rain days/yr = {mean:.1}",
            cells.len()
        );
        band_means.push(mean);
    }
    let defined: Vec<f32> = band_means
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let monotone_up = defined.windows(2).all(|w| w[1] > w[0]);
    if !monotone_up && defined.len() >= 2 {
        climatology.push(format!(
            "[seed {}] non-monotonic altitudinal rain gradient: bands = {:?}",
            stats.seed, band_means
        ));
    }
    let mid_cells: Vec<HexCoord> = stats
        .elevations
        .iter()
        .filter(|(_, e)| (300.0..700.0).contains(*e))
        .map(|(c, _)| *c)
        .collect();
    let mid_zero: Vec<HexCoord> = mid_cells
        .iter()
        .copied()
        .filter(|c| stats.rain_days.get(c).copied().unwrap_or(0) == 0)
        .collect();
    eprintln!(
        "  mid-alt 300-700 cells={} at 0 rain = {}",
        mid_cells.len(),
        mid_zero.len()
    );
    if !mid_zero.is_empty() {
        climatology.push(format!(
            "[seed {}] mid-alt 300-700 m: {} cells at 0 mm/yr (examples {:?})",
            stats.seed,
            mid_zero.len(),
            mid_zero.iter().take(3).collect::<Vec<_>>()
        ));
    }
}

fn check_lake_halo(stats: &SeedStats, climatology: &mut Vec<String>) {
    let permanent_lakes: std::collections::HashSet<HexCoord> = stats
        .lake_ticks
        .iter()
        .filter(|(_, n)| **n >= LAKE_PERM_TICKS)
        .map(|(c, _)| *c)
        .collect();
    let halo: std::collections::HashSet<HexCoord> = permanent_lakes
        .iter()
        .flat_map(|c| c.neighbors())
        .filter(|c| stats.elevations.contains_key(c) && !permanent_lakes.contains(c))
        .collect();
    let measure_ticks_f = f32::from(u16::try_from(MEASURE_TICKS).expect("fits u16"));
    let halo_overrain: Vec<(HexCoord, u32)> = halo
        .iter()
        .map(|c| (*c, stats.rain_days.get(c).copied().unwrap_or(0)))
        .filter(|(_, d)| f32::from(u16::try_from(*d).expect("fits u16")) / measure_ticks_f > 0.6)
        .collect();
    eprintln!(
        "  lake halo: lakes_perm={} halo={} halo>60%rain={}",
        permanent_lakes.len(),
        halo.len(),
        halo_overrain.len()
    );
    if !halo_overrain.is_empty() {
        let top: Vec<(HexCoord, f32)> = halo_overrain
            .iter()
            .take(3)
            .map(|(c, d)| {
                (
                    *c,
                    f32::from(u16::try_from(*d).expect("fits u16")) / measure_ticks_f,
                )
            })
            .collect();
        climatology.push(format!(
            "[seed {}] saturated lake halo: {} cells > 60% rain (ex. {:?})",
            stats.seed,
            halo_overrain.len(),
            top
        ));
    }
}

fn check_rain_ceiling(stats: &SeedStats, must_pass: &mut Vec<String>) {
    let over_ceiling: Vec<(HexCoord, u32)> = stats
        .rain_days
        .iter()
        .filter(|(_, d)| **d > 320)
        .map(|(c, d)| (*c, *d))
        .collect();
    let max_days = stats.rain_days.values().copied().max().unwrap_or(0);
    eprintln!(
        "  rain ceiling: max = {max_days} days/yr, cells > 320 days = {}",
        over_ceiling.len()
    );
    if !over_ceiling.is_empty() {
        must_pass.push(format!(
            "[seed {}] rain ceiling: {} cells > 320 days/yr (max = {} days, ex. {:?})",
            stats.seed,
            over_ceiling.len(),
            max_days,
            over_ceiling.iter().take(3).collect::<Vec<_>>()
        ));
    }
}

fn check_seasonal_amplitude(stats: &SeedStats, climatology: &mut Vec<String>) {
    let deltas: Vec<f32> = stats
        .temp_min
        .iter()
        .filter_map(|(c, tmin)| stats.temp_max.get(c).map(|tmax| tmax - tmin))
        .collect();
    let mean_delta = if deltas.is_empty() {
        0.0_f32
    } else {
        let n_f = f32::from(u16::try_from(deltas.len()).expect("fits u16"));
        deltas.iter().sum::<f32>() / n_f
    };
    eprintln!("  mean seasonal amplitude (temp_max - temp_min) = {mean_delta:.1} °C");
    if mean_delta < 10.0 {
        climatology.push(format!(
            "[seed {}] mean seasonal amplitude {mean_delta:.1} °C < 10 °C (season_amplitude = 13 °C expected)",
            stats.seed
        ));
    }
}

fn check_permanent_lakes(stats: &SeedStats, climatology: &mut Vec<String>) {
    let permanent_lake_cells = stats
        .lake_ticks
        .values()
        .filter(|n| **n >= LAKE_PERM_TICKS)
        .count();
    let cell_count_f = f32::from(u16::try_from(stats.cell_count).expect("fits u16"));
    let lake_perm_f = f32::from(u16::try_from(permanent_lake_cells).expect("fits u16"));
    let lake_ratio = lake_perm_f / cell_count_f;
    eprintln!(
        "  permanent lakes = {permanent_lake_cells}/{} ({:.1}%)",
        stats.cell_count,
        lake_ratio * 100.0
    );
    if permanent_lake_cells == 0 {
        climatology.push(format!(
            "[seed {}] no permanent lake over the measurement window",
            stats.seed
        ));
    }
    if lake_ratio > 0.5 {
        climatology.push(format!(
            "[seed {}] {:.1}% of the map is permanent lake (> 50%): flood",
            stats.seed,
            lake_ratio * 100.0
        ));
    }
}

fn check_dry_streak(stats: &SeedStats, climatology: &mut Vec<String>) {
    eprintln!(
        "  max dry streak (consecutive ticks with no rain anywhere) = {}",
        stats.dry_streak_max
    );
    if stats.dry_streak_max < 5 {
        climatology.push(format!(
            "[seed {}] max dry streak = {} ticks < 5 (no real calm periods)",
            stats.seed, stats.dry_streak_max
        ));
    }
}

fn check_rose_vents(stats: &SeedStats, climatology: &mut Vec<String>) {
    let wind_total: u64 = stats.wind_sector_counts.iter().sum();
    let wind_mean_mag = if stats.wind_magnitude_samples > 0 {
        // wind_magnitude_samples fits in u32 (MEASURE_TICKS * cells ~ 365 * 1300 ~ 500k)
        stats.wind_magnitude_sum
            / f64::from(u32::try_from(stats.wind_magnitude_samples).expect("fits u32"))
    } else {
        0.0
    };
    eprintln!(
        "  wind: |w| avg = {:.3}, max = {:.3}, rose = {:?}",
        wind_mean_mag, stats.wind_magnitude_max, stats.wind_sector_counts
    );
    if wind_total > 0 {
        // Compute dominant/minimal as fractions in [0, 1] using integer ratio
        // (avoid float precision issues; wind_total fits in u32 for our grid sizes)
        let wt = u32::try_from(wind_total).expect("fits u32");
        let fractions: Vec<f64> = stats
            .wind_sector_counts
            .iter()
            .map(|n| f64::from(u32::try_from(*n).expect("fits u32")) / f64::from(wt))
            .collect();
        let dominant = fractions.iter().copied().fold(0.0_f64, f64::max);
        let minimal = fractions.iter().copied().fold(1.0_f64, f64::min);
        if dominant > 0.80 {
            climatology.push(format!(
                "[seed {}] wind rose monopolized: dominant sector = {:.1}% (> 80%)",
                stats.seed,
                dominant * 100.0
            ));
        }
        if minimal < 0.02 {
            climatology.push(format!(
                "[seed {}] wind rose has a gap: minimal sector = {:.2}% (< 2%)",
                stats.seed,
                minimal * 100.0
            ));
        }
    }
    if wind_mean_mag < 0.05 {
        climatology.push(format!(
            "[seed {}] anemic average wind: |w| = {:.3} (< 0.05)",
            stats.seed, wind_mean_mag
        ));
    }
}

fn check_snow_cycle(stats: &SeedStats, climatology: &mut Vec<String>) {
    let snow_ratio = if stats.snow_total_max > 1e-3 {
        stats.snow_total_min / stats.snow_total_max
    } else {
        1.0
    };
    eprintln!(
        "  snow cycle: min={:.1} max={:.1} min/max={:.2}",
        stats.snow_total_min, stats.snow_total_max, snow_ratio
    );
    if stats.snow_total_max < 1e-3 {
        climatology.push(format!(
            "[seed {}] zero snowpack over the whole window: no snow at all",
            stats.seed
        ));
    } else if snow_ratio > 0.5 {
        climatology.push(format!(
            "[seed {}] flat snow cycle: min/max = {:.2} > 0.5 (no clear summer melt)",
            stats.seed, snow_ratio
        ));
    }
}

#[test]
fn scale_universal_invariants_multi_seed() {
    let mut timer = PerfTimer::start("scale_universal_invariants");

    let all_stats: Vec<SeedStats> = SEEDS.iter().map(|&s| run_one_seed(s, &mut timer)).collect();
    timer.ticks(TOTAL_TICKS * u64::from(u32::try_from(SEEDS.len()).expect("fits u32")));

    let mut must_pass: Vec<String> = Vec::new();
    let mut climatology: Vec<String> = Vec::new();

    for stats in &all_stats {
        check_seed(stats, &mut must_pass, &mut climatology);
    }

    timer.report();

    if !climatology.is_empty() {
        eprintln!(
            "\n=== Climatology warnings ({}), non-blocking ===",
            climatology.len()
        );
        for f in &climatology {
            eprintln!("  WARN {f}");
        }
    }
    if !must_pass.is_empty() {
        eprintln!(
            "\n=== MUST-PASS failures ({}), blocking ===",
            must_pass.len()
        );
        for f in &must_pass {
            eprintln!("  FAIL {f}");
        }
        panic!(
            "{} structural invariant(s) violated in scale_universal_invariants",
            must_pass.len()
        );
    }
}
