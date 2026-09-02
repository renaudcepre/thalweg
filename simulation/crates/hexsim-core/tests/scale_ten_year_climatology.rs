//! Scale test: 1 year of warm-up + 10 years of measurement, fixed seed.
//!
//! Philosophy: this test doesn't try to validate a local rule, it
//! characterizes the *climate* the simulation produces over the long run.
//! Many assertions are bets on what the world should look like; they will
//! fail as long as the simulation isn't calibrated. We let them fail: they
//! are a visible to-do list.
//!
//! Warm-up: the sim starts from a uniform state (1 unit of water everywhere,
//! 1.5 unit of groundwater everywhere). During the first year the water
//! table saturates, rivers carve their bed, lakes find their equilibrium;
//! it's a pure transient. We ignore these first 365 ticks and measure over
//! the following 10 years. Without this, water table drift and river
//! stability would be measured against initial noise rather than the
//! steady state.

mod common;

use std::collections::{HashMap, HashSet};

use common::{PerfTimer, pct};
use hexsim_core::atmosphere::{AtmosphereParams, saturation_upper};
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::WindParams;

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const YEAR: u64 = 365;
/// Years ignored at the start of the sim.
const WARMUP_TICKS: u64 = YEAR;
/// Years of measurement after the warm-up.
const YEARS: u64 = 3;
const MEASURE_TICKS: u64 = YEARS * YEAR;
const TOTAL_TICKS: u64 = WARMUP_TICKS + MEASURE_TICKS;

/// 80% threshold of measurement ticks for permanent glacier/lake (1095 * 80 / 100 = 876).
const THRESHOLD_80_TICKS: u32 = 876;
/// 5% threshold of measurement ticks for plains cloud/snow (1095 * 5 / 100 = 54).
const THRESHOLD_5_TICKS: u32 = 54;
/// 50% threshold of the year for a lake in its first year (365 / 2 = 182).
const THRESHOLD_50_YEAR: u32 = 182;
/// 80% threshold of the year for a persistent lake in its last year (365 * 80 / 100 = 292).
const THRESHOLD_80_YEAR: u32 = 292;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Season {
    Winter,
    Spring,
    Summer,
    Autumn,
}

fn season_of(tick_of_year: u64) -> Season {
    match tick_of_year {
        0..=54 | 335..=364 => Season::Winter,
        55..=145 => Season::Spring,
        146..=237 => Season::Summer,
        _ => Season::Autumn,
    }
}

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

fn total_water(sim: &Simulation) -> f32 {
    sim.grid()
        .iter()
        .map(|(_, c)| c.water_level + c.humidity_total() + c.groundwater + c.snow_level)
        .sum()
}

/// Average of the wind vectors over the grid.
fn mean_wind(sim: &Simulation) -> (f32, f32) {
    let mut sx = 0.0_f32;
    let mut sy = 0.0_f32;
    let mut n = 0_u32;
    for v in sim.wind_field() {
        sx += v.x;
        sy += v.y;
        n += 1;
    }
    if n == 0 {
        (0.0, 0.0)
    } else {
        let nf = f32::from(u16::try_from(n).expect("fits u16"));
        (sx / nf, sy / nf)
    }
}

/// Buckets a direction into 8 sectors of 45° (0=E, 1=NE, ...).
fn direction_bucket(x: f32, y: f32) -> usize {
    let norm = (y.atan2(x).to_degrees() + 360.0) % 360.0;
    if !(22.5..337.5).contains(&norm) {
        0
    } else if norm < 67.5 {
        1
    } else if norm < 112.5 {
        2
    } else if norm < 157.5 {
        3
    } else if norm < 202.5 {
        4
    } else if norm < 247.5 {
        5
    } else if norm < 292.5 {
        6
    } else {
        7
    }
}

const DIRECTION_NAMES: [&str; 8] = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"];

/// Set of per-cell, per-year accumulators updated during the simulation.
struct SimAccumulators {
    years_usize: usize,
    wind_buckets: [u32; 8],
    wind_mean_mag_cum: f64,
    precip_by_season: HashMap<Season, (f64, u32)>,
    temp_by_season: HashMap<Season, (f64, u32)>,
    snow_coverage_by_season: HashMap<Season, Vec<u32>>,
    fronts_per_year: Vec<u32>,
    dry_days_per_year: Vec<u32>,
    snow_ticks_per_cell: HashMap<HexCoord, u32>,
    snow_max_per_cell: HashMap<HexCoord, f32>,
    glacier_ticks_per_cell: HashMap<HexCoord, u32>,
    water_sum_per_cell: HashMap<HexCoord, f32>,
    temp_min_per_cell: HashMap<HexCoord, f32>,
    temp_max_per_cell: HashMap<HexCoord, f32>,
    temp_sum_per_cell: HashMap<HexCoord, f64>,
    humidity_upper_sum_per_cell: HashMap<HexCoord, f64>,
    cloudy_ticks_per_cell: HashMap<HexCoord, u32>,
    precip_sum_per_cell: HashMap<HexCoord, f64>,
    gw_sum_year: Vec<f64>,
    gw_count_year: Vec<u64>,
    discharge_year_first: HashMap<HexCoord, f32>,
    discharge_year_last: HashMap<HexCoord, f32>,
    lake_ticks_year_first: HashMap<HexCoord, u32>,
    lake_ticks_year_last: HashMap<HexCoord, u32>,
    tick_perf_log: Vec<(u64, f64)>,
    lapse_rate_first_year: Option<f32>,
    water_initial_measure: f32,
}

impl SimAccumulators {
    fn new(years_usize: usize, water_initial_sim: f32) -> Self {
        Self {
            years_usize,
            wind_buckets: [0; 8],
            wind_mean_mag_cum: 0.0,
            precip_by_season: HashMap::new(),
            temp_by_season: HashMap::new(),
            snow_coverage_by_season: HashMap::new(),
            fronts_per_year: vec![0; years_usize],
            dry_days_per_year: vec![0; years_usize],
            snow_ticks_per_cell: HashMap::new(),
            snow_max_per_cell: HashMap::new(),
            glacier_ticks_per_cell: HashMap::new(),
            water_sum_per_cell: HashMap::new(),
            temp_min_per_cell: HashMap::new(),
            temp_max_per_cell: HashMap::new(),
            temp_sum_per_cell: HashMap::new(),
            humidity_upper_sum_per_cell: HashMap::new(),
            cloudy_ticks_per_cell: HashMap::new(),
            precip_sum_per_cell: HashMap::new(),
            gw_sum_year: vec![0.0; years_usize],
            gw_count_year: vec![0; years_usize],
            discharge_year_first: HashMap::new(),
            discharge_year_last: HashMap::new(),
            lake_ticks_year_first: HashMap::new(),
            lake_ticks_year_last: HashMap::new(),
            tick_perf_log: Vec::new(),
            lapse_rate_first_year: None,
            water_initial_measure: water_initial_sim,
        }
    }
}

fn run_main_loop(sim: &mut Simulation, cell_count: usize, acc: &mut SimAccumulators) {
    let temp_params = TemperatureParams::default();
    let atmosphere_params = AtmosphereParams::default();
    let year_f64 = f64::from(u32::try_from(YEAR).expect("fits u32"));
    let snapshot_ticks: HashMap<u64, &'static str> = [
        (WARMUP_TICKS + 4 * YEAR + 80, "spring measurement year 5"),
        (WARMUP_TICKS + 4 * YEAR + 170, "summer measurement year 5"),
        (WARMUP_TICKS + 4 * YEAR + 260, "autumn measurement year 5"),
        (WARMUP_TICKS + 4 * YEAR + 355, "winter measurement year 5"),
    ]
    .into_iter()
    .collect();
    let mut last_year_instant = std::time::Instant::now();

    for tick in 0..TOTAL_TICKS {
        sim.step();
        if (tick + 1) % YEAR == 0 {
            let dt = last_year_instant.elapsed().as_secs_f64();
            last_year_instant = std::time::Instant::now();
            acc.tick_perf_log.push((tick + 1, (dt * 1000.0) / year_f64));
        }
        if let Some(label) = snapshot_ticks.get(&tick) {
            let diag = sim.diagnostics();
            eprintln!(
                "\n[snapshot {label} tick={tick}] water_total={:.1} humidity_mean={:.3} \
                 snow_total={:.1} temp_mean={:.1} raining_cells={}",
                diag.water_budget.total,
                diag.humidity.mean,
                diag.snow.total,
                diag.temperature.mean,
                diag.hydrology.raining_cells,
            );
        }
        if tick + 1 == WARMUP_TICKS {
            acc.water_initial_measure = total_water(sim);
        }
        if tick < WARMUP_TICKS {
            continue;
        }

        let measure_tick = tick - WARMUP_TICKS;
        let year_idx = usize::try_from(measure_tick / YEAR).expect("fits usize");
        let season = season_of(tick % YEAR);

        let (wx, wy) = mean_wind(sim);
        let mag = (wx * wx + wy * wy).sqrt();
        acc.wind_mean_mag_cum += f64::from(mag);
        acc.wind_buckets[direction_bucket(wx, wy)] += 1;

        accumulate_cells(sim, year_idx, season, &temp_params, &atmosphere_params, acc);

        let (raining, daily_total) = accumulate_precip(sim, acc);
        let cell_u16 = u16::try_from(cell_count.max(1)).expect("fits u16");
        let raining_u16 = u16::try_from(raining).expect("fits u16");
        let coverage = f32::from(raining_u16) / f32::from(cell_u16);
        let p_entry = acc.precip_by_season.entry(season).or_insert((0.0, 0));
        p_entry.0 += daily_total;
        p_entry.1 += 1;
        if coverage >= 0.9 && year_idx < acc.years_usize {
            acc.fronts_per_year[year_idx] += 1;
        }
        if raining == 0 && year_idx < acc.years_usize {
            acc.dry_days_per_year[year_idx] += 1;
        }
        if tick + 1 == WARMUP_TICKS + YEAR {
            acc.lapse_rate_first_year = Some(compute_lapse_rate(sim.grid()));
        }
    }
}

fn accumulate_cells(
    sim: &Simulation,
    year_idx: usize,
    season: Season,
    temp_params: &TemperatureParams,
    atmosphere_params: &AtmosphereParams,
    acc: &mut SimAccumulators,
) {
    let mut snow_covered = 0_u32;
    let mut temp_sum = 0.0_f64;
    let mut temp_n = 0_u32;
    let mut gw_sum = 0.0_f64;
    let mut gw_n = 0_u32;

    for (coord, cell) in sim.grid().iter() {
        if cell.snow_level > 0.1 {
            snow_covered += 1;
            *acc.snow_ticks_per_cell.entry(*coord).or_default() += 1;
        }
        let max_e = acc.snow_max_per_cell.entry(*coord).or_default();
        if cell.snow_level > *max_e {
            *max_e = cell.snow_level;
        }
        if cell.snow_level > 5.0 {
            *acc.glacier_ticks_per_cell.entry(*coord).or_default() += 1;
        }
        *acc.water_sum_per_cell.entry(*coord).or_default() += cell.water_level;
        let t = cell.temperature;
        acc.temp_min_per_cell
            .entry(*coord)
            .and_modify(|v| *v = v.min(t))
            .or_insert(t);
        acc.temp_max_per_cell
            .entry(*coord)
            .and_modify(|v| *v = v.max(t))
            .or_insert(t);
        *acc.temp_sum_per_cell.entry(*coord).or_default() += f64::from(t);
        *acc.humidity_upper_sum_per_cell.entry(*coord).or_default() +=
            f64::from(cell.humidity_upper);
        let t_upper = cell.temperature - temp_params.lapse_rate * 1.5;
        let sat = saturation_upper(t_upper, atmosphere_params);
        let hr = if sat > 0.0 {
            cell.humidity_upper / sat
        } else {
            0.0
        };
        if hr > 0.6 {
            *acc.cloudy_ticks_per_cell.entry(*coord).or_default() += 1;
        }
        temp_sum += f64::from(cell.temperature);
        temp_n += 1;
        gw_sum += f64::from(cell.groundwater);
        gw_n += 1;
    }

    // Update season-based accumulators with aggregated tick values.
    acc.snow_coverage_by_season
        .entry(season)
        .or_default()
        .push(snow_covered);
    let t_entry = acc.temp_by_season.entry(season).or_insert((0.0, 0));
    t_entry.0 += temp_sum;
    t_entry.1 += temp_n;

    if year_idx < acc.years_usize {
        acc.gw_sum_year[year_idx] += gw_sum / f64::from(gw_n.max(1));
        acc.gw_count_year[year_idx] += 1;
    }

    // lake + discharge
    let coords_disc = sim.grid().coords_slice();
    let is_first = year_idx == 0;
    let is_last = year_idx + 1 == acc.years_usize;
    if is_first {
        for (i, &d) in sim.discharge_map().iter().enumerate() {
            *acc.discharge_year_first.entry(coords_disc[i]).or_default() += d;
        }
        for (coord, cell) in sim.grid().iter() {
            if cell.water_level > 0.1 * cell.water_capacity {
                *acc.lake_ticks_year_first.entry(*coord).or_default() += 1;
            }
        }
    } else if is_last {
        for (i, &d) in sim.discharge_map().iter().enumerate() {
            *acc.discharge_year_last.entry(coords_disc[i]).or_default() += d;
        }
        for (coord, cell) in sim.grid().iter() {
            if cell.water_level > 0.1 * cell.water_capacity {
                *acc.lake_ticks_year_last.entry(*coord).or_default() += 1;
            }
        }
    }
}

fn accumulate_precip(sim: &Simulation, acc: &mut SimAccumulators) -> (u32, f64) {
    let precip = sim.last_precipitation();
    let coords = sim.grid().coords_slice();
    let mut raining = 0_u32;
    let mut daily_total = 0.0_f64;
    for (i, rec) in precip.iter().enumerate() {
        if rec.rain > 1e-4 || rec.snow > 1e-4 {
            raining += 1;
        }
        let daily = f64::from(rec.rain) + f64::from(rec.snow);
        daily_total += daily;
        *acc.precip_sum_per_cell.entry(coords[i]).or_default() += daily;
    }
    (raining, daily_total)
}

fn print_band_diagnostics(elevations: &HashMap<HexCoord, f32>, acc: &SimAccumulators) {
    let bands: &[(&str, f32, f32)] = &[
        ("<0m", f32::NEG_INFINITY, 0.0),
        ("0-300m", 0.0, 300.0),
        ("300-800m", 300.0, 800.0),
        ("800-1500m", 800.0, 1500.0),
        (">1500m", 1500.0, f32::INFINITY),
    ];
    eprintln!(
        "\n=== Climate by band ({YEARS} years) ===\n  {:>10}  {:>5}  {:>8} {:>8} {:>8}  {:>10}  {:>10}",
        "band", "cells", "t_min", "t_max", "t_mean", "precip/yr", "snow_max"
    );
    let measure_f64 = f64::from(u32::try_from(MEASURE_TICKS).expect("fits u32"));
    let years_f64 = f64::from(u32::try_from(YEARS).expect("fits u32"));
    for (name, lo, hi) in bands {
        let cells: Vec<HexCoord> = elevations
            .iter()
            .filter(|(_, e)| **e >= *lo && **e < *hi)
            .map(|(c, _)| *c)
            .collect();
        if cells.is_empty() {
            continue;
        }
        let n_f = f32::from(u16::try_from(cells.len()).expect("fits u16"));
        let n_f64 = f64::from(u32::try_from(cells.len()).expect("fits u32"));
        let t_min = cells
            .iter()
            .filter_map(|c| acc.temp_min_per_cell.get(c).copied())
            .sum::<f32>()
            / n_f;
        let t_max = cells
            .iter()
            .filter_map(|c| acc.temp_max_per_cell.get(c).copied())
            .sum::<f32>()
            / n_f;
        let t_mean = cells
            .iter()
            .filter_map(|c| acc.temp_sum_per_cell.get(c).copied())
            .sum::<f64>()
            / (n_f64 * measure_f64);
        let precip_yearly = cells
            .iter()
            .filter_map(|c| acc.precip_sum_per_cell.get(c).copied())
            .sum::<f64>()
            / (n_f64 * years_f64);
        let snow_max = cells
            .iter()
            .filter_map(|c| acc.snow_max_per_cell.get(c).copied())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "  {:>10}  {:>5}  {:>+8.2} {:>+8.2} {:>+8.2}  {:>10.2}  {:>10.2}",
            name,
            cells.len(),
            t_min,
            t_max,
            t_mean,
            precip_yearly,
            snow_max
        );
    }
    eprintln!();
}

fn check_glaciers(
    elevations: &HashMap<HexCoord, f32>,
    acc: &SimAccumulators,
    climatology: &mut Vec<String>,
) {
    let high_cells: Vec<HexCoord> = elevations
        .iter()
        .filter(|(_, e)| **e > 1500.0)
        .map(|(c, _)| *c)
        .collect();
    let non_glaciers: Vec<HexCoord> = high_cells
        .iter()
        .copied()
        .filter(|c| acc.snow_ticks_per_cell.get(c).copied().unwrap_or(0) < THRESHOLD_80_TICKS)
        .collect();
    eprintln!(
        "Glaciers: high cells (>1500m) = {}, without permanent snow = {}",
        high_cells.len(),
        non_glaciers.len()
    );
    let measure_f = f32::from(u16::try_from(MEASURE_TICKS).expect("fits u16"));
    let mut high_stats: Vec<(HexCoord, f32, u32, f32, f32, u32, f32)> = high_cells
        .iter()
        .map(|c| {
            let elev = *elevations.get(c).unwrap_or(&0.0);
            let ticks = acc.snow_ticks_per_cell.get(c).copied().unwrap_or(0);
            let frac = f32::from(u16::try_from(ticks).expect("fits u16")) / measure_f;
            let snow_max = acc.snow_max_per_cell.get(c).copied().unwrap_or(0.0);
            let gl_ticks = acc.glacier_ticks_per_cell.get(c).copied().unwrap_or(0);
            let water_mean = acc.water_sum_per_cell.get(c).copied().unwrap_or(0.0) / measure_f;
            (*c, elev, ticks, frac, snow_max, gl_ticks, water_mean)
        })
        .collect();
    high_stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (coord, elev, ticks, frac, snow_max, gl_ticks, water_mean) in &high_stats {
        eprintln!(
            "  {:?} elev={:.0}m snow_ticks={}/{} ({:.0}%) snow_max={:.2} glacier_ticks={} water_mean={:.3}",
            coord,
            elev,
            ticks,
            MEASURE_TICKS,
            frac * 100.0,
            snow_max,
            gl_ticks,
            water_mean
        );
    }
    if !high_cells.is_empty() && !non_glaciers.is_empty() {
        climatology.push(format!(
            "glaciers: {} cells > 1500 m without snow > 80%",
            non_glaciers.len()
        ));
    }
}

fn check_clouds_over_relief(
    elevations: &HashMap<HexCoord, f32>,
    acc: &SimAccumulators,
    climatology: &mut Vec<String>,
) {
    let high: Vec<HexCoord> = elevations
        .iter()
        .filter(|(_, e)| **e > 1000.0)
        .map(|(c, _)| *c)
        .collect();
    let with_clouds: Vec<HexCoord> = high
        .iter()
        .copied()
        .filter(|c| acc.cloudy_ticks_per_cell.get(c).copied().unwrap_or(0) > THRESHOLD_5_TICKS)
        .collect();
    let fraction = if high.is_empty() {
        0.0_f32
    } else {
        f32::from(u16::try_from(with_clouds.len()).expect("fits u16"))
            / f32::from(u16::try_from(high.len()).expect("fits u16"))
    };
    let hum_mean: f64 = if high.is_empty() {
        0.0
    } else {
        let n_f64 = f64::from(u32::try_from(high.len()).expect("fits u32"));
        let m_f64 = f64::from(u32::try_from(MEASURE_TICKS).expect("fits u32"));
        high.iter()
            .map(|c| {
                acc.humidity_upper_sum_per_cell
                    .get(c)
                    .copied()
                    .unwrap_or(0.0)
            })
            .sum::<f64>()
            / (n_f64 * m_f64)
    };
    eprintln!(
        "Cloud coverage relief (>1000m): {}/{} ({:.0}%), hum_up mean = {:.4}",
        with_clouds.len(),
        high.len(),
        fraction * 100.0,
        hum_mean
    );
    if !high.is_empty() && fraction < 0.50 {
        climatology.push(format!(
            "relief clouds: {:.0}% of cells > 1000 m see clouds > 5% of the time (target >= 50%)",
            fraction * 100.0
        ));
    }
}

fn check_nappes(acc: &SimAccumulators, climatology: &mut Vec<String>) {
    let gw_annual: Vec<f64> = acc
        .gw_sum_year
        .iter()
        .zip(&acc.gw_count_year)
        .map(|(s, &n)| s / f64::from(u32::try_from(n.max(1)).expect("fits u32")))
        .collect();
    let n_f64 = f64::from(u32::try_from(gw_annual.len()).expect("fits u32"));
    let gw_total_mean = gw_annual.iter().sum::<f64>() / n_f64;
    let gw_drift =
        ((gw_annual[0] - gw_annual[gw_annual.len() - 1]) / gw_total_mean.max(1e-6)).abs();
    eprintln!(
        "Groundwater per year: {:?} (drift = {:.1}%)",
        gw_annual
            .iter()
            .map(|x| format!("{x:.2}"))
            .collect::<Vec<_>>(),
        gw_drift * 100.0
    );
    if gw_drift > 0.10 {
        climatology.push(format!(
            "groundwater: drift {:.1}% between first year and last year (> 10%)",
            gw_drift * 100.0
        ));
    }
}

fn check_fronts_secs(acc: &SimAccumulators, climatology: &mut Vec<String>) {
    let n_f = f32::from(u8::try_from(acc.fronts_per_year.len()).expect("fits u8"));
    let front_avg = acc
        .fronts_per_year
        .iter()
        .map(|&n| f32::from(u16::try_from(n).expect("fits u16")))
        .sum::<f32>()
        / n_f;
    eprintln!(
        "Fronts per year: {:?} (mean {front_avg:.1})",
        acc.fronts_per_year
    );
    if !(2.0..=30.0).contains(&front_avg) {
        climatology.push(format!("fronts: {front_avg:.1}/yr out of range [2, 30]"));
    }
    let dry_avg = acc
        .dry_days_per_year
        .iter()
        .map(|&n| f32::from(u16::try_from(n).expect("fits u16")))
        .sum::<f32>()
        / n_f;
    eprintln!(
        "Fully dry days per year: {:?} (mean {dry_avg:.1})",
        acc.dry_days_per_year
    );
    if dry_avg < 10.0 {
        climatology.push(format!(
            "dry days: {dry_avg:.1}/yr < 10 (permanent rain somewhere)"
        ));
    }
}

fn check_wind_rose(acc: &SimAccumulators, climatology: &mut Vec<String>) {
    let total_wind: u32 = acc.wind_buckets.iter().sum();
    let wind_f = f32::from(u16::try_from(total_wind.max(1)).expect("fits u16"));
    let bucket_fracs: Vec<f32> = acc
        .wind_buckets
        .iter()
        .map(|n| f32::from(u16::try_from(*n).expect("fits u16")) / wind_f)
        .collect();
    let (dom_idx, dom_frac) = bucket_fracs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or((0, 0.0), |(i, v)| (i, *v));
    let mean_mag = acc.wind_mean_mag_cum / f64::from(total_wind.max(1));
    eprintln!(
        "Wind: dominant {} ({}), mean magnitude {:.3}",
        DIRECTION_NAMES[dom_idx],
        pct(dom_frac),
        mean_mag
    );
    for (i, f) in bucket_fracs.iter().enumerate() {
        eprintln!("  {:<3} {}", DIRECTION_NAMES[i], pct(*f));
    }
    if dom_frac < 0.35 {
        climatology.push(format!(
            "wind: no dominant direction (max {} at {})",
            DIRECTION_NAMES[dom_idx],
            pct(dom_frac)
        ));
    }
    if dom_frac > 0.80 {
        climatology.push(format!(
            "wind: fixed direction ({} at {})",
            DIRECTION_NAMES[dom_idx],
            pct(dom_frac)
        ));
    }
}

fn check_seasonality(acc: &SimAccumulators, climatology: &mut Vec<String>) {
    let p_s = acc
        .precip_by_season
        .get(&Season::Summer)
        .copied()
        .unwrap_or((0.0, 0));
    let p_w = acc
        .precip_by_season
        .get(&Season::Winter)
        .copied()
        .unwrap_or((0.0, 0));
    let precip_summer_tick = p_s.0 / f64::from(p_s.1.max(1));
    let precip_winter_tick = p_w.0 / f64::from(p_w.1.max(1));
    let ratio = precip_summer_tick / precip_winter_tick.max(1e-6);
    eprintln!(
        "Precip summer={precip_summer_tick:.2} winter={precip_winter_tick:.2} ratio={ratio:.2}"
    );
    if !(0.3..=3.0).contains(&ratio) {
        climatology.push(format!(
            "precip seasonality out of range: summer/winter ratio = {ratio:.2}"
        ));
    }

    let t_s = acc
        .temp_by_season
        .get(&Season::Summer)
        .copied()
        .unwrap_or((0.0, 0));
    let t_w = acc
        .temp_by_season
        .get(&Season::Winter)
        .copied()
        .unwrap_or((0.0, 0));
    let temp_summer = t_s.0 / f64::from(t_s.1.max(1));
    let temp_winter = t_w.0 / f64::from(t_w.1.max(1));
    let delta = temp_summer - temp_winter;
    eprintln!("Temp summer {temp_summer:.1} °C, winter {temp_winter:.1} °C, delta {delta:.1} °C");
    if delta < 10.0 {
        climatology.push(format!(
            "temperature seasonality: summer-winter delta {delta:.1} °C < 10 °C"
        ));
    }

    if let Some(lr) = acc.lapse_rate_first_year {
        let expected = -0.0065_f32;
        let rel_err = ((lr - expected) / expected).abs();
        eprintln!(
            "Lapse rate = {:.5} °C/m (theoretical {:.5}, diff {})",
            lr,
            expected,
            pct(rel_err)
        );
        if rel_err > 0.20 {
            climatology.push(format!("lapse rate diff {} > 20%", pct(rel_err)));
        }
    }

    let win_cov = acc
        .snow_coverage_by_season
        .get(&Season::Winter)
        .cloned()
        .unwrap_or_default();
    let sum_cov = acc
        .snow_coverage_by_season
        .get(&Season::Summer)
        .cloned()
        .unwrap_or_default();
    let win_max = win_cov.iter().copied().max().unwrap_or(0);
    let sum_min = sum_cov.iter().copied().min().unwrap_or(0);
    eprintln!("Snow coverage max winter {win_max}, min summer {sum_min}");
    if f32::from(u16::try_from(win_max).expect("fits u16"))
        <= 5.0 * f32::from(u16::try_from(sum_min).expect("fits u16"))
    {
        climatology.push(format!(
            "snow coverage: max winter {win_max} not > 5x min summer {sum_min}"
        ));
    }
}

fn check_rivers_and_lakes(
    elevations: &HashMap<HexCoord, f32>,
    acc: &SimAccumulators,
    climatology: &mut Vec<String>,
) {
    let low: Vec<HexCoord> = elevations
        .iter()
        .filter(|(_, e)| **e < 300.0)
        .map(|(c, _)| *c)
        .collect();
    let snowy = low
        .iter()
        .copied()
        .filter(|c| acc.snow_ticks_per_cell.get(c).copied().unwrap_or(0) > THRESHOLD_5_TICKS)
        .count();
    eprintln!(
        "Plains: low cells (<300m) = {}, with snow > 5% = {}",
        low.len(),
        snowy
    );
    if snowy > 0 {
        climatology.push(format!(
            "plains: {snowy} cells < 300 m with snow > 5% of ticks"
        ));
    }

    let top10_map = |map: &HashMap<HexCoord, f32>| -> HashSet<HexCoord> {
        let mut v: Vec<(HexCoord, f32)> = map.iter().map(|(c, d)| (*c, *d)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.into_iter().take(10).map(|(c, _)| c).collect()
    };
    let top_first = top10_map(&acc.discharge_year_first);
    let top_last = top10_map(&acc.discharge_year_last);
    let intersection: HashSet<_> = top_first.intersection(&top_last).collect();
    eprintln!("Rivers top 10: intersection = {}/10", intersection.len());
    if intersection.len() < 7 {
        climatology.push(format!(
            "unstable river network: intersection = {}/10 (expected ≥ 7)",
            intersection.len()
        ));
    }

    let lake_perm: HashSet<HexCoord> = acc
        .lake_ticks_year_first
        .iter()
        .filter(|(_, n)| **n >= THRESHOLD_50_YEAR)
        .map(|(c, _)| *c)
        .collect();
    let lake_lost = lake_perm
        .iter()
        .filter(|c| acc.lake_ticks_year_last.get(*c).copied().unwrap_or(0) < THRESHOLD_80_YEAR)
        .count();
    eprintln!(
        "Persistent lakes: {} in first year, {lake_lost} gone in last year",
        lake_perm.len()
    );
    if lake_lost > 0 {
        climatology.push(format!(
            "lakes: {lake_lost}/{} first-year lakes do not persist > 80% in last year",
            lake_perm.len()
        ));
    }
}

fn check_conservation_and_perf(
    water_initial_sim: f32,
    water_final: f32,
    acc: &SimAccumulators,
    must_pass: &mut Vec<String>,
    climatology: &mut Vec<String>,
) {
    let drift =
        (water_final - acc.water_initial_measure).abs() / acc.water_initial_measure.max(1e-6);
    eprintln!(
        "Conservation: init post-warmup {:.1}, final {:.1}, drift {} (init pre-warmup = {:.1})",
        acc.water_initial_measure,
        water_final,
        pct(drift),
        water_initial_sim
    );
    if drift > 0.03 {
        climatology.push(format!(
            "conservation {YEARS} years measured: drift {} > 3%",
            pct(drift)
        ));
    }

    eprintln!(
        "\nms/tick per year: {:?}",
        acc.tick_perf_log
            .iter()
            .map(|(_, m)| format!("{m:.1}"))
            .collect::<Vec<_>>()
    );
    if !acc.tick_perf_log.is_empty() && !cfg!(debug_assertions) {
        let mut ms: Vec<f64> = acc.tick_perf_log.iter().map(|(_, m)| *m).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = ms[ms.len() / 2];
        eprintln!("ms/tick median: {median:.2} (release threshold 15.0)");
        if median > 15.0 {
            must_pass.push(format!("perf: ms/tick median {median:.2} > 15 in release (x{:.1} expected value ~5.7 ms/tick)", median / 5.7));
        }
    }
}

fn run_checks(
    elevations: &HashMap<HexCoord, f32>,
    cell_count: usize,
    water_initial_sim: f32,
    water_final: f32,
    acc: &SimAccumulators,
    must_pass: &mut Vec<String>,
    climatology: &mut Vec<String>,
) {
    eprintln!(
        "\n=== Climatology analysis ({YEARS} years measured, seed {SEED}, {cell_count} cells) ==="
    );
    check_wind_rose(acc, climatology);
    check_seasonality(acc, climatology);
    print_band_diagnostics(elevations, acc);
    check_glaciers(elevations, acc, climatology);
    check_clouds_over_relief(elevations, acc, climatology);
    check_rivers_and_lakes(elevations, acc, climatology);
    check_nappes(acc, climatology);
    check_fronts_secs(acc, climatology);
    check_conservation_and_perf(water_initial_sim, water_final, acc, must_pass, climatology);
}

#[test]
fn scale_ten_year_climatology() {
    let mut timer = PerfTimer::start("scale_ten_year_climatology");
    let mut sim = build_sim();
    timer.lap("setup");

    let cell_count = sim.grid().len();
    let elevations: HashMap<HexCoord, f32> = sim
        .grid()
        .iter()
        .map(|(c, cell)| (*c, cell.elevation))
        .collect();
    let water_initial_sim = total_water(&sim);

    let years_usize = usize::try_from(YEARS).expect("fits usize");
    let mut acc = SimAccumulators::new(years_usize, water_initial_sim);

    run_main_loop(&mut sim, cell_count, &mut acc);
    timer.lap("simulation");
    timer.ticks(TOTAL_TICKS);

    let water_final = total_water(&sim);
    let mut must_pass: Vec<String> = Vec::new();
    let mut climatology: Vec<String> = Vec::new();

    run_checks(
        &elevations,
        cell_count,
        water_initial_sim,
        water_final,
        &acc,
        &mut must_pass,
        &mut climatology,
    );
    timer.lap("analyse");
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
            "{} structural invariant(s) violated in scale_ten_year_climatology",
            must_pass.len()
        );
    }
}

/// Simple linear regression temp = a * elev + b, returns a (°C/m).
fn compute_lapse_rate(grid: &HexGrid) -> f32 {
    let pts: Vec<(f32, f32)> = grid
        .iter()
        .map(|(_, c)| (c.elevation, c.temperature))
        .collect();
    let n_f = f32::from(u16::try_from(pts.len()).expect("fits u16"));
    if n_f < 2.0 {
        return 0.0;
    }
    let mean_x: f32 = pts.iter().map(|(x, _)| x).sum::<f32>() / n_f;
    let mean_y: f32 = pts.iter().map(|(_, y)| y).sum::<f32>() / n_f;
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for (x, y) in &pts {
        num += (x - mean_x) * (y - mean_y);
        den += (x - mean_x).powi(2);
    }
    if den.abs() < 1e-6 { 0.0 } else { num / den }
}
