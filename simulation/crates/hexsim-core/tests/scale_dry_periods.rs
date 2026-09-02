//! `scale_dry_periods`: validates that a given cell experiences real
//! dry periods (the world isn't in permanent drizzle everywhere).
//! Assertion: median longest dry streak per cell ≥ 21 days.
//!
//! HISTORY (2026-07-10): this test ALSO required "≥ 1 high-altitude cell
//! with perennial snow (glacier) over 2 years". Expectation REMOVED: it was
//! a RENDERING scale artifact: the front exaggerated relief ×12, the peaks
//! looked alpine, a glacier seemed plausible. The actual Drôme relief
//! (restored to 1:1, `CELL_SPACING_M=130`) tops out too low to sustain a
//! perennial glacier; the model is RIGHT not to produce one, it wasn't a
//! regression. The snow-accumulation mechanism is still tested under
//! synthetic cold conditions by `glacier_formation.rs` and the unit tests
//! in `snow.rs`. Cf. issue #70 (closed not-planned) and JOURNAL 2026-07-10.
//! The altitude snow diagnostic is still printed (informational), but is no
//! longer asserted.
//!
//! Philosophy `feedback_scale_tests_eval_style`: we write the test like
//! an eval, it can fail for a long time, failures are to-dos, not bugs.

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

const YEARS: u64 = 3;
const WARMUP_DAYS: u64 = 90;
const TOTAL_TICKS: u64 = YEARS * 365;
const RADIUS: i32 = 30;
const SEED: u32 = 42;
const HIGH_ALT_THRESHOLD: f32 = 1000.0;

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

struct EventStats {
    avg_duration: f32,
    n_events: usize,
    n_one_tick: usize,
    pct_flashes: f32,
}

fn analyse_event_durations(rain_history_per_sample: &[Vec<bool>]) -> EventStats {
    let mut all_durations: Vec<usize> = Vec::new();
    for history in rain_history_per_sample {
        let mut cur = 0usize;
        for &b in history {
            if b {
                cur += 1;
            } else if cur > 0 {
                all_durations.push(cur);
                cur = 0;
            }
        }
        if cur > 0 {
            all_durations.push(cur);
        }
    }
    let n_events = all_durations.len();
    let n_one_tick = all_durations.iter().filter(|&&d| d == 1).count();
    let avg_duration = if n_events == 0 {
        0.0
    } else {
        all_durations
            .iter()
            .map(|&d| f32::from(u16::try_from(d).expect("fits u16")))
            .sum::<f32>()
            / f32::from(u16::try_from(n_events).expect("fits u16"))
    };
    let pct_flashes = if n_events > 0 {
        100.0 * f32::from(u16::try_from(n_one_tick).expect("fits u16"))
            / f32::from(u16::try_from(n_events).expect("fits u16"))
    } else {
        0.0
    };
    EventStats {
        avg_duration,
        n_events,
        n_one_tick,
        pct_flashes,
    }
}

struct DryStreakStats {
    max_streak: usize,
    zero_days: usize,
    median_dry_streak: usize,
    p25_dry_streak: usize,
    n_sample: usize,
}

fn analyse_dry_streaks(
    raining_per_tick: &[usize],
    rain_history_per_sample: &[Vec<bool>],
) -> DryStreakStats {
    let mut max_streak = 0usize;
    let mut cur = 0usize;
    for &n in raining_per_tick {
        if n == 0 {
            cur += 1;
            if cur > max_streak {
                max_streak = cur;
            }
        } else {
            cur = 0;
        }
    }
    let zero_days = raining_per_tick.iter().filter(|&&n| n == 0).count();

    let mut max_dry_per_cell: Vec<usize> = rain_history_per_sample
        .iter()
        .map(|history| {
            let mut max_dry = 0;
            let mut streak = 0;
            for &raining in history {
                if raining {
                    streak = 0;
                } else {
                    streak += 1;
                    if streak > max_dry {
                        max_dry = streak;
                    }
                }
            }
            max_dry
        })
        .collect();
    max_dry_per_cell.sort_unstable();
    let n_sample = max_dry_per_cell.len();
    let median_dry_streak = if n_sample > 0 {
        max_dry_per_cell[n_sample / 2]
    } else {
        0
    };
    let p25_dry_streak = if n_sample > 0 {
        max_dry_per_cell[n_sample / 4]
    } else {
        0
    };

    DryStreakStats {
        max_streak,
        zero_days,
        median_dry_streak,
        p25_dry_streak,
        n_sample,
    }
}

fn print_high_alt_diag(sim: &Simulation, high_coords: &[HexCoord]) {
    if high_coords.is_empty() {
        return;
    }
    let n_f = f32::from(u16::try_from(high_coords.len()).expect("fits u16"));
    let avg_elev = high_coords
        .iter()
        .filter_map(|c| sim.grid().get(*c))
        .map(|c| c.elevation)
        .sum::<f32>()
        / n_f;
    let avg_temp = high_coords
        .iter()
        .filter_map(|c| sim.grid().get(*c))
        .map(|c| c.temperature)
        .sum::<f32>()
        / n_f;
    let avg_hum = high_coords
        .iter()
        .filter_map(|c| sim.grid().get(*c))
        .map(|c| c.humidity_upper)
        .sum::<f32>()
        / n_f;
    let snow_now_sum: f32 = high_coords
        .iter()
        .filter_map(|c| sim.grid().get(*c))
        .map(|c| c.snow_level)
        .sum();
    eprintln!(
        "  Haute-alt instant final: elev_mean={avg_elev:.0} temp_mean={avg_temp:.1} \
         hum_up_mean={avg_hum:.3} snow_now_sum={snow_now_sum:.2}"
    );
}

struct SimData {
    raining_per_tick: Vec<usize>,
    snow_total_per_tick: Vec<f32>,
    rain_history_per_sample: Vec<Vec<bool>>,
    min_snow_last_year: Vec<f32>,
    high_coords: Vec<HexCoord>,
}

fn run_sim_loop(sim: &mut Simulation) -> SimData {
    let total_ticks_usize = usize::try_from(TOTAL_TICKS).expect("fits usize");
    let mut raining_per_tick: Vec<usize> = Vec::with_capacity(total_ticks_usize);
    let mut snow_total_per_tick: Vec<f32> = Vec::with_capacity(total_ticks_usize);

    let sample_coords: Vec<HexCoord> = sim
        .grid()
        .iter()
        .filter(|(_, c)| c.elevation > 50.0 && c.elevation < 600.0)
        .map(|(c, _)| *c)
        .take(200)
        .collect();
    let mut rain_history_per_sample: Vec<Vec<bool>> =
        vec![Vec::with_capacity(total_ticks_usize); sample_coords.len()];

    let high_coords: Vec<HexCoord> = sim
        .grid()
        .iter()
        .filter(|(_, c)| c.elevation > HIGH_ALT_THRESHOLD)
        .map(|(c, _)| *c)
        .collect();
    let mut min_snow_last_year: Vec<f32> = vec![f32::INFINITY; high_coords.len()];
    let last_year_start = TOTAL_TICKS.saturating_sub(365);

    for tick in 0..TOTAL_TICKS {
        sim.step();
        let diag = sim.diagnostics();
        raining_per_tick.push(diag.hydrology.raining_cells);
        snow_total_per_tick.push(diag.snow.total);

        for (i, coord) in sample_coords.iter().enumerate() {
            let precipitated = sim
                .grid()
                .cell_index(*coord)
                .and_then(|idx| sim.last_precipitation().get(idx))
                .is_some_and(|d| d.rain > 1e-5);
            rain_history_per_sample[i].push(precipitated);
        }

        if tick >= last_year_start {
            for (i, coord) in high_coords.iter().enumerate() {
                if let Some(cell) = sim.grid().get(*coord)
                    && cell.snow_level < min_snow_last_year[i]
                {
                    min_snow_last_year[i] = cell.snow_level;
                }
            }
        }
    }

    SimData {
        raining_per_tick,
        snow_total_per_tick,
        rain_history_per_sample,
        min_snow_last_year,
        high_coords,
    }
}

fn print_run_diagnostics(
    sim: &Simulation,
    data: &SimData,
    events: &EventStats,
    dry: &DryStreakStats,
    perennial_high: usize,
    max_snow_ever_high: f32,
) {
    let min_raining = *data.raining_per_tick.iter().min().unwrap_or(&0);
    let max_raining = *data.raining_per_tick.iter().max().unwrap_or(&0);
    let mean_raining = data
        .raining_per_tick
        .iter()
        .map(|&n| f32::from(u16::try_from(n).expect("fits u16")))
        .sum::<f32>()
        / f32::from(u16::try_from(TOTAL_TICKS).expect("fits u16"));
    let snow_total_final = data.snow_total_per_tick.last().copied().unwrap_or(0.0);
    let snow_total_max = data
        .snow_total_per_tick
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let mid = data.snow_total_per_tick.len() / 2;
    let snow_total_min_late = data.snow_total_per_tick[mid..]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    eprintln!("\n=== scale_dry_periods (seed 42, {YEARS} years, warmup {WARMUP_DAYS}j) ===");
    eprintln!("Raining cells: min={min_raining} max={max_raining} mean={mean_raining:.0}");
    eprintln!(
        "Zero-rain days (everywhere): {}/{TOTAL_TICKS}",
        dry.zero_days
    );
    eprintln!("MAX zero-rain streak (global): {} days", dry.max_streak);
    eprintln!(
        "Max dry streak PER CELL (sample {}): p25={}d median={}d (target median ≥ 21)",
        dry.n_sample, dry.p25_dry_streak, dry.median_dry_streak
    );
    eprintln!(
        "Showers: {} events, avg duration={:.1} ticks, 1-tick flashes = {} ({:.1}%)",
        events.n_events, events.avg_duration, events.n_one_tick, events.pct_flashes
    );
    eprintln!(
        "Snow total: max={snow_total_max:.0} final={snow_total_final:.0} \
         min-after-1-summer={snow_total_min_late:.1}"
    );
    eprintln!(
        "Perennial cells >{HIGH_ALT_THRESHOLD:.0}m (snow>100mm always): {perennial_high}/{} \
         (max snow-always-held = {max_snow_ever_high:.2})",
        data.high_coords.len()
    );
    print_high_alt_diag(sim, &data.high_coords);
    print_snow_by_elevation(sim);
}

fn print_snow_by_elevation(sim: &Simulation) {
    for thr in [500.0_f32, 800.0, 1000.0, 1200.0] {
        let cs: Vec<HexCoord> = sim
            .grid()
            .iter()
            .filter(|(_, c)| c.elevation > thr)
            .map(|(c, _)| *c)
            .collect();
        let snow_now: Vec<f32> = cs
            .iter()
            .filter_map(|c| sim.grid().get(*c))
            .map(|c| c.snow_level)
            .collect();
        // Phase 3 (#32) : rescale ×200 (0.5 → 100 mm), snow_level en mm maintenant.
        let with_snow = snow_now.iter().filter(|&&s| s > 100.0).count();
        let total_snow: f32 = snow_now.iter().sum();
        eprintln!(
            "  >{thr:.0}m: {}/{} cells with snow>100mm (total={total_snow:.1})",
            with_snow,
            cs.len()
        );
    }
}

fn check_and_panic(dry: &DryStreakStats) {
    let mut failures: Vec<String> = Vec::new();
    if dry.median_dry_streak < 21 {
        failures.push(format!(
            "median max dry-streak per cell = {}d < 21 \
             (p25={}d), rain never stops long enough on a given cell",
            dry.median_dry_streak, dry.p25_dry_streak
        ));
    }
    // NB: the "perennial snow / glacier" assertion was removed (rendering
    // scale artifact ×12, cf. the module doc-comment). `perennial_high` is
    // still computed and printed as a diagnostic, but is no longer blocking.
    if !failures.is_empty() {
        eprintln!("\n=== Failures ({}) ===", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        panic!("{} scale_dry_periods check(s) failed", failures.len());
    }
}

#[test]
fn dry_periods_exist() {
    let mut sim = build_sim();
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    let data = run_sim_loop(&mut sim);
    let events = analyse_event_durations(&data.rain_history_per_sample);
    let dry = analyse_dry_streaks(&data.raining_per_tick, &data.rain_history_per_sample);

    // Phase 3 (#32) : rescale ×200 (0.5 → 100 mm), snow_level en mm maintenant.
    let perennial_high = data
        .min_snow_last_year
        .iter()
        .filter(|&&m| m > 100.0)
        .count();
    let max_snow_ever_high = data
        .min_snow_last_year
        .iter()
        .copied()
        .filter(|m| m.is_finite())
        .fold(0.0f32, f32::max);

    print_run_diagnostics(
        &sim,
        &data,
        &events,
        &dry,
        perennial_high,
        max_snow_ever_high,
    );
    check_and_panic(&dry);
}
