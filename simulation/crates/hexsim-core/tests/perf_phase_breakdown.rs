//! Per-phase instrumentation bench to identify tick hotspots (issue #41).
//! For each top-level phase called in `Simulation::step_hour`, measures
//! the cumulative time over N ticks.
//!
//! Run with: `cargo test --release --test perf_phase_breakdown -- --nocapture`
//! This test has no assertion; it prints a breakdown.
//!
//! ⚠ This bench REPLAYS the tick phase by phase: any phase added to
//! `Simulation::step_hour` must be added HERE too, with the same cadence
//! gate (prod cadences are exposed via
//! `simulation::{wind,synoptic}_subsample_hours`). The cost of forgetting
//! is a silent blind spot: from 2026-07-07 to 2026-07-10 the breakdown
//! ignored the synoptic phase (82% of the tick at r45, A/B via the
//! `synoptic.enabled` param, #88) and recomputed wind every hour when
//! prod subsamples it at 3 h (#89).

use std::time::Instant;

use hexsim_core::atmosphere::{
    AtmoForcing, AtmoScratch, AtmosphereParams, PrecipitationMap, smooth_upper_air_mean_t,
    step_atmosphere_into, surface_means,
};
use hexsim_core::climate::DayRecord;
use hexsim_core::dynamics::{SynopticParams, SynopticState};
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::{GroundwaterParams, step_groundwater};
use hexsim_core::hydro::{
    DischargeMap, EdgeFluxMap, FlowVecMap, FluxMap, HydroParams, step_hydro_mfd_into,
};
use hexsim_core::simulation::{synoptic_subsample_hours, wind_subsample_hours};
use hexsim_core::snow::{SnowForcing, SnowParams, step_snow};
use hexsim_core::synoptic_mesh::SynopticMesh;
use hexsim_core::temperature::{
    SolarBeam, TemperatureForcing, TemperatureParams, compute_illumination, solar_beam_at_tick,
    step_temperature,
};
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::{
    WindField, WindParams, WindVec, compute_wind_field_into, compute_wind_magnitudes_into,
};

const RADIUS: i32 = 30;
const SEED: u32 = 42;
const WARMUP_HOURS: u32 = 365 * 24;
const MEASURE_HOURS: u32 = 30 * 24;

#[derive(Default)]
struct PhaseTimings {
    temp: f64,
    synoptic: f64,
    wind: f64,
    snow: f64,
    atmo: f64,
    gw: f64,
    hydro: f64,
}

fn time_phase(measuring: bool, acc: &mut f64, body: impl FnOnce()) {
    let t = Instant::now();
    body();
    if measuring {
        *acc += t.elapsed().as_secs_f64();
    }
}

fn print_breakdown(t: &PhaseTimings, n: usize) {
    let total = t.temp + t.synoptic + t.wind + t.snow + t.atmo + t.gw + t.hydro;
    let measure_hours_f = f64::from(MEASURE_HOURS);
    eprintln!(
        "\n=== Phase breakdown over {MEASURE_HOURS} hours = {} days, radius {RADIUS} ({n} cells) ===",
        MEASURE_HOURS / 24
    );
    let row = |name: &str, dt: f64| {
        let pct = if total > 0.0 { 100.0 * dt / total } else { 0.0 };
        let per_hour_ms = 1000.0 * dt / measure_hours_f;
        eprintln!("  {name:<14} {dt:>7.2} s  ({pct:>5.1} %)  {per_hour_ms:>6.3} ms/hour-tick");
    };
    row("temperature", t.temp);
    row("synoptic", t.synoptic);
    row("wind", t.wind);
    row("snow", t.snow);
    row("atmosphere", t.atmo);
    row("groundwater", t.gw);
    row("hydro x8", t.hydro);
    eprintln!("  {:<14} {total:>7.2} s", "TOTAL");
    eprintln!(
        "  ms/hour-tick: {:.3}  ->  ms/day-tick: {:.2}",
        1000.0 * total / measure_hours_f,
        1000.0 * total / (measure_hours_f / 24.0)
    );
}

/// Illumination (#102) + heat budget, counted together in the temp phase.
/// `beam` computed once per tick by the caller (single source of truth,
/// reused afterward by the snow phase; mirrors `Simulation::step_hour`).
fn illum_and_temp(
    current: &HexGrid,
    next: &mut HexGrid,
    p: &AllParams,
    hour_tick: u64,
    beam: &SolarBeam,
    ff: &mut Vec<f32>,
    il: &mut Vec<f32>,
) {
    compute_illumination(current, beam, p.temp.cloud_albedo_coef, 1500.0, ff, il);
    step_temperature(
        current,
        next,
        &p.temp,
        &TemperatureForcing {
            hour_tick,
            flux_factor: ff,
            snow: &p.snow,
        },
    );
}

struct AllParams {
    hydro: HydroParams,
    atmo: AtmosphereParams,
    gw: GroundwaterParams,
    snow: SnowParams,
    temp: TemperatureParams,
    wind: WindParams,
    synoptic: SynopticParams,
}

impl AllParams {
    fn defaults() -> Self {
        let temp = TemperatureParams::default();
        // Mirrors `Simulation::new`: seed + latitude inherited from the world.
        let synoptic = SynopticParams {
            seed: SEED,
            latitude_deg: temp.latitude_deg,
            ..SynopticParams::default()
        };
        Self {
            hydro: HydroParams::default(),
            atmo: AtmosphereParams::default(),
            gw: GroundwaterParams::default(),
            snow: SnowParams::default(),
            temp,
            wind: WindParams {
                seed: SEED,
                ..WindParams::default()
            },
            synoptic,
        }
    }
}

/// Tick scratch buffers, bundled so `run_phases` stays readable (same
/// spirit as `AtmoScratch`); otherwise the list of `let mut` overflows the
/// function past the clippy `too_many_lines` limit.
struct Scratch {
    current: HexGrid,
    next: HexGrid,
    flux_factor: Vec<f32>,
    illum: Vec<f32>,
    wind_field: WindField,
    wind_mag: Vec<f32>,
    wind_snap: WindField,
    atmo: AtmoScratch,
    hydro_flux: FluxMap,
    hydro_flow_vec: FlowVecMap,
    hydro_edge_flux: EdgeFluxMap,
    precip_tick: PrecipitationMap,
    discharge_map: DischargeMap,
    flow_vec_map: FlowVecMap,
    precip_gate_open: bool,
    last_precipitation: PrecipitationMap,
    synoptic_mesh: SynopticMesh,
    synoptic_state: SynopticState,
    synoptic_base: WindField,
    synoptic_coarse_base: WindField,
}

impl Scratch {
    fn new(grid: HexGrid, n: usize, synoptic: &mut SynopticParams) -> Self {
        // Mirrors `Simulation::new`: the solver integrates on the coarse
        // torus, the params are recalibrated to its spacing (seed and
        // latitude kept).
        let mut synoptic_mesh = SynopticMesh::build(&grid);
        *synoptic = SynopticParams {
            seed: synoptic.seed,
            latitude_deg: synoptic.latitude_deg,
            ..SynopticParams::for_spacing(synoptic_mesh.spacing_m())
        };
        synoptic_mesh.aggregate_temperature(&grid);
        let n_coarse = synoptic_mesh.grid().len();
        let synoptic_state = SynopticState::new(n_coarse, synoptic);
        let mut synoptic_coarse_base: WindField = vec![WindVec::default(); n_coarse];
        synoptic_state.write_base_wind(synoptic, &mut synoptic_coarse_base);
        let mut synoptic_base: WindField = vec![WindVec::default(); n];
        synoptic_mesh.interpolate_wind(&synoptic_coarse_base, &mut synoptic_base);
        Self {
            next: grid.clone(),
            current: grid,
            flux_factor: vec![0.0; n],
            illum: vec![1.0; n],
            wind_field: vec![WindVec::default(); n],
            wind_mag: Vec::with_capacity(n),
            wind_snap: vec![WindVec::default(); n],
            atmo: AtmoScratch::new(n),
            hydro_flux: vec![0.0; n],
            hydro_flow_vec: vec![(0.0, 0.0); n],
            hydro_edge_flux: vec![[0.0; 6]; n],
            precip_tick: vec![DayRecord::default(); n],
            discharge_map: vec![0.0; n],
            flow_vec_map: vec![(0.0, 0.0); n],
            precip_gate_open: false,
            last_precipitation: vec![DayRecord::default(); n],
            synoptic_mesh,
            synoptic_state,
            synoptic_base,
            synoptic_coarse_base,
        }
    }
}

/// Groundwater + hydro (8 substeps), run once per day (24 ticks).
fn run_daily_phases(
    measuring: bool,
    t: &mut PhaseTimings,
    n: usize,
    p: &AllParams,
    s: &mut Scratch,
) {
    time_phase(measuring, &mut t.gw, || {
        step_groundwater(&s.current, &mut s.next, &p.gw);
        std::mem::swap(&mut s.current, &mut s.next);
    });
    time_phase(measuring, &mut t.hydro, || {
        for _ in 0..8 {
            step_hydro_mfd_into(
                &s.current,
                &mut s.next,
                &p.hydro,
                &mut s.hydro_flux,
                &mut s.hydro_flow_vec,
                &mut s.hydro_edge_flux,
            );
            for i in 0..n {
                s.discharge_map[i] += s.hydro_flux[i];
                s.flow_vec_map[i].0 += s.hydro_flow_vec[i].0;
                s.flow_vec_map[i].1 += s.hydro_flow_vec[i].1;
            }
            std::mem::swap(&mut s.current, &mut s.next);
        }
    });
}

fn run_phases(grid: HexGrid) -> (PhaseTimings, usize) {
    let n = grid.len();
    let mut p = AllParams::defaults();
    let mut s = Scratch::new(grid, n, &mut p.synoptic);
    let p = p;

    let mut hour_tick: u64 = 0;
    // Upper-air anchor (EMA τ = 24 h), same bookkeeping as `Simulation`.
    let mut upper_air_mean_t = surface_means(&s.current).0;
    let mut t = PhaseTimings::default();
    let mut measuring = false;
    for h in 0..u64::from(WARMUP_HOURS + MEASURE_HOURS) {
        if h == u64::from(WARMUP_HOURS) {
            measuring = true;
        }
        if h.is_multiple_of(24) {
            s.last_precipitation.fill(DayRecord::default());
            s.discharge_map.fill(0.0);
            s.flow_vec_map.fill((0.0, 0.0));
        }

        // Tick solar beam: computed once, reused by
        // the illumination/temp below AND by the snow forcing further down
        // (same source of truth as in prod, cf `Simulation::step_hour`).
        let solar = solar_beam_at_tick(&p.temp, hour_tick);
        time_phase(measuring, &mut t.temp, || {
            illum_and_temp(
                &s.current,
                &mut s.next,
                &p,
                hour_tick,
                &solar,
                &mut s.flux_factor,
                &mut s.illum,
            );
            std::mem::swap(&mut s.current, &mut s.next);
        });
        // Mirrors `Simulation::step_hour`: synoptic then wind, each at
        // its prod cadence (subsample #89 / SYNOPTIC_SUBSAMPLE_HOURS).
        time_phase(measuring, &mut t.synoptic, || {
            if hour_tick.is_multiple_of(synoptic_subsample_hours()) {
                s.synoptic_mesh.aggregate_temperature(&s.current);
                s.synoptic_state
                    .step_hour(s.synoptic_mesh.grid(), &p.synoptic);
                s.synoptic_state
                    .write_base_wind(&p.synoptic, &mut s.synoptic_coarse_base);
                s.synoptic_mesh
                    .interpolate_wind(&s.synoptic_coarse_base, &mut s.synoptic_base);
            }
        });
        time_phase(measuring, &mut t.wind, || {
            if hour_tick.is_multiple_of(wind_subsample_hours()) {
                compute_wind_field_into(
                    &s.current,
                    &p.wind,
                    hour_tick,
                    &mut s.wind_field,
                    &mut s.wind_snap,
                    Some(&s.synoptic_base),
                );
                compute_wind_magnitudes_into(&s.wind_field, &mut s.wind_mag);
            }
        });
        time_phase(measuring, &mut t.snow, || {
            // Minimal realistic forcing (#60 Phase 1): reuses the buffers
            // already computed this hour by the temp/wind phases (same
            // source of truth as in prod). `s.precip_tick` still carries
            // the PREVIOUS tick's precipitation here; the atmo phase,
            // which overwrites it, runs afterward (mirrors the one-tick
            // lag documented in `snow.rs`).
            step_snow(
                &s.current,
                &mut s.next,
                &p.snow,
                &SnowForcing {
                    beam_w_m2: solar.beam,
                    ground_albedo: p.temp.ground_albedo,
                    flux_factor: &s.flux_factor,
                    wind_mag: &s.wind_mag,
                    rain_last_tick: &s.precip_tick,
                    gw_max_capacity: p.gw.max_capacity,
                },
            );
            std::mem::swap(&mut s.current, &mut s.next);
        });
        upper_air_mean_t = smooth_upper_air_mean_t(upper_air_mean_t, surface_means(&s.current).0);
        time_phase(measuring, &mut t.atmo, || {
            step_atmosphere_into(
                &s.current,
                &mut s.next,
                &p.atmo,
                &AtmoForcing {
                    temp_params: &p.temp,
                    wind_params: &p.wind,
                    wind_field: &s.wind_field,
                    wind_mag: &s.wind_mag,
                    hour_tick,
                    upper_air_mean_t,
                },
                &mut s.precip_gate_open,
                &mut s.atmo,
                &mut s.precip_tick,
            );
            std::mem::swap(&mut s.current, &mut s.next);
        });

        for (total, tick) in s.last_precipitation.iter_mut().zip(s.precip_tick.iter()) {
            total.rain += tick.rain;
            total.snow += tick.snow;
        }
        hour_tick += 1;

        if hour_tick.is_multiple_of(24) {
            run_daily_phases(measuring, &mut t, n, &p, &mut s);
        }
    }
    (t, n)
}

// Benchmark, not a test: only measures time, asserts nothing. Outside
// the default suite (like diag_*); run via `just perf-phases`.
#[test]
#[ignore = "benchmark (~30 s release) — just perf-phases"]
fn perf_phase_breakdown() {
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: SEED,
            ..TerrainParams::default()
        },
    );
    let (t, n) = run_phases(grid);
    print_breakdown(&t, n);
}
