//! Pure diagnostic (read-only, no `assert`): does the orographic rain
//! shadow exist, and does precipitation respond to the uplift trigger
//! `w`?
//!
//! Context (session 2026-07-12): `cloud_water` advection moved to a
//! fraction of the physical Courant number
//! (`atmosphere::advect_cloud_water_into`), clouds now travel ~1 cell
//! per transport pass, versus a much slower regime before this work.
//! Observed consequence over 10 years:
//! `snow_max` (300-800 m band) dropped from 171 to 41 mm (x4), clouds
//! seem to overshoot the mountains before they've had time to
//! precipitate on them. The model already has a physics piece meant to
//! compensate for this: an uplift velocity `w = H·(−∇·v) + v·∇z` that
//! modulates precipitation efficiency via
//! `clamp(updraft_floor + w/updraft_ref_ms, 0, 1)` (see the doc comment
//! on `AtmosphereParams::updraft_ref_ms` in `atmosphere.rs`).
//!
//! This file changes NEITHER default parameters NOR model logic: it
//! builds test-local sims with *test-only* overrides (same technique as
//! `diag_wind_rain_distribution.rs` with its `DIAG_W_REF` /
//! `DIAG_FLOOR` env vars, or `phys_lake_feeds_mountain.rs` with its
//! pre-filled lake) to observe the shipped model's behavior, and to
//! probe what would happen if the already-coded trigger were activated.
//!
//! Does not duplicate:
//! - `diag_wind_rain_distribution.rs` (elevation bands on a procedural
//!   radius-60 map, no controlled ridge and no explicit
//!   upwind/downwind ratio: complementary, not a replacement);
//! - `phys_lake_feeds_mountain.rs` (qualitative lake-to-snow guard rail,
//!   no quantitative measurement of spatial gradient or correlation
//!   with `w`);
//! - `phys_humidity_advection.rs` (pure transport direction on a flat
//!   grid, one-day horizon; here the terrain is NOT flat, that's the
//!   whole point).
//!
//! `#[ignore]` on both tests, run explicitly:
//! `cargo test --release -p hexsim-core --test diag_orographic_precip -- --ignored --nocapture`

// Statistical conversions (cell count -> f64/f32): same justification as
// `diag_wind_rain_distribution.rs` / `bench_metrics.rs` (descriptive-stats
// module over a few thousand observations max, f32/f64 precision is more
// than enough). We avoid `as`: see `count_f64`/`world_x` below
// (`f64::from`/`f32::from` + `try_from` with a loud `.expect()` rather than
// a silent cast).

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::bench_metrics::{AtmosphereParamsOverride, BenchParams, build_bench_sim};
use hexsim_core::cell::CellProperties;
use hexsim_core::coord::HexCoord;
use hexsim_core::dynamics::CELL_SPACING_M;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::{WindParams, WindVec};

// ====================================================================
// Small numeric utilities (no `as`, no silent clamp)
// ====================================================================

/// `usize` -> `f64` without loss: the grid size used here (a few thousand
/// cells max) always fits in a `u32`. A loud `.expect()` rather than an
/// `unwrap_or` that would silently mask a real sizing problem (no
/// defensive clamp/fallback without a physical justification).
fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("cell count fits in u32"))
}

/// Axial coordinate -> world position `x = q + r/2` (unit = cell width,
/// cf `CELL_SPACING_M`), consistent with `hex_direction_to_world`
/// (E = (1,0) -> (1.0, 0.0)); same formula as `synoptic_mesh::axial_to_world`
/// (not pub, duplicated here). `i16`: the radii used here (<= 30) fit with
/// plenty of room, i16->f32 conversion is exact.
fn world_x(c: HexCoord) -> f32 {
    f32::from(i16::try_from(c.q).expect("q fits in i16 for these radii"))
        + 0.5 * f32::from(i16::try_from(c.r).expect("r fits in i16 for these radii"))
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = count_f64(xs.len());
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in xs.iter().zip(ys) {
        let (dx, dy) = (x - mx, y - my);
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    cov / (vx.sqrt() * vy.sqrt()).max(1e-12)
}

/// Index of percentile `p` in a sorted slice of length `len` (rounded
/// "nearest-rank" method). No `f64 as usize` (`cast_possible_truncation`
/// / `cast_sign_loss`): linear search bounded by `len` (a few thousand
/// iterations max, negligible cost for a diagnostic).
fn percentile_index(len: usize, p: f64) -> usize {
    let last = count_f64(len - 1);
    let target = ((p / 100.0) * last).round();
    let mut idx = 0_usize;
    while count_f64(idx) < target && idx + 1 < len {
        idx += 1;
    }
    idx
}

fn mean_where(values: &[f64], keys: &[f64], pred: impl Fn(f64) -> bool) -> f64 {
    let (mut sum, mut cnt) = (0.0_f64, 0_usize);
    for (&v, &k) in values.iter().zip(keys) {
        if pred(k) {
            sum += v;
            cnt += 1;
        }
    }
    if cnt == 0 {
        f64::NAN
    } else {
        sum / count_f64(cnt)
    }
}

// ====================================================================
// Test 1: controlled ridge perpendicular to a uniform wind
// ====================================================================
//
// Geography (x = q + r/2, in cell widths ~CELL_SPACING_M):
//   sea (persistent moisture source, evaporation) ............ x >= 20
//   upwind (low plain, measured) .................... 11 <= x < 20
//   windward slope .................................... 2 < x < 10
//   ridge (measured) .................................. -2 <= x <= 2
//   lee slope ........................................... -10 < x < -2
//   downwind (low plain, measured) .................. -20 < x <= -11
//   hinterland (not measured, just flat terrain) ............ x <= -20
//
// The wind (uniform and deterministic, set via the seam
// `Simulation::set_uniform_wind` (#108, mapping of the old `west_bias`;
// noise/thermal/deflection cut out in `WindParams`, same pattern as
// `phys_humidity_advection.rs`) blows toward -x. Upwind is therefore the
// side the wind comes from (large x), downwind the side it goes toward
// (small x).
//
// The ridge forms a complete diagonal wall (every value of r) across
// the whole toroidal domain: one transport step only changes x by
// 0/±0.5/±1 (hexagonal neighbor), so no shortcut can move an air
// parcel from x>10 to x<-10 without crossing the ridge band; true even
// if toroidal wraparound drifts the trajectory in r over successive
// laps. A single ridge band is enough to isolate upwind from downwind.

const RIDGE_RADIUS: i32 = 25;
const RIDGE_BASE_ELEV_M: f32 = 100.0;
const RIDGE_PEAK_ELEV_M: f32 = 1500.0;
const RIDGE_HALF_WIDTH: f32 = 10.0;
const SEA_ZONE_X: f32 = 20.0;
const SEA_WATER_LEVEL_MM: f32 = 5000.0;
const AMONT_MIN_X: f32 = 11.0;
const AMONT_MAX_X: f32 = 20.0;
const CREST_HALF_WIDTH_X: f32 = 2.0;
/// Magnitude of the old `west_bias`, set via `Simulation::set_uniform_wind`
/// (mapping #108: `west_bias = v` -> `WindVec { x: -v, y: 0.0 }`, wind
/// toward -x, ~5 m/s).
const WEST_BIAS: f32 = 0.5;

/// Ridge elevation at point `x`: triangular ramp, flat on both sides
/// beyond `RIDGE_HALF_WIDTH`.
fn ridge_elevation(x: f32) -> f32 {
    let t = (1.0 - x.abs() / RIDGE_HALF_WIDTH).max(0.0);
    RIDGE_BASE_ELEV_M + (RIDGE_PEAK_ELEV_M - RIDGE_BASE_ELEV_M) * t
}

/// Grid: sea to the east (moisture source via evaporation, same
/// technique as `phys_lake_feeds_mountain.rs` but as a continuous band
/// rather than a single cell), ridge in the center, low plain to the
/// west. No scripted mid-run moisture injection (the `Simulation` API
/// doesn't expose grid mutation while running): the sea feeds the cycle
/// via production evaporation (Meyer), then transported by the scripted
/// wind.
fn build_ridge_grid() -> HexGrid {
    let mut grid = HexGrid::from_radius(RIDGE_RADIUS);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        let x = world_x(coord);
        if let Some(cell) = grid.get_mut(coord) {
            *cell = if x >= SEA_ZONE_X {
                CellProperties {
                    elevation: 0.0,
                    temperature: 15.0,
                    water_level: SEA_WATER_LEVEL_MM,
                    ..CellProperties::default()
                }
            } else {
                CellProperties {
                    elevation: ridge_elevation(x),
                    temperature: 15.0,
                    ..CellProperties::default()
                }
            };
        }
    }
    grid
}

/// "Clean" scripted wind: same pattern as `phys_humidity_advection.rs`
/// (noise/thermal/deflection cut out -> the uniform, deterministic
/// vector is then set on the whole grid via `set_uniform_wind`, cf
/// `build_ridge_sim`).
fn wind_params_scripted() -> WindParams {
    WindParams {
        noise_direction_amplitude: 0.0,
        noise_strength_amplitude: 0.0,
        thermal_strength: 0.0,
        terrain_deflection: 0.0,
        ..WindParams::default()
    }
}

fn build_ridge_sim(atmo: AtmosphereParams) -> Simulation {
    let mut sim = Simulation::new(
        build_ridge_grid(),
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        wind_params_scripted(),
    );
    // `set_uniform_wind` forces the desired uniform wind vector AND
    // automatically disables the synoptic dynamics (otherwise on by
    // default, `HEXSIM_SYNOPTIC` unwrap_or true): it would otherwise
    // silently replace our scripted wind with a seed-dependent
    // geostrophic field. This isolates the pure ridge/uniform-wind
    // response (same reason as `phys_humidity_advection.rs`).
    sim.set_uniform_wind(WindVec {
        x: -WEST_BIAS,
        y: 0.0,
    });
    sim
}

/// Builds the sim, runs `warmup` days (steady regime), then accumulates
/// precipitation (rain+snow) per cell over `measure` days. Reports the
/// per-band averages (upwind/ridge/downwind) and the ratios.
fn run_ridge(label: &str, atmo: AtmosphereParams, warmup: u32, measure: u32) {
    let mut sim = build_ridge_sim(atmo);
    let coords: Vec<HexCoord> = sim.grid().coords_slice().to_vec();

    let mut amont_idx = Vec::new();
    let mut crest_idx = Vec::new();
    let mut aval_idx = Vec::new();
    for (i, &c) in coords.iter().enumerate() {
        let x = world_x(c);
        if (AMONT_MIN_X..AMONT_MAX_X).contains(&x) {
            amont_idx.push(i);
        } else if (-CREST_HALF_WIDTH_X..=CREST_HALF_WIDTH_X).contains(&x) {
            crest_idx.push(i);
        } else if (-AMONT_MAX_X..-AMONT_MIN_X).contains(&x) {
            aval_idx.push(i);
        }
    }

    for _ in 0..warmup {
        sim.step();
    }

    let n = sim.last_precipitation().len();
    let mut cum = vec![0.0_f64; n];
    for _ in 0..measure {
        sim.step();
        for (i, d) in sim.last_precipitation().iter().enumerate() {
            cum[i] += f64::from(d.rain + d.snow);
        }
    }

    let mean_band = |idx: &[usize]| -> f64 {
        if idx.is_empty() {
            f64::NAN
        } else {
            idx.iter().map(|&i| cum[i]).sum::<f64>() / count_f64(idx.len())
        }
    };
    let (amont, crest, aval) = (
        mean_band(&amont_idx),
        mean_band(&crest_idx),
        mean_band(&aval_idx),
    );

    eprintln!(
        "[{label}] precip accumulated over {measure}d (mm, mean/cell): \
         upwind={amont:.2} ({}c)  ridge={crest:.2} ({}c)  downwind={aval:.2} ({}c) \
         | ratio upwind/downwind={:.2}x  upwind/ridge={:.2}x  ridge/downwind={:.2}x",
        amont_idx.len(),
        crest_idx.len(),
        aval_idx.len(),
        amont / aval.max(1e-6),
        amont / crest.max(1e-6),
        crest / aval.max(1e-6),
    );
}

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_orographic_rain_shadow_ridge() {
    eprintln!(
        "\n=== DIAG rain shadow: controlled ridge (radius={RIDGE_RADIUS}, \
         cell~{CELL_SPACING_M}m, half_width={RIDGE_HALF_WIDTH}c, peak={RIDGE_PEAK_ELEV_M}m, \
         base={RIDGE_BASE_ELEV_M}m) ==="
    );
    eprintln!(
        "bands (x = q + r/2): sea x>={SEA_ZONE_X} | upwind [{AMONT_MIN_X},{AMONT_MAX_X}) | \
         ridge [-{CREST_HALF_WIDTH_X},{CREST_HALF_WIDTH_X}] | downwind (-{AMONT_MAX_X},-{AMONT_MIN_X}]"
    );
    eprintln!(
        "wind: set_uniform_wind magnitude={WEST_BIAS:.1} (~5 m/s), uniform, blows toward -x (upwind=large x -> downwind=small x)"
    );

    // Variant A: shipped configuration as-is (uplift trigger OFF by
    // default, cf `AtmosphereParams::default()`).
    run_ridge(
        "A: shipped default (updraft trigger OFF)",
        AtmosphereParams::default(),
        120,
        60,
    );

    // Variant B: test-only exploration. Is the already-coded trigger
    // capable of producing a rain shadow once activated, with plausible
    // values (cf doc comment `updraft_ref_ms`: v·∇z ~1 m/s order of
    // magnitude)? Does NOT change the model default: override local to
    // this test sim only.
    run_ridge(
        "B: trigger ON exploratory (updraft_ref_ms=1.0, updraft_floor=0.1)",
        AtmosphereParams {
            updraft_ref_ms: 1.0,
            updraft_floor: 0.1,
            ..AtmosphereParams::default()
        },
        120,
        60,
    );
}

// ====================================================================
// Test 2: precip x w response on a realistic generated map
// ====================================================================

#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_updraft_precip_correlation_realistic_map() {
    let defaults = AtmosphereParams::default();
    eprintln!("\n=== DIAG default values of the updraft trigger (AtmosphereParams::default()) ===");
    eprintln!(
        "updraft_ref_ms={:.3}  updraft_floor={:.3}  => trigger {}",
        defaults.updraft_ref_ms,
        defaults.updraft_floor,
        if defaults.updraft_ref_ms > 0.0 {
            "ACTIVE"
        } else {
            "INACTIVE: `fill_updraft_into` is never called (gate `updraft_ref_ms > 0.0`), \
             the precip factor is 1 everywhere, w is not even computed"
        }
    );

    let radius: i32 = 30;
    let seed: u32 = 42;
    let warmup: u32 = 150;
    let measure: u32 = 60;

    // The `w` field is only filled if `updraft_ref_ms > 0` (cf
    // `Simulation::updraft_field` doc comment). To observe WHERE the
    // model would make air rise (and whether precip follows), we force
    // the trigger to plausible test-only values: a local override, not a
    // change to the production default (same technique as
    // `diag_wind_rain_distribution.rs` `DIAG_W_REF`/`DIAG_FLOOR`).
    let overrides = BenchParams {
        atmosphere: AtmosphereParamsOverride {
            updraft_ref_ms: Some(1.0),
            updraft_floor: Some(0.1),
            ..AtmosphereParamsOverride::default()
        },
        ..BenchParams::default()
    };
    let (mut sim, _effective) = build_bench_sim(seed, radius, &overrides);

    for _ in 0..warmup {
        sim.step();
    }

    let n = sim.last_precipitation().len();
    eprintln!(
        "  (sanity) updraft_field.len()={} right after warmup (expected {n}, 0 if trigger inactive)",
        sim.updraft_field().len()
    );
    let mut precip_sum = vec![0.0_f64; n];
    let mut w_sum = vec![0.0_f64; n];
    for _ in 0..measure {
        sim.step();
        for (i, d) in sim.last_precipitation().iter().enumerate() {
            precip_sum[i] += f64::from(d.rain + d.snow);
        }
        for (i, &w) in sim.updraft_field().iter().enumerate() {
            w_sum[i] += f64::from(w);
        }
    }

    let days = count_f64(measure as usize);
    let precip_mean: Vec<f64> = precip_sum.iter().map(|&s| s / days).collect();
    let w_mean: Vec<f64> = w_sum.iter().map(|&s| s / days).collect();

    let r = pearson(&precip_mean, &w_mean);

    // Narrow +/-0.05 m/s band: used to characterize the distribution of
    // w, not to build a reliable "flat" bucket (see percentiles below;
    // if this band is nearly empty, it means w is almost never close to
    // 0 over the 60-day average, not a measurement error).
    let narrow_thresh = 0.05_f64;
    let narrow_count = w_mean.iter().filter(|&&w| w.abs() < narrow_thresh).count();
    let narrow_frac = 100.0 * count_f64(narrow_count) / count_f64(n);

    // Wide cutoff, physically significant for the trigger: w<=0 =>
    // clamp(floor + w/ref, 0, floor) <= floor => precip factor CAPPED at
    // the `updraft_floor` floor ("floor-dominated" zone). w>0 => the
    // factor can exceed the floor (net uplift).
    let floor_dominated_count = w_mean.iter().filter(|&&w| w <= 0.0).count();
    let floor_dominated_frac = 100.0 * count_f64(floor_dominated_count) / count_f64(n);
    let precip_on_floor_dominated = mean_where(&precip_mean, &w_mean, |w| w <= 0.0);
    let precip_on_ascending = mean_where(&precip_mean, &w_mean, |w| w > 0.0);

    let mut w_sorted = w_mean.clone();
    w_sorted.sort_by(f64::total_cmp);
    let pct = |p: f64| -> f64 { w_sorted[percentile_index(w_sorted.len(), p)] };

    eprintln!(
        "\n=== DIAG precip x w correlation (generated map radius={radius} seed={seed}, \
         trigger FORCED ON to populate w: updraft_ref_ms=1.0 updraft_floor=0.1) ==="
    );
    eprintln!("n cells = {n}, measured {measure}d after {warmup}d of warmup");
    eprintln!("pearson(precip_mean_mm_per_day, w_mean_m_per_s) = {r:.3}");
    eprintln!(
        "distribution w_mean (m/s): p5={:.3} p25={:.3} p50={:.3} p75={:.3} p95={:.3}",
        pct(5.0),
        pct(25.0),
        pct(50.0),
        pct(75.0),
        pct(95.0)
    );
    eprintln!(
        "cells |w| < {narrow_thresh} m/s (narrow band around 0)         : {narrow_frac:.2}% of the map \
         ({narrow_count} cells out of {n})"
    );
    eprintln!(
        "cells w <= 0 (precip factor capped at floor={:.2})             : {floor_dominated_frac:.1}% of the map, \
         mean precip = {precip_on_floor_dominated:.4} mm/d",
        0.1_f32
    );
    eprintln!(
        "cells w > 0 (net updraft, factor can exceed floor)             : mean precip = {precip_on_ascending:.4} mm/d"
    );
    eprintln!(
        "ratio precip(w>0)/precip(w<=0) = {:.2}x",
        precip_on_ascending / precip_on_floor_dominated.max(1e-6),
    );
}
