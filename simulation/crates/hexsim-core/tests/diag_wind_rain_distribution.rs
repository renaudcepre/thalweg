// Diagnostic tool: statistical conversions (counts ↔ floats,
// percentiles), ubiquitous and benign. Precedent: `scale_climate_lapse_rate`,
// `climate_history` use the same allow for test stats code.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Wind × rain diagnostic, **cell by cell** (drizzle session 2026-07).
//! The 2026-05 triangulation looked at cloud clusters; never at the
//! wind×rain map. Hypothesis to test: `wind_mag` ~0 everywhere → humidity
//! transport is off (`fraction = humidity_advection_rate × wind_mag`) → each
//! cell precipitates its own vapor → widespread local drizzle.
//!
//! Configurable: `DIAG_RADIUS` / `DIAG_WARMUP` / `DIAG_MEASURE` (days), plus
//! the physics knockouts `DIAG_NC` / `DIAG_COND` / `DIAG_ADV`.
//! `cargo test -p hexsim-core --test diag_wind_rain_distribution -- --nocapture`

use hexsim_core::atmosphere::saturation_upper_pw;
use hexsim_core::bench_metrics::{BenchParams, build_bench_sim};

fn env_u<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn envf(k: &str) -> Option<f32> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

fn pct(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        f32::NAN
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// Per-cell quantities accumulated over the measurement window.
struct Data {
    rain: Vec<f32>, // average mm/day
    wind: Vec<f32>, // average magnitude
    is_water: Vec<bool>,
    elev: Vec<f32>,
    rh: Vec<f32>, // RH_upper (humidity_upper / saturation_upper)
    /// Longest run of consecutive "dry" days (< 0.1 mm/day) per
    /// cell over the measurement window; the "deserts persist" criterion
    /// from the synoptic plan (§2: `dry_streak` must stay long despite the
    /// de-concentration of lakes).
    dry_streak: Vec<u32>,
    /// Average total ascent `w` (m/s) per cell over the window; used to
    /// check WHERE the trigger fires (elevation bands). Zero when the
    /// trigger is inactive (the scratch isn't filled).
    conv: Vec<f32>,
}

/// Builds the sim (with `DIAG_NC/COND/ADV` overrides + Phase 3 synoptic:
/// `DIAG_W_REF`/`DIAG_FLOOR`/`DIAG_WCRIT` for the ascent trigger,
/// `DIAG_SYN_FRICTION`/`DIAG_SYN_RELAX` for synoptic damping
/// (a transient regime), and synoptic active by default hardcoded,
/// overridable via `update_param("synoptic.enabled", …)`), warmup, measure,
/// derives the per-cell quantities. Prints the run header.
fn collect() -> Data {
    let radius: i32 = env_u("DIAG_RADIUS", 60);
    let warmup: u32 = env_u("DIAG_WARMUP", 200);
    let measure: u32 = env_u("DIAG_MEASURE", 120);
    let (nc, cond, adv) = (envf("DIAG_NC"), envf("DIAG_COND"), envf("DIAG_ADV"));
    let (w_ref, w_floor, wcrit) = (envf("DIAG_W_REF"), envf("DIAG_FLOOR"), envf("DIAG_WCRIT"));
    let (syn_friction, syn_relax) = (envf("DIAG_SYN_FRICTION"), envf("DIAG_SYN_RELAX"));
    let syn_anom = envf("DIAG_SYN_ANOM");
    // The synoptic is active by default hardcoded (no more env flag
    // `HEXSIM_SYNOPTIC`): this sim never disables it, so the
    // truth is simply `true`; source of truth = prod behavior,
    // not a missing environment variable.
    let synoptic = true;
    let mut params = BenchParams::default();
    params.atmosphere.kk2000_droplet_count = nc;
    params.atmosphere.condensation_rate = cond;
    params.atmosphere.updraft_ref_ms = w_ref;
    params.atmosphere.updraft_floor = w_floor;
    params.atmosphere.precip_crit_mm = wcrit;
    params.wind.humidity_advection_rate = adv;

    let (mut sim, _) = build_bench_sim(42, radius, &params);
    let water_t0 = sim.diagnostics().water_budget.total;
    if let Some(f) = syn_friction {
        assert!(sim.update_param("synoptic.friction_days", f));
    }
    if let Some(r) = syn_relax {
        assert!(sim.update_param("synoptic.relax_days", r));
    }
    if let Some(a) = syn_anom {
        assert!(sim.update_param("synoptic.thermal_anomaly_days", a));
    }
    for _ in 0..warmup {
        sim.step();
    }
    let n = sim.last_precipitation().len();
    let (mut rain_sum, mut wind_sum) = (vec![0.0f64; n], vec![0.0f64; n]);
    let (mut cur_streak, mut max_streak) = (vec![0u32; n], vec![0u32; n]);
    let mut conv_sum = vec![0.0f64; n];
    for _ in 0..measure {
        sim.step();
        for (i, d) in sim.last_precipitation().iter().enumerate() {
            let day = d.rain + d.snow;
            rain_sum[i] += f64::from(day);
            if day < 0.1 {
                cur_streak[i] += 1;
                max_streak[i] = max_streak[i].max(cur_streak[i]);
            } else {
                cur_streak[i] = 0;
            }
        }
        for (i, w) in sim.wind_field().iter().enumerate() {
            wind_sum[i] += f64::from((w.x * w.x + w.y * w.y).sqrt());
        }
        let divfield = sim.updraft_field();
        if divfield.len() == n {
            for (s, &c) in conv_sum.iter_mut().zip(divfield) {
                *s += f64::from(c);
            }
        }
    }
    let snap = sim.snapshot();
    let water_t1 = sim.diagnostics().water_budget.total;
    let m = f64::from(measure);
    println!(
        "\n=== DIAG wind×rain  radius={radius}  n={n}  warmup={warmup}j measure={measure}j ==="
    );
    println!(
        "(overrides: N_c={nc:?} cond={cond:?} adv={adv:?} | synoptic={synoptic} w_ref={w_ref:?} floor={w_floor:?} W_crit={wcrit:?} syn_friction={syn_friction:?} syn_relax={syn_relax:?} syn_anom={syn_anom:?})"
    );
    println!(
        "water conservation (closed terrarium): total {water_t0:.1} → {water_t1:.1} (drift {:+.3} = {:+.4}%)",
        water_t1 - water_t0,
        100.0 * (water_t1 - water_t0) / water_t0.max(1e-6)
    );
    Data {
        rain: rain_sum.iter().map(|s| (s / m) as f32).collect(),
        wind: wind_sum.iter().map(|s| (s / m) as f32).collect(),
        is_water: snap.cells.iter().map(|c| c.is_open_water).collect(),
        elev: snap.cells.iter().map(|c| c.elevation).collect(),
        // t_offset = lapse_rate(6.5) × 1500/1000 = 9.75.
        rh: snap
            .cells
            .iter()
            .map(|c| c.humidity_upper / saturation_upper_pw(c.temperature - 9.75, 1500.0).max(1e-6))
            .collect(),
        dry_streak: max_streak,
        conv: conv_sum.iter().map(|s| (s / m) as f32).collect(),
    }
}

fn sorted(v: &[f32]) -> Vec<f32> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

// The only diag_* that wasn't #[ignore]: it ran in the suite (254 s
// opt!) and slipped past `just diag-tools` (--run-ignored=only). Aligned.
#[test]
#[ignore = "exploratory diagnostic, run with --ignored --nocapture"]
fn diag_wind_rain_distribution() {
    let d = collect();
    let n = d.rain.len();
    let (rs, ws, rhs) = (sorted(&d.rain), sorted(&d.wind), sorted(&d.rh));

    println!(
        "rain/j (mm): p10={:.2} p50={:.2} p90={:.2} p99={:.2} max={:.2} mean={:.2}",
        pct(&rs, 10.0),
        pct(&rs, 50.0),
        pct(&rs, 90.0),
        pct(&rs, 99.0),
        pct(&rs, 100.0),
        mean(&d.rain)
    );
    println!(
        "wind_mag   : p10={:.3} p50={:.3} p90={:.3} max={:.3} mean={:.3}",
        pct(&ws, 10.0),
        pct(&ws, 50.0),
        pct(&ws, 90.0),
        pct(&ws, 100.0),
        mean(&d.wind)
    );
    let f_sat = d.rh.iter().filter(|&&r| r > 0.9).count() as f32 / n as f32;
    println!(
        "RH_upper   : p10={:.2} p50={:.2} p90={:.2} mean={:.2}   cells RH>0.9: {:.0}%",
        pct(&rhs, 10.0),
        pct(&rhs, 50.0),
        pct(&rhs, 90.0),
        mean(&d.rh),
        100.0 * f_sat
    );
    // 1e-3 = display threshold of the front overlay (PRECIP_MIN): a lot of
    // cells > 1e-3 but few > 0.1 ⇒ "rains everywhere" = rendering artifact.
    let frac = |t: f32| d.rain.iter().filter(|&&r| r > t).count() as f32 / n as f32;
    println!(
        "cells rain/j : >1e-3(overlay)={:.0}%  >0.01={:.0}%  >0.1={:.0}%  >1mm={:.0}%   open water={:.0}%",
        100.0 * frac(1e-3),
        100.0 * frac(0.01),
        100.0 * frac(0.1),
        100.0 * frac(1.0),
        100.0 * d.is_water.iter().filter(|&&w| w).count() as f32 / n as f32
    );

    println!("\nrain/j by wind quartile (transport test):");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d.wind[a].partial_cmp(&d.wind[b]).unwrap());
    for q in 0..4 {
        let s = &idx[q * n / 4..(q + 1) * n / 4];
        let mr = s.iter().map(|&i| d.rain[i]).sum::<f32>() / s.len() as f32;
        let mw = s.iter().map(|&i| d.wind[i]).sum::<f32>() / s.len() as f32;
        println!("  Q{} wind~{mw:.3} : rain/j={mr:.2} mm", q + 1);
    }

    let wr: Vec<f32> = (0..n)
        .filter(|&i| d.is_water[i])
        .map(|i| d.rain[i])
        .collect();
    let lr: Vec<f32> = (0..n)
        .filter(|&i| !d.is_water[i])
        .map(|i| d.rain[i])
        .collect();
    println!(
        "\nrain/j open water: {:.2} ({} cells)  |  land: {:.2} ({} cells)  |  ratio: {:.1}x",
        mean(&wr),
        wr.len(),
        mean(&lr),
        lr.len(),
        mean(&wr) / mean(&lr).max(1e-6)
    );

    // Deserts (synoptic plan §2): lake de-concentration must not
    // drown out the dry zones; the longest dry streak per cell must
    // stay long for a significant fraction of the domain.
    let streaks = sorted(&d.dry_streak.iter().map(|&s| s as f32).collect::<Vec<_>>());
    let deserts = d.dry_streak.iter().filter(|&&s| s >= 30).count() as f32 / n as f32;
    let window: f32 = envf("DIAG_MEASURE").unwrap_or(120.0);
    println!(
        "dry_streak (j, window {window:.0}j) : p10={:.0} p50={:.0} p90={:.0} max={:.0}   cells streak>=30j: {:.0}%",
        pct(&streaks, 10.0),
        pct(&streaks, 50.0),
        pct(&streaks, 90.0),
        pct(&streaks, 100.0),
        100.0 * deserts
    );

    println!("\nrain/j & wind by elevation band:");
    for (lo, hi, lab) in [
        (-9999.0, 0.0, "<0m"),
        (0.0, 300.0, "0-300"),
        (300.0, 800.0, "300-800"),
        (800.0, 1500.0, "800-1500"),
        (1500.0, 9999.0, ">1500"),
    ] {
        let band: Vec<usize> = (0..n)
            .filter(|&i| d.elev[i] >= lo && d.elev[i] < hi)
            .collect();
        let mr = mean(&band.iter().map(|&i| d.rain[i]).collect::<Vec<_>>());
        let mw = mean(&band.iter().map(|&i| d.wind[i]).collect::<Vec<_>>());
        let mc = mean(&band.iter().map(|&i| d.conv[i]).collect::<Vec<_>>());
        println!(
            "  {lab:<9} rain/j={mr:.2}  wind={mw:.3}  w={mc:+.3} m/s  ({} cells)",
            band.len()
        );
    }
    println!();
}
