//! Pure perf measurement on a radius-60 grid (~10,981 cells).
//!
//! Goal: characterize the per-tick cost at this scale to judge the
//! feasibility of real-time rendering + simulation. Two years of
//! simulation are enough to warm up the cache and stabilize the measurement.

mod common;

use std::time::Instant;

use common::build_prod_sim;

const RADIUS: i32 = 45;
const SEED: u32 = 42;
const TICKS: u64 = 730;
/// I/O: cost of a snapshot + JSON serialization (what goes over the WS
/// on every frame on the CLI side). Averaged over 50 iterations to smooth.
const IO_ITERS: u32 = 50;

// Benchmark, not a test: only measures time, asserts nothing. Outside
// the default suite (like diag_*), run via `just perf-scale`.
#[test]
#[ignore = "benchmark (2 years, ~11k cells), just perf-scale"]
fn scale_perf_radius_60() {
    let setup_start = Instant::now();
    let mut sim = build_prod_sim(SEED, RADIUS);
    let cells = sim.grid().len();
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Setup: {cells} cells (radius {RADIUS}) in {setup_ms:.0} ms");

    let run_start = Instant::now();
    for _ in 0..TICKS {
        sim.step();
    }
    let elapsed = run_start.elapsed().as_secs_f64();
    let ticks_f = f64::from(u32::try_from(TICKS).unwrap_or(u32::MAX));
    let ticks_per_sec = ticks_f / elapsed;
    let ms_per_tick = (elapsed * 1000.0) / ticks_f;

    eprintln!(
        "Simulation: {TICKS} ticks in {elapsed:.2} s -> {ticks_per_sec:.1} ticks/s \
         ({ms_per_tick:.2} ms/tick)"
    );

    let mut snap_total_ms = 0.0_f64;
    let mut json_total_ms = 0.0_f64;
    let mut last_json_bytes = 0_usize;
    for _ in 0..IO_ITERS {
        let t0 = Instant::now();
        let state = sim.snapshot();
        snap_total_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let json = serde_json::to_string(&state).expect("serde_json");
        json_total_ms += t1.elapsed().as_secs_f64() * 1000.0;
        last_json_bytes = json.len();
    }
    let snap_ms = snap_total_ms / f64::from(IO_ITERS);
    let json_ms = json_total_ms / f64::from(IO_ITERS);
    let total_ms = snap_ms + json_ms;
    let json_kb = f64::from(u32::try_from(last_json_bytes).unwrap_or(u32::MAX)) / 1024.0;
    eprintln!(
        "I/O : snapshot {snap_ms:.2} ms, serialize JSON {json_ms:.2} ms \
         (total {total_ms:.2} ms), payload {json_kb:.1} KiB \
         -> {:.1} frames/s max",
        1000.0 / total_ms.max(1e-6)
    );
}
