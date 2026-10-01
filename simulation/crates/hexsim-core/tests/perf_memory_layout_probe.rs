//! Memory-layout probe for the r250 effort (JOURNAL 2026-09-05): how much
//! of an hourly tick is the cost of streaming the whole ~88-byte
//! `CellProperties` in and out of every pass, against the cost of touching
//! only the fields a pass actually writes.
//!
//! Three layouts run the same synthetic "day": `PASSES_PER_HOUR` passes per
//! hour, 24 hours, each pass reading one field of every cell, writing it
//! back slightly changed into the other buffer, and copying the rest of the
//! cell's state as the double buffer requires. No physics: the arithmetic
//! is one multiply-add per cell, so the wall-clock is the memory traffic.
//!
//! - `aos`: `Vec<CellProperties>` × 2, the production layout; a pass streams
//!   two full cells per cell (read `current`, write `next`).
//! - `hot_cold`: the seven hourly-written fields in a 28-byte struct × 2,
//!   the rest in one shared cold vector never touched by the pass.
//! - `soa`: one `Vec<f32>` per field × 2; a pass streams the 8 bytes of the
//!   field it touches.
//!
//! Threads follow `HEXSIM_THREADS` like the engine (one region per worker);
//! `HEXSIM_PERF_RADIUS` sets the grid (default 250).
//!
//! ```text
//! HEXSIM_THREADS=10 HEXSIM_PERF_RADIUS=250 cargo test --release -p hexsim-core \
//!     --test perf_memory_layout_probe -- --ignored --nocapture
//! ```

use std::time::Instant;

use hexsim_core::cell::CellProperties;
use hexsim_core::grid::HexGrid;
use rayon::prelude::*;

const PASSES_PER_HOUR: usize = 25;
const HOURS: usize = 24;
const CELL_BYTES: usize = std::mem::size_of::<CellProperties>();

#[derive(Clone, Copy, Default)]
struct HotCell {
    temperature: f32,
    water_level: f32,
    humidity_surface: f32,
    humidity_upper: f32,
    cloud_water: f32,
    groundwater: f32,
    snow_level: f32,
}

impl HotCell {
    fn checksum(&self) -> f32 {
        self.temperature
            + self.water_level
            + self.humidity_surface
            + self.humidity_upper
            + self.cloud_water
            + self.groundwater
            + self.snow_level
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn pool() -> rayon::ThreadPool {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(n) = std::env::var("HEXSIM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        builder = builder.num_threads(n);
    }
    builder.build().expect("probe thread pool")
}

fn as_f64(x: usize) -> f64 {
    f64::from(u32::try_from(x).unwrap_or(u32::MAX))
}

fn region_len(n: usize) -> usize {
    n.div_ceil(rayon::current_num_threads())
}

fn pass_aos(cur: &[CellProperties], next: &mut [CellProperties]) {
    let len = region_len(cur.len());
    next.par_chunks_mut(len)
        .zip(cur.par_chunks(len))
        .for_each(|(out, inp)| {
            for (nc, c) in out.iter_mut().zip(inp) {
                *nc = c.clone();
                nc.temperature = c.temperature * 0.999 + 0.01;
            }
        });
}

fn pass_hot(cur: &[HotCell], next: &mut [HotCell]) {
    let len = region_len(cur.len());
    next.par_chunks_mut(len)
        .zip(cur.par_chunks(len))
        .for_each(|(out, inp)| {
            for (nc, c) in out.iter_mut().zip(inp) {
                *nc = *c;
                nc.temperature = c.temperature * 0.999 + 0.01;
            }
        });
}

fn pass_soa(cur: &[f32], next: &mut [f32]) {
    let len = region_len(cur.len());
    next.par_chunks_mut(len)
        .zip(cur.par_chunks(len))
        .for_each(|(out, inp)| {
            for (n, c) in out.iter_mut().zip(inp) {
                *n = c * 0.999 + 0.01;
            }
        });
}

fn run_day<S>(mut a: S, mut b: S, mut pass: impl FnMut(&S, &mut S)) -> (f64, S) {
    let t0 = Instant::now();
    for _ in 0..HOURS * PASSES_PER_HOUR {
        pass(&a, &mut b);
        std::mem::swap(&mut a, &mut b);
    }
    (t0.elapsed().as_secs_f64() * 1000.0, a)
}

fn report(label: &str, cells: usize, bytes_per_cell_pass: usize, ms_per_day: f64, checksum: f32) {
    let passes = as_f64(HOURS * PASSES_PER_HOUR);
    let ms_per_h_tick = ms_per_day / as_f64(HOURS);
    let gb_s = as_f64(cells) * as_f64(bytes_per_cell_pass) * passes / (ms_per_day / 1000.0) / 1e9;
    eprintln!(
        "  {label:9} {bytes_per_cell_pass:4} B/cell/pass  {ms_per_day:8.1} ms/day  \
         {ms_per_h_tick:6.3} ms/h-tick  {gb_s:6.1} GB/s streamed  (checksum {checksum:.3e})"
    );
}

#[test]
#[ignore = "probe, not a test: cargo test --release --test perf_memory_layout_probe -- --ignored --nocapture"]
fn perf_memory_layout_probe() {
    let radius = i32::try_from(env_usize("HEXSIM_PERF_RADIUS", 250)).expect("radius");
    let grid = HexGrid::from_radius(radius);
    let n = grid.len();
    let pool = pool();
    let threads = pool.current_num_threads();
    eprintln!(
        "memory layout probe: {n} cells (radius {radius}), {threads} threads, \
         {PASSES_PER_HOUR} passes/hour × {HOURS} h, CellProperties = {CELL_BYTES} B"
    );

    let aos_a: Vec<CellProperties> = grid.cells_slice().to_vec();
    let aos_b = aos_a.clone();
    let hot_a = vec![HotCell::default(); n];
    let hot_b = hot_a.clone();
    let soa_a = vec![1.0_f32; n];
    let soa_b = soa_a.clone();

    pool.install(|| {
        let (ms, out) = run_day(aos_a, aos_b, |a, b| pass_aos(a, b));
        let sum: f32 = out.iter().map(|c| c.temperature).sum();
        report("aos", n, 2 * CELL_BYTES, ms, sum);
        let (ms, out) = run_day(hot_a, hot_b, |a, b| pass_hot(a, b));
        let sum: f32 = out.iter().map(HotCell::checksum).sum();
        report("hot_cold", n, 2 * std::mem::size_of::<HotCell>(), ms, sum);
        let (ms, out) = run_day(soa_a, soa_b, |a, b| pass_soa(a, b));
        let sum: f32 = out.iter().sum();
        report("soa", n, 2 * std::mem::size_of::<f32>(), ms, sum);
    });
}
