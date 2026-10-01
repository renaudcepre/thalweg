//! Byte-level golden dump of the grid after a short run, the gate every
//! "bit-identical" refactor of the tick is held to (r250 perf effort:
//! chunks B1, B2, the `SoA` layout, and this instrument's own successors).
//!
//! Writes the msgpack encoding of `cells_slice()` to `HEXSIM_GOLDEN_OUT`
//! (the order of the cells is the grid's own, canonical since
//! `HexGrid::from_radius`); two dumps made with the same env are compared
//! with `cmp`. Not a test: it asserts nothing and only runs when
//! `HEXSIM_GOLDEN_OUT` is set, so a forgotten `--ignored` costs nothing.
//!
//! ```text
//! HEXSIM_GOLDEN_OUT=/tmp/before.bin HEXSIM_THREADS=4 \
//!   cargo test --release --test perf_golden_state -- --ignored
//! ```
//!
//! Env: `HEXSIM_PERF_RADIUS` (130, the smallest grid above
//! `par::PAR_MIN_CELLS` so the parallel path is the one exercised),
//! `HEXSIM_PERF_SEED` (42), `HEXSIM_GOLDEN_DAYS` (3),
//! `HEXSIM_GOLDEN_FIRE_EROSION` (unset: fire and erosion at their live
//! defaults; `1`: both forced on, to cover their daily passes),
//! `HEXSIM_THREADS` (the pool size, see `par`).

mod common;

use common::build_prod_sim;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "instrument — HEXSIM_GOLDEN_OUT=path cargo test --release --test perf_golden_state -- --ignored"]
fn perf_golden_state() {
    let Ok(out) = std::env::var("HEXSIM_GOLDEN_OUT") else {
        eprintln!("HEXSIM_GOLDEN_OUT unset: nothing to do");
        return;
    };
    let radius = i32::try_from(env_u64("HEXSIM_PERF_RADIUS", 130)).expect("radius i32");
    let seed = u32::try_from(env_u64("HEXSIM_PERF_SEED", 42)).expect("seed u32");
    let days = env_u64("HEXSIM_GOLDEN_DAYS", 3);
    let fire_erosion = std::env::var("HEXSIM_GOLDEN_FIRE_EROSION").is_ok_and(|v| v == "1");

    let mut sim = build_prod_sim(seed, radius);
    if fire_erosion {
        sim.update_param("fire.enabled", 1.0);
        sim.update_param("erosion.enabled", 1.0);
    }
    for _ in 0..days * 24 {
        sim.step_hour();
    }
    let bytes = rmp_serde::to_vec(sim.grid().cells_slice()).expect("encode cells");
    std::fs::write(&out, &bytes).expect("write golden dump");
    eprintln!(
        "golden r{radius} seed {seed} {days} d ({} cells, fire+erosion {fire_erosion}) -> {out} ({} bytes)",
        sim.grid().len(),
        bytes.len()
    );
}
