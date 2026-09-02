//! `hexsim-bench`, standalone binary to run a simulation for
//! parameter optimization.
//!
//! Reads a JSON of parameters (partial override of the defaults), runs the sim
//! for warmup + measure ticks, computes a set of climate metrics, and
//! writes the result as JSON to stdout or a file.
//!
//! Stateless: each invocation rebuilds a fresh sim, no shared state.
//! Trivially parallelizable by an external orchestrator
//! (`scripts/optim/random_search.py`).
//!
//! Exit codes:
//! - 0: run completed, metrics.status == "ok"
//! - 1: input error (malformed JSON, unknown key, I/O)
//! - 2: run completed but NaN/Inf detected (metrics.status == "`nan_inf`")

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;

use hexsim_core::bench_metrics::{
    BenchParams, EffectiveParams, Metrics, MetricsAccumulator, RunStatus, build_bench_sim,
};

#[derive(Parser, Debug)]
#[command(
    name = "hexsim-bench",
    about = "Standalone simulation run for parameter optimization",
    long_about = None
)]
struct Cli {
    /// JSON of parameter overrides (partial OK, hierarchical by group).
    #[arg(long)]
    params: PathBuf,

    /// Seed for terrain + wind generation.
    #[arg(long)]
    seed: u32,

    /// Radius of the hex grid.
    #[arg(long, default_value_t = 30)]
    radius: i32,

    /// Warmup ticks before the measurement phase (default 1 year).
    #[arg(long, default_value_t = 365)]
    warmup_ticks: u64,

    /// Measurement ticks, duration over which the metrics are computed.
    #[arg(long, default_value_t = 1095)]
    measure_ticks: u64,

    /// Output file. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Serialize)]
struct BenchOutput {
    schema_version: u32,
    seed: u32,
    radius: i32,
    warmup_ticks: u64,
    measure_ticks: u64,
    params_effective: EffectiveParams,
    metrics: Metrics,
    status: RunStatus,
    elapsed_ms: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let params_json = std::fs::read_to_string(&cli.params)
        .with_context(|| format!("reading params file {}", cli.params.display()))?;
    let overrides: BenchParams = serde_json::from_str(&params_json)
        .with_context(|| format!("parsing JSON params {}", cli.params.display()))?;

    let (mut sim, effective) = build_bench_sim(cli.seed, cli.radius, &overrides);

    // Warmup: initial transient, metrics not collected
    for _ in 0..cli.warmup_ticks {
        sim.step();
    }

    let mut acc = MetricsAccumulator::start(&sim, &effective, cli.measure_ticks);
    let t0 = Instant::now();
    for _ in 0..cli.measure_ticks {
        sim.step();
        acc.observe(&sim);
    }
    let elapsed = t0.elapsed();
    let (metrics, status) = acc.finalize(&sim, elapsed);

    let out = BenchOutput {
        schema_version: 1,
        seed: cli.seed,
        radius: cli.radius,
        warmup_ticks: cli.warmup_ticks,
        measure_ticks: cli.measure_ticks,
        params_effective: effective,
        metrics,
        status,
        elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
    };

    let json = serde_json::to_string_pretty(&out)?;
    if let Some(path) = cli.output {
        std::fs::write(&path, &json).with_context(|| format!("ecriture {}", path.display()))?;
    } else {
        println!("{json}");
    }

    // Exit 2 if the run is structurally corrupted (NaN/Inf observed).
    // The Python orchestrator can distinguish this from a plain "bad
    // fitness" and discard the config without flagging it as an input error.
    if status != RunStatus::Ok {
        std::process::exit(2);
    }
    Ok(())
}
