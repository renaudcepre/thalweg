#!/usr/bin/env python3
"""Random search over the atmo/hydro/temp/wind/ground parameter space.

Runs `hexsim-bench` N times with configurations drawn uniformly from
the search space defined in `search_space.toml`. Each configuration is
evaluated across several seeds, results are aggregated (mean), sorted
by fitness and saved as CSV + JSONL.

Why random search and not Bayesian/genetic: this is the *foundation*
step. Once the params JSON -> metrics + fitness pipeline runs, swapping
the sampler for `optuna` or `cma-es` is a 50-line fork. We first validate
that the fitness captures what we want.

Usage (from the repo root):

    python scripts/optim/random_search.py --n 200 --workers 8

Outputs:
    scripts/optim/results/run_<timestamp>/
        runs.jsonl   : one line per individual run (append, crash-robust)
        summary.csv  : rollup per config, sorted by fitness_mean desc
        top.txt      : readable top 20 on stdout + file
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import fitness.py (sibling module)
sys.path.insert(0, str(Path(__file__).parent))
from fitness import evaluate  # noqa: E402


# ====================================================================
# Types
# ====================================================================


@dataclass
class ParamSpec:
    group: str
    name: str
    min_v: float
    max_v: float
    scale: str  # "linear" or "log"

    def sample(self, rng: random.Random) -> float:
        if self.scale == "log":
            import math

            log_min, log_max = math.log(self.min_v), math.log(self.max_v)
            return math.exp(rng.uniform(log_min, log_max))
        return rng.uniform(self.min_v, self.max_v)


@dataclass
class RunResult:
    run_id: int
    seed: int
    params: dict[str, dict[str, float]]  # hierarchical {group: {name: val}}
    fitness: float
    per_metric_scores: dict[str, float]
    metrics: dict[str, Any]
    status: str
    elapsed_ms: int


# ====================================================================
# Search space
# ====================================================================


def load_search_space(path: Path) -> list[ParamSpec]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    specs: list[ParamSpec] = []
    for group, params in data.items():
        for name, cfg in params.items():
            specs.append(
                ParamSpec(
                    group=group,
                    name=name,
                    min_v=float(cfg["min"]),
                    max_v=float(cfg["max"]),
                    scale=cfg.get("scale", "linear"),
                )
            )
    return specs


def sample_config(specs: list[ParamSpec], rng: random.Random) -> dict[str, dict[str, float]]:
    """Returns the hierarchical structure expected by hexsim-bench."""
    config: dict[str, dict[str, float]] = {}
    for spec in specs:
        config.setdefault(spec.group, {})[spec.name] = spec.sample(rng)
    return config


# ====================================================================
# Launching a run
# ====================================================================


def run_bench(
    binary: Path,
    params: dict[str, dict[str, float]],
    seed: int,
    radius: int,
    warmup_ticks: int,
    measure_ticks: int,
    run_id: int,
) -> RunResult:
    """Serializes params to a temp JSON file, runs hexsim-bench, parses
    stdout, returns a RunResult."""
    t0 = time.monotonic()
    # Tempfile for params (cleaned up automatically).
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(params, f)
        params_path = f.name
    try:
        proc = subprocess.run(
            [
                str(binary),
                "--params", params_path,
                "--seed", str(seed),
                "--radius", str(radius),
                "--warmup-ticks", str(warmup_ticks),
                "--measure-ticks", str(measure_ticks),
            ],
            capture_output=True,
            text=True,
            timeout=180,  # safety net: if a run takes over 3min it's a bug
        )
    finally:
        os.unlink(params_path)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if proc.returncode == 1:
        # Input error, bug in the orchestrator, let it propagate
        raise RuntimeError(
            f"hexsim-bench refused the params (run {run_id} seed {seed}): {proc.stderr}"
        )

    # Exit 0 or 2: parse stdout, 2 = NaN/Inf detected
    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"hexsim-bench produced invalid JSON (run {run_id} seed {seed}): {e}\n"
            f"stdout: {proc.stdout[:500]}"
        ) from e

    status = output.get("status", "unknown")
    metrics = output.get("metrics", {})
    fitness, per_metric = evaluate(metrics, status)

    return RunResult(
        run_id=run_id,
        seed=seed,
        params=params,
        fitness=fitness,
        per_metric_scores=per_metric,
        metrics=metrics,
        status=status,
        elapsed_ms=elapsed_ms,
    )


# ====================================================================
# Orchestration
# ====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="Number of configurations")
    parser.add_argument("--seeds", type=str, default="42,1337,7", help="Seeds separated by ,")
    parser.add_argument("--radius", type=int, default=30)
    parser.add_argument("--warmup-ticks", type=int, default=365)
    parser.add_argument("--measure-ticks", type=int, default=1095)
    parser.add_argument("--space", type=Path, default=Path(__file__).parent / "search_space.toml")
    parser.add_argument(
        "--bin",
        type=Path,
        default=Path(__file__).parent.parent.parent / "simulation" / "target" / "release" / "hexsim-bench",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--optim-seed", type=int, default=0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    specs = load_search_space(args.space)
    print(f"Search space: {len(specs)} parameters, {len(seeds)} seeds.")
    print(f"Binary: {args.bin}")
    if not args.bin.exists():
        print(f"ERROR: binary not found. Build with `cargo build --release --bin hexsim-bench`.")
        return 1

    # Prepare results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    csv_path = out_dir / "summary.csv"
    top_path = out_dir / "top.txt"

    rng = random.Random(args.optim_seed)
    # Generate N fixed configs, each config is tested across all seeds
    configs = [(run_id, sample_config(specs, rng)) for run_id in range(args.n)]

    total_jobs = args.n * len(seeds)
    print(f"Total runs: {total_jobs} ({args.n} configs × {len(seeds)} seeds), {args.workers} workers.")
    print(f"Output: {out_dir}")

    all_results: list[RunResult] = []
    start = time.monotonic()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor, jsonl_path.open("w") as jsonl_f:
        futures = []
        for run_id, cfg in configs:
            for seed in seeds:
                futures.append(
                    executor.submit(
                        run_bench,
                        args.bin,
                        cfg,
                        seed,
                        args.radius,
                        args.warmup_ticks,
                        args.measure_ticks,
                        run_id,
                    )
                )

        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                result = fut.result()
            except Exception as e:
                print(f"  ! run failed: {e}")
                continue
            all_results.append(result)
            jsonl_f.write(json.dumps(_result_to_jsonl(result)) + "\n")
            jsonl_f.flush()
            if completed % max(1, total_jobs // 20) == 0 or completed == total_jobs:
                pct = 100 * completed / total_jobs
                elapsed = time.monotonic() - start
                eta = elapsed * (total_jobs - completed) / max(1, completed)
                print(f"  [{completed}/{total_jobs}] {pct:.0f}%, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.monotonic() - start
    print(f"\nAll runs completed in {elapsed:.0f}s.")

    # Aggregate by config (average over seeds)
    configs_aggregated = _aggregate(all_results, args.n)
    configs_aggregated.sort(key=lambda c: c["fitness_mean"], reverse=True)

    # Write CSV
    _write_csv(csv_path, configs_aggregated, specs)
    print(f"CSV: {csv_path}")

    # Top 20
    top_text = _format_top(configs_aggregated[:20], specs, seeds)
    top_path.write_text(top_text)
    print(f"Top 20: {top_path}")
    print()
    print(top_text)
    return 0


def _result_to_jsonl(r: RunResult) -> dict:
    return {
        "run_id": r.run_id,
        "seed": r.seed,
        "fitness": r.fitness,
        "status": r.status,
        "elapsed_ms": r.elapsed_ms,
        "params": r.params,
        "metrics": r.metrics,
        "per_metric_scores": r.per_metric_scores,
    }


def _aggregate(all_results: list[RunResult], n_configs: int) -> list[dict]:
    """Aggregates by run_id: average fitness + keep fitness per seed + average metrics."""
    by_config: dict[int, list[RunResult]] = {}
    for r in all_results:
        by_config.setdefault(r.run_id, []).append(r)

    aggregated: list[dict] = []
    for run_id in range(n_configs):
        runs = by_config.get(run_id, [])
        if not runs:
            continue
        fitness_by_seed = {r.seed: r.fitness for r in runs}
        fitness_mean = sum(fitness_by_seed.values()) / len(fitness_by_seed) if fitness_by_seed else 0.0

        # Average of metrics (keys that are float/int)
        metric_keys: set[str] = set()
        for r in runs:
            metric_keys.update(k for k, v in r.metrics.items() if isinstance(v, (int, float)))
        metrics_mean: dict[str, float] = {}
        for k in metric_keys:
            vals: list[float] = []
            for r in runs:
                v = r.metrics.get(k)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            metrics_mean[k] = sum(vals) / len(vals) if vals else 0.0

        aggregated.append(
            {
                "run_id": run_id,
                "fitness_mean": fitness_mean,
                "fitness_by_seed": fitness_by_seed,
                "params": runs[0].params,  # identical for all seeds
                "metrics_mean": metrics_mean,
                "n_seeds_ok": sum(1 for r in runs if r.status == "ok"),
                "n_seeds_total": len(runs),
            }
        )
    return aggregated


def _write_csv(path: Path, configs: list[dict], specs: list[ParamSpec]) -> None:
    # Columns: run_id, fitness_mean, fitness_seed<X>..., n_ok,
    #          p_<group>_<name>..., m_<metric>...
    param_cols = [f"p_{s.group}_{s.name}" for s in specs]
    # All metrics seen
    metric_names: set[str] = set()
    for c in configs:
        metric_names.update(c["metrics_mean"].keys())
    metric_cols = [f"m_{m}" for m in sorted(metric_names)]

    all_seeds: set[int] = set()
    for c in configs:
        all_seeds.update(c["fitness_by_seed"].keys())
    fit_cols = [f"fit_seed{s}" for s in sorted(all_seeds)]

    header = ["run_id", "fitness_mean"] + fit_cols + ["n_seeds_ok", "n_seeds_total"] + param_cols + metric_cols

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in configs:
            row = [c["run_id"], f"{c['fitness_mean']:.4f}"]
            for s in sorted(all_seeds):
                row.append(f"{c['fitness_by_seed'].get(s, 0):.4f}")
            row.append(c["n_seeds_ok"])
            row.append(c["n_seeds_total"])
            for spec in specs:
                v = c["params"].get(spec.group, {}).get(spec.name)
                row.append(f"{v:.6g}" if v is not None else "")
            for m in sorted(metric_names):
                v = c["metrics_mean"].get(m)
                if v is None:
                    row.append("")
                else:
                    row.append(f"{v:.6g}")
            w.writerow(row)


def _format_top(configs: list[dict], specs: list[ParamSpec], seeds: list[int]) -> str:
    lines = [f"=== TOP {len(configs)} (sorted by fitness_mean desc) ==="]
    for rank, c in enumerate(configs, 1):
        seed_str = " ".join(f"s{s}={c['fitness_by_seed'].get(s, 0):.2f}" for s in seeds)
        lines.append(
            f"\n#{rank}  run_id={c['run_id']}  fitness={c['fitness_mean']:.3f}  [{seed_str}]  "
            f"ok={c['n_seeds_ok']}/{c['n_seeds_total']}"
        )
        # Show params that deviate notably from the middle of their range
        for spec in specs:
            v = c["params"].get(spec.group, {}).get(spec.name)
            if v is None:
                continue
            mid = (spec.min_v + spec.max_v) / 2
            half = (spec.max_v - spec.min_v) / 2
            if half > 0 and abs(v - mid) / half > 0.4:  # > 40% of the half-range
                lines.append(f"    {spec.group}.{spec.name} = {v:.4g}  (range [{spec.min_v:g}, {spec.max_v:g}])")
        # Key metrics
        m = c["metrics_mean"]
        lines.append(
            f"    metrics: rain_max={m.get('rain_days_max_per_cell', 0):.0f}j  "
            f"dry_streak={m.get('dry_streak_median_per_cell_days', 0):.0f}j  "
            f"summer_winter={m.get('ratio_precip_summer_winter', 0):.2f}  "
            f"cloud_mtn={m.get('cloud_cover_mountain_pct', 0)*100:.0f}%  "
            f"drift={m.get('water_drift_pct', 0)*100:.3f}%"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
