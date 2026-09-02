#!/usr/bin/env python3
"""Extracts the top-N of a random_search run into params.json files ready
to be tested via `just bench <params>` or injected into the live sim via
`just param` for a visual check.

Usage:

    python scripts/optim/extract_top.py <results_dir> [--n 5]

Example:

    python scripts/optim/extract_top.py scripts/optim/results/run_20260422_105432 --n 3

Produces <results_dir>/top_1.json, top_2.json, ... with the hierarchical
params expected by hexsim-bench and a header comment summarizing the
fitness and key metrics of the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="run_<timestamp> directory")
    parser.add_argument("--n", type=int, default=5, help="Number of configs to extract (default 5)")
    args = parser.parse_args()

    jsonl = args.results_dir / "runs.jsonl"
    if not jsonl.exists():
        print(f"runs.jsonl not found in {args.results_dir}", file=sys.stderr)
        return 1

    # Aggregate by run_id (average of fitness across seeds)
    by_run: dict[int, list[dict]] = {}
    with jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            by_run.setdefault(rec["run_id"], []).append(rec)

    aggregated = []
    for run_id, runs in by_run.items():
        fitnesses = [r["fitness"] for r in runs]
        if not fitnesses:
            continue
        mean_fit = sum(fitnesses) / len(fitnesses)
        # Take the params from the first one (identical across all seeds)
        params = runs[0]["params"]
        # Average of scalar metrics
        metric_keys: set[str] = set()
        for r in runs:
            metric_keys.update(k for k, v in r["metrics"].items() if isinstance(v, (int, float)))
        metrics_mean = {}
        for k in metric_keys:
            vals = [r["metrics"][k] for r in runs if isinstance(r["metrics"].get(k), (int, float))]
            metrics_mean[k] = sum(vals) / len(vals) if vals else 0.0
        aggregated.append(
            {
                "run_id": run_id,
                "fitness_mean": mean_fit,
                "fitness_by_seed": {r["seed"]: r["fitness"] for r in runs},
                "params": params,
                "metrics_mean": metrics_mean,
            }
        )

    aggregated.sort(key=lambda c: c["fitness_mean"], reverse=True)
    top = aggregated[: args.n]

    for rank, c in enumerate(top, 1):
        out_path = args.results_dir / f"top_{rank}.json"
        header = {
            "_comment": (
                f"Top {rank} of run {args.results_dir.name}. "
                f"Fitness mean {c['fitness_mean']:.3f}, seeds "
                + " ".join(f"s{s}={f:.2f}" for s, f in c["fitness_by_seed"].items())
                + ". Metrics: "
                + ", ".join(
                    f"{k}={v:.2f}"
                    for k, v in sorted(c["metrics_mean"].items())
                    if not k.startswith("cell_count")
                )
            ),
        }
        # params + _comment (the leading _ on the field isn't recognized by
        # serde deny_unknown_fields, so it must be stripped before injecting).
        # We write _comment only into a separate .txt file so params.json
        # can be injected directly into hexsim-bench.
        (args.results_dir / f"top_{rank}.txt").write_text(
            f"{header['_comment']}\n\nparams:\n{json.dumps(c['params'], indent=2)}\n"
        )
        out_path.write_text(json.dumps(c["params"], indent=2))
        print(f"{out_path}  (fitness={c['fitness_mean']:.3f})")

    if top:
        print(f"\nTest a top-N with:  just bench {args.results_dir / 'top_1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
