"""Fitness function for the HexSim optimization harness.

Computes a score in [0, 1] from the metrics produced by `hexsim-bench`.
Targets and weights are deliberately *explicit and editable*: this isn't
an oracle, it's an aid to judgment. If a target looks wrong after visual
inspection of a top-10 config, the user should adjust it here rather than
reinterpret the results.

Philosophy:
- Score per metric = 1.0 at `target`, descends linearly to 0.0 at
  `target ± tol`, clamped [0, 1].
- Global fitness = weighted sum / sum of weights -> [0, 1].
- HARD_REJECT: one violated condition = fitness forced to 0, the config
  is rejected unambiguously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Direction = Literal["above", "below", "center"]


@dataclass
class Target:
    target: float
    tol: float
    weight: float
    direction: Direction


# Starter targets, derived from the visual observations of April 21, 2026 +
# the current scale tests. Commenting out a line removes that metric from
# the fitness (the user can also double a weight if a metric feels
# underrepresented).
TARGETS: dict[str, Target] = {
    # Rain ceiling: a cell raining more than 280 days/year is pathological.
    # Tol = 100 -> above 380 days/year, score = 0. Widened from 40 -> 100
    # because our current configs all sit in [346, 365] and too narrow a
    # tol crushed everything to 0: impossible for the optimizer to
    # distinguish wb=0.4 (rain_max=346) from wb=0 (365). Now a visible
    # gradient: 346 -> score 0.34, 365 -> score 0.15.
    "rain_days_max_per_cell": Target(target=280.0, tol=100.0, weight=2.0, direction="below"),
    # Median dry streak per cell: target 25 days (~3 dry weeks) BUT
    # symmetric tol in "center" mode, beyond ~60 days we fall into a
    # desert climate (first v1 batch had optima at 316 days, pushing
    # toward desert). "center" prevents the extreme.
    "dry_streak_median_per_cell_days": Target(target=30.0, tol=30.0, weight=1.5, direction="center"),
    # Seasonality: summer/winter ratio, target centered on 1.0 but wide
    # tolerance (tropical climate legitimately ratio 2, Mediterranean 0.5).
    "ratio_precip_summer_winter": Target(target=1.0, tol=0.7, weight=1.0, direction="center"),
    # Cloud cover over relief: target ≥ 55% (currently at 78% with
    # the defaults, showing it's achievable).
    "cloud_cover_mountain_pct": Target(target=0.55, tol=0.15, weight=1.5, direction="above"),
    # Clean snow cycle: min_summer / max_winter < 0.2 (= winter has 5x
    # more snow than summer). Tol = 0.2 -> >= 0.4 = score 0.
    "snow_ratio_winter_max_summer_min": Target(target=0.2, tol=0.2, weight=1.0, direction="below"),
    # Plain rain (<300m): ≥ 60 days/year (minimum temperate climate).
    # Below that = desert climate, not what we're after for a fireplace
    # fire on a "habitable" world.
    "rain_days_median_plain": Target(target=80.0, tol=50.0, weight=1.0, direction="above"),
    # Spatial rain concentration: gap between the rainiest cell
    # (rain_max) and the plains median. Above a 150-day gap = one or a
    # few cells rain 10x more than the median = pathological
    # concentration (typically lakes dropping their own evap back onto
    # themselves). Target 50, tol 150 -> > 200 days gap = score 0, a
    # heavy penalty in the fitness.
    "rain_concentration_spread": Target(target=50.0, tol=150.0, weight=2.0, direction="below"),
}

# Hard rejects: if the condition returns True, the config is rejected
# (fitness = 0). Guardrail against conservation bugs or corrupted runs,
# and against the extreme desert climates the optimizer otherwise finds
# trivially (first v1 batch: top 1 had cloud=0% and dry_streak=316 days,
# i.e. 10 months with no rain per cell).
HARD_REJECT = {
    "status": lambda v: v != "ok",                                # NaN/Inf
    "water_drift_pct": lambda v: v > 0.01,                        # > 1% = leak
    "ms_per_tick": lambda v: v > 15.0,                            # pathological perf
    "cloud_cover_mountain_pct": lambda v: v < 0.10,               # < 10% = no clouds at all
    "dry_streak_median_per_cell_days": lambda v: v > 300,         # > 10 months dry = desert
    "rain_days_median_plain": lambda v: v < 30.0,                 # < 30 days/year plain = too arid
}


def extract_plain_rain_days(metrics: dict) -> float | None:
    """Derived metric: days/year of plain rain (<300m), from the
    rain_days_median_by_altitude[0] array. Returns None if absent."""
    arr = metrics.get("rain_days_median_by_altitude")
    if isinstance(arr, list) and len(arr) >= 1:
        v = arr[0]
        if isinstance(v, (int, float)):
            return float(v)
    return None


def extract_concentration_spread(metrics: dict, plain_rain: float | None) -> float | None:
    """Gap between rain_max (rainiest cell) and the plains median. A high
    value = very localized rain (typically lakes)."""
    rain_max = metrics.get("rain_days_max_per_cell")
    if plain_rain is None or not isinstance(rain_max, (int, float)):
        return None
    return float(rain_max) - plain_rain


def score_metric(value: float, tgt: Target) -> float:
    """Score in [0, 1] for a given metric.

    - `above`  : 1.0 at target+, descends to 0.0 at target - tol.
    - `below`  : 1.0 at target-, descends to 0.0 at target + tol.
    - `center` : 1.0 at target, descends symmetrically.
    """
    if tgt.tol <= 0:
        return 1.0 if value == tgt.target else 0.0
    if tgt.direction == "center":
        delta = abs(value - tgt.target)
    elif tgt.direction == "above":
        delta = max(0.0, tgt.target - value)
    else:  # below
        delta = max(0.0, value - tgt.target)
    return max(0.0, 1.0 - delta / tgt.tol)


def evaluate(metrics: dict, status: str) -> tuple[float, dict[str, float]]:
    """Returns (fitness, per_metric_scores).

    `metrics` = content of the "metrics" field of the JSON produced by
    `hexsim-bench`. `status` = the "status" field (same JSON, root level).
    """
    # Enrich the metrics with derived values (plain rain days from
    # rain_days_median_by_altitude[0], and concentration spread).
    enriched = dict(metrics)
    plain_rain = extract_plain_rain_days(metrics)
    if plain_rain is not None:
        enriched["rain_days_median_plain"] = plain_rain
    spread = extract_concentration_spread(metrics, plain_rain)
    if spread is not None:
        enriched["rain_concentration_spread"] = spread

    # Hard rejects
    for key, predicate in HARD_REJECT.items():
        value = status if key == "status" else enriched.get(key)
        if value is None:
            continue
        if predicate(value):
            return 0.0, {}

    # Weighted score
    per_metric: dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    for key, tgt in TARGETS.items():
        value = enriched.get(key)
        if value is None:
            continue
        s = score_metric(float(value), tgt)
        per_metric[key] = s
        weighted_sum += s * tgt.weight
        total_weight += tgt.weight

    fitness = weighted_sum / total_weight if total_weight > 0 else 0.0
    return fitness, per_metric
