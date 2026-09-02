#!/usr/bin/env bash
# rain_proximity.sh - Measures the rain halo around lakes.
# Goal: a ratio (% raining / % baseline) close to 1.0 at all
# distances = rain spatially uniformly distributed.
# Bias > 1.5 at dist=0-2 = persistent captive halo.
#
# Usage: bash scripts/rain_proximity.sh [seed] [ticks]

set -e
SEED=${1:-42}
TICKS=${2:-365}

CTL=./simulation/target/release/hexsim-ctl

$CTL reset "$SEED" > /dev/null
# Re-applies TUNING params passed via env vars PARAM_X_Y=value
# e.g.: PARAM_atmosphere_transpiration_coef=0.5 bash scripts/rain_proximity.sh
for var in $(env | grep '^PARAM_' | cut -d= -f1); do
    key="${var#PARAM_}"
    key="${key//_/.}"
    # Restores the first "." to "_" for "atmosphere_X" -> "atmosphere.X"
    # actually we want exactly one "." after the group prefix, the rest as "_"
    key=$(echo "$var" | sed -e 's/^PARAM_//' -e 's/_/./')
    val="${!var}"
    $CTL param "$key" "$val" > /dev/null
    echo "# applied $key=$val"
done
$CTL step "$TICKS" > /dev/null
$CTL region 0 0 50 > /tmp/hexgrid.json

python3 - << 'PYEOF'
import json
from collections import Counter

with open('/tmp/hexgrid.json') as f:
    data = json.load(f)

cells = data['cells']
lakes = [(c['q'], c['r']) for c in cells if c['water_level'] > 3.0]
raining = [(c['q'], c['r']) for c in cells if c['is_raining']]

def hex_dist(a, b):
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq+dr)) // 2

dists_r = [min(hex_dist(r, l) for l in lakes) for r in raining] if lakes else []
dists_all = [min(hex_dist((c['q'], c['r']), l) for l in lakes) for c in cells]
hist_r = Counter(dists_r)
hist_all = Counter(dists_all)

print(f"tick≈{data.get('tick','?')} | lakes={len(lakes)} raining={len(raining)} ({100*len(raining)/len(cells):.1f}%)")
print(f"{'dist':>4} | {'rain%':>6} | {'base%':>6} | {'ratio':>6}")
print("-" * 36)
for d in sorted(set(hist_r) | set(hist_all)):
    r_pct = 100*hist_r.get(d,0)/max(1,len(dists_r))
    a_pct = 100*hist_all.get(d,0)/len(dists_all)
    ratio = r_pct/a_pct if a_pct > 0 else 0
    marker = "##" if ratio > 1.5 else ("--" if ratio < 0.7 else "")
    print(f"{d:>4} | {r_pct:>5.1f}% | {a_pct:>5.1f}% | {ratio:>5.2f} {marker}")

# Synthetic score: KL divergence (rain vs baseline)
# Lower = more uniform distribution
import math
kl = 0.0
for d in set(dists_r):
    p = hist_r[d]/len(dists_r)
    q = hist_all[d]/len(dists_all)
    if p > 0 and q > 0:
        kl += p * math.log(p/q)
print(f"\nKL divergence rain||baseline = {kl:.3f} (ideal: 0, < 0.1 = very good, > 0.3 = strong halo)")

# Pct raining cells near a lake
for n in [2, 3, 5]:
    pct = 100 * sum(1 for d in dists_r if d <= n) / max(1, len(dists_r))
    baseline = 100 * sum(1 for d in dists_all if d <= n) / len(dists_all)
    print(f"raining within <={n} hex: {pct:.1f}% (baseline: {baseline:.1f}%)")
PYEOF
