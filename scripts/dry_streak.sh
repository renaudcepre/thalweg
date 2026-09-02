#!/usr/bin/env bash
# dry_streak.sh: measures the longest consecutive period with NO rain anywhere
# on the map, plus a count of permanent high-altitude snow.
# A convincing "fireplace fire" needs:
#   - at least 60 consecutive days with no rain anywhere (>=2 months of visible dryness)
#   - at least 1 cell >1500m with snow_level > 0 permanently over 2 years
#
# Usage: bash scripts/dry_streak.sh [seed] [years]
# Params via env: PARAM_<group>_<key>=<value>

set -e
SEED=${1:-42}
YEARS=${2:-2}
TICKS=$((YEARS * 365))

CTL=./simulation/target/release/hexsim-ctl

$CTL reset "$SEED" > /dev/null
# Apply params from PARAM_* env vars
for var in $(env | grep '^PARAM_' | cut -d= -f1); do
    key=$(echo "$var" | sed -e 's/^PARAM_//' -e 's/_/./')
    val="${!var}"
    $CTL param -- "$key" "$val" > /dev/null
    echo "# applied $key=$val"
done

# 90-day warmup (lets the sim settle)
$CTL step 90 > /dev/null

echo "tick,raining,snow_max_all,snow_total"
max_snow_high_ever=0
for day in $(seq 1 $TICKS); do
    out=$($CTL step 1 2>/dev/null)
    r=$(echo "$out" | grep -E '"raining_cells"' | head -1 | grep -oE '[0-9]+')
    r=${r:-0}
    snow_max=$(echo "$out" | grep -oE '"snow"[^}]*"max": *[0-9.eE+-]+' | head -1 | grep -oE '[0-9.eE+-]+$')
    snow_max=${snow_max:-0}
    snow_tot=$(echo "$out" | grep -oE '"snow"[^}]*"total": *[0-9.eE+-]+' | head -1 | grep -oE '[0-9.eE+-]+$')
    snow_tot=${snow_tot:-0}
    echo "$day,$r,$snow_max,$snow_tot"
done > /tmp/raining_log.csv

# Final analysis
python3 << 'PYEOF'
import csv
rows = []
snow_max_all = []
snow_tot_all = []
with open('/tmp/raining_log.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    for r in reader:
        if len(r) == 4:
            rows.append((int(r[0]), int(r[1])))
            snow_max_all.append(float(r[2]) if r[2] else 0.0)
            snow_tot_all.append(float(r[3]) if r[3] else 0.0)

# Streaks at different thresholds of "negligible rain"
def max_streak(rows, thr):
    m = 0; c = 0
    for _, n in rows:
        if n < thr:
            c += 1
            m = max(m, c)
        else:
            c = 0
    return m

zero = sum(1 for _, n in rows if n == 0)
print(f"Total ticks: {len(rows)}")
print(f"Days with 0 raining cells: {zero} ({100*zero/len(rows):.1f}%)")
for thr in [1, 5, 10, 20, 50]:
    s = max_streak(rows, thr)
    days_below = sum(1 for _, n in rows if n < thr)
    marker = "✅" if s >= 60 else ("🟡" if s >= 30 else "❌")
    print(f"  streak <{thr:3}: max={s:3} days, total={days_below:3} days  {marker}")

# Min/max raining cells
mins = [n for _, n in rows]
print(f"raining_cells min={min(mins)} max={max(mins)} mean={sum(mins)/len(mins):.0f}")

# Snow total over time
if snow_tot_all:
    print(f"snow_total min={min(snow_tot_all):.0f} max={max(snow_tot_all):.0f} final={snow_tot_all[-1]:.0f}")
    print(f"snow_max (max snow_level any cell) max_ever={max(snow_max_all):.2f}")
    # Is there an accumulation that survives a summer?
    # Find the min of snow_total over the 2nd half of the sim (after 1 summer of melt)
    mid = len(snow_tot_all) // 2
    min_late = min(snow_tot_all[mid:]) if mid < len(snow_tot_all) else 0
    print(f"snow_total min over 2nd half (after 1 summer) = {min_late:.1f}")
PYEOF

# Perennial snow at high altitude
echo "--- Perennial snow >1500m ---"
$CTL region 0 0 50 > /tmp/final_grid.json 2>/dev/null
python3 << 'PYEOF'
import json
with open('/tmp/final_grid.json') as f:
    d = json.load(f)
high = [c for c in d['cells'] if c['elevation'] > 1500]
with_snow = [c for c in high if c['snow_level'] > 0.1]
print(f"Cells >1500m: {len(high)}, with snow>0.1 (instant): {len(with_snow)}  (target: ≥1)")
if with_snow and high:
    pcts = sorted([c['snow_level'] for c in with_snow], reverse=True)[:5]
    print(f"top snow_level >1500m: {[f'{x:.2f}' for x in pcts]}")
    print("  ✅ AT LEAST ONE snowy cell high")
else:
    print("  ❌ FAIL no snow on high altitudes")
PYEOF
