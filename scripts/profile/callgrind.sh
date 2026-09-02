#!/usr/bin/env bash
# Callgrind profiling of the simulation tick (profiling effort, see JOURNAL).
#
# Runs `hexsim-bench` (the real Simulation: tiered scheduler, wind and
# transport subsampling) under valgrind/callgrind, then prints the hexsim
# functions sorted by exclusive cost (Ir = instructions retired, deterministic).
#
# Gotcha: the normal build uses `target-cpu=native` (AVX-512 on recent
# machines), which valgrind can't decode (SIGILL at startup).
# So we rebuild in x86-64-v3 (AVX2) with debug symbols: same code, slightly
# different vectorization, identical cost hierarchy.
#
# Usage:
#   scripts/profile/callgrind.sh [radius] [days] [seed]
#   scripts/profile/callgrind.sh            # r30, 365 days, seed 42
#   scripts/profile/callgrind.sh 60 90      # r60, 90 days
#
# Order of magnitude: r30 x 365 days ~= 15-20 min (callgrind ~50x).
# Output: raw profile + annotation in scripts/profile/out/.

set -euo pipefail

RADIUS="${1:-30}"
DAYS="${2:-365}"
SEED="${3:-42}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIM="$ROOT/simulation"
OUT_DIR="$ROOT/scripts/profile/out"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/callgrind.out.r${RADIUS}_d${DAYS}_s${SEED}"
ANNOT="$OUT_DIR/annotate.r${RADIUS}_d${DAYS}_s${SEED}.txt"
PARAMS="$OUT_DIR/empty_params.json"

command -v valgrind >/dev/null || { echo "valgrind missing" >&2; exit 1; }

echo "== build x86-64-v3 + debuginfo (valgrind can't decode AVX-512) =="
(cd "$SIM" && RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_PROFILE_RELEASE_DEBUG=true \
    cargo build --release --bin hexsim-bench)

echo '{}' > "$PARAMS"

echo "== callgrind: r${RADIUS}, ${DAYS} days, seed ${SEED} (long: ~50x the native run) =="
valgrind --tool=callgrind \
    --callgrind-out-file="$OUT" \
    --cache-sim=no \
    "$SIM/target/release/hexsim-bench" \
    --params "$PARAMS" --seed "$SEED" --radius "$RADIUS" \
    --warmup-ticks "$DAYS" --measure-ticks 1 \
    --output "$OUT_DIR/bench_result.json"

echo "== annotation: top functions by exclusive cost =="
callgrind_annotate --auto=no --threshold=99 "$OUT" > "$ANNOT"
# Readable summary: project symbols + libm only.
grep -E "hexsim|libm|PROGRAM|^-|Ir\b" "$ANNOT" | head -50

echo
echo "== breakdown by subphase (inlining resolved) =="
python3 "$ROOT/scripts/profile/subphase_breakdown.py" "$OUT" "$SIM"

echo
echo "Full profile: $ANNOT"
echo "Zoom on source lines: callgrind_annotate --auto=yes $OUT | less"
