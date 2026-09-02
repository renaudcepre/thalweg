#!/usr/bin/env bash
# Stands in for xfail for tracked reds.
#
# nextest 0.9.132 has NO native xfail: the `expected-failure` key is
# ignored as an unknown key (measured on 2026-08-28). And excluding a red from
# a gate with nothing else makes it disappear from every run entirely, which
# is exactly what produced issue #146. Hence this script: gates exclude
# tracked reds (a green run then means "nothing NEW is broken"), and this
# script checks that they are ALL still red.
#
# Three possible alarms, all actionable:
#   - a tracked red passes         -> the physics fix landed, delist it
#   - fewer tests than expected    -> test renamed or removed, list is stale
#   - more tests than expected     -> the filter is catching something else
#
# Usage: known-reds.sh "<nextest filter>" <expected count>
set -uo pipefail

FILTER="${1:?expected nextest filter expression}"
EXPECTED="${2:?expected red count}"

cd "$(dirname "$0")/../simulation" || exit 1

echo "▸ tracked reds: $EXPECTED test(s) must fail"
OUT=$(cargo nextest run --profile heavy --no-fail-fast -E "$FILTER" 2>&1)

SUMMARY=$(printf '%s\n' "$OUT" | grep -E "[0-9]+ tests? run:" | tail -1)
if [ -z "$SUMMARY" ]; then
    echo "✗ no nextest summary line found, the run didn't complete:"
    printf '%s\n' "$OUT" | tail -20
    exit 1
fi

num() { printf '%s\n' "$SUMMARY" | grep -oE "[0-9]+ $1" | grep -oE "^[0-9]+" || echo 0; }
RUN=$(printf '%s\n' "$SUMMARY" | grep -oE "[0-9]+ tests? run" | grep -oE "^[0-9]+")
PASSED=$(num passed)
FAILED=$(num failed)

# nextest prints each status twice (once as it runs, once in the summary),
# and test output itself can also start with "FAIL": keep only the actual
# nextest status lines (the ones carrying a duration), deduplicated.
printf '%s\n' "$OUT" \
    | grep -E "^[[:space:]]+(FAIL|PASS) \[[[:space:]]*[0-9.]+s\]" \
    | sort -u | sed 's/^ */  /'

if [ "$RUN" != "$EXPECTED" ]; then
    echo "✗ $RUN test(s) ran, $EXPECTED expected."
    echo "  The known_reds list in the justfile is stale (test renamed, removed, or filter too broad)."
    exit 1
fi
if [ "$PASSED" != "0" ]; then
    echo "✗ $PASSED tracked red(s) now PASS."
    echo "  Good news: delist them from known_reds in the justfile, and close the issue."
    exit 1
fi
if [ "$FAILED" != "$EXPECTED" ]; then
    echo "✗ $FAILED failure(s) instead of $EXPECTED, unexpected state."
    exit 1
fi

echo "✓ the $EXPECTED tracked reds still fail (#111, #146), debt unchanged"
