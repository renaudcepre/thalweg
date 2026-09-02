# HexSim: project commands

set working-directory := "simulation"

# Private recipes (journal, mirror publication). `?` = optional: a clone of
# the public `thalweg` mirror doesn't have this file and works without it.
import? 'justfile.local'

jq_compact := "walk(if type == \"object\" then del(.top, .bottom) else . end)"

# Tracked reds, physics not hygiene (#111 for phys_wet_peak_snows,
# #146 for the three other binaries). Excluded from the gates so a
# green means "nothing NEW is broken", and checked to always stay red
# by `just reds`, which alarms if one heals. Measured on 2026-08-28:
# these 4 binaries contain ONLY these 5 tests, so filtering by binary
# hides no green test. Delist here once the physics fix lands.
known_reds := "binary(phys_wet_peak_snows) | binary(physics_lake_concentration) | binary(scale_seasonal_climatology) | binary(scale_universal_invariants)"
known_reds_count := "5"

# ── Dev ──────────────────────────────────────────────

# Build
build:
    cargo build

# Default suite = dev loop (~20s on 4 cores). The nextest `default` profile
# (.config/nextest.toml) excludes the 10-year sims (scale_*) and the
# tracked red; the cargo `dev` profile compiles at opt-level 3 (Cargo.toml).
#
# `--no-fail-fast` everywhere (2026-08-06): without it the suite stops at the
# first red binary, and since the tracked reds (`phys_wet_peak_snows`,
# `physics_lake_concentration`) sort before `scale_*` alphabetically,
# `check-all` used to report a verdict on 37 tests that never ran, all scale_*
# included. A gate hiding 3/4 of its result behind the first known red isn't
# doing its job.
test:
    cargo nextest run --no-fail-fast -E "not ({{known_reds}})"

# Old name for the fast loop, `test` does this job now.
alias test-fast := test

# Full suite, scale_* 10-year runs included (~2 min on 4 cores). Run
# before merging a physics change; target a single heavy one with:
#   cargo nextest run --profile heavy -E 'test(dry_periods)'
#
# The 5 tracked reds (`known_reds`) are excluded from it: they're measured by
# `just reds`, which checks that they ALWAYS fail all 5. See
# scripts/known-reds.sh for why exclusion alone isn't enough.
test-all:
    cargo nextest run --profile heavy --no-fail-fast -E "not ({{known_reds}})"

# Tracked reds: checks that they all fail, and nothing more. Stands in
# for xfail; nextest 0.9.132 has no `expected-failure`. Alarms if one
# heals (delist it) or if the list is stale.
reds:
    ../scripts/known-reds.sh "{{known_reds}}" {{known_reds_count}}

# Run tests with output (for metrics)
test-verbose name:
    cargo test -p hexsim-core --test {{name}} -- --nocapture

# Water cycle metrics
metrics:
    cargo test -p hexsim-core --test water_cycle_metrics -- --nocapture

# Runs all structural diagnostic tools (tests/diag_*.rs).
# Eval-style: no assert, output on stderr. Slow tools (~30-60s each),
# meant to be run by hand when investigating an open physics question
# (#68 stationary clouds, #24 cell-lake, #69 drizzle, etc). Compilation
# is already gated by the pre-commit hook (clippy --all-targets); this
# recipe covers execution.
#
# Selects only binaries whose name starts with `diag_` (nextest
# filter), runs the `#[ignore]` tests (run-ignored=only), and prints
# stderr/stdout (no-capture). Doesn't depend on a list to maintain:
# any new `tests/diag_X.rs` is picked up automatically.
diag-tools:
    cargo nextest run --release --no-capture --run-ignored=only -E 'binary(/^diag_/)'

# Targeted diag on a single tool. Example: just diag-tool cloud_clusters
diag-tool name:
    cargo test --release -p hexsim-core --test diag_{{name}} -- --ignored --nocapture

# Strict lint: `-D warnings` hardens pedantic into deny. Matches the
# pre-commit hook, whatever passes here passes the commit.
# --keep-going: without it, clippy stops at the first failing crate and
# hides the lints of crates that depend on it (measured on 28-08: 11
# sites reported, 16 real; hexsim-mcp was hidden behind hexsim-wsclient).
lint:
    cargo clippy --all-targets --all-features --keep-going -- -D warnings

# Format
fmt:
    cargo fmt

# Format check
fmt-check:
    cargo fmt --check

# Full check (fmt + lint + default suite)
check:
    cargo fmt --check && cargo clippy --all-targets --all-features --keep-going -- -D warnings && cargo nextest run --no-fail-fast -E "not ({{known_reds}})"

# Pre-merge check: like `check` but with the full suite (test-all)
check-all:
    cargo fmt --check && cargo clippy --all-targets --all-features --keep-going -- -D warnings && cargo nextest run --profile heavy --no-fail-fast -E "not ({{known_reds}})" && ../scripts/known-reds.sh "{{known_reds}}" {{known_reds_count}}

# Run the server (release = ~10x faster, also builds hexsim-ctl and hexsim-mcp)
run:
    cargo build --release && cargo run --release --bin hexsim-cli

# Stops the hexsim-cli server if it's running (silent pkill if absent).
stop:
    #!/usr/bin/env bash
    if pkill -f "hexsim-cli"; then
        sleep 1
        echo "✓ Server stopped"
    else
        echo "ℹ No hexsim-cli server running"
    fi

# Clean rebuild: clean + full rebuild + restart server
rebuild:
    #!/usr/bin/env bash
    set -e
    pkill -f "hexsim-cli" || true
    sleep 1
    cargo clean
    VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
    HASH=$(git -C .. rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
    if [ -n "$(git -C .. status --porcelain --untracked-files=no 2>/dev/null)" ]; then HASH="$HASH-dirty"; fi
    echo "─── rebuild @ $(date +%H:%M:%S) ─────────────"
    echo "  version:    v$VERSION"
    echo "  build hash: $HASH"
    cargo build --release -p hexsim-cli
    nohup ./target/release/hexsim-cli > /tmp/hexsim.log 2>&1 &
    for i in $(seq 1 150); do
        curl -sf -o /dev/null http://localhost:8355/ && break
        if [ "$i" -eq 150 ]; then echo "✗ Server unreachable after 30s, see /tmp/hexsim.log" >&2; exit 1; fi
        sleep 0.2
    done
    echo "✓ Server restarted, v$VERSION · $HASH"
    echo "  logs: tail -f /tmp/hexsim.log"

# Doc tests
doc:
    cargo test --doc

# ── Optim: parameter evaluation harness ─────
bench_bin := "./target/release/hexsim-bench"

# Isolated run of the bench runner with a params JSON (default seed 42).
# Example: just bench /tmp/params.json
bench params seed="42":
    cargo build --release --bin hexsim-bench
    {{bench_bin}} --params {{params}} --seed {{seed}}

# Random search over N configurations (default 50). 3 seeds, radius 30.
# Results go to scripts/optim/results/run_<timestamp>/.
bench-search n="50":
    cargo build --release --bin hexsim-bench
    cd .. && python3 scripts/optim/random_search.py --n {{n}}

# ── Profiling ────────────────────────────────────────

# Wall-clock breakdown by top-level tick phase (dedicated bench, ~30s).
# `#[ignore]`: it's a benchmark, not a test, excluded from the default suite.
perf-phases:
    cargo test --release --test perf_phase_breakdown -- --ignored --nocapture

# Per-tick cost on a radius-45 grid (~6k cells), 2 years. Same status.
perf-scale:
    cargo test --release --test scale_perf_radius_60 -- --ignored --nocapture

# Full callgrind profile (cost per function, deterministic). Long: ~15-20 min
# for the default 365 days. Short example: just profile 30 90
profile radius="30" days="365":
    cd .. && scripts/profile/callgrind.sh {{radius}} {{days}}

# ── Simulation API (server on :8355) ───────────────
ctl := "./target/release/hexsim-ctl"

# Compact diagnostics (without top/bottom cells)
diag:
    {{ctl}} diag | jq '{{jq_compact}}'

# Full diagnostics (with top/bottom cells per property)
diag-full:
    {{ctl}} diag | jq

# Play (runs the simulation continuously)
play:
    {{ctl}} play

# Pause
pause:
    {{ctl}} pause

# Advances N ticks (default: 1)
step n="1":
    {{ctl}} step {{n}} | jq '{{jq_compact}}'

# Advances N hours (default: 1), diurnal cycle, intra-day synoptic drift
step-hour n="1":
    {{ctl}} step-hour {{n}} | jq '{{jq_compact}}'

# Advances by one month (30 days)
month:
    {{ctl}} step 30 | jq '{{jq_compact}}'

# Advances by one year (365 days)
year:
    {{ctl}} step 365 | jq '{{jq_compact}}'

# Advances N years and prints a summary per year (default: 5)
monitor years="5":
    #!/usr/bin/env bash
    for i in $(seq 1 {{years}}); do
        ./target/release/hexsim-ctl step 365 \
        | jq -r '"tick=\(.tick | tostring | (" " * (5 - length)) + .) | surface=\(.water_budget.surface | round) humidity=\(.water_budget.humidity | round) gw=\(.water_budget.groundwater | round) total=\(.water_budget.total | . * 10 | round / 10) | rain=\(.hydrology.raining_cells) rivers=\(.hydrology.river_cells) | temp \(.temperature.min | . * 10 | round / 10)..\(.temperature.max | . * 10 | round / 10)"'
    done

# Change a parameter at runtime (e.g. just param "atmosphere.cloud_evap_rate" 0.15)
param key value:
    {{ctl}} param {{key}} {{value}}

# Climate report: rain/snow by elevation band over 30/180/365 days
climate:
    {{ctl}} climate | jq

# Resets the simulation (optional seed)
reset seed="":
    {{ctl}} reset {{seed}} | jq '{{jq_compact}}'

# ── Screenshots (Playwright) ────────────────────
# The server must be running (just run / just rebuild). Drive the sim with
# play/pause/step/reset, then capture the 3D scene as rendered by the front end.

# Installs the front end's npm deps (three.js, served locally, no CDN)
front-setup:
    cd ../frontend && npm install

# Installs the WASM tooling, one-shot after a fresh clone (#138)
wasm-setup:
    #!/usr/bin/env bash
    set -e
    rustup target add wasm32-unknown-unknown
    # binaryen comes from brew, not wasm-pack's automatic download: the latter
    # hits GitHub Releases and fails behind a restricted network.
    # Hence `wasm-opt = false` in the crate's Cargo.toml.
    command -v wasm-pack >/dev/null || brew install wasm-pack
    command -v wasm-opt  >/dev/null || brew install binaryen

# Builds the WASM module into frontend/wasm/ (artifact, gitignored)
wasm:
    #!/usr/bin/env bash
    set -e
    wasm-pack build crates/hexsim-wasm --target web --release --out-dir ../../../frontend/wasm
    OUT=../frontend/wasm/hexsim_wasm_bg.wasm
    # `-O3` and not `-Oz`: measured at build time (#138), all four wasm-opt
    # levels produce the same size once gzipped (272-276 KB); the raw gap
    # is 2.5%. Nothing to gain from sacrificing execution speed, the sim
    # is CPU-bound, which is the whole point of `lto = "fat"` (#41).
    wasm-opt -O3 "$OUT" -o "$OUT.opt"
    # Guardrail (#141): a wasm-opt older than the target features emitted
    # by rustc won't recognize them; it strips them from the declaration,
    # rewrites the code without understanding it, and produces a binary that
    # instantiates, exposes the same exports, and computes nothing. Zero
    # build error, zero console error. This is what shipped in release
    # v0.8.0: CI was pulling binaryen 108 (2022) from apt, versus 132 locally.
    ../scripts/check-wasm-opt.py "$OUT" "$OUT.opt"
    mv "$OUT.opt" "$OUT"
    # `stat -f%z` is the BSD form (macOS), `-c%s` the GNU form (Linux); the
    # recipe also runs on Ubuntu's CI runner (#138).
    RAW=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
    GZ=$(gzip -9 -c "$OUT" | wc -c | tr -d ' ')
    printf '✓ %s\n  raw   %6.1f KB\n  gzip  %6.1f KB\n' "$OUT" "$(echo "scale=1; $RAW/1024" | bc)" "$(echo "scale=1; $GZ/1024" | bc)"

# Builds dist/: the complete static site (front end + WASM), deployable
# as-is on any file host. Doesn't deploy anything, see scripts/build-web.sh.
build-web:
    ../scripts/build-web.sh

# Serves dist/ statically, without hexsim-cli or anything Rust; it's the
# deployed site's configuration, reproduced locally. `just build-web` first.
serve-web port="8099":
    #!/usr/bin/env bash
    set -e
    [ -d ../dist ] || { echo "✗ dist/ missing, run first: just build-web" >&2; exit 1; }
    echo "▸ http://localhost:{{port}}  (Ctrl-C to stop)"
    echo "  no simulation server: the world runs in the tab"
    cd ../dist && python3 -m http.server {{port}} --bind 127.0.0.1

# Installs Playwright + Chromium (one-shot, after a fresh clone)
shot-setup:
    cd ../scripts/shot && npm install && npx playwright install chromium

# Captures the scene. Args passed through as-is to shot.mjs (see its header).
# Examples:
#   just shot                              # framed map, oblique view
#   just shot --zoom-factor 0.5            # zoom in x2
#   just shot --azimuth 90 --polar 30      # different angle
#   just shot --view temperature --clean   # temperature background, no sidebars
#   just shot --target "5,-3" --zoom 12    # close-up on (q,r)≈(5,-3)
shot *ARGS:
    cd .. && node scripts/shot/shot.mjs {{ARGS}}

# Checks that an embed starts up correctly: shipped world loaded, playback
# started, zero page errors. Server required. With no argument, tests WASM
# mode in `chrome=none`, the real path for an embed on a third-party site (#147).
#   just embed-check
#   just embed-check 'http://localhost:8355/?chrome=none'             # via the WS server
#   just embed-check 'http://localhost:8355/?mode=wasm&world=neuf&chrome=none'
#
# Named parameter and quotes in the recipe, not a `*ARGS`: just
# interpolates raw text into the shell line, so a `&` in a URL used to cut
# the command short. `?mode=wasm&world=neuf&chrome=none` used to turn into
# `node … '?mode=wasm' &` plus two background jobs; the check ran against
# the full UI, without the embed, and came back red for the wrong reasons
# (2026-08-26).
embed-check url="http://localhost:8355/?mode=wasm&chrome=none" ms="":
    cd .. && node scripts/shot/embed-check.mjs "{{url}}" {{ms}}

# Top-down map view, bare map. Extra args passed through.
shot-top *ARGS:
    cd .. && node scripts/shot/shot.mjs --top --clean {{ARGS}}

# Vegetation cover composition of the current state (species, mix per hex).
cover:
    cd .. && node scripts/shot/cover.mjs
