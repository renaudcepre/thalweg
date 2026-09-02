#!/usr/bin/env python3
"""Breakdown by subphase from a raw callgrind profile (profiling effort).

Aggressive release inlining melts the sub-functions into the
`Simulation::step_hour` symbol: `callgrind_annotate` alone can no longer say
"how much does step_orographic_convection cost". This script reattributes:

1. the *self* cost of each hexsim source line to its originating function
   (bucketing by line ranges, extracted by grepping `fn `);
2. the *inclusive* cost of outgoing calls to external code (libm,
   memcpy/memset, malloc) to the **calling function**: inlining places
   these call sites in std sources, so line-based bucketing doesn't see
   them; the "caller -> external" table recovers them.

No double counting: hexsim -> hexsim calls are not added
(the callee's lines are already counted via their self cost).

Usage: subphase_breakdown.py callgrind.out.rXX [src_root=simulation/]
"""

import os
import re
import subprocess
import sys
from collections import defaultdict

RAW = sys.argv[1]
SRC_ROOT = sys.argv[2] if len(sys.argv) > 2 else "simulation"

EXTERNALS = (
    "expf", "powf", "logf", "sinf", "cosf",
    "memcpy", "memset", "memmove", "malloc", "calloc", "free",
)

id_re = re.compile(r"^(?:\((\d+)\))?\s*(.*)$")


def norm_id(match, table):
    ident, name = match.group(1), match.group(2)
    if name:
        if ident:
            table[ident] = name
        return name
    return table.get(ident, "?")


# --- pass 1: parse the raw file ---
files, fns = {}, {}
line_cost = defaultdict(int)      # (file, line) -> self Ir
ext_calls = defaultdict(int)      # (caller, external) -> inclusive Ir
ext_count = defaultdict(int)      # (caller, external) -> call count
total = 0

cur_file, cur_fn, cfn = None, None, None
pending_calls = 0
last_line = 0

with open(RAW) as f:
    for raw in f:
        line = raw.rstrip("\n")
        if line.startswith("summary:"):
            total = int(line.split()[1])
        elif line.startswith(("fl=", "fi=", "fe=")):
            cur_file = norm_id(id_re.match(line[3:]), files)
            last_line = 0
        elif line.startswith("fn="):
            cur_fn = norm_id(id_re.match(line[3:]), fns)
            last_line = 0
        elif line.startswith(("cfi=", "cfl=")):
            norm_id(id_re.match(line[4:]), files)
        elif line.startswith("cfn="):
            cfn = norm_id(id_re.match(line[4:]), fns)
        elif line.startswith("calls="):
            pending_calls = int(line.split("=")[1].split()[0])
        elif line and (line[0].isdigit() or line[0] in "+-*"):
            pos, cost = line.split()[:2]
            cost = int(cost)
            if pos == "*":
                ln = last_line
            elif pos[0] in "+-":
                ln = last_line + int(pos)
            else:
                ln = int(pos)
            last_line = ln
            if pending_calls:
                if cfn and "hexsim" not in cfn:
                    key = (cur_fn or "?", cfn)
                    ext_calls[key] += cost
                    ext_count[key] += pending_calls
                pending_calls = 0
                cfn = None
            else:
                line_cost[(cur_file, ln)] += cost

# --- pass 2: bucketing hexsim lines by source function ---
range_cache = {}


def function_marks(path):
    if path not in range_cache:
        out = subprocess.run(
            ["grep", "-nE", r"^\s*(pub )?(const )?fn [a-zA-Z0-9_]+", path],
            capture_output=True, text=True,
        ).stdout
        marks = []
        for entry in out.splitlines():
            n, rest = entry.split(":", 1)
            name = re.search(r"fn\s+([a-zA-Z0-9_]+)", rest)
            if name:
                marks.append((int(n), name.group(1)))
        range_cache[path] = marks
    return range_cache[path]


buckets = defaultdict(int)
for (fname, ln), cost in line_cost.items():
    if not fname or "hexsim" not in fname:
        continue
    path = os.path.join(SRC_ROOT, fname)
    if not os.path.exists(path):
        continue
    fn_name = "?"
    for start, name in function_marks(path):
        if ln >= start:
            fn_name = name
        else:
            break
    buckets[f"{os.path.basename(fname)}::{fn_name}"] += cost

# --- output ---
print(f"TOTAL program: {total:,} Ir\n")
print("== Self cost by subphase (hexsim code, inlining resolved) ==")
print(f"{'subphase':<52} {'Ir':>16} {'% total':>8}")
for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
    if v / total < 0.001:
        continue
    print(f"{k:<52} {v:>16,} {100 * v / total:>7.2f}%")

print("\n== External calls (libm / libc) by calling function ==")
print(f"{'caller -> external':<74} {'Ir incl':>14} {'%':>7} {'calls':>13}")
rows = [(v, k) for k, v in ext_calls.items()
        if any(e in k[1] for e in EXTERNALS)]
for v, k in sorted(rows, reverse=True):
    if v / total < 0.001:
        continue
    caller = k[0].split("[")[0][:50]
    callee = k[1].split("@")[0][:18]
    print(f"{caller:<52} -> {callee:<18} {v:>14,} {100 * v / total:>6.2f}% {ext_count[k]:>13,}")
