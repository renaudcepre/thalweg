#!/usr/bin/env python3
"""Checks that wasm-opt hasn't amputated the binary of features it ignores.

rustc declares in the custom `target_features` section the WebAssembly
extensions the binary uses (`bulk-memory`, `reference-types`,
`call-indirect-overlong`...). A wasm-opt older than these extensions doesn't
recognize them: it strips them from the declaration, rewrites the code
without understanding it, and produces a binary that instantiates normally,
exposes the same exports, and computes nothing.

This is the failure mode that produced the v0.8.0 release (binaryen 108
served by apt on Ubuntu noble, versus 132 locally): simulation frozen, no
error, no log. Nothing in the build chain flagged it — hence this guard.

Usage: check-wasm-opt.py <before.wasm> <after.wasm>
Exits 1 if `after` has lost features present in `before`.
"""

import subprocess
import sys

SECTION = "target_features"


def _leb128(buf, i):
    """Reads an unsigned LEB128 integer, returns (value, next offset)."""
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def target_features(path):
    """Returns the set of features declared by the module, or None if the
    section is absent (a binary with no section isn't an amputation)."""
    buf = open(path, "rb").read()
    if buf[:4] != b"\0asm":
        sys.exit(f"✗ {path} is not a WebAssembly module")
    i = 8
    while i < len(buf):
        section_id = buf[i]
        i += 1
        size, i = _leb128(buf, i)
        end = i + size
        if section_id == 0:  # custom section: the name follows
            name_len, j = _leb128(buf, i)
            if buf[j : j + name_len].decode("utf8", "replace") == SECTION:
                k = j + name_len
                count, k = _leb128(buf, k)
                found = set()
                for _ in range(count):
                    prefix = chr(buf[k])  # '+' required, '-' disallowed, '=' fixed
                    k += 1
                    flen, k = _leb128(buf, k)
                    found.add(prefix + buf[k : k + flen].decode())
                    k += flen
                return found
        i = end
    return None


def _wasm_opt_version():
    try:
        out = subprocess.run(
            ["wasm-opt", "--version"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unreadable version"
    except (OSError, subprocess.CalledProcessError):
        return "not found in PATH"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    before, after = sys.argv[1], sys.argv[2]
    was, now = target_features(before), target_features(after)

    if was is None:
        print(f"⚠ {before} doesn't declare a `{SECTION}` section — nothing to check")
        return
    if now is None:
        sys.exit(f"✗ wasm-opt removed the whole `{SECTION}` section")

    lost = was - now
    if lost:
        sys.exit(
            "✗ wasm-opt stripped target features from the binary:\n"
            + "".join(f"    {f}\n" for f in sorted(lost))
            + "\n  It's too old to understand them. The resulting binary\n"
            "  instantiates but computes nothing (cf. release v0.8.0).\n"
            f"  wasm-opt used here: {_wasm_opt_version()}\n"
            "  macOS: brew upgrade binaryen\n"
            "  CI     : see .github/workflows/dist.yml, binaryen is pinned\n"
            "           from GitHub releases, not from apt."
        )
    print(f"✓ target_features intact ({len(now)} features)")


if __name__ == "__main__":
    main()
