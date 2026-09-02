#!/usr/bin/env bash
# Builds `dist/`: the front end + the WASM module, servable by any static
# file host, without `hexsim-cli` behind it.
#
# The project is zero-bundler: `index.html` loads three, msgpack and the fonts
# via an importmap that points at `node_modules/`. So we don't copy a bundle:
# we copy the handful of files actually loaded into `vendor/` and rewrite the
# importmap to point at them. No Vite, no rollup: the front stays readable
# exactly as served.
#
# Usage: just build-web
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONT="$ROOT/frontend"
NM="$FRONT/node_modules"
DIST="$ROOT/dist"

[ -d "$NM" ] || { echo "✗ frontend/node_modules missing, run first: just front-setup" >&2; exit 1; }

# 1. The WASM module. `frontend/wasm/` is gitignored: on a fresh clone it
#    doesn't exist, and on a dev machine it can predate the last Rust
#    change. We rebuild it unconditionally.
echo "▸ building the WASM module"
( cd "$ROOT/simulation" && just wasm )

# 2. The front end, without dev-only files.
echo "▸ copying the front end"
rm -rf "$DIST"
mkdir -p "$DIST"
for f in index.html main.js logs.js style.css; do cp "$FRONT/$f" "$DIST/"; done
cp -R "$FRONT/transport" "$DIST/transport"
# The shipped world (#147): without it, an embed starts on a fresh, flat
# world with no history. Missing from the repo? We don't fail, the embed
# falls back to a fresh world while logging a warning.
if [ -d "$FRONT/worlds" ]; then
  cp -R "$FRONT/worlds" "$DIST/worlds"
else
  echo "⚠ frontend/worlds/ missing, the embed will start on a fresh world" >&2
fi
mkdir -p "$DIST/wasm"
# The .d.ts files and .gitignore are only used on the dev machine.
cp "$FRONT/wasm/hexsim_wasm.js" "$FRONT/wasm/hexsim_wasm_bg.wasm" "$DIST/wasm/"

# 3. The dependencies actually loaded, and only those.
echo "▸ vendor"
mkdir -p "$DIST/vendor/three/addons/controls" "$DIST/vendor/msgpack" "$DIST/vendor/fontsource/files"
# three.module.js isn't self-contained: since three 0.183 it re-exports
# from three.core.js, which must therefore come along. The min/webgpu/tsl/cjs
# variants are never loaded by this front end.
cp "$NM/three/build/three.module.js" "$NM/three/build/three.core.js" "$DIST/vendor/three/"
# OrbitControls is the only addon imported by main.js, and it only imports `three`.
cp "$NM/three/examples/jsm/controls/OrbitControls.js" "$DIST/vendor/three/addons/controls/"
# index.mjs re-exports about a dozen neighboring modules, including a
# `utils/` subfolder: we copy the tree as-is (flattening it breaks relative
# imports), minus types and source maps.
cp -R "$NM/@msgpack/msgpack/dist.esm/." "$DIST/vendor/msgpack/"
find "$DIST/vendor/msgpack" \( -name '*.d.ts' -o -name '*.map' -o -name '*.tsbuildinfo' \) -delete
# Fonts: the 4 weights used by style.css, in woff2 only. Any browser capable
# of WebAssembly and WebGL2 can read it, and .woff would double the weight
# for no one.
for w in 400 500 600 700; do
  cp "$NM/@fontsource/ibm-plex-mono/$w.css" "$DIST/vendor/fontsource/"
  cp "$NM/@fontsource/ibm-plex-mono/files/"*-"$w"-normal.woff2 "$DIST/vendor/fontsource/files/"
done

# 4. Rewriting index.html: importmap pointing at vendor/, and switch to WASM mode.
echo "▸ index.html"
python3 - "$DIST/index.html" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()

s = s.replace("./node_modules/@fontsource/ibm-plex-mono/", "./vendor/fontsource/")
s = s.replace('"three": "./node_modules/three/build/three.module.js"',
              '"three": "./vendor/three/three.module.js"')
s = s.replace('"three/addons/": "./node_modules/three/examples/jsm/"',
              '"three/addons/": "./vendor/three/addons/"')
s = s.replace('"@msgpack/msgpack": "./node_modules/@msgpack/msgpack/dist.esm/index.mjs"',
              '"@msgpack/msgpack": "./vendor/msgpack/index.mjs"')

# Hook left by L2 (frontend/transport/index.js): without a server, the
# front end must start on the WASM module.
s = s.replace('  <script type="importmap">',
              '  <!-- Static build: no server behind it, the simulation runs\n'
              '       in the browser (see frontend/transport/index.js). -->\n'
              '  <script>window.HEXSIM_MODE = "wasm";</script>\n'
              '  <script type="importmap">', 1)

assert "node_modules" not in s, "a reference to node_modules survived the rewrite"
assert 'window.HEXSIM_MODE = "wasm"' in s, "WASM mode was not injected"
p.write_text(s)
PY

# 5. Zero-bundler safety net: nobody resolves imports on our behalf. A
#    forgotten vendored file only shows up when the page loads, as a 404.
echo "▸ verifying imports"
python3 - "$DIST" <<'CHECK'
import pathlib, re, sys

dist = pathlib.Path(sys.argv[1])
# `from "./x.js"`, `import "./x.js"`, dynamic imports included.
pattern = re.compile(r"""(?:from|import)\s*\(?\s*['"](\.[^'"]+)['"]""")
manquants = []
for f in list(dist.rglob("*.js")) + list(dist.rglob("*.mjs")):
    for rel in pattern.findall(f.read_text(errors="ignore")):
        if not (f.parent / rel).resolve().exists():
            manquants.append(f"{f.relative_to(dist)} -> {rel}")

if manquants:
    print("X unresolved imports:", file=sys.stderr)
    for m in manquants:
        print("   " + m, file=sys.stderr)
    sys.exit(1)
print("  ok - all relative imports resolve")
CHECK

echo
echo "✓ $DIST"
du -sh "$DIST" | awk '{printf "  total %s\n", $1}'
find "$DIST" -type f | wc -l | awk '{printf "  %s files\n", $1}'
