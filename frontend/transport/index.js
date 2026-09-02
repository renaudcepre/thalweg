// Transport choice: WebSocket server or local WASM module.
//
// **The default is WebSocket**, and it stays that way: it's the dev mode,
// the one `just run` opens, and `hexsim-ctl` / `hexsim-mcp` depend on it.
// WASM mode only activates when explicitly requested.
//
// Three ways to request it, from most local to most general:
//
//   1. `?mode=wasm` in the URL, to compare the two modes in dev without
//      rebuilding anything. `?mode=ws` forces the reverse.
//   2. `window.HEXSIM_MODE`, injected by a `<script>` in `index.html`.
//      This is the static build's hook point (#138 L3): a page deployed
//      without a server sets it to `"wasm"`.
//   3. nothing: WebSocket.
//
// `?radius=` and `?seed=` only apply to WASM mode: without a server, there
// is no `hexsim.toml` to supply them. The default is radius 45 (~6200
// cells, 3.1 ms/tick) rather than the local config file's 120: a web page
// opens onto a world that's running right away.

import { WsTransport } from "./ws.js";
import { WasmTransport } from "./wasm.js";

const DEFAULT_WASM_RADIUS = 45;
const DEFAULT_WASM_SEED = 42;

// Randomly drawn seed, within the bounds of the `u32` the engine expects.
// Used in bare mode: on a third-party site, each visitor opens a world
// that's their own, whereas the full mode keeps 42 to stay reproducible.
function randomSeed() {
  return Math.floor(Math.random() * 4294967296);
}

export function resolveMode(search = location.search) {
  const requested = new URLSearchParams(search).get("mode");
  if (requested === "wasm" || requested === "ws") return requested;
  return window.HEXSIM_MODE === "wasm" ? "wasm" : "ws";
}

export function createTransport(search = location.search) {
  if (resolveMode(search) !== "wasm") return new WsTransport();

  const params = new URLSearchParams(search);
  const int = (key, fallback) => {
    const n = parseInt(params.get(key), 10);
    return Number.isFinite(n) ? n : fallback;
  };
  const minimal = params.get("chrome") === "minimal";
  return new WasmTransport({
    seed: int("seed", minimal ? randomSeed() : DEFAULT_WASM_SEED),
    radius: int("radius", DEFAULT_WASM_RADIUS),
  });
}
