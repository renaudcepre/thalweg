// Web Worker: the WASM simulation and its time loop.
//
// This file is the browser twin of `hexsim-cli/src/main.rs`. It plays the
// same role (run the world and route what the protocol asks for), with
// browser tools instead of tokio: a `setTimeout` instead of
// `spawn_tick_loop`, a `postMessage` instead of a `broadcast`.
//
// Why a Worker and not the main thread: a tick costs 17.5 ms at r120
// (measured in V8, milestone #138 L1). On the render thread, that's more
// than half a frame eaten every 30 ms. Here the sim runs alongside, and
// three.js keeps its thread to itself.
//
// This file knows no protocol rules: command names, bounds and formats
// live in `hexsim-proto`, translated by `hexsim-wasm`. Everything decided
// here is scheduling.

import init, { HexSim, batch_stride } from "../wasm/hexsim_wasm.js";

/** The simulation. `null` until the wasm is initialized. */
let sim = null;
/** Module memory, to report the world's actual footprint. */
let wasmMemory = null;

// Scheduling: the equivalent of `AppState`'s atomics. They don't describe
// the world, but how this shell makes it advance.
let paused = true;
let tickMs = 30;
let stepping = false;
let tickTimer = null;
let perfTimer = null;
/** Cost of the last tick (µs), republished on every perf sample. */
let lastTickUs = 0;

/** Sends a snapshot to the main thread without copying it. */
function postSnapshot(bytes) {
  self.postMessage({ type: "snapshot", buffer: bytes.buffer }, [bytes.buffer]);
}

/** JSON message (log, perf, progress, query response). */
function post(msg) {
  self.postMessage({ type: "json", payload: msg });
}

/**
 * Logs the way `tracing` would on the server side.
 *
 * The front's log panel consumes `tracing` events; in WASM there's no
 * subscriber, so we emit at the same points as the `info!` and `warn!`
 * calls in `main.rs`. The panel stays alive and says the same thing.
 */
function log(level, message, fields) {
  post({
    type: "log",
    level,
    message,
    fields,
    target: "hexsim_wasm",
    ts_ms: Date.now(),
  });
}

/**
 * Perf sample, in the server's format.
 *
 * Two of the three numbers don't exist in a browser: `cpu_percent` and
 * process RSS come from `sysinfo`, out of a page's reach. `cpu_percent`
 * therefore defaults to `null` (the front shows "—" rather than making
 * something up), and the reported memory is the WASM module's: that's
 * where the world actually lives, and it's an exact measurement rather
 * than a heap approximation.
 */
function postPerf(tickUs) {
  post({
    type: "perf",
    cpu_percent: null,
    rss_mb: wasmMemory ? wasmMemory.buffer.byteLength / 1048576 : null,
    tick_us: tickUs,
    tick: 0,
  });
}

/**
 * Samples at 1 Hz, like `spawn_perf_task` on the server side.
 *
 * Emitting perf only from the tick loop would leave the panel empty as
 * long as the world is paused, and it starts paused, which is exactly
 * when we want to see how much the world we just generated weighs.
 */
function startPerfLoop() {
  clearInterval(perfTimer);
  perfTimer = setInterval(() => postPerf(lastTickUs), 1000);
}

/** Yields back to the Worker's event loop. */
function yieldToLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// --- Auto-tick loop ---------------------------------------------------------

function scheduleTick() {
  clearTimeout(tickTimer);
  tickTimer = setTimeout(runTick, tickMs);
}

/**
 * One auto-tick cycle: advances by one simulated **hour** and emits the
 * snapshot.
 *
 * The hour and not the day, for the same reason as the server (#47): a
 * daily `step()` only emits a snapshot at midnight, and the front thinks
 * it's an endless night.
 */
function runTick() {
  if (!sim || paused || stepping) {
    scheduleTick();
    return;
  }
  const t0 = performance.now();
  const bytes = sim.step_hour();
  lastTickUs = Math.round((performance.now() - t0) * 1000);
  postSnapshot(bytes);
  scheduleTick();
}

/**
 * Long advance, split into batches: the transposition of `run_stepped`.
 *
 * The stride comes from `batch_stride()` exported by the wasm: it's the
 * server's, so the progress bar advances the same way in both modes. The
 * loop itself is specific to this shell: we yield between two batches so
 * the Worker stays reachable (a pause sent mid `+1 year` must get
 * through) and so snapshots go out as they come.
 *
 * `stepping` disables the auto-tick for the duration of the job, like on
 * the server side.
 */
async function runAdvance(n, hourly) {
  stepping = true;
  const stride = batch_stride(n);
  let done = 0;
  while (done < n) {
    const batch = Math.min(stride, n - done);
    sim.advance(batch, hourly);
    done += batch;
    postSnapshot(sim.snapshot());
    // `tick` is deliberately absent: reading it would require decoding
    // the snapshot here, at every step. The front falls back on the tick
    // of the last snapshot rendered.
    post({ type: "progress", done, total: n, finished: done >= n });
    await yieldToLoop();
  }
  stepping = false;
}

// --- Envelope translation ---------------------------------------------------

/**
 * Executes what the protocol asks for: the exact counterpart of
 * `run_outcome`.
 *
 * The `switch` operates on the `Envelope` variants produced by
 * `hexsim-wasm`, not on command names: commands are `hexsim-proto`'s to
 * know.
 */
async function runEnvelope(env) {
  switch (env.kind) {
    case "nothing":
      break;
    case "snapshot":
      postSnapshot(sim.snapshot());
      break;
    // `reply` and `broadcast` differ by their recipient on the server
    // side (one client / all). In a tab there's only one: same path.
    case "reply":
    case "broadcast":
      post(env.payload);
      break;
    case "play":
      paused = false;
      log("INFO", "play");
      break;
    case "pause":
      paused = true;
      log("INFO", "pause");
      break;
    case "speed":
      if (env.tick_ms !== env.requested) {
        log("WARN", "speed: value out of bounds, clamped", {
          requested: env.requested,
          applied: env.tick_ms,
        });
      }
      tickMs = env.tick_ms;
      log("INFO", "speed", { tick_ms: env.tick_ms });
      scheduleTick();
      break;
    case "advance":
      await runAdvance(env.n, env.hourly);
      break;
    case "error":
      log("WARN", env.message);
      break;
    default:
      log("WARN", "enveloppe inconnue", { kind: env.kind });
  }
}

// --- Receiving orders from the main thread ----------------------------------

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    switch (msg.op) {
      case "boot": {
        const mod = await init();
        wasmMemory = mod?.memory ?? null;
        sim = new HexSim(msg.seed, msg.radius);
        log("INFO", "monde genere", { seed: sim.seed(), radius: sim.radius() });
        self.postMessage({ type: "ready" });
        postSnapshot(sim.snapshot());
        scheduleTick();
        startPerfLoop();
        break;
      }
      case "command": {
        if (!sim) return;
        // Logs long-running commands before executing them, like
        // `log_started`: resetting a large world takes several hundred
        // ms, and a log that only arrives afterward doesn't say it's
        // working.
        if (msg.cmd.cmd === "reset") {
          log("INFO", "reset", { seed: msg.cmd.seed ?? null });
        } else if (msg.cmd.cmd === "step" || msg.cmd.cmd === "step_hour") {
          log("INFO", msg.cmd.cmd, { n: msg.cmd.n ?? 1 });
        }
        const env = JSON.parse(sim.command(JSON.stringify(msg.cmd)));
        // `set_param` is only logged once the key is accepted: otherwise
        // a misleading INFO would precede the "unknown parameter" WARN.
        if (msg.cmd.cmd === "set_param" && env.kind !== "error") {
          log("INFO", "set_param", { key: msg.cmd.key, value: msg.cmd.value });
        }
        await runEnvelope(env);
        break;
      }
      case "export": {
        if (!sim) return;
        const bytes = sim.export_checkpoint();
        self.postMessage({ type: "checkpoint", buffer: bytes.buffer }, [bytes.buffer]);
        break;
      }
      case "import": {
        if (!sim) return;
        // An invalid blob is rejected without touching the current
        // world: that's `import_checkpoint`'s contract, identical to the
        // server's 400.
        sim.import_checkpoint(new Uint8Array(msg.buffer));
        log("INFO", "import checkpoint", { size: msg.buffer.byteLength });
        self.postMessage({ type: "imported" });
        postSnapshot(sim.snapshot());
        break;
      }
      default:
        log("WARN", "ordre inconnu", { op: msg.op });
    }
  } catch (e) {
    const message = String(e?.message ?? e);
    log("ERROR", message);
    self.postMessage({ type: "failed", op: msg.op, message });
  }
};
