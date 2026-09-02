// WASM transport: the simulation runs in a Web Worker, inside the tab.
//
// On the front side, nothing distinguishes this transport from the
// WebSocket one: same JSON commands in, same msgpack snapshots out. The
// WASM module runs the same `hexsim-proto` as the server, so the two
// modes can't diverge on what a command is or on a snapshot's format.
//
// The difference is one of nature, not of protocol: there's no more
// network, hence no more disconnection, no more reconnection, and a
// single client.

/**
 * Returns an `ArrayBuffer` we own, hence transferable to the Worker
 * without neutralizing the caller's. A `Blob` already gives us a fresh
 * one; a buffer supplied by a host is copied.
 *
 * **Structural typing, never `instanceof`.** A host that passes us an
 * `ArrayBuffer` built it in ITS realm, the page's; ours is the iframe's,
 * and `data instanceof ArrayBuffer` returns `false` there. v0.11.0 then
 * fell through to the `Blob` branch and called `arrayBuffer()` on a
 * buffer: `data.arrayBuffer is not a function`, on the most ordinary
 * case there is, a state reloaded from IndexedDB. Reported by the
 * integrator.
 *
 * The trap is that the test path passed: `load(await export())` gets a
 * buffer born in the iframe, so from the right realm. Only a genuine
 * boundary crossing reveals the failure; that's what the last section of
 * `just embed-check` verifies.
 *
 * `ArrayBuffer.isView`, on the other hand, crosses realms: that's why
 * typed views already worked.
 */
async function toOwnedBuffer(data) {
  // `Blob`/`File`: the only one of the three carrying an `arrayBuffer`
  // method.
  if (typeof data?.arrayBuffer === "function") return data.arrayBuffer();
  if (ArrayBuffer.isView(data)) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  // What remains is an `ArrayBuffer`, from here or elsewhere: `slice`
  // makes a copy that we own. Methods, though, work cross-realm.
  if (typeof data?.slice === "function" && typeof data?.byteLength === "number") {
    return data.slice(0);
  }
  throw new TypeError(
    "__hexsim.load expects an ArrayBuffer, a typed view, or a Blob, " +
      `received ${Object.prototype.toString.call(data)}`,
  );
}

export class WasmTransport {
  constructor({ seed = 42, radius = 45 } = {}) {
    this.worker = null;
    this.seed = seed;
    this.radius = radius;
    this.pendingImport = null;
    // FIFO: the Worker processes orders serially, responses come back in
    // emission order. A queue rather than a single slot: two concurrent
    // `export()` calls (a host + the SAVE button) must not steal each
    // other's response.
    this.pendingExports = [];
    this.onBinary = () => {};
    this.onJson = () => {};
    this.onStatus = () => {};
  }

  connect() {
    this.onStatus("connecting", "Loading WASM module…");
    this.worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });

    this.worker.onmessage = (event) => {
      const msg = event.data;
      switch (msg.type) {
        case "ready":
          this.onStatus("connected", `Simulation locale (WASM) · rayon ${this.radius}`);
          // The server pushes `meta` and `params` when the socket opens;
          // we do the same, so the version badge and the sliders align
          // through the same path in both modes.
          this.send({ cmd: "meta" });
          this.send({ cmd: "params" });
          break;
        case "snapshot":
          this.onBinary(msg.buffer);
          break;
        case "json":
          this.onJson(msg.payload);
          break;
        case "checkpoint":
          this.pendingExports.shift()?.resolve(msg.buffer);
          break;
        case "imported":
          this.pendingImport?.resolve();
          this.pendingImport = null;
          break;
        case "failed":
          // A failed import must surface to the caller, who shows the
          // alert. Other failures are already logged by the Worker.
          if (msg.op === "import" && this.pendingImport) {
            this.pendingImport.reject(new Error(msg.message));
            this.pendingImport = null;
          } else if (msg.op === "export") {
            this.pendingExports.shift()?.reject(new Error(msg.message));
          }
          break;
        default:
          break;
      }
    };

    // A Worker that dies (OOM on a very large world, unreadable module)
    // would leave the front frozen without saying anything.
    this.worker.onerror = (e) => {
      this.onStatus("disconnected", `Worker WASM en erreur : ${e.message}`);
    };

    this.worker.postMessage({ op: "boot", seed: this.seed, radius: this.radius });
  }

  send(cmd) {
    this.worker?.postMessage({ op: "command", cmd });
  }

  /**
   * The complete state of the world, as `loadCheckpoint` picks it back
   * up.
   *
   * Serialization is synchronous **inside the Worker**: the simulation
   * doesn't tick during that time (rendering, meanwhile, lives on the
   * main thread and stays smooth). A 42-year-old world weighs about 50
   * MB, see EMBED.md for the measured cost.
   */
  exportCheckpoint() {
    return new Promise((resolve, reject) => {
      this.pendingExports.push({ resolve, reject });
      this.worker?.postMessage({ op: "export" });
    });
  }

  saveCheckpoint() {
    this.exportCheckpoint().then((buffer) => this.#downloadCheckpoint(buffer));
  }

  /**
   * Accepts a `Blob`/`File` (LOAD button, `fetch`) or an `ArrayBuffer` /
   * `TypedArray` (embed host reading back its own storage).
   *
   * The buffer goes to the Worker as a transfer, so it's neutralized on
   * the caller's side: the one the host hands us is **copied** first.
   * Silently making it unusable after a successful `load()` would be the
   * kind of bug you only find on the second call.
   */
  async loadCheckpoint(data) {
    const buffer = await toOwnedBuffer(data);
    return new Promise((resolve, reject) => {
      this.pendingImport = { resolve, reject };
      this.worker?.postMessage({ op: "import", buffer }, [buffer]);
    });
  }

  /**
   * The server names the file via `Content-Disposition`; without a
   * server, it's up to the front to do it. The timestamp stands in for
   * the tick, which the main thread doesn't know without decoding a
   * snapshot.
   */
  #downloadCheckpoint(buffer) {
    const url = URL.createObjectURL(new Blob([buffer], { type: "application/octet-stream" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `hexsim-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "")}.ckpt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
}
