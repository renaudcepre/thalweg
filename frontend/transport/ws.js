// WebSocket transport: the simulation runs in `hexsim-cli`.
//
// This is the daily dev mode: `just run`, and `hexsim-ctl` / `hexsim-mcp`
// talk to the same server in parallel with the front. The code below is
// what used to live in `main.js`, moved with no change in behavior.

const RECONNECT_MS = 2000;

export class WsTransport {
  constructor() {
    this.ws = null;
    this.onBinary = () => {};
    this.onJson = () => {};
    this.onStatus = () => {};
  }

  connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this.onStatus("connected", "WebSocket connected");
      this.send({ cmd: "meta" });
      // Sync UI (#2): aligns the sliders on the real server state as soon as
      // it connects; the static value/min/max in index.html are only
      // pre-connection placeholders.
      this.send({ cmd: "params" });
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.onBinary(event.data);
        return;
      }
      this.onJson(JSON.parse(event.data));
    };

    ws.onclose = () => {
      this.ws = null;
      this.onStatus("disconnected", "WebSocket disconnected");
      setTimeout(() => this.connect(), RECONNECT_MS);
    };

    ws.onerror = () => {
      this.onStatus("disconnected", "WebSocket error");
    };
  }

  send(cmd) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
    }
  }

  // Byte-for-byte the same as what `saveCheckpoint` downloads, and what
  // `loadCheckpoint` reads back: it's the response body, without the
  // download header.
  async exportCheckpoint() {
    const resp = await fetch("/checkpoint");
    if (!resp.ok) throw new Error(await resp.text());
    return resp.arrayBuffer();
  }

  // SAVE: GET /checkpoint → the server returns the state as a
  // Content-Disposition attachment; the browser downloads the .ckpt
  // (name provided by the server).
  saveCheckpoint() {
    const a = document.createElement("a");
    a.href = "/checkpoint";
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // LOAD: POST /checkpoint with the file as the body. On success, the
  // server broadcasts a fresh snapshot → the scene updates itself.
  // `fetch` accepts a `Blob`, an `ArrayBuffer`, or a view indifferently:
  // the three forms the embed API allows.
  async loadCheckpoint(data) {
    const resp = await fetch("/checkpoint", { method: "POST", body: data });
    if (!resp.ok) throw new Error(await resp.text());
  }
}
