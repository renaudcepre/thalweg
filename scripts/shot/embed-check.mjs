// End-to-end check that an embed starts up correctly: shipped world loaded,
// playback running, no page error.
//
// There is no other honest way to check it. The three embed bugs of the
// #142-#144 series all had in common that they were invisible from the code
// (a command sent before the connection is up vanishes without an error) and
// that they only showed up by watching what the host watches: the stream of
// `tick` events.
//
// The first `tick` is not enough: in WASM mode the worker renders a snapshot
// of the fresh world before the checkpoint import is done (~300 ms). So we
// watch a window, not an instant.
//
// Usage (server running):
//   node scripts/shot/embed-check.mjs                    # WASM, the real path of an embed
//   node scripts/shot/embed-check.mjs '...?chrome=none'  # through the WS server
//   node scripts/shot/embed-check.mjs '...&world=neuf'            # without the shipped world
//
// Exits 1 if the page raised an error or if no tick arrived.
import { chromium } from "playwright";

const EMBED_URL =
  process.argv[2] ?? "http://localhost:8355/?mode=wasm&chrome=none";
const WINDOW_MS = Number(process.argv[3] ?? 20000);
const EXPECT_AGED = !EMBED_URL.includes("world=neuf");
// `?world=neuf` means "do not import the shipped world", not "reset". In WS
// mode the world lives in the server, which keeps what it had; an already
// aged server stays aged, and that is correct. Only WASM mode, where the front
// owns the world, lets us assert the opposite.
const WASM_MODE = EMBED_URL.includes("mode=wasm");

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 900, height: 600 } });

const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(e.message));

// The clipboard, made observable. This is the exact complaint from the
// playtest (#150): a click on the scene, the gesture that orbits the camera,
// wrote into the host page's clipboard. We cannot read it from Playwright
// without permissions, so we swap `writeText` for a recorder. A test that
// merely looked at whether the tooltip is visible would miss the point: the
// click does not consult visibility.
await page.addInitScript(() => {
  window.__clipWrites = [];
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: (t) => { window.__clipWrites.push(t); return Promise.resolve(); } },
  });
});

await page.goto(EMBED_URL, { waitUntil: "domcontentloaded" });

const ticks = await page.evaluate(
  (ms) =>
    new Promise((resolve) => {
      const seen = [];
      const t0 = performance.now();
      window.__hexsim.on("tick", (d) =>
        seen.push({ ...d, ms: Math.round(performance.now() - t0) }),
      );
      setTimeout(() => resolve(seen), ms);
    }),
  WINDOW_MS,
);

// --- API surface: views, overlays, persistence -----------------------------
//
// Same principle as above: we do not inspect the code, we do what a host does
// and look at what it gets back. The lists come from the API (no hard-coded id
// here, otherwise the test validates its own copy), and every command is
// judged on the `tick` that follows, never on the absence of an error.
const api = await page.evaluate(async () => {
  const sim = window.__hexsim;
  const waitNextTick = () =>
    new Promise((resolve) => {
      let first = true;
      const off = sim.on("tick", (d) => {
        // `on` fires back immediately with the last known tick: that one is
        // the state from BEFORE the command, we want the next one.
        if (first) { first = false; return; }
        off();
        resolve(d);
      });
    });

  const apply = async (fn) => {
    const p = waitNextTick();
    fn();
    return p;
  };

  // The current state, without waiting for anything: `on` fires back
  // immediately with the last known tick, so we subscribe and unsubscribe
  // right away.
  const readState = () => {
    let seen = null;
    const off = sim.on("tick", (d) => { seen = d; });
    off();
    return seen;
  };

  const views = sim.views;
  const layers = sim.layers;

  // A view other than the current one, taken from the list the API gives.
  const other = views.find((v) => v !== readState().view);
  const afterView = await apply(() => sim.view(other));

  // Unknown id: turned down, and the `tick` must keep reporting the view that
  // is actually on screen. A silent clamp would slip through here (#147).
  sim.view("this-view-does-not-exist");
  const afterReject = readState();

  // Toggling an overlay.
  const layer = layers[0];
  const wasOn = afterView.layers.includes(layer);
  const afterLayer = await apply(() => sim.layer(layer, !wasOn));

  // Persistence: export, browser gzip, re-import of the same buffer.
  const t0 = performance.now();
  const buf = await sim.export();
  const msExport = Math.round(performance.now() - t0);

  const gz = await new Response(
    new Response(buf).body.pipeThrough(new CompressionStream("gzip")),
  ).arrayBuffer();

  const t1 = performance.now();
  await sim.load(buf);
  const msLoad = Math.round(performance.now() - t1);
  const afterLoad = await waitNextTick();

  return {
    views,
    layers,
    viewRequested: other,
    viewApplied: afterView.view,
    viewAfterUnknownId: afterReject.view,
    layerToggled: layer,
    layerBefore: wasOn,
    layerAfter: afterLayer.layers.includes(layer),
    exportBytes: buf.byteLength,
    exportGzipBytes: gz.byteLength,
    msExport,
    msLoad,
    // The buffer returned by `export` must still be readable after `load`:
    // if it were transferred to the Worker, it would be detached here.
    bufferSurvivesLoad: buf.byteLength > 0,
    tickAfterLoad: afterLoad.tick,
  };
});

// --- Cell inspector: off in embed, can be switched back on (#150) ---------
//
// Two distinct properties, and the second one is what hurt in playtest: the
// tooltip does not show, AND the click never reaches the clipboard. They are
// tested separately because they drifted apart; the CSS was hiding (thought it
// was hiding) the element while the `click` listener kept writing to the
// clipboard.
//
// Hovering is done by sweeping: nothing guarantees that a given point of the
// window lands on a hexagon, and a tooltip missing because we aimed at empty
// space would be a false green. We ask for "at least one point" when on and
// "no point" when off.
const POINTS = [[450, 300], [400, 260], [500, 340], [380, 320], [520, 270]];

const hover = async () => {
  for (const [x, y] of POINTS) {
    await page.mouse.move(x, y);
    const visible = await page.evaluate(
      () => getComputedStyle(document.getElementById("tooltip")).display !== "none",
    );
    if (visible) return [x, y];
  }
  return null;
};

// No unsubscribe: `sim.on` fires back immediately, so the callback runs before
// `on` has returned its unsubscribe function; reading `off` there would hit
// its temporal dead zone (the v0.11.1 trap). A test page that closes within
// the second does not need to clean up.
const waitInspect = (expected) =>
  page.evaluate((exp) => new Promise((resolve) => {
    const t = setTimeout(() => resolve("no tick"), 8000);
    window.__hexsim.on("tick", (d) => {
      if (d.inspect === exp) { clearTimeout(t); resolve(d.inspect); }
    });
  }), expected);

await page.evaluate(() => { window.__clipWrites.length = 0; });
const hoverOff = await hover();
await page.mouse.click(...POINTS[0]);
const clipWritesOff = await page.evaluate(() => window.__clipWrites.length);
const tickOff = await waitInspect(false);

await page.evaluate(() => window.__hexsim.inspect(true));
const hoverOn = await hover();
await page.mouse.click(...(hoverOn ?? POINTS[0]));
const clipWritesOn = await page.evaluate(() => window.__clipWrites.length);
const tickOn = await waitInspect(true);

// Switch it back off while a tooltip is on screen: no `mousemove` will come to
// refresh it any more, so switching off is what has to clear it.
await page.evaluate(() => window.__hexsim.inspect(false));
const tooltipAfterOff = await page.evaluate(
  () => getComputedStyle(document.getElementById("tooltip")).display !== "none",
);

api.inspector = {
  hoverOff: Boolean(hoverOff),
  clipWritesOff,
  tickOff,
  hoverOn: Boolean(hoverOn),
  clipWritesOn,
  tickOn,
  tooltipAfterOff,
};

// A foreign buffer being turned down, kept apart and last: in WS mode it goes
// through a POST that the server answers with a 400, and Chrome logs every 4xx
// as a page error. Counting errors BEFORE avoids failing the test on the 400
// it causes itself.
const errorsBeforeReject = errors.length;
const rejection = await page.evaluate(async () => {
  const sim = window.__hexsim;
  const currentTick = () => {
    let seen = null;
    const off = sim.on("tick", (d) => { seen = d; });
    off();
    return seen.tick;
  };
  const before = currentTick();
  let message = null;
  try {
    await sim.load(new Uint8Array([1, 2, 3, 4]).buffer);
  } catch (e) {
    message = String(e?.message ?? e);
  }
  // The world must be intact: same tick, or further along if it kept running.
  // Compared, not tested for truth; a fresh world sits at tick 0.
  return { message, tickBefore: before, tickAfter: currentTick() };
});
api.foreignBufferRejected = rejection.message;
api.tickAfterReject = rejection.tickAfter;
api.worldIntactAfterReject = rejection.tickAfter >= rejection.tickBefore;

// --- The API seen from a REAL host page --------------------------------------
//
// Everything above tests the embed page by calling it from itself. That is
// enough for almost everything, and it has one exact blind spot: the objects
// exchanged then belong to the same JavaScript realm. A host builds its own
// objects in ITS realm and passes them across the iframe boundary.
//
// That is what let the v0.11.0 bug through: `sim.load(await sim.export())`
// worked (the buffer came from the iframe) while the real use, a buffer read
// back from IndexedDB by the host, failed on a cross-realm `instanceof`. The
// happy path and the real path were not the same path.
//
// Hence this section: a synthetic host page served from the SAME origin (via
// `route`, so nothing to ship in `dist/`), an iframe inside it, and objects
// built on the host side.
const HOST_URL = new URL("/__embed-host-test.html", EMBED_URL).href;
const host = await browser.newPage({ viewport: { width: 900, height: 600 } });
const hostErrors = [];
host.on("pageerror", (e) => hostErrors.push(e.message));

await host.route("**/__embed-host-test.html", (r) =>
  r.fulfill({
    contentType: "text/html; charset=utf-8",
    body: `<!doctype html><meta charset="utf-8"><title>host</title>
<body style="margin:0;background:#123">
<iframe id="sim" src="${EMBED_URL}" style="width:900px;height:520px;border:0"></iframe>`,
  }),
);
await host.goto(HOST_URL, { waitUntil: "domcontentloaded" });

const crossRealm = await host.evaluate(async (expectAged) => {
  const frame = document.getElementById("sim");
  await new Promise((r) => {
    if (frame.contentWindow?.__hexsim) r();
    else frame.addEventListener("load", r, { once: true });
  });
  const sim = frame.contentWindow.__hexsim;
  await new Promise((r) => sim.on("ready", r));

  // Wait for the shipped world: it is what gives a realistic checkpoint.
  //
  // `let off = null` before subscribing, and never unsubscribe FROM the
  // callback: `sim.on` fires back immediately with the last known tick, so the
  // callback runs before `sim.on` has returned anything. With a
  // `const off = sim.on(...)` we read `off` in its temporal dead zone.
  // Invisible in WASM, where the iframe has just started and has no aged world
  // yet; immediate in WS, where the server already has one.
  if (expectAged) {
    await new Promise((r) => {
      let off = null;
      let seen = false;
      off = sim.on("tick", (d) => {
        if (d.tick > 1000) { seen = true; r(); }
      });
      if (seen) off();
      setTimeout(() => { off?.(); r(); }, 15000);
    });
  }

  const src = await sim.export();

  // The point of the test: an ArrayBuffer built in THIS page's realm, not in
  // the iframe's. That is what an `idb.get()` hands back on a host.
  const hostBuf = new ArrayBuffer(src.byteLength);
  new Uint8Array(hostBuf).set(new Uint8Array(src));

  const tryLoad = async (label, data) => {
    try {
      await sim.load(data);
      return [label, "ok"];
    } catch (e) {
      return [label, String(e?.message ?? e)];
    }
  };

  return Object.fromEntries([
    await tryLoad("hostArrayBuffer", hostBuf),
    await tryLoad("hostTypedArray", new Uint8Array(hostBuf)),
    await tryLoad("hostBlob", new Blob([hostBuf])),
  ]);
}, EXPECT_AGED);

await browser.close();

// A shipped world is more than 1000 days old; a fresh world has none.
const switchTick = ticks.find((t) => t.tick > 1000);
const last = ticks[ticks.length - 1] ?? null;
const report = {
  url: EMBED_URL,
  tickCount: ticks.length,
  lastTick: last && { date: last.date, tick: last.tick, playing: last.playing },
  shippedWorld: Boolean(switchTick),
  switchMs: switchTick?.ms ?? null,
  api,
  crossRealm,
  errors,
};
console.log(JSON.stringify(report, null, 2));

const failures = [];
if (!ticks.length) failures.push("no tick received; the host would see nothing");
if (!last?.playing) failures.push("playback did not start");
if (EXPECT_AGED && !switchTick) failures.push("the shipped world was not loaded");
if (!EXPECT_AGED && switchTick && WASM_MODE) failures.push("world=neuf loaded a shipped world anyway");
if (!api.views.length) failures.push("`views` lists no base map");
if (!api.layers.length) failures.push("`layers` lists no overlay");
if (api.viewApplied !== api.viewRequested) {
  failures.push(`view("${api.viewRequested}") not applied (tick says "${api.viewApplied}")`);
}
if (api.viewAfterUnknownId !== api.viewRequested) {
  failures.push("an unknown view id changed the display instead of being turned down");
}
if (api.layerAfter === api.layerBefore) {
  failures.push(`layer("${api.layerToggled}") did not toggle`);
}
if (!api.exportBytes) failures.push("`export` returns an empty buffer");
if (!api.bufferSurvivesLoad) failures.push("`load` detached the caller's buffer");
if (!api.foreignBufferRejected) failures.push("a foreign buffer was accepted by `load`");
if (!api.worldIntactAfterReject) failures.push("the world did not survive a rejected `load`");
const insp = api.inspector;
if (insp.hoverOff) failures.push("the tooltip shows in embed while the inspector is off");
if (insp.clipWritesOff) failures.push(`${insp.clipWritesOff} write(s) to the host clipboard, inspector off`);
if (insp.tickOff !== false) failures.push(`the tick does not report the inspector as off (${insp.tickOff})`);
if (!insp.hoverOn) failures.push("inspect(true) did not bring the tooltip back");
if (insp.clipWritesOn <= insp.clipWritesOff) failures.push("inspect(true) did not restore copy-on-click");
if (insp.tickOn !== true) failures.push(`the tick does not report the inspector as on (${insp.tickOn})`);
if (insp.tooltipAfterOff) failures.push("inspect(false) leaves a tooltip on screen");
if (errorsBeforeReject) failures.push(`${errorsBeforeReject} page error(s)`);
// The three shapes the contract announces, built by a host in ITS realm. This
// is the only place in the test where the iframe boundary is real.
for (const [shape, verdict] of Object.entries(crossRealm)) {
  if (verdict !== "ok") failures.push(`load(${shape}) from a host page: ${verdict}`);
}

if (failures.length) {
  console.error("✗ " + failures.join(" ; "));
  process.exit(1);
}
console.error("✓ embed contract ok");
