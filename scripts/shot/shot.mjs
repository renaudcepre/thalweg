#!/usr/bin/env node
//
// hexsim-shot: Playwright screenshot of the HexSim frontend.
//
// The engine produces the state, the frontend (localhost:8355) renders it in
// 3D; this tool drives the camera deterministically via the `window.__hexcam`
// hook (frontend/main.js), then captures the WebGL canvas. Designed to be
// called by Claude from `just shot …` to *see* the simulation, not just read
// JSON diagnostics.
//
// The sim itself is driven separately with `just play|pause|step|reset` (the
// hexsim-cli server shares the same WebSocket): set the state, then capture.
//
// Usage:
//   node shot.mjs [options]
//
// Framing options (camera):
//   --zoom <r>          camera→target distance in world units (1 hex ≈ 1.0).
//                       Default: fitAll (whole map framed).
//   --zoom-factor <f>   multiplies the fitAll distance: <1 zooms in, >1 zooms out.
//   --azimuth <deg>     rotation around the vertical axis (0 = from the south).
//   --polar <deg>       tilt: 0 = top-down view, 90 = grazing angle.
//   --top               shortcut: --polar 1 (top-down map view).
//   --target "x,z"      recenters the camera on this world point.
//
// Scene options:
//   --view <mode>       map background: terrain | species | permeability |
//                       groundwater | humidity_upper | humidity_surface.
//   --wind, --clouds, --temperature, --precipitation   enables the overlay.
//   --no-clouds         disables clouds (checked by default).
//   --clean             hides both sidebars ("bare map" capture).
//
// Output options:
//   --out <path>        PNG file (default: out/shot-<tick>-<ts>.png).
//   --width <px>        viewport width (default 1600).
//   --height <px>       viewport height (default 1000).
//   --scale <n>         deviceScaleFactor / pixel density (default 1).
//   --settle <ms>       wait before capture, to let the scene settle
//                       (default 600).
//   --canvas-only       captures only the <canvas> (ignores the surrounding DOM).
//   --url <url>         default http://localhost:8355.
//   --timeout <ms>      max wait for the __hexcam hook to be ready (default 20000).
//   --shot-timeout <ms> max wait for the capture itself (default 120000;
//                       large worlds in headless software GL render slowly).
//
// Stdout output: one JSON line { path, tick, hourTick, view, bytes }.

import { chromium } from "playwright";
import { mkdir, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

// --- parsing minimal -------------------------------------------------------
function parseArgs(argv) {
  const flags = new Set([
    "top",
    "clean",
    "canvas-only",
    "wind",
    "clouds",
    "no-clouds",
    "temperature",
    "precipitation",
    "help",
  ]);
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    let a = argv[i];
    if (!a.startsWith("--")) continue;
    a = a.slice(2);
    if (flags.has(a)) {
      o[a] = true;
    } else {
      o[a] = argv[++i];
    }
  }
  return o;
}

const args = parseArgs(process.argv.slice(2));

if (args.help) {
  // File header = doc; we point the user there.
  console.error("hexsim-shot, see the header of scripts/shot/shot.mjs (--help).");
  process.exit(0);
}

// Don't name this constant `URL`: it would shadow the global `new URL(...)`
// constructor used further below (bug seen in the Phase 2 synoptic session).
const baseUrl = args.url ?? "http://localhost:8355";
const width = Number(args.width ?? 1600);
const height = Number(args.height ?? 1000);
const scale = Number(args.scale ?? 1);
const settle = Number(args.settle ?? 600);
const readyTimeout = Number(args.timeout ?? 20000);
const shotTimeout = Number(args["shot-timeout"] ?? 120_000);

const num = (v) => (v === undefined ? undefined : Number(v));
const parseTarget = (v) => {
  if (!v) return undefined;
  const [x, z] = v.split(",").map(Number);
  return { x, z };
};

/**
 * Launches the browser, falling back to the system Chrome.
 *
 * Playwright pins a specific build of its chromium; a package update is
 * enough to make it disappear, and `npx playwright install` can't always
 * re-download it (restricted network). Rather than blocking every capture
 * on that, we fall back to the Chrome installed on the machine.
 */
async function launchBrowser() {
  const args = ["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"];
  // arm64 machine workaround: the Playwright cache contains an incomplete
  // chromium-1194 x64; HEXSIM_SHOT_CHROMIUM points to a working binary
  // (e.g. chromium_headless_shell-1234 .../chrome-headless-shell).
  const exe = process.env.HEXSIM_SHOT_CHROMIUM;
  if (exe) return await chromium.launch({ executablePath: exe, args });
  try {
    return await chromium.launch({ args });
  } catch (e) {
    if (!/Executable doesn't exist/.test(String(e?.message))) throw e;
    console.error("[shot] Playwright chromium missing, falling back to system Chrome");
    return await chromium.launch({ channel: "chrome", args });
  }
}

async function main() {
  const browser = await launchBrowser();
  const page = await browser.newPage({
    viewport: { width, height },
    deviceScaleFactor: scale,
  });

  // Surfaces page errors so we don't capture a silently broken scene.
  page.on("pageerror", (e) => console.error("[page error]", e.message));

  // `?capture` enables preserveDrawingBuffer on the frontend side: without
  // it, the WebGL buffer is cleared on swap and the capture comes out black
  // (the frontend disables it by default because it hurts interactive
  // rendering performance, see main.js).
  const captureUrl = new URL(baseUrl);
  captureUrl.searchParams.set("capture", "1");
  await page.goto(captureUrl.href, { waitUntil: "domcontentloaded" });

  // Wait for the hook to be ready: terrain built + first snapshot received.
  await page.waitForFunction(() => window.__hexcam && window.__hexcam.ready(), {
    timeout: readyTimeout,
  });

  // --- overlays / map background -------------------------------------------
  if (args.view) {
    await page.selectOption("#view-mode", args.view).catch(() => {});
  }
  const setToggle = async (id, on) => {
    await page
      .evaluate(
        ([sel, val]) => {
          const el = document.getElementById(sel);
          if (el && el.checked !== val) el.click();
        },
        [id, on],
      )
      .catch(() => {});
  };
  if (args.wind) await setToggle("toggle-wind", true);
  if (args.temperature) await setToggle("toggle-temperature", true);
  if (args.precipitation) await setToggle("toggle-precipitation", true);
  if (args["no-clouds"]) await setToggle("toggle-clouds", false);
  else if (args.clouds) await setToggle("toggle-clouds", true);

  if (args.clean) {
    await page.evaluate(() => window.__hexcam.setChromeVisible(false));
  }

  // Lets the scene rebuild after an overlay/background change.
  await page.waitForTimeout(250);

  // --- camera framing -------------------------------------------------------
  const camOpts = {
    zoom: num(args.zoom),
    zoomFactor: num(args["zoom-factor"]),
    azimuthDeg: num(args.azimuth),
    polarDeg: args.top ? 1 : num(args.polar),
    target: parseTarget(args.target),
  };

  const view = await page.evaluate((opts) => {
    const cam = window.__hexcam;
    // Base: frames the whole map, unless an absolute radius is requested.
    if (opts.zoom === undefined) cam.fitAll();
    const set = {};
    if (opts.zoom !== undefined) set.radius = opts.zoom;
    if (opts.azimuthDeg !== undefined) set.azimuthDeg = opts.azimuthDeg;
    if (opts.polarDeg !== undefined) set.polarDeg = opts.polarDeg;
    if (opts.target) set.target = opts.target;
    if (Object.keys(set).length) cam.setView(set);
    if (opts.zoomFactor !== undefined) cam.zoom(opts.zoomFactor);
    return cam.view();
  }, camOpts);

  // Wait for stabilization (OrbitControls damping, particles, clouds),
  // then a synchronous render right before the capture.
  await page.waitForTimeout(settle);
  const meta = await page.evaluate(() => {
    window.__hexcam.render();
    return { ...window.__hexcam.state(), cover: window.__hexcam.cover() };
  });

  // --- output ----------------------------------------------------------------
  const tick = meta?.tick ?? "na";
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const out = resolve(args.out ? args.out : `${HERE}/out/shot-${tick}-${ts}.png`);
  await mkdir(dirname(out), { recursive: true });

  if (args["canvas-only"]) {
    // #scene-canvas = the scene's WebGL canvas (not the #compass canvas).
    const canvas = await page.$("#scene-canvas");
    await canvas.screenshot({ path: out, timeout: shotTimeout });
  } else {
    await page.screenshot({ path: out, timeout: shotTimeout });
  }

  await browser.close();

  const { size } = await stat(out);
  console.log(
    JSON.stringify({
      path: out,
      tick: meta?.tick ?? null,
      hourTick: meta?.hourTick ?? null,
      view,
      cover: meta?.cover ?? null,
      bytes: size,
    }),
  );
}

main().catch((e) => {
  console.error("hexsim-shot failed:", e.message);
  console.error(
    "Is the server running? (just run / just rebuild) And is Playwright installed? (just shot-setup)",
  );
  process.exit(1);
});
