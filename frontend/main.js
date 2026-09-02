import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { decode as msgpackDecode } from "@msgpack/msgpack";
import { appendLog, initLogPanel } from "./logs.js";
import { createTransport, resolveMode } from "./transport/index.js";

// --- DOM refs ---
const tickCountEl = document.getElementById("tick-count");
const wsStatusEl = document.getElementById("ws-status");
const statSurface = document.getElementById("stat-surface");
const statHumidity = document.getElementById("stat-humidity");
const statCloud = document.getElementById("stat-cloud");
const statRainMm = document.getElementById("stat-rain-mm");
const statGroundwater = document.getElementById("stat-groundwater");
const statSnow = document.getElementById("stat-snow");
const statTotal = document.getElementById("stat-total");
const viewModeEl = document.getElementById("view-mode");

// --- Constants ---
const SQRT3 = Math.sqrt(3);
const HEX_SIZE = 1.0; // size of a hex in 3D world units
// Real center-to-center distance (m) between two neighboring hexes. Comes
// from the core (CELL_SPACING_M, dynamics.rs) via the "meta" command on
// connect, NOT from the snapshots' wire protocol (anti-pattern #2: the front
// must never recompute/guess a physical value already known to the core).
// The value below is only a fallback until the "meta" response arrives.
let CELL_SPACING_M = 1074.569;
// A hex = ~1 km² (100 ha) by default. Real side length = spacing / sqrt(3).
let KM_PER_WORLD = (CELL_SPACING_M / 1000) / SQRT3 / HEX_SIZE;
// TRUE vertical scale (1:1, no exaggeration). 2 neighboring cells = √3
// world units = CELL_SPACING_M (core side) → 1 world unit = CELL_SPACING_M/√3
// meters horizontally. For a 1:1 render, 1 m of elevation = 1/(CELL_SPACING_M/√3) unit.
// (Before: 0.02, i.e. ×12.4 vertical exaggeration, removed: the world is gentle,
// it is displayed as-is.)
let ELEVATION_SCALE = SQRT3 / CELL_SPACING_M;

// Recomputes the constants derived from CELL_SPACING_M when the core
// announces a value different from the fallback ("meta" message).
function applyCellSpacingM(spacingM) {
  CELL_SPACING_M = spacingM;
  KM_PER_WORLD = (CELL_SPACING_M / 1000) / SQRT3 / HEX_SIZE;
  ELEVATION_SCALE = SQRT3 / CELL_SPACING_M;
  // The snow gain is pegged to ELEVATION_SCALE (see the SNOW_VIZ_GAIN block):
  // resync it here, otherwise it stays on the 1074 m fallback and snow
  // renders ×8.3 too high at 130 m.
  SNOW_VIZ_GAIN = snowVizGainFor(ELEVATION_SCALE);
  // If the first snapshot beat the "meta" response (race at connect), the
  // terrain was built with the fallback ELEVATION_SCALE and the view bounds
  // (max zoom, fog, km-to-world budget) with the wrong KM_PER_WORLD, and
  // nothing recomputed them (measured at r200: clamped to 32 wu instead of
  // 266). Invalidate both: geometry rebuilt on the next rebuild, bounds
  // recomputed right after.
  terrainSig = null;
  viewWorldWidth = 0;
  needsRebuild = true;
}
// Since Phase 3 (#34, rescale x200 stocks in mm), a water/snow height
// rendered at physical scale would be invisible (200 mm of water = 0.004 world y).
// Damped by 1/200 to recover the pre-rescale visual (2 visual meters
// for 200 real mm, accepting an amplification for legibility).
const STOCK_VISUAL = 1 / 200;

// Visual snow height (elevation units, pre-ELEVATION_SCALE), #18.
// The `snow_level` stock (mm) spans ~3 orders of magnitude (dusting 0.5 →
// glacier / cold sink 800+). The old `min(snow * 0.5 * STOCK_VISUAL, 5.0)`
// crushed it entirely under the anti-z-fight floor +0.02, regardless of the
// stock. log1p compresses the top and lifts the bottom; the gain aims for a
// "typical" stock (SNOW_TYPICAL_MM) to render SNOW_TYPICAL_WORLD world-units
// once multiplied by ELEVATION_SCALE.
//
// The gain is DERIVED from ELEVATION_SCALE, not fixed: otherwise it drifts
// out of sync as soon as CELL_SPACING_M changes. This happened: #18 froze it
// at 25 by pegging it to the old 1074 m spacing (ELEVATION_SCALE≈0.0016)
// while the runtime was already running at 130 m (≈0.0133) → snow rendered
// ×8.3 too high (100 m slabs to the eye). Recomputed in applyCellSpacingM
// when the core announces the spacing.
// Single source of truth for the 3 renders (snow mesh, evaporation haze, precip overlay).
const SNOW_TYPICAL_MM = 10; // "typical" snow stock, calibration reference
const SNOW_TYPICAL_WORLD = 0.1; // world-height targeted for SNOW_TYPICAL_MM
function snowVizGainFor(elevScale) {
  return SNOW_TYPICAL_WORLD / (Math.log1p(SNOW_TYPICAL_MM) * elevScale);
}
let SNOW_VIZ_GAIN = snowVizGainFor(ELEVATION_SCALE);
const snowVisualHeight = (snowLevel) => Math.log1p(Math.max(0, snowLevel || 0)) * SNOW_VIZ_GAIN;

// Cloud rendering window (mm of `cloud_water`), DERIVED from the core's
// autoconversion threshold (`atmosphere.precip_crit_mm`, received via the
// "params" broadcast on connect and on every set_param). The core accumulates
// cloud_water up to this threshold before triggering rain: it is THE ONE that
// sets the order of magnitude of the visible stocks. The old fixed window
// [0.04, 0.13] was pegged to the old 0.05 floor (CLOUD_MIN_PRECIP); once
// switched to crit=0.15, the 0.13 ceiling fell BELOW the threshold → any
// cloud heading toward rain rendered the same saturated white, with all the
// dynamics (loading, transport, purge) playing out outside the window, the
// same desync mechanism as the snow gain (#131). The 0.8×crit / 2.6×crit
// ratios reproduce exactly [0.04, 0.13] at crit=0.05: they capture the
// original intent, made robust to a change in regime. Fallback = current
// core default (0.15), until the "params" response arrives.
const CLOUD_WINDOW_MIN_RATIO = 0.8; // appearance threshold, in × precip_crit_mm
const CLOUD_WINDOW_FULL_RATIO = 2.6; // visual saturation, in × precip_crit_mm
let CLOUD_RENDER_MIN = 0.15 * CLOUD_WINDOW_MIN_RATIO;
let CLOUD_RENDER_FULL = 0.15 * CLOUD_WINDOW_FULL_RATIO;
// crit=0 = gate disabled on the core side → historical continuous regime: the
// original window (base 0.05) becomes the right one again. Returns true if
// the window moved.
function applyPrecipCritMm(critMm) {
  const ref = critMm > 0 ? critMm : 0.05;
  const min = ref * CLOUD_WINDOW_MIN_RATIO;
  if (min === CLOUD_RENDER_MIN) return false;
  CLOUD_RENDER_MIN = min;
  CLOUD_RENDER_FULL = ref * CLOUD_WINDOW_FULL_RATIO;
  return true;
}
// PHYSICAL mm→m conversion (= `units::MM_PER_M` on the core side) for the
// SURFACE HEIGHT of a lake. A lake's surface must be rendered at its real
// `effective_elevation` (elevation + surplus/1000), otherwise a lake that is
// physically flat across multiple hexes (#106, `lake::step_lake_leveling`)
// reappears stair-stepped: any factor ≠ 1/1000 amplifies the surplus that
// compensates the terrain and breaks the flatness (topY = elev + surplus·k;
// only k = 1/1000 gives a flat surface). The ×5 exaggeration of STOCK_VISUAL
// remains for the THICKNESS of sub-capacity puddles (invisible otherwise),
// not for a lake's free surface.
const WATER_SURFACE_M = 1 / 1000;
// Since v0.2.0 (physical units refactor, #29), rain_amount is already
// in mm/tick, no conversion needed.
// River display threshold: cell discharge (mm accumulated over the hydro
// slice) above which the cell draws its segment. Default 0.5 = the core
// diagnostics' `river_threshold`, a "river" in the diag = a segment on
// screen. Slider to 0 to see all runoff when debugging hydro.
let RIVER_THRESHOLD = 0.5;
// Ambient light multiplier (dev slider). 0 = no ambient (night = total
// darkness, only what the sun lights is visible); 1 = default; >1 = everything
// brightened. Fixes the inconsistency "night brighter than a late-afternoon
// shadow": at 0, night becomes as black as the shadow.
let AMBIENT_GAIN = 1.0;
// Two kinds of frames travel over the binary WS channel: grid snapshots
// (large: ~140 B/cell) and perf samples (~70 B, 1/s). They are told apart by
// size so snapshots are only decoded at rebuild time (see maybeRebuild). A
// frame bigger than this threshold is necessarily a snapshot; below it, it's
// perf (or a tiny world, decoded at no cost).
const PERF_FRAME_MAX_BYTES = 4096;

// Axial neighbors in the order of hexVertices' edges: edge i between vertex i and vertex (i+1)%6
const HEX_NEIGHBORS = [
  [1, 0],   // 0: E
  [0, 1],   // 1: SE
  [-1, 1],  // 2: SW
  [-1, 0],  // 3: W
  [0, -1],  // 4: NW
  [1, -1],  // 5: NE
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
// `tickJours` = simulated day (state.tick), `hourTick` (optional) = cumulative
// hour (state.hour_tick). If provided, the hour of day is appended in "HHh"
// format, useful since #47 where the diurnal cycle is observable second by
// second in the render.
function tickToDate(tickJours, hourTick) {
  const year = Math.floor(tickJours / 365);
  let dayOfYear = tickJours % 365;
  let month = 0;
  while (month < 11 && dayOfYear >= DAYS_PER_MONTH[month]) {
    dayOfYear -= DAYS_PER_MONTH[month];
    month++;
  }
  const date = `${dayOfYear + 1} ${MONTHS[month]} Year ${year}`;
  if (hourTick === undefined || hourTick === null) return date;
  const hod = hourTick % 24;
  return `${date} ${String(hod).padStart(2, "0")}h`;
}

// Speed: slider 0..100 → log mapping to simulated hours per second.
// Since #47 (diurnal cycle work), auto-tick advances 1 simulated hour per
// cycle (instead of 1 day before). The natural unit for the slider therefore
// becomes h/s. Range [1 h/s, 1000 h/s]: from contemplative (24 s for a full
// diurnal cycle) to fast scroll (~41 days/s).
const HPS_MIN = 1;
const HPS_MAX = 1000;

function hpsFromPos(pos) {
  const t = pos / 100;
  return HPS_MIN * Math.pow(HPS_MAX / HPS_MIN, t);
}

// Inverse of `hpsFromPos`. Used by the embed API: a host requests hours per
// second, the slider position is a detail of our UI it doesn't need to know
// about. The slider's integer step rounds the obtained value, hence the
// `speed` returned in the `tick` event, which gives what is actually applied
// rather than what was requested.
function posFromHps(hps) {
  const borne = Math.min(HPS_MAX, Math.max(HPS_MIN, hps));
  const t = Math.log(borne / HPS_MIN) / Math.log(HPS_MAX / HPS_MIN);
  return Math.round(t * 100);
}

function msFromHps(hps) {
  return Math.max(1, Math.round(1000 / hps));
}

// Display: `h/s` for low speeds (reading the diurnal cycle), switching
// to `d/s` above 24 h/s (= 1 d/s) where reading an annual cycle becomes
// more meaningful.
function formatHps(hps) {
  if (hps >= 24) {
    const dps = hps / 24;
    if (dps >= 100) return `${Math.round(dps)} d/s`;
    if (dps >= 10) return `${dps.toFixed(1)} d/s`;
    return `${dps.toFixed(2)} d/s`;
  }
  if (hps >= 10) return `${Math.round(hps)} h/s`;
  return `${hps.toFixed(1)} h/s`;
}

function windLabel(wx, wy) {
  const mag = Math.sqrt((wx || 0) * (wx || 0) + (wy || 0) * (wy || 0));
  if (mag < 0.01) return "calm";
  const angle = ((Math.atan2(wy, wx) * 180 / Math.PI) + 360) % 360;
  const dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"];
  const idx = Math.round(angle / 45) % 8;
  return `${mag.toFixed(2)} → ${dirs[idx]}`;
}

// --- Compass + weather ---
const compassCanvas = document.getElementById("compass");
const seasonLabel = document.getElementById("season-label");
const avgTempLabel = document.getElementById("avg-temp-label");
const avgWindLabel = document.getElementById("avg-wind-label");

function getSeason(tick) {
  const day = tick % 365;
  if (day < 80 || day >= 355) return { name: "Winter", color: "#6699ff" };
  if (day < 172) return { name: "Spring", color: "#66cc66" };
  if (day < 264) return { name: "Summer", color: "#eebb33" };
  return { name: "Autumn", color: "#cc8844" };
}

// Average wind stored across frames for continuous redraw
let lastAvgWind = { x: 0, y: 0 };
// Solar intensity memorized across frames to draw the sun on the
// compass even when the sim is paused.
let lastSunIntensity = 0.5;

// Sun in the top-right of the compass. Size, color and rays vary
// with solar intensity (season). It's a fixed indicator: it does not
// move, only its appearance changes.
function drawSunIndicator(ctx, sx, sy, intensity) {
  const haloR = 5 + 7 * intensity;
  const grad = ctx.createRadialGradient(sx, sy, 0.5, sx, sy, haloR);
  grad.addColorStop(0, `rgba(255, 230, 140, ${0.55 + 0.35 * intensity})`);
  grad.addColorStop(1, "rgba(255, 200, 80, 0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(sx, sy, haloR, 0, 2 * Math.PI);
  ctx.fill();

  const coreR = 2.0 + 1.8 * intensity;
  const r = 255;
  const g = Math.round(180 + 60 * intensity);
  const b = Math.round(70 + 120 * intensity);
  ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
  ctx.beginPath();
  ctx.arc(sx, sy, coreR, 0, 2 * Math.PI);
  ctx.fill();

  const rayCount = 8;
  const rayLen = 1.5 + 3.5 * intensity;
  ctx.strokeStyle = `rgba(255, 220, 100, ${0.35 + 0.55 * intensity})`;
  ctx.lineWidth = 1.0;
  ctx.lineCap = "round";
  for (let i = 0; i < rayCount; i++) {
    const a = (i / rayCount) * 2 * Math.PI;
    const x1 = sx + Math.cos(a) * (coreR + 1);
    const y1 = sy + Math.sin(a) * (coreR + 1);
    const x2 = sx + Math.cos(a) * (coreR + 1 + rayLen);
    const y2 = sy + Math.sin(a) * (coreR + 1 + rayLen);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}

function drawCompass(avgWx, avgWy, cameraRot) {
  const ctx = compassCanvas.getContext("2d");
  const w = compassCanvas.width;
  const h = compassCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(cx, cy) - 10;

  ctx.clearRect(0, 0, w, h);

  // Sun in the top-right corner, does not interfere with the rose.
  drawSunIndicator(ctx, w - 13, 13, lastSunIntensity);

  // Circle
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(136, 136, 170, 0.4)";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Cardinals rotated with the camera
  // World angles: E=0, S=PI/2, W=PI, N=-PI/2
  // cameraRot is added to follow the view's rotation
  const cardinals = [
    { label: "N", base: -Math.PI / 2 },
    { label: "E", base: 0 },
    { label: "S", base: Math.PI / 2 },
    { label: "W", base: Math.PI },
  ];
  ctx.font = "bold 10px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const c of cardinals) {
    const a = c.base + cameraRot;
    const lx = cx + Math.cos(a) * (r - 2);
    const ly = cy + Math.sin(a) * (r - 2);
    ctx.fillStyle = c.label === "N" ? "#cc4444" : "#666688";
    ctx.fillText(c.label, lx, ly);
  }

  // Thin center cross
  ctx.beginPath();
  ctx.moveTo(cx - 3, cy);
  ctx.lineTo(cx + 3, cy);
  ctx.moveTo(cx, cy - 3);
  ctx.lineTo(cx, cy + 3);
  ctx.strokeStyle = "rgba(136, 136, 170, 0.3)";
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // Average wind arrow (rotated with the camera)
  const mag = Math.sqrt(avgWx * avgWx + avgWy * avgWy);
  if (mag < 0.005) return;

  const worldDx = avgWx / mag;
  const worldDy = avgWy / mag;
  // Camera rotation applied to the wind vector
  const cosR = Math.cos(cameraRot);
  const sinR = Math.sin(cameraRot);
  const dx = worldDx * cosR - worldDy * sinR;
  const dy = worldDx * sinR + worldDy * cosR;

  const arrowLen = (r - 14) * Math.min(mag / 0.6, 1.0);

  const tipX = cx + dx * arrowLen;
  const tipY = cy + dy * arrowLen;

  // Shaft
  ctx.beginPath();
  ctx.moveTo(cx - dx * 4, cy - dy * 4);
  ctx.lineTo(tipX, tipY);
  ctx.strokeStyle = "#55ccff";
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.stroke();

  // Arrowhead
  const headLen = 7;
  const angle = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - headLen * Math.cos(angle - 0.45), tipY - headLen * Math.sin(angle - 0.45));
  ctx.lineTo(tipX - headLen * Math.cos(angle + 0.45), tipY - headLen * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fillStyle = "#55ccff";
  ctx.fill();
}

function updateCompass(state) {
  if (!state || !state.cells || state.cells.length === 0) return;

  // Average wind
  let sumWx = 0, sumWy = 0, sumTemp = 0;
  for (const c of state.cells) {
    sumWx += c.wind_x || 0;
    sumWy += c.wind_y || 0;
    sumTemp += c.temperature;
  }
  const n = state.cells.length;
  const avgWx = sumWx / n;
  const avgWy = sumWy / n;
  const avgTemp = sumTemp / n;
  const avgMag = Math.sqrt(avgWx * avgWx + avgWy * avgWy);

  lastAvgWind = { x: avgWx, y: avgWy };

  // Season
  const season = getSeason(state.tick);
  seasonLabel.textContent = season.name;
  seasonLabel.style.color = season.color;

  // Average temperature
  avgTempLabel.textContent = `${avgTemp.toFixed(1)}\u00B0C`;
  // Color based on temperature
  if (avgTemp < 5) avgTempLabel.style.color = "#6699ff";
  else if (avgTemp < 15) avgTempLabel.style.color = "#88aa88";
  else if (avgTemp < 25) avgTempLabel.style.color = "#ccaa44";
  else avgTempLabel.style.color = "#cc6633";

  // Average wind
  const dir = windLabel(avgWx, avgWy);
  avgWindLabel.textContent = `Wind ${dir}`;
  avgWindLabel.style.color = "#55ccff";
}

// --- Three.js setup ---

// Embed on a third-party site: `?chrome=minimal` keeps a date bar +
// play/pause, `?chrome=none` renders only the scene and lets the host draw
// its own UI (#142). CSS alone decides what stays visible; here we adapt
// the scene, identically in both cases: transparent background to let the
// host page show through, and no sun disk.
const EMBED = ["minimal", "none"].includes(
  new URLSearchParams(location.search).get("chrome"),
);

// --- Shipped world (#147) ---
// A fresh world is flat and history-less: bare ground, trees barely
// sprouted, no lake settled in. A visitor arriving on an embed should see a
// world that has lived, not tick 0, and they won't wait the three minutes of
// computation that represents. So we ship an already-aged checkpoint.
//
// Gzipped (50 MB -> 1.9 MB) and decompressed here rather than by the server:
// the embed can be served by any static host, whose `Content-Encoding`
// negotiation we don't control.
//
// Hence the `.ckptz` extension rather than `.ckpt.gz`. Many static servers
// (sirv, and therefore `vite preview`) see a `.gz` and announce
// `Content-Encoding: gzip`: the browser decompresses it on its own, ends up
// with 52 MB where the header promised 2, and aborts the response, a
// `TypeError: Failed to fetch` on a request that answered 200. An extension
// no server recognizes removes the trap at the root rather than documenting
// it. Reported by the integrator on v0.10.0.
//
// `?world=neuf` forces the fresh world, `?world=<name>` loads `worlds/<name>.ckptz`.
// Outside embed nothing loads by default: the dev loop wants a reproducible
// world at tick 0, not a frozen 50 MB state.
const WORLD_PARAM = new URLSearchParams(location.search).get("world");
const BOOT_WORLD_URL = (() => {
  if (WORLD_PARAM === "neuf") return null;
  if (WORLD_PARAM) return `worlds/${WORLD_PARAM}.ckptz`;
  return EMBED ? "worlds/aged.ckptz" : null;
})();

// An embed's lighting used to follow the simulation's real time… no: it was
// frozen at noon on the summer solstice, whatever the actual time was. On a
// fresh world this wasn't visible; on the 42-year shipped world the bar
// announces 10 PM while the terrain sits in broad noon. Reported by the
// integrator, and it's indefensible.
//
// The two arguments that had justified the freeze don't share the same fate.
// "A visitor arriving at night sees a black rectangle": false since #147, the
// shipped world opens at noon on June 30, it takes six simulated hours for
// the sun to go down. "It flickers at fast speed": true, and already handled,
// `updateSceneLighting` freezes the light above 24 h/s, always has, for the
// same reason, in normal mode. The embed freeze was an unconditional
// duplicate stacked on top of a guard that already existed.
//
// Hence its removal: the embed follows real time, and inherits the project's
// only "readable pace" threshold, the same one that already decides the hour
// display in `date`. An embed opens at 1 h/s: 24 s cycle, a dozen-second
// night.

const scene = new THREE.Scene();
// Sky color, animated over the diurnal cycle by `renderLighting`. Kept in its
// own variable rather than read from `scene.background`: in bare mode the
// background is transparent (`background = null`), but the fog must keep
// following this color.
//
// Outside bare mode, the value is aligned with the sidebars' --bg-1 to unify
// the visual field: the eye perceives the scene and the panels as the same
// dark continuum, rather than a framed black arena.
const skyColor = new THREE.Color(0x0a0c14);
scene.background = EMBED ? null : skyColor;

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 30, 25);

// `preserveDrawingBuffer`: the WebGL buffer stays readable after the swap,
// which makes captures (Playwright / canvas.toDataURL) reliable without a
// black frame. But preserving the backbuffer breaks Apple GPUs' tile-based
// optimization on every frame, negligible at ~500 hexes, costly at 120k. So
// it's only enabled for the capture tool, which loads the page with
// `?capture` (see scripts/shot/). Same for pixelRatio: capped at 1.5
// interactively (2× on Mac = 4× the fragments of a flat-shading that already
// hides aliasing well), but respects the density requested in capture
// (--scale).
const CAPTURE_MODE = new URLSearchParams(location.search).has("capture");
const renderer = new THREE.WebGLRenderer({
  antialias: true,
  preserveDrawingBuffer: CAPTURE_MODE,
  // The alpha channel costs a composite with the page on every frame: enabled
  // only for the one case that needs it, the embed.
  alpha: EMBED,
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(CAPTURE_MODE ? window.devicePixelRatio : Math.min(window.devicePixelRatio, 1.5));
// Explicit id: distinguishes the scene's WebGL canvas from the sidebar's
// #compass canvas, so the capture tool can target the right one (scripts/shot/).
renderer.domElement.id = "scene-canvas";
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;

// --- Fog + zoom bounds, derived from the map's real-world extent -----------
// Two answers to the same need ("avoid seeing too far", #110):
// 1. Background-colored distance fog: the far distance blends into the
//    background instead of floating crisp in a black arena, atmospheric
//    depth at the current zoom, horizon that fades out at max zoom.
// 2. maxDistance: at maximum zoom-out the map fills the screen (~fit height
//    for a 50° fov), never a "diorama seen from space" view where cloud
//    movement (1-2 cells/h) becomes sub-pixel.
// All of it DERIVED from the real world width (terrainBBox, recomputed on
// every rebuild so it's robust to a reset with a different radius), no fixed
// constant that drifts out of sync when the scale changes (#131). In capture
// mode: no clamp (scripts/shot frames programmatically via __hexcam); `?nofog`
// cuts the fog for pixel-exact diagnostic captures.
const NO_FOG = new URLSearchParams(location.search).has("nofog");
const FOG_NEAR_RATIO = 0.9; // × view width, below this, no haze
const FOG_FAR_RATIO = 2.8; // × view width, full fade into the background
const MAX_DIST_RATIO = 1.25; // × view width, the window fills the screen
// Absolute view budget (#132): beyond this equivalent width (in real km),
// max zoom, fog and the far plane stop following the map's size. At r45
// (11.9 km) nothing changes; at r120+ the map is no longer viewed as a whole
// diorama, it gets EXPLORED, the zoom clamp guarantees the frustum only sees
// a bounded window, so per-tile culling bounds the per-frame cost to the
// visibility radius, not the map radius. This is the render-side counterpart
// of "avoid seeing too far": the engine ticks everything, the screen only
// draws a region of it (the wire still transmits everything, separate effort).
const VIEW_BUDGET_KM = 16;
let viewWorldWidth = 0; // current world width (0 = no terrain yet)

function updateViewDistances() {
  if (!terrainBBox) return;
  const w = Math.max(
    terrainBBox.max.x - terrainBBox.min.x,
    terrainBBox.max.z - terrainBBox.min.z,
  );
  if (!(w > 0) || Math.abs(w - viewWorldWidth) < 1e-6) return;
  viewWorldWidth = w;
  // VIEW width: the whole map under the budget, a bounded window beyond it.
  // In capture, never capped, scripts/shot frames the whole map.
  const wView = CAPTURE_MODE ? w : Math.min(w, VIEW_BUDGET_KM / KM_PER_WORLD);
  if (!CAPTURE_MODE) {
    controls.maxDistance = wView * MAX_DIST_RATIO;
    controls.minDistance = 2; // prevents clipping through terrain when zooming in
  }
  if (!NO_FOG) {
    scene.fog = new THREE.Fog(0x000000, wView * FOG_NEAR_RATIO, wView * FOG_FAR_RATIO);
    // SAME Color instance as the sky: renderLighting animates `skyColor`
    // in place (setHSL, day/night cycle), by sharing it, the fog follows the
    // sky color without a per-frame copy or a second source of truth.
    scene.fog.color = skyColor;
  }
  // The far plane must encompass the end of the fog, otherwise clipping cuts
  // before the full fade (hard silhouette against the background).
  camera.far = Math.max(1000, wView * (FOG_FAR_RATIO + 1));
  camera.updateProjectionMatrix();
}

// Lighting: ambient and dirLight are modulated by the season.
// `updateSceneLighting(hourTick)` is called on every WS snapshot and at boot
// so intensities/colors reflect the current hour (diurnal cycle).
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);
// Radius for placing the sun on the sky-dome. For a directional light only
// the DIRECTION lights (intensity doesn't depend on distance), so this radius
// is arbitrary. Cast shadows are computed in the core (per-cell
// `illumination` field, #102); the front only colors the albedo with it.
const SUN_RADIUS = 25;

// Sun disk visible "in the distance", placed on the REAL azimuth (same vector
// as the core's sun). Decorative + visual debug of the direction. Unlit
// (MeshBasicMaterial: the sun doesn't shade itself), positioned/colored every
// frame by updateSceneLighting, hidden below the horizon. Distance >> terrain
// extent (~150 u) to read as "far away".
const SUN_DISTANCE = 320;
const sunMesh = new THREE.Mesh(
  new THREE.CircleGeometry(16, 6), // 6 segments = hexagon
  new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }),
);
sunMesh.frustumCulled = false;
// The disk is only positioned on the first renderLighting(): without this it
// stays visible at its default position as long as no snapshot has arrived,
// and it fills the frame on load (#140, seen in bare mode).
// Hidden until the first render, in every mode, the problem had nothing
// specific to the embed, only its symptom showed up there.
sunMesh.visible = false;
scene.add(sunMesh);

// Issue #47, visual day/night cycle in the 3D render.
//
// JS replicas of temperature.rs's formulas (Cooper 1969 for declination,
// D&B 2013 for solar elevation). Result: ambient color + directional
// intensity + sky background naturally follow the hourly cycle, the
// seasonal cycle emerges from variations in declination & day length
// depending on latitude.

// Reads the current latitude from the front slider. 44.5 by default if
// the slider isn't (yet) in the DOM.
function getLatitudeDeg() {
  const slider = document.querySelector('input[data-key="temperature.latitude_deg"]');
  return slider ? parseFloat(slider.value) : 44.5;
}

// Instantaneous sin(solar_elevation) for `hourTick` (hourly ticks since
// the sim started) and latitude. Formula strictly equivalent to
// temperature.rs::solar_elevation_at_hour.
//
// WARNING: takes `hour_tick` (hours) NOT the snapshot's `tick`, which is
// in *days* (v0.2.x compat). Otherwise the "day/night" cycle lasts 24
// simulated days instead of 24 hours, a bug observed visually and fixed here.
function computeSolarSinElevation(hourTick, latitudeDeg) {
  const day = Math.floor(hourTick / 24) % 365;
  const hour = hourTick % 24;
  const latRad = (latitudeDeg * Math.PI) / 180;
  // Cooper 1969: declination
  const decRad = (23.45 * Math.PI / 180) * Math.sin((2 * Math.PI * (284 + day)) / 365);
  // Hour angle ω: solar noon = 0, morning < 0, afternoon > 0
  const omega = ((hour - 12) * Math.PI) / 12;
  const sinElev =
    Math.sin(latRad) * Math.sin(decRad)
    + Math.cos(latRad) * Math.cos(decRad) * Math.cos(omega);
  return Math.max(-1, Math.min(1, sinElev));
}

// Unit vector TOWARD the sun in the ENU frame (East, North, Up), for
// `hourTick` (hours) and latitude. SAME geometry as the core
// (temperature.rs::solar_beam_at_tick): it's the real solar direction, not
// the sky-dome's stylized shape. Conversion to world space: x=e (East), y=u
// (up), z=-n (South = -North). This stops the front from "lying" about where
// the sun is.
function computeSunVectorENU(hourTick, latitudeDeg) {
  const day = Math.floor(hourTick / 24) % 365;
  const hour = hourTick % 24;
  const latRad = (latitudeDeg * Math.PI) / 180;
  const decRad = ((23.45 * Math.PI) / 180) * Math.sin((2 * Math.PI * (284 + day)) / 365);
  const omega = ((hour - 12) * Math.PI) / 12;
  const sphi = Math.sin(latRad);
  const cphi = Math.cos(latRad);
  const sdec = Math.sin(decRad);
  const cdec = Math.cos(decRad);
  const sw = Math.sin(omega);
  const cw = Math.cos(omega);
  return {
    e: -cdec * sw,
    n: sdec * cphi - cdec * cw * sphi,
    u: sdec * sphi + cdec * cw * cphi,
  };
}

// --- Sub-tick light interpolation (smoothed render of the day/night cycle) ---
//
// The core only emits ONE snapshot per simulated hour: at 1 h/s the sun used
// to jump a notch every real second (visible stutter reported by the user).
// This is smoothed on the front ONLY, without touching the engine: on every
// snapshot we know the hour REACHED; the light is then animated from the
// previous snapshot's hour to this one, spread over the real duration
// measured between the two arrivals. The render is therefore "1 tick behind"
// (an already-received interval is interpolated, never extrapolated), at the
// cost of a one-tick lag, invisible to the eye, in exchange for a perfectly
// continuous day/night cycle at 60 fps.
let lightPrevHour = null; // start hour of the current interpolation segment
let lightTargetHour = null; // arrival hour (last snapshot received)
let lightCurrentHour = null; // last fractional hour actually rendered
let lightSegStart = 0; // performance.now() at the last snapshot (segment start)
let lightSegDur = 0; // real measured duration of the segment (ms); 0 → interp inactive
let lightPlaying = false; // were we playing at the previous snapshot?
// Beyond this jump in hours between two snapshots we do NOT smooth: it's set
// directly. Protects large jumps (resuming after a fast-mode freeze, reset,
// multi-hour step) from a "spinning sun" over one second. At low speed the
// jump is 1-2 h, always smoothed.
const LIGHT_INTERP_MAX_GAP = 6;
// True iff the LAST snapshot armed an interpolation segment (slow play).
// Read by the illumination capture to smooth SHADOWS on the same timeline
// (same snapshots, same interval) as the directional light.
let lightInterpArmed = false;

// Schedules the light render for `hourTick` (snapshot hour).
// In slow play: arms an interpolation segment. Otherwise (pause, step, first
// snapshot, resuming play, big jump, or >= 24 h/s): sets the light directly.
function updateSceneLighting(hourTick) {
  // In play at >= 24 h/s, the 24h cycle would pass in < 1 real second →
  // strobe effect. Solution: freeze the light at its last stable state
  // (the current hour before switching to fast mode). To shift the light,
  // the user switches to pause and uses +1H.
  if (isPlaying && currentHps >= 24) {
    // Interpolation is cut and `lightSegStart` is kept fresh (called on
    // every snapshot even while frozen) to measure a correct interval when
    // returning to slow mode. The current light state stays frozen.
    lightSegDur = 0;
    lightSegStart = performance.now();
    lightPlaying = true;
    lightInterpArmed = false;
    return;
  }
  const now = performance.now();
  const canInterp =
    isPlaying &&
    lightPlaying &&
    lightCurrentHour !== null &&
    hourTick !== lightTargetHour &&
    Math.abs(hourTick - lightCurrentHour) <= LIGHT_INTERP_MAX_GAP;
  if (canInterp) {
    // New segment: starts from the CURRENTLY rendered hour (perfect
    // continuity regardless of jitter) toward the new target, over the real
    // duration elapsed since the previous snapshot.
    lightPrevHour = lightCurrentHour;
    lightTargetHour = hourTick;
    lightSegDur = Math.max(1, now - lightSegStart);
  } else {
    // No interpolation → immediate render and interp disarmed.
    lightPrevHour = hourTick;
    lightTargetHour = hourTick;
    lightCurrentHour = hourTick;
    lightSegDur = 0;
    renderLighting(hourTick);
  }
  lightSegStart = now;
  lightPlaying = isPlaying;
  lightInterpArmed = canInterp;
}

// Advances the light interpolation by one frame (called from animate()).
// Without an armed segment (lightSegDur == 0) the light is already set by
// updateSceneLighting, nothing to do.
function tickLightInterpolation(now) {
  if (lightSegDur <= 0) return;
  const t = Math.min(1, (now - lightSegStart) / lightSegDur);
  const hour = lightPrevHour + (lightTargetHour - lightPrevHour) * t;
  lightCurrentHour = hour;
  renderLighting(hour);
}

// --- Sub-tick SHADOW interpolation (per-cell `illumination` field) ---
//
// The sun direction (dirLight) already glides smoothly (interp above), but
// the CAST SHADOW and the sunny-slope/shaded-slope look come from the
// `illumination` field that the core computes PER CELL (toroidal raymarch +
// clouds, #102) and that the front writes into the terrain texture's alpha.
// This value only used to change at rebuild → shadows jumped a notch every
// hour just like the light.
//
// The front does NOT recompute the shadow (anti-pattern #2: the core remains
// the single source of truth): it only interpolates, cell by cell, between
// the LAST TWO values provided by the core, on the same timeline as the
// light (`lightSegStart`/`lightSegDur`). This is a temporal fade of core
// data, not a shadow computation, exactly the spirit of the light interp.
//
// Cost: when interp is active (slow play only), the terrain texture's alpha
// is rewritten every frame (one byte ×2 texels/cell + 1 upload). Negligible
// at current radii; bounded to slow play (frozen ≥ 24 h/s).
let illumPrev = null; // Float32Array, illumination at the start of the current segment
let illumTarget = null; // Float32Array, illumination from the last snapshot
let illumCurrent = null; // Float32Array, illumination currently displayed
let illumActive = false; // shadow interp in progress?

// Captures a snapshot's illumination. `interpolated` = did this snapshot arm
// a segment (provided by updateSceneLighting via `lightInterpArmed`). Otherwise
// it's set directly (the rebuild already writes the target alpha, nothing to
// smooth).
function captureIllumination(cells, interpolated) {
  const n = cells.length;
  // (Re)allocates if the grid changed size (reset with a different radius).
  if (!illumTarget || illumTarget.length !== n) {
    illumPrev = new Float32Array(n);
    illumTarget = new Float32Array(n);
    illumCurrent = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = cells[i].illumination ?? 1;
      illumPrev[i] = v;
      illumTarget[i] = v;
      illumCurrent[i] = v;
    }
    illumActive = false;
    return;
  }
  if (interpolated) {
    // New segment: starts from the CURRENTLY displayed illumination toward
    // the new target (perfect continuity, robust to jitter).
    illumPrev.set(illumCurrent);
    for (let i = 0; i < n; i++) illumTarget[i] = cells[i].illumination ?? 1;
    illumActive = true;
  } else {
    // No interp: set directly (the rebuild wrote / will write the target alpha).
    for (let i = 0; i < n; i++) {
      const v = cells[i].illumination ?? 1;
      illumTarget[i] = v;
      illumCurrent[i] = v;
    }
    illumActive = false;
  }
}

// Advances the shadow interpolation by one frame and rewrites the terrain
// texture's alpha. Shares the light's timeline (`lightSegStart/Dur`).
function tickIlluminationInterpolation(now) {
  if (!illumActive || lightSegDur <= 0 || !terrainTexData) return;
  const t = Math.min(1, (now - lightSegStart) / lightSegDur);
  const data = terrainTexData;
  const n = illumCurrent.length;
  for (let i = 0; i < n; i++) {
    const v = illumPrev[i] + (illumTarget[i] - illumPrev[i]) * t;
    illumCurrent[i] = v;
    const a = (v * 255) | 0;
    const o = i * 8; // 2 texels RGBA/cell; alpha = bytes 3 (top) and 7 (side)
    data[o + 3] = a;
    data[o + 7] = a;
  }
  if (terrainColorTex) terrainColorTex.needsUpdate = true;
}

// --- Sub-tick CLOUD interpolation (per-cell `cloud_water` field) ---
//
// Same principle as the shadows above: the core only emits one snapshot per
// simulated hour, and the cloud layer jumped a notch every hour, entire
// prisms appearing/disappearing at once, cell by cell. Yet at the scale of a
// tick the physical field moves little (hourly corr ~0.97, #110): this
// binary popping at a 1 h cadence is a big part of the perceived "nothing is
// moving". The per-cell RENDER DENSITY is faded between the last two
// snapshots, on the light timeline (lightSegStart/Dur): growth, dissipation
// and transfer to the neighboring cell become continuous slides. This is a
// temporal fade of core data, not a recomputation (anti-pattern #2
// respected, same spirit as the illumination interp).
//
// Cost: in slow play only, rewrites the active instances every frame
// (16+3 floats per visible instance). Same regime as the shadow alpha.
let cloudDensPrev = null; // Float32Array, densities at the start of the segment
let cloudDensTarget = null; // Float32Array, densities from the last snapshot
let cloudDensCurrent = null; // Float32Array, densities currently displayed
let cloudInterpActive = false;
let cloudCells = null; // cells from the last snapshot (positions/elevations)

// Captures a snapshot's densities. `interpolated` = same signal as the
// light/shadows (segment armed by updateSceneLighting). Otherwise set directly.
function captureClouds(cells, interpolated) {
  const n = cells.length;
  cloudCells = cells;
  // (Re)allocates if the grid changed size (reset with a different radius).
  if (!cloudDensTarget || cloudDensTarget.length !== n) {
    cloudDensPrev = new Float32Array(n);
    cloudDensTarget = new Float32Array(n);
    cloudDensCurrent = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const d = cloudDensityOf(cells[i].cloud_water);
      cloudDensPrev[i] = d;
      cloudDensTarget[i] = d;
      cloudDensCurrent[i] = d;
    }
    cloudInterpActive = false;
    return;
  }
  if (interpolated) {
    // New segment: starts from the CURRENTLY displayed density toward the
    // new target (perfect continuity, robust to jitter).
    cloudDensPrev.set(cloudDensCurrent);
    for (let i = 0; i < n; i++) cloudDensTarget[i] = cloudDensityOf(cells[i].cloud_water);
    cloudInterpActive = true;
  } else {
    for (let i = 0; i < n; i++) {
      const d = cloudDensityOf(cells[i].cloud_water);
      cloudDensTarget[i] = d;
      cloudDensCurrent[i] = d;
    }
    cloudInterpActive = false;
  }
}

// Advances the cloud fade by one frame and rewrites the instances. Shares
// the light's timeline (lightSegStart/Dur), like the shadows.
function tickCloudInterpolation(now) {
  if (!cloudInterpActive || lightSegDur <= 0 || !cloudCells) return;
  const t = Math.min(1, (now - lightSegStart) / lightSegDur);
  const n = cloudDensCurrent.length;
  for (let i = 0; i < n; i++) {
    cloudDensCurrent[i] = cloudDensPrev[i] + (cloudDensTarget[i] - cloudDensPrev[i]) * t;
  }
  fillCloudInstances(cloudCells, cloudDensCurrent);
}

// Modulates the scene light according to the instantaneous solar cycle.
// `dayness` (0 night, 1 zenith) drives the intensity, `horizon` (1 near
// the horizon, 0 otherwise) adds the red-orange dawn/dusk tint.
//
// DirectionalLight position: follows the real solar azimuth/elevation so
// shadows point the right way throughout the day. At night, intensity = 0 →
// no visible DirectionalLight.
//
// `hourTick` can be FRACTIONAL (sub-tick interpolation): the solar formulas
// natively accept a float (hour % 24, floor(hour/24)).
function renderLighting(hourTick) {
  const latDeg = getLatitudeDeg();
  const sinElev = computeSolarSinElevation(hourTick, latDeg);
  const dayness = Math.max(0, sinElev);
  // Gamma for visual perception: the eye compresses high brightness
  // (gamma ≈ 0.5). Without this, at 7 PM in August (sin_elev ≈ 0.04) the
  // brightness is near zero while the sun doesn't set until 9 PM. With
  // gamma 0.5: visibility = 0.2 at 7 PM in August, map still readable.
  // Applied only to ambient + background (reading the scene), not to
  // dirLight (the grazing sun MUST be weak, physically it crosses more
  // atmosphere → elongated shadows).
  const visibility = Math.sqrt(dayness);
  // "red" horizon: narrow peak around sunrise/sunset (sin_elev ≈ 0).
  // 1 - 4×|sinElev| cuts off at sinElev = ±0.25 (~14° above/below), beyond
  // which it's broad daylight (horizon = 0).
  const horizon = Math.max(0, 1 - Math.abs(sinElev) * 4);

  // Ambient: deep night 0.08 (map readability preserved), noon ~0.50.
  // Driven by `visibility` (gamma 0.5). Kept low so directional contrast
  // stays readable (a north-facing Lambert≈0 face must stay dark).
  ambientLight.intensity = AMBIENT_GAIN * (0.08 + 0.42 * visibility);

  // Directional: off at night, noon ~1.75. Linear in dayness.
  dirLight.intensity = dayness * 1.75;

  // Sun color: blue-white at zenith (warm at altitude), warm orange at
  // dawn/dusk (rays crossing more atmosphere). Decomposed into R/G/B for
  // a smooth transition.
  dirLight.color.setRGB(
    1.0,
    0.85 + 0.15 * dayness - 0.30 * horizon,
    0.70 + 0.30 * dayness - 0.50 * horizon,
  );

  // Sun position: follows the hourly azimuth (East at sunrise, West at
  // sunset) and the real elevation. The position drives the shadow
  // direction in Three.js. At night, positioned below the horizon
  // (intensity=0 so invisible but coherent).
  // Sun direction = REAL solar vector (ENU → world: x=East, y=up,
  // z=South). No more stylized ×0.8/×0.5 direction: the Lambert term now
  // points where the sun really is, consistent with the core's solar
  // geometry (#102) that computes per-cell shadow/illumination.
  const sun = computeSunVectorENU(hourTick, latDeg);
  dirLight.position.set(
    sun.e * SUN_RADIUS, // East
    sun.u * SUN_RADIUS, // height (Up)
    -sun.n * SUN_RADIUS, // South = −North
  );

  // Visible sun disk: same direction, projected far away. Colored like
  // dirLight (blue-white at zenith, warm orange near the horizon), hidden
  // below the horizon (otherwise it would shine below the map at night).
  //
  // Also visible in embed, since the lighting freeze was lifted: there was
  // no point showing a sun that never moved, there is one in showing a
  // sunrise and a sunset. The scene background is transparent in embed, so
  // the disk floats over the host page, an accepted tradeoff.
  sunMesh.visible = sun.u > 0.0;
  if (sunMesh.visible) {
    sunMesh.position.set(
      sun.e * SUN_DISTANCE,
      sun.u * SUN_DISTANCE,
      -sun.n * SUN_DISTANCE,
    );
    sunMesh.material.color.copy(dirLight.color);
    sunMesh.lookAt(camera.position);
  }

  // Sky background: dark blue at night (HSL hue 0.6, lum 0.05), light blue
  // at noon (HSL hue 0.6, lum 0.7), purple at dawn/dusk (hue 0.55, sat 0.8).
  // Lum driven by `visibility` (gamma 0.5) so the sky keeps some intensity
  // until sunset.
  skyColor.setHSL(
    0.60 - 0.05 * horizon,
    0.50 + 0.30 * horizon,
    0.05 + 0.65 * visibility,
  );
}

// --- State ---
let terrainMesh = null;
let terrainBBox = null;
let waterMesh = null;
// Snow: instanced like the clouds (uniform hex sheet per snowy cell, Y
// scale = thickness). SNOW_MAT is solid white → no instance color, only
// matrices.
let snowInst = null;
let snowProtoGeom = null;
let forestMesh = null;
// Clouds: rendered as instances (one hex prism prototype, one instance per
// cloudy cell, scale = thickness/footprint, color = density). The per-frame
// loop only fills instance matrices (data PER CELL, not 38 vertices/cell) →
// small and fast, unlike F10.
let cloudsInst = null;
let cloudProtoGeom = null;
const _cloudMtx = new THREE.Matrix4();
const _cloudCol = new THREE.Color();
let evapParticles = null;
let windArrows = null;
let temperatureContours = null;
let precipitationOverlay = null;
let showWind = false;
let showClouds = true;
let showTemperature = false;
let showPrecipitation = false;
let cellIndex = {}; // "q,r" → index into the state
let lastState = null;
// Last binary snapshot received, lazily decoded at rebuild (F2). Only a raw
// ArrayBuffer is kept: O(1) store on WS arrival, decode at most once per
// rebuild rather than on every frame received.
let latestSnapshotBuf = null;

// --- Hex geometry helpers ---

// Axial (q, r) → world (x, z). Y will be the elevation.
function hexToWorld(q, r) {
  const x = HEX_SIZE * (SQRT3 * q + (SQRT3 / 2) * r);
  const z = HEX_SIZE * (3 / 2) * r;
  return { x, z };
}

// Generates the 6 vertices of a flat-top hexagon (on the XZ plane)
function hexVertices(cx, cz) {
  const verts = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i - 30);
    verts.push({
      x: cx + HEX_SIZE * Math.cos(angle),
      z: cz + HEX_SIZE * Math.sin(angle),
    });
  }
  return verts;
}

// --- Color functions by view mode ---

// Terrain colored by permeability: impermeable rock (slate gray) → porous soil (sandy beige).
// Elevation is already visible through the columns' 3D height.
// Snow is rendered as a separate 3D mesh (buildSnow).
// Terrain color: blend (permeability × climatic humidity).
// Atmospheric humidity drives vegetation, local groundwater adds a bonus.
// Dry + rock  → stone gray     | Wet + rock  → dark moss
// Dry + soil → earthy brown    | Wet + soil → meadow green
function terrainColor(cell) {
  const p = Math.min(1, Math.max(0, cell.permeability));
  // Greenery fed by precipitable humidity (upper layer): this is what
  // waters the cell. humidity_upper typically ~ [0.02, 0.45].
  const wetFromAir = (cell.humidity_upper || 0) / 0.25;
  const wetFromSoil = (cell.groundwater || 0) / 1.5;
  const h = Math.min(1, Math.max(0, wetFromAir + wetFromSoil));

  // Dry palette: rock (gray) → soil (warm brown)
  const dryR = 0.40 + 0.38 * p;
  const dryG = 0.38 + 0.24 * p;
  const dryB = 0.34 + 0.10 * p;

  // Wet palette: rock (moss) → soil (meadow green)
  const wetR = 0.22 + 0.20 * p;
  const wetG = 0.44 + 0.30 * p;
  const wetB = 0.18 + 0.10 * p;

  return new THREE.Color(
    dryR + (wetR - dryR) * h,
    dryG + (wetG - dryG) * h,
    dryB + (wetB - dryB) * h,
  );
}

// Gradient red (0) → yellow (0.5) → blue (1)
function permeabilityColor(perm) {
  const t = Math.min(1, Math.max(0, perm));
  if (t < 0.5) {
    const s = t / 0.5;
    return new THREE.Color(0.8 - 0.3 * s, 0.2 + 0.6 * s, 0.1);
  }
  const s = (t - 0.5) / 0.5;
  return new THREE.Color(0.5 - 0.4 * s, 0.8 - 0.4 * s, 0.1 + 0.7 * s);
}

// Gradient black (0) → blue (max)
function groundwaterColor(gw, maxCapacity) {
  const t = Math.min(1, gw / maxCapacity);
  return new THREE.Color(0.05 + 0.1 * t, 0.1 + 0.2 * t, 0.2 + 0.7 * t);
}

// Gradient dry (brown) → wet (cyan).
// Scale: humidity in mm PW since Phase 3. Initial floor ~10 mm, Tetens
// saturation ~25 mm at 20 C (Phase 6). 50 mm = very humid (oversaturated).
function humidityColor(hum) {
  const t = Math.min(1, hum / 50.0);
  return new THREE.Color(0.4 - 0.2 * t, 0.3 + 0.3 * t, 0.2 + 0.6 * t);
}

// Blue (<0°C) → green (~15°C) → red (>30°C)
function temperatureColor(temp) {
  // Normalized over [-10, 35] → [0, 1]
  const t = Math.min(1, Math.max(0, (temp + 10) / 45));
  if (t < 0.33) {
    const s = t / 0.33;
    return new THREE.Color(0.1, 0.2 + 0.5 * s, 0.8 - 0.3 * s);
  }
  if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return new THREE.Color(0.1 + 0.7 * s, 0.7 - 0.2 * s, 0.3 - 0.2 * s);
  }
  const s = (t - 0.66) / 0.34;
  return new THREE.Color(0.8 + 0.2 * s, 0.5 - 0.3 * s, 0.1);
}

// PRECIPITATION overlay: blue→violet tint rendered as a separate mesh
// floating above all layers (terrain + snow + water), so the accumulated
// snow sheet doesn't mask the tint. Physically, rain_amount and snow_amount
// share the same scale, the engine picks one or the other depending on T,
// we aggregate on the total intensity.
//
// The intensity is encoded only by the HUE (not the alpha) so the mesh can
// use a single material with constant alpha. Square root on t to make
// drizzle distinguishable from downpour.
// mm/tick scale: physical ceiling max_precip_per_tick=4 mm (Chow 1988).
// At 2 mm/day it's already a proper downpour → t=0.71, marked violet.
// At 4 mm/day (cap reached) → deep violet, visual signal of extreme downpour.
const PRECIP_MAX = 4.0;
const PRECIP_MIN = 1e-3;

function precipitationTintColor(cell) {
  const total = (cell.rain_amount ?? 0) + (cell.snow_amount ?? 0);
  if (total < PRECIP_MIN) return null;
  const t = Math.sqrt(Math.min(1.0, total / PRECIP_MAX));
  // Saturated blue (0.35, 0.60, 1.00) → deep violet (0.24, 0.06, 0.63).
  // Starting from a solid blue (not a pale cyan) to stay visible even
  // when the overlay is rendered above the white snow.
  return new THREE.Color(
    0.35 * (1 - t) + 0.24 * t,
    0.60 * (1 - t) + 0.06 * t,
    1.00 * (1 - t) + 0.63 * t,
  );
}

// Color by DOMINANT SPECIES (vegetation layer, epic #78 #84). Blind
// consumption of `cell.dominant_species` + `cell.is_open_water` computed by
// the core (no reclassification here: the front end never recomputes what
// the core already computed). The total
// cover `cell.vegetation` only modulates vividness (dense cover more vivid).
// Keys = `SpeciesId` serialized snake_case by the core (species.rs), they
// must match exactly, otherwise `dominant_species` falls back to BARE_COLOR.
// Past bug: keys `oak`/`grass` ≠ `oak_pubescent`/`alpine_grass` → oak and
// grassland rendered as beige "bare soil", landscape diversity invisible.
const SPECIES_COLORS = {
  oak_pubescent: [0.45, 0.55, 0.20], // downy oak, warm green
  pine:          [0.22, 0.46, 0.42], // pine / juniper, blue-green
  beech:         [0.40, 0.62, 0.28], // beech, light green
  fir:           [0.12, 0.34, 0.22], // fir / spruce, dark green
  alpine_grass:  [0.66, 0.72, 0.30], // alpine grassland, yellow-green
};
const WATER_COLOR = [0.12, 0.30, 0.58]; // lake blue
const BARE_COLOR = [0.55, 0.52, 0.48];  // rock / bare soil

// Snow on column walls (buildTerrain). A bit darker than the snow roof
// (0xf0f0f8) so the walls read as walls under Lambert lighting, in the same
// spirit as the terrain's side = top·0.5.
const SNOW_SIDE_COLOR = new THREE.Color(0.82, 0.82, 0.88);
// snow_level at which the walls are fully white. Below that, whitening is
// proportional: a thin film whitens the roof but barely the sides.
const SNOW_WALL_FULL = 0.5;

function speciesColor(cell) {
  // Open water carries no vegetation: fixed tint.
  if (cell.is_open_water) {
    return new THREE.Color(WATER_COLOR[0], WATER_COLOR[1], WATER_COLOR[2]);
  }
  const sp = cell.dominant_species;
  if (!sp) {
    // Bare soil / rock (no species holds on).
    return new THREE.Color(BARE_COLOR[0], BARE_COLOR[1], BARE_COLOR[2]);
  }
  const base = SPECIES_COLORS[sp] || BARE_COLOR;
  const v = Math.min(1, Math.max(0, cell.vegetation || 0));
  const shade = 0.6 + 0.4 * v;
  return new THREE.Color(base[0] * shade, base[1] * shade, base[2] * shade);
}

// Canopy age: young (bright green) → old (dark green/rust brown).
// Saturates around ~80 years. Bare/water cell → gray/blue.
function ageColor(cell) {
  if (cell.is_open_water) return new THREE.Color(WATER_COLOR[0], WATER_COLOR[1], WATER_COLOR[2]);
  const v = Math.min(1, Math.max(0, cell.vegetation || 0));
  if (v < 0.02) return new THREE.Color(BARE_COLOR[0], BARE_COLOR[1], BARE_COLOR[2]);
  const t = Math.min(1, (cell.stand_age || 0) / 80);
  // young [0.5,0.8,0.35] → old [0.35,0.22,0.12]
  const r = 0.5 + t * (0.35 - 0.5);
  const g = 0.8 + t * (0.22 - 0.8);
  const b = 0.35 + t * (0.12 - 0.35);
  return new THREE.Color(r, g, b);
}

// Fire map: intensity 0 (dark) → 1 (bright red-orange).
function fireMapColor(cell) {
  const f = Math.min(1, Math.max(0, cell.fire_intensity || 0));
  if (f < 1e-3) return new THREE.Color(0.08, 0.09, 0.11);
  return new THREE.Color(1.0, 0.35 + 0.45 * (1 - f), 0.05);
}

// Fire tint that overlays any background: a burning cell shifts to
// orange/red proportionally to intensity (visible in any view).
function applyFire(color, cell) {
  const f = Math.min(1, Math.max(0, cell.fire_intensity || 0));
  if (f < 1e-3) return color;
  return color.lerp(new THREE.Color(1.0, 0.3, 0.04), 0.55 + 0.4 * f);
}

function cellColor(cell, mode) {
  let c;
  switch (mode) {
    case "permeability": c = permeabilityColor(cell.permeability); break;
    case "groundwater": c = groundwaterColor(cell.groundwater, (cell.permeability || 1) * 100.0); break;
    case "humidity_upper": c = humidityColor(cell.humidity_upper ?? 0); break;
    case "humidity_surface": c = humidityColor(cell.humidity_surface ?? 0); break;
    case "species": c = speciesColor(cell); break;
    case "age": c = ageColor(cell); break;
    case "fire": return fireMapColor(cell); // dedicated view: no re-tinting.
    case "pressure": return pressureColor(cell); // dedicated view: no re-tinting.
    default: c = terrainColor(cell);
  }
  return applyFire(c, cell);
}

// --- Synoptic pressure view (isobars, Phase 2 synoptic dynamics) ---
// Anomaly h−⟨h⟩ normalized by the frame's max (the amplitude ranges from a
// few mm at spin-up to tens of m in forced regime, a fixed scale would be
// unreadable). Weather convention: H (high) blue, L (low) red, neutral light
// gray. The core exports synoptic_h (m); the front consumes it blindly
// (anti-pattern #2).
let pressureScale = { mean: 0, span: 1 };

function updatePressureScale(cells) {
  let mean = 0;
  for (const c of cells) mean += c.synoptic_h ?? 0;
  mean /= Math.max(1, cells.length);
  // Scale at the p95 of |anomaly| (not the max): a few local extremes
  // would crush everything else into gray. Beyond the p95, color saturates.
  const abs = cells.map((c) => Math.abs((c.synoptic_h ?? 0) - mean)).sort((a, b) => a - b);
  const p95 = abs[Math.min(abs.length - 1, Math.floor(abs.length * 0.95))] ?? 0;
  pressureScale = { mean, span: Math.max(p95, 1e-9) };
}

function pressureColor(cell) {
  const a = ((cell.synoptic_h ?? 0) - pressureScale.mean) / pressureScale.span;
  const t = Math.max(-1, Math.min(1, a));
  // Neutral → L (red) for t<0, neutral → H (blue) for t>0.
  const n = { r: 0.88, g: 0.88, b: 0.86 };
  const target = t < 0 ? { r: 0.82, g: 0.20, b: 0.16 } : { r: 0.15, g: 0.35, b: 0.78 };
  const s = Math.abs(t);
  return new THREE.Color(
    n.r + (target.r - n.r) * s,
    n.g + (target.g - n.g) * s,
    n.b + (target.b - n.b) * s,
  );
}

// --- Shared materials (F5) --------------------------------------------------
// A `new *Material` per layer and per rebuild used to trigger shader
// pipeline recompilation (10% of the CPU profile at r60, `getProgramInfoLog`)
// and leaked VRAM on dispose. The configs are static: they're hoisted into
// constants reused from one rebuild to the next. `disposeObject3D` therefore
// only frees the geometries (unique per rebuild), never these materials.
//
// All Lambert layers use `flatShading`: the shader reconstructs the normal
// from screen-space derivatives (dFdx/dFdy) and does NOT use the `normal`
// attribute, hence removing the `computeVertexNormals()` calls in the builds
// (F4). Snow also switches to flatShading (visually identical on prisms).
// Terrain: colors in a DataTexture (F3-step-2). Rather than a per-vertex
// color attribute rewritten + re-uploaded every frame (~14-45 MB), 2 colors
// per cell (roof + side) are stored in a small texture (~1 MB) and the vertex
// shader looks up each vertex's color via its texel index `aTexel`. The
// lookup is injected into MeshLambertMaterial via onBeforeCompile to keep its
// Lambert lighting + flatShading (normal from screen-space derivatives, F4).
// The uniforms point to stable objects updated on world change (new texture)
// without recompiling the shader.
const terrainColorUniform = { value: null }; // sampler2D of cell colors
const terrainTexSizeUniform = { value: new THREE.Vector2(1, 1) };
const TERRAIN_MAT = new THREE.MeshLambertMaterial({ flatShading: true });
TERRAIN_MAT.onBeforeCompile = (shader) => {
  shader.uniforms.uColorTex = terrainColorUniform;
  shader.uniforms.uTexSize = terrainTexSizeUniform;
  shader.vertexShader = shader.vertexShader
    .replace(
      "#include <common>",
      `#include <common>
      attribute float aTexel;
      uniform sampler2D uColorTex;
      uniform vec2 uTexSize;
      varying vec3 vTerrColor;
      varying float vIllum;`,
    )
    .replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>
      {
        float tx = mod(aTexel, uTexSize.x);
        float ty = floor(aTexel / uTexSize.x);
        vec4 terr = texture2D(uColorTex, (vec2(tx, ty) + 0.5) / uTexSize);
        vTerrColor = terr.rgb;   // pure cell color (albedo)
        vIllum = terr.a;         // illumination [0,1] (aspect × occlusion × cloud)
      }`,
    );
  // vIllum modulates ONLY the directional light (sun), not the ambient:
  // a cell in shadow receives `ambient + sun×0 = ambient` (the floor,
  // never darker). This is the correct physical model; baking illumination
  // into the albedo also darkened the ambient (shadows blacker than night).
  shader.fragmentShader = shader.fragmentShader
    .replace(
      "#include <common>",
      "#include <common>\n      varying vec3 vTerrColor;\n      varying float vIllum;",
    )
    .replace(
      "vec4 diffuseColor = vec4( diffuse, opacity );",
      "vec4 diffuseColor = vec4( diffuse * vTerrColor, opacity );",
    )
    .replace(
      "#include <lights_fragment_end>",
      "#include <lights_fragment_end>\n      reflectedLight.directDiffuse *= vIllum;",
    );
};
const WATER_MAT = new THREE.MeshLambertMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.88,
  flatShading: true,
  side: THREE.DoubleSide,
});
const CLOUDS_MAT = new THREE.MeshLambertMaterial({
  vertexColors: true,
  side: THREE.DoubleSide,
  flatShading: true,
});
const SNOW_MAT = new THREE.MeshLambertMaterial({
  color: 0xf0f0f8,
  side: THREE.DoubleSide,
  flatShading: true,
});
const TREE_MAT = new THREE.MeshLambertMaterial({ flatShading: true });
const WIND_MAT = new THREE.MeshBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.75,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const TEMP_MAT = new THREE.MeshBasicMaterial({
  vertexColors: true,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const PRECIP_MAT = new THREE.MeshBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.7,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const EVAP_MAT = new THREE.PointsMaterial({
  color: 0xccddff,
  size: 0.08,
  transparent: true,
  opacity: 0.4,
});

// --- Persistent terrain (F3) + neighbor-aware walls (F6) + tiles (#132) -----
// Terrain topology (positions + indices) depends only on (q, r, elevation),
// immutable as long as the world doesn't change (a reset regenerates the
// relief). So we build the geometry ONCE per world, then each snapshot only
// rewrites the color DataTexture: no more rebuilding from scratch or
// re-uploading positions/indices.
//
// F6: we only emit a wall on edges that are actually exposed: neighbor lower
// (wall = the step, from its roof to ours) or absent (map edge: skirt down
// to the base). Each shared interior edge is therefore drawn only once (by
// the higher cell) instead of twice: ~half the walls, without changing
// the silhouette.
//
// #132: the terrain is no longer ONE merged mesh but a Group of tiles (axial
// blocks of TERRAIN_CHUNK × TERRAIN_CHUNK cells), one bounding sphere per
// tile:
// - per-tile frustum culling (`frustumCulled` was inoperative on the single
//   object): up close, only the on-screen tiles go through vertex
//   processing, so the per-frame cost follows the VISIBILITY radius, not
//   the map radius. Founding measurement (ablation r200, JOURNAL
//   2026-07-22): 4.6M vertices processed every frame regardless of what's
//   in view, free instanced layers, ÷9 pixels ≈ +12%: the entire per-frame
//   cost was this mesh.
// - distance culling as a bonus (updateTileVisibility): a tile entirely
//   beyond `fog.far` is invisible by construction (100% faded into the
//   background), so we don't draw it. Visually free as long as the fog
//   is there.
// All tiles share TERRAIN_MAT and the color DataTexture (`aTexel` stays a
// GLOBAL index 2·ci): updateTerrainColors, illumination interpolation, and
// analytic picking (F8) don't change by a single byte.
const TERRAIN_CHUNK = 32; // cells per tile side (axial) — ~1k cells/tile
let terrainTiles = []; // tile meshes (children of the terrainMesh Group)
const TERRAIN_TEX_W = 2048; // width of the color DataTexture (F3-step-2)
let terrainSig = null; // topological signature of the rendered world
// Cell color DataTexture (2 texels/cell: roof then side).
let terrainColorTex = null; // THREE.DataTexture
let terrainTexData = null; // underlying Uint8Array RGBA (rewritten per snapshot)
let terrainTexH = 1; // texture height (width = TERRAIN_TEX_W)

// Cheap signature of the relief: if it changes, the world has changed (reset)
// and the geometry must be rebuilt. O(n) in pure arithmetic (~1 ms at
// r200), negligible next to the cost of the full rebuild it avoids.
function terrainSignature(cells) {
  let h = cells.length >>> 0;
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    h = Math.imul(h ^ (c.q & 0xffff), 16777619);
    h = Math.imul(h ^ (c.r & 0xffff), 16777619);
    h = Math.imul(h ^ (Math.round(c.elevation * 16) & 0x7fffff), 16777619);
  }
  return h >>> 0;
}

// Builds the terrain Group of tiles (frozen topology, neighbor-aware walls,
// one BufferGeometry per tile). Rebuilds `cellIndex` (stable per world,
// consumed by the tooltip). Called only on first render and on each world
// change. `push` on JS arrays is acceptable here: once per world, not per
// frame.
function buildTerrainGeometry(cells) {
  const n = cells.length;
  // Pass 1: cellIndex + elevations (needed to test neighbors).
  cellIndex = {};
  const elev = new Float32Array(n);
  let minElev = Infinity;
  for (let ci = 0; ci < n; ci++) {
    const c = cells[ci];
    cellIndex[`${c.q},${c.r}`] = ci;
    elev[ci] = c.elevation;
    if (c.elevation < minElev) minElev = c.elevation;
  }
  const baseY = (minElev - 20) * ELEVATION_SCALE;

  // Pass 2: geometry, accumulated PER TILE (axial block ⌊q/C⌋,⌊r/C⌋). Walls
  // only on exposed edges (F6). Each vertex carries `aTexel` = its texel
  // index in the DataTexture: the 7 roof vertices point to texel 2*ci
  // (roof color), the wall vertices to 2*ci+1 (side).
  const tiles = new Map(); // "tq,tr" -> {positions, indices, texels, vc}

  for (let ci = 0; ci < n; ci++) {
    const cell = cells[ci];
    const key = `${Math.floor(cell.q / TERRAIN_CHUNK)},${Math.floor(cell.r / TERRAIN_CHUNK)}`;
    let tile = tiles.get(key);
    if (!tile) {
      tile = { positions: [], indices: [], texels: [], vc: 0 };
      tiles.set(key, tile);
    }
    const { positions, indices, texels } = tile;
    const { x, z } = hexToWorld(cell.q, cell.r);
    const topY = cell.elevation * ELEVATION_SCALE;
    const verts = hexVertices(x, z);
    const topTexel = 2 * ci;
    const sideTexel = 2 * ci + 1;

    // Roof: center + ring (7 vertices, up front).
    const topCenter = tile.vc;
    positions.push(x, topY, z); texels.push(topTexel); tile.vc++;
    for (const v of verts) { positions.push(v.x, topY, v.z); texels.push(topTexel); tile.vc++; }
    for (let i = 0; i < 6; i++) {
      const next = (i + 1) % 6;
      indices.push(topCenter, topCenter + 1 + next, topCenter + 1 + i);
    }

    // Walls: edge i (between verts[i] and verts[i+1]) borders neighbor
    // HEX_NEIGHBORS[i]. We draw the wall only if the neighbor is lower
    // (wall from the current roof down to the neighbor's roof) or absent
    // (edge → base). Tile note: the wall belongs to the HIGH cell; at a
    // tile boundary it is emitted in that cell's tile, the silhouette is
    // identical to the old merged mesh.
    for (let i = 0; i < 6; i++) {
      const nq = cell.q + HEX_NEIGHBORS[i][0];
      const nr = cell.r + HEX_NEIGHBORS[i][1];
      const nIdx = cellIndex[`${nq},${nr}`];
      let bottomY;
      if (nIdx === undefined) {
        bottomY = baseY; // map edge: full skirt down to the base
      } else {
        const nTopY = elev[nIdx] * ELEVATION_SCALE;
        if (nTopY >= topY) continue; // neighbor as high: hidden face, skip
        bottomY = nTopY; // wall = the step, from the neighbor's roof to ours
      }
      const next = (i + 1) % 6;
      const v0 = verts[i];
      const v1 = verts[next];
      const sb = tile.vc;
      positions.push(v0.x, topY, v0.z); texels.push(sideTexel); tile.vc++;
      positions.push(v1.x, topY, v1.z); texels.push(sideTexel); tile.vc++;
      positions.push(v1.x, bottomY, v1.z); texels.push(sideTexel); tile.vc++;
      positions.push(v0.x, bottomY, v0.z); texels.push(sideTexel); tile.vc++;
      indices.push(sb, sb + 1, sb + 2, sb, sb + 2, sb + 3);
    }
  }

  // Color DataTexture: 2 texels per cell, GLOBAL (shared by all tiles).
  // Size = TERRAIN_TEX_W × ceil(2n / W). Byte RGBA in NoColorSpace: the
  // shader reads the bytes as-is (linear).
  terrainTexH = Math.max(1, Math.ceil((2 * n) / TERRAIN_TEX_W));
  terrainTexData = new Uint8Array(TERRAIN_TEX_W * terrainTexH * 4);
  terrainColorTex = new THREE.DataTexture(
    terrainTexData,
    TERRAIN_TEX_W,
    terrainTexH,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  terrainColorTex.magFilter = THREE.NearestFilter;
  terrainColorTex.minFilter = THREE.NearestFilter;
  terrainColorTex.generateMipmaps = false;
  terrainColorTex.colorSpace = THREE.NoColorSpace;
  terrainColorUniform.value = terrainColorTex;
  terrainTexSizeUniform.value.set(TERRAIN_TEX_W, terrainTexH);

  // One mesh per tile, explicit bounding sphere (three's frustum culling
  // relies on it), all under one Group: the outer API (Box3, dispose,
  // scene.add) always sees a SINGLE `terrainMesh` object.
  const group = new THREE.Group();
  terrainTiles = [];
  for (const tile of tiles.values()) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(tile.positions, 3));
    geometry.setAttribute("aTexel", new THREE.Float32BufferAttribute(tile.texels, 1));
    geometry.setIndex(tile.indices);
    // No computeVertexNormals: flatShading reconstructs the normal from
    // screen-space derivatives, the `normal` attribute would be dead (F4).
    geometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(geometry, TERRAIN_MAT);
    group.add(mesh);
    terrainTiles.push(mesh);
  }
  return group;
}

// Per-tile distance culling (#132): a tile entirely beyond the fog's end is
// 100% faded into the background, invisible by construction, so we drop it
// from the draw. Called every frame (≤ ~200 distances, negligible); per-tile
// frustum culling is handled by three via the bounding spheres.
// In capture mode: never cull (captures frame the whole map, often beyond
// the view budget; diagnostic views).
function updateTileVisibility() {
  if (!terrainTiles.length || CAPTURE_MODE) return;
  const farWorld = scene.fog ? scene.fog.far : Infinity;
  for (const t of terrainTiles) {
    const s = t.geometry.boundingSphere;
    t.visible = camera.position.distanceTo(s.center) - s.radius <= farWorld;
  }
}

// --- Illumination (consumed from the core, #102) --------------------------
// No more shadow computed in the front: the core exports `cell.illumination`
// ∈ [0,1] per cell = fraction of sun received (aspect × relief occlusion ×
// cloud shadow, toroidal raymarch on the Rust side). `updateTerrainColors`
// multiplies the albedo by this value: the front now only colors, single
// source of truth (anti-pattern #2). The old `computeCellShadow` (JS raymarch
// on elevation alone, cosmetic, non-toroidal, no cloud) is removed.

// Writes cell colors (roof + side) into the DataTexture per view mode:
// 2 texels/cell instead of ~10 vertices, ~5× fewer writes and ~15-45× less
// upload (F3-step-2). Same color logic as the old buildTerrain. Bytes 0-255
// (linear, NoColorSpace on the texture side). Each color is modulated by the
// illumination factor `cell.illumination` supplied by the core.
function updateTerrainColors(cells, mode) {
  if (mode === "pressure") updatePressureScale(cells);
  const data = terrainTexData;
  const snowR = SNOW_SIDE_COLOR.r;
  const snowG = SNOW_SIDE_COLOR.g;
  const snowB = SNOW_SIDE_COLOR.b;
  for (let ci = 0; ci < cells.length; ci++) {
    const cell = cells[ci];
    const color = cellColor(cell, mode);
    // Sides = roof × 0.5, whitened toward snow per snow_level (same as the
    // old buildTerrain: otherwise a snowy hex shows a dark wall).
    let sr = color.r * 0.5;
    let sg = color.g * 0.5;
    let sb = color.b * 0.5;
    const snow = cell.snow_level ?? 0;
    if (snow >= 0.01) {
      const t = Math.min(1, snow / SNOW_WALL_FULL);
      sr += (snowR - sr) * t;
      sg += (snowG - sg) * t;
      sb += (snowB - sb) * t;
    }
    // PURE color in RGB (albedo) + core illumination in ALPHA. The shader
    // only applies illumination to the directional light (sun), not to the
    // ambient, so a cell in shadow = `ambient` (floor), never darker.
    // 1.0 if the field is absent (old core).
    const sf = cell.illumination ?? 1;
    const a = (sf * 255) | 0;
    let o = ci * 8; // 2 RGBA texels = 8 bytes per cell
    data[o] = (color.r * 255) | 0;
    data[o + 1] = (color.g * 255) | 0;
    data[o + 2] = (color.b * 255) | 0;
    data[o + 3] = a;
    data[o + 4] = (sr * 255) | 0;
    data[o + 5] = (sg * 255) | 0;
    data[o + 6] = (sb * 255) | 0;
    data[o + 7] = a;
  }
  terrainColorTex.needsUpdate = true;
}

// --- Build water mesh (#103: per-edge flux ribbons + solid-hex lakes) -----
// We draw the FLUX, not the stock (anti-pattern #3 applied to rendering):
//   • river = CURVED band from one edge-midpoint to another, inside each
//     hex (quadratic curve whose control point is the hex center: the bend
//     is rounded, not folded square). Since the edge-midpoint is shared with
//     the neighbor, inter-hex continuity is geometric: an edge above the
//     threshold is drawn on both sides, each on its own. Each tributary
//     joins the dominant outlet; a source is born at the center; a sink
//     dies there by thinning out.
//   • lake / overflow = solid hexagonal prism for the surplus above
//     capacity. At this resolution (~130 m/hex), a one-hex lake is rendered
//     as a hex filled with water, full stop, and NO ribbon is drawn inside
//     a lake: the river stops at the shore (cascading down to the surface)
//     and resumes from the outlet's shore.
// No more puddle-disc: sub-capacity water is micro-retention, invisible
// at map scale.
//
// Ribbon source: `cell.edge_flux` exported by the core (never reconstructed
// here, anti-pattern #2), 6 bytes per cell in the core's DIRECTIONS order
// (E, NE, NW, W, SW, SE, clockwise from east), quantized on a square-root
// scale relative to the frame max: b/255 = √(flux/edge_flux_max).
// b/255 therefore directly serves as visual intensity (the old pow 0.5 ramp).
const EDGE_DIR_WORLD = [
  [1, 0],
  [0.5, -SQRT3 / 2],
  [-0.5, -SQRT3 / 2],
  [-1, 0],
  [-0.5, SQRT3 / 2],
  [0.5, SQRT3 / 2],
];
const EDGE_DIR_QR = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
];
// Opposite edge: the inflow through edge d of my cell = the outflow of the
// neighbor through its edge (d+3)%6. It's the same wire byte on both sides.
const EDGE_OPPOSITE = [3, 4, 5, 0, 1, 2];
// Sampling segments for a ribbon curve (7 points, 12 triangles).
const CURVE_SEGS = 6;

function buildWater(cells, edgeFluxMax) {
  const positions = [];
  const colors = [];
  const indices = [];
  let vc = 0;

  const OVERFLOW_Y = 0.03;
  const TRAIL_Y = 0.04;
  const TOP_COLOR = [0.22, 0.52, 0.88];
  const SIDE_COLOR = [0.08, 0.22, 0.52];
  const TRAIL_COLOR = [0.15, 0.40, 0.75];

  const TRAIL_WIDTH = 0.40;
  // Minimum overflow threshold to render a "lake" prism. 1 mm of
  // surplus = drizzle that exceeded the micro-retention capacity; on a
  // hex that's a trace, not a lake visible at this scale. Protects
  // against the "entire column blue" bug when water_capacity is tiny
  // (0.05 mm post-Phase 3).
  const OVERFLOW_MIN_MM = 1.0;

  // Center → edge-midpoint distance = half the center-to-center (√3·HEX_SIZE).
  const EDGE_MID_DIST = (SQRT3 / 2) * HEX_SIZE;

  // Index (q,r) → cell to read the downstream neighbor's elevation
  // (the ribbon's slope). A missing neighbor = toroidal seam edge: the
  // ribbon stops at the edge midpoint (the water exits through the
  // map's edge, the wrapped neighbor redraws the continuation on its
  // side).
  const byCoord = new Map();
  for (const c of cells) byCoord.set(c.q + "," + c.r, c);

  // Frame's max discharge to normalize widths. Loop rather than
  // spread: the V8 stack crashes beyond ~130k arguments (F11).
  let maxDischarge = 1e-3;
  for (const c of cells) {
    const o = c.outflow_flux || 0;
    if (o > maxDischarge) maxDischarge = o;
  }
  // Width of a segment from the TOTAL discharge of its source cell
  // (square-root ramp: a trickle stays distinguishable, a big river
  // dominates).
  const dischargeWidth = (o) =>
    TRAIL_WIDTH * (0.03 + 0.97 * Math.sqrt(Math.min(1, (o || 0) / maxDischarge)));
  // Dominant outgoing edge of a cell (-1 if nothing flows). This is
  // the only use of `edge_flux` at render time: routing the network,
  // not painting it.
  function dominantEdge(c) {
    const ef = c.edge_flux;
    if (!ef) return -1;
    let best = -1;
    let bestB = 0;
    for (let d = 0; d < 6; d++) {
      const b = ef[d] || 0;
      if (b > bestB) {
        bestB = b;
        best = d;
      }
    }
    return best;
  }
  // Tiny vertical offset between curves of the same cell: they share
  // their endpoints (confluences), being coplanar they would z-fight.
  // ~15 visual cm, imperceptible.
  const Y_JITTER = 0.002;

  // Curved band at constant height: quadratic (ax,az) → control
  // (cx,cz) → (bx,bz), width interpolated w0 → w1. The whole path lives
  // inside one hex (flat roof), elevation change is taken as a cascade
  // at the edge (pushFall), never by tilting the band (a tilted ribbon
  // dives into the terrain prism, rendering as dotted).
  function pushCurve(ax, az, cx, cz, bx, bz, y, w0, w1) {
    const base = vc;
    const pts = [];
    for (let k = 0; k <= CURVE_SEGS; k++) {
      const t = k / CURVE_SEGS;
      const m0 = (1 - t) * (1 - t);
      const m1 = 2 * t * (1 - t);
      const m2 = t * t;
      pts.push([m0 * ax + m1 * cx + m2 * bx, m0 * az + m1 * cz + m2 * bz]);
    }
    for (let k = 0; k <= CURVE_SEGS; k++) {
      // Tangent via centered differences → normal (flat ribbon in XZ).
      const kp = pts[Math.min(CURVE_SEGS, k + 1)];
      const km = pts[Math.max(0, k - 1)];
      let tx = kp[0] - km[0];
      let tz = kp[1] - km[1];
      const tm = Math.hypot(tx, tz) || 1;
      tx /= tm;
      tz /= tm;
      const w = (w0 + ((w1 - w0) * k) / CURVE_SEGS) / 2;
      const px = -tz * w;
      const pz = tx * w;
      positions.push(pts[k][0] - px, y, pts[k][1] - pz);
      positions.push(pts[k][0] + px, y, pts[k][1] + pz);
      colors.push(...TRAIL_COLOR);
      colors.push(...TRAIL_COLOR);
      vc += 2;
    }
    for (let k = 0; k < CURVE_SEGS; k++) {
      const i0 = base + 2 * k;
      indices.push(i0, i0 + 1, i0 + 3);
      indices.push(i0, i0 + 3, i0 + 2);
    }
  }

  // Cascade: vertical quad at the edge, drop from y0 to y1 over width
  // w. Hex roofs are flat, a ribbon tilted in a straight line would
  // dive into the upstream prism (rendering as dotted). So the ribbon
  // stays flat above each hex and the elevation change is taken here,
  // as a stair step at the edge midpoint. WATER_MAT is DoubleSide:
  // winding order doesn't matter.
  function pushFall(cx, cz, y0, y1, ux, uz, w) {
    const px = (-uz * w) / 2;
    const pz = (ux * w) / 2;
    const base = vc;
    positions.push(cx - px, y0, cz - pz);
    positions.push(cx + px, y0, cz + pz);
    positions.push(cx + px, y1, cz + pz);
    positions.push(cx - px, y1, cz - pz);
    for (let j = 0; j < 4; j++) colors.push(...TRAIL_COLOR);
    vc += 4;
    indices.push(base, base + 1, base + 2);
    indices.push(base, base + 2, base + 3);
  }

  for (const cell of cells) {
    const { x, z } = hexToWorld(cell.q, cell.r);
    const wl = cell.water_level || 0;
    const cap = cell.water_capacity || 1.0;

    // OVERFLOW: solid hex prism for the surplus above capacity, the
    // "one-hex lake = hex filled with water" render. Thresholded to
    // avoid a cell with near-zero capacity (post-Phase 3, water_capacity
    // ≈ 0.05-0.5 mm) showing a "lake" as soon as one mm of rain crosses
    // it; below the threshold, nothing (trace invisible at this scale).
    if (wl > cap + OVERFLOW_MIN_MM) {
      const surplus = (wl - cap) * WATER_SURFACE_M;
      const bottomY = cell.elevation * ELEVATION_SCALE + OVERFLOW_Y;
      const topY = (cell.elevation + surplus) * ELEVATION_SCALE + OVERFLOW_Y;
      const verts = hexVertices(x, z);

      const centerTop = vc;
      positions.push(x, topY, z);
      colors.push(...TOP_COLOR);
      vc++;
      for (const v of verts) {
        positions.push(v.x, topY, v.z);
        colors.push(...TOP_COLOR);
        vc++;
      }
      for (let i = 0; i < 6; i++) {
        indices.push(centerTop, centerTop + 1 + ((i + 1) % 6), centerTop + 1 + i);
      }

      for (let i = 0; i < 6; i++) {
        const v0 = verts[i];
        const v1 = verts[(i + 1) % 6];
        const sideBase = vc;
        positions.push(v0.x, topY, v0.z);
        positions.push(v1.x, topY, v1.z);
        positions.push(v1.x, bottomY, v1.z);
        positions.push(v0.x, bottomY, v0.z);
        for (let j = 0; j < 4; j++) colors.push(...SIDE_COLOR);
        vc += 4;
        indices.push(sideBase, sideBase + 1, sideBase + 2);
        indices.push(sideBase, sideBase + 2, sideBase + 3);
      }
    }

    // RIVERS: network reduction at display time (cartographic D8). The
    // MFD spreads the flux over 2-3 edges, it's a FLOW FIELD, and
    // drawing every edge braids and crosses (unreadable, see #103
    // rollback). Like a topo map extracted from a DEM, we only draw the
    // NETWORK: a river-cell (discharge > threshold, same definition as
    // the core diag's `river_cells`) draws ONE curve toward its
    // dominant outgoing edge, width ∝ √(discharge/max), the total
    // discharge, not the edge's flux. The full field stays in the wire
    // protocol and the diags: the simplification is purely
    // cartographic.
    if (edgeFluxMax <= 0) continue;
    const isLake = wl > cap + OVERFLOW_MIN_MM;
    // Reference height of the ribbon/surface: the lake's roof for a
    // cell in overflow (a cascade lands there), the ground otherwise.
    const ySurf = isLake
      ? (cell.elevation + (wl - cap) * WATER_SURFACE_M) * ELEVATION_SCALE + OVERFLOW_Y
      : cell.elevation * ELEVATION_SCALE + TRAIL_Y;

    // Dominant outlet; the cell is a river if its discharge passes the
    // threshold. Inflows: river-neighbors whose dominant edge points
    // here, the criterion is the same on both sides of the edge, so a
    // drawn segment is always drawn on both sides (no dotted line
    // possible).
    const outD = dominantEdge(cell);
    const isRiver = outD >= 0 && (cell.outflow_flux || 0) > RIVER_THRESHOLD;
    const ins = [];
    let outNb = null;
    for (let d = 0; d < 6; d++) {
      const nb = byCoord.get(cell.q + EDGE_DIR_QR[d][0] + "," + (cell.r + EDGE_DIR_QR[d][1]));
      if (!nb) continue; // toroidal seam: the continuation is drawn on the other side
      if (d === outD) outNb = nb;
      if (
        dominantEdge(nb) === EDGE_OPPOSITE[d] &&
        (nb.outflow_flux || 0) > RIVER_THRESHOLD
      ) {
        ins.push({ d, w: dischargeWidth(nb.outflow_flux) });
      }
    }
    if (!isRiver && !ins.length) continue;

    const wOut = dischargeWidth(cell.outflow_flux || 0);

    // Cascade at the outgoing edge (drawn by the flow's owner): from
    // the level here to the level there, lake surface included, this is
    // how a river DIVES into a lake or RESURFACES at the outlet.
    if (isRiver && outNb) {
      const nwl = outNb.water_level || 0;
      const ncap = outNb.water_capacity || 1.0;
      const yN =
        nwl > ncap + OVERFLOW_MIN_MM
          ? (outNb.elevation + (nwl - ncap) * WATER_SURFACE_M) * ELEVATION_SCALE + OVERFLOW_Y
          : outNb.elevation * ELEVATION_SCALE + TRAIL_Y;
      if (Math.abs(yN - ySurf) > 1e-6) {
        const [ux, uz] = EDGE_DIR_WORLD[outD];
        pushFall(x + ux * EDGE_MID_DIST, z + uz * EDGE_MID_DIST, ySurf, yN, ux, uz, wOut);
      }
    }

    // In a lake, no ribbon: the prism IS the water.
    if (isLake) continue;

    let lift = 0;
    const midOf = (d) => [x + EDGE_DIR_WORLD[d][0] * EDGE_MID_DIST, z + EDGE_DIR_WORLD[d][1] * EDGE_MID_DIST];
    if (isRiver) {
      const [ox, oz] = midOf(outD);
      if (ins.length) {
        // Every tributary joins the outlet via a curve through the center.
        for (const i of ins) {
          const [ix, iz] = midOf(i.d);
          pushCurve(ix, iz, x, z, ox, oz, ySurf + (lift += Y_JITTER), i.w, wOut);
        }
      } else {
        // Source: the ribbon is born thin at the hex's center.
        pushCurve(x, z, (x + ox) / 2, (z + oz) / 2, ox, oz, ySurf, wOut * 0.35, wOut);
      }
    } else {
      // Terminus (local sink): the trickle thins out and dies at the center.
      for (const i of ins) {
        const [ix, iz] = midOf(i.d);
        pushCurve(ix, iz, (ix + x) / 2, (iz + z) / 2, x, z, ySurf + (lift += Y_JITTER), i.w, TRAIL_WIDTH * 0.02);
      }
    }
  }

  if (positions.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  // flatShading → no per-vertex normals (F4).

  return new THREE.Mesh(geometry, WATER_MAT);
}

// --- Build clouds mesh (hex puffs above terrain, proxy humidity_upper) ---
//
// Principle: we render a "cloud layer" as translucent hexagonal prisms
// floating above the terrain, at the same x/z as the cell but at a
// common ceiling altitude. humidity_upper drives the cloud's THICKNESS
// and TINT, the more humid the cell is aloft, the thicker and whiter the
// prism (visually denser column at constant material opacity). Below
// the threshold, no prism.
//
// Choice: no per-vertex alpha (complicates transparent rendering in
// Three.js). Visual density variation goes through the prism's
// thickness + color (dense white vs light bluish gray), which gives the
// impression of scattered cumulus versus thick cover.
// Cloud prototype: unit hex prism (radius HEX_SIZE, y ∈ [0,1]) built
// ONCE. Per-vertex color carries the vertical gradient (white roof
// [1,1,1], bluish base [0.72,0.76,0.88]); multiplied by the instance
// color (= density tint), it reproduces exactly the old
// topCol/botCol. shrink and thick become instance scales, position
// comes from the cell.
const CLOUD_TOP_TINT = [1, 1, 1];
const CLOUD_BOT_TINT = [0.72, 0.76, 0.88];
function buildCloudProto() {
  const verts = hexVertices(0, 0); // 6 corners, radius HEX_SIZE, centered at origin
  const pos = [];
  const col = [];
  const idx = [];
  let vc = 0;
  // Roof (y=1)
  const tc = vc;
  pos.push(0, 1, 0); col.push(...CLOUD_TOP_TINT); vc++;
  for (const v of verts) { pos.push(v.x, 1, v.z); col.push(...CLOUD_TOP_TINT); vc++; }
  for (let i = 0; i < 6; i++) idx.push(tc, tc + 1 + ((i + 1) % 6), tc + 1 + i);
  // Bottom (y=0)
  const bc = vc;
  pos.push(0, 0, 0); col.push(...CLOUD_BOT_TINT); vc++;
  for (const v of verts) { pos.push(v.x, 0, v.z); col.push(...CLOUD_BOT_TINT); vc++; }
  for (let i = 0; i < 6; i++) idx.push(bc, bc + 1 + i, bc + 1 + ((i + 1) % 6));
  // Walls (roof→base gradient)
  for (let i = 0; i < 6; i++) {
    const v0 = verts[i];
    const v1 = verts[(i + 1) % 6];
    const b = vc;
    pos.push(v0.x, 1, v0.z); col.push(...CLOUD_TOP_TINT);
    pos.push(v1.x, 1, v1.z); col.push(...CLOUD_TOP_TINT);
    pos.push(v1.x, 0, v1.z); col.push(...CLOUD_BOT_TINT);
    pos.push(v0.x, 0, v0.z); col.push(...CLOUD_BOT_TINT);
    vc += 4;
    idx.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  return g;
}

// Since the 3-stock model: we paint cloud_water (condensed, visible
// droplets) and not humidity_upper (vapor, invisible). Render density ∈
// [0, 1]: linear ramp over the window [CLOUD_RENDER_MIN,
// CLOUD_RENDER_FULL] derived from precip_crit_mm (see the
// CLOUD_WINDOW_* block at the top of the file).
function cloudDensityOf(cloudWaterMm) {
  const h = cloudWaterMm || 0;
  if (h < CLOUD_RENDER_MIN) return 0;
  return Math.min(1, (h - CLOUD_RENDER_MIN) / (CLOUD_RENDER_FULL - CLOUD_RENDER_MIN));
}

const CLOUD_BASE_HEIGHT = 3.5;
const CLOUD_THICK_MIN = 0.05;
const CLOUD_THICK_MAX = 0.7;
const CLOUD_SHRINK_MIN = 0.25;
const CLOUD_SHRINK_MAX = 0.95;

// Fills `cloudsInst` instances (created in rebuildScene) from a
// density array [0, 1] per cell. One instance per cell with density
// > 0: scale (shrink, thick, shrink), position (x, ceilingY, z), color =
// density tint. `count` = active cells; buffers uploaded over
// [0, count]. SINGLE writer of the instances: called by buildClouds (direct
// placement on rebuild) AND by tickCloudInterpolation (per-frame fade).
function fillCloudInstances(cells, dens) {
  if (!cloudsInst || cloudsInst.instanceMatrix.count < cells.length) return;
  if (!showClouds) {
    cloudsInst.visible = false;
    return;
  }
  let maxElev = 0;
  for (const cell of cells) if (cell.elevation > maxElev) maxElev = cell.elevation;
  // Common absolute ceiling (keeps the clouds from undulating with the relief).
  const ceilingY = maxElev * ELEVATION_SCALE + CLOUD_BASE_HEIGHT;

  let i = 0;
  for (let c = 0; c < cells.length; c++) {
    const density = dens[c];
    if (density <= 0) continue;
    const thick = CLOUD_THICK_MIN + (CLOUD_THICK_MAX - CLOUD_THICK_MIN) * density;
    const shrink = CLOUD_SHRINK_MIN + (CLOUD_SHRINK_MAX - CLOUD_SHRINK_MIN) * density;
    const { x, z } = hexToWorld(cells[c].q, cells[c].r);
    _cloudMtx.makeScale(shrink, thick, shrink);
    _cloudMtx.setPosition(x, ceilingY, z);
    cloudsInst.setMatrixAt(i, _cloudMtx);
    cloudsInst.setColorAt(
      i,
      _cloudCol.setRGB(0.78 + 0.20 * density, 0.82 + 0.16 * density, 0.88 + 0.10 * density),
    );
    i++;
  }
  cloudsInst.count = i;
  cloudsInst.visible = i > 0;
  cloudsInst.instanceMatrix.clearUpdateRanges();
  cloudsInst.instanceMatrix.addUpdateRange(0, i * 16);
  cloudsInst.instanceMatrix.needsUpdate = true;
  if (cloudsInst.instanceColor) {
    cloudsInst.instanceColor.clearUpdateRanges();
    cloudsInst.instanceColor.addUpdateRange(0, i * 3);
    cloudsInst.instanceColor.needsUpdate = true;
  }
}

// Placement on rebuild: paints the current fade state if it matches the
// grid (don't overwrite an ongoing interpolation with its target),
// otherwise the snapshot's densities directly (first render, reset,
// toggle before any captureClouds).
function buildClouds(cells) {
  if (!showClouds) {
    if (cloudsInst) cloudsInst.visible = false;
    return;
  }
  if (cloudDensCurrent && cloudDensCurrent.length === cells.length) {
    fillCloudInstances(cells, cloudDensCurrent);
    return;
  }
  const dens = new Float32Array(cells.length);
  for (let c = 0; c < cells.length; c++) dens[c] = cloudDensityOf(cells[c].cloud_water);
  fillCloudInstances(cells, dens);
}

// --- Instanced snow (white sheet per snowy cell) ---------------------------
// Prototype: unit hex sheet, cap (y=1) + 6 walls down to the terrain (y=0),
// radius HEX_SIZE, built ONCE. No bottom (the terrain closes it below),
// no per-vertex color (SNOW_MAT is solid white).
function buildSnowProto() {
  const verts = hexVertices(0, 0);
  const pos = [];
  const idx = [];
  let vc = 0;
  // Cap (y=1)
  const tc = vc;
  pos.push(0, 1, 0); vc++;
  for (const v of verts) { pos.push(v.x, 1, v.z); vc++; }
  for (let i = 0; i < 6; i++) idx.push(tc, tc + 1 + ((i + 1) % 6), tc + 1 + i);
  // Walls (y=1 → y=0): give the sheet its thickness seen in profile.
  for (let i = 0; i < 6; i++) {
    const v0 = verts[i];
    const v1 = verts[(i + 1) % 6];
    const sb = vc;
    pos.push(v0.x, 1, v0.z);
    pos.push(v1.x, 1, v1.z);
    pos.push(v1.x, 0, v1.z);
    pos.push(v0.x, 0, v0.z);
    vc += 4;
    idx.push(sb, sb + 1, sb + 2, sb, sb + 2, sb + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

// Fills `snowInst` instances: one per snowy cell, base at the terrain
// (topY), scale Y = snow thickness (+0.02 anti-z-fight), scale XZ = 1
// (the prototype is already at radius HEX_SIZE). No instance color.
function buildSnow(cells) {
  let i = 0;
  for (const cell of cells) {
    if (cell.snow_level < 0.01) continue;
    const { x, z } = hexToWorld(cell.q, cell.r);
    const topY = cell.elevation * ELEVATION_SCALE;
    const snowHeight = snowVisualHeight(cell.snow_level) * ELEVATION_SCALE;
    _cloudMtx.makeScale(1, snowHeight + 0.02, 1);
    _cloudMtx.setPosition(x, topY, z);
    snowInst.setMatrixAt(i, _cloudMtx);
    i++;
  }
  snowInst.count = i;
  snowInst.visible = i > 0;
  snowInst.instanceMatrix.clearUpdateRanges();
  snowInst.instanceMatrix.addUpdateRange(0, i * 16);
  snowInst.instanceMatrix.needsUpdate = true;
}

// --- Build forest (instanced low-poly trees) -------------------------------
// One hex = ~100 ha: we don't render actual trees but a *representative
// cover*. Each vegetated hex scatters N stylized trees, driven blindly
// by the core's data (anti-pattern #2):
//   • count        ∝ vegetation (total cover)
//   • species      drawn from species_mix (the mix, not just the dominant one)
//   • shape        conifer (pine/fir) = cone; broadleaf (oak/beech) = blob
//   • height       ∝ stand_age (young sapling → old-growth stand)
//   • position     hash(q, r, i) → deterministic, stable tick to tick (no
//                  flicker), same spirit as the sub-hex decor seeded by (q,r)
// Alpine grass doesn't carry a tree: it's a ground cover, rendered via the
// terrain color. A pure-lawn hex therefore stays treeless (correct).
const FOREST_MODES = new Set(["terrain", "species", "age"]);
const CONIFER_SPECIES = new Set(["pine", "fir"]);
const GRASS_SPECIES = new Set(["alpine_grass"]);
const MAX_TREES_PER_HEX = 20;
const MIN_VEG_FOR_TREES = 0.05;
// Global budget of instanced trees (F16 / LOD). Without a ceiling, a mature
// world at r200 generates >750k trees (round(veg·20) per cell over ~100k
// cells), more triangles than everything else combined, and sub-pixel at
// full-map view. We cap the total: beyond it, we thin all cells
// proportionally (dense forests keep the most trees). At r60/r100 the
// natural total is under the budget, so no effect (it's an anti-explosion
// safeguard for large maps).
const FOREST_BUDGET = 120000;

// Deterministic integer hash (q, r, i) → [0, 1). Used to seed positions,
// sizes, and species draws without Math.random (determinism = no flicker).
function hash3(a, b, c) {
  let h = 2166136261 >>> 0;
  h = Math.imul(h ^ (a & 0xffff), 16777619);
  h = Math.imul(h ^ (b & 0xffff), 16777619);
  h = Math.imul(h ^ (c & 0xffff), 16777619);
  h ^= h >>> 13;
  h = Math.imul(h, 0x5bd1e995);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

// Draws a species proportionally to the mix (species_mix normalized). `r`
// is in [0,1). Returns the name (SPECIES_COLORS key) or null for a bare hex.
function pickSpeciesByMix(mix, order, r) {
  let sum = 0;
  for (const m of mix) sum += m;
  if (sum <= 0) return null;
  let x = r * sum;
  for (let i = 0; i < order.length; i++) {
    x -= mix[i];
    if (x <= 0) return order[i];
  }
  return order[order.length - 1];
}

function buildForest(cells, mode, speciesOrder) {
  if (!FOREST_MODES.has(mode) || !Array.isArray(speciesOrder)) return null;

  // Pass 1 (F16): natural total of trees. Beyond the budget, we trim all
  // cells by the same factor → total ≈ budget, relative density preserved.
  let naturalTotal = 0;
  for (const cell of cells) {
    if (cell.is_open_water) continue;
    const veg = cell.vegetation ?? 0;
    if (veg < MIN_VEG_FOR_TREES || !Array.isArray(cell.species_mix)) continue;
    naturalTotal += Math.round(veg * MAX_TREES_PER_HEX);
  }
  const treesPerHex =
    naturalTotal > FOREST_BUDGET
      ? MAX_TREES_PER_HEX * (FOREST_BUDGET / naturalTotal)
      : MAX_TREES_PER_HEX;

  // Two silhouette categories, each its own separate InstancedMesh.
  const conifer = []; // { x, y, z, r, h, cr, cg, cb }
  const broadleaf = [];

  for (const cell of cells) {
    if (cell.is_open_water) continue;
    const veg = cell.vegetation ?? 0;
    if (veg < MIN_VEG_FOR_TREES || !Array.isArray(cell.species_mix)) continue;

    const n = Math.round(veg * treesPerHex);
    if (n < 1) continue;

    const { x: cx, z: cz } = hexToWorld(cell.q, cell.r);
    const verts = hexVertices(cx, cz);
    const topY = cell.elevation * ELEVATION_SCALE;
    const ageF = 0.55 + 0.45 * Math.min(1, (cell.stand_age ?? 0) / 70);

    for (let i = 0; i < n; i++) {
      const b = i * 8; // base salt per tree: disjoint hash indices
      const sp = pickSpeciesByMix(cell.species_mix, speciesOrder, hash3(cell.q, cell.r, b + 4));
      if (!sp || GRASS_SPECIES.has(sp)) continue; // grass = no tree
      const base = SPECIES_COLORS[sp];
      if (!base) continue;

      // Position: uniform sample over the WHOLE hexagon (up to the edges),
      // not a disc centered on it, otherwise trees clump in the center and
      // two adjacent hexes don't connect (a "polka-dot" effect on plains).
      // We draw one of the 6 triangles (center, corner t, corner t+1) then
      // a uniform point inside it (barycentric fold of the unit square onto
      // the triangle).
      const t = Math.floor(hash3(cell.q, cell.r, b) * 6) % 6;
      let s1 = hash3(cell.q, cell.r, b + 1);
      let s2 = hash3(cell.q, cell.r, b + 2);
      if (s1 + s2 > 1) { s1 = 1 - s1; s2 = 1 - s2; }
      const va = verts[t];
      const vb = verts[(t + 1) % 6];
      const px = cx + s1 * (va.x - cx) + s2 * (vb.x - cx);
      const pz = cz + s1 * (va.z - cz) + s2 * (vb.z - cz);

      // Size: height ∝ age, with per-tree jitter.
      const jit = 0.8 + 0.35 * hash3(cell.q, cell.r, b + 3);
      const shade = 0.58 + 0.3 * hash3(cell.q, cell.r, b + 5);
      const cr = base[0] * shade;
      const cg = base[1] * shade;
      const cb = base[2] * shade;

      if (CONIFER_SPECIES.has(sp)) {
        const h = 0.5 * ageF * jit;
        const r = 0.16 * ageF * jit;
        conifer.push({ x: px, y: topY, z: pz, r, h, cr, cg, cb });
      } else {
        const h = 0.3 * ageF * jit;
        const r = 0.2 * ageF * jit;
        broadleaf.push({ x: px, y: topY, z: pz, r, h, cr, cg, cb });
      }
    }
  }

  if (conifer.length === 0 && broadleaf.length === 0) return null;

  const group = new THREE.Group();
  const mtx = new THREE.Matrix4();
  const col = new THREE.Color();

  // Cone (radius 1, height 1, centered) → conifers. 6 segments = low-poly.
  if (conifer.length > 0) {
    const geo = new THREE.ConeGeometry(1, 1, 6);
    const mesh = new THREE.InstancedMesh(geo, TREE_MAT, conifer.length);
    conifer.forEach((t, i) => {
      // ConeGeometry is vertically centered: base at -h/2, place it on the ground.
      mtx.makeScale(t.r, t.h, t.r);
      mtx.setPosition(t.x, t.y + t.h / 2, t.z);
      mesh.setMatrixAt(i, mtx);
      mesh.setColorAt(i, col.setRGB(t.cr, t.cg, t.cb));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    group.add(mesh);
  }

  // Icosahedron (radius 1, detail 0 = 20 faces) → broadleaves, slightly flattened.
  if (broadleaf.length > 0) {
    const geo = new THREE.IcosahedronGeometry(1, 0);
    const mesh = new THREE.InstancedMesh(geo, TREE_MAT, broadleaf.length);
    broadleaf.forEach((t, i) => {
      mtx.makeScale(t.r, t.h, t.r);
      mtx.setPosition(t.x, t.y + t.h, t.z); // vertical radius = h → base on the ground
      mesh.setMatrixAt(i, mtx);
      mesh.setColorAt(i, col.setRGB(t.cr, t.cg, t.cb));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    group.add(mesh);
  }

  return group;
}

// --- Weather particles (evaporation only) ---
// Rain-as-particles was removed: too subtle to be a useful diagnostic,
// replaced by the Precipitation overlay (blue→violet tint on the hex). We
// keep the evaporation points, which read well above puddles as a light
// mist effect.
const EVAP_HEIGHT = 1.5;

function buildWeatherParticles(cells) {
  const evapPositions = [];
  const evapBase = [];

  for (const cell of cells) {
    const { x, z } = hexToWorld(cell.q, cell.r);
    const waterH = Math.max(0, cell.water_level || 0) * STOCK_VISUAL;
    const snowH = snowVisualHeight(cell.snow_level);
    const surfaceY = (cell.elevation + waterH + snowH) * ELEVATION_SCALE;

    // Evaporation: only on actual puddles (sub-capacity); lakes (overflow)
    // are already rendered as a solid prism, no need to add blue points
    // there that would blur the reading.
    const cap = cell.water_capacity || 1.0;
    if (cell.water_level > 0.3 && cell.water_level <= cap) {
      const ox = (Math.random() - 0.5) * HEX_SIZE * 0.6;
      const oz = (Math.random() - 0.5) * HEX_SIZE * 0.6;
      const oy = surfaceY + 0.2 + EVAP_HEIGHT * Math.random();
      evapPositions.push(x + ox, oy, z + oz);
      evapBase.push(surfaceY);
    }
  }

  if (evapPositions.length === 0) return null;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(evapPositions, 3));
  const mesh = new THREE.Points(geo, EVAP_MAT);
  mesh.userData.baseY = new Float32Array(evapBase);
  return mesh;
}

// --- Build wind arrows (one arrow per cell, merged geometry) ---
function buildWindArrows(cells, mode) {
  const positions = [];
  const colors = [];
  const indices = [];
  let vertCount = 0;

  // Pressure view: arrows = raw synoptic wind (m/s SI → WindVec unit ÷10),
  // the geostrophic circulation around the isobars, without thermal breeze
  // or orographic deflection. Elsewhere: the usual consumed wind.
  const synoptic = mode === "pressure";
  for (const cell of cells) {
    const wx = synoptic ? (cell.synoptic_u ?? 0) * 0.1 : cell.wind_x || 0;
    const wy = synoptic ? (cell.synoptic_v ?? 0) * 0.1 : cell.wind_y || 0;
    const mag = Math.sqrt(wx * wx + wy * wy);
    if (mag < 0.01) continue;

    const { x, z } = hexToWorld(cell.q, cell.r);
    const y = cell.elevation * ELEVATION_SCALE + 0.25;

    // Normalized direction (wind_x → world X, wind_y → world Z)
    const dx = wx / mag;
    const dz = wy / mag;

    // Perpendicular in the XZ plane
    const px = -dz;
    const pz = dx;

    // Non-linear size (sqrt) to make magnitude differences visible
    const t = Math.sqrt(Math.min(mag, 1.0));
    const arrowLen = 0.1 + 0.75 * t;
    const shaftLen = arrowLen * 0.55;
    const shaftW = 0.02 + 0.04 * t;
    const headW = 0.05 + 0.15 * t;

    // Shaft : rectangle (4 vertices, 2 triangles)
    positions.push(
      x - px * shaftW, y, z - pz * shaftW,
      x + px * shaftW, y, z + pz * shaftW,
      x + dx * shaftLen + px * shaftW, y, z + dz * shaftLen + pz * shaftW,
      x + dx * shaftLen - px * shaftW, y, z + dz * shaftLen - pz * shaftW,
    );

    // Head : triangle (3 vertices, 1 triangle)
    positions.push(
      x + dx * shaftLen - px * headW, y, z + dz * shaftLen - pz * headW,
      x + dx * shaftLen + px * headW, y, z + dz * shaftLen + pz * headW,
      x + dx * arrowLen, y, z + dz * arrowLen,
    );

    // Couleur : bleu sombre (faible) → blanc vif (fort)
    const ci = Math.min(1, mag * 3.0);
    const cr = 0.2 + 0.8 * ci;
    const cg = 0.35 + 0.65 * ci;
    const cb = 0.6 + 0.4 * ci;
    for (let j = 0; j < 7; j++) colors.push(cr, cg, cb);

    // Shaft : 2 triangles
    indices.push(vertCount, vertCount + 1, vertCount + 2);
    indices.push(vertCount, vertCount + 2, vertCount + 3);
    // Head : 1 triangle
    indices.push(vertCount + 4, vertCount + 5, vertCount + 6);
    vertCount += 7;
  }

  if (positions.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);

  return new THREE.Mesh(geometry, WIND_MAT);
}

// --- Build temperature contours (colored ribbon around each hex's roof) ---
// Stackable overlay: a blue→green→red rim traced just above the hex's
// roof, on the periphery only. Leaves the fill visible underneath.
// Rendered as a ribbon (quad per edge) for a real thickness in world
// units, more reliable than GL_LINES, whose width > 1 isn't portable.
function buildTemperatureContours(cells) {
  const positions = [];
  const colors = [];
  const indices = [];
  let vc = 0;

  const RIM_WIDTH = 0.12;              // ribbon thickness (world units)
  const Y_LIFT = 0.04;                 // avoids z-fighting with the roof
  const inner = 1 - RIM_WIDTH / HEX_SIZE;

  for (const cell of cells) {
    const { x, z } = hexToWorld(cell.q, cell.r);
    const y = cell.elevation * ELEVATION_SCALE + Y_LIFT;
    const col = temperatureColor(cell.temperature ?? 15);
    const outer = hexVertices(x, z);

    for (let i = 0; i < 6; i++) {
      const next = (i + 1) % 6;
      const o0 = outer[i];
      const o1 = outer[next];
      const i0x = x + (o0.x - x) * inner;
      const i0z = z + (o0.z - z) * inner;
      const i1x = x + (o1.x - x) * inner;
      const i1z = z + (o1.z - z) * inner;

      positions.push(o0.x, y, o0.z);
      positions.push(o1.x, y, o1.z);
      positions.push(i1x,  y, i1z);
      positions.push(i0x,  y, i0z);
      for (let k = 0; k < 4; k++) colors.push(col.r, col.g, col.b);

      indices.push(vc, vc + 1, vc + 2);
      indices.push(vc, vc + 2, vc + 3);
      vc += 4;
    }
  }

  if (positions.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);

  return new THREE.Mesh(geometry, TEMP_MAT);
}

// --- Build precipitation overlay (tinted hex top floating above everything) ---
// A hexagonal disc per cell with active precipitation, positioned above the
// highest surface (terrain + any lake + accumulated snow) so the white snow
// doesn't mask the tint. Single alpha on the material, intensity encoded by
// the hue from pale cyan to deep violet.
function buildPrecipitationOverlay(cells) {
  const positions = [];
  const colors = [];
  const indices = [];
  let vc = 0;

  const Y_LIFT = 0.10; // above the snow roof (snow Top = topY + snowH + 0.02)

  for (const cell of cells) {
    const col = precipitationTintColor(cell);
    if (!col) continue;

    const { x, z } = hexToWorld(cell.q, cell.r);
    const cap = cell.water_capacity || 1.0;
    // Water surface at the physical height (consistent with the lake prism).
    const overflowH = Math.max(0, (cell.water_level || 0) - cap) * WATER_SURFACE_M;
    const snowH = snowVisualHeight(cell.snow_level);
    const y = (cell.elevation + overflowH + snowH) * ELEVATION_SCALE + Y_LIFT;

    const centerIdx = vc;
    positions.push(x, y, z);
    colors.push(col.r, col.g, col.b);
    vc++;

    const verts = hexVertices(x, z);
    for (const v of verts) {
      positions.push(v.x, y, v.z);
      colors.push(col.r, col.g, col.b);
      vc++;
    }

    for (let i = 0; i < 6; i++) {
      const next = (i + 1) % 6;
      indices.push(centerIdx, centerIdx + 1 + next, centerIdx + 1 + i);
    }
  }

  if (positions.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);

  return new THREE.Mesh(geometry, PRECIP_MAT);
}

// Animates the particles: evaporation rises (relative to the cell's surface).
// Rain used to be rendered here as points, but it wasn't very legible; it is
// now carried by the blue→violet tint of the Precipitation overlay.
function animateParticles() {
  if (evapParticles) {
    const pos = evapParticles.geometry.attributes.position;
    const baseY = evapParticles.userData.baseY;
    for (let i = 0; i < pos.count; i++) {
      let y = pos.getY(i);
      y += 0.03;
      if (y > baseY[i] + EVAP_HEIGHT) y = baseY[i] + 0.2;
      pos.setY(i, y);
    }
    pos.needsUpdate = true;
  }
}


// --- Full scene rebuild ---
// Releases the VRAM (geometry + material) of an object BEFORE removing it
// from the scene. `scene.remove()` alone does NOT free the GPU: without
// this, every rebuild (1 per snapshot) leaks 8 geometries + 8 materials, the
// VRAM fills up, and the WebGL context is lost / crashes after a few
// minutes (the "days-long leak" bug). `traverse()` covers Mesh, Points,
// Line and Group alike.
function disposeObject3D(obj) {
  if (!obj) return;
  scene.remove(obj);
  obj.traverse((child) => {
    // InstancedMesh.dispose() also frees instanceMatrix/instanceColor (VRAM).
    if (child.isInstancedMesh) child.dispose();
    if (child.geometry) child.geometry.dispose();
    // Materials are module-level constants shared across rebuilds (F5): we
    // never dispose them, only the (unique) geometries are freed.
  });
}

function rebuildScene(state) {
  // Persistent terrain (F3): geometry rebuilt only if the world has changed
  // (relief signature). Otherwise, we only rewrite the colors: no
  // dispose/rebuild, no re-uploading of positions.
  const sig = terrainSignature(state.cells);
  if (!terrainMesh || sig !== terrainSig) {
    disposeObject3D(terrainMesh);
    if (terrainColorTex) terrainColorTex.dispose(); // the old texture (VRAM)
    terrainMesh = buildTerrainGeometry(state.cells); // Group of tiles (#132)
    scene.add(terrainMesh);
    terrainSig = sig;
    terrainBBox = new THREE.Box3().setFromObject(terrainMesh);
    updateViewDistances();
  }
  updateTerrainColors(state.cells, viewModeEl.value);

  // Dynamic layers still rebuilt on every snapshot (water, …).
  // Clouds and snow are instanced (updated further below, not disposed).
  disposeObject3D(waterMesh);
  disposeObject3D(forestMesh);
  disposeObject3D(evapParticles);
  disposeObject3D(windArrows);
  disposeObject3D(temperatureContours);
  disposeObject3D(precipitationOverlay);

  // Pressure view: bare isobar map; the water (blue, the color of highs)
  // and the snow (covers everything in winter) would mask the colored field.
  const bareMap = viewModeEl.value === "pressure";

  waterMesh = bareMap ? null : buildWater(state.cells, state.edge_flux_max || 0);
  if (waterMesh) scene.add(waterMesh);

  // Snow: persistent InstancedMesh (one prototype, N instances).
  if (!snowProtoGeom) snowProtoGeom = buildSnowProto();
  if (!snowInst || snowInst.instanceMatrix.count < state.cells.length) {
    if (snowInst) { scene.remove(snowInst); snowInst.dispose(); }
    snowInst = new THREE.InstancedMesh(snowProtoGeom, SNOW_MAT, state.cells.length);
    snowInst.frustumCulled = false;
    snowInst.count = 0;
    scene.add(snowInst);
  }
  buildSnow(bareMap ? [] : state.cells);

  // Forest: instanced trees, only on the map backgrounds where vegetation
  // makes sense (terrain / species / age).
  forestMesh = buildForest(state.cells, viewModeEl.value, state.species_order);
  if (forestMesh) scene.add(forestMesh);

  if (showWind) {
    windArrows = buildWindArrows(state.cells, viewModeEl.value);
    if (windArrows) scene.add(windArrows);
  } else {
    windArrows = null;
  }

  if (showTemperature) {
    temperatureContours = buildTemperatureContours(state.cells);
    if (temperatureContours) scene.add(temperatureContours);
  } else {
    temperatureContours = null;
  }

  if (showPrecipitation) {
    precipitationOverlay = buildPrecipitationOverlay(state.cells);
    if (precipitationOverlay) scene.add(precipitationOverlay);
  } else {
    precipitationOverlay = null;
  }

  // Clouds: persistent InstancedMesh (one prototype, N instances). Created
  // once, sized to the cell count; buildClouds only fills in the active
  // instance matrices/colors. Independent of the map background.
  if (!cloudProtoGeom) cloudProtoGeom = buildCloudProto();
  if (!cloudsInst || cloudsInst.instanceMatrix.count < state.cells.length) {
    if (cloudsInst) {
      scene.remove(cloudsInst);
      cloudsInst.dispose();
    }
    cloudsInst = new THREE.InstancedMesh(cloudProtoGeom, CLOUDS_MAT, state.cells.length);
    cloudsInst.frustumCulled = false; // footprint = the whole map
    cloudsInst.count = 0;
    scene.add(cloudsInst);
  }
  buildClouds(state.cells);

  evapParticles = buildWeatherParticles(state.cells);
  if (evapParticles) scene.add(evapParticles);

  // Center the camera on the terrain on first render
  if (state.cells.length > 0 && camera.position.y === 30) {
    const box = new THREE.Box3().setFromObject(terrainMesh);
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    camera.position.set(center.x, 30, center.z + 25);
  }
}

// --- Map width label (bar on the south edge of the terrain) ---
const mapWidthEl = document.getElementById("map-width");
const mapWidthLabelEl = document.getElementById("map-width-label");
const mwLeft = new THREE.Vector3();
const mwRight = new THREE.Vector3();

function updateMapWidthLabel() {
  // The graduated ruler is a reading tool for whoever drives the simulation:
  // out of place in an embed, where it would float under the map without
  // context. Decided here rather than in CSS: this element's visibility is
  // driven by inline style further down, which no stylesheet can override.
  if (EMBED || !terrainBBox) { mapWidthEl.style.display = "none"; return; }
  // The map is one big regular hex. Side = flat-to-flat / √3 (world units).
  const bx = terrainBBox.max.x - terrainBBox.min.x;
  const bz = terrainBBox.max.z - terrainBBox.min.z;
  const sideWorld = Math.min(bx, bz) / SQRT3;
  const centerX = (terrainBBox.min.x + terrainBBox.max.x) / 2;
  const zSouth = terrainBBox.max.z; // +Z = south (top-down convention)
  mwLeft.set(centerX - sideWorld / 2, 0, zSouth).project(camera);
  mwRight.set(centerX + sideWorld / 2, 0, zSouth).project(camera);
  const W = window.innerWidth;
  const H = window.innerHeight;
  const lx = (mwLeft.x + 1) * 0.5 * W;
  const ly = (1 - mwLeft.y) * 0.5 * H;
  const rx = (mwRight.x + 1) * 0.5 * W;
  const ry = (1 - mwRight.y) * 0.5 * H;
  const dx = rx - lx;
  const dy = ry - ly;
  const len = Math.hypot(dx, dy);
  if (!Number.isFinite(len) || len <= 0) { mapWidthEl.style.display = "none"; return; }
  const angle = Math.atan2(dy, dx);
  const cx = (lx + rx) / 2;
  const cy = (ly + ry) / 2;
  const km = sideWorld * KM_PER_WORLD;
  mapWidthEl.style.display = "block";
  mapWidthEl.style.width = `${len}px`;
  mapWidthEl.style.transform = `translate(${cx - len / 2}px, ${cy + 8}px) rotate(${angle}rad)`;
  mapWidthLabelEl.textContent = `${km.toFixed(1)} km`;
}

// --- Scale bar ---
const scaleBarLineEl = document.getElementById("scale-bar-line");
const scaleBarLabelEl = document.getElementById("scale-bar-label");
const NICE_KM = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500];
const scaleTmpA = new THREE.Vector3();
const scaleTmpB = new THREE.Vector3();
const scaleTmpRight = new THREE.Vector3();

function updateScaleBar() {
  // "Screen right" direction projected onto the ground plane (y=0)
  scaleTmpRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
  scaleTmpRight.y = 0;
  if (scaleTmpRight.lengthSq() < 1e-6) return;
  scaleTmpRight.normalize();

  const target = controls.target;
  const worldPerKm = 1 / KM_PER_WORLD;
  scaleTmpA.copy(target);
  scaleTmpB.copy(target).addScaledVector(scaleTmpRight, worldPerKm);
  scaleTmpA.project(camera);
  scaleTmpB.project(camera);

  const W = window.innerWidth;
  const H = window.innerHeight;
  const dx = (scaleTmpB.x - scaleTmpA.x) * 0.5 * W;
  const dy = (scaleTmpB.y - scaleTmpA.y) * 0.5 * H;
  const pxPerKm = Math.hypot(dx, dy);
  if (!Number.isFinite(pxPerKm) || pxPerKm <= 0) return;

  // Find the largest "round" value that fits under 180 px
  let km = NICE_KM[0];
  for (let i = NICE_KM.length - 1; i >= 0; i--) {
    if (NICE_KM[i] * pxPerKm <= 180) { km = NICE_KM[i]; break; }
  }
  const widthPx = Math.max(20, Math.round(km * pxPerKm));
  scaleBarLineEl.style.width = `${widthPx}px`;
  scaleBarLabelEl.textContent = km < 1 ? `${km * 1000} m` : `${km} km`;
}

// --- Render loop ---
let needsRebuild = false;
let isRebuilding = false;
let forceRebuild = false; // bypasses the throttle (view/toggle change)
const MIN_REBUILD_INTERVAL = 300; // floor (ms) between two rebuilds
let rebuildInterval = MIN_REBUILD_INTERVAL; // adaptive (F13), see maybeRebuild
let lastRebuildTime = -1e9; // very negative → the very first rebuild is immediate
let hudDirty = true; // compass + bars: redraw only if the camera moves or a new snapshot arrives

// Decodes the latest pending snapshot (F2) then rebuilds the scene, at an
// adaptive rate. Decoding msgpack + inflating cells costs hundreds of ms
// at r200: this is only done here, on the most recent frame, never on
// every WS arrival. A slow rebuild won't trigger another one before 4x
// its duration (F13) → ~75% of the time stays available for interactive
// frames, whatever the radius. View/toggle changes go through
// `forceRebuild` so they don't have to wait for this delay.
function maybeRebuild(now) {
  if (isRebuilding) return;
  if (!forceRebuild && now - lastRebuildTime <= rebuildInterval) return;
  if (latestSnapshotBuf) {
    const state = decodeBinaryFrame(latestSnapshotBuf);
    latestSnapshotBuf = null;
    applySnapshot(state); // lastState + stats + light + compass
  }
  if (!needsRebuild || !lastState) return;
  forceRebuild = false;
  isRebuilding = true;
  const t0 = performance.now();
  try {
    rebuildScene(lastState);
  } finally {
    // finally: an exception in rebuildScene must NOT freeze rendering
    // forever by leaving isRebuilding=true (F12).
    isRebuilding = false;
  }
  rebuildInterval = Math.max(MIN_REBUILD_INTERVAL, (performance.now() - t0) * 4);
  needsRebuild = false;
  lastRebuildTime = now;
}

// `ready` in the embed sense: terrain built AND first snapshot received,
// the same definition as `__hexcam.ready()`. Emitted from the render loop
// rather than from applySnapshot, since the terrain arrives at rebuild,
// not at snapshot, and the loop is the only point that reliably sees both.
let embedReadyEmitted = false;

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  maybeRebuild(now);
  if (!embedReadyEmitted && terrainMesh && lastState) {
    embedReadyEmitted = true;
    window.__hexsim?._emit("ready");
  }
  // Smoothed day/night cycle: interpolates light AND shadows (per-cell
  // illumination) between two hourly snapshots at 60 fps (the core only
  // sends one per simulated hour).
  tickLightInterpolation(now);
  tickIlluminationInterpolation(now);
  tickCloudInterpolation(now);
  animateParticles();
  // Compass + scale bars: depend on the camera and the latest snapshot. We
  // only recompute them when one of the two changes (F14); otherwise a
  // still camera means constant DOM style recalc at 60 fps for nothing.
  const moved = controls.update();
  if (moved || hudDirty) {
    drawCompass(lastAvgWind.x, lastAvgWind.y, controls.getAzimuthalAngle());
    updateScaleBar();
    updateMapWidthLabel();
    hudDirty = false;
  }
  // Distance culling of terrain tiles (#132), after the camera movement,
  // before the render. Per-tile frustum culling is handled by three.
  updateTileVisibility();
  renderer.render(scene, camera);
}
animate();

// --- Resize ---
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

initLogPanel();

// --- Transport + controls ---
// The transport connects this front end to a simulation: the `hexsim-cli`
// server via WebSocket, or the WASM module in a Worker. The rest of the file
// doesn't know which of the two feeds it; same commands, same snapshots.
const transport = createTransport();
// The mode is carried by the <body>: CSS uses it to reveal what only makes
// sense without a server (the warning about the in-memory world).
document.body.dataset.mode = resolveMode();
// `data-chrome` is set by the inline script at the top of index.html, not
// here: by this point in the file, the UI has already been painted (#140).

function send(cmd) {
  transport.send(cmd);
}

// Binary frame = msgpack (wire snapshots, perf); text frame = JSON
// (logs, targeted responses). A snapshot's cells arrive as positional
// rows plus a `cell_fields` manifest (sent once per frame, not per
// cell, that's the whole gain of the format): we inflate them back into
// objects here so the rest of the code stays identical to the JSON protocol.
function decodeBinaryFrame(buffer) {
  const msg = msgpackDecode(buffer);
  if (msg.cell_fields && msg.cells) {
    const fields = msg.cell_fields;
    msg.cells = msg.cells.map((row) => {
      const cell = {};
      for (let i = 0; i < fields.length; i++) cell[fields[i]] = row[i];
      return cell;
    });
    delete msg.cell_fields;
  }
  return msg;
}

// Applies a decoded snapshot: sidebar stats, light, compass, and arms a
// rebuild. Called at the rebuild rate (not on every WS arrival); stats
// and the compass therefore update at ~rebuild cadence, imperceptibly.
// `tick` in the embed sense: "what a host displays has changed". Emitted
// on every snapshot, but also on every play/pause toggle; otherwise a
// host that subscribes right after a `play()` gets an immediate callback
// with the last known tick, with a `playing` that's already false.
//
// `date` is the string HexSim displays itself: a host that redraws the
// bar shows the same date as normal mode, without having to reproduce
// the hour-display rule.
function emitEmbedTick() {
  if (!lastState) return;
  window.__hexsim?._emit("tick", {
    date: tickCountEl.textContent,
    playing: isPlaying,
    tick: lastState.tick,
    speed: currentHps,
    // What is actually displayed, not what a host requested: an unknown
    // `view` is rejected, and the button bar must see it. Same principle
    // as `speed`, which states the applied value, not the requested one.
    view: viewModeEl.value,
    layers: activeLayers(),
    // So that a host drawing an "inspector" button can show it in the
    // right state without keeping its own copy, and see it move if dev
    // mode or another call changed it.
    inspect: inspectorOn,
  });
}

/** Identifiers of the currently enabled overlays. */
function activeLayers() {
  return Array.from(document.querySelectorAll("[data-layer]"))
    .filter((el) => el.checked)
    .map((el) => el.dataset.layer);
}

function applySnapshot(state) {
  // Hour shown if:
  //  - we're paused (the user just stepped, the hour is frozen and useful)
  //  - or if running in slow mode (< 24 h/s ≈ 1 j/s) or the eye can follow
  // Beyond 24 h/s in play, the hour scrolls by too fast, so we hide it.
  const showHour = !isPlaying || currentHps < 24;
  tickCountEl.textContent = tickToDate(state.tick, showHour ? state.hour_tick : null);
  const s = state.total_surface_water?.toFixed(1) ?? "-";
  const h = state.total_humidity?.toFixed(1) ?? "-";
  const cw = state.total_cloud_water?.toFixed(2) ?? "-";
  const g = state.total_groundwater?.toFixed(1) ?? "-";
  const sn = state.total_snow?.toFixed(1) ?? "-";
  // total_humidity already includes cloud_water (humidity_total = surface +
  // upper + cloud). So we don't add it a second time in the global total,
  // otherwise the same stock would be counted twice.
  const total = (
    (state.total_surface_water ?? 0) +
    (state.total_humidity ?? 0) +
    (state.total_groundwater ?? 0) +
    (state.total_snow ?? 0)
  ).toFixed(1);
  statSurface.textContent = s;
  statHumidity.textContent = h;
  statCloud.textContent = cw;
  // Average rainfall over the map: (total_precip / cells) × mm/unit gives
  // mm of rain on the average cell for this tick (=1 day). Annualized =
  // × 365 under the assumption of climate stability across the tick.
  const precipPerCell =
    (state.total_precip_this_tick ?? 0) / Math.max(1, state.cell_count ?? 1);
  const mmPerDay = precipPerCell;
  const mmPerYear = mmPerDay * 365;
  statRainMm.textContent = `${mmPerDay.toFixed(1)} mm/d (${Math.round(mmPerYear)} mm/yr)`;
  statGroundwater.textContent = g;
  statSnow.textContent = sn;
  statTotal.textContent = total;

  lastState = state;
  needsRebuild = true;
  hudDirty = true;
  // `date` is the string HexSim displays itself: a host that redraws the
  // bar shows the same date as normal mode, without reproducing the
  // hour-display rule.
  emitEmbedTick();
  // #47: solar intensity for the compass = instantaneous sin_elev_pos (hourly
  // cycle). CRITICAL: use `state.hour_tick` (hours), not `state.tick`
  // which is in *days* (v0.2.x compat). Otherwise a "day/night" cycle that
  // lasts 24 simulated days, a visual bug reported by the user.
  const hourTick = state.hour_tick ?? (state.tick * 24); // fallback for the old protocol
  lastSunIntensity = Math.max(0, computeSolarSinElevation(hourTick, getLatitudeDeg()));
  updateSceneLighting(hourTick);
  // Smooths shadows (core `illumination` field) on the same timeline as the
  // light: `updateSceneLighting` just decided whether this snapshot interpolates.
  captureIllumination(state.cells, lightInterpArmed);
  // Cloud fade on the same timeline as light and shadows.
  captureClouds(state.cells, lightInterpArmed);
  updateCompass(state);
}

// Dispatch of a decoded message (JSON text, or small perf binary).
//  - "log"       : event tracing -> log panel
//  - "perf"      : perf sample (binary, same channel as snapshots)
//  - "meta"      : version/build
//  - other type  : targeted responses not consumed here -> ignore
//  - no type     : grid snapshot (small world decoded without deferring)
function dispatchMessage(msg) {
  if (msg.type === "log") {
    appendLog(msg);
    return;
  }
  if (msg.type === "perf") {
    // `cpu_percent` and `rss_mb` come from `sysinfo`, which doesn't exist in
    // a browser. In WASM mode the memory is that of the module (exact) and
    // the CPU has no equivalent: its row is hidden by CSS, so we write
    // into an invisible element rather than add a branch here.
    document.getElementById("stat-cpu").textContent =
      typeof msg.cpu_percent === "number" ? msg.cpu_percent.toFixed(1) + " %" : "—";
    document.getElementById("stat-ram").textContent =
      typeof msg.rss_mb === "number" ? msg.rss_mb.toFixed(0) + " MB" : "—";
    // As long as no tick has run yet (the world starts paused), there's
    // nothing to measure: "—", not a "0 µs" that no real tick can produce.
    document.getElementById("stat-tick-us").textContent =
      !msg.tick_us
        ? "—"
        : msg.tick_us < 1000
          ? `${msg.tick_us} µs`
          : `${(msg.tick_us / 1000).toFixed(1)} ms`;
    return;
  }
  if (msg.type === "progress") {
    updateProgress(msg);
    return;
  }
  if (msg.type === "meta") {
    if (typeof msg.cell_spacing_m === "number" && msg.cell_spacing_m > 0) {
      applyCellSpacingM(msg.cell_spacing_m);
    }
    const el = document.getElementById("version-badge");
    if (el) {
      el.textContent = `v${msg.version} · ${msg.build_hash}`;
      const iso = msg.build_unix
        ? new Date(msg.build_unix * 1000).toISOString().slice(0, 16).replace("T", " ")
        : "unknown";
      el.title = `Build ${iso} UTC · hash ${msg.build_hash}`;
      el.classList.toggle("dirty", String(msg.build_hash).endsWith("-dirty"));
    }
    return;
  }
  if (msg.type === "params") {
    applyServerParams(msg);
    return;
  }
  if (msg.type) {
    // Typed responses we don't handle here. We ignore them to avoid polluting
    // the stats with missing fields.
    return;
  }
  applySnapshot(msg);
}

let embedAutostarted = false;

// The shipped world is only loaded once, even after a reconnection: the
// server kept the state, reimporting it would erase what has happened since.
let bootWorldDone = false;

/**
 * Serializes checkpoint imports, whatever their origin.
 *
 * Two sources compete for the world when an embed starts: the shipped world
 * (`worlds/aged.ckptz`, launched on connection) and a `__hexsim.load()` from
 * a host restoring its stored state, which nothing prevents from arriving
 * while the first is still in flight. Without a queue, two concurrent
 * `/checkpoint` POSTs land in whatever order the network decides, and the
 * host ends up seeing a world it didn't ask for about half the time.
 *
 * Call order becomes application order. A failed import doesn't block the
 * queue: the current world stays intact, the next one can go through.
 */
let importChain = Promise.resolve();
function queueCheckpointImport(fn) {
  const p = importChain.then(fn, fn);
  importChain = p.catch(() => {});
  return p;
}

/**
 * Resolved on the first `connected`, never before.
 *
 * `_attach` wires up the embed API **before** `connect()`: an `export()` or a
 * `load()` popped off the boot queue would land on a Worker that doesn't
 * exist yet (`this.worker?.postMessage`, silent no-op) or on a null `sim`
 * on the worker side. The promise would never resolve and the host would
 * wait without learning anything, the exact shape of the `play()` lost in #143.
 */
let signalerConnecte;
const moteurJoignable = new Promise((resolve) => {
  signalerConnecte = resolve;
});

/**
 * Loads the shipped world and returns control once the engine has accepted it.
 *
 * Called from `connected` and not at initialization: it's the only moment
 * when the engine is reachable (#143, #144). A failure isn't fatal, a
 * fresh world beats an empty rectangle, so we swallow the error after
 * logging it.
 */
async function loadBootWorld() {
  if (bootWorldDone || !BOOT_WORLD_URL) return;
  bootWorldDone = true;
  try {
    const resp = await fetch(BOOT_WORLD_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} sur ${BOOT_WORLD_URL}`);
    // `DecompressionStream` avoids materializing the 50 MB decompressed in
    // a single allocation: the stream is reassembled into a Blob, which
    // both transports already know how to consume (POST /checkpoint or postMessage).
    const flux = resp.body.pipeThrough(new DecompressionStream("gzip"));
    const blob = await new Response(flux).blob();
    await queueCheckpointImport(() => transport.loadCheckpoint(blob));
  } catch (e) {
    console.warn("Delivered world not loaded, keeping the fresh world:", e);
    // A `TypeError` here (a `Failed to fetch` on a request that nonetheless
    // answered 200, or an unreadable gzip stream) has an almost unique
    // cause: the server announced a `Content-Encoding` on a file that's
    // already compressed, the browser decompressed it once too many and
    // aborted. The raw error says nothing about it and costs the
    // integrator an hour, hence this panel.
    if (e instanceof TypeError) {
      console.warn(
        `Likely cause: the server serving ${BOOT_WORLD_URL} adds a ` +
          "`Content-Encoding` (often inferred from the extension). This file is " +
          "already gzipped and must be served as-is, the page decompresses it.",
      );
    }
  }
}

function connect() {
  transport.onStatus = (state, title) => {
    wsStatusEl.dataset.state = state === "connecting" ? "disconnected" : state;
    wsStatusEl.title = title;
    // A job in progress will never receive its `finished`: unfreeze the UI
    // so it doesn't stay stuck. `setBusy` is defined further down (hoisted at runtime).
    if (state === "disconnected" && busy) setBusy(false);
    // Embed (#140): nobody is going to click PLAY inside an iframe, so we
    // start playback as soon as the transport responds. Only once
    // (otherwise a reconnection would override a pause requested in the
    // meantime), and never against an explicit intent from the host (#142).
    if (state === "connected") {
      signalerConnecte();
      // The engine starts at `tick_ms = 30`, i.e. 33 h/s, while the
      // default slider shows 19.5 h/s: the label had been lying forever,
      // in every mode, because nobody was passing the position on open. In
      // embed the gap became glaring, slider at a contemplative pace,
      // simulation measured at 28 h/s (#144).
      //
      // Here rather than at initialization: it's the only point where the
      // engine is reachable. Also covers a `speed()` received from the host
      // before the connection, which set the slider but whose command was lost.
      applySpeedFromSlider(true);
    }
    if (state === "connected") {
      const premierEmbed = EMBED && !embedAutostarted;
      if (premierEmbed) embedAutostarted = true;
      // An embed starts on its own (#140). Outside of embed nothing starts on
      // its own, so only an intent from the host justifies touching the state,
      // but it then has to be replayed here: without an autostart to catch up,
      // a `play()` popped off the queue before `connect()` would be lost,
      // exactly like #143. `applyPlayState` and not `setPlaying`: the local
      // state already reflects the intent, an idempotent toggle would send nothing.
      const demarre = () => {
        if (premierEmbed || hostIntent) applyPlayState(hostIntent !== "pause");
      };
      // The shipped world must replace the fresh world BEFORE the first playback,
      // otherwise the visitor sees tick 0 scroll by for a second before being overwritten.
      if (BOOT_WORLD_URL && !bootWorldDone) loadBootWorld().finally(demarre);
      else demarre();
    }
  };

  transport.onBinary = (buffer) => {
    // Two binary frames: grid snapshots (large, frequent) and perf
    // samples (~70 B, 1/s). We defer decoding snapshots to the
    // rebuild (F2): no point decoding+inflating 120k cells on every
    // arrival when we'll only render a fraction of them, and it avoids
    // stacking 17 MB frames in the event queue when the main thread is
    // struggling. A large frame = snapshot (stored as-is); a small one =
    // perf, decoded right away (negligible cost).
    if (buffer.byteLength > PERF_FRAME_MAX_BYTES) {
      latestSnapshotBuf = buffer;
      needsRebuild = true;
      return;
    }
    dispatchMessage(decodeBinaryFrame(buffer));
  };

  transport.onJson = dispatchMessage;

  transport.connect();
}

// Buttons
// Local play/pause state: the server starts paused and doesn't rebroadcast
// this flag, so we track it client-side. Any single source of truth must
// converge with the server via playPauseToggle().
let isPlaying = false;
const btnPlayPause = document.getElementById("btn-playpause");

function updatePlayPauseButton() {
  btnPlayPause.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
}

// Latest intent expressed by the host page via `window.__hexsim` (#142),
// `null` if it never showed up. It takes priority over the embed's
// automatic startup: a host that calls `pause()` even before the
// simulation is connected is explicitly asking not to see it run,
// typically because the iframe is off-screen. Without this, autostart
// would restart the simulation right after popping the `pause()`.
let hostIntent = null;

// Brings the play/pause state to the desired value, without sending
// anything if we're already there. Don't use at connection time: see `applyPlayState` (#143).
function setPlaying(want) {
  if (want !== isPlaying) applyPlayState(want);
}

function playPauseToggle() {
  applyPlayState(!isPlaying);
}

// Sets the state AND sends it to the engine, unconditionally on the current state.
//
// Both transports silently drop commands issued before the connection
// (`worker?.postMessage` with a worker still `null` on the WASM side,
// the `readyState === OPEN` guard on the WebSocket side). A `play()`
// popped off the queue before `connect()` therefore leaves `isPlaying`
// at `true` facing an engine still paused, and an idempotent toggle
// locks in this divergence: nothing ever sends the command again. This
// is what was freezing an embed whose host called `play()` on the
// iframe's `load`, reporting `playing: true` while nothing was advancing (#143).
function applyPlayState(want) {
  isPlaying = want;
  send({ cmd: isPlaying ? "play" : "pause" });
  updatePlayPauseButton();
  emitEmbedTick();
  // Immediate label + light refresh: switching to pause must make the
  // hour reappear AND exit the noon-freeze even if the speed was
  // > 24 h/s. Same in reverse for play→fast-play.
  if (lastState) {
    if (tickCountEl) {
      const showHour = !isPlaying || currentHps < 24;
      tickCountEl.textContent = tickToDate(
        lastState.tick,
        showHour ? lastState.hour_tick : null,
      );
    }
    updateSceneLighting(lastState.hour_tick);
  }
}

// --- "Busy" state during a long computation (step/step_hour multi-ticks) ---
// The server splits the job into batches and pushes `progress` messages
// (date + progress); we show a bar and FREEZE transport commands.
// Graying out at the source prevents stacking: without this, 30 clicks on
// +Y would send 30 jobs processed in series → ~10 min of blocking. Disabled
// button = ignored click, so only one job at a time.
const playbackFooter = document.getElementById("side-playback");
const progressBox = document.getElementById("pb-progress");
const progressFill = document.getElementById("pb-progress-fill");
const progressPct = document.getElementById("pb-progress-pct");
const progressLabel = document.getElementById("pb-progress-label");
const btnStepHour = document.getElementById("btn-step-hour");
const btnStep = document.getElementById("btn-step");
const btnMonth = document.getElementById("btn-month");
const btnYear = document.getElementById("btn-year");
const seedInput = document.getElementById("seed-input");
const btnRandomSeed = document.getElementById("btn-random-seed");
const btnReset = document.getElementById("btn-reset");

// Controls frozen for the duration of a job. View toggles and param sliders
// stay active (purely client-side / harmless set_param).
const BUSY_CONTROLS = [
  btnPlayPause, btnStepHour, btnStep, btnMonth, btnYear,
  seedInput, btnRandomSeed, btnReset,
];

// `busy` (logical) = a job is running: switches to true AS SOON as the
// click happens and freezes intake of new jobs (anti-stacking, instant).
// `busyVisible` (visual) = the loader is actually shown: it's only
// activated if the computation exceeds REVEAL_DELAY, so an instant step
// (small map) doesn't make the UI flicker. A one-day step on a large map
// (a few seconds) exceeds the threshold → loader visible.
const BUSY_REVEAL_MS = 200;
let busy = false;
let busyVisible = false;
let busyRevealTimer = null;
let busyWatchdog = null;

function applyBusyVisuals(on, label) {
  busyVisible = on;
  playbackFooter.classList.toggle("busy", on);
  progressBox.hidden = !on;
  for (const el of BUSY_CONTROLS) el.disabled = on;
  if (on) {
    progressLabel.textContent = label || "Computing…";
    // Starts as an indeterminate shimmer: until we have a usable multi-batch
    // milestone, a bar frozen at 0% would look like a freeze.
    progressFill.classList.add("indeterminate");
    progressFill.style.width = "";
    progressPct.textContent = "…";
  } else {
    progressFill.classList.remove("indeterminate");
    progressFill.style.width = "0%";
  }
}

function setBusy(on, label) {
  if (on) {
    if (busy) return;
    busy = true;
    // Shows the loader only if the job takes time: an instant step
    // finishes before REVEAL_DELAY and never shows anything.
    clearTimeout(busyRevealTimer);
    busyRevealTimer = setTimeout(() => applyBusyVisuals(true, label), BUSY_REVEAL_MS);
    // Safety net: if `finished` gets lost (server killed mid-job), we don't
    // stay frozen forever.
    clearTimeout(busyWatchdog);
    busyWatchdog = setTimeout(() => setBusy(false), 180000);
  } else {
    busy = false;
    clearTimeout(busyRevealTimer);
    busyRevealTimer = null;
    clearTimeout(busyWatchdog);
    busyWatchdog = null;
    if (busyVisible) applyBusyVisuals(false);
  }
}

function updateProgress(msg) {
  if (!busy) return; // progress from a job we didn't initiate (another client)
  if (msg.finished) {
    setBusy(false);
    return;
  }
  if (!busyVisible) return; // job still under the display threshold
  // Determinate bar only for multi-batch jobs (month/year); a single-tick
  // job has no intermediate milestone → we keep the shimmer.
  if (msg.total > 1) {
    const frac = Math.min(1, msg.done / msg.total);
    progressFill.classList.remove("indeterminate");
    progressFill.style.width = `${(frac * 100).toFixed(1)}%`;
    progressPct.textContent = `${Math.round(frac * 100)} %`;
  }
  if (typeof msg.tick === "number") {
    progressLabel.textContent = `→ ${tickToDate(msg.tick)}`;
  } else if (lastState) {
    // WASM mode: the Worker doesn't send `tick` along with progress. The
    // last rendered snapshot is authoritative, one milestone behind.
    progressLabel.textContent = `→ ${tickToDate(lastState.tick)}`;
  }
}

// Starts a computation job while freezing input. No-op if a job is already running.
function startJob(cmd, label) {
  if (busy) return;
  setBusy(true, label);
  send(cmd);
}

btnPlayPause.addEventListener("click", playPauseToggle);
btnStepHour.addEventListener("click", () => startJob({ cmd: "step_hour" }, "+1 hour…"));
btnStep.addEventListener("click", () => startJob({ cmd: "step" }, "+1 day…"));
btnMonth.addEventListener("click", () => startJob({ cmd: "step", n: 30 }, "+1 month…"));
btnYear.addEventListener("click", () => startJob({ cmd: "step", n: 365 }, "+1 year…"));
btnRandomSeed.addEventListener("click", () => {
  seedInput.value = Math.floor(Math.random() * 999999);
});
btnReset.addEventListener("click", () => {
  send({ cmd: "reset", seed: parseInt(seedInput.value, 10) || 42 });
});

// --- Checkpoint save/load (binary file) ---
// The transport knows where the state lives: on the server (HTTP `/checkpoint`)
// or in the WASM module (`export_checkpoint`/`import_checkpoint`). In both
// cases, a successful import produces a fresh snapshot that refreshes the scene.
document.getElementById("btn-save").addEventListener("click", () => {
  transport.saveCheckpoint();
});

const fileLoad = document.getElementById("file-load");
document.getElementById("btn-load").addEventListener("click", () => fileLoad.click());
fileLoad.addEventListener("change", async () => {
  const file = fileLoad.files && fileLoad.files[0];
  if (!file) return;
  try {
    await transport.loadCheckpoint(file);
  } catch (e) {
    console.error("Import checkpoint refused:", e);
    alert("Import refused: " + (e?.message ?? e));
  } finally {
    fileLoad.value = ""; // allows reloading the same file
  }
});

// forceRebuild: bypasses the adaptive throttle for an immediate rebuild
// (fixing `lastRebuildTime = 0` was no longer enough, the adaptive interval
// can exceed the time elapsed since boot).
viewModeEl.addEventListener("change", () => {
  needsRebuild = true;
  forceRebuild = true;
  emitEmbedTick();
});

document.getElementById("toggle-wind").addEventListener("change", (e) => {
  showWind = e.target.checked;
  needsRebuild = true;
  forceRebuild = true;
  emitEmbedTick();
});

const toggleCloudsEl = document.getElementById("toggle-clouds");
toggleCloudsEl.checked = showClouds;
toggleCloudsEl.addEventListener("change", (e) => {
  showClouds = e.target.checked;
  needsRebuild = true;
  forceRebuild = true;
  emitEmbedTick();
});

document.getElementById("toggle-temperature").addEventListener("change", (e) => {
  showTemperature = e.target.checked;
  needsRebuild = true;
  forceRebuild = true;
  emitEmbedTick();
});

document.getElementById("toggle-precipitation").addEventListener("change", (e) => {
  showPrecipitation = e.target.checked;
  needsRebuild = true;
  forceRebuild = true;
  emitEmbedTick();
});

const speedSlider = document.getElementById("speed-slider");
const speedLabel = document.getElementById("speed-label");

// Bare mode: the embed opens at a contemplative pace, position 0 = `HPS_MIN`,
// i.e. 1 h/s, 24 s for a full diurnal cycle. Full mode's speed (~19 h/s)
// scrolls through a year in eight minutes: readable when you're driving it,
// agitated in the corner of a page you're reading.
//
// We move the slider rather than calling `speed` directly: the slider also
// drives the label, the hour-display toggle, and the light rendering.
// Bypassing the chain would desynchronize them.
if (EMBED) speedSlider.value = "0";

// Tracks the current speed to decide whether to display the hour in the
// tick label. Beyond 24 h/s (= 1 j/s) the hour scrolls by too fast to be
// readable, so we switch to date-only display.
let currentHps = 1;

function applySpeedFromSlider(sendToServer) {
  const pos = parseInt(speedSlider.value, 10);
  const hps = hpsFromPos(pos);
  currentHps = hps;
  const ms = msFromHps(hps);
  speedLabel.textContent = formatHps(hps);
  if (sendToServer) send({ cmd: "speed", value: ms });
  // Forces a refresh of the tick label + light if we have a last state
  // at hand (changing the speed must toggle the visual noon-freeze and
  // the hour display without waiting for the next snapshot).
  if (lastState) {
    if (tickCountEl) {
      const showHour = !isPlaying || currentHps < 24;
      tickCountEl.textContent = tickToDate(
        lastState.tick,
        showHour ? lastState.hour_tick : null,
      );
    }
    updateSceneLighting(lastState.hour_tick);
  }
}

speedSlider.addEventListener("input", () => applySpeedFromSlider(true));
// Display only: at this point in the file the transport isn't connected, a
// `send` would go nowhere (#144). Transmission to the engine happens in
// `onStatus("connected")`.
applySpeedFromSlider(false);

// Keyboard navigation: Space = play/pause. Ignored if focus is on an input
// field, so as not to interfere with editing the seed or any params.
window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  e.preventDefault();
  playPauseToggle();
});

/**
 * Applies a map background requested by the host.
 *
 * We set the `<select>` and let its `change` do the work, rather than
 * replaying the rebuild here: a single path, so nothing to desynchronize
 * the day a view's rendering needs one more step.
 *
 * An unknown identifier is **loudly rejected**. Silence is the trap
 * from #147 (`MAX_STEP_DAYS` clamped without saying so): here the host
 * sees the rejection twice, in the console and in the `view` of the next
 * `tick`, which won't have moved.
 */
function applyHostView(id) {
  const connu = Array.from(viewModeEl.options).some((o) => o.value === id);
  if (!connu) {
    console.warn(
      `__hexsim.view: unknown map background "${id}", ignored. ` +
        `Available: ${Array.from(viewModeEl.options, (o) => o.value).join(", ")}`,
    );
    return;
  }
  viewModeEl.value = id;
  viewModeEl.dispatchEvent(new Event("change"));
}

/** Same for an overlay. Comparison in JS, not a CSS selector: `id`
 *  comes from the host, and a string with quotes would break `querySelector`. */
function applyHostLayer(id, on) {
  const boites = Array.from(document.querySelectorAll("[data-layer]"));
  const boite = boites.find((el) => el.dataset.layer === id);
  if (!boite) {
    console.warn(
      `__hexsim.layer: unknown overlay "${id}", ignored. ` +
        `Available: ${boites.map((el) => el.dataset.layer).join(", ")}`,
    );
    return;
  }
  if (boite.checked === on) return; // already in this state, no unnecessary rebuild
  boite.checked = on;
  boite.dispatchEvent(new Event("change"));
}

// --- Cell inspector (hover tooltip + click-to-copy) -----------------------
//
// Declared here, above `_attach`, and not in the tooltip block 80 lines
// further down: `_attach` empties the boot queue **synchronously**, so an
// `inspect()` called by the host before the module loads executes right
// at this line. Whatever it touches must already be initialized, otherwise
// a dead zone, the third encounter with this trap in this repo.
//
// Off by default in embed (#150). Not a comfort setting: the click was
// writing into the host page's clipboard, on a scene where clicking is
// primarily used to rotate the camera. Flagged during the first playtest.
//
// A flag short-circuited at the top of both handlers rather than a
// conditional `addEventListener`: `inspect(true)` must be able to turn it
// back on, so the listeners have to exist regardless.
//
// CSS couldn't do this job. `body[data-chrome="none"] #tooltip
// { display: none }` existed since #142 and never hid anything: the
// handler writes `style.display = "block"` inline, which beats the
// stylesheet. A UI driven from JS doesn't turn off via CSS, the rule was
// removed rather than left there suggesting otherwise.
let inspectorOn = !EMBED;
const tooltipEl = document.getElementById("tooltip");
let tooltipData = null;

function hideTooltip() {
  tooltipEl.style.display = "none";
  tooltipData = null;
}

/**
 * `inspect(on)` from the embed contract.
 *
 * Turning it off also clears what's on screen: a tooltip shown at the
 * moment of the toggle would otherwise stay stuck there, with no more
 * `mousemove` coming to update it. And `tooltipData` goes with it, so
 * that turning it back on doesn't resume on the cell hovered before it was turned off.
 */
function applyHostInspect(on) {
  inspectorOn = on !== false;
  if (!inspectorOn) hideTooltip();
}

// Wires the `window.__hexsim` bootstrap set up by index.html to the simulation.
// Every call received before this line has been queued and now goes out.
// `playPauseToggle` rather than a direct `send`: it holds `isPlaying` and
// the button label, which a host must not be able to desynchronize.
//
// What's returned here becomes the promise's value on the host side: `undefined`
// for silent commands, the transport's promise for `export`/`load`.
window.__hexsim?._attach((name, arg) => {
  switch (name) {
    case "speed":
      // Goes through the slider rather than calling `send`: it also drives the
      // label, the hour-display toggle, and the light freeze.
      speedSlider.value = String(posFromHps(arg));
      applySpeedFromSlider(true);
      return undefined;
    case "view":
      return applyHostView(arg);
    case "layer":
      return applyHostLayer(arg.id, arg.on);
    case "inspect":
      return applyHostInspect(arg);
    // The world's state lives in the transport (HTTP `/checkpoint` or
    // WASM's `export_checkpoint`): nothing to translate here, we let it through.
    case "export":
      return moteurJoignable.then(() => transport.exportCheckpoint());
    case "load":
      // A host restoring its state has no use for the shipped world: we
      // cancel the one that hasn't gone out yet, and queue behind the one
      // already in flight. The host has the last word either way.
      bootWorldDone = true;
      return queueCheckpointImport(() =>
        moteurJoignable.then(() => transport.loadCheckpoint(arg)),
      );
    default:
      hostIntent = name;
      setPlaying(name === "play");
      return undefined;
  }
});

connect();

// --- Tooltip (hover cell inspector) ---
// `tooltipEl`, `tooltipData` and the `inspectorOn` flag are declared further
// up, with `applyHostInspect`, see the why over there.
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function worldToHex(x, z) {
  const q = (x * SQRT3 / 3 - z / 3) / HEX_SIZE;
  const r = (z * 2 / 3) / HEX_SIZE;
  // Round to cube coordinates
  let rx = Math.round(q), ry = Math.round(-q - r), rz = Math.round(r);
  const dx = Math.abs(rx - q), dy = Math.abs(ry - (-q - r)), dz = Math.abs(rz - r);
  if (dx > dy && dx > dz) rx = -ry - rz;
  else if (dy > dz) ry = -rx - rz;
  else rz = -rx - ry;
  return { q: rx, r: rz };
}

renderer.domElement.addEventListener("mousemove", (e) => {
  // Inspector off: we do nothing at all, not even the raycast.
  if (!inspectorOn) return;
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  if (!lastState || !terrainMesh) { hideTooltip(); return; }

  // F8: analytic picking. `intersectObject(terrainMesh)` was testing the
  // terrain's ~2M triangles on EVERY mousemove (no BVH), tens of ms at
  // r200. Instead we intersect the camera ray with a horizontal plane then
  // convert to hex via `worldToHex` (O(1), already analytic). A refinement
  // at the targeted cell's real altitude corrects the parallax of tall
  // columns in oblique view.
  raycaster.setFromCamera(mouse, camera);
  const ray = raycaster.ray;
  const hexAtHeight = (h) => {
    if (Math.abs(ray.direction.y) < 1e-6) return null;
    const t = (h - ray.origin.y) / ray.direction.y;
    if (t < 0) return null;
    const wx = ray.origin.x + t * ray.direction.x;
    const wz = ray.origin.z + t * ray.direction.z;
    return { hex: worldToHex(wx, wz) };
  };
  let g = hexAtHeight(controls.target.y); // first pass at the camera target's level
  if (g) {
    const i0 = cellIndex[`${g.hex.q},${g.hex.r}`];
    if (i0 !== undefined) {
      const g2 = hexAtHeight(lastState.cells[i0].elevation * ELEVATION_SCALE);
      if (g2) g = g2;
    }
  }
  if (!g) { hideTooltip(); return; }
  const hex = g.hex;
  const key = `${hex.q},${hex.r}`;
  const idx = cellIndex[key];
  if (idx === undefined) { hideTooltip(); return; }

  const c = lastState.cells[idx];
  const cap = (c.permeability * 100).toFixed(1);
  const sat = c.permeability > 0 ? ((c.groundwater / (c.permeability * 100)) * 100).toFixed(0) : "—";
  const fvMag = Math.hypot(c.flow_vec_x || 0, c.flow_vec_y || 0);
  const fvLabel = fvMag > 1e-4
    ? `${fvMag.toFixed(3)} → ${windLabel(c.flow_vec_x, c.flow_vec_y).split(" → ")[1] || ""}`
    : "—";
  const lines = [
    `(${c.q}, ${c.r})`,
    `elev     ${c.elevation.toFixed(1)} m`,
    `temp     ${c.temperature.toFixed(1)} °C`,
    `water    ${c.water_level.toFixed(2)} mm / cap ${(c.water_capacity || 1).toFixed(2)} mm`,
    `hum surf ${(c.humidity_surface ?? 0).toFixed(2)} mm`,
    `hum up   ${(c.humidity_upper ?? 0).toFixed(2)} mm PW`,
    `cloud    ${(c.cloud_water ?? 0).toFixed(2)} mm  (drives the render)`,
    `rain/d   ${(c.rain_amount ?? 0).toFixed(2)} mm   snow/d ${(c.snow_amount ?? 0).toFixed(2)} mm`,
    `gwater   ${c.groundwater.toFixed(2)} mm / ${cap} mm (${sat}%)`,
    `perm     ${c.permeability.toFixed(2)}`,
    `cover    ${c.is_open_water ? "water" : (c.dominant_species ?? "bare")}   veg ${(c.vegetation ?? 0).toFixed(2)}`,
    `outflow  ${(c.outflow_flux || 0).toFixed(3)}`,
    `flow dir ${fvLabel}`,
    `wind     ${windLabel(c.wind_x, c.wind_y)}`,
  ];
  tooltipData = lines.join("\n");
  tooltipEl.textContent = tooltipData;
  tooltipEl.style.display = "block";
  tooltipEl.style.left = (e.clientX + 16) + "px";
  tooltipEl.style.top = (e.clientY + 16) + "px";
});

renderer.domElement.addEventListener("click", () => {
  // The host page's clipboard belongs to it. Short-circuit at the top, not
  // a test on the tooltip's visibility: the two get desynchronized.
  if (inspectorOn && tooltipData) {
    navigator.clipboard.writeText(tooltipData);
    tooltipEl.style.outline = "1px solid #8f8";
    setTimeout(() => { tooltipEl.style.outline = ""; }, 200);
  }
});

// --- Params panel (sidebar params region, always visible) ---
const paramsContent = document.getElementById("params-content");

// Server sliders (data-key) → WS command
for (const input of paramsContent.querySelectorAll("input[data-key]")) {
  const span = input.nextElementSibling;
  input.addEventListener("input", () => {
    const value = parseFloat(input.value);
    span.textContent = value;
    send({ cmd: "set_param", key: input.dataset.key, value });
  });
}

// Server → sliders sync (#2): response to the connect's {cmd:"params"}, and
// broadcast received after each successful set_param (another client,
// `just param`, MCP...). The server is the source of truth, the UI only reflects it.
function applyServerParams(msg) {
  for (const input of paramsContent.querySelectorAll("input[data-key]")) {
    const [module, field] = input.dataset.key.split(".");
    const raw = msg[module]?.[field];
    if (raw === undefined || raw === null) continue;
    const value = typeof raw === "boolean" ? (raw ? 1 : 0) : raw;
    if (!Number.isFinite(value)) continue;
    // Slider currently being manipulated: don't fight the user (the
    // broadcast is an echo of their own set_param).
    if (input === document.activeElement) continue;
    // Real value outside the HTML range (core default that moved,
    // out-of-bounds CLI set_param...): extend the range rather than show a
    // slider that's capped and lying, cf. wind.humidity_advection_rate 3.0 on a max=1.
    if (value < parseFloat(input.min)) input.min = value;
    if (value > parseFloat(input.max)) input.max = value;
    input.value = value;
    const span = input.nextElementSibling;
    if (span) span.textContent = value;
  }
  // The cloud rendering window follows the core's auto-conversion threshold
  // (cf. CLOUD_WINDOW_* block at the top), both connect AND hot-tuning (`just param`).
  const crit = msg.atmosphere?.precip_crit_mm;
  if (Number.isFinite(crit) && applyPrecipCritMm(crit)) {
    // Reapplies the densities under the new window (a fade in progress
    // would keep the old one) and repaints even while paused.
    if (lastState?.cells) captureClouds(lastState.cells, false);
    needsRebuild = true;
  }
}

// Local sliders (data-local) → JS variables
for (const input of paramsContent.querySelectorAll("input[data-local]")) {
  const span = input.nextElementSibling;
  input.addEventListener("input", () => {
    const value = parseFloat(input.value);
    span.textContent = value;
    // water_threshold removed, no more separate water mesh
    if (input.dataset.local === "river_threshold") RIVER_THRESHOLD = value;
    if (input.dataset.local === "ambient_gain") {
      AMBIENT_GAIN = value;
      // Light only: reapplies the lighting (animate() renders every frame),
      // no geometry rebuild.
      if (lastState) updateSceneLighting(lastState.hour_tick ?? lastState.tick * 24);
      return;
    }
    needsRebuild = true;
    forceRebuild = true;
  });
}

// --- Capture hook (__hexcam) ----------------------------------------
//
// Camera control surface exposed on `window` for headless driving
// (Playwright). Lets an external tool frame the scene deterministically
// (zoom, azimuth, tilt, recentering), then force a render before capture.
// See `scripts/shot/shot.mjs`.
//
// Angular convention (THREE.Spherical):
//   - radius     : camera → target distance, in world units (1 hex ≈ 1.0)
//   - polarDeg   : tilt from vertical. 0 = top-down view,
//                  90 = grazing view at the horizon.
//   - azimuthDeg : rotation around the vertical axis. 0 = view from the south.
{
  const _sph = new THREE.Spherical();
  const _vec = new THREE.Vector3();
  const RAD = Math.PI / 180;
  const DEG = 180 / Math.PI;

  function currentView() {
    _vec.copy(camera.position).sub(controls.target);
    _sph.setFromVector3(_vec);
    return {
      radius: _sph.radius,
      polarDeg: _sph.phi * DEG,
      azimuthDeg: _sph.theta * DEG,
      target: { x: controls.target.x, y: controls.target.y, z: controls.target.z },
    };
  }

  window.__hexcam = {
    // Ready to capture: terrain built AND at least one snapshot received.
    ready() {
      return !!(terrainMesh && lastState);
    },

    // Metadata of the last state received, useful for naming/annotating a capture.
    state() {
      if (!lastState) return null;
      return {
        tick: lastState.tick ?? null,
        hourTick: lastState.hour_tick ?? null,
        cellCount: lastState.cell_count ?? (lastState.cells?.length ?? null),
      };
    },

    // Histogram of the dominant cover on the map (from the last
    // snapshot): cell count per species + average vegetation cover.
    // Diagnostic tool to judge landscape diversity without recomputation
    // (consumes `dominant_species`/`vegetation` already provided by the core).
    cover() {
      if (!lastState || !lastState.cells) return null;
      const hist = {};
      let vegSum = 0;
      let land = 0;
      let ageSum = 0;
      let ageWeight = 0;
      let maxAge = 0;
      let burning = 0;
      for (const c of lastState.cells) {
        if (c.is_open_water) {
          hist.water = (hist.water ?? 0) + 1;
          continue;
        }
        const sp = c.dominant_species ?? "bare";
        hist[sp] = (hist[sp] ?? 0) + 1;
        const v = c.vegetation ?? 0;
        vegSum += v;
        land += 1;
        // Age weighted by cover (the age of a bare hex is meaningless).
        ageSum += (c.stand_age ?? 0) * v;
        ageWeight += v;
        if ((c.stand_age ?? 0) > maxAge) maxAge = c.stand_age;
        if ((c.fire_intensity ?? 0) > 1e-3) burning += 1;
      }
      const pct = {};
      for (const [k, v] of Object.entries(hist)) {
        pct[k] = +((100 * v) / lastState.cells.length).toFixed(1);
      }
      return {
        tick: lastState.tick ?? null,
        cells: lastState.cells.length,
        land,
        meanVeg: +(vegSum / Math.max(1, land)).toFixed(3),
        meanAge: +(ageSum / Math.max(1e-6, ageWeight)).toFixed(2),
        maxAge: +maxAge.toFixed(1),
        burningCells: burning,
        count: hist,
        pct,
        mix: this.mix(),
      };
    },

    // Species mix *inside* a hex (from `species_mix`, exported by the
    // core). For each vegetated cell: share of the dominant species
    // (max/sum) and effective number of species (inverse Simpson,
    // Σpᵢ²). Aggregates to answer "mono vs mixed". Cover threshold to
    // ignore nearly-bare cells (noise).
    mix(minVeg = 0.05) {
      if (!lastState || !lastState.cells || !lastState.species_order) return null;
      let n = 0;
      let domShareSum = 0;
      let effSpeciesSum = 0;
      let mono = 0; // dominant > 90%
      let mixed = 0; // dominant < 70%
      const compo = {}; // average community composition (share per species)
      for (const o of lastState.species_order) compo[o] = 0;
      for (const c of lastState.cells) {
        if (c.is_open_water || !Array.isArray(c.species_mix)) continue;
        const m = c.species_mix;
        const total = m.reduce((a, b) => a + b, 0);
        if (total < minVeg) continue;
        n += 1;
        const max = Math.max(...m);
        const domShare = max / total;
        domShareSum += domShare;
        const simpson = m.reduce((a, b) => a + (b / total) ** 2, 0);
        effSpeciesSum += 1 / simpson;
        if (domShare > 0.9) mono += 1;
        else if (domShare < 0.7) mixed += 1;
        lastState.species_order.forEach((sp, i) => {
          compo[sp] += m[i] / total;
        });
      }
      if (n === 0) return { vegetatedCells: 0 };
      const compoPct = {};
      for (const [k, v] of Object.entries(compo)) {
        compoPct[k] = +((100 * v) / n).toFixed(1);
      }
      return {
        vegetatedCells: n,
        meanDominantShare: +(domShareSum / n).toFixed(3),
        meanEffSpecies: +(effSpeciesSum / n).toFixed(2),
        monoPct: +((100 * mono) / n).toFixed(1), // % hexes > 90% one species
        mixedPct: +((100 * mixed) / n).toFixed(1), // % hexes < 70% dominant
        communityPct: compoPct,
      };
    },

    // Reads the current framing (radius / polarDeg / azimuthDeg / target).
    view: currentView,

    // Positions the camera. All fields are optional and merged with the
    // current framing; `target` can be {x,z} (y kept).
    setView(opts = {}) {
      const cur = currentView();
      const radius = Math.max(0.1, opts.radius ?? cur.radius);
      const polarDeg = opts.polarDeg ?? cur.polarDeg;
      const azimuthDeg = opts.azimuthDeg ?? cur.azimuthDeg;
      if (opts.target) {
        controls.target.set(
          opts.target.x ?? controls.target.x,
          opts.target.y ?? controls.target.y,
          opts.target.z ?? controls.target.z,
        );
      }
      // phi (polar) bounded away from the poles: OrbitControls becomes unstable at 0°.
      const phi = Math.min(Math.max(polarDeg * RAD, 0.001), Math.PI - 0.001);
      _sph.set(radius, phi, azimuthDeg * RAD);
      _vec.setFromSpherical(_sph);
      camera.position.copy(controls.target).add(_vec);
      controls.update();
      this.render();
      return currentView();
    },

    // Multiplies the distance: <1 = zoom in, >1 = zoom out.
    zoom(factor) {
      return this.setView({ radius: currentView().radius * factor });
    },

    // Recenters the camera on the whole terrain and adjusts distance to frame it.
    fitAll(margin = 1.25) {
      if (!terrainMesh) return currentView();
      const box = new THREE.Box3().setFromObject(terrainMesh);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.z);
      const fov = camera.fov * RAD;
      const radius = (maxDim * margin) / (2 * Math.tan(fov / 2));
      return this.setView({
        radius,
        target: { x: center.x, y: center.y, z: center.z },
      });
    },

    // Forces a synchronous render (clean capture with preserveDrawingBuffer).
    render() {
      renderer.render(scene, camera);
    },

    // Rendering counters for the last frame (triangles, draw calls, VRAM
    // buffers). Objective numbers to validate a front-end optimization
    // against, rather than an impression of smoothness.
    info() {
      return {
        render: { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles },
        memory: { geometries: renderer.info.memory.geometries, textures: renderer.info.memory.textures },
      };
    },

    // Shows/hides both sidebars for a "bare map" capture.
    setChromeVisible(visible) {
      for (const id of ["sidebar", "right-sidebar"]) {
        const el = document.getElementById(id);
        if (el) el.style.display = visible ? "" : "none";
      }
      window.dispatchEvent(new Event("resize"));
    },
  };
}

// Bare mode: inside an iframe, nobody is going to manipulate OrbitControls
// to find the map. We frame the overall view as soon as the terrain exists,
// via the same path as the capture tool (`scripts/shot` calls `fitAll`
// from the outside). Only once: after that the view belongs to the visitor.
if (EMBED) {
  // `fitAll` measures the terrain's bounding box, which is built tile by
  // tile: framing just once on the first `ready()` targets a still-partial
  // map and gives a ground-level view (measured: distance 32 instead of
  // 197). Waiting for two consecutive measurements to be equal isn't
  // enough either, the box stays stable between two tile arrivals, and
  // the framing then locks onto a fragment (seen: 1 launch out of 3).
  //
  // So we reframe at a fixed interval during the first two seconds,
  // then the view belongs to the visitor.
  const framing = setInterval(() => {
    if (window.__hexcam?.ready()) window.__hexcam.fitAll();
  }, 200);
  setTimeout(() => clearInterval(framing), 2000);
}
