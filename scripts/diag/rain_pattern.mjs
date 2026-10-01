// Rain pattern probe on a LIVE hexsim server, read in data rather than in
// pixels: what the precipitation overlay would paint per hour, how sticky the
// rained-on cells are, how many islets, daily totals. Written on 2026-09-06
// for #63 ("it always rains somewhere, in islets that never move"), it found
// the wire exporting the daily accumulator instead of the hour's flux.
//
// It RESETS the target world (reset seed, warmup, then 48 hourly steps), so
// never point it at the server someone is watching: run a second instance on
// another port first, e.g. from `simulation/`:
//     HEXSIM_PORT=8356 ./target/release/hexsim-cli
// then, from the repo root:
//     node scripts/diag/rain_pattern.mjs            (or `just rain-pattern`)
// Env: WS_URL (ws://localhost:8356/ws), SEED (42), WARMUP_DAYS (545 = July 1
// of year 2), HOURS (48), PAINT (0.1 mm/h, the overlay threshold used for the
// persistence / Jaccard / islet metrics; the per-hour table always lists the
// fixed thresholds 1e-4 .. 1 mm/h).
//
// Wire pitfalls handled: during a `step n` the server broadcasts
// intermediate snapshots, and after it a small binary perf message
// (`type, cpu_percent, rss_mb, tick_us, tick`) with no `cells`; frames are
// filtered on the presence of `cells` and on a strictly increasing
// `hour_tick`. The `diagnostics` command is answered after the running job,
// so it doubles as a barrier. Decoder: the front's own @msgpack/msgpack
// (`just front-setup` installs it).
import { decode } from "../../frontend/node_modules/@msgpack/msgpack/dist.esm/index.mjs";

const URL = process.env.WS_URL ?? "ws://localhost:8356/ws";
const SEED = Number(process.env.SEED ?? 42);
const WARMUP_DAYS = Number(process.env.WARMUP_DAYS ?? 545); // day 545 = Jul 1 of year 2
const HOURS = Number(process.env.HOURS ?? 48);
const THRESHOLDS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]; // mm per hour-tick
const PAINT = Number(process.env.PAINT ?? 0.1); // front PRECIP_MIN (0.1 mm/h since L1)
const PARAMS = JSON.parse(process.env.PARAMS ?? "{}"); // hot params applied after reset, e.g. {"atmosphere.regime_enabled":1}
const ROW_EVERY = Number(process.env.ROW_EVERY ?? 6); // per-hour table stride
const CLOUD_MIN = Number(process.env.CLOUD_MIN ?? 0.12); // mm of cloud_water, the front's cloud window floor

const ws = new WebSocket(URL);
ws.binaryType = "arraybuffer";
const queue = [];
const waiters = [];
ws.onmessage = (ev) => {
  const item = ev.data instanceof ArrayBuffer ? { bin: new Uint8Array(ev.data) } : { txt: ev.data };
  if (waiters.length) waiters.shift()(item); else queue.push(item);
};
const next = () => (queue.length ? Promise.resolve(queue.shift()) : new Promise((r) => waiters.push(r)));
const send = (o) => ws.send(JSON.stringify(o));
async function barrier() {
  send({ cmd: "diagnostics" });
  for (;;) {
    const m = await next();
    if (m.txt) { try { const j = JSON.parse(m.txt); if (j.type === "diagnostics") return j; } catch {} }
  }
}
async function frame(minHourTick) {
  for (;;) {
    const m = await next(); if (!m.bin) continue;
    const f = decode(m.bin);
    if (!f || !Array.isArray(f.cells)) { console.error("skip non-snapshot binary: " + (f && typeof f === "object" ? Object.keys(f).join(",") : typeof f)); continue; }
    if (minHourTick !== undefined && !(f.hour_tick > minHourTick)) { console.error(`skip stale frame hour_tick=${f.hour_tick}`); continue; }
    return f;
  }
}
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

const t0 = Date.now();
send({ cmd: "pause" });
send({ cmd: "reset", seed: SEED });
await barrier();
for (const [key, value] of Object.entries(PARAMS)) send({ cmd: "set_param", key, value: Number(value) });
if (Object.keys(PARAMS).length) { await barrier(); console.error(`params applied: ${JSON.stringify(PARAMS)}`); }
send({ cmd: "step", n: WARMUP_DAYS });
await barrier();
// drain any intermediate frames broadcast during the warmup
while (queue.length) queue.shift();
console.error(`warmup ${WARMUP_DAYS} d done in ${((Date.now() - t0) / 1000).toFixed(0)} s`);

let fields = null, idx = null, coords = null, elev = null, N = 0;
let lastHourTick = WARMUP_DAYS * 24;
const hourly = []; // per hour: Float32Array total (rain+snow), Float32Array rain
const perHourStats = [];
for (let h = 0; h < HOURS; h++) {
  send({ cmd: "step_hour", n: 1 });
  const f = await frame(lastHourTick);
  lastHourTick = f.hour_tick;
  await barrier();
  while (queue.length) queue.shift();
  if (!fields) {
    fields = f.cell_fields; idx = Object.fromEntries(fields.map((n, i) => [n, i]));
    N = f.cells.length;
    coords = f.cells.map((c) => [c[idx.q], c[idx.r]]);
    elev = Float32Array.from(f.cells, (c) => c[idx.elevation]);
  }
  const rain = Float32Array.from(f.cells, (c) => c[idx.rain_amount]);
  const snow = Float32Array.from(f.cells, (c) => c[idx.snow_amount]);
  const total = rain.map((v, i) => v + snow[i]);
  const temp = f.cells.reduce((s, c) => s + c[idx.temperature], 0) / N;
  const counts = THRESHOLDS.map((t) => total.reduce((s, v) => s + (v > t ? 1 : 0), 0));
  const rainOnly = rain.reduce((s, v) => s + (v > PAINT ? 1 : 0), 0);
  const snowOnly = snow.reduce((s, v) => s + (v > PAINT ? 1 : 0), 0);
  const painted = total.filter((v) => v > PAINT).sort();
  const q = (p) => (painted.length ? painted[Math.floor(p * (painted.length - 1))] : 0);
  const paintedN = total.reduce((s, v) => s + (v > PAINT ? 1 : 0), 0);
  // cloud cover as the front renders it: cloud_water above the cloud window floor
  // (CLOUD_WINDOW_MIN_RATIO 0.8 x precip_crit_mm 0.15 = 0.12 mm)
  const cloudN = f.cells.reduce((s, c) => s + (c[idx.cloud_water] > CLOUD_MIN ? 1 : 0), 0);
  perHourStats.push({ h, tick: f.tick, hour_tick: f.hour_tick, temp, counts, rainOnly, snowOnly, paintedN, cloudN, med: q(0.5), p90: q(0.9), max: q(1), wet: f.weather_regime_wet, sky: f.total_sky_water });
  hourly.push({ total, rain });
}
console.error(`sampled ${HOURS} h in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
ws.close();

// ---- analysis ----
const pct = (n) => ((100 * n) / N).toFixed(1).padStart(5) + " %";
console.log(`\n# world: ${N} cells, seed ${SEED}, from day ${perHourStats[0].tick} (mean T ${perHourStats[0].temp.toFixed(1)} °C)`);

console.log(`\n## per-hour map fraction above threshold (persistence/Jaccard/islets below use PAINT=${PAINT} mm/h)`);
const hasRegime = perHourStats.some((s) => s.wet !== undefined);
console.log(`hour | >1e-4 | >1e-3 | >1e-2 | >1e-1 | >1 mm/h | rain>${PAINT} | snow>${PAINT} | median | p90 | max (mm/h among painted) | cloud>${CLOUD_MIN}${hasRegime ? " | regime | sky (cell-mm total)" : ""}`);
for (const s of perHourStats) {
  if (s.h % ROW_EVERY !== 0 && s.h !== HOURS - 1) continue;
  const reg = hasRegime ? ` | ${s.wet ? "WET" : "dry"} | ${(s.sky ?? 0).toFixed(0)}` : "";
  console.log(`${String(s.h).padStart(4)} | ${s.counts.map(pct).join(" | ")} | ${pct(s.rainOnly)} | ${pct(s.snowOnly)} | ${s.med.toFixed(3)} | ${s.p90.toFixed(3)} | ${s.max.toFixed(2)} | ${pct(s.cloudN)}${reg}`);
}
const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
console.log(`mean over ${HOURS} h: ` + THRESHOLDS.map((t, i) => `>${t}: ${pct(avg(perHourStats.map((s) => s.counts[i])))}`).join(", "));
const paintedFr = perHourStats.map((s) => s.paintedN / N).sort((a, b) => a - b);
console.log(`\n## regime, at the overlay threshold PAINT=${PAINT} mm/h`);
console.log(`painted fraction per hour: min ${(100 * paintedFr[0]).toFixed(1)} % | median ${(100 * paintedFr[Math.floor(paintedFr.length / 2)]).toFixed(1)} % | max ${(100 * paintedFr[paintedFr.length - 1]).toFixed(1)} %`);
console.log(`dry hours (< 0.1 % of map painted): ${paintedFr.filter((f) => f < 0.001).length} / ${HOURS}`);
console.log(`cloud cover (cells with cloud_water > ${CLOUD_MIN} mm): mean ${pct(avg(perHourStats.map((s) => s.cloudN))).trim()}, min ${pct(Math.min(...perHourStats.map((s) => s.cloudN))).trim()}, max ${pct(Math.max(...perHourStats.map((s) => s.cloudN))).trim()}`);
console.log(`map-wide hours (> 50 % of map painted): ${paintedFr.filter((f) => f > 0.5).length} / ${HOURS}`);
if (hasRegime) {
  const wetH = perHourStats.filter((s) => s.wet).length;
  console.log(`regime: ${wetH} WET hours / ${HOURS - wetH} dry hours; sky stock min ${Math.min(...perHourStats.map((s) => s.sky ?? 0)).toFixed(0)} / max ${Math.max(...perHourStats.map((s) => s.sky ?? 0)).toFixed(0)} cell-mm`);
}

// per-cell persistence
const hoursRaining = new Uint16Array(N);
for (const { total } of hourly) for (let i = 0; i < N; i++) if (total[i] > PAINT) hoursRaining[i]++;
const bins = [[0, 0], [1, 3], [4, 12], [13, 24], [25, HOURS - 1], [HOURS, HOURS]];
console.log(`\n## per-cell persistence: hours painted out of ${HOURS}`);
for (const [a, b] of bins) {
  const n = hoursRaining.reduce((s, v) => s + (v >= a && v <= b ? 1 : 0), 0);
  console.log(`${String(a).padStart(3)}-${String(b).padEnd(3)} h : ${pct(n)}`);
}
// what share of all painted cell-hours comes from cells painted > half the time
const totalCH = hoursRaining.reduce((s, v) => s + v, 0);
const stickyCH = hoursRaining.reduce((s, v) => s + (v > HOURS / 2 ? v : 0), 0);
console.log(`share of painted cell-hours from cells painted > ${HOURS / 2} h: ${((100 * stickyCH) / totalCH).toFixed(0)} %`);

// Jaccard at lags
function jaccard(a, b) {
  let inter = 0, uni = 0;
  for (let i = 0; i < N; i++) { const x = a[i] > PAINT, y = b[i] > PAINT; if (x && y) inter++; if (x || y) uni++; }
  return uni ? inter / uni : 1;
}
console.log(`\n## Jaccard of the painted set between hours (1 = same islets, 0 = fully renewed)`);
for (const lag of [1, 6, 24]) {
  const vals = [];
  for (let h = 0; h + lag < HOURS; h++) vals.push(jaccard(hourly[h].total, hourly[h + lag].total));
  console.log(`lag ${String(lag).padStart(2)} h : mean ${avg(vals).toFixed(2)}  (min ${Math.min(...vals).toFixed(2)}, max ${Math.max(...vals).toFixed(2)})`);
}

// connected patches (no torus wrap: edge effect negligible at r120)
const key = (q, r) => `${q},${r}`;
const byKey = new Map(coords.map(([q, r], i) => [key(q, r), i]));
const DIRS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]];
function patches(total) {
  const seen = new Uint8Array(N); const sizes = [];
  for (let i = 0; i < N; i++) {
    if (seen[i] || total[i] <= PAINT) continue;
    let size = 0; const stack = [i]; seen[i] = 1;
    while (stack.length) {
      const c = stack.pop(); size++;
      const [q, r] = coords[c];
      for (const [dq, dr] of DIRS) { const j = byKey.get(key(q + dq, r + dr)); if (j !== undefined && !seen[j] && total[j] > PAINT) { seen[j] = 1; stack.push(j); } }
    }
    sizes.push(size);
  }
  return sizes.sort((a, b) => b - a);
}
console.log(`\n## islets: connected components of painted cells, per hour`);
const pstats = hourly.map(({ total }) => patches(total));
const nPatch = pstats.map((s) => s.length);
const biggest = pstats.map((s) => s[0] ?? 0);
const singles = pstats.map((s) => s.filter((x) => x <= 2).length);
console.log(`patches/hour: mean ${avg(nPatch).toFixed(0)} (min ${Math.min(...nPatch)}, max ${Math.max(...nPatch)})`);
console.log(`biggest patch: mean ${avg(biggest).toFixed(0)} cells = ${pct(avg(biggest)).trim()} of map`);
console.log(`patches of 1-2 cells: mean ${avg(singles).toFixed(0)} per hour (${((100 * avg(singles)) / avg(nPatch)).toFixed(0)} % of patches)`);

// daily totals
console.log(`\n## daily totals per cell (mm/day), 2 days`);
for (let d = 0; d * 24 + 23 < HOURS; d++) {
  const daily = new Float32Array(N);
  for (let h = d * 24; h < d * 24 + 24; h++) for (let i = 0; i < N; i++) daily[i] += hourly[h].total[i];
  const c = (t) => daily.reduce((s, v) => s + (v >= t ? 1 : 0), 0);
  const sum = daily.reduce((s, v) => s + v, 0);
  console.log(`day ${d}: ≥0.1 mm ${pct(c(0.1))} | ≥1 mm ${pct(c(1))} | ≥5 mm ${pct(c(5))} | ≥10 mm ${pct(c(10))} | map mean ${(sum / N).toFixed(2)} mm/day`);
}

// by altitude band (fraction painted, mean over hours)
console.log(`\n## painted fraction by altitude band (mean over hours)`);
const bands = [[-1e9, 300], [300, 800], [800, 1500], [1500, 1e9]];
for (const [lo, hi] of bands) {
  const ids = []; for (let i = 0; i < N; i++) if (elev[i] >= lo && elev[i] < hi) ids.push(i);
  if (!ids.length) continue;
  const fr = avg(hourly.map(({ total }) => ids.reduce((s, i) => s + (total[i] > PAINT ? 1 : 0), 0) / ids.length));
  console.log(`${String(lo === -1e9 ? "<300" : hi === 1e9 ? ">1500" : `${lo}-${hi}`).padEnd(9)} m : ${(100 * fr).toFixed(1).padStart(5)} %  (${ids.length} cells)`);
}
