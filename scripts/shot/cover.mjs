#!/usr/bin/env node
//
// hexsim-cover, reads the vegetation cover composition of the current state.
//
// Companion diagnostic to shot.mjs: connects to the front end (localhost:8355),
// reads `window.__hexcam.cover()` (frontend/main.js) and prints the JSON. Gives,
// without recomputing (the core provides `dominant_species` + `species_mix`):
//   - count / pct  : number & % of cells per dominant species (+ water, bare soil)
//   - meanVeg      : average total vegetation cover [0,1]
//   - mix          : mix *within* the hexes:
//       meanDominantShare : average share of the dominant species (max/Σ)
//       meanEffSpecies    : effective number of species/hex (inverse Simpson)
//       monoPct / mixedPct: % of hexes > 90% one species / < 70% dominant
//       communityPct      : average composition (biomass share per species)
//
// Drive the sim alongside (just play|pause|step|reset) then `just cover`.
//
// Usage : node cover.mjs [--url http://localhost:8355] [--timeout 20000]

import { chromium } from "playwright";

function parse(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) o[argv[i].slice(2)] = argv[i + 1];
  }
  return o;
}
const opt = parse(process.argv.slice(2));
const URL = opt.url ?? "http://localhost:8355";
const timeout = Number(opt.timeout ?? 20000);

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  // `?capture`: preserveDrawingBuffer on the front end, otherwise a black capture (main.js).
  const captureUrl = new URL(URL);
  captureUrl.searchParams.set("capture", "1");
  await page.goto(captureUrl.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__hexcam && window.__hexcam.ready(), {
    timeout,
  });
  const cover = await page.evaluate(() => window.__hexcam.cover());
  console.log(JSON.stringify(cover, null, 1));
} finally {
  await browser.close();
}
