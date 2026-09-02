# Embedding HexSim in another page

Public contract between this repo and the sites that embed the simulation. It's
consumed from another repo: what's written here doesn't change without a version
bump, and without saying so.

## Display modes

Passed as a URL parameter, they decide what HexSim draws itself.

| `?chrome=` | What remains |
|---|---|
| absent | the full interface: params, telemetry, logs |
| `minimal` | a bar at the bottom: date, play/pause. Nothing else |
| `none` | the scene alone, no UI. The host draws its own |

In both embed modes, the scene background and the page background are
transparent: the host page shows through behind the terrain. Provide a background
on the host side, or the iframe sits on white.

## `window.__hexsim`

Set by an inline script **before** the main module loads: a host can call it
as soon as the iframe's `load` fires, without waiting for anything else. Calls
received before the simulation is ready are queued and delivered at
initialization.

```js
const sim = iframe.contentWindow.__hexsim;

sim.play();
sim.pause();
sim.speed(1);              // simulated hours per second

sim.view("temperature");   // map background, exclusive
sim.layer("clouds", false);// overlay, stackable
sim.views;                 // ["terrain", "species", …]
sim.layers;                // ["wind", "clouds", …]

sim.inspect(true);         // cell inspector, off by default in embed

const buf = await sim.export();   // ArrayBuffer: the full world state
await sim.load(buf);              // restores it

const off = sim.on("ready", () => { /* terrain built, first state received */ });
const off2 = sim.on("tick", ({ date, playing, tick, speed, view, layers, inspect }) => { /* … */ });
off();  // `on` returns its unsubscribe function
```

`on` calls back **immediately** if the event has already happened: `on("ready")`
registered after initialization still fires, and `on("tick")` delivers
the last known state instead of leaving an empty bar until the next one.
Without this, a subscription set up a millisecond too late would never
fire, and that's painful to debug from the other repo.

### `tick`

Emitted on every state received from the simulation, **and** on every
play/pause toggle: "what a host displays has changed", not just "time has
moved forward". A `playing` value received via immediate callback is
therefore never stale.

- `date`: the string HexSim displays itself (`8 Jan An 0 15h`). The time
  disappears above 24 h/s, where it scrolls too fast to read.
- `playing`: `true` while playing
- `tick`: the raw counter, in days
- `speed`: the actual applied speed, in hours per second
- `view`: the background actually displayed
- `layers`: the ids of the overlays turned on, `["clouds"]` at start
- `inspect`: whether the cell inspector is active (`false` at start in embed)

The last five fields state **what is applied**, never what was requested.
A button bar painted from `tick` stays correct even when a command is
refused (and it will be: an unknown id changes nothing).

### `speed(hoursPerSecond)`

Bounded to `[1, 1000]` h/s. Below 24 h/s the time stays shown in
`date`; above it, it scrolls too fast to read and disappears.

The value sits on a logarithmic scale with integer steps, so it's rounded:
it's the `speed` field of the `tick` event that states what's actually
applied, never the requested value. Two paces that read well: **1 h/s** (one
diurnal cycle in 24 s) and **12 h/s** (weather and erosion visible).

An embed opens at 1 h/s. Until v0.9.1 it actually opened at ~28 h/s:
the slider position was never passed to the engine (#144).

**24 h/s is also the day/night threshold.** Below it, the scene is lit at
the simulation's actual time of day: the sun sets, night falls, shadows
turn. Above it, a full cycle would pass in under a second, so lighting
freezes on its last state instead of strobing. A single notion of
"readable pace" drives both.

At 1 h/s, the default, one cycle lasts 24 s. An embed opens on the world
shipped at June 30 noon: it takes six simulated hours, that is six seconds,
before the sun starts to go down. Nobody lands on a black screen.

Until v0.10.0, an embed's lighting was frozen at noon on the solstice,
regardless of the actual time: the bar would announce 22h on a terrain in
broad daylight. It didn't show on a fresh world; it was glaring on the
shipped one.

### `view(id)` and `layer(id, on)`

One **map background** at a time, as many **overlays** as you want. Both
lists are read from the API (`sim.views` and `sim.layers`), never hardcoded:
a host that hardcodes the ids silently breaks the day a view gets renamed
here.

```js
for (const id of sim.views) boutonFond(id);
for (const id of sim.layers) boutonSurcouche(id);
```

The lists come from the embedded page's DOM, so they're valid as soon as
the iframe's `load` fires. These are stable ids, not labels: the host
applies its own, in its own language.

**The DOM as the source is deliberate, and it's a constraint we hold to.**
The settings panel is still built in every mode: `?chrome=none` **hides** it
in CSS, it doesn't skip building it, and the `value`s of its `<select>` are
literally the ids of this contract. That's what guarantees a view renamed
here can't leave a dead button on your side. Two safeguards rather than one
promise: `just embed-check` runs in `chrome=none` and fails if the list is
empty, and an empty list writes a console error instead of silently
rendering an empty array. If you see `__hexsim.views is empty`, that's a
regression on our side, not misuse on yours.

As of September 1, 2026, nine backgrounds (`terrain`, `species`,
`permeability`, `groundwater`, `humidity_upper`, `humidity_surface`, `age`,
`fire`, `pressure`) and four overlays (`wind`, `clouds`, `temperature`,
`precipitation`). An embed opens on `terrain` + `clouds`.

An unknown id is **rejected**, with a console warning, and the next
`tick` keeps announcing what's actually displayed. Nothing is silently
clamped.

**What's worth it in an embed.** The nine are exposed, none is
filtered upstream: what deserves a button is your call, not ours.
Eight show something at all times. The four overlays stack without
clashing; `wind` and `precipitation` together visually clutter the map,
that's all.

**`fire` is a diagnostic instrument, not a show.** It renders a dark map
where only burning cells stand out, so, most of the time, a dark map and
nothing else. Fire is dormant by default (`fire.enabled = false`, in the
shipped world as everywhere else), and **turning it on would change almost
nothing**: the ignition rate is calibrated for "very low risk", about one
start every three years at radius 30 (roughly one every eighteen months at
radius 45, proportional to the cell count), for fires that die out on
their own in two to four days. A visitor would therefore have less than a
one in a hundred chance of hitting one. That's why there's no command to
light a fire in this contract: it wouldn't make the view more alive, it
would just relocate the disappointment.

The rendering cost is that of a mesh rebuild, the same for every
background except `pressure`, which renders the bare map and so costs
less. No view requests data that's absent in wasm; the synoptic runs there
too, `pressure` is populated. Nothing has been measured as an issue at
12 h/s; that's an observation from use, not a measurement.

### `inspect(on)`

The **cell inspector**: a tooltip that follows the mouse with the full
reading of the hovered hexagon (elevation, temperature, water, water table,
cover, wind, …), and a click that copies that reading to the clipboard.

**Off by default under `?chrome=minimal` and `?chrome=none`.** This is
debugging tooling: the tooltip carries HexSim's own styling and renders on
top of your page with no awareness of its style, and above all the click
writes to the visitor's clipboard, on a 3D scene where clicking is first
used to rotate the camera. Reported at the embed's first playtest (#150).

```js
sim.inspect(true);   // to wire to your own button, if you want
sim.inspect(false);
```

Turning it off also clears the on-screen tooltip: without `mousemove` to
update it, it would stay frozen on the last hovered cell. And the click is
short-circuited **at the top of the listener**, not based on the tooltip's
visibility; the two have already gone out of sync once.

Without `?chrome=`, in the full interface, the inspector stays on: it's
the project's own debugging tool, it doesn't change.

> **What this says about the bare mode.** There was indeed, since #142, a
> rule `body[data-chrome="none"] #tooltip { display: none }`. It never hid
> anything: the handler writes `style.display = "block"` **inline**, which
> beats any stylesheet. A JS-driven UI doesn't turn off via CSS, and hiding
> the element wouldn't have stopped the click from reaching the clipboard
> anyway. The rule was removed in v0.12.0.

### `export()` and `load(buffer)`

The full world state, so a returning visitor finds the one they left
rather than January 1st, Year 0.

```js
const buf = await sim.export();        // ArrayBuffer
await idb.put("hexsim", buf);          // IndexedDB: localStorage is too small

const stocke = await idb.get("hexsim");
if (stocke) {
  try { await sim.load(stocke); }
  catch { /* incompatible: keep the shipped world */ }
}
```

`load` accepts an `ArrayBuffer`, a typed view, or a `Blob`. The buffer you
pass is **not** neutered: HexSim takes a copy of it before transferring it
to its worker.

All three forms work from your realm, which isn't obvious and isn't free:
in v0.11.0, an `ArrayBuffer` built by the host page (that is, a state read
back from IndexedDB, the very use case of this API) was rejected with
`data.arrayBuffer is not a function`. An `instanceof ArrayBuffer` evaluated
in the iframe's realm returns `false` on an object coming from yours.
Fixed in v0.11.1 with structural recognition, and verified from a real
host page: the previous test passed a buffer born inside the iframe, so it
couldn't see the failure.

**Version and invalidation.** A checkpoint carries a `HEXSIM_CKPT` marker,
an integer format version, and the version of the engine that wrote it.
`load` checks the first two and **cleanly rejects** a format version that
isn't its own: no migration, no inconsistent world. The host therefore has
nothing to compare before calling: it stores, it retries on return, it
falls back to the shipped world if that's rejected. It's the simplest of
the three behaviors considered, and it's the one that's implemented.

A rejected `load` leaves the current world **intact**: the simulation
carries on as if nothing happened.

**What it costs.** The export is synchronous in the worker: **85 ms** on
the shipped world, measured by `just embed-check`. The simulation doesn't
advance during that time, but rendering lives on the main thread and stays
smooth: the image freezes, it doesn't stutter. A `load` costs the same
order of magnitude (~95 ms). That's little enough not to show at 1 h/s,
and enough that a periodic export is pointless: `pagehide` is sufficient.

**What it weighs.** A 42-year world at radius 45 is **52.4 MB** raw,
**2.0 MB** gzipped (`CompressionStream("gzip")`, 96% less). That's ten
times more than a world a few hours old, which weighs 4.8 MB: the gap is
a 365-day-per-cell climate history that fills up over the first simulated
year and caps there. `localStorage` is out of the question (5 MB quota,
strings only); IndexedDB takes the `ArrayBuffer` as is.

**Gzip is on you.** `export` returns the native format, the one `load`
takes back and the SAVE button downloads: a single format, no compressed
variant to tell apart. The browser gzips natively and off the main thread,
there's nothing to gain doing it on the wasm side.

**The shipped world is only the world.** Speed, play/pause, and view stay
controlled by the host, which already knows them; the checkpoint carries
none of it.

**A host that restores its own state** can open the iframe on
`?world=neuf`: the shipped 1.9 MB world is then not downloaded just to be
overwritten right away. It's only an optimization: without this parameter,
a `load()` from the host still wins last, regardless of arrival order.

### Startup

An embed starts playing as soon as the simulation responds: nobody clicks
PLAY inside an iframe. **A call to `pause()` takes priority**, even if
received before initialization: a host whose section is offscreen at load
time does get a stopped simulation.

`play()` and `pause()` both work from the queue: there is **no** need to
wait for `ready` before calling either one. This wasn't true in v0.9.0,
where a `play()` passed through the queue was lost (#143).

### The starting world

An embed doesn't start on a fresh world. It loads `worlds/aged.ckptz`,
a **42-year** world shipped in the archive: forests established, lakes
filled, forty-year-old trees. A fresh world is a bare board, and the three
minutes of computation it takes to grow out of that, nobody waits for that
on a page.

The file weighs 1.9 MB (50 MB decompressed). It's loaded after the
connection, so the scene shows the fresh world before switching over:
**300 ms locally, 2.5 s behind a CDN** (measured on Cloudflare Pages:
there are 2 MB to download, and that's the figure that matters to a
visitor). During that time, playback hasn't started: the first `tick` a
host receives can therefore be tick 0, and the next one tick 15510, two
and a half seconds later. **Don't infer the world's state from the first
`tick` received**, and don't paint `1 Jan An 0` as stable information.

The file is served from the same origin as the page, like everything
else. If it's missing or rejected, the embed starts on a fresh world
rather than showing nothing, and says so in the console.

**It must be served as is, without `Content-Encoding`.** It's already
gzipped, and the page decompresses it. A server that announces
`Content-Encoding: gzip` makes the browser decompress it once too often:
52 MB arrive where the header promised 2, the response is aborted, and
your console shows a `TypeError: Failed to fetch` on a request that
nonetheless responded 200. The `.ckptz` extension is chosen for exactly
this: static servers that guess encoding do it based on `.gz`, and none
know `.ckptz`. If you still see that `TypeError`, this is the lead to
follow, and the page spells it out right below the error.

`?world=neuf` doesn't import any shipped world. `?world=<nom>` loads
`worlds/<nom>.ckptz` instead, if you ship others of your own: also
gzipped, and named with the same extension for the same reason.

`neuf` means "import nothing", not "reset": for an embed, which simulates
inside the tab, that amounts to the same thing, since it does start from
a fresh world. In server mode, where the world lives elsewhere and
survives reloads, the parameter leaves in place whatever the server
already had.

### Scope

`play`, `pause`, `speed`, `view`, `layer`, `views`, `layers`, `export`, `load`,
`ready`, `tick`, and nothing else. `step()` and `reset()` will come if
they're needed. A contract between two repos holds over time.

What HexSim won't do for you: decide when to export, under what key to
store it, when to discard it. The host knows when its page closes, HexSim
doesn't.

## Same origin required

`window.__hexsim` assumes the host and the iframe share the same origin:
that's the case when the site copies the contents of `dist.tar.gz` onto its
own server, the intended use. A cross-origin embed of a remote HexSim can't
reach the object. The internal dispatch is written by command name so that
a `postMessage` bridge would be an addition, not a rewrite, but it doesn't
exist today.

## Suspend when offscreen

Nothing does this on its own: a loaded iframe keeps the simulation running
even outside the viewport. It's up to the host, which is the only one that
knows where things stand.

```js
new IntersectionObserver(([e]) => {
  const sim = iframe.contentWindow.__hexsim;
  e.isIntersecting ? sim.play() : sim.pause();
}).observe(iframe);
```
