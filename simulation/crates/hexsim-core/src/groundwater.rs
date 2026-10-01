use serde::{Deserialize, Serialize};

use crate::cell::CellProperties;
use crate::coord::opposite_direction;
use crate::dynamics::{CELL_AREA_M2, CELL_SPACING_M};
use crate::grid::HexGrid;
use crate::par::{for_each_chunk_mut, for_each_chunk_mut6, sum_dir_out};

/// Default capacity of a cell at `permeability=1.0`. Single source of
/// truth for components that need a static reference for "full water
/// table" (e.g. the evapotranspiration proxy in `atmosphere.rs`).
pub const DEFAULT_MAX_CAPACITY_MM: f32 = 100.0;

#[derive(Clone, Serialize, Deserialize)]
pub struct GroundwaterParams {
    /// Saturated hydraulic conductivity of the soil surface (mm/day) when
    /// `permeability = 1`: the infiltration capacity once the soil is wet
    /// (Green-Ampt's rate tends to `Ks`, never below it). A cell takes in
    /// `permeability × Ks × frozen_factor` per day, as much as its root
    /// zone has room for. 317 mm/day is a loam, 13.2 mm/h (Rawls,
    /// Brakensiek & Saxton 1982); sand 2 820, clay 14. Replaces the
    /// fraction-of-ponded-water-per-day `infiltration_rate` (0.05) that let
    /// 10 mm of standing water into a perm-0.5 soil at 0.25 mm/day: the
    /// rain ran off to the lakes instead of wetting the soil (2026-10-01,
    /// `diag_water_cycle`).
    #[serde(default = "default_saturated_conductivity")]
    pub saturated_conductivity_mm_per_day: f32,
    pub diffusion_rate: f32,
    // Max capacity when permeability=1.0. The actual capacity of a cell
    // = max_capacity * permeability. Impermeable rock stores little.
    pub max_capacity: f32,
    /// Baseflow recession coefficient of the deep aquifer (linear
    /// reservoir, /day). At each daily slice a fraction `baseflow_coef`
    /// of the cell's `aquifer` seeps back to the surface:
    /// `seepage = baseflow_coef × aquifer`. Maillet's recession (1905): a
    /// spring's discharge decays as `Q₀·e^(−αt)` between two recharges.
    /// Drawn from the aquifer only, never from the root zone: in July
    /// (#107) it was drawn from the single soil bucket the plants empty
    /// every summer and had nothing to give back (perennial 4 → 5), and
    /// re-measured on 2026-10-01 it still emptied a root zone already at
    /// 0.7 mm. Conservative: `aquifer → water_level`.
    #[serde(default)]
    pub baseflow_coef: f32,
    /// Field capacity, as a fraction of the cell's total storage
    /// (`permeability × max_capacity`). Water below that level is held
    /// by capillary forces against gravity and does NOT drain
    /// laterally: only `groundwater − field_capacity` takes part in the
    /// piezometric flow of step 2 (Veihmeyer & Hendrickson 1931, bucket
    /// model of FAO-56 §22). Without it, the slopes' water table
    /// emptied into the valley in 1-2 days at 130 m spacing and left
    /// half the map bare (#151: ablation `diffusion = 0` → 2% bare
    /// instead of 57% at 800-1500 m).
    ///
    /// 0.65 of total pore storage is the loam end of the range: the
    /// proper value comes from the lithology of each cell, which the
    /// engine does not model yet, which is why this is a single
    /// world-wide fraction for now and not a per-cell soil property.
    #[serde(default = "default_field_capacity_frac")]
    pub field_capacity_frac: f32,
    /// Fraction of the root zone's water above field capacity that
    /// percolates down to the aquifer each day (/day). Field capacity is
    /// by definition what a soil holds 2-3 days after saturation once
    /// free drainage has stopped (Veihmeyer & Hendrickson 1931): the
    /// excess leaves with an e-folding time of ~1.5 day, 0.5/day. The
    /// drainage is vertical first, lateral interflow (step 2) works on
    /// what percolation left.
    #[serde(default = "default_percolation_rate")]
    pub percolation_rate: f32,
    /// Saturated thickness of a full aquifer when `permeability = 1`
    /// (m). The aquifer of a cell spans `permeability × aquifer_thickness_m`
    /// below the surface: thin in tight rock, thick in porous rock. 10 m
    /// is a shallow unconfined aquifer (alluvium, weathered bedrock).
    #[serde(default = "default_aquifer_thickness_m")]
    pub aquifer_thickness_m: f32,
    /// Specific yield (dimensionless, m³ of water drained per m³ of
    /// aquifer): converts the stock (mm of water) into a water-table
    /// height. 0.15 is the middle of sand and gravel, 0.1-0.3 (Johnson
    /// 1967). Storage of a full aquifer: `1000 × S_y × thickness` mm,
    /// 1500 mm at `permeability = 1`.
    #[serde(default = "default_specific_yield")]
    pub specific_yield: f32,
    /// Hydraulic conductivity of the aquifer (m/day), Darcy's `K`. 10 m/day
    /// is fine to medium sand, 1-50 m/day (Freeze & Cherry 1979, table
    /// 2.2). Transmissivity is `K × b` with `b` the saturated thickness of
    /// the upstream cell (Dupuit): an empty aquifer does not conduct.
    #[serde(default = "default_aquifer_conductivity")]
    pub aquifer_conductivity_m_per_day: f32,
}

fn default_saturated_conductivity() -> f32 {
    DEFAULT_SATURATED_CONDUCTIVITY_MM_PER_DAY
}

/// See [`GroundwaterParams::saturated_conductivity_mm_per_day`].
pub const DEFAULT_SATURATED_CONDUCTIVITY_MM_PER_DAY: f32 = 317.0;

fn default_percolation_rate() -> f32 {
    DEFAULT_PERCOLATION_RATE
}

fn default_aquifer_thickness_m() -> f32 {
    DEFAULT_AQUIFER_THICKNESS_M
}

fn default_specific_yield() -> f32 {
    DEFAULT_SPECIFIC_YIELD
}

fn default_aquifer_conductivity() -> f32 {
    DEFAULT_AQUIFER_CONDUCTIVITY_M_PER_DAY
}

/// See [`GroundwaterParams::percolation_rate`].
pub const DEFAULT_PERCOLATION_RATE: f32 = 0.5;

/// See [`GroundwaterParams::aquifer_thickness_m`].
pub const DEFAULT_AQUIFER_THICKNESS_M: f32 = 10.0;

/// See [`GroundwaterParams::specific_yield`].
pub const DEFAULT_SPECIFIC_YIELD: f32 = 0.15;

/// See [`GroundwaterParams::aquifer_conductivity_m_per_day`].
pub const DEFAULT_AQUIFER_CONDUCTIVITY_M_PER_DAY: f32 = 10.0;

/// Millimetres per metre, for the aquifer's mm-of-water stock against its
/// metre-scale heads.
const MM_PER_M: f32 = 1000.0;

/// See [`GroundwaterParams::baseflow_coef`].
pub const DEFAULT_BASEFLOW_COEF: f32 = 0.01;

/// Serde default for [`GroundwaterParams::field_capacity_frac`], so a
/// checkpoint written before #151 reloads with the current physics
/// rather than with a silent 0.0 (no capillary water at all).
fn default_field_capacity_frac() -> f32 {
    DEFAULT_FIELD_CAPACITY_FRAC
}

/// See [`GroundwaterParams::field_capacity_frac`].
pub const DEFAULT_FIELD_CAPACITY_FRAC: f32 = 0.65;

/// Scratch buffers for `step_groundwater_into`'s two-phase piezometric
/// flow (r250 perf effort, chunk C1), owned by the caller and reused
/// every call: content between two calls is undefined, same convention
/// as `atmosphere::AtmoScratch`. `dir_out[d][i]` is the amount cell `i`
/// routes toward its toric neighbor in direction `d`
/// (`coord::DIRECTIONS[d]`) this step, filled by a parallel per-source
/// outflow pass and read back by the following per-destination gather
/// pass via `coord::opposite_direction`. `snap_*` are the
/// post-infiltration snapshot of groundwater/permeability/elevation the
/// outflow phase reads, replacing 3 freshly-allocated `Vec`s per call.
pub struct GroundwaterScratch {
    pub dir_out: [Vec<f32>; 6],
    pub snap_gw: Vec<f32>,
    pub snap_perm: Vec<f32>,
    pub snap_elev: Vec<f32>,
    pub snap_aquifer: Vec<f32>,
}

impl GroundwaterScratch {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            dir_out: std::array::from_fn(|_| Vec::with_capacity(n)),
            snap_gw: Vec::with_capacity(n),
            snap_perm: Vec::with_capacity(n),
            snap_elev: Vec::with_capacity(n),
            snap_aquifer: Vec::with_capacity(n),
        }
    }
}

impl Default for GroundwaterParams {
    fn default() -> Self {
        Self {
            saturated_conductivity_mm_per_day: DEFAULT_SATURATED_CONDUCTIVITY_MM_PER_DAY,
            diffusion_rate: 0.03,
            // 100 (vs 5): lets the water table be a real primary
            // freshwater stock (~50 mm × 2790 cells = ~140,000 mm
            // cumulative, vs 7000 mm of surface currently). With this
            // capacity, the water table can feed rivers in the dry
            // season via resurgence (groundwater > capacity → return to
            // water_level). A true physical aquifer means meters of
            // column, but 100 mm is already a big improvement over the
            // initial 5 mm and avoids breaking the global calibration.
            max_capacity: DEFAULT_MAX_CAPACITY_MM,
            baseflow_coef: DEFAULT_BASEFLOW_COEF,
            field_capacity_frac: DEFAULT_FIELD_CAPACITY_FRAC,
            percolation_rate: DEFAULT_PERCOLATION_RATE,
            aquifer_thickness_m: DEFAULT_AQUIFER_THICKNESS_M,
            specific_yield: DEFAULT_SPECIFIC_YIELD,
            aquifer_conductivity_m_per_day: DEFAULT_AQUIFER_CONDUCTIVITY_M_PER_DAY,
        }
    }
}

// Underground cycle: infiltration → diffusion → resurgence.
//
// The water table is the "slow path" of water, as opposed to runoff
// (fast path). The infiltration rate controls the proportion of
// surface water that enters the soil on each tick.
//
// Underground diffusion is much slower than atmospheric diffusion: water
// in the soil moves slowly laterally through the porous rock. This is
// the mechanism that distributes water under hills and feeds the
// springs downhill.
pub fn step_groundwater(current: &HexGrid, next: &mut HexGrid, params: &GroundwaterParams) {
    let mut scratch = GroundwaterScratch::new(current.len());
    step_groundwater_into(current, next, params, &mut scratch);
}

/// Zero-malloc variant of [`step_groundwater`]: reuses `scratch` across
/// calls instead of allocating its snapshot/outflow buffers each time.
/// Same 4 steps, infiltration/baseflow/resurgence are pure per-cell
/// maps parallelized directly (`par::for_each_chunk_mut`); step 2's
/// piezometric flow is a two-phase scatter -> gather split (r250 perf
/// effort, chunk C1), documented on `fill_groundwater_outflow`.
pub fn step_groundwater_into(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &GroundwaterParams,
    scratch: &mut GroundwaterScratch,
) {
    let cur_cells = current.cells_slice();

    // Step 1: Infiltration (surface → water table)
    // Weighted by local permeability: impermeable soil → little
    // infiltration. Read on `current`, cell-local write on `next` → a
    // pure per-cell map, parallelizable (`par::for_each_chunk_mut`).
    // Folds the historical `current → next` full-grid copy into this
    // same sweep (r250 perf effort, chunk B2): this is the phase's
    // first per-cell pass, so starting each cell from `*nc = cell.clone()`
    // before applying infiltration is bit-identical to the separate
    // copy, one fewer 88-byte full-grid stream per tick.
    for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
        for (local, nc) in chunk.iter_mut().enumerate() {
            let i = start + local;
            let cell = &cur_cells[i];
            *nc = cell.clone();
            if cell.water_level <= 0.0 {
                continue;
            }
            let capacity = cell.permeability * params.max_capacity;
            // Frozen soil: linear transition between -2°C (impermeable) and
            // 0°C. Meltwater at the foot of glaciers stays on the surface →
            // runs off. Written as an explicit ramp rather than
            // `midpoint(T, 2.0).clamp(0.0, 1.0)` (#49): both compute the same
            // numbers, but the midpoint form reads as a formula that keeps
            // rising above 0°C, where it has no physical meaning, and hides
            // the plateau inside the clamp.
            let frozen_factor = if cell.temperature <= -2.0 {
                0.0
            } else if cell.temperature >= 0.0 {
                1.0
            } else {
                (cell.temperature + 2.0) / 2.0
            };
            let infiltration_capacity =
                params.saturated_conductivity_mm_per_day * cell.permeability * frozen_factor;
            let room = (capacity - cell.groundwater).max(0.0);
            let infiltration = cell.water_level.min(infiltration_capacity).min(room);
            nc.water_level -= infiltration;
            nc.groundwater += infiltration;
        }
        for nc in chunk.iter_mut() {
            percolate(nc, params);
        }
    });

    // Step 2: Underground flow (conservative transfers, gravity-driven)
    //
    // Groundwater follows gravity via the piezometric level: piezometric
    // = elevation + groundwater. This is the "pressure" of water
    // underground. Water flows from the high piezo level to the low
    // one: a water table in the mountains (elev=400, gw=1 → piezo=401)
    // flows toward the plain (elev=50, gw=3 → piezo=53) even if the
    // plain has more groundwater.
    //
    // ASSUMED mm/m MIX (discovered in #104): gw is in mm, elevation in
    // m. The same hybrid as fixed in `effective_elevation`, but here it
    // drives a slow diffusion capped at ~100 mm of stock, not surface
    // topology, and no ablation has measured it. Transition to SI: to be
    // settled with the Darcy rework of the water table (post-#105), not
    // sneaked in as part of the surface effort.
    //
    // Post-infiltration snapshot of the needed fields (gw, perm, elev)
    // in per-cell indexed Vecs, reused tick to tick (chunk C1) instead
    // of 3 fresh Vecs per call. Avoids the intermediate HashMap and lets
    // `next.cells_slice_mut()` be mutated freely in the gather phase.
    scratch.snap_gw.clear();
    scratch.snap_perm.clear();
    scratch.snap_elev.clear();
    for cell in next.cells_slice() {
        scratch.snap_gw.push(cell.groundwater);
        scratch.snap_perm.push(cell.permeability);
        scratch.snap_elev.push(cell.elevation);
    }

    for dir in &mut scratch.dir_out {
        dir.clear();
        dir.resize(cur_cells.len(), 0.0);
    }
    fill_groundwater_outflow(
        current,
        &scratch.snap_gw,
        &scratch.snap_perm,
        &scratch.snap_elev,
        params,
        &mut scratch.dir_out,
    );
    gather_groundwater_transfers(current, &scratch.dir_out, params, next);

    // Step 3: the deep aquifer's lateral flow, springs and baseflow
    // (#107). Elevation and permeability are untouched since the
    // snapshot, only the aquifer stock is re-read.
    scratch.snap_aquifer.clear();
    scratch
        .snap_aquifer
        .extend(next.cells_slice().iter().map(|c| c.aquifer));
    fill_aquifer_outflow(
        current,
        &scratch.snap_aquifer,
        &scratch.snap_perm,
        &scratch.snap_elev,
        params,
        &mut scratch.dir_out,
    );
    gather_aquifer_transfers(current, &scratch.dir_out, params, next);
}

/// Phase 1 of step 2's two-phase scatter -> gather split (r250 perf
/// effort, chunk C1): per source cell, the amount routed toward each of
/// its 6 toric neighbors this step. Reads only the post-infiltration
/// snapshot (`snap_*`, untouched by this function): fully independent
/// per source, parallelizable (`par::for_each_chunk_mut6`) — EVEN
/// THOUGH the historical serial loop computed one source's 6 outgoing
/// transfers sequentially, each capped by what the PREVIOUS transfer of
/// the SAME source left in `drainable` (`next_cells[i].groundwater -
/// field_capacity`, re-read live after every transfer): that
/// self-depleting budget is order-dependent (the first downhill
/// neighbor in `coord::DIRECTIONS` order gets first claim on the
/// surplus), but the dependency is entirely WITHIN one source's own 6
/// directions. `remaining_drainable` below walks the same 6 directions
/// in the same fixed order inside a single source's closure invocation,
/// replaying it exactly — never across sources, so splitting sources
/// across workers changes nothing.
fn fill_groundwater_outflow(
    current: &HexGrid,
    snap_gw: &[f32],
    snap_perm: &[f32],
    snap_elev: &[f32],
    params: &GroundwaterParams,
    dir_out: &mut [Vec<f32>; 6],
) {
    let base_rate = params.diffusion_rate / 6.0;
    for_each_chunk_mut6(dir_out, |start, chunks| {
        for local in 0..chunks[0].len() {
            let i = start + local;
            let piezometric = snap_elev[i] + snap_gw[i];
            // Toric neighborhood: the water table also diffuses across
            // the seam (physical piezometric gradient, periodic
            // terrain). j == i (degenerate grid) → zero diff, zero
            // transfer: conservative.
            let neighbors = current.neighbor_indices_toric(i);
            // Only the water above field capacity is drainable (#151):
            // below that level capillary forces hold it against
            // gravity, whatever the piezometric gradient.
            let field_capacity = snap_perm[i] * params.max_capacity * params.field_capacity_frac;
            let mut remaining_drainable = (snap_gw[i] - field_capacity).max(0.0);
            for (d, &j) in neighbors.iter().enumerate() {
                let neighbor_piezo = snap_elev[j] + snap_gw[j];
                if piezometric <= neighbor_piezo {
                    chunks[d][local] = 0.0;
                    continue;
                }
                let perm_factor = snap_perm[i].min(snap_perm[j]);
                let diff = piezometric - neighbor_piezo;
                let transfer = (base_rate * perm_factor * diff).min(remaining_drainable);
                if transfer > 0.0 {
                    remaining_drainable -= transfer;
                    chunks[d][local] = transfer;
                } else {
                    chunks[d][local] = 0.0;
                }
            }
        }
    });
}

/// Phase 2 of step 2 (r250 perf effort, chunk C1), fused with steps 3-4
/// (chunk B2, see below): per destination cell, applies the net
/// `groundwater` change — self-loss (`Σ_d dir_out[d][j]`, exactly what
/// `j` itself routed away in the outflow phase) and the inflow gathered
/// from its neighbors via `coord::opposite_direction`. `next` already
/// holds the post-infiltration snapshot at every cell (nothing has
/// touched `next`'s `groundwater` since [`step_groundwater_into`]
/// copied it into `snap_gw`), so this only needs to add the delta. A
/// pure per-cell map over already-fully-computed data, parallelizable
/// (`par::for_each_chunk_mut`).
///
/// `k == j` skips a neighbor slot that
/// [`HexGrid::neighbor_indices_toric`]'s doc calls out as its
/// self-transfer fallback ("wrap unreachable (non-hexagonal grid)"): on
/// a genuine `HexGrid::from_radius` torus this never fires, but on any
/// grid built by hand a filler self-loop at direction `d` makes
/// `opposite_direction(d)` alias one of `j`'s OWN real outgoing
/// directions — gathering it back would double-count `j`'s outflow as
/// its own inflow (see the identical guard and its worked example on
/// `hydro::gather_hydro_water`, which pins this exact hazard).
///
/// Resurgence of the root zone above its capacity (always active)
/// folded in (r250 perf effort, chunk B2): a pure per-cell map that reads
/// only THIS cell's own, already-final `groundwater`, never a neighbor's,
/// so appending it to this same sweep is bit-identical to a separate
/// full-grid pass. The aquifer's baseflow moved to
/// [`gather_aquifer_transfers`] (#107).
fn gather_groundwater_transfers(
    current: &HexGrid,
    dir_out: &[Vec<f32>; 6],
    params: &GroundwaterParams,
    next: &mut HexGrid,
) {
    for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            let j = start + local;
            let self_loss = sum_dir_out(dir_out, j);
            let neighbors = current.neighbor_indices_toric(j);
            let mut gathered_in = 0.0_f32;
            for (d, &k) in neighbors.iter().enumerate() {
                if k == j {
                    continue;
                }
                gathered_in += dir_out[opposite_direction(d)][k];
            }
            cell.groundwater += gathered_in - self_loss;

            let capacity = cell.permeability * params.max_capacity;
            if cell.groundwater > capacity {
                let surplus = cell.groundwater - capacity;
                cell.groundwater = capacity;
                cell.water_level += surplus;
            }
        }
    });
}

/// Step 1b, vertical drainage of the root zone into the aquifer (#107):
/// a fraction `percolation_rate` of the water above field capacity, as
/// much as the aquifer has room for. Water below field capacity is held
/// by capillarity and never percolates, so a root zone the plants keep
/// dry recharges nothing; the aquifer fills where the soil gets wet
/// (under ponds, lakes and channels, after a wet spell). Cell-local.
fn percolate(cell: &mut CellProperties, params: &GroundwaterParams) {
    let field_capacity = cell.permeability * params.max_capacity * params.field_capacity_frac;
    let drainable = cell.groundwater - field_capacity;
    if drainable <= 0.0 {
        return;
    }
    let room = aquifer_storage(cell.permeability, params) - cell.aquifer;
    if room <= 0.0 {
        return;
    }
    let percolation = (params.percolation_rate * drainable).min(room);
    let before = cell.groundwater;
    cell.groundwater -= percolation;
    cell.aquifer += before - cell.groundwater;
}

/// Storage of a full aquifer (mm of water): `1000 × S_y × b_max`.
fn aquifer_storage(permeability: f32, params: &GroundwaterParams) -> f32 {
    MM_PER_M * params.specific_yield * permeability * params.aquifer_thickness_m
}

/// Saturated thickness of the aquifer (m): the stock over the specific
/// yield.
fn saturated_thickness(aquifer_mm: f32, params: &GroundwaterParams) -> f32 {
    aquifer_mm / (MM_PER_M * params.specific_yield)
}

/// Hydraulic head of the aquifer (m, same datum as `elevation`): the
/// aquifer base sits `permeability × aquifer_thickness_m` below the
/// surface, the water table `saturated_thickness` above it. A full
/// aquifer has its head at the surface.
fn aquifer_head(
    elevation: f32,
    permeability: f32,
    aquifer_mm: f32,
    params: &GroundwaterParams,
) -> f32 {
    elevation - permeability * params.aquifer_thickness_m + saturated_thickness(aquifer_mm, params)
}

/// Darcy conductance of one hex edge per unit of saturated thickness and
/// head drop, in mm of the source cell's stock per day: the edge flux
/// `K × b × Δh / L` (m²/day per m of edge) times the edge length
/// `L/√3`, over the cell area, in mm. A pure geometry × `K` factor.
fn aquifer_edge_conductance(params: &GroundwaterParams) -> f32 {
    let edge_length = CELL_SPACING_M / 3.0_f32.sqrt();
    MM_PER_M * params.aquifer_conductivity_m_per_day * edge_length / (CELL_SPACING_M * CELL_AREA_M2)
}

/// Step 3, lateral flow of the aquifer (#107), Darcy-Dupuit in SI: from
/// each cell toward every neighbor of lower head, `K × b × Δh` through
/// the shared edge, with `b` the upstream saturated thickness. The
/// outflows of a cell are scaled down together if they would exceed its
/// stock, so no direction gets first claim. A cell with no stock (or the
/// one-ulp overdraft that scaling can leave) conducts nothing, Dupuit's
/// `b = 0`: without that guard a zero total over a negative stock made
/// the scale `−∞` and the flows NaN (measured, seed 42, first sweep). Reads only the snapshot,
/// parallel per source, same scatter → gather split as the root zone.
fn fill_aquifer_outflow(
    current: &HexGrid,
    snap_aquifer: &[f32],
    snap_perm: &[f32],
    snap_elev: &[f32],
    params: &GroundwaterParams,
    dir_out: &mut [Vec<f32>; 6],
) {
    let conductance = aquifer_edge_conductance(params);
    for_each_chunk_mut6(dir_out, |start, chunks| {
        for local in 0..chunks[0].len() {
            let i = start + local;
            let stock = snap_aquifer[i];
            if stock <= 0.0 {
                for chunk in chunks.iter_mut() {
                    chunk[local] = 0.0;
                }
                continue;
            }
            let head = aquifer_head(snap_elev[i], snap_perm[i], stock, params);
            let transmissive = conductance * saturated_thickness(stock, params);
            let neighbors = current.neighbor_indices_toric(i);
            let mut total = 0.0_f32;
            for (d, &j) in neighbors.iter().enumerate() {
                let drop = head - aquifer_head(snap_elev[j], snap_perm[j], snap_aquifer[j], params);
                let flow = if j != i && drop > 0.0 {
                    transmissive * drop
                } else {
                    0.0
                };
                chunks[d][local] = flow;
                total += flow;
            }
            if total > stock {
                let scale = stock / total;
                for chunk in chunks.iter_mut() {
                    chunk[local] *= scale;
                }
            }
        }
    });
}

/// Gather of step 3, then the aquifer's two exits to the surface, both
/// per-cell on the cell's own final stock: a spring where the lateral
/// inflow fills it past its storage (the water table reaches the
/// surface), then Maillet's baseflow.
fn gather_aquifer_transfers(
    current: &HexGrid,
    dir_out: &[Vec<f32>; 6],
    params: &GroundwaterParams,
    next: &mut HexGrid,
) {
    for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            let j = start + local;
            let self_loss = sum_dir_out(dir_out, j);
            let neighbors = current.neighbor_indices_toric(j);
            let mut gathered_in = 0.0_f32;
            for (d, &k) in neighbors.iter().enumerate() {
                if k == j {
                    continue;
                }
                gathered_in += dir_out[opposite_direction(d)][k];
            }
            cell.aquifer += gathered_in - self_loss;

            let storage = aquifer_storage(cell.permeability, params);
            if cell.aquifer > storage {
                let spring = cell.aquifer - storage;
                cell.aquifer = storage;
                cell.water_level += spring;
            }

            if params.baseflow_coef > 0.0 && cell.aquifer > 0.0 {
                let seepage = (params.baseflow_coef * cell.aquifer).min(cell.aquifer);
                cell.aquifer -= seepage;
                cell.water_level += seepage;
            }
        }
    });
}

#[must_use]
pub fn total_groundwater(grid: &HexGrid) -> f32 {
    grid.iter().map(|(_, cell)| cell.groundwater).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::total_humidity;
    use crate::hydro::total_water;

    fn total_moisture(grid: &HexGrid) -> f32 {
        total_water(grid) + total_humidity(grid) + total_groundwater(grid)
    }

    fn make_wet_grid() -> HexGrid {
        let mut grid = HexGrid::from_radius(3);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 3.0;
                cell.groundwater = 1.0;
                cell.permeability = 0.5;
            }
        }
        grid
    }

    #[test]
    fn moisture_conserved_with_groundwater() {
        let current = make_wet_grid();
        let mut next = current.clone();
        let params = GroundwaterParams::default();

        let before = total_moisture(&current);
        step_groundwater(&current, &mut next, &params);
        let after = total_moisture(&next);

        assert!(
            (before - after).abs() < 1e-2,
            "Conservation violated: {before} → {after}"
        );
    }

    #[test]
    fn infiltration_moves_water_underground() {
        let current = make_wet_grid();
        let mut next = current.clone();
        step_groundwater(&current, &mut next, &GroundwaterParams::default());

        let center = next.get(crate::coord::HexCoord::new(0, 0)).unwrap();
        assert!(center.water_level < 3.0);
        assert!(center.groundwater > 1.0);
    }

    #[test]
    fn resurgence_when_over_capacity() {
        let mut grid = HexGrid::from_radius(1);
        // Isolated test: fix max_capacity=5.0 to validate the resurgence
        // mechanism without depending on the default calibration (which
        // may shift).
        let params = GroundwaterParams {
            max_capacity: 5.0,
            ..GroundwaterParams::default()
        };
        // permeability=0.5, max_capacity=5.0 → capacity=2.5
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 0.0;
                cell.groundwater = 10.0; // well above the capacity (2.5)
                cell.permeability = 0.5;
            }
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let center = next.get(crate::coord::HexCoord::new(0, 0)).unwrap();
        assert!(center.water_level > 0.0, "Water should rise to the surface");
        let capacity = 0.5 * 5.0; // permeability * max_capacity
        assert!(
            center.groundwater <= capacity + 0.1,
            "Water table should not exceed local capacity"
        );
    }

    #[test]
    fn frozen_ground_blocks_infiltration() {
        // Below -2°C, the soil is fully frozen: meltwater stays on the
        // surface instead of filling the water table.
        let mut grid = HexGrid::from_radius(0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.water_level = 3.0;
            cell.groundwater = 0.0;
            cell.permeability = 0.5;
            cell.temperature = -3.0;
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &GroundwaterParams::default());

        let center = next.get(c0).unwrap();
        assert!(
            (center.groundwater - 0.0).abs() < 1e-6,
            "No infiltration under frozen soil: gw={}",
            center.groundwater
        );
        assert!(
            (center.water_level - 3.0).abs() < 1e-6,
            "Surface water stays intact: {}",
            center.water_level
        );
    }

    #[test]
    fn dry_grid_no_change() {
        let grid = HexGrid::from_radius(2);
        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &GroundwaterParams::default());

        for (coord, cell) in next.iter() {
            let orig = grid.get(*coord).unwrap();
            assert!((cell.groundwater - orig.groundwater).abs() < 1e-6);
        }
    }

    #[test]
    fn conservation_after_many_steps() {
        let mut current = make_wet_grid();
        let initial = total_moisture(&current);
        let params = GroundwaterParams::default();

        for _ in 0..100 {
            let mut next = current.clone();
            step_groundwater(&current, &mut next, &params);
            current = next;
        }

        let final_val = total_moisture(&current);
        assert!(
            (initial - final_val).abs() < 1e-1,
            "Conservation after 100 steps: {initial} → {final_val}"
        );
    }

    #[test]
    fn piezometric_gradient_flows_uphill_to_downhill() {
        // Mountain and plain both above field capacity, the plain
        // richer: the piezometric level (elev + gw) stays higher on the
        // mountain, so water must flow toward the plain even though the
        // plain holds more groundwater. This is the mechanism that feeds
        // natural springs.
        //
        // Both stocks are set above field capacity on purpose (#151):
        // below it the water is capillary-held and drains nowhere, which
        // is what `water_below_field_capacity_does_not_drain` pins. The
        // earlier version of this test ran on 1 mm of mountain water
        // table, i.e. exactly the leak #151 measured.
        let params = GroundwaterParams::default();
        let field_capacity = params.max_capacity * params.field_capacity_frac;

        let mut grid = HexGrid::from_radius(1);
        let mountain = crate::coord::HexCoord::new(0, 0);
        let plain_neighbors: Vec<_> = grid.neighbors(mountain).iter().map(|(c, _)| *c).collect();

        if let Some(cell) = grid.get_mut(mountain) {
            cell.elevation = 500.0;
            cell.groundwater = field_capacity + 10.0; // piezo = 575
            cell.water_level = 0.0;
            cell.permeability = 1.0;
        }
        for &coord in &plain_neighbors {
            if let Some(cell) = grid.get_mut(coord) {
                cell.elevation = 0.0;
                cell.groundwater = field_capacity + 30.0; // piezo = 95
                cell.water_level = 0.0;
                cell.permeability = 1.0;
            }
        }

        let before_mountain = grid.get(mountain).unwrap().groundwater;
        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);
        let after_mountain = next.get(mountain).unwrap().groundwater;

        assert!(
            after_mountain < before_mountain,
            "mountain water table (high piezo) should flow to the plain \
             (low piezo): before={before_mountain} after={after_mountain}"
        );
    }

    #[test]
    fn saturated_cell_resurges_excess_to_surface() {
        // If the water table exceeds the local capacity (perm ×
        // max_capacity), the excess must rise into water_level: this is
        // the "spring" mechanism.
        // Isolated test: max_capacity=5.0 to validate the mechanism
        // without depending on the default calibration. Percolation off:
        // with an aquifer below (#107) part of the excess would go down
        // instead of up, this test pins the spring.
        let params = GroundwaterParams {
            max_capacity: 5.0,
            percolation_rate: 0.0,
            ..GroundwaterParams::default()
        };
        let mut grid = HexGrid::from_radius(0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.permeability = 0.5; // capacity = 0.5 * 5 = 2.5
            cell.groundwater = 5.0; // well above 2.5
            cell.water_level = 0.0;
            cell.temperature = 10.0;
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let cell = next.get(c0).unwrap();
        assert!(
            cell.groundwater <= 2.5 + 1e-4,
            "gw should be brought back to capacity: {}",
            cell.groundwater
        );
        assert!(
            cell.water_level >= 2.4,
            "the excess (~2.5 units) should appear at the surface: {}",
            cell.water_level
        );
    }

    // --- Aquifer e2e-unit micro-tests (#107: percolation, Maillet recession) ---

    /// Params that isolate the vertical exchanges of one cell: no
    /// infiltration, no lateral flow, no baseflow unless asked.
    fn vertical_only(percolation_rate: f32, baseflow_coef: f32) -> GroundwaterParams {
        GroundwaterParams {
            saturated_conductivity_mm_per_day: 0.0,
            diffusion_rate: 0.0,
            baseflow_coef,
            percolation_rate,
            ..GroundwaterParams::default()
        }
    }

    /// One cell, radius 0: percolation and baseflow are cell-local, no
    /// transport (micro-test rule: no transport at radius 0).
    fn one_cell(groundwater: f32, aquifer: f32) -> HexGrid {
        let mut grid = HexGrid::from_radius(0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.groundwater = groundwater;
            cell.aquifer = aquifer;
            cell.water_level = 0.0;
            cell.permeability = 1.0;
            cell.temperature = 10.0;
        }
        grid
    }

    fn step_one(grid: &HexGrid, params: &GroundwaterParams) -> CellProperties {
        let mut next = grid.clone();
        step_groundwater(grid, &mut next, params);
        next.get(crate::coord::HexCoord::new(0, 0))
            .expect("radius 0 has its center")
            .clone()
    }

    /// Baseflow returns a fraction of the AQUIFER to the surface at each
    /// slice, without waiting for overflow, and leaves the root zone
    /// alone.
    #[test]
    fn baseflow_seeps_aquifer_to_surface() {
        let cell = step_one(&one_cell(20.0, 100.0), &vertical_only(0.0, 0.1));
        assert!(
            (cell.water_level - 10.0).abs() < 1e-3,
            "10% of the aquifer should resurge: water_level={}",
            cell.water_level
        );
        assert!(
            (cell.aquifer - 90.0).abs() < 1e-3,
            "aquifer={}",
            cell.aquifer
        );
        assert!(
            (cell.groundwater - 20.0).abs() < 1e-6,
            "baseflow must not touch the root zone: gw={}",
            cell.groundwater
        );
    }

    /// Maillet recession: without recharge, the aquifer decreases
    /// monotonically and strictly, `200 × 0.95ⁿ`. Pins the property that
    /// makes baseflow "sustained between two rains".
    #[test]
    fn baseflow_recedes_monotonically_without_recharge() {
        let params = vertical_only(0.0, 0.05);
        let mut current = one_cell(0.0, 200.0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        let mut prev = 200.0_f32;
        for _ in 0..20 {
            let mut next = current.clone();
            step_groundwater(&current, &mut next, &params);
            current = next;
            let aquifer = current.get(c0).unwrap().aquifer;
            assert!(aquifer < prev, "strict recession: {aquifer} !< {prev}");
            prev = aquifer;
        }
        let expected = 200.0 * 0.95_f32.powi(20);
        assert!(
            (prev - expected).abs() < 0.5,
            "Maillet recession: aquifer={prev}, expected≈{expected}"
        );
    }

    /// Only the water above field capacity percolates, at
    /// `percolation_rate` per day: 65 mm of capillary water stay in the
    /// root zone, half of the 20 mm excess goes down.
    #[test]
    fn percolation_drains_only_above_field_capacity() {
        let field_capacity = 100.0 * DEFAULT_FIELD_CAPACITY_FRAC;
        let cell = step_one(
            &one_cell(field_capacity + 20.0, 0.0),
            &vertical_only(0.5, 0.0),
        );
        assert!(
            (cell.aquifer - 10.0).abs() < 1e-3,
            "aquifer={}",
            cell.aquifer
        );
        assert!(
            (cell.groundwater - (field_capacity + 10.0)).abs() < 1e-3,
            "gw={}",
            cell.groundwater
        );
    }

    /// A root zone below field capacity recharges nothing: the plants'
    /// water is never handed to the aquifer.
    #[test]
    fn dry_root_zone_does_not_percolate() {
        let cell = step_one(&one_cell(30.0, 5.0), &vertical_only(0.5, 0.0));
        assert!(
            (cell.groundwater - 30.0).abs() < 1e-6,
            "gw={}",
            cell.groundwater
        );
        assert!(
            (cell.aquifer - 5.0).abs() < 1e-6,
            "aquifer={}",
            cell.aquifer
        );
    }

    /// A full aquifer takes nothing more: the water table has reached the
    /// root zone, the excess stays above (where resurgence and lateral
    /// flow handle it).
    #[test]
    fn full_aquifer_stops_percolation() {
        let capacity = aquifer_storage(1.0, &GroundwaterParams::default());
        let cell = step_one(&one_cell(95.0, capacity - 2.0), &vertical_only(0.5, 0.0));
        assert!(
            (cell.aquifer - capacity).abs() < 1e-3,
            "aquifer filled to its room only: {}",
            cell.aquifer
        );
        assert!(
            (cell.groundwater - 93.0).abs() < 1e-3,
            "gw={}",
            cell.groundwater
        );
    }

    /// The vertical exchanges are exact transfers: root zone + aquifer +
    /// surface is the same mass after a year of daily steps, bit for bit
    /// up to f32 rounding.
    #[test]
    fn percolation_and_baseflow_conserve_the_column() {
        let params = vertical_only(0.5, 0.02);
        let mut current = one_cell(90.0, 40.0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        for _ in 0..365 {
            let mut next = current.clone();
            step_groundwater(&current, &mut next, &params);
            current = next;
        }
        let c = current.get(c0).unwrap();
        let total = c.groundwater + c.aquifer + c.water_level;
        assert!((total - 130.0).abs() < 1e-3, "column total {total} != 130");
    }

    /// Params that isolate the aquifer's lateral flow: nothing enters or
    /// leaves it vertically.
    fn lateral_only() -> GroundwaterParams {
        GroundwaterParams {
            saturated_conductivity_mm_per_day: 0.0,
            diffusion_rate: 0.0,
            baseflow_coef: 0.0,
            percolation_rate: 0.0,
            ..GroundwaterParams::default()
        }
    }

    /// Radius 2 (transport, never radius 0): every cell at `elevation`,
    /// permeability 1, empty root zone, `aquifer` mm in its aquifer.
    fn flat_aquifer(elevation: f32, aquifer: f32) -> HexGrid {
        let mut grid = HexGrid::from_radius(2);
        for cell in grid.cells_slice_mut() {
            cell.elevation = elevation;
            cell.permeability = 1.0;
            cell.groundwater = 0.0;
            cell.aquifer = aquifer;
            cell.water_level = 0.0;
            cell.temperature = 10.0;
        }
        grid
    }

    fn aquifer_column(grid: &HexGrid) -> f32 {
        grid.cells_slice()
            .iter()
            .map(|c| c.aquifer + c.water_level + c.groundwater)
            .sum()
    }

    /// A raised cell drains its aquifer toward its lower neighbors, by
    /// Darcy through the edges, and the column is conserved.
    #[test]
    fn aquifer_flows_down_the_head_gradient() {
        let params = lateral_only();
        let mut grid = flat_aquifer(100.0, 300.0);
        let center = crate::coord::HexCoord::new(0, 0);
        grid.get_mut(center).unwrap().elevation = 105.0;
        let before = aquifer_column(&grid);

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let east = center + crate::coord::DIRECTIONS[0];
        assert!(
            next.get(center).unwrap().aquifer < 300.0,
            "the raised cell drains"
        );
        assert!(
            next.get(east).unwrap().aquifer > 300.0,
            "its neighbor receives"
        );
        let after = aquifer_column(&next);
        assert!(
            (after - before).abs() < 1e-2,
            "Darcy transfer is a transfer: {before} → {after}"
        );
    }

    /// The edge flux is `K × b × Δh / L × edge / area` in mm/day: 2 m of
    /// saturated thickness under a 5 m head drop, K = 10 m/day, out of a
    /// 130 m hex, is exactly that number, toward each of the 6 neighbors.
    #[test]
    fn aquifer_edge_flux_is_darcy_in_si() {
        let params = lateral_only();
        let thickness = 2.0_f32;
        let stock = thickness * MM_PER_M * DEFAULT_SPECIFIC_YIELD;
        let mut grid = flat_aquifer(100.0, stock);
        let center = crate::coord::HexCoord::new(0, 0);
        grid.get_mut(center).unwrap().elevation = 105.0;

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let edge = CELL_SPACING_M / 3.0_f32.sqrt();
        let per_edge = MM_PER_M * 10.0 * thickness * 5.0 / CELL_SPACING_M * edge / CELL_AREA_M2;
        let lost = stock - next.get(center).unwrap().aquifer;
        assert!(
            (lost - 6.0 * per_edge).abs() < 1e-3 * per_edge.max(1.0),
            "lost {lost} mm, Darcy says {}",
            6.0 * per_edge
        );
    }

    /// Dupuit: transmissivity is `K × b`, an empty aquifer carries
    /// nothing whatever the head drop below it.
    #[test]
    fn empty_aquifer_does_not_conduct() {
        let params = lateral_only();
        let mut grid = flat_aquifer(100.0, 0.0);
        let center = crate::coord::HexCoord::new(0, 0);
        grid.get_mut(center).unwrap().elevation = 200.0;

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        for cell in next.cells_slice() {
            assert!(
                cell.aquifer.abs() < 1e-9,
                "water out of nothing: {}",
                cell.aquifer
            );
        }
    }

    /// An overdrawn aquifer (the one-ulp negative stock that scaling the
    /// outflows can leave) on flat ground sends nothing and stays finite:
    /// the NaN of the first sweep.
    #[test]
    fn overdrawn_aquifer_stays_finite() {
        let params = lateral_only();
        let mut grid = flat_aquifer(100.0, 0.0);
        let center = crate::coord::HexCoord::new(0, 0);
        grid.get_mut(center).unwrap().aquifer = -1e-6;

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        for cell in next.cells_slice() {
            assert!(cell.aquifer.is_finite(), "aquifer went {}", cell.aquifer);
            assert!(
                cell.water_level.is_finite(),
                "surface went {}",
                cell.water_level
            );
        }
    }

    /// Where the lateral inflow fills a full aquifer past its storage,
    /// the water table reaches the ground and the excess comes out as a
    /// spring.
    #[test]
    fn full_aquifer_downhill_becomes_a_spring() {
        let params = lateral_only();
        let storage = aquifer_storage(1.0, &params);
        let mut grid = flat_aquifer(100.0, storage);
        let center = crate::coord::HexCoord::new(0, 0);
        grid.get_mut(center).unwrap().elevation = 90.0;

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let cell = next.get(center).unwrap();
        assert!(
            (cell.aquifer - storage).abs() < 1e-3,
            "aquifer stays full: {}",
            cell.aquifer
        );
        assert!(cell.water_level > 0.0, "the inflow surfaces as a spring");
    }

    /// `frozen_factor` is a linear ramp between -2°C and 0°C: no
    /// infiltration below -2°C, half rate at -1°C, full rate from 0°C up.
    /// Pins the explicit form against a rewrite back to `midpoint()`,
    /// which is what produced the unreadable version of #49.
    #[test]
    fn frozen_factor_is_a_linear_ramp() {
        // Radius 0 is legitimate here: infiltration is cell-local, the
        // ramp involves no transport between neighbours.
        let params = GroundwaterParams {
            saturated_conductivity_mm_per_day: 4.0,
            percolation_rate: 0.0,
            ..GroundwaterParams::default()
        };
        let c0 = crate::coord::HexCoord::new(0, 0);

        for (temperature, factor) in [(-3.0, 0.0), (-1.0, 0.5), (0.0, 1.0), (15.0, 1.0)] {
            let mut grid = HexGrid::from_radius(0);
            if let Some(cell) = grid.get_mut(c0) {
                cell.water_level = 10.0;
                cell.groundwater = 0.0;
                cell.permeability = 1.0;
                cell.temperature = temperature;
            }
            let mut next = grid.clone();
            step_groundwater(&grid, &mut next, &params);

            // Ks × permeability × factor, below the 10 mm ponded and the
            // 100 mm of room, so only the capacity binds.
            let expected: f32 = 4.0 * factor;
            let gw = next.get(c0).unwrap().groundwater;
            assert!(
                (gw - expected).abs() < 1e-4,
                "at {temperature}°C the ramp should give {factor}: gw={gw}, expected≈{expected}"
            );
        }
    }
    /// Capillary water does not drain (#151): with a water table below
    /// field capacity, the piezometric gradient of step 2 moves nothing,
    /// however steep the slope. Radius 2, not 0: lateral drainage is
    /// transport, and on the torus a radius-0 cell is its own neighbour
    /// six times over.
    #[test]
    fn water_below_field_capacity_does_not_drain() {
        let params = GroundwaterParams::default();
        let field_capacity = params.max_capacity * params.field_capacity_frac;

        let mut grid = HexGrid::from_radius(2);
        // A slope: elevation grows with q, so the piezometric gradient
        // has somewhere to push the water table towards.
        let mut elevation = 0.0_f32;
        for cell in grid.cells_slice_mut() {
            cell.elevation = elevation;
            elevation += 100.0;
            cell.permeability = 1.0;
            cell.water_level = 0.0;
            cell.temperature = 10.0;
            cell.groundwater = field_capacity - 5.0;
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        for (coord, cell) in next.iter() {
            let before = grid.get(*coord).unwrap().groundwater;
            assert!(
                (cell.groundwater - before).abs() < 1e-6,
                "held water moved: {before} → {}",
                cell.groundwater
            );
        }
    }

    /// Above field capacity, only the excess drains, and a cell never
    /// falls below field capacity through lateral flow alone (#151).
    #[test]
    fn only_the_excess_above_field_capacity_drains() {
        let params = GroundwaterParams::default();
        let field_capacity = params.max_capacity * params.field_capacity_frac;

        let mut grid = HexGrid::from_radius(2);
        let mut elevation = 0.0_f32;
        for cell in grid.cells_slice_mut() {
            cell.elevation = elevation;
            elevation += 100.0;
            cell.permeability = 1.0;
            cell.water_level = 0.0;
            cell.temperature = 10.0;
            cell.groundwater = field_capacity + 10.0;
        }

        let column = |g: &HexGrid| -> f32 {
            g.cells_slice()
                .iter()
                .map(|c| c.water_level + c.groundwater + c.aquifer)
                .sum()
        };
        let mut next = grid.clone();
        let total_before = column(&grid);
        step_groundwater(&grid, &mut next, &params);

        let mut drained = false;
        for cell in next.cells_slice() {
            assert!(
                cell.groundwater >= field_capacity - 1e-4,
                "drained below field capacity: {}",
                cell.groundwater
            );
            if cell.groundwater < field_capacity + 10.0 - 1e-4 {
                drained = true;
            }
        }
        assert!(drained, "the excess should drain somewhere");
        assert!(
            (column(&next) - total_before).abs() < 1e-2,
            "percolation, baseflow and step 2 are transfers, not sources: {total_before} → {}",
            column(&next)
        );
    }

    /// r250 perf effort, chunk C1 gate: pins that the two-phase outflow
    /// of step 2 still replays the historical serial loop's
    /// order-dependent budget depletion — the first downhill direction
    /// in `coord::DIRECTIONS` order gets first claim on the drainable
    /// surplus, a later one only the leftover, NOT a globally
    /// rebalanced share (unlike hydro's MFD). Two downhill neighbors,
    /// each individually "wanting" more than half the drainable
    /// surplus:
    ///   `field_capacity = perm(1) * max_capacity(100) * frac(0.5) = 50`
    ///   `drainable = groundwater(60) - field_capacity(50) = 10`
    ///   `base_rate = diffusion_rate(0.6) / 6 = 0.1`
    ///   dir 0: `diff = 80` -> desired `8.0`, `min(8.0, 10) = 8.0`,
    ///           remaining drainable `10 - 8 = 2`
    ///   dir 1: `diff = 80` -> desired `8.0`, but `min(8.0, 2) = 2.0`
    /// If this test breaks by dir 0 and dir 1 receiving equal shares
    /// (e.g. 5.0 each), the two-phase split has been "fixed" into a
    /// hydro-like global rebalance, which is NOT what the historical
    /// loop did and would silently change the physics.
    #[test]
    fn groundwater_two_phase_outflow_depletes_budget_in_direction_order() {
        let params = GroundwaterParams {
            saturated_conductivity_mm_per_day: 0.0,
            diffusion_rate: 0.6,
            max_capacity: 100.0,
            baseflow_coef: 0.0,
            field_capacity_frac: 0.5,
            percolation_rate: 0.0,
            aquifer_thickness_m: DEFAULT_AQUIFER_THICKNESS_M,
            specific_yield: DEFAULT_SPECIFIC_YIELD,
            aquifer_conductivity_m_per_day: DEFAULT_AQUIFER_CONDUCTIVITY_M_PER_DAY,
        };
        let mut grid = HexGrid::from_radius(2);
        let center = crate::coord::HexCoord::new(0, 0);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                c.permeability = 1.0;
                c.groundwater = 0.0;
                c.elevation = 2000.0; // uphill of center by default: inert
                c.water_level = 0.0;
                c.temperature = 10.0;
            }
        }
        if let Some(c) = grid.get_mut(center) {
            c.elevation = 1000.0;
            c.groundwater = 60.0; // drainable above field_capacity(50) = 10
        }
        let n0 = center + crate::coord::DIRECTIONS[0];
        let n1 = center + crate::coord::DIRECTIONS[1];
        grid.get_mut(n0).unwrap().elevation = 980.0; // diff = (1000+60)-(980+0) = 80
        grid.get_mut(n1).unwrap().elevation = 980.0; // same diff = 80

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let gw_center = next.get(center).unwrap().groundwater;
        let gw_n0 = next.get(n0).unwrap().groundwater;
        let gw_n1 = next.get(n1).unwrap().groundwater;
        assert!(
            (gw_n0 - 8.0).abs() < 1e-3,
            "first downhill direction should get its full desired transfer: {gw_n0}"
        );
        assert!(
            (gw_n1 - 2.0).abs() < 1e-3,
            "second downhill direction should get only the leftover budget: {gw_n1}"
        );
        assert!(
            (gw_center - 50.0).abs() < 1e-3,
            "center should drain exactly down to field capacity: {gw_center}"
        );
    }
}
