//! Synoptic atmospheric dynamics, prognostic core (Phase 1 of the
//! synoptic-dynamics design: the shallow-water pressure/wind core itself,
//! before system drift and precip coupling in later phases).
//!
//! **Single-layer shallow-water, f-plane, periodic torus** model, thermally
//! forced. It makes pressure systems (geostrophic lows) emerge from the
//! existing temperature field, with no scripting and no hand-calibrated
//! dimensionless parameter. It's the smallest model that produces a coherent
//! synoptic wind (isobars + geostrophy) instead of decorative noise.
//!
//! State (SI) per cell:
//! - `h` (m): geopotential height, proxy for synoptic pressure (`p ≈ ρ g h`).
//! - `u`, `v` (m/s): prognostic synoptic wind.
//!
//! Equations (tangent plane, single layer, linearized around a mean zonal
//! flow `U`):
//! ```text
//! ∂u/∂t + U ∂ₓu = +f v − g ∂ₓh − r u + ν ∇²u
//! ∂v/∂t + U ∂ₓv = −f u − g ∂_yh − r v + ν ∇²v
//! ∂h/∂t + U ∂ₓh = −H ∇·u + Q(T) − (h − h₀)/τ
//! ```
//! - `f = 2Ω sin φ`: Coriolis, giving the geostrophic balance
//!   `u_g = (g/f) k×∇h`, the wind circulates around L/H.
//! - `−g ∇h`: pressure gradient force.
//! - `−r u`: Rayleigh friction (turbulent dissipation, return to
//!   equilibrium).
//! - `Q(T) ∝ −(T − T̄)`: thermal forcing, a warm cell digs a low. **This is
//!   where temperature drives the weather.**
//! - `−(h − h₀)/τ`: gentle relaxation towards the base state (prevents
//!   drift).
//! - `−U ∂ₓ(·)` (Phase 2): advection by the mean zonal flow. This is the
//!   term that makes systems **travel across** the map; without it, on an
//!   f-plane, eddies adjust geostrophically and then stay put.
//!   `(u, v)` is the *perturbation* around the base state `(U, 0)`; the base
//!   state's `f k̂×U` term is balanced by an implicit meridional slope of
//!   `h` (impossible to represent periodically on the torus), so it does not
//!   appear (standard linearization, Vallis 2017 §5.7).
//!
//! Time scheme: subsampled **forward-backward** (N substeps / hour),
//! validated at the Phase 0 spike (`tests/diag_synoptic_spike.rs`), stable,
//! ~12 substeps suffice at the operating point (H tuned for `L_d ≈ 10`
//! cells, `c ≈ 1 m/s`). Zero-malloc per tick (reused scratch), deterministic.
//!
//! **Integration status**: the state is evolved and exposed, wired into the
//! production wind via the `synoptic.enabled` param (hardcoded ON by default
//! since #108, cf. `simulation.rs`). Not coupled to precipitation (Phase 3).

use serde::{Deserialize, Serialize};

use crate::coord::hex_direction_to_world;
use crate::grid::HexGrid;
use crate::wind::{WindField, WindVec};

/// Gravity (m/s²).
const GRAVITY: f32 = 9.81;
/// Angular velocity of the Earth (rad/s): `f = 2Ω sin φ` (Coriolis f = 2Ω sin φ).
const EARTH_OMEGA: f32 = 7.292e-5;

/// Center-to-center distance between two neighboring cells (m). A cell is
/// ~1.7 ha = 1.69e4 m²; regular hexagon `A = (√3/2) d²` ⇒ `d = √(2A/√3)`.
/// This is the first time the core encodes the physical size of a cell:
/// the rest of the engine works in unit cells, synoptic dynamics needs real
/// meters for SI units (Coriolis, waves, deformation).
///
/// Went from 1074.569 m (1 km²/hex) to 130 m on `feat/dem-terrain-validation`
/// (validated against a real SRTM survey over Bégude/Dieulefit/Saoû/Crest:
/// at 1074.569 m no real relief was distinguishable, Forêt de Saoû invisible
/// even from afar; at 130 m it becomes recognizable again). Modules that
/// correctly derive from this constant (automatic recalibration):
/// `temperature.rs` (slope, rain-shadow ray-marching), `atmosphere.rs`
/// (orographic uplift), `hydro.rs`/`wind.rs` (`STEEP_SLOPE_GRADE`).
/// The frontend (`frontend/main.js`) receives it via the `meta` command
/// (`cell_spacing_m`), not through the snapshot wire protocol; this is
/// deliberate (fixed value for the process's lifetime, no need to repeat
/// it every frame).
pub const CELL_SPACING_M: f32 = 130.0;

/// Area of a hexagonal cell (m²): `A = (√3/2)·d²` with `d =
/// CELL_SPACING_M` (√3/2 ≈ 0.866 025 4, regular hexagon, same formula as
/// the `CELL_SPACING_M` doc inverted). ≈ 14 637 m² at 130 m. Used for
/// water depth conversions (mm ≡ kg/m² over the column) → SI volume (m³),
/// notably the stream power discharge `Q` (`erosion.rs`).
pub const CELL_AREA_M2: f32 = 0.866_025_4 * CELL_SPACING_M * CELL_SPACING_M;

/// Slope (m/m) empirically calibrated as the threshold "elevation delta
/// between neighboring hexes steep enough to saturate an effect", shared by
/// several modules that compare a raw elevation delta (m, not yet divided
/// by a distance) to a threshold: `hydro.rs` (mobilization of trapped
/// water), `wind.rs` (deflection/catabatic acceleration of wind by relief).
/// Historical value: 50 m of delta over the original ~1074.569 m of
/// `CELL_SPACING_M` ≈ 4.65%. Multiply by `CELL_SPACING_M` to get the
/// threshold in meters at the current resolution; otherwise the threshold
/// ends up silently miscalibrated on every change of the engine's
/// resolution (cf. `feat/dem-terrain-validation`, where `wind.rs` had this
/// threshold hardcoded).
pub const STEEP_SLOPE_GRADE: f32 = 50.0 / 1074.569;

/// Conversion factor from synoptic wind → `WindVec`. Consumers of the wind
/// field interpret `magnitude × 10 = m/s` (`wind.rs` convention, #33). The
/// synoptic wind is in SI m/s, so we divide by 10 to express it in the
/// `WindField` unit.
const MS_TO_WINDVEC: f32 = 0.1;

/// **Calibration** spacing of the synoptic solver (m). All the default
/// quantities (`L_d` = 10 cells, ν = 40 m²/s, `thermal_coupling`) were
/// measured and validated at this spacing (Phase 0 spike, Phase 4
/// calibration). [`SynopticParams::for_spacing`] re-derives the defaults
/// for any other spacing, preserving the absolute PHYSICAL quantities.
pub const SYNOPTIC_REFERENCE_SPACING_M: f32 = 1074.569;

/// Serde default for `cell_spacing_m`: params serialized before this field
/// (older checkpoints) integrated on the fine grid.
fn fine_spacing() -> f32 {
    CELL_SPACING_M
}

/// Physical parameters of the synoptic model. **Strict SI** (Vallis 2017,
/// Holton & Hakim 2013).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynopticParams {
    /// World seed, seed-derived initial perturbation (determinism).
    pub seed: u32,
    /// Latitude of the single-latitude terrarium (°) → Coriolis `f`. Mirrors
    /// `TemperatureParams::latitude_deg` for "one single world" consistency.
    pub latitude_deg: f32,
    /// Spacing (m) of the grid on which THIS parameter set integrates. The
    /// solver is parametric in spacing: gradients, laplacian, upwind
    /// advection and CFL all derive from it, no more global constant
    /// hardcoded in the sweeps. This is what allows solving the synoptics
    /// on a grid at its natural scale (~km) rather than on the fine terrain
    /// grid (chantier #88: the CFL in 1/Δx cost 163 substeps/h at
    /// 130 m versus 20 at the calibration spacing).
    #[serde(default = "fine_spacing")]
    pub cell_spacing_m: f32,
    /// Target Rossby deformation radius, in **cells**. This is THE design
    /// lever (Phase 0 spike): it sets the size of pressure systems, hence
    /// the scale depth `H = (f · L_d)² / g` and the wave speed `c = √(gH)`.
    /// ~10 cells ⇒ systems that fit within the map (66 km at r30).
    ///
    /// **`L_d = deformation_radius_cells · cell_spacing_m` must stay on the
    /// order of its reference value (~10.7 km) regardless of spacing**: `H`
    /// collapses as `spacing²` (via `L_d`), and below a certain `H` the
    /// continuity term `−H·∇·u` becomes too weak to contain the geostrophic
    /// feedback → `h` diverges within a few hours (`NaN` measured at 130 m
    /// with 10 fixed cells, cf. feat/dem-terrain-validation). `for_spacing`
    /// therefore recomputes it from the spacing to preserve the absolute
    /// `L_d`, not a fixed cell count. An explicit override (tests) remains
    /// expressed in cells and follows the params' `cell_spacing_m`, which is
    /// intended when the goal is just "negligible" (`0.1` ⇒ frozen waves)
    /// rather than a precise system size.
    pub deformation_radius_cells: f32,
    /// Rayleigh friction time (days) → `r = 1/(friction_days · 86400)`.
    pub friction_days: f32,
    /// Relaxation time towards the base state (days) → `τ`. Prevents drift.
    pub relax_days: f32,
    /// Numerical diffusion `ν` (m²/s), damps the checkerboard (spike §7 risk
    /// #2).
    ///
    /// The discrete laplacian `lap = 2/(3·cell_spacing_m²)` (`substep`)
    /// means that the real effect of `ν` is `ν/spacing²`: an absolute `ν`
    /// implicitly calibrated at the reference spacing becomes ~68x stronger
    /// at 130 m even though the cell hasn't changed physical size, enough
    /// to break the forcing/relaxation balance of a thermal `L` (measured
    /// on `anomaly_mode_extinguishes_standing_forcing`). `for_spacing`
    /// therefore rescales by `spacing²` to preserve the same *relative*
    /// checkerboard damping regardless of resolution.
    pub viscosity: f32,
    /// Thermal coupling `k` (s⁻¹·K⁻¹, **relative to `depth()`**):
    /// `Q = −k·depth()·(T − T̄)`. A warm anomaly digs a low `L`.
    ///
    /// Relative to `depth()`, like `eps` in `SynopticState::new` (1% of
    /// `depth()`), because `h` fluctuates on the fictitious scale `depth()`,
    /// itself derived from `deformation_radius_cells · CELL_SPACING_M`
    /// (#DEM-terrain-validation). An absolute coupling (m·s⁻¹·K⁻¹) stays
    /// calibrated for the single `depth()` value it was measured at:
    /// reducing `CELL_SPACING_M` reduces `depth()` by `CELL_SPACING_M²`
    /// (via `L_d`) without reducing an absolute forcing, which then becomes
    /// huge compared to `depth()` → `h` diverges within a few hours (NaN
    /// observed at 130 m before this fix).
    ///
    /// **Transition to SI (finalized by a later overhaul)**: `k` here is
    /// the linearized slope of the column's thermal expansion. Its value is
    /// set to a plausible physical scale (tens of meters of digging for a
    /// ~10 K anomaly over `τ`, at the reference `depth()`, 1074.569 m,
    /// 10 cells, 44.5°N), not derived from a full radiative flux; to be
    /// refined in Phase 4 (the project tolerates a transitional mix of
    /// units mid-refactor as long as it's flagged in comments with a
    /// planned follow-up, which is what this comment does).
    pub thermal_coupling: f32,
    /// Mean zonal flow `U` (m/s, +East). Linear advection `−U ∂ₓ(·)` of
    /// `h`, `u`, `v`: the mechanism that makes systems travel (Phase 2).
    /// Physical successor to the old scripted `west_bias` of the historical
    /// wind (0.32 `WindVec` ≈ 3.2 m/s, calibrated #176, removed in #108):
    /// same role as a dominant westerly wind, but it also advects pressure
    /// systems instead of only pushing humidity. Composed in
    /// `write_base_wind`.
    pub mean_flow_ms: f32,
    /// Time constant (days) of the **per-cell temperature rolling average**
    /// used as the reference for thermal forcing (Phase 3).
    /// - `0.0` (default): historical Phase 1 forcing, `Q ∝ −(T − T̄_spatial)`;
    ///   the *permanent* contrasts (altitude!) dig *permanent* systems: the
    ///   anticyclone anchors on the cold massif and its divergence (measured
    ///   −1.6e-3 s⁻¹, i.e. w ≈ −2.4 m/s) crushes everything else; ablation
    ///   Phase 3, r60 seed 42.
    /// - `> 0`: the forcing only sees the **anomaly** `T − ⟨T⟩_rolling`
    ///   (spatially centered); the stationary response dies out over ~τ,
    ///   only the fluctuations remain (diurnal land/lake differential,
    ///   weather passages), which create **transient** systems advected by
    ///   `U`: weather that travels. Classic anomaly/mean-state
    ///   decomposition (the mean state is already balanced, only the
    ///   deviation drives the dynamics; Holton §6, quasi-geostrophy on a
    ///   base state).
    pub thermal_anomaly_days: f32,
    /// Forward-backward substeps per hour. 12 = comfortable margin at the
    /// operating point (spike: 4 suffice for CFL, 12 for accuracy). Floor:
    /// `step_hour` adds more if the combined waves + advection CFL requires
    /// it (cf. `substeps_per_hour`).
    pub substeps: u32,
}

impl Default for SynopticParams {
    /// Defaults for integration on the **fine grid** (`CELL_SPACING_M`),
    /// the historical behavior. Production now integrates on the dedicated
    /// synoptic grid via [`Self::for_spacing`].
    fn default() -> Self {
        Self::for_spacing(CELL_SPACING_M)
    }
}

impl SynopticParams {
    /// Parameter set for a grid with spacing `spacing_m`: same **absolute
    /// physical** quantities as the reference calibration
    /// (`SYNOPTIC_REFERENCE_SPACING_M`), regardless of spacing.
    /// - absolute `L_d` ≈ 10.7 km (10 reference cells); otherwise `depth()`
    ///   collapses as `spacing²` and `h` diverges (NaN measured at 130 m,
    ///   feat/dem-terrain-validation);
    /// - `ν` rescaled as `spacing²`; the discrete laplacian is `2/(3·Δx²)`,
    ///   an absolute ν would change relative strength with the cell size.
    #[must_use]
    pub fn for_spacing(spacing_m: f32) -> Self {
        Self {
            seed: 42,
            latitude_deg: 44.5,
            cell_spacing_m: spacing_m,
            deformation_radius_cells: 10.0 * SYNOPTIC_REFERENCE_SPACING_M / spacing_m,
            friction_days: 3.0,
            relax_days: 10.0,
            viscosity: 40.0 * (spacing_m / SYNOPTIC_REFERENCE_SPACING_M).powi(2),
            // 5.0e-6 (m·s⁻¹·K⁻¹, old absolute) / reference depth() (0.122939 m,
            // at deformation_radius_cells=10, reference spacing, lat=44.5°)
            // → same absolute forcing as before at the reference config, but
            // correctly proportional to depth() if it changes.
            thermal_coupling: 4.0669e-5,
            mean_flow_ms: 3.0,
            thermal_anomaly_days: 10.0,
            substeps: 12,
        }
    }

    /// Coriolis `f = 2Ω sin φ` (1/s).
    #[must_use]
    pub fn coriolis(&self) -> f32 {
        2.0 * EARTH_OMEGA * self.latitude_deg.to_radians().sin()
    }

    /// Scale depth `H = (f · L_d)² / g` (m), tuned to the target deformation
    /// radius. `L_d = deformation_radius_cells · cell_spacing_m`.
    #[must_use]
    pub fn depth(&self) -> f32 {
        let ld = self.deformation_radius_cells * self.cell_spacing_m;
        (self.coriolis() * ld).powi(2) / GRAVITY
    }

    /// Gravity wave speed `c = √(gH)` (m/s).
    #[must_use]
    pub fn wave_speed(&self) -> f32 {
        (GRAVITY * self.depth()).sqrt()
    }

    fn friction(&self) -> f32 {
        1.0 / (self.friction_days * 86_400.0)
    }

    fn relax_tau(&self) -> f32 {
        self.relax_days * 86_400.0
    }

    /// Maximum Courant number per substep. Forward-backward is stable up to
    /// a Courant number ≈ 1.5 measured at the Phase 0 spike; the hexagonal
    /// upwind scheme is monotone as long as `dt · Σ outgoing weights ≤ 1`,
    /// i.e. an advective Courant number ≤ 0.75. 0.7 = below both limits,
    /// with margin.
    const CFL_MAX: f32 = 0.7;

    /// Effective substeps for one hour: at least `substeps`, more if the
    /// combined signal speed (gravity waves `c` + advection `|U|`) requires
    /// it: `(c + |U|) · dt / d ≤ CFL_MAX`. This is the
    /// Courant-Friedrichs-Lewy stability criterion, not an arbitrary
    /// safeguard: a larger step makes the scheme diverge. At the default
    /// operating point (`c ≈ 1.1`, `U = 3`): 20 substeps.
    #[must_use]
    pub fn substeps_per_hour(&self) -> u32 {
        let signal = self.wave_speed() + self.mean_flow_ms.abs();
        let max_dt = Self::CFL_MAX * self.cell_spacing_m / signal.max(f32::EPSILON);
        // Smallest n ≥ substeps such that 3600/n ≤ max_dt, without an
        // f32→u32 cast: n is bounded (signal ≤ 300 m/s ⇒ n ≤ ~1600), the
        // loop is trivial.
        let mut n = self.substeps.max(1);
        while 3600.0 / f32::from(u16::try_from(n).unwrap_or(u16::MAX)) > max_dt {
            n += 1;
        }
        n
    }
}

/// Upwind advection weights per hex direction (s⁻¹ per unit of Δφ) for a
/// mean zonal flow `U`: finite-volume flux form `−∇·(Uφ)` on the hexagon,
/// upwind donor per face. Since `Σ_k n̂_k = 0`, it can be rewritten as
/// `dφ_i/dt = Σ_k w_k (φ_j − φ_i)` with `w_k = (2/3d)·max(−U·n̂ₖ, 0) ≥ 0`:
/// only the *incoming* faces contribute. Conservative (face fluxes
/// telescope on the torus → ⟨φ⟩ exactly conserved), monotone, and exact
/// for a linear field (the discrete operator reproduces `−U ∂ₓφ` exactly).
fn advection_weights(mean_flow_ms: f32, spacing_m: f32, dirs: &[(f32, f32); 6]) -> [f32; 6] {
    let geom = 2.0 / (3.0 * spacing_m);
    let mut w = [0.0; 6];
    for (wk, &(dx, _)) in w.iter_mut().zip(dirs) {
        *wk = geom * (-mean_flow_ms * dx).max(0.0);
    }
    w
}

/// World unit vectors of the 6 hex directions (constants).
fn direction_units() -> [(f32, f32); 6] {
    let mut d = [(0.0, 0.0); 6];
    for (i, slot) in d.iter_mut().enumerate() {
        *slot = hex_direction_to_world(i);
    }
    d
}

/// Small deterministic hash seed×index → initial perturbation in [-1, 1].
/// Breaks the perfect symmetry of the rest state without a runtime RNG.
fn seed_perturbation(seed: u32, index: usize) -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9).wrapping_add(
        u32::try_from(index & 0xFFFF_FFFF)
            .unwrap_or(0)
            .wrapping_mul(0x8542_5D1B),
    );
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    // u32 → [0,1) via the 24 high-order bits (f32 mantissa), then [-1,1].
    let unit = f32::from(u16::try_from(x >> 16).unwrap_or(0)) / f32::from(u16::MAX);
    2.0 * unit - 1.0
}

/// Double-buffered synoptic state + scratch (zero-malloc per tick).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynopticState {
    /// Geopotential height h (m), pressure proxy.
    h: Vec<f32>,
    /// Synoptic wind (m/s).
    u: Vec<f32>,
    v: Vec<f32>,
    /// Base height `h₀ = H` (relaxation + continuity).
    depth: f32,
    // Scratch reused every substep.
    div: Vec<f32>,
    du: Vec<f32>,
    dv: Vec<f32>,
    forcing: Vec<f32>,
    /// Advection tendency of h by the mean flow (`−U ∂ₓh`, m/s).
    adv: Vec<f32>,
    /// Per-cell rolling average temperature (°C), reference for the forcing
    /// in anomaly mode (`thermal_anomaly_days > 0`). Empty while the mode
    /// is inactive; seeded on the first step with the current T field.
    t_ref: Vec<f32>,
}

impl SynopticState {
    /// Initial state: rest (u=v=0), `h = H + ε(seed)` (small seed-derived
    /// perturbation). Deterministic.
    #[must_use]
    pub fn new(n: usize, params: &SynopticParams) -> Self {
        let depth = params.depth();
        let eps = 1.0e-2 * depth;
        let h: Vec<f32> = (0..n)
            .map(|i| depth + eps * seed_perturbation(params.seed, i))
            .collect();
        Self {
            h,
            u: vec![0.0; n],
            v: vec![0.0; n],
            depth,
            div: vec![0.0; n],
            du: vec![0.0; n],
            dv: vec![0.0; n],
            forcing: vec![0.0; n],
            adv: vec![0.0; n],
            t_ref: Vec::new(),
        }
    }

    /// Geopotential height (m), indexed by `cell_index`.
    #[must_use]
    pub fn height(&self) -> &[f32] {
        &self.h
    }

    /// Perturbed synoptic wind (m/s), `(u, v)` components indexed by
    /// `cell_index`. This is the perturbation around the mean flow: the
    /// total synoptic wind is `(u + mean_flow_ms, v)`, which is what
    /// `write_base_wind` composes for the wind pipeline.
    #[must_use]
    pub fn velocity(&self) -> (&[f32], &[f32]) {
        (&self.u, &self.v)
    }

    /// Maximum speed of the synoptic wind (m/s), stability diagnostic.
    #[must_use]
    pub fn max_speed(&self) -> f32 {
        self.u
            .iter()
            .zip(&self.v)
            .map(|(a, b)| a.hypot(*b))
            .fold(0.0, f32::max)
    }

    /// Mean height ⟨h⟩ (m), conservation invariant outside forcing.
    #[must_use]
    pub fn mean_height(&self) -> f32 {
        let n = f32::from(u16::try_from(self.h.len()).unwrap_or(u16::MAX));
        self.h.iter().sum::<f32>() / n
    }

    /// Writes the **total** synoptic wind `(u + U, v)` into a `WindField`
    /// (`WindVec` unit, `magnitude × 10 = m/s`). The mean zonal flow `U` is
    /// part of the real wind (it's the dominant westerly wind, successor to
    /// the old `west_bias` removed in #108); the perturbation adds the
    /// circulation around L/H systems on top. Base onto which `wind.rs`
    /// adds thermal breeze and orographic deflection.
    pub fn write_base_wind(&self, params: &SynopticParams, out: &mut WindField) {
        out.resize(self.u.len(), WindVec::default());
        let u0 = params.mean_flow_ms;
        for (o, (&u, &v)) in out.iter_mut().zip(self.u.iter().zip(&self.v)) {
            *o = WindVec {
                x: (u + u0) * MS_TO_WINDVEC,
                y: v * MS_TO_WINDVEC,
            };
        }
    }

    /// Advances the state by one hour (3600 s) in `params.substeps`
    /// forward-backward substeps. The thermal forcing `Q ∝ −(T − T̄)` is
    /// computed from `grid`'s temperature field. Internal double-buffer,
    /// deterministic.
    pub fn step_hour(&mut self, grid: &HexGrid, params: &SynopticParams) {
        debug_assert_eq!(
            self.h.len(),
            grid.len(),
            "synoptic state misaligned with grid"
        );
        self.fill_thermal_forcing(grid, params);
        let dirs = direction_units();
        let substeps = params.substeps_per_hour();
        let dt = 3600.0 / f32::from(u16::try_from(substeps).unwrap_or(1));
        let adv_w = advection_weights(params.mean_flow_ms, params.cell_spacing_m, &dirs);
        for _ in 0..substeps {
            self.substep(grid, params, &dirs, &adv_w, dt);
        }
    }

    /// Diagnostic summary (`synoptic` block of the WS `diag` command).
    /// Strict SI.
    #[must_use]
    pub fn stats(&self, params: &SynopticParams, enabled: bool) -> SynopticStats {
        let mean = self.mean_height();
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &h in &self.h {
            lo = lo.min(h);
            hi = hi.max(h);
        }
        SynopticStats {
            enabled,
            mean_flow_ms: params.mean_flow_ms,
            mean_h_m: mean,
            anomaly_min_m: lo - mean,
            anomaly_max_m: hi - mean,
            max_wind_ms: self.max_speed(),
        }
    }

    /// Thermal forcing. Two modes (cf. `SynopticParams::thermal_anomaly_days`):
    /// - historical: `Q_i = −k (T_i − T̄_spatial)`, a warm anomaly is a
    ///   negative source for h → digs a low. Permanent contrasts (altitude)
    ///   create permanent systems.
    /// - anomaly (`τ > 0`): `Q_i = −k (a_i − ā)` with `a_i = T_i − ⟨T_i⟩_τ`
    ///   (deviation from the cell's rolling average, spatially centered to
    ///   keep ⟨Q⟩ = 0). The stationary component of the forcing dies out
    ///   over ~τ → only transient systems remain, advected by U.
    fn fill_thermal_forcing(&mut self, grid: &HexGrid, params: &SynopticParams) {
        let cells = grid.cells_slice();
        let n = f32::from(u16::try_from(cells.len()).unwrap_or(u16::MAX));
        if params.thermal_anomaly_days > 0.0 {
            // Deterministic seeding: the first reference is the current T
            // field (zero anomaly at first, it builds up afterward).
            if self.t_ref.len() != cells.len() {
                self.t_ref = cells.iter().map(|c| c.temperature).collect();
            }
            // Anomaly measured against the reference *before* updating it
            // (first-order high-pass filter, one step = 1 h).
            let mean_anom = cells
                .iter()
                .zip(&self.t_ref)
                .map(|(c, r)| c.temperature - r)
                .sum::<f32>()
                / n;
            for ((f, cell), r) in self.forcing.iter_mut().zip(cells).zip(&self.t_ref) {
                *f = -params.thermal_coupling * self.depth * ((cell.temperature - r) - mean_anom);
            }
            let alpha = 3600.0 / (params.thermal_anomaly_days * 86_400.0);
            for (r, cell) in self.t_ref.iter_mut().zip(cells) {
                *r += alpha * (cell.temperature - *r);
            }
        } else {
            let mean_t = cells.iter().map(|c| c.temperature).sum::<f32>() / n;
            for (f, cell) in self.forcing.iter_mut().zip(cells) {
                *f = -params.thermal_coupling * self.depth * (cell.temperature - mean_t);
            }
        }
    }

    /// One forward-backward substep of `dt` seconds.
    fn substep(
        &mut self,
        grid: &HexGrid,
        params: &SynopticParams,
        dirs: &[(f32, f32); 6],
        adv_w: &[f32; 6],
        dt: f32,
    ) {
        let (f, r, nu, h0, tau) = (
            params.coriolis(),
            params.friction(),
            params.viscosity,
            self.depth,
            params.relax_tau(),
        );
        let inv = 1.0 / (3.0 * params.cell_spacing_m);
        let lap = 2.0 / (3.0 * params.cell_spacing_m * params.cell_spacing_m);

        // 1) ∇h, then velocity tendencies (reads the old state → double-buffer).
        for i in 0..self.h.len() {
            let nbr = grid.neighbor_indices_toric(i);
            let (hc, uc, vc) = (self.h[i], self.u[i], self.v[i]);
            let (mut gx, mut gy) = (0.0, 0.0);
            let (mut lu, mut lv) = (0.0, 0.0);
            let (mut au, mut av) = (0.0, 0.0);
            for (k, &(dx, dy)) in dirs.iter().enumerate() {
                let j = nbr[k];
                let dh = self.h[j] - hc;
                gx += dh * dx;
                gy += dh * dy;
                lu += self.u[j] - uc;
                lv += self.v[j] - vc;
                // Upwind advection by the mean flow: −U ∂ₓ(u, v).
                au += adv_w[k] * (self.u[j] - uc);
                av += adv_w[k] * (self.v[j] - vc);
            }
            self.du[i] = f * vc - GRAVITY * gx * inv - r * uc + nu * lu * lap + au;
            self.dv[i] = -f * uc - GRAVITY * gy * inv - r * vc + nu * lv * lap + av;
        }
        for (((uu, vv), &du), &dv) in self
            .u
            .iter_mut()
            .zip(&mut self.v)
            .zip(&self.du)
            .zip(&self.dv)
        {
            *uu += dt * du;
            *vv += dt * dv;
        }

        // 2) ∇·u with the fresh velocity + advection of h (read from the
        //    old h state), then continuity (the "backward" step).
        for i in 0..self.h.len() {
            let nbr = grid.neighbor_indices_toric(i);
            let (hc, uc, vc) = (self.h[i], self.u[i], self.v[i]);
            let mut d = 0.0;
            let mut ah = 0.0;
            for (k, &(dx, dy)) in dirs.iter().enumerate() {
                let j = nbr[k];
                d += (self.u[j] - uc) * dx + (self.v[j] - vc) * dy;
                ah += adv_w[k] * (self.h[j] - hc);
            }
            self.div[i] = d * inv;
            self.adv[i] = ah;
        }
        for (((hh, &d), &src), &adv) in self
            .h
            .iter_mut()
            .zip(&self.div)
            .zip(&self.forcing)
            .zip(&self.adv)
        {
            let relax = (*hh - h0) / tau;
            *hh += dt * (-h0 * d + adv + src - relax);
        }
    }
}

/// Diagnostic summary of the synoptic state, exposed by the WS `diag`
/// command. All quantities in SI (m, m/s).
#[derive(Debug, Serialize)]
pub struct SynopticStats {
    /// Does the dynamics drive the wind (`synoptic.enabled` param,
    /// hardcoded ON by default)?
    pub enabled: bool,
    /// Mean zonal flow `U` (m/s, +East).
    pub mean_flow_ms: f32,
    /// Mean geopotential height ⟨h⟩ (m).
    pub mean_h_m: f32,
    /// Extreme negative anomaly `h − ⟨h⟩` (m), the deepest low's trough.
    pub anomaly_min_m: f32,
    /// Extreme positive anomaly (m), the strongest high's peak.
    pub anomaly_max_m: f32,
    /// Maximum speed of the perturbation (m/s), stability diagnostic.
    pub max_wind_ms: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellProperties;
    use crate::coord::HexCoord;

    fn flat_grid(radius: i32) -> HexGrid {
        HexGrid::from_radius(radius)
    }

    /// Parameters at the **calibration** spacing: fixtures keep their
    /// historical sizes in cells (radius 10, patch 3, …). Since the solver
    /// is parametric in spacing, there's no longer a need to inflate the
    /// grid to preserve real sizes; that was the cost of the
    /// `scale_cells`/`cells_to_radius` helpers removed here (radius ~83,
    /// ~21,000 cells, x8 CFL substeps: `anomaly_mode_extinguishes_standing_forcing`
    /// at 1151 s in debug, measured during the 2026-07-10 test-suite overhaul).
    fn ref_params() -> SynopticParams {
        SynopticParams::for_spacing(SYNOPTIC_REFERENCE_SPACING_M)
    }

    #[test]
    fn depth_and_wave_speed_are_physical() {
        let p = SynopticParams::default();
        // L_d ≈ 10 cells → c on the order of m/s, H on the order of dm.
        assert!(
            p.wave_speed() > 0.3 && p.wave_speed() < 5.0,
            "c = {}",
            p.wave_speed()
        );
        assert!(p.depth() > 0.0);
        assert!(p.coriolis() > 0.0, "f = {}", p.coriolis());
    }

    #[test]
    fn evolution_is_deterministic() {
        let grid = flat_grid(8);
        let params = ref_params();
        let run = || {
            let mut s = SynopticState::new(grid.len(), &params);
            for _ in 0..24 {
                s.step_hour(&grid, &params);
            }
            s.height().to_vec()
        };
        assert_eq!(run(), run(), "same seed → same state, bit-identical");
    }

    #[test]
    fn stays_finite_and_bounded_over_a_month() {
        let mut grid = flat_grid(10);
        // Structured temperature field (warm in the east) → non-trivial forcing.
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                c.temperature = f32::from(i8::try_from(coord.q).unwrap_or(0));
            }
        }
        let params = ref_params();
        let mut s = SynopticState::new(grid.len(), &params);
        for _ in 0..(24 * 30) {
            s.step_hour(&grid, &params);
        }
        assert!(s.height().iter().all(|x| x.is_finite()));
        assert!(
            s.max_speed().is_finite() && s.max_speed() < 100.0,
            "max|u| = {}",
            s.max_speed()
        );
    }

    #[test]
    fn mean_height_conserved_without_forcing_or_relax() {
        // Cuts forcing and relaxation → only the conservative dynamics act:
        // ⟨h⟩ must not drift (bijective toric neighborhood).
        let grid = flat_grid(8);
        let params = SynopticParams {
            thermal_coupling: 0.0,
            relax_days: 1.0e9, // τ → ∞ ⇒ negligible relaxation
            ..ref_params()
        };
        let mut s = SynopticState::new(grid.len(), &params);
        // Non-zero initial perturbation (via seed) → there is something to conserve.
        let h0 = s.mean_height();
        for _ in 0..(24 * 5) {
            s.step_hour(&grid, &params);
        }
        let drift = (s.mean_height() - h0).abs();
        assert!(drift < 1e-2, "⟨h⟩ drift = {drift} m (expected ~0)");
    }

    #[test]
    fn warm_anomaly_digs_a_low() {
        // A central warm patch must dig a low pressure (h < ⟨h⟩) and
        // generate a circulation. This test targets the *historical*
        // forcing (permanent contrast → permanent system): it therefore
        // explicitly sets `thermal_anomaly_days = 0`. In anomaly mode (the
        // default since Phase 4), a *static* contrast dies out over ~τ by
        // construction; that's the behavior covered by the dedicated test
        // `anomaly_mode_extinguishes_standing_forcing`.
        let mut grid = flat_grid(10);
        let center = HexCoord::new(0, 0);
        let patch_radius = 3;
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                let d = coord.distance(center);
                c.temperature = if d <= patch_radius { 12.0 } else { 0.0 };
            }
        }
        // Mean flow cut: we study the *stationary* response to forcing
        // (with U ≠ 0 the low is swept downstream of the warm patch; that's
        // the translation test that covers that regime).
        let params = SynopticParams {
            mean_flow_ms: 0.0,
            thermal_anomaly_days: 0.0,
            ..ref_params()
        };
        let mut s = SynopticState::new(grid.len(), &params);
        for _ in 0..(24 * 3) {
            s.step_hour(&grid, &params);
        }
        let ci = grid.cell_index(center).unwrap();
        let anomaly = s.height()[ci] - s.mean_height();
        assert!(
            anomaly < 0.0,
            "warm anomaly → low expected, h−⟨h⟩ = {anomaly}"
        );
        assert!(
            s.max_speed() > 0.01,
            "circulation should emerge, max|u| = {}",
            s.max_speed()
        );
    }

    #[test]
    fn base_wind_has_right_magnitude_scale() {
        let grid = flat_grid(6);
        let mut grid = grid;
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                c.temperature = f32::from(i8::try_from(coord.r).unwrap_or(0));
            }
        }
        let params = ref_params();
        let mut s = SynopticState::new(grid.len(), &params);
        for _ in 0..48 {
            s.step_hour(&grid, &params);
        }
        let mut wind: WindField = Vec::new();
        s.write_base_wind(&params, &mut wind);
        assert_eq!(wind.len(), grid.len());
        assert!(wind.iter().all(|w| w.x.is_finite() && w.y.is_finite()));
        // The mean flow is part of the real wind: zonal mean ≈ U in
        // WindVec units (the perturbation, with mean ~0, doesn't shift the mean).
        let n = f32::from(u16::try_from(wind.len()).unwrap_or(u16::MAX));
        let mean_x = wind.iter().map(|w| w.x).sum::<f32>() / n;
        let expected = params.mean_flow_ms * 0.1;
        assert!(
            (mean_x - expected).abs() < 0.05,
            "zonal mean {mean_x} vs expected U {expected}"
        );
    }

    /// Toric cycle of the East neighbors (dir 0) from `start`. This is the
    /// exact path along which zonal advection translates the field: the
    /// cycle's wrap IS the toric identification that the dynamics uses.
    fn east_cycle(grid: &HexGrid, start: usize) -> Vec<usize> {
        let mut cycle = vec![start];
        let mut i = grid.neighbor_indices_toric(start)[0];
        while i != start {
            cycle.push(i);
            i = grid.neighbor_indices_toric(i)[0];
        }
        cycle
    }

    /// Phase (rad) of the first Fourier mode of the anomaly `h − ⟨h⟩` along
    /// the cycle. A translation of +1 position along the cycle rotates this
    /// phase by `−2π/L`, a sub-cell measurement, robust to noise.
    fn cycle_phase(s: &SynopticState, cycle: &[usize]) -> f32 {
        let mean = s.mean_height();
        let len = f32::from(u16::try_from(cycle.len()).unwrap_or(u16::MAX));
        let (mut re, mut im) = (0.0_f32, 0.0_f32);
        for (k, &i) in cycle.iter().enumerate() {
            let theta = std::f32::consts::TAU * f32::from(u16::try_from(k).unwrap_or(0)) / len;
            let a = s.h[i] - mean;
            re += a * theta.cos();
            im -= a * theta.sin();
        }
        im.atan2(re)
    }

    /// Wraps an angle into (−π, π], to unwrap the phase between two measurements.
    fn wrap_angle(mut a: f32) -> f32 {
        use std::f32::consts::{PI, TAU};
        while a > PI {
            a -= TAU;
        }
        while a < -PI {
            a += TAU;
        }
        a
    }

    /// Overwrites the state with a Gaussian bump of h (σ = 2.5 cells,
    /// center (0,0)) **geostrophically balanced**: `u_g = (g/f) k̂×∇h`, set
    /// with the same discrete stencil as the dynamics. Without this
    /// balance, the initial adjustment radiates gravity waves that blur
    /// the phase measurement.
    fn inject_balanced_bump(s: &mut SynopticState, grid: &HexGrid, params: &SynopticParams) {
        let center = HexCoord::new(0, 0);
        let amp = 0.05 * s.depth;
        let sigma = 2.5;
        for (i, coord) in grid.coords().copied().enumerate().collect::<Vec<_>>() {
            let d = f32::from(u16::try_from(coord.distance(center)).unwrap_or(0));
            s.h[i] = s.depth + amp * (-d * d / (2.0 * sigma * sigma)).exp();
        }
        let dirs = direction_units();
        let inv = 1.0 / (3.0 * params.cell_spacing_m);
        let g_over_f = GRAVITY / params.coriolis();
        for i in 0..grid.len() {
            let nbr = grid.neighbor_indices_toric(i);
            let (mut gx, mut gy) = (0.0, 0.0);
            for (k, &(dx, dy)) in dirs.iter().enumerate() {
                let dh = s.h[nbr[k]] - s.h[i];
                gx += dh * dx;
                gy += dh * dy;
            }
            s.u[i] = -g_over_f * gy * inv;
            s.v[i] = g_over_f * gx * inv;
        }
    }

    /// THE Phase 2 test: a balanced pressure bump must *travel across* the
    /// map, carried by the mean flow, at speed U; its center covers `U·t`
    /// on the torus (measured via the phase along the East cycle, unwrapped
    /// hour by hour).
    #[test]
    fn mean_flow_translates_a_balanced_bump_at_u_times_t() {
        let grid = flat_grid(8);
        let params = SynopticParams {
            thermal_coupling: 0.0, // pure advection: no forcing…
            relax_days: 1.0e9,     // …nor pull back towards h₀
            mean_flow_ms: 2.0,
            ..ref_params()
        };
        let mut s = SynopticState::new(grid.len(), &params);
        inject_balanced_bump(&mut s, &grid, &params);

        let ci = grid.cell_index(HexCoord::new(0, 0)).unwrap();
        let cycle = east_cycle(&grid, ci);
        let len = f32::from(u16::try_from(cycle.len()).unwrap_or(u16::MAX));
        assert!(cycle.len() > 20, "East cycle too short: {}", cycle.len());

        // Unwraps the phase hour by hour (hourly displacement ≈ 6.7 cells
        // ≪ L/2: no wrap-around ambiguity). 6 h window: beyond that, the
        // upwind scheme's transverse diffusion has spread the bump over
        // several rows whose contributions interfere along the wrapped
        // cycle and mode 1's phase breaks down (measured: clean at 7 h,
        // chaotic at 9 h). 6 h ≈ 40 cells = 2.4 map widths: the crossing
        // is proven.
        let hours = 6;
        let mut shift_rad = 0.0;
        let mut prev = cycle_phase(&s, &cycle);
        for _ in 0..hours {
            s.step_hour(&grid, &params);
            let now = cycle_phase(&s, &cycle);
            shift_rad += wrap_angle(now - prev);
            prev = now;
        }
        // Translation of +1 position ⇒ phase −2π/L, so cells = −Δθ·L/2π.
        let cells_travelled = -shift_rad * len / std::f32::consts::TAU;
        let hours_f = f32::from(u16::try_from(hours).unwrap_or(0));
        let expected = params.mean_flow_ms * 3600.0 * hours_f / params.cell_spacing_m;
        assert!(
            cells_travelled > 0.0,
            "the bump should drift east, measured {cells_travelled} cells"
        );
        let ratio = cells_travelled / expected;
        assert!(
            (0.8..=1.2).contains(&ratio),
            "measured drift {cells_travelled:.1} cells vs U·t = {expected:.1} (ratio {ratio:.2})"
        );
    }

    /// The advection operator alone, without wave dynamics (c ≈ 0 via a
    /// tiny `L_d`, u=v=0, ν=0): h must translate at exactly U, the
    /// "analytical" version of the translation test, without geostrophy.
    #[test]
    fn pure_advection_translates_h_at_exactly_u() {
        let grid = flat_grid(8);
        let params = SynopticParams {
            thermal_coupling: 0.0,
            relax_days: 1.0e9,
            mean_flow_ms: 2.0,
            deformation_radius_cells: 0.1, // c = f·L_d ≈ 0.01 m/s: frozen waves
            friction_days: 1.0e9,
            viscosity: 0.0,
            ..ref_params()
        };
        let mut s = SynopticState::new(grid.len(), &params);
        let center = HexCoord::new(0, 0);
        let amp = 0.05 * s.depth;
        for (i, coord) in grid.coords().copied().enumerate().collect::<Vec<_>>() {
            let d = f32::from(u16::try_from(coord.distance(center)).unwrap_or(0));
            s.h[i] = s.depth + amp * (-d * d / (2.0 * 2.5 * 2.5)).exp();
            s.u[i] = 0.0;
            s.v[i] = 0.0;
        }
        let ci = grid.cell_index(center).unwrap();
        let cycle = east_cycle(&grid, ci);
        let len = f32::from(u16::try_from(cycle.len()).unwrap_or(u16::MAX));
        let mut shift_rad = 0.0;
        let mut prev = cycle_phase(&s, &cycle);
        for _ in 0..6 {
            s.step_hour(&grid, &params);
            let now = cycle_phase(&s, &cycle);
            shift_rad += wrap_angle(now - prev);
            prev = now;
        }
        let cells = -shift_rad * len / std::f32::consts::TAU;
        let expected = params.mean_flow_ms * 3600.0 * 6.0 / params.cell_spacing_m;
        let ratio = cells / expected;
        assert!(
            (0.95..=1.05).contains(&ratio),
            "pure advection: {cells:.1} cells vs U·t = {expected:.1} (ratio {ratio:.2})"
        );
    }

    /// Without mean flow, the same balanced bump goes nowhere: the negative
    /// control for the translation test (this was exactly the Phase 1
    /// behavior, intended here, pathological in production).
    #[test]
    fn without_mean_flow_a_balanced_bump_stays_put() {
        let grid = flat_grid(8);
        let params = SynopticParams {
            thermal_coupling: 0.0,
            relax_days: 1.0e9,
            mean_flow_ms: 0.0,
            ..ref_params()
        };
        let mut s = SynopticState::new(grid.len(), &params);
        inject_balanced_bump(&mut s, &grid, &params);
        let ci = grid.cell_index(HexCoord::new(0, 0)).unwrap();
        let cycle = east_cycle(&grid, ci);
        let len = f32::from(u16::try_from(cycle.len()).unwrap_or(u16::MAX));
        let mut shift_rad = 0.0;
        let mut prev = cycle_phase(&s, &cycle);
        for _ in 0..12 {
            s.step_hour(&grid, &params);
            let now = cycle_phase(&s, &cycle);
            shift_rad += wrap_angle(now - prev);
            prev = now;
        }
        let cells = (-shift_rad * len / std::f32::consts::TAU).abs();
        assert!(
            cells < 3.0,
            "without U the bump should not drift: {cells} cells"
        );
    }

    /// The combined waves + advection CFL drives the effective time step:
    /// `(c + |U|)·dt ≤ CFL_MAX·d`, and an extreme U tightens the step.
    #[test]
    fn substeps_respect_combined_cfl() {
        let p = SynopticParams::default();
        let n = p.substeps_per_hour();
        assert!(n >= p.substeps);
        let dt = 3600.0 / f32::from(u16::try_from(n).unwrap_or(1));
        let signal = p.wave_speed() + p.mean_flow_ms.abs();
        assert!(
            signal * dt <= 0.7 * p.cell_spacing_m * 1.0001,
            "Courant = {}",
            signal * dt / p.cell_spacing_m
        );
        let fast = SynopticParams {
            mean_flow_ms: 30.0,
            ..SynopticParams::default()
        };
        assert!(
            fast.substeps_per_hour() > n,
            "extreme U should tighten the step"
        );
    }

    /// `for_spacing` preserves the ABSOLUTE physical quantities of the
    /// reference calibration, regardless of spacing: `L_d` (hence `c` and
    /// `depth`), and the relative checkerboard damping `ν/Δx²`. The number
    /// of CFL substeps, however, follows `1/Δx`; that's the whole point of
    /// the coarse-grid overhaul (#88): 20 substeps/h at the calibration
    /// spacing, 163 at 130 m, for the same physics.
    #[test]
    fn for_spacing_preserves_absolute_physics() {
        let reference = ref_params();
        let ld_ref = reference.deformation_radius_cells * reference.cell_spacing_m;
        let damp_ref = reference.viscosity / (reference.cell_spacing_m * reference.cell_spacing_m);
        for spacing in [130.0_f32, 650.0, SYNOPTIC_REFERENCE_SPACING_M, 1040.0] {
            let p = SynopticParams::for_spacing(spacing);
            let ld = p.deformation_radius_cells * p.cell_spacing_m;
            assert!(
                ((ld - ld_ref) / ld_ref).abs() < 1e-5,
                "L_d should be absolute: {ld} m at {spacing} m vs {ld_ref} m"
            );
            assert!(
                (p.wave_speed() - reference.wave_speed()).abs() < 1e-4,
                "c should be invariant: {} vs {}",
                p.wave_speed(),
                reference.wave_speed()
            );
            let damp = p.viscosity / (spacing * spacing);
            assert!(
                ((damp - damp_ref) / damp_ref).abs() < 1e-5,
                "ν/Δx² should be invariant: {damp} vs {damp_ref}"
            );
        }
        assert_eq!(ref_params().substeps_per_hour(), 20);
        assert_eq!(SynopticParams::for_spacing(130.0).substeps_per_hour(), 163);
    }

    /// Anomaly mode (`thermal_anomaly_days > 0`): a *permanent* T contrast
    /// stops forcing once absorbed by the rolling average, the initial
    /// thermal low fills back in (contrast with `warm_anomaly_digs_a_low`
    /// where it persists). This is the anti-anchoring mechanism measured
    /// in Phase 3.
    #[test]
    fn anomaly_mode_extinguishes_standing_forcing() {
        let mut grid = flat_grid(10);
        let center = HexCoord::new(0, 0);
        let patch_radius = 3;
        let params = SynopticParams {
            mean_flow_ms: 0.0,         // pure local response, no advection
            thermal_anomaly_days: 2.0, // short filter for a fast test
            ..ref_params()
        };
        let mut s = SynopticState::new(grid.len(), &params);
        // T=0 everywhere first: this first step seeds t_ref=0 over the
        // whole grid (`fill_thermal_forcing`, reference = current T on the
        // first call). The contrast is only applied AFTERWARDS, so t_ref
        // has a real gap to catch up on; otherwise t_ref would already
        // initialize equal to T and the forcing would stay zero from the
        // first to the last tick, whatever T is (a fixture bug found on
        // feat/dem-terrain-validation: a T that's *static* from the start
        // never tests the extinction).
        s.step_hour(&grid, &params);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(c) = grid.get_mut(coord) {
                let d = coord.distance(center);
                c.temperature = if d <= patch_radius { 12.0 } else { 0.0 };
            }
        }
        let ci = grid.cell_index(center).unwrap();
        // Digging phase: the contrast is fresh, t_ref hasn't caught up to
        // the new T plateau yet.
        for _ in 0..(24 * 2) {
            s.step_hour(&grid, &params);
        }
        let dug = (s.height()[ci] - s.mean_height()).abs();
        assert!(dug > 0.0, "the fresh contrast should dig a low, dug={dug}");
        // T stays at the new plateau ≫ τ: the reference has caught up, the
        // forcing dies out and relaxation fills the trough back in.
        for _ in 0..(24 * 20) {
            s.step_hour(&grid, &params);
        }
        let residual = (s.height()[ci] - s.mean_height()).abs();
        assert!(
            residual < dug * 0.35,
            "the low should fill back in once forcing dies out: dug {dug:.4} m, residual {residual:.4} m"
        );
    }

    #[test]
    fn default_cell_used() {
        // Safeguard: CellProperties::default() indeed has a finite
        // temperature (the forcing depends on it).
        assert!(CellProperties::default().temperature.is_finite());
    }
}
