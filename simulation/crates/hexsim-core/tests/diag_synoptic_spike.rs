//! Phase 0 — Numerical spike for synoptic dynamics.
//!
//! Goal (the synoptic-dynamics design flags CFL/time-step stability as its
//! risk #1, to retire at a dedicated Phase 0 spike before writing anything
//! real): **retire the time-step numerical risk** before writing
//! anything into prod. This prototype is **outside the sim**: it does not
//! touch any engine phenomenon, and the only thing it reuses from
//! `hexsim-core` is the **hex torus topology** (`HexGrid` for toric
//! neighbors, `hex_direction_to_world` for the geometry). All the
//! shallow-water physics lives here, in the test.
//!
//! What we measure (the verdict):
//!   1. **Real CFL** — at what sub-step count the dynamics becomes stable,
//!      compared to the theoretical bound `dt < dx / c`, `c = √(gH)`.
//!   2. **Perf** — wall cost of one hourly step at the stable sub-step count,
//!      compared to the prod tick budget (~14-55 ms depending on grid radius).
//!   3. **Geostrophic adjustment** — does a pressure bump organize into a
//!      balanced vortex (Coriolis) instead of dispersing or blowing up.
//!   4. **Thermal forcing** — does a warm anomaly dig a **low pressure**
//!      (the §3.2 hook `Q ∝ −(T−T̄)`).
//!   5. **Conservation** — does the mass ∮h drift (terrarium invariant).
//!
//! Expected verdict: settle **explicit shallow-water vs QG** (§8) on numbers,
//! not assumptions.
//!
//! Scheme chosen for the spike: **forward-backward** (updates velocity with
//! `h^n`, then `h` with the fresh velocity). It's the smallest scheme
//! *stable for gravity waves* — plain forward Euler is unconditionally
//! unstable on an oscillatory system, which would bias the CFL measurement.
//! It's also what a real minimal SW kernel would use, so the measured perf
//! is representative.
//!
//! Run with: `just diag-tool synoptic_spike` (or via `just diag-tools`).

use std::time::Instant;

use hexsim_core::coord::hex_direction_to_world;
use hexsim_core::grid::HexGrid;

// --- SI constants (documented) ----------------------------------------------

/// Gravity (m/s²).
const G: f64 = 9.81;
/// Earth's angular velocity (rad/s), Coriolis f = 2Ω sin φ.
const OMEGA: f64 = 7.292e-5;
/// Reference latitude of the single-latitude terrarium (f-plane, §8). 45°.
const LAT_DEG: f64 = 45.0;

/// Surface of a cell: ~100 ha = 1e6 m² (the grid convention at the time of
/// this spike, before the later rescale to a 130 m cell spacing).
/// Regular hexagon: `A = (√3/2) d²` with `d` = center-to-center distance of
/// neighbors → `d = √(2A/√3) ≈ 1074.6 m`. Consistent with the "dx ~1 km" from
/// the journal.
const DX: f64 = 1074.569;

/// Hex radius of the tested domain, prod scale (`synoptic_front_coverage` = 30).
const RADIUS: i32 = 30;

// --- Physical parameters of a scenario --------------------------------------

struct Physics {
    /// Scale depth H (m). Sets the wave speed `c = √(gH)`.
    depth: f64,
    /// Coriolis f (1/s).
    coriolis: f64,
    /// Rayleigh friction r (1/s). Dissipates, prevents runaway growth (§3.2).
    friction: f64,
    /// Relaxation time τ (s) toward the base state. 0 = disabled.
    relax_tau: f64,
    /// Numerical diffusion ν (m²/s), controls the checkerboard (§7 risk #2).
    /// 0 = off.
    viscosity: f64,
}

impl Physics {
    fn wave_speed(&self) -> f64 {
        (G * self.depth).sqrt()
    }

    /// Theoretical CFL time step `dt < dx / c` (gravity stability bound).
    fn cfl_dt(&self) -> f64 {
        DX / self.wave_speed()
    }

    /// Rossby deformation radius `L_d = c / f` (m), scale of the systems.
    fn deformation_radius(&self) -> f64 {
        self.wave_speed() / self.coriolis
    }
}

fn coriolis_45() -> f64 {
    2.0 * OMEGA * (LAT_DEG.to_radians()).sin()
}

/// Scale depth `H` (m) such that the deformation radius `L_d = √(gH)/f`
/// equals `target_ld` meters. `H = (f · L_d)² / g`. This is the **real design
/// lever** on this small domain: `L_d` sets the size of the pressure systems,
/// so it must be tuned to the map's scale (§8, Phase 1 decision).
fn depth_for_deformation_radius(target_ld: f64, f: f64) -> f64 {
    (f * target_ld).powi(2) / G
}

// --- Shallow-water state -----------------------------------------------------

struct State {
    /// Height h (m), proxy for synoptic pressure.
    h: Vec<f64>,
    /// Synoptic wind (u, v) (m/s).
    u: Vec<f64>,
    v: Vec<f64>,
}

impl State {
    fn at_rest(n: usize, depth: f64) -> Self {
        Self {
            h: vec![depth; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    fn max_speed(&self) -> f64 {
        self.u
            .iter()
            .zip(&self.v)
            .map(|(a, b)| a.hypot(*b))
            .fold(0.0, f64::max)
    }

    fn mean_h(&self) -> f64 {
        self.h.iter().sum::<f64>() / count_f64(self.h.len())
    }

    fn is_finite(&self) -> bool {
        self.h
            .iter()
            .chain(&self.u)
            .chain(&self.v)
            .all(|x| x.is_finite())
    }
}

// --- Differential operators on the hexagonal torus --------------------------

/// World unit vectors of the 6 hex directions (precomputed, f64).
fn direction_units() -> [(f64, f64); 6] {
    let mut d = [(0.0, 0.0); 6];
    for (i, slot) in d.iter_mut().enumerate() {
        let (x, y) = hex_direction_to_world(i);
        *slot = (f64::from(x), f64::from(y));
    }
    d
}

/// Gradient of a scalar field by least squares on the 6 hex neighbors:
/// `Σ d̂_i d̂_iᵀ = 3 I` ⇒ `∇φ ≈ (1/(3 dx)) Σ_i (φ_i − φ_c) d̂_i`. Isotropic,
/// periodic via the toric neighbors.
fn gradient(field: &[f64], nbr: &[[usize; 6]], dirs: &[(f64, f64); 6], out: &mut [(f64, f64)]) {
    let inv = 1.0 / (3.0 * DX);
    for (i, slot) in out.iter_mut().enumerate() {
        let (mut gx, mut gy) = (0.0, 0.0);
        let phi_c = field[i];
        for (k, &(dx, dy)) in dirs.iter().enumerate() {
            let delta = field[nbr[i][k]] - phi_c;
            gx += delta * dx;
            gy += delta * dy;
        }
        *slot = (gx * inv, gy * inv);
    }
}

/// Divergence `∇·u ≈ (1/(3 dx)) Σ_i (u_i − u_c)·d̂_i`. On the torus, the
/// global sum telescopes to ~0 (bijective toric neighborhood) → conserves ∮h.
fn divergence(u: &[f64], v: &[f64], nbr: &[[usize; 6]], dirs: &[(f64, f64); 6], out: &mut [f64]) {
    let inv = 1.0 / (3.0 * DX);
    for (i, slot) in out.iter_mut().enumerate() {
        let (uc, vc) = (u[i], v[i]);
        let mut acc = 0.0;
        for (k, &(dx, dy)) in dirs.iter().enumerate() {
            acc += (u[nbr[i][k]] - uc) * dx + (v[nbr[i][k]] - vc) * dy;
        }
        *slot = acc * inv;
    }
}

/// Hex Laplacian `∇²φ ≈ (2/(3 dx²)) Σ_i (φ_i − φ_c)` (numerical diffusion).
fn laplacian(field: &[f64], nbr: &[[usize; 6]], i: usize) -> f64 {
    let inv = 2.0 / (3.0 * DX * DX);
    let phi_c = field[i];
    let mut acc = 0.0;
    for &j in &nbr[i] {
        acc += field[j] - phi_c;
    }
    acc * inv
}

// --- Time integration (forward-backward) -------------------------------------

struct Scratch {
    grad_h: Vec<(f64, f64)>,
    div_u: Vec<f64>,
    /// Velocity tendencies (du/dt, dv/dt), buffered so diffusion reads the
    /// old state (double-buffer, no order-dependent Gauss-Seidel).
    du: Vec<f64>,
    dv: Vec<f64>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            grad_h: vec![(0.0, 0.0); n],
            div_u: vec![0.0; n],
            du: vec![0.0; n],
            dv: vec![0.0; n],
        }
    }
}

/// One forward-backward sub-step of `dt` seconds.
/// Momentum: `du/dt = f v − g ∂ₓh − r u (+ ν∇²u)`; `dv/dt = −f u − g ∂_yh − r v`.
/// Continuity: `dh/dt = −H ∇·u + Q − (h−h₀)/τ`, using the **fresh** velocity.
fn substep(
    s: &mut State,
    phys: &Physics,
    nbr: &[[usize; 6]],
    dirs: &[(f64, f64); 6],
    forcing: &[f64],
    dt: f64,
    scratch: &mut Scratch,
) {
    let (f, g, r, h0) = (phys.coriolis, G, phys.friction, phys.depth);
    gradient(&s.h, nbr, dirs, &mut scratch.grad_h);
    // 1) forward on velocity (explicit Coriolis: |f dt| ≪ 1 → harmless).
    //    Tendencies computed first (reads the old state for diffusion),
    //    applied after → simultaneous update, order-independent.
    for (i, (du_i, dv_i)) in scratch.du.iter_mut().zip(scratch.dv.iter_mut()).enumerate() {
        let (gx, gy) = scratch.grad_h[i];
        let (uc, vc) = (s.u[i], s.v[i]);
        let mut du = f * vc - g * gx - r * uc;
        let mut dv = -f * uc - g * gy - r * vc;
        if phys.viscosity > 0.0 {
            du += phys.viscosity * laplacian(&s.u, nbr, i);
            dv += phys.viscosity * laplacian(&s.v, nbr, i);
        }
        *du_i = du;
        *dv_i = dv;
    }
    for (((uu, vv), du_i), dv_i) in
        s.u.iter_mut()
            .zip(&mut s.v)
            .zip(&scratch.du)
            .zip(&scratch.dv)
    {
        *uu += dt * du_i;
        *vv += dt * dv_i;
    }
    // 2) backward on h with the fresh velocity (the "backward" of the scheme).
    divergence(&s.u, &s.v, nbr, dirs, &mut scratch.div_u);
    for (i, hh) in s.h.iter_mut().enumerate() {
        let mut dh = -h0 * scratch.div_u[i] + forcing[i];
        if phys.relax_tau > 0.0 {
            dh -= (*hh - h0) / phys.relax_tau;
        }
        *hh += dt * dh;
    }
}

/// Advances one hourly tick (3600 s) split into `substeps` sub-steps.
fn step_hour(
    s: &mut State,
    phys: &Physics,
    nbr: &[[usize; 6]],
    dirs: &[(f64, f64); 6],
    forcing: &[f64],
    substeps: u32,
    scratch: &mut Scratch,
) {
    let dt = 3600.0 / f64::from(substeps);
    for _ in 0..substeps {
        substep(s, phys, nbr, dirs, forcing, dt, scratch);
    }
}

// --- Utilities ---------------------------------------------------------------

fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// World position (x, y) in meters of an axial cell (q, r).
/// Basis consistent with `hex_direction_to_world`: `e_q=(1,0)`, `e_r=(0.5, √3/2)`.
fn world_xy(q: i32, r: i32) -> (f64, f64) {
    let (qf, rf) = (f64::from(q), f64::from(r));
    (DX * (qf + 0.5 * rf), DX * (0.866_025_4 * rf))
}

fn topology() -> (HexGrid, Vec<[usize; 6]>) {
    let grid = HexGrid::from_radius(RADIUS);
    let nbr: Vec<[usize; 6]> = (0..grid.len())
        .map(|i| grid.neighbor_indices_toric(i))
        .collect();
    (grid, nbr)
}

/// Topology references (grid + toric neighbors + world directions), grouped
/// together to avoid lugging 3 arguments around in every scenario.
struct Mesh<'a> {
    grid: &'a HexGrid,
    nbr: &'a [[usize; 6]],
    dirs: &'a [(f64, f64); 6],
}

/// Gaussian anomaly centered on the cell at index `center`, amplitude `amp`,
/// width `sigma` (m). Used as a pressure bump and as thermal forcing.
fn gaussian_field(grid: &HexGrid, center: usize, amp: f64, sigma: f64) -> Vec<f64> {
    let coords = grid.coords_slice();
    let (cx, cy) = world_xy(coords[center].q, coords[center].r);
    coords
        .iter()
        .map(|c| {
            let (x, y) = world_xy(c.q, c.r);
            let d2 = (x - cx).powi(2) + (y - cy).powi(2);
            amp * (-d2 / (2.0 * sigma * sigma)).exp()
        })
        .collect()
}

// --- Scenario 1: CFL sweep --------------------------------------------------

struct SweepPoint {
    substeps: u32,
    dt: f64,
    cfl_ratio: f64,
    stable: bool,
    max_speed: f64,
}

/// Runs `hours` ticks with `substeps` sub-steps and reports stability.
/// Instability = NaN/inf or an aberrant speed (> 1e4 m/s).
fn run_stability(mesh: &Mesh, phys: &Physics, substeps: u32, hours: u32) -> SweepPoint {
    let (grid, nbr, dirs) = (mesh.grid, mesh.nbr, mesh.dirs);
    let center = grid.len() / 2;
    let mut s = State::at_rest(grid.len(), phys.depth);
    // Pressure bump: enough to excite the fast gravity waves.
    let bump = gaussian_field(grid, center, 40.0, 6.0 * DX);
    for (h, b) in s.h.iter_mut().zip(&bump) {
        *h += b;
    }
    let zero = vec![0.0; grid.len()];
    let mut scratch = Scratch::new(grid.len());
    for _ in 0..hours {
        step_hour(&mut s, phys, nbr, dirs, &zero, substeps, &mut scratch);
        if !s.is_finite() || s.max_speed() > 1.0e4 {
            break;
        }
    }
    let stable = s.is_finite() && s.max_speed() < 1.0e4;
    let dt = 3600.0 / f64::from(substeps);
    SweepPoint {
        substeps,
        dt,
        cfl_ratio: dt / phys.cfl_dt(),
        stable,
        max_speed: if s.is_finite() {
            s.max_speed()
        } else {
            f64::INFINITY
        },
    }
}

// --- Scenario 2: geostrophic adjustment --------------------------------------

struct Adjustment {
    max_speed: f64,
    rossby: f64,
    mean_h_drift: f64,
    finite: bool,
}

/// Pressure bump of width `sigma` (≈ `L_d`) with no forcing → should organize
/// into a balanced vortex (Coriolis turns the wind, energy is retained when
/// `L ~ L_d`), not disperse entirely nor blow up. Also measures the mass
/// drift (conservation): neither forcing nor relaxation here, only friction
/// acts.
fn run_geostrophic(
    mesh: &Mesh,
    phys: &Physics,
    substeps: u32,
    hours: u32,
    sigma: f64,
    amp: f64,
) -> Adjustment {
    let (grid, nbr, dirs) = (mesh.grid, mesh.nbr, mesh.dirs);
    let center = grid.len() / 2;
    let mut s = State::at_rest(grid.len(), phys.depth);
    let bump = gaussian_field(grid, center, amp, sigma);
    for (h, b) in s.h.iter_mut().zip(&bump) {
        *h += b;
    }
    let h_start = s.mean_h();
    let zero = vec![0.0; grid.len()];
    let mut scratch = Scratch::new(grid.len());
    for _ in 0..hours {
        step_hour(&mut s, phys, nbr, dirs, &zero, substeps, &mut scratch);
    }
    let umax = s.max_speed();
    Adjustment {
        max_speed: umax,
        rossby: umax / (phys.coriolis * sigma),
        mean_h_drift: (s.mean_h() - h_start).abs(),
        finite: s.is_finite(),
    }
}

// --- Scenario 3: thermal forcing ---------------------------------------------

struct ThermalLow {
    center_anomaly: f64,
    max_speed: f64,
    finite: bool,
}

/// A warm anomaly → `Q ∝ −(T−T̄)` digs a low pressure at the center.
/// We check the sign: h at the center goes **below** the mean.
fn run_thermal(mesh: &Mesh, phys: &Physics, substeps: u32, hours: u32, sigma: f64) -> ThermalLow {
    let (grid, nbr, dirs) = (mesh.grid, mesh.nbr, mesh.dirs);
    let center = grid.len() / 2;
    let mut s = State::at_rest(grid.len(), phys.depth);
    // Q < 0 at the hot point: negative height source → digs the L.
    // Toy amplitude: ~4e-5 m/s of settling (toy spike, not a prod constant).
    let forcing = gaussian_field(grid, center, -4.0e-5, sigma);
    let mut scratch = Scratch::new(grid.len());
    for _ in 0..hours {
        step_hour(&mut s, phys, nbr, dirs, &forcing, substeps, &mut scratch);
    }
    ThermalLow {
        center_anomaly: s.h[center] - s.mean_h(),
        max_speed: s.max_speed(),
        finite: s.is_finite(),
    }
}

// --- Perf -------------------------------------------------------------------

/// Average wall cost of an hourly tick, in ms, at the given sub-step count.
fn bench_ms_per_hour(mesh: &Mesh, phys: &Physics, substeps: u32, hours: u32) -> f64 {
    let (grid, nbr, dirs) = (mesh.grid, mesh.nbr, mesh.dirs);
    let mut s = State::at_rest(grid.len(), phys.depth);
    let bump = gaussian_field(grid, grid.len() / 2, 40.0, 8.0 * DX);
    for (h, b) in s.h.iter_mut().zip(&bump) {
        *h += b;
    }
    let zero = vec![0.0; grid.len()];
    let mut scratch = Scratch::new(grid.len());
    let t0 = Instant::now();
    for _ in 0..hours {
        step_hour(&mut s, phys, nbr, dirs, &zero, substeps, &mut scratch);
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms / f64::from(hours)
}

/// Smallest stable sub-step count in a sorted (ascending) list.
fn first_stable(points: &[SweepPoint]) -> Option<u32> {
    points.iter().find(|p| p.stable).map(|p| p.substeps)
}

/// Approximate width of the hexagonal domain in km (diameter × dx).
fn domain_width_km() -> f64 {
    count_f64(2 * usize::try_from(RADIUS).unwrap_or(0) + 1) * DX / 1000.0
}

fn print_regime(name: &str, phys: &Physics) {
    println!("\n--- Regime {name} ---");
    println!(
        "  c = √(gH) = {:.2} m/s  →  CFL dt_max = dx/c = {:.1} s",
        phys.wave_speed(),
        phys.cfl_dt()
    );
    println!(
        "  theoretical min sub-steps / tick = 3600/dt_max = {:.0}",
        3600.0 / phys.cfl_dt()
    );
    println!(
        "  deformation radius L_d = c/f = {:.0} km ({:.1} cells)",
        phys.deformation_radius() / 1000.0,
        phys.deformation_radius() / DX
    );
}

/// CFL sweep on the "stress" regime (full external gravity): the risk #1
/// value from the plan. Prints the table and returns (points, first stable
/// sub-step).
fn cfl_sweep(mesh: &Mesh, stress: &Physics) -> (Vec<SweepPoint>, u32) {
    println!("\n--- CFL sweep (stress regime H=8000, 48 ticks, pressure bump) ---");
    println!(
        "  {:>8} {:>10} {:>10} {:>8} {:>14}",
        "sub-step", "dt (s)", "dt/dt_cfl", "stable", "max|u| (m/s)"
    );
    let mut sweep = Vec::new();
    for &sub in &[1_u32, 200, 600, 900, 950, 1024, 1500] {
        let p = run_stability(mesh, stress, sub, 48);
        println!(
            "  {:>8} {:>10.2} {:>10.3} {:>8} {:>14}",
            p.substeps,
            p.dt,
            p.cfl_ratio,
            if p.stable { "YES" } else { "NO" },
            if p.max_speed.is_finite() {
                format!("{:.2}", p.max_speed)
            } else {
                "∞/NaN".to_string()
            }
        );
        sweep.push(p);
    }
    let stable_sub = first_stable(&sweep).unwrap_or(u32::MAX);
    println!(
        "  → 1st stable sub-step measured: {stable_sub} (theory ≈ {:.0})",
        3600.0 / stress.cfl_dt()
    );
    (sweep, stable_sub)
}

// --- The report (main test) --------------------------------------------------

/// Scenario 2 (printed): geostrophic adjustment in the synoptic regime
/// (L ~ `L_d`), where Coriolis really organizes a balanced vortex.
fn report_geostrophic_adjustment(
    mesh: &Mesh,
    synoptic: &Physics,
    syn_sub: u32,
    ld_target: f64,
) -> Adjustment {
    println!("\n--- Geostrophic adjustment (synoptic regime, {syn_sub} sub-steps, 36 h) ---");
    let adj = run_geostrophic(mesh, synoptic, syn_sub, 36, ld_target, 0.1);
    println!(
        "  finite: {}   max|u| = {:.3} m/s   Rossby = {:.3}",
        adj.finite, adj.max_speed, adj.rossby
    );
    println!(
        "  mass drift ⟨h⟩ = {:.3e} m (terrarium invariant)",
        adj.mean_h_drift
    );
    println!(
        "  interpretation: {}",
        if adj.max_speed > 0.02 && adj.max_speed < 50.0 {
            "✓ the bump organizes into a bounded balanced vortex (Coriolis turns the wind, energy retained)"
        } else {
            "⚠ degenerate flow, check L_d / L scale"
        }
    );
    adj
}

/// Scenario 3 (printed): thermal forcing in the same regime → low pressure.
fn report_thermal_forcing(
    mesh: &Mesh,
    synoptic: &Physics,
    syn_sub: u32,
    ld_target: f64,
) -> ThermalLow {
    println!("\n--- Thermal forcing Q ∝ −(T−T̄) (synoptic regime, {syn_sub} sub-steps, 48 h) ---");
    let therm = run_thermal(mesh, synoptic, syn_sub, 48, ld_target);
    println!(
        "  h anomaly at hot center = {:.3} m   max|u| = {:.3} m/s",
        therm.center_anomaly, therm.max_speed
    );
    println!(
        "  interpretation: {}",
        if therm.center_anomaly < 0.0 {
            "✓ a LOW pressure deepens over the warm anomaly + circulation (correct sign)"
        } else {
            "⚠ unexpected sign"
        }
    );
    therm
}

/// Perf (printed): the worst case (stress) vs the operating point (synoptic).
fn report_perf(
    mesh: &Mesh,
    stress: &Physics,
    synoptic: &Physics,
    syn_sub: u32,
    n_cells: usize,
) -> (f64, f64) {
    println!("\n--- Perf (ms per hourly tick, {n_cells} cells) ---");
    let ms_stress = bench_ms_per_hour(mesh, stress, 1024, 24);
    let ms_synoptic = bench_ms_per_hour(mesh, synoptic, syn_sub, 24);
    println!("  stress   (H=8000, 1024 sub-steps): {ms_stress:.2} ms/tick");
    println!("  synoptic (H tuned, {syn_sub} sub-steps):   {ms_synoptic:.2} ms/tick");
    println!("  production tick budget: ~14-55 ms (depending on grid radius)");
    (ms_stress, ms_synoptic)
}

#[test]
#[ignore = "heavy numerical spike, run via `just diag-tool synoptic_spike`"]
fn synoptic_spike_report() {
    let (grid, nbr) = topology();
    let dirs = direction_units();
    let mesh = Mesh {
        grid: &grid,
        nbr: &nbr,
        dirs: &dirs,
    };
    let f = coriolis_45();

    println!("\n=== PHASE 0: SYNOPTIC DYNAMIC SPIKE (shallow-water hex/torus) ===");
    println!(
        "domain: radius {RADIUS}, {} cells, dx = {DX:.1} m",
        grid.len()
    );
    println!(
        "domain width ≈ {:.0} km (mono-latitude {LAT_DEG:.0}°, f-plane)",
        domain_width_km()
    );
    println!("Coriolis f = {f:.3e} 1/s");

    // Two regimes. `stress` = full external gravity (H=8000): the worst
    // numerical case, used to bound risk #1. `synoptic` = H tuned so that
    // L_d ≈ 10 cells (⇒ systems that fit within the map): the physically
    // relevant operating point for this 66 km terrarium.
    let visc = 1.0 / (3.0 * 86400.0);
    let stress = Physics {
        depth: 8000.0,
        coriolis: f,
        friction: visc,
        relax_tau: 0.0,
        viscosity: 200.0,
    };
    let ld_target = 10.0 * DX;
    let synoptic = Physics {
        depth: depth_for_deformation_radius(ld_target, f),
        coriolis: f,
        friction: visc,
        relax_tau: 0.0,
        viscosity: 40.0,
    };
    print_regime("stress (H=8000, external gravity)", &stress);
    print_regime("synoptic (H tuned to L_d≈10 cells)", &synoptic);

    let (sweep, stress_stable_sub) = cfl_sweep(&mesh, &stress);

    let syn_sub = 12;
    let adj = report_geostrophic_adjustment(&mesh, &synoptic, syn_sub, ld_target);
    let therm = report_thermal_forcing(&mesh, &synoptic, syn_sub, ld_target);
    let (ms_stress, ms_synoptic) = report_perf(&mesh, &stress, &synoptic, syn_sub, grid.len());

    print_verdict(ms_stress, ms_synoptic, stress_stable_sub, syn_sub);

    // Guardrails (anti-regression, not the core of the spike).
    assert!(
        !sweep[0].stable,
        "CFL must be REAL: 1 sub-step (dt=3600 s ≫ dt_cfl) must blow up"
    );
    assert!(
        stress_stable_sub != u32::MAX,
        "at least one sweep config must be stable"
    );
    assert!(
        adj.finite && adj.max_speed < 50.0,
        "unstable geostrophic adjustment: max|u|={:.3}",
        adj.max_speed
    );
    assert!(
        adj.max_speed > 0.02,
        "the bump fully dissipated, no geostrophic equilibrium"
    );
    assert!(
        adj.mean_h_drift < 1e-3,
        "mass drift ∮h too large: {:.3e} m",
        adj.mean_h_drift
    );
    assert!(
        therm.finite && therm.center_anomaly < 0.0,
        "thermal forcing must dig an L (h_center < ⟨h⟩)"
    );
}

fn print_verdict(ms_stress: f64, ms_synoptic: f64, stress_stable_sub: u32, syn_sub: u32) {
    println!("\n=== VERDICT PHASE 0 ===");
    let budget = 55.0;
    println!(
        "1. Explicit forward-backward scheme: STABLE beyond the measured CFL \
         (stress: {stress_stable_sub} sub-steps/tick). Pure Euler-forward was excluded (waves → unstable)."
    );
    println!(
        "2. The REAL operating point is not H=8000. On a domain of \
         {:.0} km, the scale depth H must be tuned so that L_d ≈ the desired \
         system size (here ~10 cells). This makes c small (~1 m/s) → trivial CFL: \
         **{syn_sub} sub-steps/tick suffice**, and a geostrophic vortex plus a \
         thermal low emerge cleanly (scenarios 2 & 3).",
        domain_width_km()
    );
    if ms_stress < budget {
        println!(
            "3. Perf: even the worst case (stress H=8000, 1024 sub-steps) holds at {ms_stress:.1} ms/tick \
             < budget {budget:.0} ms (the grid is small, ~2800 cells). The \
             synoptic operating point is nearly free ({ms_synoptic:.1} ms/tick). **Risk #1 CLEARED.**"
        );
    } else {
        println!(
            "3. Perf: the worst case (stress) costs {ms_stress:.1} ms/tick ≥ budget {budget:.0} ms, BUT \
             the synoptic operating point holds at {ms_synoptic:.1} ms/tick, target this regime."
        );
    }
    println!(
        "4. Phase 1 reco: **explicit forward-backward shallow-water, f-plane, H tuned to L_d**. \
         Neither QG nor semi-implicit required, the perf lever is physical (choice of L_d/H), not numerical. \
         QG remains the fallback if the grid grows a lot (sub-stepping ∝ c·N_cells)."
    );
    println!(
        "5. Scale caveat: the terrarium is ~{:.0} km, \"synoptic\" here is analogical. \
         f-plane + mean flow is enough for \"the rain travels\" (§8); beta-plane is an extension.",
        domain_width_km()
    );
    println!("=======================\n");
}

// --- Light unit tests (run in `just test`, validate the operators) -----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dx_matches_hundred_hectare_hexagon() {
        // A = (√3/2) d²  →  d = √(2A/√3), A = 1e6 m².
        let d = (2.0e6 / 3.0_f64.sqrt()).sqrt();
        assert!((d - DX).abs() < 0.1, "DX={DX}, expected {d:.3}");
    }

    #[test]
    fn gradient_of_planar_field_is_exact_on_interior() {
        // Planar field h = a·x + b·y: ∇h = (a, b) exactly, tested on an
        // interior cell (non-wrapped neighbors, toric == planar).
        let grid = HexGrid::from_radius(6);
        let nbr: Vec<[usize; 6]> = (0..grid.len())
            .map(|i| grid.neighbor_indices_toric(i))
            .collect();
        let dirs = direction_units();
        let (a, b) = (3.0e-4, -1.5e-4);
        let field: Vec<f64> = grid
            .coords_slice()
            .iter()
            .map(|c| {
                let (x, y) = world_xy(c.q, c.r);
                a * x + b * y
            })
            .collect();
        let mut grads = vec![(0.0, 0.0); grid.len()];
        gradient(&field, &nbr, &dirs, &mut grads);
        // Central cell (0,0): interior, 6 in-grid neighbors.
        let center = grid
            .index_of(hexsim_core::coord::HexCoord::new(0, 0))
            .unwrap();
        let (gx, gy) = grads[center];
        assert!((gx - a).abs() < 1e-9, "gx={gx}, expected {a}");
        assert!((gy - b).abs() < 1e-9, "gy={gy}, expected {b}");
    }

    #[test]
    fn divergence_of_uniform_flow_is_zero() {
        // Uniform wind → divergence zero everywhere (including at the toric wrap).
        let grid = HexGrid::from_radius(5);
        let nbr: Vec<[usize; 6]> = (0..grid.len())
            .map(|i| grid.neighbor_indices_toric(i))
            .collect();
        let dirs = direction_units();
        let u = vec![2.7; grid.len()];
        let v = vec![-1.3; grid.len()];
        let mut div = vec![0.0; grid.len()];
        divergence(&u, &v, &nbr, &dirs, &mut div);
        let max_abs = div.iter().fold(0.0_f64, |m, d| m.max(d.abs()));
        assert!(
            max_abs < 1e-9,
            "divergence of a uniform flow non-zero: {max_abs:.3e}"
        );
    }

    #[test]
    fn direction_units_are_unit_and_sum_to_zero() {
        let dirs = direction_units();
        let (mut sx, mut sy) = (0.0, 0.0);
        for &(x, y) in &dirs {
            assert!((x.hypot(y) - 1.0).abs() < 1e-6);
            sx += x;
            sy += y;
        }
        // 6 directions opposite in pairs → sum is zero (isotropic torus).
        assert!(sx.hypot(sy) < 1e-6, "Σ d̂ ≠ 0: ({sx:.3e}, {sy:.3e})");
    }
}
