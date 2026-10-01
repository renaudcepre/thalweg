//! The moist upper layer and its own ~1 km torus: steps 1 to 2c of its
//! migration (design notes kept private).
//!
//! # Two modes, one switch
//!
//! [`MoistCoarseMode`] (`HEXSIM_MOIST_COARSE`, carried by
//! `Ablation::moist_coarse`, never read from the environment down here):
//!
//! - **`Fine`** (`0`, the ablation value since the 2026-09-30 flip): the
//!   historical fine pipeline, bit for bit. The coarse state is step 1's
//!   read-only mirror, re-gathered from the fine grid every hour so the
//!   water budget reads the same accessor either way, and nothing feeds
//!   back.
//! - **`CoarsePrecip`** (`1`, the compiled-in default since 2026-09-30
//!   — [`MOIST_COARSE_DEFAULT`] — also selected by `2`, see
//!   [`MoistCoarseMode::of`]):
//!   the fine grid keeps the entire moist physics — `humidity_upper` and
//!   `cloud_water` ARE the state — and the only thing that lives on the
//!   ~1 km torus is **the decision to precipitate and the sheet it
//!   drops**. See below.
//!
//! A third mode, `CoarseStock` (`2`, the step 2/2b variant: the coarse
//! state was the reference stock and the fine fields were **broadcast
//! views** of it), was retired 2026-09-07: step 2c's instrument (below)
//! measured it drying the plains by exposing pooled condensate to
//! subsaturated columns, `CoarsePrecip` replaced it, and it was never a
//! shipped configuration. `HEXSIM_MOIST_COARSE=2` stays truthy and now
//! selects `CoarsePrecip`, same as `1`.
//!
//! # `CoarsePrecip`: fine microphysics, coarse precipitation unit
//!
//! Nothing in the moist pipeline moves. The orographic pump,
//! `step_uplift`, both advections, the cloud diffusion, the imposed
//! regime, the vapour ↔ droplet transition and the radiative fog all run
//! on the fine grid, on the fine fields, exactly as on the `Fine` path,
//! with no view and no transfer. Then, once an hour
//! ([`step_moist_precip`]):
//!
//! 1. **Gather two numbers per coarse cell** from the fine
//!    `cloud_water`: the grid-box mean `q_c`
//!    ([`SynopticMesh::aggregate_mean`]) and the **in-cloud content**
//!    `q_in = Σ cw² / Σ cw`, the mean cloud water weighted by the cloud
//!    water itself ([`SynopticMesh::aggregate_mass_weighted_content`]) —
//!    what a droplet sits in, not what a column holds. It implies a
//!    cloud fraction `q_c / q_in ∈ (0, 1]`, no threshold and no counting.
//! 2. **Drain in-cloud** with the fine path's own function,
//!    `precipitation::precip_amount_mm(q_in, 1.0, …)` — `1.0` because the
//!    caller has already done the sub-grid reduction, `q_in` *is* the
//!    in-cloud content. The result is a per-cloud drain `d ≤ q_in`
//!    (guaranteed by the KK2000 closed form, no clamp), so
//!    `φ = d / q_in ∈ [0, 1]` **exactly in f32**, a rounded division of a
//!    numerator by a larger denominator. The global hysteresis gate is
//!    tested on the `N_c`-weighted map mean of `q_c` (design note §9
//!    risk 4: coarse cells have unequal fine-cell counts), and the ascent
//!    factor on the coarse mean of `convergence`.
//! 3. **Take the mass out of the cloudy columns, pro rata**: every fine
//!    cell of the coarse cell does `cw_i ← cw_i × (1 − φ)`. A column with
//!    no cloud loses nothing, a column with twice the cloud loses twice
//!    as much, and `1 − φ ∈ [0, 1]` makes a negative stock unreachable —
//!    the partition is exact by construction rather than guarded by a
//!    `.max(0.0)` (anti-pattern 4). The mass leaving the cell
//!    is `φ · Σ_{i∈c} cw_i = φ · N_c · q_c`.
//! 4. **Drop the sheet `P_c = q_c · φ` uniformly** on every fine cell of
//!    the coarse cell (`≤ q_c`, exactly, for the same reason as φ), with
//!    each fine cell deciding **its own** rain/snow phase from its own
//!    temperature, and the small-radius footprint rule
//!    ([`CoarseFootprint`]) unchanged. `N_c · P_c` is the mass step 3
//!    removed, to f32 rounding.
//! 5. The coarse stock is re-gathered as a mirror, like on the `Fine`
//!    path: it is a read-only accessor for the water budget, not state.
//!
//! Uniform, and not weighted by where the cloud was: the sheet falls at
//! ~1 km resolution because that is roughly how far a droplet drifts
//! while falling (0.4 to 3 km at this layer's thickness and wind), so
//! "the rain falls under its own 130 m column" is the artefact, not the
//! islet.
//!
//! ## What this abandons, and what it keeps
//!
//! Abandoned, deliberately:
//! - **the cloud is no longer "at 1 km" in the core.** The design note's
//!   target 2 (coherent clouds at render time) is now a *display*
//!   question: step 5 can send a coarse aggregation of the cloud (`q_c`,
//!   or the fraction `q_c / q_in`) on the wire without any physics
//!   moving.
//! - **most of the performance gain of the note's §2.** What is saved is
//!   the KK2000 post alone (~0.45 ms at r120); the transfer, the views
//!   and the coarse transition are gone with the stock.
//! - **step 3 (coarse advection) loses its stated purpose**: there is no
//!   coarse field left to advect.
//!
//! Kept: the rain in 1 km islets (the whole point of the effort),
//! strict conservation, the fine physics **intact** — no drying mechanism
//! anywhere — and both sentinels of step 2b (a cloud concentrated on a
//! fraction `f` of the box drains `f^(1−2.47)` more; the sheet never
//! exceeds the stock).
//!
//! ## The sub-grid closure, and why it stopped counting columns (#158)
//!
//! Until 2026-09-07 step 1 gathered a **counting** cloud fraction,
//! `f_c = #{i ∈ c : cw_i > 0} / N_c`, and took `q_in = q_c / f_c`. That
//! weighs a column holding 0.003 mm of the cloud diffusion's skirt
//! exactly as much as one holding 1.0 mm of cumulonimbus, so the skirt
//! inflates the count and `q_in` collapses. Measured at r8, on the
//! coarse cell of `phys_rain_footprint_is_a_disc` (13 fine cells, 11 of
//! them in the sample the JOURNAL quotes): one loaded column ringed by
//! six skirt columns gives `f_c` = 7/11 and `q_in` = 0.143 mm, under
//! `precip_crit_mm` = 0.15 mm — **a 1 mm cloud rains nothing**.
//!
//! `q_in` is now the **mass-weighted** content, `Σ cw² / Σ cw`
//! ([`SynopticMesh::aggregate_mass_weighted_content`]): the same skirt
//! now weighs what its mass weighs, and the same distribution gives
//! `q_in` = 0.982 mm. No threshold was introduced and nothing was
//! re-tuned: on a two-valued field (a cloud of one value over `k` of the
//! `N_c` columns, zero elsewhere) the implied fraction `q_c / q_in` is
//! exactly the old `k / N_c`, so this generalises the counting closure
//! rather than replacing it (pinned in `synoptic_mesh`'s tests). What it
//! bought, and what it did not, is measured in [`MOIST_COARSE_DEFAULT`]'s
//! doc.
//!
//! ## The measurement that sent the stock back to the fine grid
//!
//! Step 2b gave the coarse-stock path its condensation trigger and its
//! sub-grid autoconversion back, and the plains still lost 19 to 46 % of
//! their rain. Step 2c instrumented the two branches of the vapour ↔
//! droplet transition per altitude band
//! (`tests/diag_cloud_budget_by_altitude.rs`, r30, 30 days after a year,
//! seeds 42/7/123, fine path against coarse-stock):
//!
//! | band | condensation ×  | evaporation × | residence time × |
//! |---|---|---|---|
//! | < 300 m | 0.58 / 0.30 / 0.62 | **4.83 / 7.30 / 5.91** | 0.37 / 0.20 / 0.28 |
//! | 300-800 m | 1.16 / 1.50 / 1.21 | **8.75 / 6.79 / 6.59** | 0.31 / 0.37 / 0.30 |
//! | 800-1500 m | 5.01 / 3.90 / 2.79 | 3.47 / 5.01 / 6.04 | 0.79 / 0.65 / 0.52 |
//! | ≥ 1500 m | 20.1 / 9.20 / 7.85 | 2.54 / 3.28 / 2.79 | 4.25 / 1.25 / 1.30 |
//!
//! In the plains the coarse-stock path **condensed less and evaporated
//! five to seven times more**: the ratio evaporated/condensed went from
//! 0.41-0.54 on the fine path to 3.4-13.2, i.e. the plains cloud is
//! evaporated several times over, and a millimetre of cloud water lived
//! 5.0-6.3 h there against 1.25-2.2 h. That is the saturation adjustment
//! (`condensation::saturation_adjustment_transfer`, `min(cw, deficit)`)
//! applied to a **view that is not where the cloud is**: the condensate
//! of the few saturated columns is pooled and rebroadcast over ~46
//! columns, most of them subsaturated, and next hour each of them
//! evaporates its share back to vapour. The vapour is then pumped and
//! advected to the relief, where every column saturates — hence
//! condensation ×8 to ×20 and rain ×2 to ×31 above 1500 m, the exact
//! signature the bench and the r120 probe showed.
//!
//! No sub-grid closure fixes that: in a statistical cloud scheme
//! (Sundqvist 1978, *Mon. Wea. Rev.* 106; Tiedtke 1993, *Mon. Wea. Rev.*
//! 121, §2) the cloud lives inside its fraction `f_c` and is eroded by
//! mixing at a finite rate, never exposed instantly to the whole box's
//! deficit. The fine grid already holds that distribution, so the cheapest
//! correct closure is to stop destroying it — which is `CoarsePrecip`.
//!
//! # The coarse-stock mode, retired 2026-09-07 (`CoarseStock`)
//!
//! Steps 2 and 2b made the coarse state the reference stock of
//! `humidity_upper`/`cloud_water`, the fine fields its broadcast **views**:
//! a fine → coarse delta transfer each hour, KK2000 run on the coarse
//! stock, precipitation redistributed coarse → fine, the views refreshed
//! by broadcast (deliberately not barycentric interpolation, which leaked
//! mass through the toric seam's rounding — the reason a stock needs
//! `Σ_{i∈c} view_i = N_c × coarse_c` exactly and `view_i ≥ 0`, not merely
//! weights that sum to 1). Step 2b then gave that coarse pass back its own
//! condensation trigger and sub-grid autoconversion closure (the sub-grid
//! distribution measured on the fine grid, not assumed, exactly as
//! `CoarsePrecip` still does above — by counting columns then, by mass
//! since #158) to
//! close a 23-40 % plains rain deficit from a Jensen-inequality trigger
//! and a diluted autoconversion.
//!
//! What that bought, bench r30, three seeds, against step 2: rain days
//! above 1500 m 22/40/54 → 66/88/85 (fine path 107/115/128). What it did
//! **not** buy: the plains, still at −26/−19/−46 %. Step 2c's instrument,
//! above, says why — the broadcast exposes pooled condensate to
//! subsaturated columns, which no sub-grid closure on the coarse trigger
//! can fix. `CoarsePrecip` fixes it by never coarsening the state at all,
//! and the mode was deleted once M covered everything it bought without
//! that cost. Its code, the fine → coarse transfer and the broadcast
//! view refresh, is gone with it; `git log` on this file has the last
//! shipped revision if the numbers above ever need reproducing.
//!
//! # What stays fine on every mode, and why
//!
//! - `humidity_surface` (evaporation, transpiration and sublimation are
//!   surface processes), and with it the whole surface half of the water
//!   cycle.
//! - `T_upper` and `sat_upper` (design note §9 risk 1): making them coarse
//!   would replace `lapse(z_i)` by `lapse(z̄)` and erase the
//!   summit/valley saturation contrast inside a 1 km cell — undoing the
//!   2026-09-02 fix that gave the engine back its lapse rate and its
//!   mountain clouds.
//! - The orographic pump, `step_uplift` and the surface advection's lift:
//!   their forcing (`Δz` between fine neighbours) is irreducibly fine.
//! - The radiative fog (`fog.rs`, design note §9 risk 5).
//!
//! # Water accounting: one source of truth
//!
//! The reference budget is `Σ_i (surface stocks) + Σ_c N_c·(hu_c + cw_c) +
//! sky reservoir` — `Simulation::water_budget_total`, what `bench_metrics`
//! and the strict conservation tests read. On both modes the coarse half
//! is an exact mean gather of the fine fields, so it is the fine sum.
//!
//! # Cadence: unchanged
//!
//! This effort changes **space, not time** (design note §1, and the
//! three cadence ablations of 2026-09-05 that refused 1 h in 3 for the
//! precipitation and for the pump). Every pass here runs every hour, on
//! the same gate as the fine one it replaces.

use serde::{Deserialize, Serialize};

use crate::climate::DayRecord;
use crate::dynamics::CELL_SPACING_M;
use crate::grid::HexGrid;
use crate::par::{for_each_chunk_mut, for_each_chunk_mut2, reduce_blocks};
use crate::phase_timing::{elapsed_s, mark};
use crate::synoptic_mesh::{MIN_COARSE_RADIUS, SynopticMesh, round_coord};

use super::precipitation::{
    PrecipRates, precip_amount_mm, precip_spread_passes, spread_precip_further, update_precip_gate,
};
use super::regime::cell_count_f32;
use super::scaling::{
    precip_boosted_params, precip_runs_this_hour, precip_subsample,
    scale_atmosphere_for_hourly_tick,
};
use super::{AtmoScratch, AtmoState, AtmosphereParams, PrecipitationMap};

/// Target spacing (m) of the moist-layer coarse torus. A clean 1 km, not
/// [`crate::dynamics::SYNOPTIC_REFERENCE_SPACING_M`] (≈1074.569 m): that
/// constant is the shallow-water solver's **calibration** spacing (every
/// solver parameter in `dynamics.rs` was tuned and validated at it), not
/// the moist layer's natural scale. The two meshes are independent
/// instances for exactly this reason (design note §6): the day either
/// constant is recalibrated, the other mesh must not move with it.
pub(crate) const MOIST_COARSE_SPACING_M: f32 = 1000.0;

/// Which of the two moist-layer modes a world runs (see this module's
/// doc). Resolved once per process from `HEXSIM_MOIST_COARSE` through
/// [`crate::ablation::Ablation`], carried to the physics as an argument —
/// nothing down here reads the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoistCoarseMode {
    /// `HEXSIM_MOIST_COARSE=0`: the historical fine pipeline, bit for
    /// bit; the coarse state is a read-only mirror.
    Fine,
    /// `HEXSIM_MOIST_COARSE=1` or `=2`: fine microphysics, coarse
    /// precipitation unit (variant M, step 2c). `=2` used to select a
    /// third mode, the legacy coarse stock, retired 2026-09-07 (see this
    /// module's doc); [`MoistCoarseMode::of`] now folds it onto this one.
    CoarsePrecip,
}

impl MoistCoarseMode {
    /// The mode the ablation boolean names.
    #[must_use]
    pub fn of(coarse: bool) -> Self {
        if coarse {
            Self::CoarsePrecip
        } else {
            Self::Fine
        }
    }

    /// Does the ~1 km torus own the precipitation this hour? `true` on
    /// [`Self::CoarsePrecip`], which is exactly what
    /// `AtmoForcing::moist_coarse` means: `step_atmosphere_into` skips its
    /// own precipitation pass and the caller runs the coarse one instead.
    #[must_use]
    pub fn precipitates_coarse(self) -> bool {
        !matches!(self, Self::Fine)
    }
}

/// Compiled-in default for the coarse moist layer: **ON** since
/// 2026-09-30, i.e. [`MoistCoarseMode::CoarsePrecip`] — the ~1 km torus
/// owns the decision to precipitate and the sheet it drops, the fine grid
/// keeps the whole moist physics. `HEXSIM_MOIST_COARSE=0` selects
/// [`MoistCoarseMode::Fine`], the historical fine pipeline bit for bit,
/// and so does `Simulation::set_moist_coarse_mode` in-process.
///
/// # The flip (2026-09-30, #158 step 6)
///
/// Owner's decision, on the two measurements below and without a further
/// bench: the physics under `=1` is exactly what the 2026-09-07 rows
/// measured, this constant is the only thing that moved. What a consumer
/// inherits with it:
///
/// - a checkpoint exported between 2026-09-07 and the flip carries
///   `moist_coarse: false` and is now **refused** on load
///   ([`crate::checkpoint::CheckpointError::Ablation`], `moist_coarse:
///   self=false other=true`); `HEXSIM_MOIST_COARSE=0` loads it as it
///   was. The refusal is the contract, not a regression: resuming the
///   same seed in a different physics is what `ablation.rs` exists to
///   prevent.
/// - a checkpoint from before 2026-09-07 has no `moist_coarse` key
///   (the embed's `frontend/worlds/aged.ckptz` of 2026-08-26 among them)
///   and reads back as the defaults, so it resumes on a coarse mode it
///   was not produced under. Well defined: the fine fields are the state
///   on both modes and the mirror is a mean re-gathered from them — the
///   same statement `Simulation::set_moist_coarse_mode` makes mid-run.
/// - the cost measured below, +3.1 % on the r120 tick, and the hotspot
///   move to the relief noted under "What the `q_in` lever moved".
///
/// # Why it shipped off until then, measured (2026-09-07, step 2c)
///
/// `CoarsePrecip` fixes what steps 2 and 2b could not, and it was one
/// criterion short of the flip. Bench r30, 3 years, 3 seeds (42 / 7 /
/// 123), fine path against `CoarseStock` (`A+B`) and `CoarsePrecip`
/// (`M`):
///
/// | metric | seed | fine | A+B | M |
/// |---|---|---|---|---|
/// | `plains_precip_mm_per_day` | 42 | 0.1126 | −25.6 % | **+8.1 %** |
/// | | 7 | 0.0692 | −19.1 % | **+15.8 %** |
/// | | 123 | 0.0872 | −45.5 % | **+10.2 %** |
/// | `fully_rain_free_days_total` | 42 | 187 | 2.03× | 1.65× |
/// | | 7 | 148 | 2.43× | 1.95× |
/// | | 123 | 314 | 1.92× | 1.61× |
/// | `water_drift_pct` | 42/7/123 | 1.1e-5 / 1.3e-5 / 6.0e-6 | 1.1e-5 / 5.3e-6 / 3.2e-6 | 1.9e-6 / 1.8e-6 / 1.0e-6 |
/// | `cloud_cover_mountain_pct` | 42 | 0.847 | 1.00 (reads the broadcast) | 0.857 |
///
/// Shape at r120 (`just rain-pattern`, seed 42, day 545, 120 h), fine
/// against M: islets of 1-2 cells 39 % of patches → **0 %**, biggest
/// patch 14 → 46 cells (**×3.3**), patches per hour 40 → 4, Jaccard at
/// 1 h 0.66 → 0.90, painted fraction 0.8 → 1.0 % (+25 %, inside ±30 %),
/// and the rain is no longer displaced to the relief: painted fraction
/// below 300 m 0.5 → **0.7 %** (A+B took it to 0.0 %), above 1500 m
/// 3.3 → 2.9 %.
///
/// Against the flip criteria: `phys_kk2000_world_stays_humid` is green on
/// `CoarsePrecip` (it read 4 440 mm against its 5 000 mm floor on A+B),
/// `fully_rain_free_days_total` is inside 1×-2× on the three seeds,
/// `water_drift_pct` is 1e-6, the r120 shape criteria all pass. **The one
/// blocker is `plains_precip_mm_per_day` on seed 7: +15.8 % against a
/// ±15 % window, 0.8 point outside** — and outside on the wet side, the
/// plains getting more rain than the fine path, not less. Nothing was
/// re-tuned to close that gap.
///
/// ## What the `q_in` lever moved (2026-09-07, #158), same bench
///
/// Replacing the counting cloud fraction by the mass-weighted content
/// (this module's doc, "the sub-grid closure") moved both symptoms the
/// same way, on the same runs — one cause, two symptoms, as suspected:
///
/// | metric | seed | fine | M counting | M mass-weighted |
/// |---|---|---|---|---|
/// | `plains_precip_mm_per_day` | 42 | 0.1126 | +8.1 % | **+3.7 %** |
/// | | 7 | 0.0692 | +15.8 % | **+8.8 %** |
/// | | 123 | 0.0872 | +10.2 % | **+7.6 %** |
/// | `fully_rain_free_days_total` | 42 | 187 | 1.65× | **1.32×** |
/// | | 7 | 148 | 1.95× | **1.22×** |
/// | | 123 | 314 | 1.61× | **1.11×** |
/// | `rain_days_median_by_altitude` ≥1500 m | 42/7/123 | 107 / 115 / 128 | 73 / 94 / 97 | 77 / 99 / 100 |
/// | `water_drift_pct` | 42/7/123 | 1.1e-5 / 1.3e-5 / 6.0e-6 | 1.9e-6 / 1.8e-6 / 1.0e-6 | 7.5e-6 / 7.3e-6 / 1.0e-6 |
///
/// The plains window (±15 %) and the rain-free-day window (1× to 1.5×)
/// are both met on the three seeds, and the drift stays under the fine
/// path's own — though four times the counting closure's on two seeds,
/// which is what more water through the same hourly partition costs in
/// f32.
///
/// What the lever does **not** fix, measured: the two
/// `phys_rain_footprint_is_a_disc` fixtures stay red under
/// `HEXSIM_MOIST_COARSE=1`, and for a different reason than before. They
/// used to fail on "the source cell itself must still rain" (the floor,
/// which this lever removes); they now fail on the footprint — at r8 the
/// coarse cell holding the seeded column is 13 fine cells reaching
/// distance 2, so a sheet that covers its coarse cell cannot be a disc of
/// radius 1 or 3 centred on one fine column. That is [`CoarseFootprint`]'s
/// subject and the same class of statement as `phys_wet_peak_snows` at
/// r2: a fixture written against the fine footprint measures the
/// footprint, not the closure. Nothing was re-fixtured **here** — but
/// `CoarseFootprint` itself had a real bug behind that red, fixed the same
/// day: see its doc's "How many extra passes, corrected 2026-09-07" for
/// the `Rc` subtraction that makes both fixtures pass on the coarse path's
/// own terms, still per-mode, still without touching an assertion.
///
/// One shape metric moved the wrong way and is worth watching now that
/// the default is on: `rain_hotspots` (top-3 rainiest cells) sat in the basins on the
/// counting closure for seeds 42 and 7, as on the fine path; on the
/// mass-weighted one seed 7 goes from −186 m to +1121 m and seed 123 from
/// +605 m to +1422 m. Seed 42 stays in its basin (−57 m). A top-3 rank
/// statistic on three seeds, not a distribution — but the direction is
/// the relief.
///
/// Cost, r120, `HEXSIM_PERF_RADIUS=120 just perf-phases` ×2, year-2
/// window: the tick goes 4.62 → 4.76 ms (**+3.1 %**). The precipitation
/// post itself is cheaper (`atmo_precipitation` 0.447 → `atmo_moist_
/// coarse` 0.205 ms, −54 %) and so is the mirror gather (0.126 → 0.076,
/// the mesh being 817 coarse cells instead of 43 561), but M puts more
/// water on the ground and the non-atmosphere phases pay +0.12 ms for it.
/// The design note's §2 performance gain is gone with the coarse stock,
/// as this module's doc says; what is left is shape, not speed.
pub(crate) const MOIST_COARSE_DEFAULT: bool = true;

/// Coarse radius for the moist-layer mesh: same formula as
/// [`SynopticMesh::build`] (`round(R·CELL_SPACING_M/target).clamp(min(2,R),
/// R)`), but at [`MOIST_COARSE_SPACING_M`] instead of the synoptic
/// solver's reference spacing — see this module's doc and design note §6.
///
/// `r120` → Rc = 16 (817 cells, ≈53.3 fine cells/coarse, 975 m spacing);
/// `r250` → Rc = 33 (3367 cells, ≈55.9 fine cells/coarse, ≈984.8 m spacing);
/// `r30` → Rc = 4 (61 cells, ≈45.8 fine cells/coarse — the design note's
/// own worked example, §9 risk 2); `r ≤ 2` degenerates to the identity
/// (`Rc = R`, [`MIN_COARSE_RADIUS`] floor), same contract as the synoptic
/// mesh's own tiny-grid fallback. **The identity mesh is not the fine
/// pipeline**: a coarse mode still drops the neighbour share and the fine
/// footprint (a coarse cell that holds one fine cell is a point
/// footprint), so a micro-test at `r ≤ 2` is NOT blind to this chantier.
#[must_use]
pub(crate) fn moist_coarse_radius(fine_radius: i32) -> i32 {
    let target = f32::from(i16::try_from(fine_radius).unwrap_or(0)) * CELL_SPACING_M
        / MOIST_COARSE_SPACING_M;
    round_coord(target).clamp(MIN_COARSE_RADIUS.min(fine_radius), fine_radius)
}

/// The moist upper layer's coarse mirror: `humidity_upper` and
/// `cloud_water` on the coarse torus, in mm **per fine cell** (intensive,
/// so a coarse cell holding `N_c` fine cells carries `N_c × value` of
/// mass). On both live modes ([`MoistCoarseMode::Fine`],
/// [`MoistCoarseMode::CoarsePrecip`]) it is a pure, read-only mean of the
/// fine grid ([`Self::gather_from_fine`]), re-derived every hour rather
/// than accumulated — nothing here can drift from the fine state it
/// mirrors. Checkpointed anyway, alongside its mesh's radius
/// (`simulation/persistence.rs`), purely to skip that gather on load; a
/// checkpoint predating this field (`None`) just re-derives it.
///
/// (Was briefly prognostic, step 2 to 2b, on a third mode that made it the
/// reference stock and the fine fields its broadcast views — retired
/// 2026-09-07, see this module's doc.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct MoistCoarseState {
    /// Coarse `humidity_upper` (mm per fine cell), see this module's doc.
    pub(crate) humidity_upper: Vec<f32>,
    /// Coarse `cloud_water` (mm per fine cell), same unit.
    pub(crate) cloud_water: Vec<f32>,
}

impl MoistCoarseState {
    /// Zeroed state sized to `n_coarse` (`mesh.coarse_len()`).
    #[must_use]
    pub(crate) fn new(n_coarse: usize) -> Self {
        Self {
            humidity_upper: vec![0.0; n_coarse],
            cloud_water: vec![0.0; n_coarse],
        }
    }

    /// (Re)derives the coarse mirror from the fine grid: coarse mean of
    /// both fields via [`SynopticMesh::aggregate_mean`], same CSR gather,
    /// worker split and ascending-fine-index summation order as
    /// `aggregate_temperature`. Exact for a stock: the coarse value times
    /// its fine-cell count reproduces the fine sum.
    ///
    /// Called at construction (`Simulation::new`, a checkpoint that
    /// predates the coarse state), and every hour on both live modes
    /// ([`Self::water_total`]'s caller, `Simulation::step_moist_layer`):
    /// on `Fine` and `CoarsePrecip` the coarse state is a read-only
    /// mirror, so re-deriving it from the fine grid is exact and cheap
    /// (a few hundred to a few thousand coarse cells), never a leak the
    /// way re-deriving a *prognostic* stock from a non-conservative view
    /// would have been (the legacy coarse-stock path, retired
    /// 2026-09-07, accumulated fine deltas for exactly that reason).
    pub(crate) fn gather_from_fine(&mut self, mesh: &SynopticMesh, fine: &HexGrid) {
        let n = fine.len();
        let mut humidity = vec![0.0_f32; n];
        let mut cloud = vec![0.0_f32; n];
        for (i, cell) in fine.cells_slice().iter().enumerate() {
            humidity[i] = cell.humidity_upper;
            cloud[i] = cell.cloud_water;
        }
        mesh.aggregate_mean(&humidity, &mut self.humidity_upper);
        mesh.aggregate_mean(&cloud, &mut self.cloud_water);
    }

    /// The mass this mirror holds, in the same mm-summed-over-fine-cells
    /// unit as every other water stock of the terrarium:
    /// `Σ_c N_c·(hu_c + cw_c)`. The upper-layer half of
    /// `Simulation::water_budget_total` — see this module's doc ("Water
    /// accounting") for why this coarse gather, not a second sum of the
    /// fine fields, is the one accessor every reader goes through.
    #[must_use]
    pub(crate) fn water_total(&self, mesh: &SynopticMesh) -> f32 {
        let mut partials: Vec<f32> = Vec::new();
        let humidity = &self.humidity_upper;
        let cloud = &self.cloud_water;
        reduce_blocks(humidity.len(), &mut partials, |range| {
            let mut sum = 0.0_f32;
            for ci in range {
                sum += (humidity[ci] + cloud[ci]) * mesh.fine_count_f32(ci);
            }
            sum
        });
        partials.iter().fold(0.0_f32, |acc, &p| acc + p)
    }
}

/// Per-tick sub-phase durations of the coarse moist step, read back by the
/// caller right after the call and folded into `PhaseTimings` (same
/// pattern as [`crate::phase_timing::AtmoStepTimings`] on `AtmoScratch`):
/// not cumulative, fully overwritten at the top of every call.
///
/// `transfer` and `views` are always `0.0` since the legacy coarse-stock
/// path (the only one that had a fine → coarse transfer and a view
/// refresh to time) was retired 2026-09-07; `step_moist_precip` only
/// ever fills `coarse`. Left as three fields rather than trimmed to one:
/// `PhaseTimings::accumulate_moist` and its per-phase diag breakdown
/// already name the three rows, and a future third mode could refill the
/// other two without another schema change.
#[derive(Debug, Default, Clone, Copy)]
pub struct MoistCoarseTimings {
    /// Fine → coarse transfer: one delta sweep and two gathers.
    pub transfer: f64,
    /// The coarse pass itself (the cloud fraction gather and KK2000) and
    /// the coarse → fine precipitation distribution.
    pub coarse: f64,
    /// Refreshing the fine views: one broadcast write sweep.
    pub views: f64,
}

/// Scratch of the coarse moist step, owned by `AtmoScratch` and reused
/// every tick: content undefined between two ticks, like every other
/// buffer there. The prognostic stock is NOT here, it is
/// [`MoistCoarseState`].
#[derive(Default)]
pub struct MoistCoarseScratch {
    /// Fine-sized staging buffer: the extracted `cloud_water` the two
    /// gathers of [`step_moist_precip`] read.
    fine_cloud: Vec<f32>,
    /// Coarse-sized: the grid-box mean cloud water `q_c` of each coarse
    /// cell.
    cloud_mean: Vec<f32>,
    /// Coarse-sized: `1 − φ`, the share of its cloud water every fine
    /// column of the cell keeps after this hour's drain. In `[0, 1]` by
    /// construction, see [`step_moist_precip`].
    cloud_keep: Vec<f32>,
    /// Coarse-sized: mean fine temperature, filled only when the global
    /// precipitation gate is enabled (it is off by default) — its snow
    /// exception is the only thing on a coarse mode that needs a
    /// temperature, the precipitation phase itself being decided on the
    /// fine cell.
    temperature: Vec<f32>,
    /// Coarse-sized: mean fine ascent, filled only when the updraft
    /// trigger is active (`updraft_ref_ms > 0`, off by default).
    updraft: Vec<f32>,
    /// Coarse-sized: the in-cloud water content `q_in` of each coarse
    /// cell — `SynopticMesh::aggregate_mass_weighted_content` of
    /// `fine_cloud`, `Σ cw² / Σ cw`, the cloud water a droplet sits in
    /// rather than the one a column holds. KK2000's sub-grid closure, see
    /// [`drain_fine_cloud_by_coarse_cell`].
    cloud_in: Vec<f32>,
    /// Coarse-sized: the sheet (mm) each coarse cell drops this hour.
    precip: Vec<f32>,
    /// Block partials of the coarse reductions' deterministic sum.
    partials: Vec<f32>,
    /// This tick's sub-phase durations, read back by the caller.
    pub timings: MoistCoarseTimings,
}

/// Read-only inputs of [`step_moist_precip`], grouped per convention #61.
/// `params` is the caller's "per day" `AtmosphereParams`, scaled to the
/// hourly tick inside, exactly like `step_atmosphere_into` does with the
/// same struct — the scaling lives in one function, called twice.
pub(crate) struct MoistCoarseForcing<'a> {
    pub(crate) mesh: &'a SynopticMesh,
    pub(crate) params: &'a AtmosphereParams,
    pub(crate) hour_tick: u64,
}

/// One hour of the **coarse precipitation unit** ([`MoistCoarseMode::
/// CoarsePrecip`], variant M): the fine grid owns the whole moist
/// physics, and this is the only pass that sees the ~1 km torus. Called
/// by the orchestrator right after `step_atmosphere_into` returns, on
/// `next`, before the `current`/`next` swap; the caller re-gathers the
/// coarse mirror afterwards.
///
/// See this module's doc for the four steps and the exactness argument
/// that makes `φ ∈ [0, 1]` a construction rather than a clamp.
pub(crate) fn step_moist_precip(
    next: &mut HexGrid,
    forcing: &MoistCoarseForcing<'_>,
    atmo_state: &mut AtmoState,
    scratch: &mut AtmoScratch,
    events: &mut PrecipitationMap,
) {
    let MoistCoarseForcing {
        mesh,
        params,
        hour_tick,
    } = *forcing;
    let params = scale_atmosphere_for_hourly_tick(params);
    scratch.moist.timings = MoistCoarseTimings::default();

    let t0 = mark();
    // Same cadence gate as the fine pass it replaces
    // (`PRECIP_SUBSAMPLE_HOURS`, hourly by default and measured to refuse
    // anything else, JOURNAL 2026-09-05). `events` was already zeroed by
    // `step_atmosphere_into` before the gate.
    let precip_sub = precip_subsample();
    if precip_runs_this_hour(hour_tick, precip_sub) {
        let params_precip = precip_boosted_params(&params, precip_sub);
        drain_fine_cloud_by_coarse_cell(
            next,
            mesh,
            &params_precip,
            f32::from(precip_sub),
            atmo_state,
            scratch,
        );
        distribute_precipitation(
            next,
            mesh,
            &CoarseFootprint::of(mesh, &params_precip),
            scratch,
            events,
        );
    }
    scratch.moist.timings.coarse += elapsed_s(t0);
}

/// Steps 1 to 3 of [`MoistCoarseMode::CoarsePrecip`]: read the fine cloud,
/// decide the drain once per coarse cell, take it back out of the fine
/// columns pro rata, and leave the sheet in `scratch.moist.precip` for
/// [`distribute_precipitation`].
///
/// # Why `precip_amount_mm` is called with `cloud_fraction = 1.0`
///
/// That parameter exists so a caller holding a **grid-box mean** can say
/// what share of the box the droplets occupy. Here the caller has already
/// done that reduction: `q_in` (`Σ cw² / Σ cw`, see this module's doc)
/// **is** the in-cloud content, so the function is being handed the
/// content of the cloud itself and there is no further sub-grid structure
/// to declare. The fraction is applied on the way out, as
/// `P_c = q_c · φ`, which is the same closure written the other way
/// round: `q_c / q_in` is the fraction, so `q_c · φ = (q_c / q_in) · d`.
///
/// Writing it that way is not cosmetic, it is what makes the partition
/// exact. `precipitation::kk2000_autoconv_over_hours` returns at most its
/// own argument **in f32** (`q_end = q₀ × factor` with `factor ≤ 1`, so
/// `q₀ − q_end ≤ q₀` with no rounding slack), the per-pass cap and the
/// ascent factor only lower it, so `d ≤ q_in` and the rounded division
/// `φ = d / q_in` lands in `[0, 1]` — a correctly rounded quotient of a
/// numerator no larger than its denominator cannot exceed 1. **That is
/// the whole conservation argument, and it does not depend on how `q_in`
/// is measured**: `φ ∈ [0, 1]` gives `cw_i × (1 − φ) ≥ 0` on every
/// column and `P_c = q_c · φ ≤ q_c` on the sheet, so a change of closure
/// (#158 moved `q_in` from `q_c / f_c` to the mass-weighted content)
/// cannot open a rounding hole here. Going through
/// `amount = precip_amount_mm(q_c, q_c / q_in, …)` and dividing by `q_c`
/// would NOT have that property: `(q_c / f) × f` is not `q_c` in f32, so
/// the ratio could land one ULP above 1 and take a column's cloud water
/// negative — which is exactly the kind of rounding hole step 2's
/// interpolated views fell into (see this module's doc).
fn drain_fine_cloud_by_coarse_cell(
    next: &mut HexGrid,
    mesh: &SynopticMesh,
    params: &AtmosphereParams,
    dt_hours: f32,
    atmo_state: &mut AtmoState,
    scratch: &mut AtmoScratch,
) {
    let n = next.len();
    let n_coarse = mesh.coarse_len();
    scratch.moist.fine_cloud.clear();
    scratch.moist.fine_cloud.resize(n, 0.0);
    for (dst, cell) in scratch
        .moist
        .fine_cloud
        .iter_mut()
        .zip(next.cells_slice().iter())
    {
        *dst = cell.cloud_water;
    }
    // `q_c`, the grid-box mean, and `q_in`, the same cloud water weighted
    // by itself — no threshold and no counting. The fine grid hands over
    // the real sub-grid distribution every hour, for the price of two
    // gathers; a GCM has to posit one (Sundqvist 1978, Smith 1990,
    // Tiedtke 1993).
    scratch.moist.cloud_mean.clear();
    scratch.moist.cloud_mean.resize(n_coarse, 0.0);
    mesh.aggregate_mean(&scratch.moist.fine_cloud, &mut scratch.moist.cloud_mean);
    scratch.moist.cloud_in.clear();
    scratch.moist.cloud_in.resize(n_coarse, 0.0);
    mesh.aggregate_mass_weighted_content(&scratch.moist.fine_cloud, &mut scratch.moist.cloud_in);

    let mean_cloud =
        weighted_coarse_mean(&scratch.moist.cloud_mean, mesh, &mut scratch.moist.partials);
    update_precip_gate(&mut atmo_state.precip_gate_open, params, mean_cloud);
    let rates = PrecipRates::new(params, dt_hours, !atmo_state.precip_gate_open);
    let cold = fill_coarse_precip_forcings(next, mesh, params, &rates, scratch);

    scratch.moist.precip.clear();
    scratch.moist.precip.resize(n_coarse, 0.0);
    scratch.moist.cloud_keep.clear();
    scratch.moist.cloud_keep.resize(n_coarse, 1.0);
    for ci in 0..n_coarse {
        let q_c = scratch.moist.cloud_mean[ci];
        let in_cloud = scratch.moist.cloud_in[ci];
        // Both are 0 on a coarse cell holding no cloud at all: there is
        // nothing to drain and nothing to divide by. Not a fallback value,
        // an empty cell. `in_cloud` is tested rather than assumed from
        // `q_c` because `Σ cw²` can underflow to zero while `Σ cw` is
        // still a denormal positive — a cloud below 1e-22 mm, which is
        // the same "no cloud here" and must not reach the division.
        if q_c <= 0.0 || in_cloud <= 0.0 {
            continue;
        }
        let temperature = if cold {
            scratch.moist.temperature[ci]
        } else {
            0.0
        };
        let w = if rates.uses_updraft() {
            scratch.moist.updraft[ci]
        } else {
            0.0
        };
        let drained = precip_amount_mm(in_cloud, 1.0, temperature, w, &rates);
        let phi = drained / in_cloud;
        debug_assert!(
            (0.0..=1.0).contains(&phi),
            "coarse cell {ci}: drained {drained} out of an in-cloud content of {in_cloud}"
        );
        scratch.moist.precip[ci] = q_c * phi;
        scratch.moist.cloud_keep[ci] = 1.0 - phi;
    }

    // Pro rata, per fine column: a column with no cloud loses nothing, a
    // column with twice the cloud loses twice as much, and `1 − φ ≥ 0`
    // makes a negative stock unreachable without a guard.
    let keep: &Vec<f32> = &scratch.moist.cloud_keep;
    let fine_to_coarse = mesh.fine_to_coarse();
    for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
        for (local, cell) in chunk.iter_mut().enumerate() {
            cell.cloud_water *= keep[fine_to_coarse[start + local]];
        }
    });
}

/// The two optional per-coarse-cell forcings [`drain_fine_cloud_by_coarse_cell`]
/// reads; both are off at the shipped defaults, and neither buffer is
/// even sized then.
///
/// Returns whether the coarse temperature was filled. That temperature
/// only feeds the global gate's snow exception ("snow is always allowed",
/// so a closed gate cannot stop a winter snowpack from building) — the
/// precipitation phase itself is decided per fine cell in
/// [`distribute_precipitation`]. The gate is off at the shipped default
/// (`global_precip_gate = 0.0`), so this gather normally never runs.
fn fill_coarse_precip_forcings(
    next: &HexGrid,
    mesh: &SynopticMesh,
    params: &AtmosphereParams,
    rates: &PrecipRates,
    scratch: &mut AtmoScratch,
) -> bool {
    let n_coarse = mesh.coarse_len();
    let cold = params.global_precip_gate > 0.0;
    if cold {
        scratch.moist.temperature.clear();
        scratch.moist.temperature.resize(n_coarse, 0.0);
        let fine_t: Vec<f32> = next.cells_slice().iter().map(|c| c.temperature).collect();
        mesh.aggregate_mean(&fine_t, &mut scratch.moist.temperature);
    }
    // Mean fine ascent per coarse cell, only when the trigger is active
    // (`updraft_ref_ms > 0`, off by default — then the factor is 1
    // everywhere and `scratch.convergence` is not even filled).
    if rates.uses_updraft() {
        scratch.moist.updraft.clear();
        scratch.moist.updraft.resize(n_coarse, 0.0);
        mesh.aggregate_mean(&scratch.convergence, &mut scratch.moist.updraft);
    }
    cold
}

/// `Σ_c N_c·field_c / N_fine`, the mean a fine-grid pass would have
/// measured, deterministic whatever the thread count
/// (`par::reduce_blocks`). `None` on an empty torus.
fn weighted_coarse_mean(
    field: &[f32],
    mesh: &SynopticMesh,
    partials: &mut Vec<f32>,
) -> Option<f32> {
    if field.is_empty() {
        return None;
    }
    reduce_blocks(field.len(), partials, |range| {
        let mut sum = 0.0_f32;
        for ci in range {
            sum += field[ci] * mesh.fine_count_f32(ci);
        }
        sum
    });
    let total = partials.iter().fold(0.0_f32, |acc, &p| acc + p);
    Some(total / cell_count_f32(mesh.fine_len()))
}

/// Cells in a hex disc of `rings` rings, `3k(k+1)+1`, capped by the
/// torus: on a small grid the disc wraps and cannot cover more than the
/// whole map. That cap is not cosmetic — at r2 (19 fine cells) the fine
/// path's radius-3 footprint covers the entire torus, which is exactly
/// why `phys_wet_peak_snows`'s 80 mm floor exists at all (JOURNAL
/// 2026-09-06, step 2).
fn disc_cells(rings: u32, fine_len: usize) -> usize {
    let k = usize::try_from(rings).expect("ring count fits usize");
    (3 * k * (k + 1) + 1).min(fine_len)
}

/// **The coarse path's footprint is never smaller than the fine path's**
/// (step 2b). Two footprints are in play:
///
/// - fine: a disc of `precip_spread_radius` rings around the source
///   column, `3k(k+1)+1` cells (37 at the shipped radius 3);
/// - coarse: the coarse cell itself, `N_c` fine cells — the sheet falls
///   uniformly on it and nowhere else.
///
/// From r30 up, `N_c` is 46 to 56 and the coarse cell is already the
/// wider of the two: the rule is inert and nothing runs. Below that it is
/// not, and the difference is not a detail. At r8, Rc = 2 gives ~11 fine
/// cells per coarse; at r ≤ 2 the mesh degenerates to the identity and a
/// "coarse" cell is **one** fine cell, a point footprint against the fine
/// path's whole-torus disc. A micro-test written against the fine
/// footprint then measures the footprint, not the physics it was aimed at.
///
/// So when `N_c` is below the disc — a **cell-count** comparison,
/// `disc_cells(passes, fine_len)` against `mean_fine_per_coarse`, both
/// plain counts — the distributed fine sheet gets extra applications of
/// `spread_precip_further`'s isotropic rule: the fine path's own
/// operator, doubly stochastic, mass-conserving for any number of passes
/// (see its doc).
///
/// **How many extra passes, corrected 2026-09-07**
/// (`phys_rain_footprint_is_a_disc`, filed against this exact function):
/// the sheet does not start from a point the way the fine path's does —
/// it starts already spread over the whole coarse cell, out to
/// `mesh.grid().radius()` fine-hex rings from its center (`Rc`; e.g. 2 at
/// r8, `SynopticMesh::with_coarse_radius`'s own radius parameter, not a
/// value this function recomputes). Running the *full* `passes` count on
/// top of that — the shipped code before this fix, on the claim "the
/// reach is the same `passes` rings" as the fine path — is true only for
/// a point source. Measured instead (probe in
/// `simulation::tests`, r8, `precip_spread_radius = 3`, one column
/// saturated): the coarse cell alone covers out to ring 2 (13 fine
/// cells), and 3 more full passes on top of it wet 24 cells at ring 4 and
/// 24 more at ring 5 — a footprint reaching ring 5, not the ring 3 the
/// parameter names. anti-pattern 3 (stock/flux confusion) in
/// its geometric form: the trigger condition above is legitimately a
/// cell count, but the passes it then spent were read as a ring radius
/// starting from a point that was never there. The fix subtracts the
/// rings the coarse cell already covers (`Rc`) from `passes` before
/// running the rest: r8, radius 3, Rc = 2 → `3 − 2 = 1` extra pass →
/// reach `2 + 1 = 3`, equal to the fine path's own disc; r8, radius 1,
/// Rc = 2 → `precip_spread_passes(1).saturating_sub(2) = 0` extra passes
/// → reach stays at the coarse cell's own 2 rings, incompressible below
/// what one coarse cell already covers. `saturating_sub` because `Rc` can
/// exceed `passes` (as it does here at radius 1); a negative pass count
/// has no meaning. From r30 up the trigger above is already false
/// (`N_c` = 46 > 37), so `Rc` is never subtracted from anything there —
/// this fix changes zero passes on any radius the project ships at.
///
/// `N_c` is taken as the mesh's **mean** (`fine_len / coarse_len`), the
/// number the design note and this module quote everywhere ("≈45.8 fine
/// cells per coarse at r30"). A per-coarse-cell decision would make the
/// operator non-uniform over the map for a spread of a few percent in the
/// counts, and `spread_precip_further` is a whole-grid pass either way.
struct CoarseFootprint {
    /// `precip_neighbor_share`, the fraction each pass moves outward.
    share: f32,
    /// Diffusion applications to run on the distributed sheet; `0` when
    /// the coarse cell is already the wider footprint, or when its own
    /// radius (`Rc`) already covers as many rings as `precip_spread_radius`
    /// would have asked for on its own.
    passes: u32,
}

impl CoarseFootprint {
    fn of(mesh: &SynopticMesh, params: &AtmosphereParams) -> Self {
        let passes = precip_spread_passes(params.precip_spread_radius);
        let fine_len = mesh.fine_len();
        let coarse_len = mesh.coarse_len().max(1);
        let mean_fine_per_coarse = fine_len / coarse_len;
        // Rings the coarse cell already covers from its own center
        // (`Rc`, this module's doc): never negative by construction
        // (`SynopticMesh::with_coarse_radius`'s `rc` is clamped at
        // `MIN_COARSE_RADIUS` and up), `unwrap_or(0)` is unreachable
        // defensive code, not a silent fallback that could mask a bug.
        let already_covered = u32::try_from(mesh.grid().radius()).unwrap_or(0);
        Self {
            share: params.precip_neighbor_share.clamp(0.0, 1.0),
            passes: if mean_fine_per_coarse < disc_cells(passes, fine_len) {
                passes.saturating_sub(already_covered)
            } else {
                0
            },
        }
    }
}

/// Step 4 of this module's doc: the coarse sheet falls on every fine cell
/// of its coarse cell, uniformly, each fine cell deciding its own
/// rain/snow phase from its own temperature, then — only when
/// [`CoarseFootprint`] says the coarse cell is the narrower footprint —
/// the fine sheet is feathered outward by the fine path's own diffusion
/// rule. Conservative by construction on both counts: `N_c` fine cells ×
/// `P_c` mm is exactly the `N_c·P_c` of mass the coarse cell lost, and
/// `spread_precip_further`'s operator is doubly stochastic.
///
/// A uniform field is a fixed point of that operator (row sums are 1), so
/// when the rule fires it only reshapes the **edge** of the coarse cell's
/// sheet — the interior is untouched. At the identity mesh (`r ≤ 2`) the
/// sheet is a point mass and the operator rebuilds the fine path's
/// footprint exactly.
fn distribute_precipitation(
    next: &mut HexGrid,
    mesh: &SynopticMesh,
    footprint: &CoarseFootprint,
    scratch: &mut AtmoScratch,
    events: &mut PrecipitationMap,
) {
    let n = next.len();
    events.resize(n, DayRecord::default());
    scratch.precip_water_delta.clear();
    scratch.precip_water_delta.resize(n, 0.0);
    scratch.precip_snow_delta.clear();
    scratch.precip_snow_delta.resize(n, 0.0);

    let fine_to_coarse = mesh.fine_to_coarse();
    let precip: &Vec<f32> = &scratch.moist.precip;
    let cells = next.cells_slice();
    for_each_chunk_mut2(
        &mut scratch.precip_water_delta,
        &mut scratch.precip_snow_delta,
        |start, rain_chunk, snow_chunk| {
            for local in 0..rain_chunk.len() {
                let i = start + local;
                let amount = precip[fine_to_coarse[i]];
                // Phase decided here and not on the coarse column
                // (design note §4 option (d), not optional): a 1 km cell
                // spans 300 to 600 m of relief at this terrain's slopes,
                // so a fine cell below 0 °C takes snow while its
                // neighbour above takes rain, out of the same sheet.
                let is_snow = cells[i].temperature < 0.0;
                rain_chunk[local] = if is_snow { 0.0 } else { amount };
                snow_chunk[local] = if is_snow { amount } else { 0.0 };
            }
        },
    );

    spread_precip_further(
        next,
        footprint.share,
        footprint.passes,
        &mut scratch.precip_water_delta,
        &mut scratch.precip_snow_delta,
        &mut scratch.precip_water_delta_tmp,
        &mut scratch.precip_snow_delta_tmp,
    );

    let rain_delta: &Vec<f32> = &scratch.precip_water_delta;
    let snow_delta: &Vec<f32> = &scratch.precip_snow_delta;
    for_each_chunk_mut2(
        next.cells_slice_mut(),
        events,
        |start, cells_chunk, events_chunk| {
            for (local, nc) in cells_chunk.iter_mut().enumerate() {
                let i = start + local;
                let rain = rain_delta[i];
                let snow = snow_delta[i];
                if rain <= 0.0 && snow <= 0.0 {
                    continue;
                }
                nc.water_level += rain;
                nc.snow_level += snow;
                events_chunk[local].rain += rain;
                events_chunk[local].snow += snow;
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fine_grid(radius: i32) -> HexGrid {
        HexGrid::from_radius(radius)
    }

    #[test]
    fn moist_coarse_radius_matches_the_worked_examples() {
        // r120 -> 16 (817 cells), r250 -> 33 (3367 cells), r30 -> 4 (61
        // cells, the design note's own example, ≈45.8 fine/coarse):
        // computed and documented on `moist_coarse_radius`'s own doc.
        for (r, rc_expected, cells_expected) in [(120, 16, 817), (250, 33, 3367), (30, 4, 61)] {
            let rc = moist_coarse_radius(r);
            assert_eq!(rc, rc_expected, "r={r}");
            let cells = HexGrid::from_radius(rc).len();
            assert_eq!(cells, cells_expected, "r={r} Rc={rc}");
        }
    }

    #[test]
    fn moist_coarse_radius_degenerates_to_identity_below_the_floor() {
        for r in [0, 1, 2] {
            assert_eq!(moist_coarse_radius(r), r, "r={r}");
        }
    }

    #[test]
    fn gather_from_fine_matches_a_direct_aggregate_mean() {
        let mut fine = fine_grid(30);
        let mut state = 0x5eed_c0de_u32;
        for c in fine.cells_slice_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            c.humidity_upper = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 5.0;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            c.cloud_water = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 2.0;
        }
        let rc = moist_coarse_radius(fine.radius());
        let mesh = SynopticMesh::with_coarse_radius(&fine, rc);
        let mut mirror = MoistCoarseState::new(mesh.coarse_len());
        mirror.gather_from_fine(&mesh, &fine);

        let humidity: Vec<f32> = fine
            .cells_slice()
            .iter()
            .map(|c| c.humidity_upper)
            .collect();
        let cloud: Vec<f32> = fine.cells_slice().iter().map(|c| c.cloud_water).collect();
        let mut expected_humidity = vec![0.0_f32; mesh.coarse_len()];
        let mut expected_cloud = vec![0.0_f32; mesh.coarse_len()];
        mesh.aggregate_mean(&humidity, &mut expected_humidity);
        mesh.aggregate_mean(&cloud, &mut expected_cloud);

        assert_eq!(mirror.humidity_upper, expected_humidity);
        assert_eq!(mirror.cloud_water, expected_cloud);
    }

    /// The gather is exact for a stock: the coarse value times its
    /// fine-cell count gives the fine sum back, so `water_total` is the
    /// mass the fine grid held. r30 (Rc = 4, 45.8 fine cells per coarse),
    /// **not** on the identity path.
    #[test]
    fn water_total_is_the_fine_sum_after_a_gather() {
        let mut fine = fine_grid(30);
        let mut state = 0x1357_9bdf_u32;
        for c in fine.cells_slice_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            c.humidity_upper = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 8.0;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            c.cloud_water = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0 * 2.0;
        }
        let mesh = SynopticMesh::with_coarse_radius(&fine, moist_coarse_radius(fine.radius()));
        assert_eq!(mesh.grid().radius(), 4, "r30 must give Rc = 4");
        let mut stock = MoistCoarseState::new(mesh.coarse_len());
        stock.gather_from_fine(&mesh, &fine);

        let fine_sum: f64 = fine
            .cells_slice()
            .iter()
            .map(|c| f64::from(c.humidity_upper) + f64::from(c.cloud_water))
            .sum();
        let coarse_sum = f64::from(stock.water_total(&mesh));
        let rel = (coarse_sum - fine_sum).abs() / fine_sum;
        assert!(
            rel < 1e-5,
            "fine {fine_sum} vs coarse {coarse_sum} (rel {rel})"
        );
    }

    /// The partition itself, at the arithmetic level and in one hour
    /// rather than through 48 h of full physics (that is
    /// `tests/phys_coarse_upper_layer.rs`'s sentinel (ii)): whatever
    /// `q_in` measures, `φ ∈ [0, 1]` must hold on every coarse cell, the
    /// sheet must stay under the box mean, no column may go negative, and
    /// the mass the fine columns lost must be the mass the sheets carry.
    ///
    /// This is the property the #158 closure change had to preserve, and
    /// the reason it could be changed at all: conservation rests on
    /// `d ≤ q_in`, not on how `q_in` was gathered
    /// ([`drain_fine_cloud_by_coarse_cell`]'s doc).
    ///
    /// r6, `Rc = 2` (19 coarse cells, ≈6.7 fine each — a real coarse
    /// torus, not the identity), on a deliberately spiky field: a loaded
    /// column, a thin diffusion skirt and exact zeros, the distribution
    /// the counting closure could not read.
    #[test]
    fn the_drain_is_an_exact_partition_whatever_the_in_cloud_content() {
        let mut fine = fine_grid(6);
        let mut state = 0x0c0f_fee5_u32;
        for c in fine.cells_slice_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0;
            c.temperature = 15.0;
            c.cloud_water = match state % 4 {
                0 => 0.0,
                1 => u * 6.0,
                _ => u * 0.004,
            };
        }
        let mesh = SynopticMesh::with_coarse_radius(&fine, moist_coarse_radius(fine.radius()));
        assert_eq!(mesh.grid().radius(), 2, "r6 must give Rc = 2");

        let before: f64 = fine
            .cells_slice()
            .iter()
            .map(|c| f64::from(c.cloud_water))
            .sum();
        let mut atmo_state = AtmoState::default();
        let mut scratch = AtmoScratch::new(fine.len());
        let params = scale_atmosphere_for_hourly_tick(&AtmosphereParams::default());
        drain_fine_cloud_by_coarse_cell(
            &mut fine,
            &mesh,
            &params,
            1.0,
            &mut atmo_state,
            &mut scratch,
        );

        let mut drained_cells = 0_usize;
        let mut sheet_mass = 0.0_f64;
        for ci in 0..mesh.coarse_len() {
            let keep = scratch.moist.cloud_keep[ci];
            let sheet = scratch.moist.precip[ci];
            let q_c = scratch.moist.cloud_mean[ci];
            assert!(
                (0.0..=1.0).contains(&keep),
                "coarse cell {ci}: kept share {keep} outside [0, 1]"
            );
            assert!(
                sheet <= q_c,
                "coarse cell {ci}: sheet {sheet} above the box mean {q_c}"
            );
            if sheet > 0.0 {
                drained_cells += 1;
            }
            sheet_mass += f64::from(sheet) * f64::from(mesh.fine_count_f32(ci));
        }
        assert!(
            drained_cells > 0,
            "the fixture must make at least one coarse cell rain"
        );

        let after: f64 = fine
            .cells_slice()
            .iter()
            .map(|c| f64::from(c.cloud_water))
            .sum();
        let lowest = fine
            .cells_slice()
            .iter()
            .map(|c| c.cloud_water)
            .fold(f32::INFINITY, f32::min);
        assert!(lowest >= 0.0, "a column holds {lowest} mm of cloud water");
        let removed = before - after;
        assert!(
            (removed - sheet_mass).abs() <= 1e-5 * before,
            "the columns lost {removed} mm and the sheets carry {sheet_mass} mm \
             (of {before} mm held)"
        );
    }

    /// Pins the reach fix (this module's doc, "How many extra passes,
    /// corrected 2026-09-07"): at r8 (`Rc = 2`, the trigger `13 < 37`
    /// fires), the extra passes are `precip_spread_passes(radius)` minus
    /// the 2 rings the coarse cell already covers, not the raw pass
    /// count. Before this fix `passes` was 3 and 1 respectively (no
    /// subtraction), which is exactly the bug
    /// `phys_rain_footprint_is_a_disc` caught: reach 5 instead of 3 at
    /// the shipped default.
    #[test]
    fn footprint_extra_passes_subtract_the_rings_the_coarse_cell_already_covers() {
        let fine = fine_grid(8);
        let mesh = SynopticMesh::with_coarse_radius(&fine, moist_coarse_radius(fine.radius()));
        assert_eq!(mesh.grid().radius(), 2, "r8 must give Rc = 2");

        let default_radius = AtmosphereParams {
            precip_spread_radius: 3.0,
            ..AtmosphereParams::default()
        };
        assert_eq!(
            CoarseFootprint::of(&mesh, &default_radius).passes,
            1,
            "radius 3, Rc 2: 3 rings asked minus 2 already covered = 1 extra pass"
        );

        let radius_one = AtmosphereParams {
            precip_spread_radius: 1.0,
            ..AtmosphereParams::default()
        };
        assert_eq!(
            CoarseFootprint::of(&mesh, &radius_one).passes,
            0,
            "radius 1, Rc 2: the coarse cell alone already covers more than asked, \
             saturating_sub floors at 0"
        );
    }

    /// From r30 up the trigger condition (`N_c < disc_cells`) is already
    /// false on its own (46 fine cells per coarse against 37), so the
    /// `Rc` subtraction never runs there: this fix must leave that branch
    /// bit-for-bit alone. Companion to the r8 test above, same doc.
    #[test]
    fn footprint_stays_inert_at_r30_regardless_of_the_rc_subtraction() {
        let fine = fine_grid(30);
        let mesh = SynopticMesh::with_coarse_radius(&fine, moist_coarse_radius(fine.radius()));
        assert_eq!(mesh.grid().radius(), 4, "r30 must give Rc = 4");
        assert_eq!(
            CoarseFootprint::of(&mesh, &AtmosphereParams::default()).passes,
            0,
            "r30 is past the design note's own inertness threshold, the rule must not fire"
        );
    }
}
