//! Imposed weather regime (#63): the synoptic scale as a boundary
//! condition, like the sun.
//!
//! # Why the box cannot make its own dry spells
//!
//! The terrarium is closed and ~2 mm/day evaporate everywhere, so that
//! water has to fall back somewhere. The orographic pump lifts vapour to
//! the summits every hour, the summit sits at RH 1.0 around the clock and
//! KK2000 drains it as it arrives: a stationary condenser, and it rains
//! somewhere on the map every single day of the year
//! (`fully_rain_free_days_total = 0` on three seeds since 2026-07-08).
//! Three fixes were measured and refuted (JOURNAL 2026-09-02/03): tuning
//! any atmosphere coefficient (7 ablations, not one dry day), a per-cell
//! threshold (`precip_crit_mm`: the cells are independent, one is always
//! above), and growing depressions inside the box (an anticyclone is
//! ~1000 km, the box 8 km at r30). No intermittence is possible without a
//! regime imposed on the *whole* map at once.
//!
//! # What this models
//!
//! A real dry spell comes from large-scale subsidence: air descends over
//! the region, the free troposphere dries out, the column's vapour leaves
//! horizontally and comes back with the next system. That circulation
//! lives outside the box, so it enters the model the way the sun does, as
//! a forcing — and the terrarium invariant is kept by giving the vapour
//! that leaves somewhere to be: a single "sky" reservoir, in the same
//! millimetres as the cell stocks, so `terrarium water + sky = constant`.
//!
//! - **Episode generator**: a two-state Markov chain {Dry, Wet}, one
//!   transition per simulated day, drawn from a hash of (world seed, day)
//!   rather than a stateful RNG, so a checkpoint restart replays the same
//!   weather ([`crate::hashing`]).
//! - **Dry phase**: each hour, each cell's `humidity_upper` relaxes toward
//!   `regime_dry_rh_target × saturation_upper(T_upper)`; the surplus goes
//!   to the sky. Existing `cloud_water` is untouched: it rains out or
//!   travels, which is what actually clears a sky over a few hours.
//! - **Wet phase**: the sky comes back, spread uniformly over the cells (a
//!   moist air mass arriving over the whole box).
//! - **Nothing else moves**: pump, KK2000, condensation and advection are
//!   not touched, and the pass is skipped whole when `regime_enabled == 0`.
//!
//! **Default ON since 2026-09-06 (#63/#146).** Measured first, then
//! shipped as the coherent package with the L2b saturation-adjustment
//! cloud evaporation; see [`AtmosphereParams::regime_enabled`] for the
//! numbers and the one-line way back to the old stationary atmosphere.

use serde::{Deserialize, Serialize};

use crate::grid::HexGrid;
use crate::hashing;
use crate::par::{for_each_chunk_mut, for_each_chunk_mut2, reduce_blocks};

use super::AtmosphereParams;

/// Salt of the regime's draw stream, so it can never move together with
/// the fire's (`crate::fire`, salts 0 and 1) even on the same seed and
/// the same day.
const REGIME_SALT: u64 = 0x5EA5_0000_0000_0063;

/// Persistent state of the imposed weather regime: which phase the map is
/// in, and the vapour currently held outside the box.
///
/// Both are real state, not derivable from the grid, so both are
/// checkpointed (same precedent as `precip_gate_open` and
/// `Simulation::upper_air_mean_t`): without `wet` a restart would redraw
/// the phase mid-episode, and without `sky_water_mm` the terrarium would
/// silently gain or lose everything the sky was holding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WeatherRegime {
    /// `true` during a wet episode (the sky returns its water),
    /// `false` during a dry one (the columns export to the sky).
    pub wet: bool,
    /// Vapour held outside the box (mm, summed over the map exactly like
    /// the per-cell stocks, so it adds straight into a water budget).
    pub sky_water_mm: f32,
}

impl WeatherRegime {
    /// Advances the two-state chain by one simulated day. Called once per
    /// day by the orchestrator, at the same hour-0 boundary as the daily
    /// precipitation reset; a no-op when the regime is disabled, so an
    /// OFF build never even draws.
    ///
    /// `day` is the absolute day index since the world started (not the
    /// day of year): the chain must not repeat itself every 365 days.
    /// `seed` is the world seed.
    pub fn advance_day(&mut self, seed: u32, day: u64, params: &AtmosphereParams) {
        if !regime_active(params) {
            return;
        }
        let leave_probability = if self.wet {
            transition_probability(params.regime_wet_mean_days)
        } else {
            transition_probability(params.regime_dry_mean_days)
        };
        let draw = hashing::hash01(&[u64::from(seed), day, REGIME_SALT]);
        if draw < leave_probability {
            self.wet = !self.wet;
        }
    }
}

/// Is the regime switched on? One predicate so "0 = off" is written once
/// (`regime_enabled` is an f32 for the same reason `updraft_ref_ms` is).
#[must_use]
pub(crate) fn regime_active(params: &AtmosphereParams) -> bool {
    params.regime_enabled != 0.0
}

/// Probability, per daily draw, of leaving an episode whose mean length
/// is `mean_days`: a geometric duration with `E[length] = mean_days`.
///
/// The chain is sampled once a day, so it cannot represent an episode
/// shorter than one day: at or below that the probability is 1 and every
/// day redraws. This is the sampling theorem, not a defensive clamp — a
/// mean of half a day is simply outside what a daily chain can express.
#[must_use]
fn transition_probability(mean_days: f32) -> f32 {
    if mean_days > 1.0 {
        1.0 / mean_days
    } else {
        1.0
    }
}

/// Fraction of the gap closed by one hourly step of a first-order
/// relaxation with time constant `tau_hours`: `1 − exp(−Δt/τ)` with
/// Δt = 1 h, the exact discretisation of `dx/dt = −x/τ` over a tick.
///
/// τ ≤ 0 means "faster than the tick can resolve", i.e. the whole gap
/// closes within the hour: the limit of the same expression, not a
/// special case bolted on.
#[must_use]
pub(crate) fn relaxation_gain(tau_hours: f32) -> f32 {
    if tau_hours > 0.0 {
        1.0 - (-1.0 / tau_hours).exp()
    } else {
        1.0
    }
}

/// Vapour (mm) one column hands to the sky in one hour of the dry phase:
/// the surplus above `rh_target × sat`, relaxed at first order with
/// `gain` = [`relaxation_gain`]`(regime_export_hours)`. Zero for a column
/// already at or below its target.
///
/// Split out of [`export_surplus_to_the_sky`] (coarse upper layer, step 2)
/// so the fine grid and the ~1 km coarse torus (`atmosphere::coarse`)
/// export by the same rule instead of two hand-kept twins. Only the loop
/// around it differs: the fine one walks `CellProperties`, the coarse one
/// an `f32` slice whose columns each stand for `N_c` fine cells, so what
/// they add to the sky is weighted by `N_c` (the sky reservoir is in mm
/// summed over FINE cells, the unit every water budget uses).
///
/// Cannot push `humidity_upper` negative: the return is
/// `excess × gain` with `gain ∈ (0, 1]` and `excess ≤ humidity_upper` for
/// a non-negative target, so subtracting it lands at worst exactly on the
/// target. No `max(0.0)`, and none wanted: a floor here would hide the day
/// the algebra stops holding.
#[must_use]
pub(crate) fn dry_export_mm(humidity_upper: f32, sat_upper: f32, rh_target: f32, gain: f32) -> f32 {
    let excess = humidity_upper - rh_target * sat_upper;
    if excess > 0.0 { excess * gain } else { 0.0 }
}

/// Wet phase, one hour: `(what the sky gives back, the share each fine
/// cell receives)`. `column_total` is the number of FINE cells the return
/// is spread over — `n` on the fine grid, the fine-cell total of the
/// coarse torus on the coarse path, which is the same number. So a coarse
/// column receives the same mm as each of its fine cells would, and
/// `Σ_c N_c × share = returned` either way (the return is uniform in mm,
/// design note §4).
///
/// The stock cannot go negative: what leaves is `sky × gain` with
/// `gain ≤ 1`. It decays toward zero and is not floored — a finite stock
/// draining exponentially is a physical statement, unlike a guard rail.
#[must_use]
pub(crate) fn wet_return_mm(sky_water_mm: f32, gain: f32, column_total: f32) -> (f32, f32) {
    let returned = sky_water_mm * gain;
    (returned, returned / column_total)
}

/// One hourly step of the imposed regime, in place on `next`.
///
/// Sits in `step_atmosphere_into` after `step_evaporation` (the pass that
/// folds the `current → next` copy, so `next` holds this tick's state)
/// and **before** the orographic pump and the vapour ↔ droplet
/// transition: the condensation that follows must see the subsaturated
/// upper layer this pass just produced, otherwise the surplus condenses
/// before it can leave and the dry phase never happens.
///
/// `sat_upper` is `AtmoScratch::sat_upper_offset`, the memoized
/// `saturation_upper(T_upper)` per cell that the condensation will use a
/// few passes later — read here rather than recomputed, so "dry" means
/// exactly the RH the rest of the tick means (anti-pattern #2).
///
/// `moved` and `partials` are reused scratch buffers, content undefined
/// between two ticks: the per-cell export, then its deterministic block
/// reduction (`par::reduce_blocks`), because summing what left the map
/// must not depend on the thread count.
pub(crate) fn step_weather_regime(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    sat_upper: &[f32],
    state: &mut WeatherRegime,
    moved: &mut Vec<f32>,
    partials: &mut Vec<f32>,
) {
    if !regime_active(params) {
        return;
    }
    let n = next.len();
    if n == 0 {
        return;
    }
    if state.wet {
        return_sky_to_the_map(next, params, state);
    } else {
        export_surplus_to_the_sky(next, params, sat_upper, state, moved, partials);
    }
}

/// Dry phase: every column relaxes toward `rh_target × sat`, the surplus
/// leaves the box. The per-column rule is [`dry_export_mm`], shared with
/// the coarse torus (`atmosphere::coarse`); this function is the fine-grid
/// loop around it plus the deterministic reduction of what left.
fn export_surplus_to_the_sky(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    sat_upper: &[f32],
    state: &mut WeatherRegime,
    moved: &mut Vec<f32>,
    partials: &mut Vec<f32>,
) {
    let n = next.len();
    let gain = relaxation_gain(params.regime_export_hours);
    let rh_target = params.regime_dry_rh_target;
    moved.clear();
    moved.resize(n, 0.0);
    for_each_chunk_mut2(next.cells_slice_mut(), moved, |start, cells, out| {
        for (local, (cell, exported)) in cells.iter_mut().zip(out.iter_mut()).enumerate() {
            let sat = sat_upper.get(start + local).copied().unwrap_or(0.0);
            let leaving = dry_export_mm(cell.humidity_upper, sat, rh_target, gain);
            cell.humidity_upper -= leaving;
            *exported = leaving;
        }
    });
    let exported: &Vec<f32> = moved;
    reduce_blocks(n, partials, |range| exported[range].iter().sum::<f32>());
    let total = partials.iter().fold(0.0_f32, |acc, &p| acc + p);
    state.sky_water_mm += total;
}

/// Wet phase: the sky gives back a first-order share of what it holds,
/// spread uniformly over the cells (an air mass arriving over the whole
/// box, not a local source). The rule is [`wet_return_mm`], shared with
/// the coarse torus (`atmosphere::coarse`); this is the fine-grid loop.
fn return_sky_to_the_map(next: &mut HexGrid, params: &AtmosphereParams, state: &mut WeatherRegime) {
    if state.sky_water_mm <= 0.0 {
        return;
    }
    let gain = relaxation_gain(params.regime_return_hours);
    let (returned, share) = wet_return_mm(state.sky_water_mm, gain, cell_count_f32(next.len()));
    for_each_chunk_mut(next.cells_slice_mut(), |_, chunk| {
        for cell in chunk {
            cell.humidity_upper += share;
        }
    });
    state.sky_water_mm -= returned;
}

/// `n` as an f32, exact below 2^24 cells, assembled from its two 16-bit
/// halves through lossless `From` conversions (same helper and same
/// reason as `condensation::exact_cell_count`: no `as` cast, no
/// precision-loss lint to silence).
pub(crate) fn cell_count_f32(n: usize) -> f32 {
    let n = u32::try_from(n).expect("cell count fits in u32");
    let high = u16::try_from(n >> 16).expect("high half of a u32 fits in u16");
    let low = u16::try_from(n & 0xFFFF).expect("low half of a u32 fits in u16");
    f32::from(high) * 65_536.0 + f32::from(low)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::saturation_upper;
    use crate::coord::HexCoord;

    fn on_params() -> AtmosphereParams {
        AtmosphereParams {
            regime_enabled: 1.0,
            ..AtmosphereParams::default()
        }
    }

    /// A grid whose upper layer is loaded well past saturation, with a
    /// matching `sat_upper` table. Radius 2: the pass is strictly
    /// per-cell (no transport), but a handful of cells makes the uniform
    /// share of the wet phase observable.
    fn loaded_grid(radius: i32, humidity_upper: f32) -> (HexGrid, Vec<f32>) {
        let mut grid = HexGrid::from_radius(radius);
        for cell in grid.cells_slice_mut() {
            cell.humidity_upper = humidity_upper;
            cell.temperature = 10.0;
        }
        let params = AtmosphereParams::default();
        let sat = vec![saturation_upper(0.0, &params); grid.len()];
        (grid, sat)
    }

    fn total_upper(grid: &HexGrid) -> f32 {
        grid.cells_slice()
            .iter()
            .fold(0.0_f32, |acc, c| acc + c.humidity_upper)
    }

    /// Disabled ⇒ strictly a no-op: not one bit of `humidity_upper`
    /// moves and the sky stays at zero. This is what makes the shipped
    /// build bit-identical to one compiled before the regime existed.
    #[test]
    fn disabled_regime_is_a_no_op() {
        let (mut grid, sat) = loaded_grid(2, 30.0);
        let before: Vec<u32> = grid
            .cells_slice()
            .iter()
            .map(|c| c.humidity_upper.to_bits())
            .collect();
        let mut state = WeatherRegime::default();
        // #63/#146: this test proves the DISABLED path is a no-op, so it
        // must pin `regime_enabled` off explicitly now that the shipped
        // default flipped to on (2026-09-06) — `AtmosphereParams::default()`
        // alone no longer gives it the disabled config it is testing.
        let params = AtmosphereParams {
            regime_enabled: 0.0,
            ..AtmosphereParams::default()
        };
        for _ in 0..48 {
            state.advance_day(42, 0, &params);
            step_weather_regime(
                &mut grid,
                &params,
                &sat,
                &mut state,
                &mut Vec::new(),
                &mut Vec::new(),
            );
        }
        let after: Vec<u32> = grid
            .cells_slice()
            .iter()
            .map(|c| c.humidity_upper.to_bits())
            .collect();
        assert_eq!(before, after, "the disabled pass must not touch a bit");
        assert_eq!(state.sky_water_mm.to_bits(), 0.0_f32.to_bits());
        assert!(!state.wet, "a disabled regime must not even draw");
    }

    /// The dry phase drains toward the target and stops there: the export
    /// never pushes `humidity_upper` below `rh_target × sat`, and never
    /// below zero, however long it runs.
    #[test]
    fn dry_export_relaxes_to_the_target_and_never_goes_negative() {
        let params = on_params();
        let (mut grid, sat) = loaded_grid(2, 30.0);
        let target = params.regime_dry_rh_target * sat[0];
        let mut state = WeatherRegime::default();
        let (mut moved, mut partials) = (Vec::new(), Vec::new());
        for _ in 0..500 {
            step_weather_regime(
                &mut grid,
                &params,
                &sat,
                &mut state,
                &mut moved,
                &mut partials,
            );
            for c in grid.cells_slice() {
                assert!(
                    c.humidity_upper >= -f32::EPSILON,
                    "humidity_upper went negative: {}",
                    c.humidity_upper
                );
                assert!(
                    c.humidity_upper >= target - 1e-3,
                    "export overshot the target: {} < {target}",
                    c.humidity_upper
                );
            }
        }
        let residual = grid.cells_slice()[0].humidity_upper;
        assert!(
            (residual - target).abs() < 1e-3,
            "after 500 h the column must sit at the target: {residual} vs {target}"
        );
        assert!(state.sky_water_mm > 0.0, "the sky must have received it");
    }

    /// One hourly step moves exactly the first-order share of the
    /// surplus, `1 − exp(−1/τ)`: the discretisation is pinned, not just
    /// "some water moved".
    #[test]
    fn one_export_step_moves_the_first_order_share() {
        let params = on_params();
        let (mut grid, sat) = loaded_grid(0, 30.0);
        let target = params.regime_dry_rh_target * sat[0];
        let expected = (30.0 - target) * (1.0 - (-1.0_f32 / params.regime_export_hours).exp());
        let mut state = WeatherRegime::default();
        step_weather_regime(
            &mut grid,
            &params,
            &sat,
            &mut state,
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert!(
            (state.sky_water_mm - expected).abs() < 1e-4,
            "exported {} mm, first-order share is {expected} mm",
            state.sky_water_mm
        );
    }

    /// The wet phase gives the sky back, uniformly, with the same
    /// first-order law and its own τ, and the stock never goes negative.
    #[test]
    fn wet_return_decays_the_sky_uniformly() {
        let params = on_params();
        let (mut grid, sat) = loaded_grid(2, 1.0);
        let n = cell_count_f32(grid.len());
        let mut state = WeatherRegime {
            wet: true,
            sky_water_mm: 100.0,
        };
        let expected_first = 100.0 * (1.0 - (-1.0_f32 / params.regime_return_hours).exp());
        step_weather_regime(
            &mut grid,
            &params,
            &sat,
            &mut state,
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert!(
            (state.sky_water_mm - (100.0 - expected_first)).abs() < 1e-3,
            "sky at {} after one hour, expected {}",
            state.sky_water_mm,
            100.0 - expected_first
        );
        let share = expected_first / n;
        for c in grid.cells_slice() {
            assert!(
                (c.humidity_upper - (1.0 + share)).abs() < 1e-4,
                "each cell must get the same share: {} vs {}",
                c.humidity_upper,
                1.0 + share
            );
        }
        for _ in 0..2000 {
            step_weather_regime(
                &mut grid,
                &params,
                &sat,
                &mut state,
                &mut Vec::new(),
                &mut Vec::new(),
            );
            assert!(state.sky_water_mm >= 0.0, "sky went negative");
        }
        assert!(
            state.sky_water_mm < 1e-6,
            "the sky must drain to (numerically) nothing: {}",
            state.sky_water_mm
        );
    }

    /// The terrarium invariant of this pass alone: `Σ humidity_upper +
    /// sky` is conserved through a long alternation of both phases, to
    /// f32 rounding. If this drifts, every water budget in the project
    /// is lying.
    #[test]
    fn export_and_return_conserve_upper_plus_sky() {
        let params = on_params();
        let (mut grid, sat) = loaded_grid(3, 22.0);
        let mut state = WeatherRegime::default();
        let initial = total_upper(&grid);
        let (mut moved, mut partials) = (Vec::new(), Vec::new());
        for hour in 0..(30 * 24) {
            if hour % 24 == 0 {
                state.advance_day(42, hour / 24, &params);
            }
            step_weather_regime(
                &mut grid,
                &params,
                &sat,
                &mut state,
                &mut moved,
                &mut partials,
            );
        }
        let total = total_upper(&grid) + state.sky_water_mm;
        let drift = (total - initial).abs() / initial;
        assert!(
            drift < 1e-4,
            "upper + sky drifted by {drift:.2e} ({initial} -> {total})"
        );
    }

    /// Same invariant on randomised stocks and saturations, including
    /// columns already below their target (nothing to export) and an
    /// empty sky (nothing to return).
    #[test]
    fn conservation_holds_on_randomised_stocks() {
        let params = on_params();
        let mut grid = HexGrid::from_radius(4);
        let mut sat = vec![0.0_f32; grid.len()];
        for (draw, (cell, s)) in grid
            .cells_slice_mut()
            .iter_mut()
            .zip(sat.iter_mut())
            .enumerate()
        {
            let draw = u64::try_from(draw).expect("cell index fits in u64");
            cell.humidity_upper = hashing::hash01(&[draw, 1]) * 40.0;
            *s = hashing::hash01(&[draw, 2]) * 30.0;
        }
        let initial = total_upper(&grid);
        let mut state = WeatherRegime::default();
        let (mut moved, mut partials) = (Vec::new(), Vec::new());
        for hour in 0..(60 * 24) {
            if hour % 24 == 0 {
                state.advance_day(7, hour / 24, &params);
            }
            step_weather_regime(
                &mut grid,
                &params,
                &sat,
                &mut state,
                &mut moved,
                &mut partials,
            );
            for c in grid.cells_slice() {
                assert!(c.humidity_upper >= 0.0, "humidity_upper went negative");
            }
        }
        let total = total_upper(&grid) + state.sky_water_mm;
        let drift = (total - initial).abs() / initial;
        assert!(drift < 1e-4, "upper + sky drifted by {drift:.2e}");
    }

    /// The chain is a pure function of (seed, day): two runs of the same
    /// seed replay the same weather, and two seeds do not.
    #[test]
    fn the_episode_sequence_is_reproducible_per_seed() {
        let params = on_params();
        let sequence = |seed: u32| {
            let mut state = WeatherRegime::default();
            (0..2000)
                .map(|day| {
                    state.advance_day(seed, day, &params);
                    state.wet
                })
                .collect::<Vec<bool>>()
        };
        assert_eq!(sequence(42), sequence(42), "same seed, same weather");
        assert_ne!(sequence(42), sequence(7), "two seeds, two weathers");
    }

    /// Mean episode lengths land on the parameters within ±20 % over
    /// 10 000 days, and the wet fraction on the stationary distribution
    /// of the chain. This is what ties `regime_dry_mean_days` to
    /// "≈112 wet days a year" rather than to a hope.
    #[test]
    fn mean_episode_durations_match_the_parameters() {
        let params = on_params();
        let mut state = WeatherRegime::default();
        let (mut dry_days, mut wet_days) = (0_u32, 0_u32);
        let (mut dry_episodes, mut wet_episodes) = (0_u32, 0_u32);
        let mut previous = state.wet;
        for day in 0..10_000_u64 {
            state.advance_day(42, day, &params);
            if state.wet != previous {
                if state.wet {
                    wet_episodes += 1;
                } else {
                    dry_episodes += 1;
                }
                previous = state.wet;
            }
            if state.wet {
                wet_days += 1;
            } else {
                dry_days += 1;
            }
        }
        let mean = |days: u32, episodes: u32| f64::from(days) / f64::from(episodes.max(1));
        let dry_mean = mean(dry_days, dry_episodes);
        let wet_mean = mean(wet_days, wet_episodes);
        let within = |got: f64, want: f64| (got - want).abs() / want < 0.20;
        assert!(
            within(dry_mean, f64::from(params.regime_dry_mean_days)),
            "mean dry episode {dry_mean:.2} d vs {} d",
            params.regime_dry_mean_days
        );
        assert!(
            within(wet_mean, f64::from(params.regime_wet_mean_days)),
            "mean wet episode {wet_mean:.2} d vs {} d",
            params.regime_wet_mean_days
        );
        // Stationary wet fraction: (1/dry) / (1/dry + 1/wet).
        let p_dry_to_wet = 1.0 / f64::from(params.regime_dry_mean_days);
        let p_wet_to_dry = 1.0 / f64::from(params.regime_wet_mean_days);
        let expected = p_dry_to_wet / (p_dry_to_wet + p_wet_to_dry);
        let measured = f64::from(wet_days) / 10_000.0;
        assert!(
            within(measured, expected),
            "wet fraction {measured:.3} vs stationary {expected:.3}"
        );
    }

    /// An episode mean at or below the daily sampling step redraws every
    /// day (probability 1), the only thing a daily chain can mean by it.
    #[test]
    fn a_sub_daily_mean_redraws_every_day() {
        assert!((transition_probability(0.25) - 1.0).abs() < 1e-9);
        assert!((transition_probability(1.0) - 1.0).abs() < 1e-9);
        assert!((transition_probability(4.0) - 0.25).abs() < 1e-6);
        let params = AtmosphereParams {
            regime_enabled: 1.0,
            regime_dry_mean_days: 0.5,
            regime_wet_mean_days: 0.5,
            ..AtmosphereParams::default()
        };
        let mut state = WeatherRegime::default();
        for day in 0..10_u64 {
            let before = state.wet;
            state.advance_day(42, day, &params);
            assert_ne!(before, state.wet, "p = 1 must flip every day");
        }
    }

    /// A cell that is already drier than the target keeps every drop:
    /// the pass only ever removes a surplus, it is not a drying rate
    /// applied to everything.
    #[test]
    fn a_column_below_the_target_is_left_alone() {
        let params = on_params();
        let mut grid = HexGrid::from_radius(0);
        let sat = vec![20.0_f32; grid.len()];
        let below = params.regime_dry_rh_target * sat[0] * 0.5;
        grid.get_mut(HexCoord::new(0, 0)).unwrap().humidity_upper = below;
        let mut state = WeatherRegime::default();
        step_weather_regime(
            &mut grid,
            &params,
            &sat,
            &mut state,
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert_eq!(
            grid.get(HexCoord::new(0, 0))
                .unwrap()
                .humidity_upper
                .to_bits(),
            below.to_bits()
        );
        assert_eq!(state.sky_water_mm.to_bits(), 0.0_f32.to_bits());
    }
}
