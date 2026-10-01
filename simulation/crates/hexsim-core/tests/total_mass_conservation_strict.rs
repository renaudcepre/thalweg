//! Strict conservation test: fundamental invariant of the terrarium.
//!
//! After closing off the boundaries (removal of `inject_boundary_humidity`,
//! `drain_edges`, external wind forcing), the sum of the 4 water stocks
//! (`water_level` + `humidity_total` + groundwater + `snow_level`) must stay
//! strictly constant over any simulation duration.
//!
//! This is THE test that validates the switch to a closed terrarium really
//! cut off all the leaks. A non-zero drift = a hidden sink or source.

mod common;

use common::{build_prod_sim, total_water_budget};

/// See the test's doc: the historical absolute bound (1e-1 on a ~462-unit
/// budget) expressed as a fraction of the initial budget.
const STRICT_DRIFT_REL: f32 = 2.2e-4;

/// Same unit correction for the yearly check: its historical absolute
/// bound (5e-2, v0.3.0 PR4 #38, on the same ~462-unit budget) as a
/// fraction of the initial budget. Since #151 step 1 (2026-09-04) the
/// r3 budget is ~1 749 and the f32 rounding bias grows with it: on main
/// the year-5 drift already sat at 77 % of the absolute bound
/// (2.2e-5 relative), and the block reductions of 2026-09-05, which move
/// no water but shift the trajectory, crossed it (3.3e-5 relative). A
/// real leak of 1e-3 mm/cell/day would make 7.7e-3 relative in one year
/// at r3, seventy times this bound.
const YEARLY_DRIFT_REL: f32 = 1.1e-4;

/// 10 years (3650 ticks): the drift of `water_budget.total` must stay
/// under `STRICT_DRIFT_REL` of the initial budget. This is pure
/// floating-point tolerance (f32 accumulation over ~90 000 hourly ticks
/// on 4 stocks: `water_level` + `humidity_total` + groundwater +
/// `snow_level`), so it scales with the budget, which is why it is
/// relative. History of the bound, in absolute units of the time:
/// Phase 6 (#29) 1.5e-3 -> 1e-2 (Tetens f32 noise), then 1e-1 (KK2000
/// micro-transfers); physical drift stayed zero (0.048 over 10 years).
///
/// 2026-09-05: #151 step 1 raised the r3 budget from ~462 to ~1749
/// units, and the r250 chunk C1 (two-phase hydro and groundwater gather,
/// same per-source transfers summed in a different order) removed a
/// coincidental cancellation: the old serial hydro scatter carried a
/// -0.040 rounding bias over 1400 days that masked a +0.039 bias of
/// `step_atmosphere_into`/`step_snow` (measured with a per-phase probe,
/// both biases present on the pre-C1 code). Net drift on this seed:
/// 0.1415, i.e. 8e-5 of the budget, against an absolute bound that
/// meant 2.2e-4 of the budget when it was set. The bound keeps that
/// relative value; a real leak (1e-3 mm/cell/day) would show as ~2e-3
/// of the budget over 10 years, an order of magnitude above it.
#[test]
fn water_budget_is_strictly_conserved_over_10_years() {
    let mut sim = build_prod_sim(42, 3);
    let initial = total_water_budget(&sim);

    // Sanity: the sim starts with something. Otherwise the test measures nothing.
    assert!(
        initial > 1.0,
        "initial stock must be non-trivial: {initial}"
    );

    for _ in 0..3650 {
        sim.step();
    }

    let final_total = total_water_budget(&sim);
    let drift = (final_total - initial).abs();
    let bound = STRICT_DRIFT_REL * initial;
    assert!(
        drift < bound,
        "strict conservation broken over 10 years: {initial:.6} -> {final_total:.6} \
         (drift {drift:.6}, bound {bound:.6})"
    );
}

/// Yearly check: the drift must stay under the threshold not only
/// at the end of the simulation but also at every intermediate tick.
/// Detects a leak that would be offset by a symmetric source at the end
/// of the cycle (unlikely but possible, e.g. evaporation offset by
/// precipitation of an amount that should have escaped).
#[test]
fn water_budget_stays_bounded_every_year() {
    let mut sim = build_prod_sim(42, 3);
    let initial = total_water_budget(&sim);

    for year in 1..=5 {
        for _ in 0..365 {
            sim.step();
        }
        let current = total_water_budget(&sim);
        let drift = (current - initial).abs();
        // v0.3.0 PR4 (#38): tolerance widened 1e-2 -> 5e-2. The Tier 1
        // regime with continuous precipitation generates far more
        // f32 operations per year. See the note on the 10-year test.
        // Relative to the initial budget since 2026-09-05, see
        // `YEARLY_DRIFT_REL`.
        let bound = YEARLY_DRIFT_REL * initial;
        assert!(
            drift < bound,
            "cumulative drift at year {year}: {initial:.6} -> {current:.6} \
             (drift {drift:.6}, bound {bound:.6})"
        );
    }
}
