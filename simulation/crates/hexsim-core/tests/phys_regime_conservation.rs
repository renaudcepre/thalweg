//! Micro-test (#63): with the imposed weather regime ON, the terrarium
//! invariant still holds, provided the sky reservoir is counted as part of
//! the terrarium.
//!
//! The regime is the first phenomenon in the engine that moves water *out
//! of the grid* on purpose. That makes "the total is conserved" a
//! statement about `cells + sky`, and it makes every budget in the project
//! that forgets `sky` wrong the day the lever is turned on. Two tests
//! here, and the second is the one that matters: the first proves the
//! invariant holds, the second proves the first would notice if `sky` were
//! dropped.
//!
//! Bound: `STRICT_DRIFT_REL` = 2.2e-4 relative over 60 days, the same
//! relative tolerance `total_mass_conservation_strict` allows itself over
//! ten years. Deliberately not looser: the regime moves water between two
//! f32 accumulators, and if that costs more rounding than a decade of the
//! whole engine, it is the pass that is wrong, not the bound.

mod common;

use common::build_prod_sim;
use hexsim_core::simulation::Simulation;

/// Same relative bound as `total_mass_conservation_strict`, over 60 days
/// instead of ten years.
const STRICT_DRIFT_REL: f32 = 2.2e-4;
const DAYS: u32 = 60;
const SEED: u32 = 42;
const RADIUS: i32 = 6;

/// The terrarium's whole water stock: `Simulation::water_budget_total`,
/// read and never re-assembled here (anti-pattern 2). It counts
/// the surface stocks per cell, the moist upper layer through its coarse
/// REFERENCE stock (coarse upper layer, step 2: the fine `humidity_upper`/
/// `cloud_water` are views rewritten by a non-conservative interpolation
/// every hour, so summing them is not the mass) and the sky reservoir of
/// the imposed weather regime (#63), which left the cells but not the
/// world.
fn total_with_sky(sim: &Simulation) -> f32 {
    sim.water_budget_total()
}

/// The same budget without the sky, i.e. what every pre-#63 helper
/// computed: the amount the map itself is holding.
fn total_without_sky(sim: &Simulation) -> f32 {
    sim.water_budget_total() - sim.sky_water_total()
}

fn regime_sim() -> Simulation {
    let mut sim = build_prod_sim(SEED, RADIUS);
    assert!(
        sim.update_param("atmosphere.regime_enabled", 1.0),
        "unknown key: atmosphere.regime_enabled"
    );
    sim
}

#[test]
fn terrarium_plus_sky_is_conserved_with_the_regime_on() {
    let mut sim = regime_sim();
    let initial = total_with_sky(&sim);
    assert!(
        initial > 1.0,
        "initial stock must be non-trivial: {initial}"
    );

    let mut worst = 0.0_f32;
    for _ in 0..DAYS {
        sim.step();
        let drift = (total_with_sky(&sim) - initial).abs() / initial;
        worst = worst.max(drift);
    }

    assert!(
        worst < STRICT_DRIFT_REL,
        "terrarium + sky drifted by {worst:.3e} over {DAYS} days, bound {STRICT_DRIFT_REL:.1e}"
    );
}

/// The counter-proof, and the reason the test above is not vacuous: over
/// the same window the *grid alone* must visibly lose water, because the
/// regime really did take some out of it. If this ever goes green the
/// regime moved nothing and the conservation test above is measuring an
/// inert pass.
#[test]
fn the_grid_alone_is_not_conserved_because_the_sky_holds_the_difference() {
    let mut sim = regime_sim();
    let initial = total_without_sky(&sim);

    let mut worst_gap = 0.0_f32;
    for _ in 0..DAYS {
        sim.step();
        let gap = (initial - total_without_sky(&sim)).abs() / initial;
        worst_gap = worst_gap.max(gap);
    }

    assert!(
        worst_gap > 10.0 * STRICT_DRIFT_REL,
        "the grid alone never departed from its initial budget by more than \
         {worst_gap:.3e}: the regime exported nothing, so the conservation test \
         next door proves nothing"
    );
    // And the difference is exactly the sky at that instant, to rounding.
    let gap = initial - total_without_sky(&sim);
    let sky = sim.sky_water_total();
    assert!(
        (gap - sky).abs() / initial < STRICT_DRIFT_REL,
        "the grid lost {gap} mm and the sky holds {sky} mm: the difference went nowhere"
    );
}
