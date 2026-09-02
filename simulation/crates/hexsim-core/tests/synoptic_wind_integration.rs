//! Phase 1 integration (of the synoptic-dynamics design's f-plane
//! shallow-water core): synoptic dynamics wired into prod pipeline (behind
//! `synoptic.enabled`).
//!
//! Validates Phase 1 acceptance criteria at full engine level:
//!   1. **Emergence**: pressure systems (L/H) appear in geopotential field,
//!      wind is no longer decorative noise.
//!   2. **Real effect**: enabling synoptic changes wind field consumed by
//!      atmosphere (noise+bias base is properly replaced).
//!   3. **Conservation**: synoptic neither injects nor destroys water (plan 6,
//!      `Q` is pressure source, not humidity). Water budget stays conserved
//!      even with synoptic wind active.
//!   4. **Stability + determinism**: no explosion over 60 days, one seed →
//!      one field, bit-reproducible.
//!
//! Not coupled to precip (Phase 3): here only verify core runs healthily in prod.

mod common;

use common::{build_prod_sim, total_water_budget};

const RADIUS: i32 = 12;
const SEED: u32 = 42;
const DAYS: usize = 60;

fn range(values: &[f32]) -> f32 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in values {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo
}

#[test]
fn synoptic_enabled_evolves_stably_and_conserves_water() {
    let mut sim = build_prod_sim(SEED, RADIUS);
    sim.update_param("synoptic.enabled", 1.0);
    assert!(
        sim.synoptic_enabled(),
        "the flag must be active after update_param"
    );

    let water0 = total_water_budget(&sim);
    let h_initial_range = range(sim.synoptic_state().height());

    for _ in 0..DAYS {
        sim.step();
    }

    let state = sim.synoptic_state();
    let h = state.height();

    // Stability: nothing diverges.
    assert!(
        h.iter().all(|x| x.is_finite()),
        "geopotential height not finite"
    );
    let umax = state.max_speed();
    assert!(
        umax.is_finite() && umax < 50.0,
        "synoptic wind unstable: max|u| = {umax}"
    );

    // Emergence: pressure field structured (L/H) beyond seed initial perturbation.
    let h_range = range(h);
    assert!(
        h_range > h_initial_range,
        "no emergent pressure systems: range h {h_range:.4} <= initial {h_initial_range:.4}"
    );

    // Conservation: synoptic wind didn't touch water budget.
    let drift = (total_water_budget(&sim) - water0).abs();
    assert!(
        drift < 5e-2,
        "synoptic must not create or destroy water: drift {drift:.4}"
    );
}

#[test]
fn synoptic_changes_the_consumed_wind_field() {
    // Two identical worlds, one with synoptic, one without: effective wind
    // field must differ (synoptic base replaces noise+bias). Since Phase 4
    // synoptic dynamics active by default, "off" control must explicitly
    // disable it to isolate field effect.
    let mut off = build_prod_sim(SEED, RADIUS);
    off.update_param("synoptic.enabled", 0.0);
    let mut on = build_prod_sim(SEED, RADIUS);
    on.update_param("synoptic.enabled", 1.0);

    for _ in 0..20 {
        off.step();
        on.step();
    }

    let differing = off
        .wind_field()
        .iter()
        .zip(on.wind_field())
        .filter(|(a, b)| (a.x - b.x).abs() > 1e-4 || (a.y - b.y).abs() > 1e-4)
        .count();
    assert!(
        differing > off.wind_field().len() / 4,
        "synoptic wind should modify substantial part of field ({differing} cells)"
    );
}

#[test]
fn synoptic_field_is_deterministic() {
    let run = || {
        let mut sim = build_prod_sim(SEED, RADIUS);
        sim.update_param("synoptic.enabled", 1.0);
        for _ in 0..30 {
            sim.step();
        }
        sim.synoptic_state().height().to_vec()
    };
    assert_eq!(
        run(),
        run(),
        "same seed - same synoptic field, bit-identical"
    );
}
