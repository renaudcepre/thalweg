//! Guard rail against oscillations of the symmetric MFD.
//!
//! Scenario: constant water inflow each tick into the basin's center.
//! At steady state, `water_level` variance at the center must stay
//! bounded. If the MFD oscillates (A→B→A→B), the variance explodes.

use hexsim_core::cell::CellProperties;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::hydro::{HydroParams, step_hydro_mfd};

#[test]
fn mfd_does_not_diverge_under_constant_input() {
    // #104 (SI): 100 mm/tick, with 1 mm/tick, 500 ticks only accumulated
    // 0.5 m of water under a 5 m rim: no overflow was possible any more
    // in SI space, the test went green by vacuity (zero flux = zero
    // oscillation). At 100 mm/tick the center exceeds the rim around
    // tick 50 and the MFD works continuously over the remaining 450 ticks.
    const INFLOW_PER_TICK: f32 = 100.0;
    let mut grid = HexGrid::from_radius(2);
    let center = HexCoord::new(0, 0);

    for coord in grid.coords().copied().collect::<Vec<_>>() {
        let d = coord.distance(center);
        if let Some(c) = grid.get_mut(coord) {
            *c = CellProperties {
                elevation: match d {
                    0 => 0.0,
                    1 => 5.0,
                    _ => 10.0,
                },
                water_capacity: 1.0,
                ..Default::default()
            };
        }
    }

    let params = HydroParams {
        flow_rate: 0.15,
        ..HydroParams::default()
    };

    let mut current = grid;
    let mut history: Vec<f32> = Vec::with_capacity(500);

    for _ in 0..500 {
        current.get_mut(center).unwrap().water_level += INFLOW_PER_TICK;
        for _ in 0..8 {
            let mut next = current.clone();
            step_hydro_mfd(&current, &mut next, &params);
            current = next;
        }
        history.push(current.get(center).unwrap().water_level);
    }

    // Anti-vacuity sanity check: the center must have exported a
    // significant share of the injection, or no oscillation is exercised.
    let injected = INFLOW_PER_TICK * 500.0;
    let center_final = current.get(center).unwrap().water_level;
    assert!(
        center_final < 0.9 * injected,
        "no significant outflow from the center ({center_final:.0} mm out of \
         {injected:.0} injected): the test doesn't exercise the MFD"
    );

    // Analyze the last 50 ticks (assumed steady state)
    let tail: &[f32] = &history[history.len() - 50..];
    let mean: f32 =
        tail.iter().sum::<f32>() / f32::from(u16::try_from(tail.len()).expect("fits u16"));
    let var: f32 = tail.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
        / f32::from(u16::try_from(tail.len()).expect("fits u16"));
    let cv = var.sqrt() / mean.abs();

    assert!(mean.is_finite(), "water_level center diverges: {mean}");
    assert!(
        cv < 0.1,
        "relative variance too high: cv={cv:.3} (mean={mean:.3}, std={:.3})",
        var.sqrt()
    );
}
