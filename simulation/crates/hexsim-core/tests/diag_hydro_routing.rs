//! Diagnostic #105→hydro: does MFD routing concentrate rivers?
//!
//! The `diag_live_rivers` measurement showed that erosion doesn't
//! converge live rivers (it grades the profile and spreads the water
//! out). Live discharge is already ultra-concentrated (Gini 0.995): the
//! "parallel" pattern comes from **routing** MFD
//! (`raw_flow_i ∝ delta_i^flow_concentration`, split across 2-3 edges),
//! not topography. This diag pushes `flow_concentration` from 6
//! (default, weighted MFD) toward ∞ (D8: all flow on the steepest edge)
//! on RAW terrain (erosion off), and looks at the live river regime
//! after 2 years (seed 7, r30):
//!   - `river_cells` (discharge > 0.5): MUST DROP if the braids merge
//!     into a single channel;
//!   - `max_discharge`: MUST RISE (the collector picks up everything);
//!   - `top3`: the three largest discharges, the "big river" signature.
//!
//! Run: `just diag-tool hydro_routing`

mod common;

use common::build_prod_sim;
use hexsim_core::erosion::discharge_gini;
use hexsim_core::simulation::Simulation;

const RADIUS: i32 = 30;
const SEED: u32 = 7;
const RIVER_THRESHOLD: f32 = 0.5;

fn stats(sim: &Simulation) -> (usize, f64, f32, [f32; 3]) {
    let mut d: Vec<f32> = sim.discharge_map().clone();
    let rivers = d.iter().filter(|&&x| x > RIVER_THRESHOLD).count();
    let gini = discharge_gini(&d);
    d.sort_by(|a, b| b.total_cmp(a));
    let top3 = [d[0], d[1], d[2]];
    let max = d[0];
    (rivers, gini, max, top3)
}

fn run_case(concentration: f32) {
    // Raw terrain (worldgen erosion off): isolates the effect of routing.
    let mut sim = build_prod_sim(SEED, RADIUS);
    assert!(sim.update_param("hydro.flow_concentration", concentration));
    for _ in 0..(2 * 365) {
        sim.step();
    }
    let (rivers, gini, max, top3) = stats(&sim);
    eprintln!(
        "  flow_conc {concentration:>5.0} : river_cells {rivers:>4} ; gini {gini:.4} ; \
         max {max:>6.1} ; top3 [{:.1}, {:.1}, {:.1}]",
        top3[0], top3[1], top3[2]
    );
}

#[test]
#[ignore = "diagnostic #105→hydro, MFD→D8 routing vs live rivers (seed 7, r30, 2 years)"]
fn hydro_routing_concentration_sweep() {
    eprintln!("=== MFD→D8 routing / live rivers 2 years / seed {SEED} (r{RADIUS}) ===");
    eprintln!("  Expected if D8 concentrates: river_cells ↓, max/top3 ↑");
    for c in [6.0_f32, 15.0, 40.0, 100.0, 1000.0] {
        run_case(c);
    }
}
