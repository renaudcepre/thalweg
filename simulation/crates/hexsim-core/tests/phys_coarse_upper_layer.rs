//! The sentinels of the **coarse** moist upper layer
//! (`atmosphere::coarse`, steps 2 to 2c of its
//! migration).
//!
//! ## Radius, and whether it is the identity path
//!
//! Both worlds are **radius 6**: `moist_coarse_radius(6) = 2`, so 127 fine
//! cells map onto 19 coarse cells, ≈6.7 fine cells each. That is a real
//! coarse torus, **not** the identity path (`Rc = R`, which the formula
//! degenerates to at `r ≤ 2`) — the rule `atmosphere::coarse` asks every
//! upper-layer micro-test to state.
//!
//! ## Why they force the mode
//!
//! The coarse mode has been the compiled-in default since 2026-09-30
//! (`MOIST_COARSE_DEFAULT`, see its doc), but `Ablation::effective()` is a
//! process-wide `OnceLock` that a test cannot set, and a sentinel that
//! inherited the default would measure whichever `HEXSIM_MOIST_COARSE`
//! happened to be set. So every test here calls
//! `Simulation::set_moist_coarse_mode(…)` right after construction, the
//! seam that exists for exactly this: the mode under test is stated, not
//! inherited. [`MoistCoarseMode::CoarsePrecip`] (step
//! 2c, variant M) is the sentinels' subject, for the precipitation unit
//! and for the fine cloud surviving in its own column.
//!
//! A third mode, `CoarseStock` (steps 2/2b, the coarse state as a
//! prognostic reference stock with the fine fields as broadcast views),
//! was retired 2026-09-07 — see `atmosphere::coarse`'s module doc for the
//! measurement that sent it back to the fine grid. Its dedicated
//! sentinel, a statement about the pooled condensation that would have
//! been a tautology on `CoarsePrecip`, was retired with it.

use hexsim_core::atmosphere::{AtmosphereParams, MoistCoarseMode};
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::terrain::{TerrainParams, generate_terrain};
use hexsim_core::wind::{WindParams, WindVec};

mod common;

/// 127 fine cells, 19 coarse cells (`Rc = 2`), ≈6.7 fine per coarse.
const RADIUS: i32 = 6;

/// A world on the given coarse mode, flat and dry unless the caller
/// sculpts it.
fn coarse_world(atmosphere: AtmosphereParams, grid: HexGrid, mode: MoistCoarseMode) -> Simulation {
    let mut sim = Simulation::new(
        grid,
        HydroParams::default(),
        atmosphere,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    );
    sim.set_moist_coarse_mode(mode);
    assert_eq!(sim.moist_coarse_mode(), mode, "the seam must have taken");
    sim
}

/// Sentinel (i): a cloud seeded on ONE fine cell becomes a cloud of its
/// **coarse cell**, and rains on every fine cell of that cell — each one
/// with **its own** rain/snow phase.
///
/// This is the whole point of the step in one test. Two halves of the map
/// straddle 0 °C, so the same coarse sheet has to fall as snow on the cold
/// fine cells and as rain on the warm ones (design note §4 option (d), not
/// optional: a 1 km cell spans 300 to 600 m of relief at this terrain's
/// slopes). And nothing may fall outside the coarse cell: the coarse cell
/// IS the footprint on this path.
///
/// **The drift spread is pinned off here** (`precip_neighbor_share = 0`),
/// the way this fixture already pins advection, fog and the pump off. It
/// is not inert at r6 any more: step 2b added the rule that the coarse
/// footprint is never narrower than the fine drift disc
/// (`atmosphere::coarse::CoarseFootprint`), and at this radius a coarse
/// cell holds ≈6.7 fine cells against a 37-cell disc, so the rule fires
/// and feathers the sheet's edge. That rule has its own sentinel
/// (`a_footprint_narrower_than_the_drift_disc_is_widened_to_it` below);
/// this one is about the coarse cell being the footprint, and a
/// zero-share diffusion is the exact identity (`1 − share = 1`,
/// `share / 6 = 0`), so pinning it off isolates that statement instead of
/// measuring the two at once.
#[test]
fn a_seeded_cloud_rains_on_its_whole_coarse_cell_with_each_cells_own_phase() {
    let mut grid = HexGrid::from_radius(RADIUS);
    for cell in grid.cells_slice_mut() {
        cell.elevation = 200.0;
        cell.water_level = 0.0;
        cell.groundwater = 0.0;
        cell.humidity_surface = 0.0;
        cell.humidity_upper = 0.0;
        cell.cloud_water = 0.0;
    }
    // Half the map below freezing, half above, split on `q` so the two
    // halves cut across coarse cells rather than following them.
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.temperature = if coord.q < 0 { -5.0 } else { 15.0 };
        }
    }
    // One fine cell carries the whole cloud (cumulonimbus regime, same
    // magnitude as `phys_kk2000_heavy_cloud_rains`).
    grid.get_mut(HexCoord::new(0, 0))
        .expect("centre")
        .cloud_water = 12.0;

    let mut atmosphere = AtmosphereParams {
        // Freeze everything that could move or create droplets, so what
        // falls can only be the seeded cloud: no transport, no fog, no
        // uplift, no pump.
        cloud_advection_rate: 0.0,
        cloud_diffusion_rate: 0.0,
        fog_condensation_rate: 0.0,
        uplift_rate: 0.0,
        uplift_thermal_coef: 0.0,
        convective_diurnal_coef: 0.0,
        orographic_lift_coef: 0.0,
        // See this test's doc: pins the step-2b footprint widening off, so
        // what is measured here is the coarse cell alone.
        precip_neighbor_share: 0.0,
        ..AtmosphereParams::default()
    };
    // …and hold the vapour ↔ droplet transition still in BOTH directions,
    // the layer supersaturated so the saturation adjustment has no deficit
    // to evaporate the seeded cloud with. Same isolation, and the same
    // reason, as every other KK2000 fixture since #63 L2b.
    common::freeze_phase_transition(&mut atmosphere);
    let mut sim = coarse_world(atmosphere, grid, MoistCoarseMode::CoarsePrecip);
    sim.step_hour();

    let events = sim.precip_this_tick().clone();
    let grid = sim.grid();
    let seeded_index = grid.index_of(HexCoord::new(0, 0)).expect("centre exists");
    let wet: Vec<usize> = (0..grid.len())
        .filter(|&i| events[i].rain + events[i].snow > 0.0)
        .collect();

    // The seeded cell is not alone any more, and the wet set is the size
    // of a coarse cell (≈6.7 fine cells), not of a fine footprint.
    assert!(
        wet.contains(&seeded_index),
        "the seeded cell itself must be wet"
    );
    assert!(
        wet.len() > 1,
        "a cloud on one fine cell must rain on its whole coarse cell, {} cell(s) wet",
        wet.len()
    );

    // Uniform sheet: every wet cell received the same millimetres (design
    // note §4 option (a) at this step).
    let first = events[wet[0]].rain + events[wet[0]].snow;
    for &i in &wet {
        let got = events[i].rain + events[i].snow;
        assert!(
            (got - first).abs() <= 1e-6 * first.max(1.0),
            "the sheet must be uniform over the coarse cell: cell {i} got {got}, first {first}"
        );
    }

    // Phase per fine cell: the cold ones took snow and no rain, the warm
    // ones rain and no snow, out of the same sheet.
    let (mut cold_wet, mut warm_wet) = (0_usize, 0_usize);
    for &i in &wet {
        let cell = &grid.cells_slice()[i];
        if cell.temperature < 0.0 {
            assert!(
                events[i].snow > 0.0 && events[i].rain == 0.0,
                "cell {i} at {} °C must take snow only, rain={} snow={}",
                cell.temperature,
                events[i].rain,
                events[i].snow
            );
            cold_wet += 1;
        } else {
            assert!(
                events[i].rain > 0.0 && events[i].snow == 0.0,
                "cell {i} at {} °C must take rain only, rain={} snow={}",
                cell.temperature,
                events[i].rain,
                events[i].snow
            );
            warm_wet += 1;
        }
    }
    assert!(
        cold_wet > 0 && warm_wet > 0,
        "the fixture is only meaningful if one coarse cell straddles 0 °C: \
         {cold_wet} cold and {warm_wet} warm wet cells"
    );
}

/// Sentinel (ii): **strict conservation on the coarse mode**, 48 h,
/// real terrain, everything running — the pump, the fog, both advections,
/// the imposed regime, the precipitation.
///
/// The budget read is `Simulation::water_budget_total`: surface stocks per
/// cell, the upper layer through its coarse mirror, and the sky
/// reservoir. The mirror is an exact mean gather of the fine fields, so
/// the two agree by construction — which the last assertion checks.
///
/// 1e-6 relative over 48 h. The pro-rata drain is taken out of the fine
/// columns and dropped back uniformly, two different distributions of the
/// same mass — a rounding hole a naive implementation could leak through
/// (the retired `CoarseStock` mode's own such hole, a barycentric
/// interpolation instead of a broadcast, leaked +98 % of the budget in
/// ten years before it was caught here and fixed, see
/// `atmosphere::coarse`'s module doc).
#[test]
fn the_coarse_modes_conserve_water_over_two_days() {
    let mode = MoistCoarseMode::CoarsePrecip;
    let mut grid = HexGrid::from_radius(RADIUS);
    generate_terrain(
        &mut grid,
        &TerrainParams {
            seed: 42,
            ..TerrainParams::default()
        },
    );
    let mut sim = coarse_world(AtmosphereParams::default(), grid, mode);

    let initial = sim.water_budget_total();
    assert!(
        initial > 1.0,
        "{mode:?}: the fixture must start with water: {initial}"
    );
    // Precondition: the fog really is running, so "conserved" covers
    // the one pass the design note flags as writing into the coarse
    // stock from the surface layer (§9 risk 5).
    assert!(
        sim.atmosphere_params().fog_condensation_rate > 0.0,
        "{mode:?}: the fog must be on for this test to cover it"
    );

    for hour in 1..=48 {
        sim.step_hour();
        let total = sim.water_budget_total();
        let drift = (total - initial).abs() / initial;
        assert!(
            drift < 1e-6,
            "{mode:?} hour {hour}: water was lost or made, {initial} -> {total} \
             (relative drift {drift:.2e})"
        );
        // Nothing may go negative: on `CoarsePrecip` the pro-rata
        // removal is `cw × (1 − φ)` with `φ ∈ [0, 1]` by
        // construction, and this is where that argument is checked
        // rather than asserted (anti-pattern 4).
        let lowest = sim
            .grid()
            .iter()
            .map(|(_, c)| c.cloud_water)
            .fold(f32::INFINITY, f32::min);
        assert!(
            lowest >= 0.0,
            "{mode:?} hour {hour}: a column holds {lowest} mm of cloud water"
        );
    }

    // The coarse state is not a bookkeeping fiction: what it holds is
    // what the fine grid shows.
    let fine_upper: f32 = sim
        .grid()
        .iter()
        .map(|(_, c)| c.humidity_upper + c.cloud_water)
        .sum();
    let stock = sim.upper_water_total();
    assert!(
        (fine_upper - stock).abs() <= 1e-4 * stock.max(1.0),
        "{mode:?}: fine {fine_upper} vs coarse state {stock}"
    );
}

/// Sentinel (v), the property `CoarsePrecip` exists for: **a cloud seeded
/// on ONE fine column stays in that column**, hour after hour, instead of
/// being spread over its 1 km cell and eaten by the dry columns around it
/// — while the sheet it drops still covers the whole coarse cell.
///
/// This is what step 2c measured the coarse-stock mode losing. Below
/// 300 m it evaporated 4.8 to 7.3× what the fine path did, for 0.30 to
/// 0.62× the condensation, and a millimetre of cloud water lived 1.25 to
/// 2.2 h there against 5.0 to 6.3 h (`atmosphere::coarse`'s module doc,
/// `tests/diag_cloud_budget_by_altitude.rs`). The mechanism is the
/// saturation adjustment applied to a broadcast view: the condensate of
/// the one saturated column is handed to ~7 subsaturated ones (r6,
/// `Rc = 2`) and they evaporate it.
///
/// The fixture is built to make that difference visible in a few hours
/// and nothing else: the seeded column is **supersaturated** (no deficit
/// of its own, so on any mode its own adjustment takes nothing), every
/// other column is bone dry (a maximal deficit), the forward transition is
/// off (`condensation_rate = 0`, so no new cloud appears anywhere), and
/// every transport is off. The only thing that can move cloud water is the
/// saturation adjustment, and KK2000.
///
/// On `CoarsePrecip` the seeded column keeps its cloud and its neighbours
/// never hold any — the contrast this used to demonstrate against the
/// retired `CoarseStock` mode (which broadcast the cloud and lost it
/// within the hour) is now `atmosphere::coarse`'s module doc.
#[test]
fn a_cloud_seeded_on_one_column_survives_its_dry_neighbours_on_the_precip_mode() {
    const HOURS: u32 = 6;
    let seeded = |mode: MoistCoarseMode| -> (Simulation, usize) {
        let mut grid = HexGrid::from_radius(RADIUS);
        for cell in grid.cells_slice_mut() {
            cell.elevation = 200.0;
            cell.temperature = 15.0;
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
            cell.snow_level = 0.0;
            cell.humidity_surface = 0.0;
            // Bone dry aloft: `sat_upper` at this profile is several mm,
            // so every column carries a large deficit and would evaporate
            // any cloud water it is handed, in one hour.
            cell.humidity_upper = 0.0;
            cell.cloud_water = 0.0;
        }
        let centre = grid.get_mut(HexCoord::new(0, 0)).expect("centre");
        centre.cloud_water = 12.0;
        // …except the seeded column, held supersaturated so its OWN
        // adjustment has nothing to take: what the test measures is what
        // the NEIGHBOURS do to the cloud, not what its own column does.
        centre.humidity_upper = common::FROZEN_TRANSITION_UPPER_MM;
        let atmosphere = AtmosphereParams {
            initial_humidity_floor: 0.0,
            // No new cloud anywhere: the forward branch is a rate, and at
            // 0 the supersaturated column condenses nothing.
            condensation_rate: 0.0,
            cloud_advection_rate: 0.0,
            cloud_diffusion_rate: 0.0,
            fog_condensation_rate: 0.0,
            uplift_rate: 0.0,
            uplift_thermal_coef: 0.0,
            convective_diurnal_coef: 0.0,
            orographic_lift_coef: 0.0,
            regime_enabled: 0.0,
            // Isolates the coarse cell as the footprint, like sentinel
            // (i): a zero share is the exact identity of the widening
            // rule, not a weakening of it.
            precip_neighbor_share: 0.0,
            ..AtmosphereParams::default()
        };
        let mut sim = coarse_world(atmosphere, grid, mode);
        sim.set_uniform_wind(WindVec::default());
        let index = sim.grid().index_of(HexCoord::new(0, 0)).expect("centre");
        for _ in 0..HOURS {
            sim.step_hour();
        }
        (sim, index)
    };

    let (precip_sim, seeded_index) = seeded(MoistCoarseMode::CoarsePrecip);
    let cloud: Vec<f32> = precip_sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| c.cloud_water)
        .collect();

    // (a) The cloud is still there after six hours of dry neighbours.
    assert!(
        cloud[seeded_index] > 0.0,
        "the seeded cloud was gone in {HOURS} h: {} mm",
        cloud[seeded_index]
    );
    // (b) And it is still in ITS column: nothing was handed to a
    // neighbour, so no neighbour can hold any.
    let elsewhere: f32 = cloud
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != seeded_index)
        .map(|(_, &c)| c)
        .sum();
    assert!(
        elsewhere == 0.0,
        "the cloud must stay in its own column: {elsewhere} mm found elsewhere"
    );
    // (c) The sheet still covers the whole coarse cell — the point of the
    // mode is the 1 km islet, not a point shower.
    let wet = precip_sim
        .precip_this_tick()
        .iter()
        .filter(|d| d.rain + d.snow > 0.0)
        .count();
    assert!(
        wet > 1,
        "the sheet must cover the coarse cell, {wet} cell(s) wet on the last hour"
    );
}

/// Sentinel (iv): **a footprint narrower than the drift disc is widened to
/// it** (step 2b, `atmosphere::coarse::CoarseFootprint`), radius 6,
/// `Rc = 2`: a coarse cell holds ≈6.7 fine cells against the 37 cells of
/// the shipped `precip_spread_radius = 3` disc, so the rule fires here.
/// From r30 up it does not (46 to 56 fine cells per coarse), and the bench
/// says so: every metric on three seeds is bit-identical with and without
/// the rule.
///
/// The two runs differ only by `precip_neighbor_share`: at `0` the
/// diffusion is the exact identity (`1 − share = 1`, `share / 6 = 0`) and
/// the sheet is the coarse cell alone; at the shipped `0.35` it is the
/// coarse cell feathered outward. What must hold: strictly more cells wet,
/// the same water fallen (the operator is doubly stochastic), and a peak
/// that goes down rather than up.
#[test]
fn a_footprint_narrower_than_the_drift_disc_is_widened_to_it() {
    let seeded = |share: f32| -> Vec<f32> {
        let mut grid = HexGrid::from_radius(RADIUS);
        let coords: Vec<HexCoord> = grid.coords().copied().collect();
        for coord in coords {
            if let Some(cell) = grid.get_mut(coord) {
                cell.elevation = 200.0;
                cell.temperature = 15.0;
                cell.water_level = 0.0;
                cell.groundwater = 0.0;
                cell.humidity_surface = 0.0;
                cell.humidity_upper = 0.0;
                cell.cloud_water = 0.0;
            }
        }
        grid.get_mut(HexCoord::new(0, 0))
            .expect("centre")
            .cloud_water = 12.0;
        let mut atmosphere = AtmosphereParams {
            cloud_advection_rate: 0.0,
            cloud_diffusion_rate: 0.0,
            fog_condensation_rate: 0.0,
            uplift_rate: 0.0,
            uplift_thermal_coef: 0.0,
            convective_diurnal_coef: 0.0,
            orographic_lift_coef: 0.0,
            precip_neighbor_share: share,
            ..AtmosphereParams::default()
        };
        common::freeze_phase_transition(&mut atmosphere);
        let mut sim = coarse_world(atmosphere, grid, MoistCoarseMode::CoarsePrecip);
        sim.step_hour();
        sim.precip_this_tick()
            .iter()
            .map(|d| d.rain + d.snow)
            .collect()
    };

    let confined = seeded(0.0);
    let widened = seeded(0.35);

    let wet = |v: &[f32]| v.iter().filter(|&&x| x > 0.0).count();
    let total = |v: &[f32]| f64::from(v.iter().sum::<f32>());
    let peak = |v: &[f32]| v.iter().copied().fold(0.0_f32, f32::max);

    assert!(
        wet(&confined) > 1,
        "the confined run must already wet a whole coarse cell: {} cells",
        wet(&confined)
    );
    assert!(
        wet(&widened) > wet(&confined),
        "the drift disc must widen the footprint: {} cells against {}",
        wet(&widened),
        wet(&confined)
    );
    let (a, b) = (total(&confined), total(&widened));
    assert!(
        (a - b).abs() <= 1e-5 * a,
        "widening must move water, not make or lose it: {a} against {b}"
    );
    assert!(
        peak(&widened) <= peak(&confined),
        "the gradient decreases outward, so the peak cannot rise: {} against {}",
        peak(&widened),
        peak(&confined)
    );
}

/// Sentinel (vi), the #158 lever: **a cloud is not made to disappear by
/// the skirt around it**. Same fixture as sentinel (i) — one loaded
/// column on an otherwise dry coarse cell — except that the six
/// neighbours of the loaded column carry the thin veil the cloud
/// diffusion leaves behind (0.003 mm, the value measured at r8 on
/// `phys_rain_footprint_is_a_disc`, JOURNAL 2026-09-07).
///
/// Under the **counting** cloud fraction the coarse mode used until
/// 2026-09-07 that veil was worth as much as the cloud: seven columns out
/// of eleven "held cloud", so `f_c` = 0.64 and the in-cloud content
/// `q_c / f_c` fell to 0.143 mm, under `precip_crit_mm` = 0.15 mm — the
/// cell rained nothing while holding a millimetre of cumulonimbus. The
/// mass-weighted content (`Σ cw² / Σ cw`) reads 0.98 mm on the same
/// distribution and the floor does not bite.
///
/// What is pinned, and why it is a physics statement rather than a
/// threshold: **adding a negligible amount of water to the neighbours of
/// a raining cloud cannot stop the rain.** The run with the skirt must
/// drop essentially the sheet the run without it drops — the extra mass
/// is 1.8 % of the stock, so the sheet may differ by a few percent, not
/// by everything.
#[test]
fn a_diffusion_skirt_around_a_cloud_does_not_switch_its_rain_off() {
    /// The veil `cloud_diffusion` leaves on the ring around a cloud.
    const SKIRT_MM: f32 = 0.003;

    let sheet_with_skirt = |skirt: f32| -> f32 {
        let mut grid = HexGrid::from_radius(RADIUS);
        for cell in grid.cells_slice_mut() {
            cell.elevation = 200.0;
            cell.temperature = 15.0;
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
            cell.humidity_surface = 0.0;
            cell.humidity_upper = 0.0;
            cell.cloud_water = 0.0;
        }
        let centre = HexCoord::new(0, 0);
        for &d in &[
            HexCoord::new(1, 0),
            HexCoord::new(1, -1),
            HexCoord::new(0, -1),
            HexCoord::new(-1, 0),
            HexCoord::new(-1, 1),
            HexCoord::new(0, 1),
        ] {
            grid.get_mut(d)
                .expect("neighbour of the centre")
                .cloud_water = skirt;
        }
        grid.get_mut(centre).expect("centre").cloud_water = 1.0;

        let mut atmosphere = AtmosphereParams {
            // Everything that could move or create droplets is off, so
            // what falls can only be the seeded cloud and its skirt.
            cloud_advection_rate: 0.0,
            cloud_diffusion_rate: 0.0,
            fog_condensation_rate: 0.0,
            uplift_rate: 0.0,
            uplift_thermal_coef: 0.0,
            convective_diurnal_coef: 0.0,
            orographic_lift_coef: 0.0,
            // The coarse cell alone is the footprint (sentinel (i)'s
            // reason): a zero share is the exact identity of the
            // widening rule.
            precip_neighbor_share: 0.0,
            ..AtmosphereParams::default()
        };
        common::freeze_phase_transition(&mut atmosphere);
        let mut sim = coarse_world(atmosphere, grid, MoistCoarseMode::CoarsePrecip);
        sim.step_hour();
        let index = sim.grid().index_of(centre).expect("centre");
        let events = sim.precip_this_tick();
        events[index].rain + events[index].snow
    };

    let bare = sheet_with_skirt(0.0);
    let skirted = sheet_with_skirt(SKIRT_MM);

    assert!(
        bare > 0.0,
        "the fixture is only meaningful if the bare cloud rains: {bare} mm"
    );
    assert!(
        skirted > 0.0,
        "a {SKIRT_MM} mm veil on the six neighbours switched the rain off: \
         {bare} mm bare against {skirted} mm skirted"
    );
    let gap = (skirted - bare).abs() / bare;
    assert!(
        gap < 0.10,
        "the skirt carries 1.8 % of the cell's water and must not change the \
         sheet by more than a few percent: {bare} mm bare against {skirted} mm \
         skirted ({:.1} %)",
        gap * 100.0
    );
}
