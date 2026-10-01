//! Micro-test for the rain footprint spread (L3, 2026-09-06): with the
//! shipped default `precip_spread_radius = 3.0`, one saturated column
//! rains not just on itself and its 6 immediate neighbors (the historical
//! footprint) but on a disc of radius 3 around it, with a gradient
//! decreasing outward — the owner's ask ("when a cloud rains, it also
//! rains on its neighbours with a gradient, up to radius 3").
//!
//! Setup (radius 8 torus, 217 cells, transport-safe per the "no
//! radius-0 transport" rule and far past the spread radius so nothing wraps
//! back onto the disc from the far side of the torus, per
//! `coord::torus_lattice_vectors(8)`'s 17-hex wrap distance): flat
//! terrain, warm (rain, not snow) everywhere, one column pre-loaded with
//! `cloud_water = 1.0` mm (cumulonimbus regime, same magnitude as
//! `phys_kk2000_heavy_cloud_rains.rs`), every other column bone dry.
//! `cloud_advection_rate = 0` freezes the cloud on its source cell for
//! the hour measured ("wind off" in the brief's sense: the only other
//! path wind could reach precipitation through, the updraft trigger, is
//! off by default — `updraft_ref_ms = 0.0` — so this is the whole
//! isolation the fixture needs, same idiom as the existing KK2000
//! fixtures). `freeze_phase_transition` holds `humidity_upper` /
//! `cloud_water`'s OTHER movers still, so only KK2000 autoconversion and
//! the footprint spread act on the one seeded cloud.
//!
//! ## Two modes, two shapes, one set of tests (2026-09-07)
//!
//! Both tests below run under either moist-layer mode
//! (`HEXSIM_MOIST_COARSE`) and branch on `sim.moist_coarse_path()`. This
//! is not the fine-path disc reformulated in coarse terms: the coarse
//! path genuinely produces a **different shape**, a uniform plateau over
//! the source's coarse cell with a feathered edge, never the fine path's
//! smooth per-ring gradient (measured while chasing #158's
//! `CoarseFootprint` reach bug, see `atmosphere::coarse::CoarseFootprint`'s
//! doc). The fine-path assertions (the `else` branches below) are
//! untouched byte for byte; the coarse branches assert a footprint and a
//! gradient property native to that shape, derived from the moist mesh's
//! own partition (`Simulation::moist_mesh_fine_to_coarse`,
//! `Simulation::moist_mesh_coarse_radius`) rather than hardcoded, so nextest
//! actually exercises both paths under the same two test names instead of
//! measuring only whichever `HEXSIM_MOIST_COARSE` happened to be set.

use hexsim_core::atmosphere::AtmosphereParams;
use hexsim_core::coord::HexCoord;
use hexsim_core::grid::HexGrid;
use hexsim_core::groundwater::GroundwaterParams;
use hexsim_core::hydro::HydroParams;
use hexsim_core::simulation::Simulation;
use hexsim_core::snow::SnowParams;
use hexsim_core::temperature::TemperatureParams;
use hexsim_core::wind::WindParams;
use std::collections::HashSet;

mod common;

const RADIUS: i32 = 8;
const SPREAD_RADIUS: i32 = 3;

fn saturated_column_sim() -> Simulation {
    let mut grid = HexGrid::from_radius(RADIUS);
    let coords: Vec<HexCoord> = grid.coords().copied().collect();
    for coord in coords {
        if let Some(cell) = grid.get_mut(coord) {
            cell.elevation = 200.0;
            cell.temperature = 15.0; // warm: rain branch, not snow
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
            cell.humidity_surface = 0.0;
            cell.humidity_upper = 0.0;
            cell.cloud_water = 0.0;
        }
    }
    if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
        cell.cloud_water = 1.0;
    }

    let mut atmo = AtmosphereParams {
        cloud_advection_rate: 0.0,
        ..AtmosphereParams::default()
    };
    common::freeze_phase_transition(&mut atmo);
    assert!(
        (atmo.precip_spread_radius - 3.0).abs() < 1e-6,
        "fixture assumption: the engine default footprint radius is 3, got {}",
        atmo.precip_spread_radius
    );

    Simulation::new(
        grid,
        HydroParams::default(),
        atmo,
        GroundwaterParams::default(),
        SnowParams::default(),
        TemperatureParams::default(),
        WindParams::default(),
    )
}

/// Mirrors `atmosphere::coarse::disc_cells` (private to the crate, so
/// re-derived here): cells in a hex disc of `rings` rings, capped by the
/// torus.
fn disc_cell_count(rings: u32, fine_len: usize) -> usize {
    let k = usize::try_from(rings).expect("ring count fits usize");
    (3 * k * (k + 1) + 1).min(fine_len)
}

/// Mirrors `atmosphere::precipitation::precip_spread_passes` (also
/// private): nearest ring count, floored at 1. Same conversion, same
/// justification for the cast as the production twin — never negative
/// (rounded then floored at 1.0 on the line above) and nowhere near
/// `u32::MAX` (`precip_spread_radius` is a human-configured knob, 1 or 3
/// in this fixture).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded and floored at 1.0 on the line above: never negative, and \
              precip_spread_radius is a human-configured knob (1 or 3 in this \
              fixture), nowhere near u32::MAX — same justification as the \
              production twin, atmosphere::precipitation::precip_spread_passes"
)]
fn passes_from_radius(radius_hexes: f32) -> u32 {
    radius_hexes.round().max(1.0) as u32
}

/// Mirrors `atmosphere::coarse::CoarseFootprint::of`'s decision, from
/// public quantities only (`Simulation::moist_mesh_fine_to_coarse`'s doc
/// explains why a black-box test must recompute this rather than read the
/// production decision back: doing the latter could not have caught the
/// exact bug this decision shipped with until 2026-09-07).
fn expected_extra_passes(spread_radius: f32, rc: i32, fine_len: usize, coarse_len: usize) -> u32 {
    let raw_passes = passes_from_radius(spread_radius);
    let mean_fine_per_coarse = fine_len / coarse_len.max(1);
    if mean_fine_per_coarse < disc_cell_count(raw_passes, fine_len) {
        raw_passes.saturating_sub(u32::try_from(rc).expect("Rc is a grid radius, never negative"))
    } else {
        0
    }
}

/// Hex-hop distance (toric, `HexGrid::neighbor_indices_toric`), multi-source
/// BFS from every index in `sources` to every cell of `grid`. `0` on the
/// sources themselves, `1` on their immediate neighbors not already a
/// source, and so on. The torus is finite and connected, so every cell is
/// eventually reached from a non-empty source set.
fn hop_distance_from(grid: &HexGrid, sources: &[usize]) -> Vec<u32> {
    let mut dist = vec![u32::MAX; grid.len()];
    let mut frontier: Vec<usize> = Vec::new();
    for &s in sources {
        if dist[s] == u32::MAX {
            dist[s] = 0;
            frontier.push(s);
        }
    }
    let mut d = 0_u32;
    while !frontier.is_empty() {
        d += 1;
        let mut next = Vec::new();
        for &i in &frontier {
            for k in grid.neighbor_indices_toric(i) {
                if dist[k] == u32::MAX {
                    dist[k] = d;
                    next.push(k);
                }
            }
        }
        frontier = next;
    }
    dist
}

/// The source's coarse-cell membership (fine indices sharing
/// `Simulation::moist_mesh_fine_to_coarse`'s coarse index with `source`),
/// its coarse radius `Rc`, and the expected extra diffusion passes for
/// `spread_radius` on this mesh — the three quantities every coarse
/// assertion below is built from, all derived from the mesh rather than
/// hardcoded.
fn coarse_footprint_ground_truth(
    sim: &Simulation,
    grid: &HexGrid,
    source: HexCoord,
    spread_radius: f32,
) -> (Vec<usize>, i32, u32) {
    let fine_to_coarse = sim.moist_mesh_fine_to_coarse();
    let rc = sim.moist_mesh_coarse_radius();
    let source_idx = grid.index_of(source).expect("source exists");
    let source_coarse = fine_to_coarse[source_idx];
    let members: Vec<usize> = (0..fine_to_coarse.len())
        .filter(|&i| fine_to_coarse[i] == source_coarse)
        .collect();
    let coarse_len = HexGrid::from_radius(rc).len();
    let passes = expected_extra_passes(spread_radius, rc, grid.len(), coarse_len);
    (members, rc, passes)
}

/// Coarse-path assertions for
/// [`one_saturated_column_wets_a_disc_of_radius_three_with_a_decreasing_gradient`]
/// (split out from the test itself only to stay under clippy's
/// `too_many_lines`): footprint containment, the interior plateau, the
/// strict hop-by-hop decrease beyond it, and mass conservation against the
/// same fixture with diffusion switched off. See that test's own doc for
/// what shape this checks and why.
fn assert_coarse_disc_footprint(
    sim: &Simulation,
    grid: &HexGrid,
    precip: &hexsim_core::atmosphere::PrecipitationMap,
    center: HexCoord,
) {
    let (members, rc, passes) = coarse_footprint_ground_truth(sim, grid, center, 3.0);
    let dist = hop_distance_from(grid, &members);

    for (idx, &coord) in grid.coords_slice().iter().enumerate() {
        let rain = precip[idx].rain;
        if dist[idx] <= passes {
            assert!(
                rain > 0.0,
                "coarse footprint (Rc={rc}, passes={passes}): cell {idx} ({coord:?}) \
                 at hop-distance {} from the source's coarse cell must be wet",
                dist[idx]
            );
        } else {
            assert_eq!(
                rain.to_bits(),
                0.0f32.to_bits(),
                "coarse footprint (Rc={rc}, passes={passes}): cell {idx} ({coord:?}) \
                 at hop-distance {} is beyond the dilated coarse cell, must stay \
                 exactly dry, got {rain}",
                dist[idx]
            );
        }
    }

    // Plateau: members whose 6 toric neighbours are themselves all
    // members take no diffusion at all (the pass's self/neighbour split
    // reduces to the identity when every term is the same value), so
    // they must read the exact same sheet, to f32 rounding. Boundary
    // members (touching a non-member) lose a little to the pass and are
    // deliberately excluded here — that loss is a real, expected effect,
    // not the plateau this checks.
    let member_set: HashSet<usize> = members.iter().copied().collect();
    let interior: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&i| {
            grid.neighbor_indices_toric(i)
                .iter()
                .all(|k| member_set.contains(k))
        })
        .collect();
    assert!(
        !interior.is_empty(),
        "fixture assumption: the source's coarse cell must have at least one \
         fully-interior member (the source itself, ringed by its 6 neighbours, \
         all members at r8), otherwise this plateau check is vacuous"
    );
    let plateau = precip[interior[0]].rain;
    for &i in &interior {
        let v = precip[i].rain;
        assert!(
            (v - plateau).abs() <= 1e-6 * plateau,
            "interior member {i} breaks the plateau: {v} vs {plateau}"
        );
    }

    // Strict decrease hop by hop beyond the plateau's hop (0), up to the
    // footprint's own edge (`passes`): each hop must be wetter than the
    // next. Valid for `passes <= 1` (true at r8 for both fixture radii):
    // a hop-0 cell here may be a lower-value boundary member rather than
    // the plateau itself, but even a boundary member only loses a
    // fraction of the sheet, comfortably above what a single diffusion
    // hop hands to hop 1 (checked below, not assumed).
    for hop in 0..passes {
        let this_layer_min = (0..grid.len())
            .filter(|&i| dist[i] == hop)
            .map(|i| precip[i].rain)
            .fold(f32::INFINITY, f32::min);
        let next_layer_max = (0..grid.len())
            .filter(|&i| dist[i] == hop + 1)
            .map(|i| precip[i].rain)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            this_layer_min > next_layer_max,
            "hop {hop} ({this_layer_min}) must dominate hop {} ({next_layer_max})",
            hop + 1
        );
    }

    // Mass conservation: the same fixture with diffusion switched off
    // must drop the identical total. `spread_precip_further` is doubly
    // stochastic (its own doc) so this is not a coincidence to re-derive
    // per radius, it is the operator's row-sum-1 invariant — but
    // exercising it end to end through this fixture's actual pipeline
    // (not just the operator in isolation, already covered by
    // `precipitation::tests::diffusion_conserves_mass_on_any_torus_
    // share_and_pass_count`) is what catches a leak anywhere else in
    // `step_moist_precip`.
    let mut sim_undiffused = saturated_column_sim();
    sim_undiffused.update_param("atmosphere.precip_neighbor_share", 0.0);
    sim_undiffused.step_hour();
    let undiffused = sim_undiffused.precip_this_tick().clone();
    let diffused_total: f64 = precip.iter().map(|p| f64::from(p.rain)).sum();
    let undiffused_total: f64 = undiffused.iter().map(|p| f64::from(p.rain)).sum();
    assert!(
        (diffused_total - undiffused_total).abs() <= 1e-5 * undiffused_total,
        "diffusion must conserve the sheet's mass: diffused {diffused_total} vs \
         undiffused {undiffused_total}"
    );
}

/// One precipitation hour after a single saturated column.
///
/// **Fine path** (`else` branch, untouched since 2026-09-06): the
/// rained-on set is exactly the disc of radius `precip_spread_radius` (3)
/// around the source, gradient decreasing outward. Distances 0/1 are
/// pinned exactly isotropic (a single diffusion pass is uniform by
/// construction, see `precipitation::spread_precip_further`'s doc);
/// distances 2/3 are pinned bounded, not exactly equal — the isotropy
/// caveat found while implementing this (a finding, not a claim of the
/// original design note): the hex lattice's short-range 6-neighbor walk
/// isn't rotationally symmetric at low step counts.
///
/// **Coarse path** (`if` branch, added 2026-09-07): the rained-on set is
/// the source's coarse cell (`Rc = 2` at r8, 13 fine cells reaching hex
/// distance 2) dilated by `expected_extra_passes` more diffusion hops (1
/// at the shipped default, per `CoarseFootprint`'s corrected reach) —
/// never a disc centred on the source, because the coarse path never
/// drops a point source. Interior members (every one of their 6 toric
/// neighbours is itself a member, so `spread_precip_further`'s single
/// pass is the identity on them) sit on an exact plateau; the hop beyond
/// it is strictly drier; nothing further out is wet at all; and the total
/// mass matches the same fixture run with diffusion switched off
/// (`precip_neighbor_share = 0`, `phys_coarse_upper_layer`'s own idiom for
/// isolating the undiffused sheet), because `spread_precip_further` is
/// doubly stochastic and cannot have moved any of it off the grid.
#[test]
fn one_saturated_column_wets_a_disc_of_radius_three_with_a_decreasing_gradient() {
    let mut sim = saturated_column_sim();
    let is_coarse = sim.moist_coarse_path();
    sim.step_hour();

    let precip = sim.precip_this_tick().clone();
    let grid = sim.grid();
    let center = HexCoord::new(0, 0);

    if is_coarse {
        assert_coarse_disc_footprint(&sim, grid, &precip, center);
    } else {
        let mut rings: [Vec<f32>; SPREAD_RADIUS as usize + 1] = Default::default();
        let mut beyond_the_disc_rained = false;
        for (idx, &coord) in grid.coords_slice().iter().enumerate() {
            let d = center.distance(coord);
            let rain = precip[idx].rain;
            if d > SPREAD_RADIUS {
                if rain != 0.0 {
                    beyond_the_disc_rained = true;
                }
            } else {
                let ring = usize::try_from(d).expect("0 <= d <= SPREAD_RADIUS");
                rings[ring].push(rain);
            }
        }

        assert!(
            !beyond_the_disc_rained,
            "a cell beyond distance {SPREAD_RADIUS} from the source received rain: \
             the footprint leaked past its configured radius"
        );

        let min_max = |v: &[f32]| -> (f32, f32) {
            (
                v.iter().copied().fold(f32::INFINITY, f32::min),
                v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            )
        };

        assert_eq!(rings[0].len(), 1, "ring 0 is the source cell alone");
        assert!(rings[0][0] > 0.0, "the source cell itself must still rain");

        assert_eq!(rings[1].len(), 6, "ring 1 has 6 cells");
        let (r1_min, r1_max) = min_max(&rings[1]);
        assert!(r1_min > 0.0, "every ring-1 cell must be wet");
        assert!(
            (r1_max - r1_min).abs() < 1e-6,
            "ring 1 must be exactly isotropic (single diffusion pass, uniform \
             by construction): {:?}",
            rings[1]
        );

        assert_eq!(rings[2].len(), 12, "ring 2 has 12 cells");
        let (r2_min, r2_max) = min_max(&rings[2]);
        assert!(r2_min > 0.0, "every ring-2 cell must be wet");
        assert!(
            r2_max / r2_min < 3.0,
            "ring 2 anisotropy grew past the bound measured in \
             precipitation::tests::footprint_gradient_decreases_with_ring_distance: \
             {r2_min}..{r2_max}"
        );

        assert_eq!(rings[3].len(), 18, "ring 3 has 18 cells");
        let (r3_min, r3_max) = min_max(&rings[3]);
        assert!(r3_min > 0.0, "every ring-3 cell must be wet");
        assert!(
            r3_max / r3_min < 5.0,
            "ring 3 anisotropy grew past the bound measured in \
             precipitation::tests::footprint_gradient_decreases_with_ring_distance: \
             {r3_min}..{r3_max}"
        );

        assert!(
            rings[0][0] > r1_min,
            "ring 0 ({}) must dominate ring 1 ({r1_min})",
            rings[0][0]
        );
        assert!(
            r1_min > r2_max,
            "ring 1 ({r1_min}) must dominate ring 2 ({r2_max})"
        );
        assert!(
            r2_min > r3_max,
            "ring 2 ({r2_min}) must dominate ring 3 ({r3_max})"
        );
    }
}

/// `precip_spread_radius = 1` (the pre-existing behavior, still reachable
/// via `just param`).
///
/// **Fine path** (`else` branch, untouched since 2026-09-06): confines
/// the same saturated column to the source plus its 6 immediate
/// neighbors — the footprint the owner's screenshots showed as isolated
/// islets.
///
/// **Coarse path** (`if` branch, added 2026-09-07): this is the **floor
/// footprint**, not a relaxed threshold. At r8 the source's coarse cell
/// (`Rc = 2`, 13 fine cells out to hex distance 2) is already wider than
/// the 7-cell disc `precip_spread_radius = 1` would ask for
/// (`disc_cell_count(1, _) = 7 < 13`), so `CoarseFootprint`'s trigger
/// never fires and `expected_extra_passes` is 0 independently of the
/// `Rc` fix (#158): the wet set is exactly the coarse cell, no more, no
/// less, and it cannot be narrower — this is a property of the mesh
/// (`Simulation::moist_mesh_coarse_radius`), not a parameter to move. See
/// `atmosphere::coarse::CoarseFootprint`'s doc.
#[test]
fn radius_one_confines_the_same_column_to_source_and_immediate_neighbors() {
    let mut sim = saturated_column_sim();
    let is_coarse = sim.moist_coarse_path();
    sim.update_param("atmosphere.precip_spread_radius", 1.0);
    sim.step_hour();

    let precip = sim.precip_this_tick().clone();
    let grid = sim.grid();
    let center = HexCoord::new(0, 0);

    if is_coarse {
        let (members, rc, passes) = coarse_footprint_ground_truth(&sim, grid, center, 1.0);
        assert_eq!(
            passes, 0,
            "fixture assumption: at r8 the coarse cell (Rc={rc}) is already wider than \
             disc(1), so the floor footprint must not be dilated any further"
        );
        let member_set: HashSet<usize> = members.iter().copied().collect();

        for (idx, &coord) in grid.coords_slice().iter().enumerate() {
            let rain = precip[idx].rain;
            if member_set.contains(&idx) {
                assert!(
                    rain > 0.0,
                    "coarse floor footprint (Rc={rc}): member cell {idx} ({coord:?}) \
                     must be wet"
                );
            } else {
                assert_eq!(
                    rain.to_bits(),
                    0.0f32.to_bits(),
                    "coarse floor footprint (Rc={rc}): cell {idx} ({coord:?}) is outside \
                     the source's coarse cell, must stay exactly dry, got {rain}"
                );
            }
        }

        // No diffusion at all (`passes == 0`): every member, not just an
        // interior subset, is the untouched, exactly uniform sheet.
        let plateau = precip[members[0]].rain;
        for &i in &members {
            let v = precip[i].rain;
            assert!(
                (v - plateau).abs() <= 1e-6 * plateau,
                "member {i} breaks the floor footprint's plateau: {v} vs {plateau}"
            );
        }

        // Mass conservation: with `passes == 0` there is no diffusion
        // pass to run at all, so the sheet the fixture drops is already
        // the undiffused one — summing it must match a
        // `precip_neighbor_share = 0` rerun of the same fixture exactly,
        // not just approximately, and confirms nothing upstream of
        // `distribute_precipitation` silently changed the mass because
        // of the `Rc` fix.
        let mut sim_undiffused = saturated_column_sim();
        sim_undiffused.update_param("atmosphere.precip_spread_radius", 1.0);
        sim_undiffused.update_param("atmosphere.precip_neighbor_share", 0.0);
        sim_undiffused.step_hour();
        let undiffused = sim_undiffused.precip_this_tick().clone();
        let diffused_total: f64 = precip.iter().map(|p| f64::from(p.rain)).sum();
        let undiffused_total: f64 = undiffused.iter().map(|p| f64::from(p.rain)).sum();
        assert!(
            (diffused_total - undiffused_total).abs() <= 1e-5 * undiffused_total,
            "the floor footprint must conserve the sheet's mass: {diffused_total} vs \
             {undiffused_total}"
        );
    } else {
        for (idx, &coord) in grid.coords_slice().iter().enumerate() {
            let d = center.distance(coord);
            let rain = precip[idx].rain;
            if d <= 1 {
                assert!(rain > 0.0, "distance {d}: cell {idx} must be wet at R=1");
            } else {
                assert_eq!(
                    rain.to_bits(),
                    0.0f32.to_bits(),
                    "distance {d}: cell {idx} must stay dry at R=1, got {rain}"
                );
            }
        }
    }
}
