use serde::{Deserialize, Serialize};

use crate::grid::HexGrid;

/// Default capacity of a cell at `permeability=1.0`. Single source of
/// truth for components that need a static reference for "full water
/// table" (e.g. the evapotranspiration proxy in `atmosphere.rs`).
pub const DEFAULT_MAX_CAPACITY_MM: f32 = 100.0;

#[derive(Clone, Serialize, Deserialize)]
pub struct GroundwaterParams {
    pub infiltration_rate: f32,
    pub diffusion_rate: f32,
    // Max capacity when permeability=1.0. The actual capacity of a cell
    // = max_capacity * permeability. Impermeable rock stores little.
    pub max_capacity: f32,
    /// Baseflow recession coefficient (linear reservoir, /day). At each
    /// daily slice, a fraction `baseflow_coef` of the water table seeps
    /// back to the surface: `seepage = baseflow_coef × groundwater`. This
    /// is Maillet's recession (1905): a spring's discharge decays
    /// exponentially between two recharges, `Q(t) = Q₀·e^(−αt)`. Without
    /// it, the only water table → surface path was resurgence at
    /// `gw > capacity`, which never triggered (measured #107: the water
    /// table settles at ~15% of its capacity, resurgence dead, "flashy"
    /// rivers with no baseflow). Since the water table is concentrated in
    /// basins/valleys (measured: 97% of the stock in 5% of the cells),
    /// this uniform term naturally feeds the low cells, the ones that
    /// carry rivers. Conservative: `groundwater → water_level`.
    #[serde(default)]
    pub baseflow_coef: f32,
}

impl Default for GroundwaterParams {
    fn default() -> Self {
        Self {
            // 0.05 (vs 0.01): at `water_level=1mm`, infiltration = 0.025
            // mm/day per cell in unsaturated regime (perm 0.5). Lets the
            // water table fill up significantly over 1-2 years instead of
            // asymptoting to empty over decades.
            infiltration_rate: 0.05,
            diffusion_rate: 0.03,
            // 100 (vs 5): lets the water table be a real primary
            // freshwater stock (~50 mm × 2790 cells = ~140,000 mm
            // cumulative, vs 7000 mm of surface currently). With this
            // capacity, the water table can feed rivers in the dry
            // season via resurgence (groundwater > capacity → return to
            // water_level). A true physical aquifer means meters of
            // column, but 100 mm is already a big improvement over the
            // initial 5 mm and avoids breaking the global calibration.
            max_capacity: DEFAULT_MAX_CAPACITY_MM,
            // 0.0 by default: baseflow is an ongoing calibration effort
            // (#107). The retained value is set after proof on the
            // sustainability metric + strict conservation + climate.
            baseflow_coef: 0.0,
        }
    }
}

// Underground cycle: infiltration → diffusion → resurgence.
//
// The water table is the "slow path" of water, as opposed to runoff
// (fast path). The infiltration rate controls the proportion of
// surface water that enters the soil on each tick.
//
// Underground diffusion is much slower than atmospheric diffusion: water
// in the soil moves slowly laterally through the porous rock. This is
// the mechanism that distributes water under hills and feeds the
// springs downhill.
pub fn step_groundwater(current: &HexGrid, next: &mut HexGrid, params: &GroundwaterParams) {
    let n = current.len();
    let cur_cells = current.cells_slice();

    // Copy current → next via indexed slice (1 memcpy instead of N lookups).
    next.cells_slice_mut().clone_from_slice(cur_cells);

    // Step 1: Infiltration (surface → water table)
    // Weighted by local permeability: impermeable soil → little
    // infiltration. Read on `current`, cell-local write on `next` → no
    // contention, can run by pure index over the slice.
    let next_cells = next.cells_slice_mut();
    for i in 0..n {
        let cell = &cur_cells[i];
        if cell.water_level <= 0.0 {
            continue;
        }
        let capacity = cell.permeability * params.max_capacity;
        let saturation = if capacity > 0.0 {
            (cell.groundwater / capacity).clamp(0.0, 1.0)
        } else {
            1.0
        };
        // Frozen soil: linear transition between -2°C (impermeable) and
        // 0°C. Meltwater at the foot of glaciers stays on the surface →
        // runs off.
        let frozen_factor = f32::midpoint(cell.temperature, 2.0).clamp(0.0, 1.0);
        let effective_rate =
            params.infiltration_rate * cell.permeability * (1.0 - saturation) * frozen_factor;
        let infiltration =
            (effective_rate * cell.water_level).min((capacity - cell.groundwater).max(0.0));
        let nc = &mut next_cells[i];
        nc.water_level -= infiltration;
        nc.groundwater += infiltration;
    }

    // Step 2: Underground flow (conservative transfers, gravity-driven)
    //
    // Groundwater follows gravity via the piezometric level: piezometric
    // = elevation + groundwater. This is the "pressure" of water
    // underground. Water flows from the high piezo level to the low
    // one: a water table in the mountains (elev=400, gw=1 → piezo=401)
    // flows toward the plain (elev=50, gw=3 → piezo=53) even if the
    // plain has more groundwater.
    //
    // ASSUMED mm/m MIX (discovered in #104): gw is in mm, elevation in
    // m. The same hybrid as fixed in `effective_elevation`, but here it
    // drives a slow diffusion capped at ~100 mm of stock, not surface
    // topology, and no ablation has measured it. Transition to SI: to be
    // settled with the Darcy rework of the water table (post-#105), not
    // sneaked in as part of the surface effort.
    //
    // Post-infiltration snapshot of the needed fields (gw, perm, elev) in
    // per-cell indexed Vecs. Avoids the intermediate HashMap and lets
    // `next.cells_slice_mut()` be mutated freely in the transfer loop.
    let mut snap_gw: Vec<f32> = Vec::with_capacity(n);
    let mut snap_perm: Vec<f32> = Vec::with_capacity(n);
    let mut snap_elev: Vec<f32> = Vec::with_capacity(n);
    for cell in next.cells_slice() {
        snap_gw.push(cell.groundwater);
        snap_perm.push(cell.permeability);
        snap_elev.push(cell.elevation);
    }

    let base_rate = params.diffusion_rate / 6.0;
    let next_cells = next.cells_slice_mut();
    for i in 0..n {
        let piezometric = snap_elev[i] + snap_gw[i];
        // Toric neighborhood: the water table also diffuses across the
        // seam (physical piezometric gradient, periodic terrain). j == i
        // (degenerate grid) → zero diff, zero transfer: conservative.
        let neighbors = current.neighbor_indices_toric(i);
        for j in neighbors {
            let neighbor_piezo = snap_elev[j] + snap_gw[j];
            if piezometric > neighbor_piezo {
                let perm_factor = snap_perm[i].min(snap_perm[j]);
                let diff = piezometric - neighbor_piezo;
                let available = next_cells[i].groundwater;
                let transfer = (base_rate * perm_factor * diff).min(available.max(0.0));
                if transfer > 0.0 {
                    next_cells[i].groundwater -= transfer;
                    next_cells[j].groundwater += transfer;
                }
            }
        }
    }

    // Step 3: Baseflow (Maillet recession): a fraction of the water
    // table seeps back to the surface at each slice, BEFORE overflow.
    // This is the mechanism that sustains rivers between two rains
    // (#107): without it the water table settles below its capacity
    // and never overflows, so no water comes back up. Conservative:
    // gw → water_level.
    if params.baseflow_coef > 0.0 {
        for nc in next.cells_slice_mut() {
            if nc.groundwater > 0.0 {
                let seepage = (params.baseflow_coef * nc.groundwater).min(nc.groundwater);
                nc.groundwater -= seepage;
                nc.water_level += seepage;
            }
        }
    }

    // Step 4: Resurgence (water table overflows → water rises to the surface)
    for nc in next.cells_slice_mut() {
        let capacity = nc.permeability * params.max_capacity;
        if nc.groundwater > capacity {
            let surplus = nc.groundwater - capacity;
            nc.groundwater = capacity;
            nc.water_level += surplus;
        }
    }
}

#[must_use]
pub fn total_groundwater(grid: &HexGrid) -> f32 {
    grid.iter().map(|(_, cell)| cell.groundwater).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::total_humidity;
    use crate::hydro::total_water;

    fn total_moisture(grid: &HexGrid) -> f32 {
        total_water(grid) + total_humidity(grid) + total_groundwater(grid)
    }

    fn make_wet_grid() -> HexGrid {
        let mut grid = HexGrid::from_radius(3);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 3.0;
                cell.groundwater = 1.0;
                cell.permeability = 0.5;
            }
        }
        grid
    }

    #[test]
    fn moisture_conserved_with_groundwater() {
        let current = make_wet_grid();
        let mut next = current.clone();
        let params = GroundwaterParams::default();

        let before = total_moisture(&current);
        step_groundwater(&current, &mut next, &params);
        let after = total_moisture(&next);

        assert!(
            (before - after).abs() < 1e-2,
            "Conservation violated: {before} → {after}"
        );
    }

    #[test]
    fn infiltration_moves_water_underground() {
        let current = make_wet_grid();
        let mut next = current.clone();
        step_groundwater(&current, &mut next, &GroundwaterParams::default());

        let center = next.get(crate::coord::HexCoord::new(0, 0)).unwrap();
        assert!(center.water_level < 3.0);
        assert!(center.groundwater > 1.0);
    }

    #[test]
    fn resurgence_when_over_capacity() {
        let mut grid = HexGrid::from_radius(1);
        // Isolated test: fix max_capacity=5.0 to validate the resurgence
        // mechanism without depending on the default calibration (which
        // may shift).
        let params = GroundwaterParams {
            max_capacity: 5.0,
            ..GroundwaterParams::default()
        };
        // permeability=0.5, max_capacity=5.0 → capacity=2.5
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 0.0;
                cell.groundwater = 10.0; // well above the capacity (2.5)
                cell.permeability = 0.5;
            }
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let center = next.get(crate::coord::HexCoord::new(0, 0)).unwrap();
        assert!(center.water_level > 0.0, "Water should rise to the surface");
        let capacity = 0.5 * 5.0; // permeability * max_capacity
        assert!(
            center.groundwater <= capacity + 0.1,
            "Water table should not exceed local capacity"
        );
    }

    #[test]
    fn frozen_ground_blocks_infiltration() {
        // Below -2°C, the soil is fully frozen: meltwater stays on the
        // surface instead of filling the water table.
        let mut grid = HexGrid::from_radius(0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.water_level = 3.0;
            cell.groundwater = 0.0;
            cell.permeability = 0.5;
            cell.temperature = -3.0;
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &GroundwaterParams::default());

        let center = next.get(c0).unwrap();
        assert!(
            (center.groundwater - 0.0).abs() < 1e-6,
            "No infiltration under frozen soil: gw={}",
            center.groundwater
        );
        assert!(
            (center.water_level - 3.0).abs() < 1e-6,
            "Surface water stays intact: {}",
            center.water_level
        );
    }

    #[test]
    fn dry_grid_no_change() {
        let grid = HexGrid::from_radius(2);
        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &GroundwaterParams::default());

        for (coord, cell) in next.iter() {
            let orig = grid.get(*coord).unwrap();
            assert!((cell.groundwater - orig.groundwater).abs() < 1e-6);
        }
    }

    #[test]
    fn conservation_after_many_steps() {
        let mut current = make_wet_grid();
        let initial = total_moisture(&current);
        let params = GroundwaterParams::default();

        for _ in 0..100 {
            let mut next = current.clone();
            step_groundwater(&current, &mut next, &params);
            current = next;
        }

        let final_val = total_moisture(&current);
        assert!(
            (initial - final_val).abs() < 1e-1,
            "Conservation after 100 steps: {initial} → {final_val}"
        );
    }

    #[test]
    fn piezometric_gradient_flows_uphill_to_downhill() {
        // Mountain with moderate gw and plain with higher gw: the
        // piezometric level (elev + gw) stays higher on the mountain, so
        // water must flow toward the plain even if the plain is "richer"
        // in groundwater. This is the mechanism that feeds natural
        // springs.
        let mut grid = HexGrid::from_radius(1);
        let mountain = crate::coord::HexCoord::new(0, 0);
        let plain_neighbors: Vec<_> = grid.neighbors(mountain).iter().map(|(c, _)| *c).collect();

        if let Some(cell) = grid.get_mut(mountain) {
            cell.elevation = 500.0;
            cell.groundwater = 1.0; // piezo = 501
            cell.water_level = 0.0;
            cell.permeability = 1.0;
        }
        for &coord in &plain_neighbors {
            if let Some(cell) = grid.get_mut(coord) {
                cell.elevation = 0.0;
                cell.groundwater = 3.0; // piezo = 3 (much lower than 501)
                cell.water_level = 0.0;
                cell.permeability = 1.0;
            }
        }

        let before_mountain = grid.get(mountain).unwrap().groundwater;
        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &GroundwaterParams::default());
        let after_mountain = next.get(mountain).unwrap().groundwater;

        assert!(
            after_mountain < before_mountain,
            "mountain water table (high piezo) should flow to the plain \
             (low piezo): before={before_mountain} after={after_mountain}"
        );
    }

    #[test]
    fn saturated_cell_resurges_excess_to_surface() {
        // If the water table exceeds the local capacity (perm ×
        // max_capacity), the excess must rise into water_level: this is
        // the "spring" mechanism.
        // Isolated test: max_capacity=5.0 to validate the mechanism
        // without depending on the default calibration.
        let params = GroundwaterParams {
            max_capacity: 5.0,
            ..GroundwaterParams::default()
        };
        let mut grid = HexGrid::from_radius(0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.permeability = 0.5; // capacity = 0.5 * 5 = 2.5
            cell.groundwater = 5.0; // well above 2.5
            cell.water_level = 0.0;
            cell.temperature = 10.0;
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let cell = next.get(c0).unwrap();
        assert!(
            cell.groundwater <= 2.5 + 1e-4,
            "gw should be brought back to capacity: {}",
            cell.groundwater
        );
        assert!(
            cell.water_level >= 2.4,
            "the excess (~2.5 units) should appear at the surface: {}",
            cell.water_level
        );
    }

    // --- Baseflow e2e-unit micro-tests (#107, Maillet recession) ---

    /// Baseflow returns a fraction of the water table to the surface at
    /// each slice, WITHOUT waiting for overflow: this is the mechanism
    /// that was missing (measured #107: the water table settles below
    /// its capacity, resurgence at `gw > capacity` never triggers, so no
    /// water comes back up). Radius 0 OK: cell-local phenomenon, no
    /// transport.
    #[test]
    fn baseflow_seeps_groundwater_to_surface() {
        let params = GroundwaterParams {
            baseflow_coef: 0.1,
            // Large capacity: isolates baseflow, not resurgence.
            max_capacity: 1000.0,
            infiltration_rate: 0.0, // no surface water to infiltrate back
            diffusion_rate: 0.0,
        };
        let mut grid = HexGrid::from_radius(0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.groundwater = 100.0;
            cell.water_level = 0.0;
            cell.permeability = 1.0;
            cell.temperature = 10.0;
        }

        let mut next = grid.clone();
        step_groundwater(&grid, &mut next, &params);

        let cell = next.get(c0).unwrap();
        // 10% of 100 mm rises to the surface, conservative.
        assert!(
            (cell.water_level - 10.0).abs() < 1e-3,
            "10% of the water table should resurge: water_level={}",
            cell.water_level
        );
        assert!(
            (cell.groundwater - 90.0).abs() < 1e-3,
            "the water table should decrease by as much: gw={}",
            cell.groundwater
        );
        assert!(
            (cell.water_level + cell.groundwater - 100.0).abs() < 1e-4,
            "strict baseflow conservation"
        );
    }

    /// Maillet recession: without recharge, the water table decreases
    /// monotonically and strictly (exponential recession `Q₀·e^(−αt)`),
    /// the surface rises by the same amount. Pins the property that
    /// makes baseflow "sustained between two rains".
    #[test]
    fn baseflow_recedes_monotonically_without_recharge() {
        let params = GroundwaterParams {
            baseflow_coef: 0.05,
            max_capacity: 10_000.0,
            infiltration_rate: 0.0,
            diffusion_rate: 0.0,
        };
        let mut current = HexGrid::from_radius(0);
        let c0 = crate::coord::HexCoord::new(0, 0);
        if let Some(cell) = current.get_mut(c0) {
            cell.groundwater = 200.0;
            cell.water_level = 0.0;
            cell.permeability = 1.0;
            cell.temperature = 10.0;
        }

        let mut prev_gw = 200.0_f32;
        for _ in 0..20 {
            let mut next = current.clone();
            step_groundwater(&current, &mut next, &params);
            current = next;
            let gw = current.get(c0).unwrap().groundwater;
            assert!(gw < prev_gw, "strict recession: {gw} !< {prev_gw}");
            prev_gw = gw;
        }
        // After 20 steps at 5%/step: 200 × 0.95²⁰ ≈ 71.7 mm remaining.
        let gw = current.get(c0).unwrap().groundwater;
        let expected = 200.0 * 0.95_f32.powi(20);
        assert!(
            (gw - expected).abs() < 0.5,
            "Maillet recession: gw={gw}, expected≈{expected}"
        );
    }

    /// Baseflow OFF by default: the mechanism is a calibratable
    /// ingredient (#107), not yet activated (no proof of gain on the
    /// sustainability metric). Pins this choice: a future default must
    /// be an explicit, proven decision, not a silent drift.
    #[test]
    fn baseflow_is_off_by_default() {
        assert!(
            GroundwaterParams::default().baseflow_coef == 0.0,
            "baseflow_coef should stay 0 by default until proven on a metric"
        );
    }
}
