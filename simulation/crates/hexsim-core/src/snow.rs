use serde::{Deserialize, Serialize};

use crate::climate::DayRecord;
use crate::grid::HexGrid;
use crate::temperature::{
    ATMO_IR_BACK_CLEAR, ATMO_IR_BACK_CLOUDY_BOOST, SECONDS_PER_HOUR, STEFAN_BOLTZMANN,
    cloud_cover_fraction,
};

/// Latent heat of fusion of ice (J/kg). Reference: Bonan G. (2008),
/// *Ecological Climatology* 2nd ed., appendix A.4 (0 °C, standard pressure).
pub const LATENT_HEAT_FUSION_J_PER_KG: f32 = 334_000.0;

/// Surface temperature of a melting snowpack (K). A melting pack is
/// isothermal at 0 °C: all the net energy goes into the phase change, not
/// into warming the snow. Reference: Bonan (2008), ch. 5 snowpack.
pub const SNOW_MELT_SURFACE_KELVIN: f32 = 273.15;

/// Air density at sea level (kg/m³), 15 °C, 1013 hPa.
pub const AIR_DENSITY_KG_PER_M3: f32 = 1.2;

/// Specific heat capacity of air at constant pressure (J/(kg·K)).
pub const AIR_SPECIFIC_HEAT_J_PER_KG_K: f32 = 1005.0;

/// Specific heat capacity of liquid water (J/(kg·K)). Same value as
/// `temperature::C_WATER_PER_METER` before multiplying by `ρ_water`.
pub const WATER_SPECIFIC_HEAT_J_PER_KG_K: f32 = 4186.0;

/// Thermal conductivity of ice at 0 °C (W/(m·K)). Reference: Bonan
/// (2008), appendix A.4. Stefan freezing bound: the latent heat released
/// at the water/ice interface must be conducted THROUGH the ice already
/// formed before being released to the atmosphere.
pub const ICE_THERMAL_CONDUCTIVITY_W_PER_M_K: f32 = 2.2;

/// Density of ice (kg/m³). Converts the frozen stock (mm of water
/// equivalent, = kg/m²) into ice thickness for the Stefan bound.
pub const ICE_DENSITY_KG_PER_M3: f32 = 917.0;

/// IR emissivity of water (Bonan 2008, table 13.1: 0.96).
pub const WATER_EMISSIVITY: f32 = 0.96;

/// Albedo of open water, high sun (Bonan 2008, table 13.1: 0.05-0.10).
/// A material constant like `Lf`, not a calibration knob.
pub const WATER_ALBEDO: f32 = 0.08;

/// Conversion of `WindVec` magnitude → m/s (convention #33, cf `wind.rs`).
const WINDVEC_TO_MS: f32 = 10.0;

/// External forcings for the snow step: the same per-cell fields consumed
/// by the temperature balance (#102), passed in rather than recomputed
/// (anti-pattern #2: a single source of truth for illumination).
pub struct SnowForcing<'a> {
    /// Clear-sky beam `S₀·τ·(1−α_ground)` (W/m²), `SolarBeam::beam` of the
    /// current tick, the same value as `step_temperature`.
    pub beam_w_m2: f32,
    /// Bare-ground albedo already folded into `beam_w_m2`. Used to convert
    /// the beam back into *incident* flux before applying the snow albedo.
    pub ground_albedo: f32,
    /// Per-cell illumination factor (aspect × relief occlusion × cloud
    /// shadow), the `scratch_flux_factor` computed by `compute_illumination`.
    pub flux_factor: &'a [f32],
    /// Wind field magnitudes (`WindVec` unit, ×10 = m/s), subsampling
    /// cadence #89.
    pub wind_mag: &'a [f32],
    /// Precipitation from the PREVIOUS tick (warm rain on snow). A
    /// one-tick lag is accepted: the current tick's accumulation doesn't
    /// exist yet when snow steps; negligible at hourly scale, documented
    /// here.
    pub rain_last_tick: &'a [DayRecord],
}

impl SnowForcing<'static> {
    /// "Calm night" forcing: no sun, no wind, no rain.
    /// For tests that isolate the non-solar terms of the balance.
    #[must_use]
    pub fn night_calm() -> Self {
        Self {
            beam_w_m2: 0.0,
            ground_albedo: 0.3,
            flux_factor: &[],
            wind_mag: &[],
            rain_last_tick: &[],
        }
    }
}

/// Snow layer parameters, all measured physical quantities (albedos,
/// emissivity, exchange coefficient), no more calibrated rates since the
/// SI refactor #60 (Phases 1-4).
///
/// **Contract #56**: any change to one of these parameters (or to the
/// freeze/melt model) requires a BEFORE/AFTER re-run of
/// `tests/scale_glacier_drift.rs` (5 seeds × 6 years) and a drift diff in
/// the JOURNAL, never eyeballed tuning on a single seed (anti-pattern #5).
#[derive(Clone, Serialize, Deserialize)]
pub struct SnowParams {
    /// Albedo of DRY snow (cold pack, T ≤ 0). 0.75 = compacted winter
    /// snowpack, seasonal average (fresh 0.8-0.9, compacted 0.7-0.75;
    /// USACE 1998, EM 1110-2-1406, table 5-1; Bonan table 13.1).
    /// WARNING, measured ice-albedo instability (ablation 2026-07-10): at
    /// 0.80, snow-covered cells never rise back above 0 °C, the melt
    /// albedo never kicks in, and annual rainfall collapses from
    /// 8,900 → 3,400 mm (tipping point between 0.78 and 0.80). Do not
    /// raise this value without re-measuring
    /// `phys_kk2000_world_stays_humid` and #56.
    pub snow_albedo_dry: f32,
    /// Albedo of MELTING snow (wet pack, T > 0). Liquid water in the
    /// pores absorbs: 0.4-0.7 depending on age, 0.6 = seasonal melt
    /// (USACE table 5-1). It's THIS drop that bounds the ice-albedo
    /// instability: a melting pack absorbs more, so it melts more;
    /// without it, the "eternal fresh snow" albedo (#60 Phases 2-3) let
    /// the stock drift at +26/+41%/year (instrument measurement #56).
    pub snow_albedo_melt: f32,
    /// IR emissivity of snow (near-blackbody at long wavelengths).
    /// Reference: Bonan (2008), table 13.1 (0.97-0.99).
    pub snow_emissivity: f32,
    /// Bulk turbulent sensible heat exchange coefficient `C_H`
    /// (dimensionless, a named physical quantity: aerodynamic transfer
    /// coefficient). 2e-3 = snow surface, neutral stability. Reference:
    /// Bonan (2008), ch. 13, typical values 1-3e-3.
    pub sensible_exchange_coef: f32,
    /// Minimum ventilation speed (m/s) for sensible exchange: free
    /// convection when the synoptic wind is zero. Equivalent to the
    /// constant term `a` of the USACE wind functions `a + b·U`.
    pub free_convection_wind_ms: f32,
    /// Snow masking depth (mm water equivalent): stock at which optical
    /// coverage reaches 50% (`f = S/(S+S_half)`). A thin film doesn't
    /// whiten a 100 ha cell; ground roughness and vegetation show
    /// through. Reference: Bonan (2008), §13.2 (snow masking, order
    /// 10-30 mm SWE for bare/herbaceous ground). Consumed by the
    /// ice-albedo feedback of the radiative balance (#60 Phase 2,
    /// `step_temperature`).
    pub snow_masking_half_mm: f32,
    /// Temperature below which water freezes.
    pub freeze_threshold: f32,
    /// Fraction of melt that percolates DIRECTLY into the water table
    /// (moist soil under the snowpack) instead of running off at the
    /// surface (`water_level`). Without this, on slopes the drainage
    /// (`step_hydro`) evacuates the meltwater before the slow
    /// infiltration (`step_groundwater`, 5%/day) can capture it, leaving
    /// mountain soil dry so the fir/beech climax stage (which requires
    /// `moisture ≥ 1 mm`) never establishes (#87). Hydrological split of
    /// melt→soil vs melt→runoff, bounded by the local water table
    /// capacity. Conservative: `snow → groundwater + water_level`.
    pub melt_recharge_frac: f32,
}

impl Default for SnowParams {
    fn default() -> Self {
        Self {
            // SI energy balance (#60 Phases 1-3): freeze and melt are no
            // longer calibrated rates but Q_net/Lf balances. The
            // parameters below are measured physical quantities, not
            // knobs.
            snow_albedo_dry: 0.75,
            snow_albedo_melt: 0.60,
            snow_emissivity: 0.98,
            sensible_exchange_coef: 2.0e-3,
            free_convection_wind_ms: 0.5,
            snow_masking_half_mm: 15.0,
            freeze_threshold: 0.0,
            // 0.5: half of the melt percolates into the water table, the
            // other half runs off. Keeps the montane stage moist in
            // spring (snow is the big water stock there) without
            // eliminating melt runoff.
            melt_recharge_frac: 0.5,
        }
    }
}

// Freeze and melt of surface water.
//
// Freeze (transition to SI, finalized by Phase 3 of #60): when the
// temperature drops below the threshold, a fraction of water_level
// freezes into snow_level, at a rate proportional to the gap to 0 °C.
//
// Melt (#60 Phase 1): SI energy balance on the surface of the snowpack.
//
//   `Q_net` [W/m²] = Q_solar + Q_IR_down − Q_IR_emitted + Q_sensible + Q_rain
//   melt [mm/tick] = max(0, Q_net) × Δt / Lf        (1 kg/m² of water = 1 mm)
//
// With a pack melting isothermally at 0 °C (T_surface = 273.15 K):
//   Q_solar      = incident × (1 − α_snow), incident = beam × ff / (1 − α_ground)
//   Q_IR_down    = same formula as `step_temperature` (Brutsaert + cloud)
//   Q_IR_emitted = ε_snow · σ · 273.15⁴  ≈ 309 W/m²
//   Q_sensible   = ρ_air · c_p_air · C_H · U · (T_air − 0)
//   Q_rain       = c_water · M_rain · T_air / Δt   (heat advection)
//
// Emergent physical consequences (vs. the old calibrated rate): melt
// follows the sun (afternoon > dawn, sunny slope > shaded slope, clear
// sky > overcast during the day), and a clear night at +5 °C does NOT
// melt (radiative deficit ≈ −29 W/m² > sensible gain in calm air): real
// snowpack behaviour.
// `max(0, Q_net)` is not an arbitrary guard rail: a negative balance
// cools the pack (cold content, not modelled) instead of "unmelting";
// freezing of liquid water is the freeze branch (Phase 3).
//
// v0.3.0 PR2 (#38): the remaining `SnowParams` rates are expressed per
// day and divided by `TICKS_PER_DAY` at use (hourly tick). The energy
// balance itself is naturally hourly (W/m² × 3600 s).
pub fn step_snow(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &SnowParams,
    gw_max_capacity: f32,
    forcing: &SnowForcing,
) {
    let n = current.len();
    let cur_cells = current.cells_slice();

    // Copy current → next via indexed slice (1 memcpy instead of N HashMap
    // lookups). Freeze/melt is purely local: read from `current`, write to
    // the local cell in `next`, so a pure index loop (#62/#65, mirrors
    // `step_groundwater`).
    next.cells_slice_mut().clone_from_slice(cur_cells);

    // Balance terms independent of the cell, precomputed.
    let q_ir_emitted = params.snow_emissivity * STEFAN_BOLTZMANN * SNOW_MELT_SURFACE_KELVIN.powi(4);
    let q_ir_emitted_water = WATER_EMISSIVITY * STEFAN_BOLTZMANN * SNOW_MELT_SURFACE_KELVIN.powi(4);
    let sensible_coef =
        AIR_DENSITY_KG_PER_M3 * AIR_SPECIFIC_HEAT_J_PER_KG_K * params.sensible_exchange_coef;
    let beam_to_incident = forcing.beam_w_m2 / (1.0 - forcing.ground_albedo).max(1e-6);

    let next_cells = next.cells_slice_mut();
    for i in 0..n {
        let cell = &cur_cells[i];
        let nc = &mut next_cells[i];

        if cell.temperature < params.freeze_threshold && cell.water_level > 0.0 {
            // Freeze under the energy balance (#60 Phase 3): freezing only
            // advances if the surface (water/ice interface at 0 °C)
            // RELEASES energy; the latent heat freed must leave toward the
            // atmosphere.
            //   freeze [mm/tick] = min(−Q_net, Q_Stefan)⁺ × Δt / Lf
            // Q_Stefan = k_ice·(0 − T_air)/h_ice: once a layer of ice has
            // formed, conduction through it limits freezing; a deep lake
            // freezes at the surface then SLOWER AND SLOWER (Stefan's law,
            // growth in √t). Replaces the old arbitrary cap
            // `max_freeze_per_tick` with self-limiting physics.
            let t_air = cell.temperature;
            let ff = forcing.flux_factor.get(i).copied().unwrap_or(0.0);
            let q_solar = beam_to_incident * ff * (1.0 - WATER_ALBEDO);
            let cover = cloud_cover_fraction(cell.cloud_water);
            let q_ir_down = ATMO_IR_BACK_CLEAR + cover * ATMO_IR_BACK_CLOUDY_BOOST;
            let wind_ms = forcing.wind_mag.get(i).copied().unwrap_or(0.0) * WINDVEC_TO_MS
                + params.free_convection_wind_ms;
            let q_sensible = sensible_coef * wind_ms * t_air; // T_air < 0 → refroidit
            let q_net = q_solar + q_ir_down - q_ir_emitted_water + q_sensible;

            // The local frozen stock proxies the ice layer thickness
            // (mm of water equivalent → m of ice via ρ_ice). Below 1 mm,
            // the film doesn't yet conduct measurable resistance.
            let h_ice_m = (cell.snow_level / 1000.0) * (1000.0 / ICE_DENSITY_KG_PER_M3);
            let q_stefan = if h_ice_m > 1e-3 {
                ICE_THERMAL_CONDUCTIVITY_W_PER_M_K * (params.freeze_threshold - t_air) / h_ice_m
            } else {
                f32::INFINITY
            };

            let evacuation = (-q_net).max(0.0).min(q_stefan);
            let amount =
                (evacuation * SECONDS_PER_HOUR / LATENT_HEAT_FUSION_J_PER_KG).min(cell.water_level);
            // Transfer bounded to what the snow stock can REPRESENT in
            // f32: a freeze amount smaller than the stock's ULP
            // (`snow + amount == snow`) would debit the water without
            // ever crediting the snow, a biased destruction measured at
            // −0.35 over 10 years (7x the historical noise of the strict
            // conservation test) before this guard.
            // We credit first, then debit EXACTLY what landed.
            let landed = (nc.snow_level + amount) - nc.snow_level;
            if landed > 0.0 {
                nc.snow_level += landed;
                nc.water_level = (nc.water_level - landed).max(0.0);
            }
        } else if cell.temperature > params.freeze_threshold && cell.snow_level > 0.0 {
            let t_air = cell.temperature;

            // SI energy balance of melt (cf. module doc), for ANY stock;
            // Phase 4 #60: no more threshold-based glacial regime, a
            // glacier is just a place where the annual balance
            // accumulates. A melting pack is WET, so melt albedo (0.60,
            // USACE), strictly more absorbing than the dry snow of the
            // temperature balance: it's this drop that bounds the
            // ice-albedo instability (measured: without it, drift
            // +26/+41%/year).
            let ff = forcing.flux_factor.get(i).copied().unwrap_or(0.0);
            let q_solar = beam_to_incident * ff * (1.0 - params.snow_albedo_melt);
            let cover = cloud_cover_fraction(cell.cloud_water);
            let q_ir_down = ATMO_IR_BACK_CLEAR + cover * ATMO_IR_BACK_CLOUDY_BOOST;
            let wind_ms = forcing.wind_mag.get(i).copied().unwrap_or(0.0) * WINDVEC_TO_MS
                + params.free_convection_wind_ms;
            let q_sensible = sensible_coef * wind_ms * t_air;
            let rain_mm = forcing.rain_last_tick.get(i).map_or(0.0, |r| r.rain);
            let q_rain = WATER_SPECIFIC_HEAT_J_PER_KG_K * rain_mm * t_air / SECONDS_PER_HOUR;

            let q_net = q_solar + q_ir_down - q_ir_emitted + q_sensible + q_rain;
            let amount = (q_net.max(0.0) * SECONDS_PER_HOUR / LATENT_HEAT_FUSION_J_PER_KG)
                .min(cell.snow_level);
            // Exact transfer (mirrors the guard in the freeze branch): we
            // distribute what ACTUALLY left the snow stock; on a glacier
            // several metres thick the f32 ULP is coarse (~0.5 mm at
            // 4 m), debiting `amount` but crediting the computed `amount`
            // created a measurable bias on the strict conservation test.
            let new_snow = nc.snow_level - amount;
            let departed = nc.snow_level - new_snow;
            nc.snow_level = new_snow;
            // Melt split: a fraction percolates directly into the water
            // table (soil under the snowpack), the rest runs off at the
            // surface. Bounded by the local water table capacity
            // (perm × max_capacity); the surplus stays on the surface.
            // Melt happens at `temperature > 0` so the soil isn't frozen:
            // percolation is legitimate. Conservative.
            // (The old "glacier" exclusion followed the binary threshold
            // removed in Phase 4; the capacity bound is enough: mountain
            // water tables, being small, saturate fast and the rest runs
            // off.)
            let gw_capacity = cell.permeability * gw_max_capacity;
            let gw_headroom = (gw_capacity - cell.groundwater).max(0.0);
            let to_gw = (params.melt_recharge_frac * departed).min(gw_headroom);
            nc.groundwater += to_gw;
            nc.water_level += departed - to_gw;
        }
    }
}

#[must_use]
pub fn total_snow(grid: &HexGrid) -> f32 {
    grid.iter().map(|(_, cell)| cell.snow_level).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::hydro::total_water;

    fn total_water_and_snow(grid: &HexGrid) -> f32 {
        total_water(grid) + total_snow(grid)
    }

    /// "Sunny noon" forcing: full sun (beam equivalent to summer noon,
    /// clear sky), full illumination on all cells, calm air, no rain.
    /// `ff` must outlive the forcing.
    fn sunny_noon(ff: &[f32]) -> SnowForcing<'_> {
        SnowForcing {
            beam_w_m2: 800.0, // ≈ S₀ × τ(0.75) × (1−α_ground 0.3) × sin_elev ~0.93
            ground_albedo: 0.3,
            flux_factor: ff,
            wind_mag: &[],
            rain_last_tick: &[],
        }
    }

    #[test]
    fn freezing_moves_water_to_snow() {
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 5.0;
                cell.temperature = -5.0;
            }
        }

        let mut next = grid.clone();
        step_snow(
            &grid,
            &mut next,
            &SnowParams::default(),
            100.0,
            &SnowForcing::night_calm(),
        );

        let center = next.get(HexCoord::new(0, 0)).unwrap();
        assert!(center.water_level < 5.0, "Water should decrease");
        assert!(center.snow_level > 0.0, "Snow should appear");
    }

    #[test]
    fn melting_moves_snow_to_water() {
        // #60 Phase 1: melt requires ENERGY, not just T > 0; this test
        // gives it full midday sun.
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.snow_level = 5.0;
                cell.water_level = 0.0;
                cell.temperature = 5.0;
            }
        }

        let ff = vec![1.0; grid.len()];
        let mut next = grid.clone();
        step_snow(
            &grid,
            &mut next,
            &SnowParams::default(),
            100.0,
            &sunny_noon(&ff),
        );

        let center = next.get(HexCoord::new(0, 0)).unwrap();
        assert!(center.snow_level < 5.0, "Snow should melt");
        assert!(center.water_level > 0.0, "Water should appear");
    }

    #[test]
    fn no_change_at_threshold() {
        let mut grid = HexGrid::from_radius(1);
        if let Some(cell) = grid.get_mut(HexCoord::new(0, 0)) {
            cell.water_level = 5.0;
            cell.snow_level = 3.0;
            cell.temperature = 0.0; // exactly at the threshold
        }

        let mut next = grid.clone();
        step_snow(
            &grid,
            &mut next,
            &SnowParams::default(),
            100.0,
            &SnowForcing::night_calm(),
        );

        let center = next.get(HexCoord::new(0, 0)).unwrap();
        assert!((center.water_level - 5.0).abs() < 1e-6);
        assert!((center.snow_level - 3.0).abs() < 1e-6);
    }

    #[test]
    fn conservation() {
        let mut grid = HexGrid::from_radius(3);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 5.0;
                cell.snow_level = 2.0;
                cell.temperature = -3.0;
            }
        }

        let before = total_water_and_snow(&grid);
        let mut next = grid.clone();
        step_snow(
            &grid,
            &mut next,
            &SnowParams::default(),
            100.0,
            &SnowForcing::night_calm(),
        );
        let after = total_water_and_snow(&next);

        assert!(
            (before - after).abs() < 1e-3,
            "Conservation violated: {before} → {after}"
        );
    }

    /// Replaces `glacier_melts_slowly` (#60 Phase 4): the threshold-based
    /// glacial regime no longer exists, but the property that motivated
    /// it, "a large stock doesn't vanish all at once", is now
    /// STRUCTURAL: melt is a SURFACE phenomenon (`Q_net`/Lf per m²),
    /// independent of volume. A 20 m glacier melts at the same mm/tick
    /// rate as a 2 m pack; it just takes 10x more years to disappear.
    #[test]
    fn melt_rate_is_energy_limited_independent_of_stock_size() {
        let run = |stock_mm: f32| -> f32 {
            let mut grid = HexGrid::from_radius(0);
            let cell = grid.get_mut(HexCoord::new(0, 0)).unwrap();
            cell.elevation = 1200.0;
            cell.snow_level = stock_mm;
            cell.water_level = 0.0;
            cell.temperature = 8.0;
            let ff = vec![1.0];
            let mut next = grid.clone();
            step_snow(
                &grid,
                &mut next,
                &SnowParams::default(),
                100.0,
                &sunny_noon(&ff),
            );
            stock_mm - next.get(HexCoord::new(0, 0)).unwrap().snow_level
        };
        let melt_pack = run(2_000.0);
        let melt_glacier = run(20_000.0);
        assert!(
            melt_pack > 0.0,
            "full sun at +8°C: melt must occur ({melt_pack})"
        );
        // Identical up to ULP noise (the stocks differ by an order of
        // magnitude, the exact transfer rounds on different grids).
        assert!(
            (melt_glacier - melt_pack).abs() / melt_pack < 1e-3,
            "melt independent of stock: 2 m → {melt_pack}, 20 m → {melt_glacier}"
        );
        // And the order of magnitude stays bounded: a few mm/h, not a purge.
        assert!(
            melt_pack < 10.0,
            "melt per tick remains a surface matter: {melt_pack} mm/h"
        );
    }

    #[test]
    fn conservation_after_many_steps() {
        let mut current = HexGrid::from_radius(3);
        for coord in current.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = current.get_mut(coord) {
                cell.water_level = 5.0;
                cell.snow_level = 2.0;
                cell.temperature = -5.0;
            }
        }

        let initial = total_water_and_snow(&current);
        let params = SnowParams::default();

        for _ in 0..100 {
            let mut next = current.clone();
            step_snow(
                &current,
                &mut next,
                &params,
                100.0,
                &SnowForcing::night_calm(),
            );
            current = next;
        }

        let final_total = total_water_and_snow(&current);
        assert!(
            (initial - final_total).abs() < 1e-1,
            "Conservation after 100 steps: {initial} → {final_total}"
        );
    }

    #[test]
    fn melt_recharges_groundwater_and_conserves() {
        // Melt on permeable soil: part of the melt must feed the water
        // table (not just the surface), and the total water+snow+table
        // stays conserved.
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.snow_level = 10.0;
            cell.water_level = 0.0;
            cell.groundwater = 0.0;
            cell.permeability = 0.6; // water table capacity = 0.6 × 100 = 60 mm
            cell.temperature = 5.0; // > 0 → melt
        }
        let before = {
            let c = grid.get(c0).unwrap();
            c.snow_level + c.water_level + c.groundwater
        };

        let ff = vec![1.0; grid.len()];
        let mut next = grid.clone();
        step_snow(
            &grid,
            &mut next,
            &SnowParams::default(),
            100.0,
            &sunny_noon(&ff),
        );

        let c = next.get(c0).unwrap();
        assert!(c.snow_level < 10.0, "snow must melt");
        assert!(
            c.groundwater > 0.0,
            "part of the melt must recharge the water table"
        );
        assert!(
            c.water_level > 0.0,
            "the other part must run off at the surface"
        );
        let after = c.snow_level + c.water_level + c.groundwater;
        assert!(
            (before - after).abs() < 1e-4,
            "Conservation (snow+water+table): {before} → {after}"
        );
    }

    #[test]
    fn melt_recharge_capped_by_groundwater_capacity() {
        // Water table already full: melt cannot overflow the capacity,
        // all the surplus stays on the surface.
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.snow_level = 10.0;
            cell.water_level = 0.0;
            cell.permeability = 0.5; // capacity = 0.5 × 100 = 50 mm
            cell.groundwater = 50.0; // already at capacity → headroom 0
            cell.temperature = 5.0;
        }
        let ff = vec![1.0; grid.len()];
        let mut next = grid.clone();
        step_snow(
            &grid,
            &mut next,
            &SnowParams::default(),
            100.0,
            &sunny_noon(&ff),
        );

        let c = next.get(c0).unwrap();
        assert!(
            (c.groundwater - 50.0).abs() < 1e-4,
            "water table full: no recharge, gw={}",
            c.groundwater
        );
        assert!(
            c.water_level > 0.0,
            "all melt runs off when the water table is full"
        );
    }

    // --- e2e-unit micro-tests for the melt energy balance (#60 Phase 1) ---

    /// 1-cell grid, snow-covered at the given temperature, ready to melt.
    fn snowy_cell(temperature: f32, cloud_water: f32) -> HexGrid {
        let mut grid = HexGrid::from_radius(0);
        let cell = grid.get_mut(HexCoord::new(0, 0)).unwrap();
        cell.snow_level = 50.0;
        cell.water_level = 0.0;
        cell.temperature = temperature;
        cell.cloud_water = cloud_water;
        grid
    }

    fn melted_after(grid: &HexGrid, forcing: &SnowForcing) -> f32 {
        let mut next = grid.clone();
        step_snow(grid, &mut next, &SnowParams::default(), 100.0, forcing);
        50.0 - next.get(HexCoord::new(0, 0)).unwrap().snow_level
    }

    /// THE behavior won by the SI balance: at +5 °C on a CLEAR, calm night,
    /// the snow does not melt. The radiative deficit (σT⁴ emitted exceeds
    /// downward atmospheric IR, ≈ −29 W/m²) outweighs the sensible gain in
    /// near-calm air. This is why snow patches survive mild spring nights.
    /// The old calibrated rate melted here without a second thought.
    #[test]
    fn clear_calm_night_above_zero_does_not_melt() {
        let grid = snowy_cell(5.0, 0.0);
        let melted = melted_after(&grid, &SnowForcing::night_calm());
        assert!(
            melted == 0.0,
            "clear calm night at +5°C: negative budget, no melt expected, melted={melted}"
        );
    }

    /// Cloudy complement: under overcast sky the IR back-radiation fills
    /// the deficit (+60 W/m²) and the same mild night melts. The
    /// clear/overcast contrast is the mechanism, not the temperature.
    #[test]
    fn cloudy_night_above_zero_melts_where_clear_night_does_not() {
        let clear = snowy_cell(5.0, 0.0);
        let cloudy = snowy_cell(5.0, 1.0);
        let f = SnowForcing::night_calm();
        let melt_clear = melted_after(&clear, &f);
        let melt_cloudy = melted_after(&cloudy, &f);
        assert!(melt_clear == 0.0, "clear night: 0 expected, {melt_clear}");
        assert!(
            melt_cloudy > 0.0,
            "cloudy night at +5°C: cloud back-radiation must cause melt"
        );
    }

    /// Melt follows the sun: at equal temperature, a better-lit cell
    /// (sunny slope, noon) melts strictly more than a shaded cell
    /// (shaded slope, dawn). Pins the dependency on `flux_factor` #102.
    #[test]
    fn melt_follows_the_sun_more_flux_more_melt() {
        let grid = snowy_cell(2.0, 0.0);
        let ff_sunny = vec![1.0];
        let ff_shaded = vec![0.2];
        let melt_sunny = melted_after(&grid, &sunny_noon(&ff_sunny));
        let melt_shaded = melted_after(&grid, &sunny_noon(&ff_shaded));
        assert!(
            melt_sunny > melt_shaded && melt_shaded >= 0.0,
            "sunny slope ({melt_sunny}) must melt more than shaded slope ({melt_shaded})"
        );
    }

    /// Warm rain speeds up melt (heat advection, the `Q_rain` term of the
    /// balance): same sky, same temperature, the rained-on cell melts
    /// strictly more.
    #[test]
    fn warm_rain_on_snow_accelerates_melt() {
        let grid = snowy_cell(8.0, 1.0); // overcast (it's raining) → cloudy night
        let dry = SnowForcing::night_calm();
        let rain_records = vec![DayRecord {
            rain: 5.0,
            ..DayRecord::default()
        }];
        let rainy = SnowForcing {
            rain_last_tick: &rain_records,
            ..SnowForcing::night_calm()
        };
        let melt_dry = melted_after(&snowy_cell(8.0, 1.0), &dry);
        let melt_rainy = melted_after(&grid, &rainy);
        assert!(
            melt_rainy > melt_dry,
            "warm rain on snow: melt {melt_rainy} must exceed {melt_dry}"
        );
    }

    /// Analytical anchor: melt is EXACTLY `Q_net` × Δt / Lf. Constructed
    /// case: fully overcast night, calm air, T=+5 °C; every term of the
    /// balance can be computed by hand. A refactor that changes the
    /// formula (factor of 2, wrong Δt, wrong constant) turns red.
    #[test]
    fn melt_magnitude_matches_energy_budget() {
        let params = SnowParams::default();
        let t_air = 5.0_f32;
        let grid = snowy_cell(t_air, 1.0);
        let melted = melted_after(&grid, &SnowForcing::night_calm());

        let q_ir_down = ATMO_IR_BACK_CLEAR + ATMO_IR_BACK_CLOUDY_BOOST; // fully overcast
        let q_ir_up = params.snow_emissivity * STEFAN_BOLTZMANN * SNOW_MELT_SURFACE_KELVIN.powi(4);
        let q_sens = AIR_DENSITY_KG_PER_M3
            * AIR_SPECIFIC_HEAT_J_PER_KG_K
            * params.sensible_exchange_coef
            * params.free_convection_wind_ms
            * t_air;
        let q_net = q_ir_down - q_ir_up + q_sens;
        let expected = q_net.max(0.0) * SECONDS_PER_HOUR / LATENT_HEAT_FUSION_J_PER_KG;

        assert!(
            expected > 0.0,
            "setup: the constructed case must have a positive budget, q_net={q_net}"
        );
        assert!(
            (melted - expected).abs() < 1e-5,
            "melt {melted} != analytical budget {expected} (Q_net={q_net} W/m²)"
        );
    }

    // --- Micro e2e-unit tests for the freeze energy budget (#60 Phase 3) ---

    /// Single-cell grid of open water at the given temperature, with an
    /// optional pre-existing frozen stock (Stefan's "ice layer").
    fn watery_cell(temperature: f32, ice_mm: f32) -> HexGrid {
        let mut grid = HexGrid::from_radius(0);
        let cell = grid.get_mut(HexCoord::new(0, 0)).unwrap();
        cell.water_level = 100.0;
        cell.snow_level = ice_mm;
        cell.temperature = temperature;
        cell.cloud_water = 0.0;
        grid
    }

    fn frozen_after(grid: &HexGrid, forcing: &SnowForcing) -> f32 {
        let mut next = grid.clone();
        step_snow(grid, &mut next, &SnowParams::default(), 100.0, forcing);
        let c0 = next.get(HexCoord::new(0, 0)).unwrap();
        100.0 - c0.water_level
    }

    /// Analytical anchor for freezing: on a clear, calm night at a given T,
    /// freezing equals exactly `(−Q_net)·Δt/Lf`, every term computable by
    /// hand (no ice yet, so Stefan's bound doesn't bite).
    #[test]
    fn freeze_magnitude_matches_energy_budget() {
        let params = SnowParams::default();
        let t_air = -10.0_f32;
        let frozen = frozen_after(&watery_cell(t_air, 0.0), &SnowForcing::night_calm());

        let q_ir_down = ATMO_IR_BACK_CLEAR; // clear sky
        let q_ir_up = WATER_EMISSIVITY * STEFAN_BOLTZMANN * SNOW_MELT_SURFACE_KELVIN.powi(4);
        let q_sens = AIR_DENSITY_KG_PER_M3
            * AIR_SPECIFIC_HEAT_J_PER_KG_K
            * params.sensible_exchange_coef
            * params.free_convection_wind_ms
            * t_air;
        let q_net = q_ir_down - q_ir_up + q_sens;
        let expected = (-q_net).max(0.0) * SECONDS_PER_HOUR / LATENT_HEAT_FUSION_J_PER_KG;

        assert!(
            expected > 0.0,
            "setup: negative budget expected, q_net={q_net}"
        );
        assert!(
            (frozen - expected).abs() < 1e-5,
            "freeze {frozen} != analytical budget {expected} (Q_net={q_net} W/m²)"
        );
    }

    /// Colder night → faster freezing (IR and sensible both grow with the
    /// gap), and strictly so.
    #[test]
    fn colder_night_freezes_more() {
        let f = SnowForcing::night_calm();
        let mild = frozen_after(&watery_cell(-2.0, 0.0), &f);
        let harsh = frozen_after(&watery_cell(-15.0, 0.0), &f);
        assert!(
            harsh > mild && mild > 0.0,
            "freeze at -15°C ({harsh}) must exceed freeze at -2°C ({mild})"
        );
    }

    /// Stefan's law: ice already formed insulates the interface, so a lake
    /// covered by 500 mm of ice freezes strictly slower than one that has
    /// just barely frozen over (5 mm), same night, same temperature. This is
    /// the physics that replaces the old arbitrary cap `max_freeze_per_tick`:
    /// a deep ocean doesn't freeze en masse, it ices over, then slows down.
    #[test]
    fn thick_ice_slows_further_freezing() {
        // Calibration checked by hand: at −10 °C on a clear, calm night,
        // |Q_net| ≈ 34 W/m²; Stefan's bound is k·ΔT/h = 2.2×10/h, i.e.
        // ~40 W/m² at 500 mm (doesn't bite yet) and ~10 W/m² at 2000 mm
        // (bites). The first run of this test at 500 mm came back red;
        // that was the test's calibration, not the model.
        let f = SnowForcing::night_calm();
        let thin = frozen_after(&watery_cell(-10.0, 5.0), &f);
        let thick = frozen_after(&watery_cell(-10.0, 2000.0), &f);
        assert!(
            thick < thin && thick > 0.0,
            "2 m of ice must slow the freeze: thin={thin}, thick={thick}"
        );
    }

    /// Winter sun stops freezing: at -5 °C but full noon sun, the dark
    /// water (albedo 0.08) absorbs enough to make the budget positive →
    /// zero freeze, where the same temperature at night does freeze.
    /// Freezing is an ENERGY phenomenon, not temperature alone.
    #[test]
    fn winter_noon_sun_stops_freezing() {
        let ff = vec![1.0];
        let sunny = SnowForcing {
            beam_w_m2: 400.0, // winter noon
            ground_albedo: 0.3,
            flux_factor: &ff,
            wind_mag: &[],
            rain_last_tick: &[],
        };
        let by_night = frozen_after(&watery_cell(-5.0, 0.0), &SnowForcing::night_calm());
        let by_noon = frozen_after(&watery_cell(-5.0, 0.0), &sunny);
        assert!(by_night > 0.0, "night at -5°C must freeze ({by_night})");
        assert!(
            by_noon == 0.0,
            "full sun at -5°C: positive budget, zero freeze expected ({by_noon})"
        );
    }
}
