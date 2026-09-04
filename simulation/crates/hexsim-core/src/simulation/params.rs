//! Runtime parameter tuning: `update_param`'s stringly-typed
//! `"group.field"` dispatch table (~180 lines, see the module doc of
//! [`super`]) and its two per-group helpers, extracted from
//! `update_param` to keep it readable.

use super::Simulation;

impl Simulation {
    /// Updates a simulation parameter by "group.field" key.
    /// Returns true if the key is recognized.
    pub fn update_param(&mut self, key: &str, value: f32) -> bool {
        if key.starts_with("atmosphere.") {
            return self.set_atmosphere_param(key, value);
        }
        if key.starts_with("erosion.") {
            return self.set_erosion_param(key, value);
        }
        match key {
            // Hydrology
            "hydro.flow_rate" => self.hydro_params.flow_rate = value,
            "hydro.slope_full_mobility" => self.hydro_params.slope_full_mobility = value,
            "hydro.flow_concentration" => self.hydro_params.flow_concentration = value,
            // Lake leveling (#106)
            "lake.enabled" => self.lake_params.enabled = value != 0.0,
            "lake.min_surplus_mm" => self.lake_params.min_surplus_mm = value,
            // Groundwater
            "groundwater.infiltration_rate" => {
                self.groundwater_params.infiltration_rate = value;
            }
            "groundwater.diffusion_rate" => self.groundwater_params.diffusion_rate = value,
            "groundwater.max_capacity" => self.groundwater_params.max_capacity = value,
            "groundwater.baseflow_coef" => self.groundwater_params.baseflow_coef = value,
            "groundwater.field_capacity_frac" => {
                self.groundwater_params.field_capacity_frac = value;
            }
            // Snow
            "snow.snow_albedo_dry" => self.snow_params.snow_albedo_dry = value,
            "snow.snow_albedo_melt" => self.snow_params.snow_albedo_melt = value,
            "snow.snow_emissivity" => self.snow_params.snow_emissivity = value,
            "snow.sensible_exchange_coef" => self.snow_params.sensible_exchange_coef = value,
            "snow.free_convection_wind_ms" => self.snow_params.free_convection_wind_ms = value,
            "snow.snow_masking_half_mm" => self.snow_params.snow_masking_half_mm = value,
            "snow.freeze_threshold" => self.snow_params.freeze_threshold = value,
            "snow.melt_recharge_frac" => self.snow_params.melt_recharge_frac = value,
            // Temperature
            "temperature.base_temp" => self.temperature_params.base_temp = value,
            "temperature.lapse_rate" => self.temperature_params.lapse_rate = value,
            "temperature.water_cooling" => self.temperature_params.water_cooling = value,
            "temperature.thermal_coupling" => self.temperature_params.thermal_coupling = value,
            "temperature.latitude_deg" => self.temperature_params.latitude_deg = value,
            "temperature.cloud_albedo_coef" => {
                self.temperature_params.cloud_albedo_coef = value;
            }
            "temperature.atmospheric_transmittance" => {
                self.temperature_params.atmospheric_transmittance = value;
            }
            "temperature.ground_albedo" => {
                self.temperature_params.ground_albedo = value;
            }
            // Wind
            "wind.noise_direction_amplitude" => {
                self.wind_params.noise_direction_amplitude = value;
            }
            "wind.noise_strength_amplitude" => {
                self.wind_params.noise_strength_amplitude = value;
            }
            "wind.noise_time_scale" => self.wind_params.noise_time_scale = value,
            "wind.thermal_strength" => self.wind_params.thermal_strength = value,
            "wind.terrain_deflection" => self.wind_params.terrain_deflection = value,
            "wind.terrain_speed_factor" => self.wind_params.terrain_speed_factor = value,
            "wind.humidity_advection_rate" => self.wind_params.humidity_advection_rate = value,
            "wind.temperature_advection_rate" => {
                self.wind_params.temperature_advection_rate = value;
            }
            "wind.wind_upper_rotation_deg" => {
                self.wind_params.wind_upper_rotation_deg = value;
            }
            "wind.wind_upper_speed_ratio" => {
                self.wind_params.wind_upper_speed_ratio = value;
            }
            // Synoptic dynamics (Phase 1). `deformation_radius_cells` is NOT
            // hot-reloadable: it fixes H = h₀, frozen at state init
            // (changing it requires a reset). The others are safe at runtime.
            "synoptic.enabled" => self.synoptic_enabled = value != 0.0,
            "synoptic.mean_flow_ms" => self.synoptic_params.mean_flow_ms = value,
            "synoptic.thermal_anomaly_days" => {
                self.synoptic_params.thermal_anomaly_days = value;
            }
            "synoptic.thermal_coupling" => self.synoptic_params.thermal_coupling = value,
            "synoptic.viscosity" => self.synoptic_params.viscosity = value,
            "synoptic.friction_days" => self.synoptic_params.friction_days = value,
            "synoptic.relax_days" => self.synoptic_params.relax_days = value,
            // Vegetation (transition to SI, finalized by #77)
            "vegetation.growth_rate" => self.vegetation_params.growth_rate = value,
            "vegetation.colonization_rate" => self.vegetation_params.colonization_rate = value,
            "vegetation.base_mortality" => self.vegetation_params.base_mortality = value,
            "vegetation.lethal_mortality" => self.vegetation_params.lethal_mortality = value,
            "vegetation.succession_rate" => self.vegetation_params.succession_rate = value,
            "vegetation.k_total" => self.vegetation_params.k_total = value,
            "vegetation.open_water_excess" => self.vegetation_params.open_water_excess = value,
            // Fire (#wildfire). `fire.enabled`: 0 = off, otherwise on.
            "fire.enabled" => self.fire_params.enabled = value != 0.0,
            "fire.ignition_rate" => self.fire_params.ignition_rate = value,
            "fire.spread_rate" => self.fire_params.spread_rate = value,
            "fire.moisture_ref_mm" => self.fire_params.moisture_ref_mm = value,
            "fire.temp_ignite_lo" => self.fire_params.temp_ignite_lo = value,
            "fire.temp_ignite_hi" => self.fire_params.temp_ignite_hi = value,
            "fire.fuel_age_half_years" => self.fire_params.fuel_age_half_years = value,
            "fire.combustion_fraction_per_day" => {
                self.fire_params.combustion_fraction_per_day = value;
            }
            "fire.extinguish_fuel_min" => self.fire_params.extinguish_fuel_min = value,
            "fire.fuel_load_kg_per_m2" => self.fire_params.fuel_load_kg_per_m2 = value,
            "fire.combustion_heat_ground_fraction" => {
                self.fire_params.combustion_heat_ground_fraction = value;
            }
            _ => return false,
        }
        true
    }

    /// Applies an `erosion.*` parameter (#105). Extracted from `update_param`
    /// to keep it readable (like `set_atmosphere_param`).
    fn set_erosion_param(&mut self, key: &str, value: f32) -> bool {
        match key {
            "erosion.enabled" => self.erosion_params.enabled = value != 0.0,
            "erosion.k_incision" => self.erosion_params.k_incision = value,
            "erosion.k_transport" => self.erosion_params.k_transport = value,
            "erosion.m_exponent" => self.erosion_params.m_exponent = value,
            "erosion.n_exponent" => self.erosion_params.n_exponent = value,
            "erosion.tau_days" => self.erosion_params.tau_days = value,
            "erosion.accel_years_per_day" => self.erosion_params.accel_years_per_day = value,
            "erosion.cfl_drop_frac" => self.erosion_params.cfl_drop_frac = value,
            _ => return false,
        }
        true
    }

    fn set_atmosphere_param(&mut self, key: &str, value: f32) -> bool {
        match key {
            "atmosphere.transpiration_coef" => self.atmosphere_params.transpiration_coef = value,
            // Ascent trigger + critical mass (synoptic Phase 3).
            "atmosphere.updraft_ref_ms" => self.atmosphere_params.updraft_ref_ms = value,
            "atmosphere.updraft_floor" => self.atmosphere_params.updraft_floor = value,
            "atmosphere.precip_crit_mm" => self.atmosphere_params.precip_crit_mm = value,
            "atmosphere.condensation_rate" => {
                self.atmosphere_params.condensation_rate = value;
            }
            "atmosphere.cloud_evap_hr_threshold" => {
                self.atmosphere_params.cloud_evap_hr_threshold = value;
            }
            "atmosphere.cloud_evap_rate" => {
                self.atmosphere_params.cloud_evap_rate = value;
            }
            "atmosphere.kk2000_droplet_count" => {
                self.atmosphere_params.kk2000_droplet_count = value;
            }
            "atmosphere.cloud_diffusion_rate" => {
                self.atmosphere_params.cloud_diffusion_rate = value;
            }
            "atmosphere.precip_neighbor_share" => {
                self.atmosphere_params.precip_neighbor_share = value;
            }
            "atmosphere.max_precip_per_tick" => {
                self.atmosphere_params.max_precip_per_tick = value;
            }
            "atmosphere.fog_condensation_threshold" => {
                self.atmosphere_params.fog_condensation_threshold = value;
            }
            "atmosphere.fog_condensation_rate" => {
                self.atmosphere_params.fog_condensation_rate = value;
            }
            "atmosphere.sublimation_rate" => self.atmosphere_params.sublimation_rate = value,
            "atmosphere.uplift_rate" => self.atmosphere_params.uplift_rate = value,
            "atmosphere.uplift_thermal_coef" => {
                self.atmosphere_params.uplift_thermal_coef = value;
            }
            "atmosphere.upper_layer_altitude_m" => {
                self.atmosphere_params.upper_layer_altitude_m = value;
            }
            "atmosphere.global_precip_gate" => {
                self.atmosphere_params.global_precip_gate = value;
            }
            "atmosphere.initial_humidity_floor" => {
                self.atmosphere_params.initial_humidity_floor = value;
            }
            "atmosphere.orographic_lift_coef" => {
                self.atmosphere_params.orographic_lift_coef = value;
            }
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::AtmosphereParams;
    use crate::grid::HexGrid;
    use crate::groundwater::GroundwaterParams;
    use crate::hydro::HydroParams;
    use crate::snow::SnowParams;
    use crate::temperature::TemperatureParams;
    use crate::wind::WindParams;

    fn default_sim(radius: i32) -> Simulation {
        let grid = HexGrid::from_radius(radius);
        Simulation::new(
            grid,
            HydroParams::default(),
            AtmosphereParams::default(),
            GroundwaterParams::default(),
            SnowParams::default(),
            TemperatureParams::default(),
            WindParams::default(),
        )
    }

    #[test]
    fn update_param_sets_each_group() {
        // 1 representative key per group: the setter must be reflected
        // by the corresponding getter (round-trip).
        let mut sim = default_sim(1);

        assert!(sim.update_param("atmosphere.uplift_rate", 0.123));
        assert!((sim.atmosphere_params().uplift_rate - 0.123).abs() < 1e-6);

        assert!(sim.update_param("hydro.flow_rate", 0.456));
        assert!((sim.hydro_params().flow_rate - 0.456).abs() < 1e-6);

        assert!(sim.update_param("groundwater.max_capacity", 7.5));
        assert!((sim.groundwater_params().max_capacity - 7.5).abs() < 1e-6);

        assert!(sim.update_param("snow.snow_albedo_dry", 0.7));
        assert!((sim.snow_params().snow_albedo_dry - 0.7).abs() < 1e-6);

        assert!(sim.update_param("temperature.base_temp", 15.0));
        assert!((sim.temperature_params().base_temp - 15.0).abs() < 1e-6);

        assert!(sim.update_param("wind.thermal_strength", 0.8));
        assert!((sim.wind_params().thermal_strength - 0.8).abs() < 1e-6);

        assert!(sim.update_param("vegetation.growth_rate", 0.33));
        assert!((sim.vegetation_params().growth_rate - 0.33).abs() < 1e-6);

        assert!(sim.update_param("erosion.accel_years_per_day", 50.0));
        assert!((sim.erosion_params().accel_years_per_day - 50.0).abs() < 1e-6);
        assert!(sim.update_param("erosion.enabled", 0.0));
        assert!(!sim.erosion_params().enabled);
    }

    #[test]
    fn update_param_unknown_key_returns_false() {
        let mut sim = default_sim(1);
        assert!(!sim.update_param("not.a.key", 0.0));
        assert!(!sim.update_param("atmosphere.unknown", 0.0));
        assert!(!sim.update_param("", 0.0));
    }
}
