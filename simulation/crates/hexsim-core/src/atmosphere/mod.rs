use crate::cell::CellProperties;
use crate::climate::DayRecord;
use crate::grid::HexGrid;
use crate::par;
use crate::phase_timing::{AtmoStepTimings, elapsed_s, mark};
use crate::temperature::{TemperatureParams, solar_declination_rad, solar_elevation_at_hour};
use crate::wind::{WindField, WindParams, compute_upper_wind_field_into};

mod advection;
mod coarse;
mod condensation;
mod fog;
mod params;
mod precipitation;
mod regime;
mod scaling;
mod scratch;
mod updraft;
mod uplift;

#[cfg(test)]
pub(crate) mod test_support;

use advection::{HumidityLayer, advect_humidity_layer_into, fill_temp_deltas};
use condensation::{cloud_dynamics_for_cell, step_cloud_diffusion};
use fog::surface_condensation_for_cell;
use precipitation::{PrecipOutflow, step_precipitation_into};
use regime::step_weather_regime;
use scaling::{
    oro_boosted_params, oro_runs_this_hour, oro_subsample, precip_boosted_params,
    precip_runs_this_hour, precip_subsample, scale_atmosphere_for_hourly_tick,
    scale_wind_for_hourly_tick, temp_advection_boosted_wind_params, temp_advection_runs_this_hour,
    temp_advection_subsample, transport_boosted_params, transport_subsample,
};
use updraft::fill_updraft_into;
use uplift::{step_orographic_convection, step_uplift};

// Symbols imported elsewhere as `hexsim_core::atmosphere::X` (stable public
// surface), re-exported here from their implementation sub-module.
pub use advection::advect_cloud_water_into;
pub use condensation::{
    UPPER_AIR_SMOOTHING_TAU_S, saturation_surface, saturation_upper, saturation_upper_pw,
    smooth_upper_air_mean_t, surface_means, upper_air_temperature,
};
pub use params::AtmosphereParams;
pub use precipitation::cloud_water_to_qc;
pub use regime::WeatherRegime;
pub use scratch::AtmoScratch;
pub use uplift::{EvapCell, EvapStats, VaporSources, step_evaporation};
// Coarse upper layer: the moist layer's own ~1 km torus. Only
// the timings struct is public, because `PhaseTimings::accumulate_moist`
// takes it; the rest is the orchestrator's business — `Simulation` calls
// `step_moist_precip` right after `step_atmosphere_into` whenever
// `AtmoForcing::moist_coarse` is set (see `coarse`'s module doc).
pub use coarse::{MoistCoarseMode, MoistCoarseTimings};
// Crate-internal: the coarse mirror and its step (`Simulation`), the radius
// formula (construction and checkpoint paths), and the compiled-in default
// (`ablation::Ablation::defaults`, so the constant is not duplicated).
pub(crate) use coarse::{
    MOIST_COARSE_DEFAULT, MoistCoarseForcing, MoistCoarseState, moist_coarse_radius,
    step_moist_precip,
};
pub(crate) use scaling::{
    ORO_SUBSAMPLE_HOURS, PRECIP_SUBSAMPLE_HOURS, TEMP_ADVECTION_SUBSAMPLE_HOURS,
    TRANSPORT_SUBSAMPLE_HOURS,
};

/// Water vapor specific constant (J/kg/K). Used to convert a saturation
/// vapor pressure into a density via the ideal gas law:
/// `rho_vap = e_s / (R_vap × T_K)`. Standard value.
pub(crate) const R_VAP: f32 = 461.5;

/// Map of precipitation events produced by one atmospheric tick, indexed by
/// `HexGrid::cell_index` (size = `grid.len()`). Cells with no precipitation:
/// `DayRecord::default()` (rain = 0, snow = 0).
pub type PrecipitationMap = Vec<DayRecord>;

/// Height of the near-ground boundary layer for radiative fog (m).
/// 50 m = typical order of magnitude for a stable night above a humid valley
/// (Stull 1988, *An Introduction to Boundary Layer Meteorology*,
/// Sect 12.3 on nocturnal inversions).
pub const SURFACE_LAYER_M: f32 = 50.0;

/// Full water table reference for normalizing transpiration water stress
/// (cf. `transpiration_coef`, `step_evaporation`). Re-export of
/// `groundwater::DEFAULT_MAX_CAPACITY_MM` (single source of truth).
pub use crate::groundwater::DEFAULT_MAX_CAPACITY_MM as SOIL_GW_REFERENCE_MM;

/// Read-only tick forcing consumed by the atmosphere: wind field and
/// magnitudes memoized at the subsample cadence (#89), params of neighboring
/// phenomena (temperature for lapse rate / LCL, wind for advection), absolute
/// hour for the diurnal cycle. Pattern common to phenomena (cf.
/// `SnowForcing`, `ErosionForcing`): shared inputs travel grouped together
/// and are never mutated (#61).
#[derive(Clone, Copy)]
pub struct AtmoForcing<'a> {
    pub temp_params: &'a TemperatureParams,
    pub wind_params: &'a WindParams,
    pub wind_field: &'a WindField,
    pub wind_mag: &'a [f32],
    /// Light-weighted transpiring cover per cell
    /// (`vegetation::transpiration_cover`), memoized by `Simulation` once a
    /// day since the vegetation only changes in the daily tail; `None`
    /// makes `step_evaporation` compute it per cell (tests, tools).
    pub transpiration_cover: Option<&'a [f32]>,
    /// The synoptic base wind interpolated onto the fine grid
    /// (`Simulation::synoptic_base`), BEFORE the thermal breeze and the
    /// terrain deflection of `compute_wind_field_into`: smooth at the
    /// solver's scale (`L_d`, the coarse synoptic mesh), which is the
    /// wind the ascent trigger derives (#110). On the composite 130 m
    /// wind, `apply_terrain_deflection` and `propagate_upstream`
    /// manufacture divergences of ~1e-2 s⁻¹ (slope noise, not weather)
    /// that, times the moist column, gave the ±225 m/s `w` of 2026-07-15;
    /// and the barrier lift `v·∇z` on an already deflected wind
    /// double-counts the relief with a downslope bias. `None` = no
    /// synoptic base (scripted wind, micro-tests): the estimator falls
    /// back to the composite wind, as before.
    pub synoptic_wind: Option<&'a WindField>,
    pub hour_tick: u64,
    /// Diurnally smoothed map-mean surface temperature (°C) that anchors
    /// the upper layer (`upper_air_temperature`): persistent state owned
    /// by `Simulation` (`Simulation::upper_air_mean_t`, EMA with
    /// τ = `UPPER_AIR_SMOOTHING_TAU_S`, checkpointed), stepped before the
    /// atmosphere and read-only here. Not the instantaneous mean of
    /// `current`: the free atmosphere keeps the seasons and the lapse
    /// with elevation, not the ~8 K day/night swing of the surface.
    pub upper_air_mean_t: f32,
    /// Map-mean elevation (m) of `current`, `surface_means(current).1`:
    /// the ground the upper layer's standard lapse starts from
    /// (`upper_air_temperature`, via `AtmoScratch::fill_upper_air`).
    /// Reduced once per tick by the caller together with the
    /// instantaneous mean temperature that steps `upper_air_mean_t`
    /// (`PhaseTimings::atmo_means`), carried here so the atmosphere
    /// doesn't stream the grid a second time for the same number.
    pub mean_elevation: f32,
    /// Coarse upper layer (`atmosphere::coarse`,
    /// `MoistCoarseMode::precipitates_coarse`): `true` means the caller
    /// will run one of the two coarse moist steps right after this
    /// function returns, so the pass that lives on the ~1 km torus — the
    /// KK2000 autoconversion and the sheet it drops — is **skipped
    /// here**. The two calls are one pipeline: setting this without
    /// making the second call is a broken tick, not a variant.
    ///
    /// Everything else in the moist pipeline stays here on every mode.
    /// The imposed weather regime and the vapour ↔ droplet transition
    /// were coarse at step 2 and came back at step 2b, then the whole
    /// stock came back at step 2c: they are per-column rules and the fine
    /// grid is where their sub-grid distribution lives. See `coarse`'s
    /// module doc for what that measured.
    ///
    /// An argument rather than a read of `Ablation::effective()`: the
    /// physics never reads a process-global switch, and the standalone
    /// [`step_atmosphere`] wrapper (used by micro-tests and by
    /// `just metrics`, neither of which owns a persistent coarse stock)
    /// keeps running the historical fine pipeline by passing `false`.
    /// `Simulation` is the only caller that passes `true`, from
    /// `Ablation::effective().moist_coarse` (`HEXSIM_MOIST_COARSE`).
    pub moist_coarse: bool,
    /// Record each fine column's signed vapour ↔ droplet transfer into
    /// [`AtmoScratch::cloud_transfer`] (see that field for the reader and
    /// for the measured cost of the store). Off in production on both
    /// modes; `Simulation` turns it on only for the by-altitude cloud
    /// budget instrument (`Simulation::set_cloud_transfer_probe`).
    pub track_cloud_transfer: bool,
}

/// Persistent state the atmosphere carries from one tick to the next, the
/// `&mut` argument of convention #61.
///
/// Grouped rather than passed field by field because `step_atmosphere_into`
/// is at the 7-argument clippy ceiling (see [`AtmoStepTimings`]'s own doc
/// for the same constraint): a second piece of persistent state would have
/// been an eighth argument, and the convention says group, never
/// `#[allow(clippy::too_many_arguments)]`. This is NOT a scratch buffer —
/// everything here has meaning between two ticks, which is exactly why it
/// is checkpointed by the caller.
#[derive(Clone, Copy, Debug, Default)]
pub struct AtmoState {
    /// Hysteresis of the global precipitation gate, see
    /// [`AtmosphereParams::global_precip_gate`].
    pub precip_gate_open: bool,
    /// Imposed weather regime (#63): the phase of the synoptic chain and
    /// the vapour held outside the box. Inert while
    /// [`AtmosphereParams::regime_enabled`] is 0 — off only on a checkpoint
    /// or params file predating the mechanism (its `#[serde(default)]`);
    /// a freshly generated world defaults to `1.0` (enabled) since
    /// 2026-09-06 (#63/#146).
    pub regime: WeatherRegime,
}

/// Two-layer atmospheric cycle, strictly closed terrarium.
///
/// Pipeline for one tick:
/// 1. Copy `current` → `next`
/// 2. Compute upper wind (Ekman drift)
/// 3. `step_evaporation`: water + snow → `humidity_surface`
/// 4. `advect_humidity_layer(Surface)`: fresh vapor dispersed BEFORE rising,
///    breaks the captive lake → rain → lake cycle
/// 5. `step_uplift`: `humidity_surface` → `humidity_upper` (thermal + diurnal
///    drive)
/// 6. `advect_humidity_layer(Upper)` advected by the upper wind
/// 7. `step_weather_regime`: imposed synoptic regime (#63), inert unless
///    enabled — the last pass to touch `humidity_upper` before the phase
///    transition reads it, which is the whole point (see its call site)
/// 8. `advect_temperature_by_wind`, then the vapour ↔ droplet transition
/// 9. `step_precipitation`: consumes `cloud_water`, with a global
///    hysteresis gate
pub fn step_atmosphere(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    forcing: &AtmoForcing<'_>,
    state: &mut AtmoState,
) -> PrecipitationMap {
    let n = current.len();
    let mut scratch = AtmoScratch::new(n);
    let mut events: PrecipitationMap = vec![DayRecord::default(); n];
    step_atmosphere_into(
        current,
        next,
        params,
        forcing,
        state,
        &mut scratch,
        &mut events,
    );
    events
}

/// Zero-malloc variant: uses the scratch buffers supplied (`AtmoScratch`),
/// their capacity is reused from one tick to the next. `forcing.wind_mag` =
/// wind field magnitudes, precomputed by the caller at the field's cadence
/// (subsample #89) instead of one `sqrt` per cell per hour here.
pub fn step_atmosphere_into(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    forcing: &AtmoForcing<'_>,
    state: &mut AtmoState,
    scratch: &mut AtmoScratch,
    events: &mut PrecipitationMap,
) {
    let AtmoForcing {
        temp_params,
        wind_params,
        wind_field,
        wind_mag,
        transpiration_cover,
        synoptic_wind,
        hour_tick,
        upper_air_mean_t,
        mean_elevation,
        moist_coarse,
        track_cloud_transfer,
    } = *forcing;

    // v0.3.0 PR2: shadow with scaled params for the hourly regime.
    // The caller passes "per day" params (v0.2.x convention); all
    // sub-functions here consume the scaled versions.
    let params_hourly = scale_atmosphere_for_hourly_tick(params);
    let wind_params_hourly = scale_wind_for_hourly_tick(wind_params);
    let params = &params_hourly;
    let wind_params = &wind_params_hourly;

    // The historical `current → next` full-grid copy now lives folded
    // into `step_evaporation`'s sweep below (r250 perf effort, chunk
    // B2): nothing between here and there ever touches `next` (wind and
    // `fill_upper_air` only read `current`, writing scratch buffers), so
    // `step_evaporation` is this phase's first per-cell pass to write
    // `next[i]` — see its doc for the fold.
    //
    // Fresh per-tick sub-phase durations (`AtmoStepTimings` carries no
    // meaning between ticks, cf. its doc): the caller (`Simulation`) reads
    // this back right after the call and accumulates it into the
    // cumulative `PhaseTimings::atmo_*` fields.
    scratch.step_timings = AtmoStepTimings::default();

    compute_upper_wind_field_into(wind_field, wind_params, &mut scratch.wind_upper);

    // Issue #46: positive sin_elev shared across all cells (single
    // latitude for the terrarium). Computed once here, passed to
    // step_uplift to drive the diurnal convective forcing.
    let lat_rad = temp_params.latitude_deg.to_radians();
    let day_of_year = crate::time::day_of_year(hour_tick);
    // Real clock hour (#sub-tick-agnostic, cf. clock_hour_of_day):
    // the diurnal convective drive must see all 24 h even if
    // TICKS_PER_DAY < 24.
    let hour_f = crate::time::clock_hour_of_day(hour_tick);
    let dec_rad = solar_declination_rad(day_of_year);
    let sin_elev_pos = solar_elevation_at_hour(lat_rad, dec_rad, hour_f)
        .sin()
        .max(0.0);

    // Upper-air temperature + Tetens memoization (#97): horizontally
    // homogeneous upper air (`upper_air_temperature`: the diurnally
    // smoothed map-mean surface T carried by the forcing, map-mean
    // elevation of the pre-advection `current` state), shared
    // identically by the orographic convection (LCL bound by upper
    // neighbor), the Surface advection's lift (LCL bound by upward flux)
    // and the vapor ↔ droplet transition.
    let t0 = mark();
    scratch.fill_upper_air(
        current,
        upper_air_mean_t,
        mean_elevation,
        params,
        temp_params,
    );
    scratch.step_timings.upper_air += elapsed_s(t0);

    let t0 = mark();
    step_evaporation(
        current,
        next,
        params,
        wind_mag,
        transpiration_cover,
        &mut scratch.evap_cells,
        &mut scratch.evap,
    );
    scratch.step_timings.evaporation += elapsed_s(t0);
    // Orographic convection BEFORE advection: fresh vapor evaporated near a
    // relief must rise orographically before being swept away by downslope
    // thermal breezes (which flow down reliefs in a closed terrarium).
    // Critical ordering, breaks if inverted.
    //
    // Cadence (ablation switch `HEXSIM_ORO_SUBSAMPLE`): like the transport
    // passes below, the pump can run one hour in `oro_sub` with
    // `orographic_lift_coef` scaled up to compensate. Measured and refuted
    // for `oro_sub > 1` (see `ORO_SUBSAMPLE_HOURS`: the per-pass cap
    // saturates and the daily transport is NOT conserved), so `1` is the
    // shipped default, bit-identical to before the switch existed. When
    // `oro_sub` equals the transport subsample, the pump still fires right
    // before the surface humidity advection on the SAME tick below, the
    // critical ordering above holds at any cadence.
    //
    // Rate law (#156): the pump's exported fraction is the bounded
    // exponential `1 − exp(−coef · Σ Δz⁺)` (see `uplift::oro_pump_rate`).
    // The pre-#156 `clamp(…, 0.0, 0.30)` A/B lever (`HEXSIM_ORO_LEGACY_
    // CLAMP`) was retired 2026-09-07 once the exponential law had been
    // green and merged since 2026-09-05.
    let oro_sub = oro_subsample();
    if oro_runs_this_hour(hour_tick, oro_sub) {
        let params_oro = oro_boosted_params(params, oro_sub);
        let t0 = mark();
        step_orographic_convection(current, next, &params_oro, scratch);
        scratch.step_timings.orographic += elapsed_s(t0);
    }

    // Horizontal transport subsampling: the humidity/cloud advection and
    // cloud diffusion passes only run one hour out of `sub`, with their
    // rates ×sub (daily transport ≈ conserved). The boosted copies are
    // local to the gated passes; `step_uplift` reads the plain hourly
    // `params` above and stays in strict hourly regime. `transport_
    // boosted_params` also boosts `orographic_lift_coef` here, but for a
    // different consumer: the *lift* half of the Surface-layer humidity
    // advection below, unrelated to `step_orographic_convection`'s OWN
    // cadence (`oro_sub`, gated above, independent of `sub`). Grouped
    // into `TransportForcing` (convention #61) so `step_atmosphere_
    // transport` stays under the 7-arg ceiling.
    let sub = transport_subsample();
    let (params_t_owned, wind_params_t_owned) = transport_boosted_params(params, wind_params, sub);
    // Temperature advection cadence (ablation switch `HEXSIM_TEMP_
    // ADVECTION_SUBSAMPLE`): its own switch, independent of `sub` above
    // (see `TEMP_ADVECTION_SUBSAMPLE_HOURS`'s doc). Only the GATHER
    // (`fill_temp_deltas`, below) is gated; the fused apply sweep that
    // follows it stays hourly.
    let tadv_sub = temp_advection_subsample();
    let wind_params_tadv_owned = temp_advection_boosted_wind_params(wind_params, tadv_sub);
    let transport = TransportForcing {
        wind_field,
        wind_params,
        temp_params,
        sin_elev_pos,
        on_transport_tick: hour_tick.is_multiple_of(u64::from(sub)),
        params_t: &params_t_owned,
        wind_params_t: &wind_params_t_owned,
        on_temp_advection_tick: temp_advection_runs_this_hour(hour_tick, tadv_sub),
        wind_params_tadv: &wind_params_tadv_owned,
        track_cloud_transfer,
    };
    step_atmosphere_transport(
        current,
        next,
        params,
        &transport,
        &mut state.regime,
        scratch,
    );

    // Ascent trigger (synoptic Phase 3, ex-design C #69): filled
    // only when active; consumed by precipitation. Derived from the
    // AMBIENT wind (the interpolated synoptic base), never from the fine
    // composite whose terrain deflection manufactured the ±225 m/s `w`
    // with an inverted altitude signature (#110, see `updraft.rs`).
    if params.updraft_ref_ms > 0.0 {
        let t0 = mark();
        fill_updraft_into(
            current,
            synoptic_wind.unwrap_or(wind_field),
            params.upper_layer_altitude_m,
            &mut scratch.convergence,
        );
        scratch.step_timings.other += elapsed_s(t0);
    }

    // `events` is zeroed BEFORE the cadence gate below, so a skipped
    // hour reports zero precipitation rather than the previous pass's
    // map: `Simulation` sums it into the daily accumulator every tick
    // and `step_snow` reads it next tick as "rain of the previous tick".
    // Both paths go through this, the coarse one included.
    events.resize(next.len(), DayRecord::default());
    events.fill(DayRecord::default());

    if !moist_coarse {
        step_fine_precipitation(next, params, hour_tick, state, scratch, events);
    }
}

/// The fine grid's precipitation pass and its cadence gate.
///
/// Cadence (ablation switch `HEXSIM_PRECIP_SUBSAMPLE`): its own switch,
/// independent of the transport and temperature advection subsamples.
/// Unlike them the compensation is not a rate boost — the pass integrates
/// the KK2000 autoconversion ODE over `dt = precip_sub` hours, and only
/// the per-pass cap `max_precip_per_tick` is scaled. See
/// [`PRECIP_SUBSAMPLE_HOURS`] and [`step_precipitation_into`]'s doc for
/// which quantities scale and which do not. `1` is the shipped default,
/// bit-identical to before the switch existed.
///
/// Not called on the coarse mode: precipitation then belongs to the
/// caller, on the ~1 km torus (`coarse::step_moist_precip`) — the same
/// KK2000 drain (`precipitation::precip_amount_mm`), no neighbour share
/// and no fine footprint (the coarse cell IS the footprint), the sheet
/// distributed back onto the fine cells with each one's own rain/snow
/// phase.
fn step_fine_precipitation(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    hour_tick: u64,
    state: &mut AtmoState,
    scratch: &mut AtmoScratch,
    events: &mut PrecipitationMap,
) {
    let precip_sub = precip_subsample();
    if precip_runs_this_hour(hour_tick, precip_sub) {
        let params_precip = precip_boosted_params(params, precip_sub);
        let t0 = mark();
        step_precipitation_into(
            next,
            &params_precip,
            f32::from(precip_sub),
            &mut state.precip_gate_open,
            events,
            scratch,
        );
        scratch.step_timings.precipitation += elapsed_s(t0);
    }
}

/// Read-only inputs of [`step_atmosphere_transport`], grouped per
/// convention #61 (a forcing carries a phenomenon's shared read-only
/// inputs so adding one is a field, not another argument): the wind
/// fields, the transport-boosted param copies used only on a transport
/// tick, the diurnal convective drive `step_uplift` also needs, and the
/// temperature advection gather's own cadence gate and boosted
/// `WindParams` (independent switch from the transport subsample above).
#[derive(Clone, Copy)]
struct TransportForcing<'a> {
    wind_field: &'a WindField,
    wind_params: &'a WindParams,
    temp_params: &'a TemperatureParams,
    sin_elev_pos: f32,
    on_transport_tick: bool,
    params_t: &'a AtmosphereParams,
    wind_params_t: &'a WindParams,
    on_temp_advection_tick: bool,
    wind_params_tadv: &'a WindParams,
    /// See [`AtmoForcing::track_cloud_transfer`]: the fused sweep stores
    /// each column's signed vapour ↔ droplet transfer.
    track_cloud_transfer: bool,
}

/// The transport half of `step_atmosphere_into` (issue split off it to
/// stay under `clippy::too_many_lines`, no behavior change): humidity
/// advection (both layers), vertical uplift, temperature advection, and
/// the cloud vapor ↔ droplet transition through its own advection and
/// diffusion. Same call order as before the split, still measured into
/// `scratch.step_timings` per sub-phase.
fn step_atmosphere_transport(
    current: &HexGrid,
    next: &mut HexGrid,
    params: &AtmosphereParams,
    forcing: &TransportForcing<'_>,
    regime: &mut WeatherRegime,
    scratch: &mut AtmoScratch,
) {
    let TransportForcing {
        wind_field,
        wind_params,
        temp_params,
        sin_elev_pos,
        on_transport_tick,
        params_t,
        wind_params_t,
        on_temp_advection_tick,
        wind_params_tadv,
        track_cloud_transfer,
    } = *forcing;

    if on_transport_tick {
        let t0 = mark();
        advect_humidity_layer_into(
            current,
            next,
            wind_field,
            wind_params_t,
            params_t,
            HumidityLayer::Surface,
            scratch,
        );
        scratch.step_timings.advection_humidity += elapsed_s(t0);
    }
    let t0 = mark();
    step_uplift(next, params, temp_params, sin_elev_pos);
    scratch.step_timings.uplift += elapsed_s(t0);
    if on_transport_tick {
        let t0 = mark();
        advect_humidity_layer_into(
            current,
            next,
            wind_field,
            wind_params_t,
            params_t,
            HumidityLayer::Upper,
            scratch,
        );
        scratch.step_timings.advection_humidity += elapsed_s(t0);
    }
    // Temperature advection's apply, the vapor <-> droplet transition
    // (`step_cloud_dynamics`) and surface condensation (`step_surface_
    // condensation`, issue #45, radiative fog — after the upper cloud
    // dynamics so fog adds its "low" cloud_water to the same stock)
    // fused into one sweep (r250 perf effort, chunk B2): temperature
    // advection's gather is unaffected (still its own barrier, done
    // BEFORE this sweep by `fill_temp_deltas`), but its apply and the
    // two per-cell phases that follow it never read a NEIGHBOR's
    // post-fusion value — cloud dynamics reads only `t_upper[i]`
    // (precomputed, unaffected by temperature) and cell `i`'s own
    // humidity_upper/cloud_water; condensation reads cell `i`'s own,
    // now-advected temperature and humidity_surface — so applying the
    // three in the same per-cell order they ran in before (temperature
    // delta, then cloud dynamics, then condensation) is bit-identical,
    // one sweep instead of three. Timed as a whole under
    // `atmo_advection_temperature` (see that bucket's doc): the
    // `atmo_cloud_dynamics`/`atmo_condensation` buckets read 0 from now on.
    //
    // Cadence (ablation switch `HEXSIM_TEMP_ADVECTION_SUBSAMPLE`): only
    // the GATHER just below is a candidate for subsampling —
    // `fill_temp_deltas` can run one hour in `tadv_sub` with
    // `temperature_advection_rate` scaled up to compensate (same pattern
    // as the orographic pump and the horizontal transport passes). The
    // fused APPLY sweep (`apply_temperature_advection_then_cloud_and_
    // condensation`, called unconditionally below regardless of
    // `advect_temp_active`) stays hourly: `cloud_dynamics_for_cell` and
    // `surface_condensation_for_cell` are hourly physics fused into it,
    // gating the whole sweep would skip them on off-hours too. `3` is
    // the shipped default since the A/B of 2026-09-05 (see
    // `TEMP_ADVECTION_SUBSAMPLE_HOURS`); `1` restores the historical
    // hourly behavior bit for bit.
    // Imposed weather regime (#63): here, and not one pass earlier.
    //
    // The design (JOURNAL 2026-09-03) asks for the pass to sit "before
    // orographic convection / condensation, so the condensation sees a
    // subsaturated upper layer". Putting it literally before the pump
    // does NOT achieve that, and the bench says so: the pump's LCL bound
    // refills `humidity_upper` to *exactly* saturation from the surface
    // layer every hour, so whatever the export removed comes straight
    // back and the map still rains every day of the year (r30, 3 seeds,
    // `fully_rain_free_days_total` 0/0/0 either way, see the report of
    // 2026-09-06). The subsaturation the design wants is the one the
    // condensation reads, so the export belongs after every pass that
    // feeds the upper layer — the pump, `step_uplift`, and the surface
    // advection's own orographic lift — and immediately before the
    // vapour ↔ droplet transition in the fused sweep below.
    //
    // Physically this is the same statement, split the other way round:
    // of the vapour delivered aloft this hour, the large-scale
    // subsidence removes the part above the target *before* it can
    // condense, instead of removing it and letting local convection put
    // it back within the same hour.
    //
    // Reads the memoized `saturation_upper(T_upper)`, the very table
    // `cloud_dynamics_for_cell` uses two lines below, so "RH" means the
    // same number in both places (anti-pattern #2). Costs nothing when
    // disabled.
    //
    // Coarse upper layer: the regime runs HERE on every mode, per fine
    // column, on `hu_i` against that column's own `sat_i` — never on a
    // coarse-torus mean, which a retired mode (step 2, `mean hu` against
    // `mean sat`) tried and lost to Jensen: a column mean never exceeds
    // its target when some of its columns do, so an export evaluated on
    // the mean under-fires exactly where the fine grid was above target.
    // The sky reservoir gets `Σ_i leaving_i` on every mode.
    let t0 = mark();
    step_weather_regime(
        next,
        params,
        &scratch.sat_upper_offset,
        regime,
        &mut scratch.regime_moved,
        &mut scratch.regime_partials,
    );
    scratch.step_timings.other += elapsed_s(t0);

    let t0 = mark();
    let advect_temp_active = wind_params.temperature_advection_rate > 0.0 && on_temp_advection_tick;
    if advect_temp_active {
        fill_temp_deltas(
            current,
            next,
            wind_field,
            wind_params_tadv,
            &mut scratch.snap,
            &mut scratch.temp_deltas,
            &mut scratch.dir_out,
        );
    }
    // Sized once, never re-zeroed: `for_each_chunk_mut2` splits the two
    // slices at the same boundaries and wants them the same length, so the
    // buffer exists on both paths. What the fine path does NOT pay is the
    // per-cell store, gated below.
    if scratch.cloud_transfer.len() != next.len() {
        scratch.cloud_transfer.clear();
        scratch.cloud_transfer.resize(next.len(), 0.0);
    }
    apply_temperature_advection_then_cloud_and_condensation(
        next,
        params,
        &FusedSweep {
            advect_temp_active,
            track_cloud_transfer,
            t_upper: &scratch.t_upper,
        },
        &scratch.temp_deltas,
        &mut scratch.cloud_transfer,
    );
    scratch.step_timings.advection_temperature += elapsed_s(t0);
    // Directional advection of cloud_water by the upper wind: droplets
    // that have formed travel with the flow before precipitation, a
    // necessary condition so rain doesn't systematically fall on the
    // vapor source (cell-lake cycle).
    if on_transport_tick {
        let t0 = mark();
        advect_cloud_water_into(
            current,
            next,
            &scratch.wind_upper,
            params_t,
            &mut scratch.snap,
            &mut scratch.deltas,
            &mut scratch.dir_out,
        );
        // Spatial diffusion of cloud_water: smooths the checkerboard
        // pattern, each cell shares a fraction of its cloud with its
        // neighbors. Shares the same measurement as the advection above.
        step_cloud_diffusion(
            current,
            next,
            params_t,
            &mut scratch.snap,
            &mut scratch.deltas,
        );
        scratch.step_timings.advection_cloud += elapsed_s(t0);
    }
}

/// Fuses three per-cell passes into one sweep (r250 perf effort, chunk
/// B2): temperature advection's apply (`temp_deltas[i]`, already fully
/// computed by [`fill_temp_deltas`]'s gather — a genuine cross-cell
/// barrier this fusion does NOT touch), [`cloud_dynamics_for_cell`]
/// (vapor <-> droplet transition) and [`surface_condensation_for_cell`]
/// (radiative fog). None of the three ever reads a neighbor's
/// post-fusion value: temperature's delta is cell `i`'s own, cloud
/// dynamics reads only the precomputed `t_upper[i]` and cell `i`'s own
/// `humidity_upper`/`cloud_water`, and condensation reads cell `i`'s own
/// temperature (just advected, own-cell only) and `humidity_surface` —
/// so running the three bodies back to back on the same cell, in the
/// same order they ran in as three separate dispatches, changes nothing
/// but the barrier count.
///
/// `advect_temp_active` mirrors the historical single
/// `advect_temperature_by_wind_into`'s early return
/// (`wind_params.temperature_advection_rate <= 0.0`): when temperature
/// advection is off, `temp_deltas` was never refilled this tick, so the
/// delta step is skipped rather than applying stale data — the same
/// behavior the historical separate call had (no full grid pass at all
/// in that case, next's temperature untouched).
///
/// The vapour ↔ droplet transition runs here on **every** moist mode. It
/// reads each fine column's own `hu_i`, `cw_i` and `sat_i`, so the
/// condensed mass is `Σ_i (hu_i − sat_i)⁺ × rate` and not the coarse
/// `N_c × (mean hu − mean sat)⁺ × rate` step 2 produced — see `coarse`'s
/// module doc for the Jensen inequality between the two, and for what
/// step 2c measured the *reverse* branch costing when it is applied to a
/// broadcast view instead of to the column the cloud is in. The
/// radiative fog has always been here and stays fine: it is
/// boundary-layer physics (50 m).
/// `cloud_transfer` (fine-sized) is this sweep's **report**: the signed
/// vapour ↔ droplet transfer of each column this hour, exactly what
/// `cloud_dynamics` returned. `AtmoScratch::cloud_transfer` documents its
/// two readers and why the store is gated rather than unconditional.
fn apply_temperature_advection_then_cloud_and_condensation(
    next: &mut HexGrid,
    params: &AtmosphereParams,
    active: &FusedSweep,
    temp_deltas: &[f32],
    cloud_transfer: &mut [f32],
) {
    let &FusedSweep {
        advect_temp_active,
        track_cloud_transfer,
        t_upper,
    } = active;
    let condensation_rate = params.condensation_rate.min(1.0);
    let cell = |nc: &mut CellProperties, i: usize| -> f32 {
        if advect_temp_active {
            let delta = temp_deltas[i];
            if delta != 0.0 {
                nc.temperature += delta;
            }
        }
        let transfer = cloud_dynamics_for_cell(nc, t_upper[i], params, condensation_rate);
        surface_condensation_for_cell(nc, params);
        transfer
    };
    // Two dispatches, ONE body (`cell` above): the sweep only needs a
    // second parallel slice when something is going to read the
    // transfers back, and pairing the two slices costs the walk itself,
    // not just the store — step 2b shipped the paired form
    // unconditionally and left the fine path +1.3 % of tick for a buffer
    // nothing on it reads (JOURNAL 2026-09-06). Splitting at the
    // dispatch rather than inside the loop keeps the per-cell physics in
    // a single place, which is the rule that matters (one system, not
    // case by case).
    if track_cloud_transfer {
        par::for_each_chunk_mut2(
            next.cells_slice_mut(),
            cloud_transfer,
            |start, chunk, transfer_chunk| {
                for (local, nc) in chunk.iter_mut().enumerate() {
                    transfer_chunk[local] = cell(nc, start + local);
                }
            },
        );
    } else {
        par::for_each_chunk_mut(next.cells_slice_mut(), |start, chunk| {
            for (local, nc) in chunk.iter_mut().enumerate() {
                cell(nc, start + local);
            }
        });
    }
}

/// Which of the fused sweep's per-cell phases are live this tick, plus
/// the read-only per-cell field it needs. A struct rather than bare
/// arguments so a gate can be added without moving the sweep toward the
/// argument ceiling, and so a caller cannot give one in the wrong
/// position.
struct FusedSweep<'a> {
    advect_temp_active: bool,
    /// Fill `cloud_transfer` with this hour's per-cell signed vapour ↔
    /// droplet transfer. See [`AtmoForcing::track_cloud_transfer`].
    track_cloud_transfer: bool,
    /// Per-cell upper-air temperature (`AtmoScratch::t_upper`), read by
    /// the vapour ↔ droplet transition.
    t_upper: &'a [f32],
}

/// Total humidity in the grid (surface + upper).
#[must_use]
pub fn total_humidity(grid: &HexGrid) -> f32 {
    grid.iter().map(|(_, cell)| cell.humidity_total()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;
    use crate::wind::compute_wind_field;
    use test_support::{
        default_temp_params, default_wind_params, make_wet_grid, stationary_atmosphere,
        total_moisture, wind_mags, zero_wind,
    };

    #[test]
    fn moisture_is_conserved() {
        let current = make_wet_grid();
        let mut next = current.clone();
        // #63/#146: grid-only conservation, one step, no sky-reservoir
        // bookkeeping here — needs the regime pinned off, not the new
        // default of on.
        let params = stationary_atmosphere();
        let tp = default_temp_params();
        let wf = zero_wind(&current);
        let wp = default_wind_params();
        let wm = wind_mags(&wf);

        let before = total_moisture(&current);
        step_atmosphere(
            &current,
            &mut next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&current).0,
                mean_elevation: surface_means(&current).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );
        let after = total_moisture(&next);

        assert!(
            (before - after).abs() < 1e-2,
            "Conservation violee : {before} -> {after}"
        );
    }

    #[test]
    fn moisture_conserved_with_wind() {
        let current = make_wet_grid();
        let mut next = current.clone();
        // #63/#146: same reason as `moisture_is_conserved` above.
        let params = stationary_atmosphere();
        let tp = default_temp_params();
        let wp = default_wind_params();
        let wf = compute_wind_field(&current, &wp, 0);
        let wm = wind_mags(&wf);

        let before = total_moisture(&current);
        step_atmosphere(
            &current,
            &mut next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&current).0,
                mean_elevation: surface_means(&current).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );
        let after = total_moisture(&next);

        assert!(
            (before - after).abs() < 1e-1,
            "Conservation with wind violated: {before} -> {after}"
        );
    }

    #[test]
    fn evaporation_feeds_surface_humidity() {
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                cell.water_level = 10.0;
                cell.temperature = 20.0;
            }
        }

        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&grid).0,
                mean_elevation: surface_means(&grid).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );

        let center = next.get(HexCoord::new(0, 0)).unwrap();
        assert!(center.water_level < 10.0, "water must decrease");
        assert!(
            center.humidity_total() > 0.0,
            "total humidity must increase"
        );
    }

    #[test]
    fn sublimation_below_freezing() {
        let mut grid = HexGrid::from_radius(0);
        let c0 = HexCoord::new(0, 0);
        if let Some(cell) = grid.get_mut(c0) {
            cell.snow_level = 2.0;
            cell.temperature = -5.0;
        }

        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&grid).0,
                mean_elevation: surface_means(&grid).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );

        let center = next.get(c0).unwrap();
        assert!(center.snow_level < 2.0, "snow must decrease");
        assert!(
            center.humidity_total() > 0.0,
            "total humidity must increase"
        );
    }

    #[test]
    fn precipitation_moves_upper_humidity_to_water() {
        let mut grid = HexGrid::from_radius(1);
        for coord in grid.coords().copied().collect::<Vec<_>>() {
            if let Some(cell) = grid.get_mut(coord) {
                // Phase 3: rescale ×200 (2.0 → 400.0), well above
                // the 24 mm saturation to trigger condensation + precipitation.
                cell.humidity_upper = 400.0;
            }
        }

        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&grid).0,
                mean_elevation: surface_means(&grid).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );

        let center = next.get(HexCoord::new(0, 0)).unwrap();
        assert!(center.humidity_upper < 400.0);
        assert!(center.water_level > 0.0);
    }

    #[test]
    fn precipitation_phase_follows_cell_temperature() {
        // Same oversaturated sky (humidity_upper well above
        // saturation), only the cell temperature differs: below 0°C
        // precipitation must fall as snow and NEVER as liquid water,
        // above as liquid water and NEVER as snow (`step_precipitation_into`,
        // branch `is_snow = nc.temperature < 0.0`).
        //
        // Complement to `phys_wet_peak_snows.rs` (integration test, verifies
        // only the cold-side accumulation over 200 ticks through the full
        // pipeline): here we isolate the phase branching in a single tick of
        // `step_atmosphere`, on two single-cell worlds (radius 0, purely
        // local, the snow/water partition depends only on the source
        // cell's temperature, not on transport), and we
        // explicitly verify that the OTHER stock stays at zero (not
        // just that the right stock increases).
        let params = AtmosphereParams::default();
        let tp = default_temp_params();
        let wp = default_wind_params();

        let mut cold = HexGrid::from_radius(0);
        if let Some(c) = cold.get_mut(HexCoord::new(0, 0)) {
            c.temperature = -10.0;
            c.humidity_upper = 400.0;
        }
        let wf_cold = zero_wind(&cold);
        let wind_mag_cold = wind_mags(&wf_cold);
        let mut cold_next = cold.clone();
        step_atmosphere(
            &cold,
            &mut cold_next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf_cold,
                wind_mag: &wind_mag_cold,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&cold).0,
                mean_elevation: surface_means(&cold).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );
        let cold_cell = cold_next.get(HexCoord::new(0, 0)).unwrap();

        let mut warm = HexGrid::from_radius(0);
        if let Some(c) = warm.get_mut(HexCoord::new(0, 0)) {
            c.temperature = 15.0;
            c.humidity_upper = 400.0;
        }
        let wf_warm = zero_wind(&warm);
        let wind_mag_warm = wind_mags(&wf_warm);
        let mut warm_next = warm.clone();
        step_atmosphere(
            &warm,
            &mut warm_next,
            &params,
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf_warm,
                wind_mag: &wind_mag_warm,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&warm).0,
                mean_elevation: surface_means(&warm).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );
        let warm_cell = warm_next.get(HexCoord::new(0, 0)).unwrap();

        assert!(
            cold_cell.snow_level > 0.0,
            "cold column (-10°C) under saturated sky must accumulate snow, snow={}",
            cold_cell.snow_level
        );
        assert!(
            cold_cell.water_level.abs() < 1e-6,
            "cold column must NOT receive liquid water, water={}",
            cold_cell.water_level
        );

        assert!(
            warm_cell.water_level > 0.0,
            "warm column (+15°C) under saturated sky must receive liquid water, water={}",
            warm_cell.water_level
        );
        assert!(
            warm_cell.snow_level.abs() < 1e-6,
            "warm column must NOT produce snow, snow={}",
            warm_cell.snow_level
        );
    }

    #[test]
    fn dry_grid_no_change() {
        let grid = HexGrid::from_radius(2);
        let mut next = grid.clone();
        let wf = zero_wind(&grid);
        let wp = default_wind_params();
        let tp = default_temp_params();
        let wm = wind_mags(&wf);
        step_atmosphere(
            &grid,
            &mut next,
            &AtmosphereParams::default(),
            &AtmoForcing {
                temp_params: &tp,
                wind_params: &wp,
                wind_field: &wf,
                wind_mag: &wm,
                transpiration_cover: None,
                synoptic_wind: None,
                hour_tick: 0,
                upper_air_mean_t: surface_means(&grid).0,
                mean_elevation: surface_means(&grid).1,
                moist_coarse: false,
                track_cloud_transfer: false,
            },
            &mut AtmoState::default(),
        );

        for (coord, cell) in next.iter() {
            let orig = grid.get(*coord).unwrap();
            assert!(
                (cell.water_level - orig.water_level).abs() < 1e-6
                    && (cell.humidity_surface - orig.humidity_surface).abs() < 1e-6
                    && (cell.humidity_upper - orig.humidity_upper).abs() < 1e-6,
                "state must stay unchanged on dry grid at {coord:?}"
            );
        }
    }

    #[test]
    fn conservation_after_many_steps() {
        let mut current = make_wet_grid();
        let initial = total_moisture(&current);
        // #63/#146: each iteration below rebuilds `AtmoState::default()`
        // from scratch, discarding the sky reservoir every tick — a
        // stationary atmosphere is the only way this loop can stay
        // conservative, and that is what this test is isolating (many
        // steps of plain advection), not the regime.
        let params = stationary_atmosphere();
        let tp = default_temp_params();
        let wp = default_wind_params();

        for tick in 0..100 {
            let wf = compute_wind_field(&current, &wp, tick);
            let wm = wind_mags(&wf);
            let mut next = current.clone();
            step_atmosphere(
                &current,
                &mut next,
                &params,
                &AtmoForcing {
                    temp_params: &tp,
                    wind_params: &wp,
                    wind_field: &wf,
                    wind_mag: &wm,
                    transpiration_cover: None,
                    synoptic_wind: None,
                    hour_tick: 0,
                    upper_air_mean_t: surface_means(&current).0,
                    mean_elevation: surface_means(&current).1,
                    moist_coarse: false,
                    track_cloud_transfer: false,
                },
                &mut AtmoState::default(),
            );
            current = next;
        }

        let final_moisture = total_moisture(&current);
        assert!(
            (initial - final_moisture).abs() < 1e-1,
            "Conservation apres 100 steps : {initial} -> {final_moisture}"
        );
    }
}
