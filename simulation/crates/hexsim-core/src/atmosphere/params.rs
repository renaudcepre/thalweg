use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct AtmosphereParams {
    /// Maximum crop coefficient `Kc_max` (dimensionless, FAO-56) for
    /// vegetal transpiration (#77). Physical transpiration replaces the old
    /// "implicit canopy" proxy: FAO-56 approach (Allen et al. 1998,
    /// *Crop evapotranspiration*, FAO Irrigation and Drainage Paper 56):
    ///   `ET = Kc × ET₀ × water_stress`
    /// with `ET₀` the reference evaporative demand (Dalton/Meyer, mm/day),
    /// `Kc = Kc_max × Σ(crop_coef_i × biomass_i)` (canopy weighted by each
    /// species' crop coefficient, #83), and water stress =
    /// `groundwater / capacity`.
    /// The transpired water is drawn *from* `groundwater` (strict
    /// conservation, no double counting). `Kc_max ≈ 1` = dense canopy
    /// transpiring at potential demand when water is not limiting.
    pub transpiration_coef: f32,
    /// Sublimation: snow turns directly into vapour below 0°C.
    pub sublimation_rate: f32,

    // --- Two-layer atmospheric model ---
    /// Base fraction of `humidity_surface` transferred to `humidity_upper`
    /// each tick.
    pub uplift_rate: f32,
    /// Uplift boost per °C of surface temperature above 0°C.
    pub uplift_thermal_coef: f32,
    /// Height (m) of the upper layer above the ground, used to compute
    /// `T_upper` via the lapse rate from the map-mean surface temperature
    /// (`upper_air_temperature`: `T̄ − lapse·(z − z̄ + h)/1000`). 1500 m =
    /// mid-low clouds.
    pub upper_layer_altitude_m: f32,
    /// Global precipitation gate with hysteresis: if the average of
    /// `humidity_upper` is below this threshold, NO cell precipitates
    /// (except snow, always allowed). Produces distinct rain waves.
    /// Hysteresis: opens at `gate`, closes at `gate × 0.75`.
    /// At 0, disabled (each cell precipitates on its own saturation).
    pub global_precip_gate: f32,
    /// Floor applied to `humidity_upper` at simulation startup: each cell
    /// starts with at least this amount of altitude vapour.
    /// Closed terrarium = no external input, so the cycle must be primed.
    /// 0.15 = median value matching the old `boundary_humidity`
    /// (0.30 with a 0.25 seasonal amplitude).
    pub initial_humidity_floor: f32,
    /// Fraction per metre of positive elevation gain of the `Surface`
    /// advection flow diverted to `humidity_upper` of the destination cell.
    /// Physical model: humid air pushed by wind against relief is forced
    /// to rise, and the vapour condenses at the relief's altitude. Without
    /// this, thermal breezes in a closed terrarium push vapour from the
    /// summits toward the plains (thermal convection), producing a
    /// permanent rain shadow over relief. Cap of 0.80 conversion per cell.
    /// Example: coef=0.005, `delta_elev=100m` → 50% of the flow goes to
    /// upper.
    pub orographic_lift_coef: f32,

    // Phase 6 (#29): `saturation_at_zero` and `saturation_doubling_celsius`
    // removed. The saturation curve is now the physical Clausius-Clapeyron
    // law via Tetens (cf. `saturation_upper`), parameterized solely by
    // `upper_layer_altitude_m` (layer height to integrate vapour density
    // in mm PW).

    // --- 3-stock model: humidity_upper (vapour) ↔ cloud_water (droplets) → precipitation ---
    /// Fraction of the supersaturation surplus (`humidity_upper` −
    /// `saturation_upper(T)`) drained into `cloud_water` each tick.
    /// Anchored to Clausius-Clapeyron (#63 P4E3): condensation kicks in
    /// at saturation, with no intermediate dimensionless RH threshold.
    pub condensation_rate: f32,
    /// RH below which droplets evaporate back into vapour.
    pub cloud_evap_hr_threshold: f32,
    /// Fraction of `cloud_water` that evaporates back to `humidity_upper`
    /// per tick when RH < `cloud_evap_hr_threshold`. Slower than
    /// condensation (clouds persist), analogous to cloud half-life times.
    pub cloud_evap_rate: f32,
    /// Cloud droplet concentration `N_c` (cm^-3) for the Khairoutdinov &
    /// Kogan 2000 autoconversion model. Negative exponent (-1.79): the
    /// more droplets there are, the smaller they are, the longer they
    /// stay suspended, the *less* the cloud rains.
    /// Typical values: ~100 cm^-3 (maritime, few aerosols, rains easily),
    /// ~600 cm^-3 (polluted continental, rains with difficulty).
    /// Default 200 = moderate continental (forests, grasslands).
    pub kk2000_droplet_count: f32,
    /// Fraction of `cloud_water` that diffuses to neighbours each tick.
    /// Represents sub-grid turbulence plus isotropic advection from local
    /// wind: clouds don't live as sharply delimited "islands", they have
    /// fuzzy edges. Breaks the checkerboard pattern that appears when
    /// each cell decides its precipitation threshold alone.
    pub cloud_diffusion_rate: f32,
    /// Directional advection rate of `cloud_water` by the upper-level wind
    /// (`wind_upper`). Analogous to `humidity_advection_rate` but for
    /// condensed droplets: a cloud formed above a vapour source (lake,
    /// forest) must travel with the wind before precipitating, otherwise
    /// the rain systematically falls back on the source (cell-lake cycle,
    /// issue #24). 3.0 by default for parity with `humidity_advection_rate`:
    /// droplets at 1500 m follow the same upper-level flow as the vapour
    /// that formed them. CFL cap at 0.95.
    ///
    /// Issue #68: the historical value (0.37) was a leftover; vapour had
    /// been bumped from 0.37 to 3.0 (2026-07-05, drizzle-on-lakes) without
    /// following the droplets, breaking the parity claimed above. Result:
    /// vapour was circulating 8x faster than the clouds it forms, hence
    /// visually motionless clouds ("smoke columns"). Diag
    /// `diag_cloud_advection_lifetime` (no-precip, scripted uniform wind
    /// ~14 m/s, synoptic OFF): at 0.37, 51% of the pulse remains on the
    /// source after 24 h; at 3.0, 0.3%. Still sub-scale versus pure
    /// physics (Courant ≫ 1 at 1 km/hourly tick); parity is a first brick,
    /// not the ceiling.
    pub cloud_advection_rate: f32,
    /// Fraction of precipitation that falls on neighbouring cells rather
    /// than the source cell. Models the spread of rain by turbulent air:
    /// a drop falls beneath the cloud but the lateral offset is on the
    /// order of the hex size (~1 km). At 0: rain falls right under the
    /// cloud. At 1: all rain goes to neighbours (absurd). 0.2-0.4:
    /// realistic spread that breaks the "checkerboard" look of rain.
    pub precip_neighbor_share: f32,
    /// Maximum precipitation per tick (units): physical limit of drop
    /// microphysics, fall speed plus max density.
    /// A cell heavily loaded with `cloud_water` cannot dump it all in one
    /// tick, it drains over several ticks, producing showers that last.
    /// 0.02 ≈ 30 mm/day (heavy storm but not absurd). At 0: disabled.
    pub max_precip_per_tick: f32,
    /// Issue #45: `HR_surface` above which vapour in the boundary layer
    /// (50 m) condenses into low droplets (radiative fog).
    /// Surface analogue of altitude condensation, but with a real RH
    /// threshold (fog kicks in before strict saturation).
    /// 0.95 = fog only kicks in very close to saturation (dew point
    /// reached).
    pub fog_condensation_threshold: f32,
    /// Issue #45: fraction of the surplus (RH - threshold) × `humidity_surface`
    /// transferred to `cloud_water` each tick when RH > threshold.
    /// Expressed per day (will be scaled /24 by
    /// `scale_atmosphere_for_hourly_tick`).
    /// 0.02 = slow, gentle condensation (fog appears progressively over
    /// ~30 simulated minutes once the dew point is reached).
    pub fog_condensation_rate: f32,
    /// Issue #46: diurnal convective drive coefficient (fraction of
    /// `humidity_surface` transferred to `humidity_upper` per K of
    /// ground-reference gap and per unit of `sin(solar_elevation)`).
    ///
    /// The drive `(T - t_ref).max(0) × sin_elev × convective_diurnal_coef`
    /// is added to `uplift_rate + temp_boost` in `step_uplift`.
    /// So: night (`sin_elev`=0) → drive = 0 (no nighttime convection);
    /// summer noon (`sin_elev`≈1, T-t_ref≈25 K) → drive ≈ 0.012 max →
    /// +1.2% of surface humidity sent aloft per tick. Over 4-5 afternoon
    /// hours: visible cumulus. Expressed per day, scaled /24.
    pub convective_diurnal_coef: f32,
    /// **Ascent trigger** (synoptic project Phase 3, ex-design C #69):
    /// reference vertical velocity (m/s) at which precipitation efficiency
    /// saturates to 1. Total ascent per cell is
    /// `w = H·(−∇·v) + v·∇z`: column convergence (core of depressions,
    /// where friction makes the trans-isobaric wind converge) **plus**
    /// orographic lift (wind against the slope); fronts and mountains go
    /// through the same physical mechanism, rising air.
    /// The factor applied to precip is
    /// `clamp(updraft_floor + w/updraft_ref_ms, 0, 1)`.
    /// Measured (Phase 3 ablation, r60 seed 42): horizontal convergence
    /// *alone* kills the mountains (conv −1.6e-3 s⁻¹ over the summits, the
    /// anticyclone anchors on the cold massif), hence the `v·∇z` term.
    /// Terrarium orders of magnitude: `H·conv` ~ ±0.3 m/s (conv ±2e-4 s⁻¹ ×
    /// H 1500 m), `v·∇z` ~ 1 m/s (5 m/s × 0.2 slope).
    /// **0.0 = trigger inactive** (precip unchanged); default activation
    /// awaits Phase 4 calibration. Requires synoptic wind (param
    /// `synoptic.enabled`, hardcoded ON by default) for coherent
    /// convergence zones (disproved on noise-wind, #69: smear).
    pub updraft_ref_ms: f32,
    /// Floor `[0,1]` of the precip factor in subsidence zones (ascent
    /// trigger): fraction of efficiency retained with no ascent.
    /// 0.0 = subsidence zones totally dry.
    pub updraft_floor: f32,
    /// Critical `cloud_water` mass (mm) below which precipitation is
    /// inhibited (ex-design A #69): the cloud loads up and travels instead
    /// of drizzling in place; above it, the super-linear KK2000 (cw^2.47)
    /// purges the excess (the "burst"). 0.0 = no-op (only
    /// `CLOUD_MIN_PRECIP`).
    /// Alone, it triggers *everywhere* (disproved #69); intended as a
    /// complement to the spatial convergence trigger for the traveling
    /// storage phase.
    pub precip_crit_mm: f32,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            // Kc_max FAO-56 (#77). The reference demand ET₀ comes from
            // Meyer/Dalton, calibrated for FREE WATER; stomatal
            // transpiration is less efficient per unit of demand, so a
            // Kc_max < 1 is physically justified (stomatal regulation +
            // canopy resistance). Calibrated to keep the terrestrial
            // humidity flux of the same order as the old proxy at world
            // scale WITHOUT tipping the sim into planetary drizzle
            // (checked via diag_water_cycle_baseline +
            // physics_lake_concentration before/after, anti-pattern #5).
            transpiration_coef: 0.5,
            sublimation_rate: 0.005,
            // 0.08 (vs 0.15 post-refactor): delays the rise of
            // `humidity_surface` toward `humidity_upper`. Combined with
            // `humidity_advection_rate=0.70`, vapour from a lake travels
            // noticeably farther before saturating: a necessary condition
            // for rain not to "fall back on the source".
            uplift_rate: 0.08,
            // 0.002: residual thermal boost. Enough to avoid a fully
            // passive atmosphere, but too weak to recreate the captive
            // halo over warm lakes.
            uplift_thermal_coef: 0.002,
            upper_layer_altitude_m: 1500.0,
            // 0.0: disabled. With the 3-stock model (cloud_water as an
            // intermediate reservoir), each cell precipitates on its own
            // local critical mass, no global synchronization needed. A
            // global gate created artificially long rain/dry cycles
            // (3 months of continuous rain).
            global_precip_gate: 0.0,
            // Phase 6 (#29): recalibrated 30 → 10 mm. With the new Tetens
            // curve, PW_sat(15°C) ≈ 19 mm. Starting at 10 mm gives an
            // initial RH ≈ 0.5, consistent with the old ratio
            // (RH_init ≈ 0.6).
            initial_humidity_floor: 10.0,
            // 0.05: 80% cap reached over 16 m of elevation gain. Very
            // strong by design: isolated summits (>1000m) are far from
            // humidity sources (lowland lakes) and cascading propagation
            // must cross several neighbour levels before arriving.
            // Lower it (0.005-0.02) if rainfall becomes too concentrated
            // on relief.
            orographic_lift_coef: 0.05,
            // Phase 6 (#29): `saturation_at_zero` + `saturation_doubling_celsius`
            // removed. Saturation now comes from physical Tetens,
            // parameterized solely by `upper_layer_altitude_m`.
            // #63 Phase 4 Step 3: physical anchoring to microphysical
            // τ_phase.
            // Pruppacher & Klett (1997), *Microphysics of Clouds and Precipitation*,
            // 2nd ed., §13.3.1 "Phase relaxation time":
            //   τ_phase = 1 / (4π·D·N·r̄)
            //   D = 2.5e-5 m²/s (vapour diffusion coef in air, 0°C, 1 atm)
            //   N = 1e7 to 1e9 m⁻³ (typical droplet concentration,
            //                       stratiform to cumuliform clouds)
            //   r̄ = 5 to 15 µm (mean radius)
            //   → τ_phase ∈ [0.7, 30] s for the realistic range.
            // At hourly tick (Δt = 3600 s), 1 - exp(-Δt/τ_phase) ≈ 1.0:
            // supersaturation resolves almost completely each tick.
            // Default expressed "per day" to follow the convention of
            // `scale_atmosphere_for_hourly_tick`: 24.0/day → 1.0/h after /24.
            // The `min(1.0)` in `step_cloud_dynamics` caps by construction
            // (drain cannot exceed 100% of the surplus per tick).
            //
            // History: 0.04 (Phase 6 #29 calibrated for the HR-fractional
            // formula with Tetens correction) → 24.0 here because the
            // formula moved to mm-absolute (cf JOURNAL pivot of
            // 2026-04-29). The 0.04 value only made sense for the
            // HR × hu formula, inseparable from its unit; the recalibration
            // is mandatory, anchored this time to cloud microphysics
            // rather than empirical behaviour.
            condensation_rate: 24.0,
            // Slower cloud evaporation: a cloud persists several ticks
            // even after the surrounding air has dried. Half-life ~7
            // ticks at rate 0.10 (1 - 0.9^7 ≈ 0.52).
            cloud_evap_hr_threshold: 0.4,
            cloud_evap_rate: 0.10,
            // KK2000: N_c = 50 cm^-3, semi-continental regime (real range
            // 30-1000 cm^-3). Phase 4 (bursts): raised from 30 (near-
            // pristine maritime, ex-default) to 50 for temporal
            // re-concentration. Exponent -1.79: autoconversion becomes
            // (30/50)^1.79 ≈ 0.40x slower, so a cloud loads up more before
            // precipitating, hence a more intense shower when it falls
            // (plains peak intensity ×1.8 measured on the bench,
            // 1.1→2.3 mm/day).
            // 50 chosen as the MAX value that re-concentrates without
            // crossing the microphysical guardrails calibrated for N_c=30
            // (heavy_cloud_rains: 1mm cloud yields 0.61>0.5 mm; no_chimney:
            // peak <3 mm; world_stays_humid green), no loosening of the
            // tripwire. N_c=100 doubled the intensity but crossed both
            // guardrails.
            kk2000_droplet_count: 50.0,
            // cloud_water diffusion: ~15% to neighbours per tick. Breaks
            // the checkerboard pattern (one cell rains, its neighbour
            // doesn't) without fully smoothing out climatic contrasts. At
            // 0.30+ clouds lose all spatial identity.
            cloud_diffusion_rate: 0.15,
            // cloud_water directional advection: aligned by default with
            // humidity_advection_rate (0.37); droplets travel with the
            // upper-level flow at the same rate as vapour, as a first
            // approximation. Lets clouds move across the map instead of
            // staying camped on the condensation zone.
            cloud_advection_rate: 3.0,
            // 35% of rain falls on neighbours, 65% on the source cell.
            // Makes showers spatially coherent (a patch of 3-6 rained-on
            // cells, not an isolated tile). Physically realistic: a real
            // storm is 5-20 km wide, so it covers several hex cells
            // (~1 km² each).
            precip_neighbor_share: 0.35,
            // Phase 3 (#32): rescaled ×200. 4 mm/tick = microphysical cap
            // (equivalent to the old 0.02 * 200). With 1 tick = 1 day,
            // gives a max of 4 mm/day: far more conservative than the
            // physical cap of 200 mm/day (terminal fall), but keeps
            // showers spread over several ticks. To revisit in Phase 5
            // with a physical bound.
            max_precip_per_tick: 4.0,
            // Issue #45: RH_surface threshold for surface condensation
            // (radiative fog). 0.95 = only kicks in very close to the dew
            // point, otherwise fog everywhere in humid zones.
            fog_condensation_threshold: 0.95,
            // Issue #45: surface condensation rate (per day, scaled /24).
            // 0.02 = fog forms progressively over ~30 simulated clock
            // minutes once RH_surface > 0.95.
            fog_condensation_rate: 0.02,
            // Issue #46: diurnal convective drive coef. 0.0005/day ≈
            // 2.1e-5/hour × (T-t_ref).max(0) × sin_elev. At summer noon,
            // lowland 44.5°N (T≈30, t_ref≈2 → t_excess=28, sin_elev≈0.93)
            // we get a boost ≈ 5.5e-4 of the uplift rate per cell, which
            // adds to the existing temp_boost to pulse convection in
            // mid-afternoon.
            convective_diurnal_coef: 0.0005,
            // Synoptic project Phase 3: triggers OFF by default; the
            // calibrated climate stays unchanged until Phase 4 has ruled.
            updraft_ref_ms: 0.0,
            updraft_floor: 0.0,
            // 0.15 mm (#63, enabled on 2026-07-15). Critical LWP mass
            // below which the cloud loads up and travels without
            // precipitating; above it, the super-linear KK2000 purges in
            // a burst. Replaces the magic number `CLOUD_MIN_PRECIP=0.05`
            // (bare numeric floor) with an autoconversion threshold in a
            // physical unit (mm of LWP, order of magnitude of the
            // observed stratocumulus drizzle onset, ~0.1-0.3 mm).
            //
            // Measured (audit #67 + `scale_precip_regime` sweep, seed 42,
            // 2 years): converts permanent drizzle into concentrated
            // showers, rainy cell-day intensity ×1.5 (0.51→0.75 mm),
            // drizzle extent −33% (354→238 cells/day), without
            // desertifying the summits (>1500m stays at ~10 days/year) or
            // touching seasonal snow (`snow_min_late` unchanged). 0.30+
            // dries out the summits too much.
            //
            // The former objection #87 ("perennial glacier" regression
            // from `scale_dry_periods`) is moot: that criterion was
            // removed (terrain <1800 m at R=30, no legitimate glacier).
            // The ascent trigger (`updraft_ref_ms`, ex-design C #69)
            // stays OFF: diagnosed as broken (aberrant `w` field) and
            // redundant; `precip_crit_mm` is the only drizzle→shower
            // texture lever.
            precip_crit_mm: 0.15,
        }
    }
}
