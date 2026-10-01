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
    /// Relative humidity floor applied to `humidity_upper` at simulation
    /// startup: each cell starts with at least this fraction of the upper
    /// layer's saturation at its own upper-air temperature
    /// (`saturation_upper(upper_air_temperature(..))`, the January profile
    /// of a fresh world). Closed terrarium = no external input, so the
    /// cycle must be primed; a profile by altitude and season instead of
    /// the former uniform 10 mm (#152), which was a summer plain's vapour
    /// over every winter summit. 0.6 = the free troposphere's mean
    /// relative humidity over temperate land (Peixoto & Oort 1996, J.
    /// Climate 9: 50-70 % in the lower troposphere). 0 disables the floor.
    pub initial_humidity_floor: f32,
    /// Per metre of positive elevation gain, per day (divided by
    /// `TICKS_PER_DAY` by `scale_atmosphere_for_hourly_tick` before use).
    /// Read by the two orographic paths, which share the physics "vapour
    /// forced up a slope reaches the upper layer" but not the geometry:
    ///
    /// - the wind-driven one, `advection::fill_lift_outflow`: fraction of
    ///   the `Surface` advection flow diverted to `humidity_upper` of the
    ///   destination cell, `clamp(coef × Δz, 0.0, 0.80)`;
    /// - the isotropic pump, `uplift::oro_pump_rate` (#156): inverse
    ///   lifting timescale per metre of convergent relief,
    ///   `rate = 1 − exp(−coef × Σ Δz⁺)`. Read its doc for the unit
    ///   breakdown — the coefficient is the group `U_slope·dt/(H·L)` and
    ///   the shipped default matches an upslope venting speed of about
    ///   0.11 m/s.
    ///
    /// Without either, thermal breezes in a closed terrarium push vapour
    /// from the summits toward the plains, producing a permanent rain
    /// shadow over relief.
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
    // #63 L2b: `cloud_evap_hr_threshold` (0.4) and `cloud_evap_rate`
    // (0.10/day) removed. The reverse transition is a saturation
    // adjustment bounded by the layer's own deficit, so it has no
    // parameter left to carry — see
    // `condensation::saturation_adjustment_transfer`. Old checkpoints and
    // params files still carrying the two keys load fine: this struct
    // does not `deny_unknown_fields`, serde drops them.
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
    /// Fraction of precipitation that leaves the source cell, spread with
    /// a gradient decreasing over `precip_spread_radius` hexes rather
    /// than landing entirely under the cloud. Models the lateral drift of
    /// falling drops in turbulent, wind-carried air.
    ///
    /// **Units and derivation (2026-09-06, floor not physics):** a
    /// raindrop's terminal velocity is 4-9 m/s (Gunn & Kinzer 1949;
    /// drizzle ~0.5 mm: ~2 m/s; snow: ~1 m/s). Falling from
    /// `upper_layer_altitude_m` = 1500 m takes 3-6 minutes (snow: ~25
    /// min), during which the upper-level wind (2-8 m/s here) carries the
    /// drop 0.4-3 km sideways; a shower cell itself spans 1-10 km (Byers
    /// & Braham 1949). At the map's 130 m hex spacing (fixed 2026-07-09)
    /// that is a footprint 3 to 25 hexes in radius — `precip_spread_radius`
    /// below picks a point in that range, not the physics itself. This
    /// doc used to read "the lateral offset is on the order of the hex
    /// size (~1 km)": stale since the spacing dropped from ~1 km to
    /// 130 m, corrected here.
    ///
    /// At 0: rain falls right under the cloud regardless of
    /// `precip_spread_radius`. At 1: all rain leaves the source (absurd:
    /// nothing rains under the cloud that formed it). 0.2-0.4: realistic
    /// spread that breaks the "checkerboard" look of rain. Orthogonal to
    /// the radius below: this is *what fraction* leaves the source,
    /// `precip_spread_radius` is *how far* it is allowed to travel before
    /// landing.
    pub precip_neighbor_share: f32,
    /// Radius (hexes) of the footprint a cloud rains onto: `1 −
    /// precip_neighbor_share` stays at the source, the rest spreads with
    /// a gradient decreasing with hex distance out to this many rings,
    /// instead of landing entirely on the source and its 6 immediate
    /// neighbours (the historical, and still the `= 1`, footprint).
    /// Rounded to the nearest integer number of diffusion passes by
    /// `precipitation::step_precipitation_into` (`precip_spread_passes`);
    /// `1` reproduces the legacy single-ring footprint bit for bit (see
    /// `precipitation::tests::radius_one_is_bit_identical_to_legacy`).
    ///
    /// See `precip_neighbor_share` above for the physical derivation of
    /// the 3-25 hex range; `3` (390 m) is the cheap, conservative end of
    /// it, sized to fix the "salt and pepper" rain pattern the owner
    /// measured on a live r120 world in July: ~140 rained-on islets per
    /// hour, the biggest averaging 19 cells, 47% of them 1-2 cells
    /// (`just rain-pattern`, threshold 0.1 mm/h) — the owner's
    /// screenshots show the same thing, isolated 1-7 hex clusters. Each
    /// additional ring costs one more full diffusion pass over the grid
    /// (the `atmo_precipitation` perf bucket): measured cost is in
    /// the project journal's entry for this change, gated at at most ×2 of the
    /// pre-change bucket by `just perf-phases`.
    ///
    /// Not physically the same column as the source: KK2000 and the
    /// rain/snow phase test both still run once, at the source cell, as
    /// before — a cell 3 rings downhill or in a different microclimate
    /// receives the source's phase and amount, not its own. Left as a
    /// simplification (documented here, not changed by this radius).
    ///
    /// `#[serde(default = "default_precip_spread_radius_legacy")]`
    /// (`1.0`), not the new `3.0`, for the same reason as
    /// `regime_enabled`'s bare `#[serde(default)]` above: a checkpoint or
    /// params file predating this radius has no opinion on it and must
    /// not wake up with a wider footprint it never asked for. Only a
    /// freshly generated world (`AtmosphereParams::default()` below) gets
    /// `3.0`.
    #[serde(default = "default_precip_spread_radius_legacy")]
    pub precip_spread_radius: f32,
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
    /// **0.0 = trigger inactive** (precip unchanged), and inactive is
    /// where it stays. Requires synoptic wind (param `synoptic.enabled`,
    /// hardcoded ON by default) for coherent convergence zones (disproved
    /// on noise-wind, #69: smear).
    ///
    /// # Why it is off, and what it would take to turn it on (#110)
    ///
    /// Not "awaiting calibration", which this doc claimed until
    /// 2026-09-07: measured broken on 2026-07-15 by `diag_updraft_field`
    /// and ruled unsalvageable *by re-calibration*, which is a different
    /// verdict. The numbers: `w` spans −225 to +314 m/s (p99 +143,
    /// std 47.6) where the design above reasons about ±1 m/s, so the
    /// implementation overshoots its own target by 50-300×; and its
    /// altitude signature is inverted, plains at +10.1 (factor saturated
    /// to 1, so plain drizzle is untouchable) against −82.6 over the
    /// summits, which is subsidence where orographic lift belongs. On top
    /// of that it is structurally redundant: the rain-weighted mean factor
    /// holds near 0.9 whatever `w_ref` does, because the cells that rain
    /// are already the cells where `w > 0` — the cloud encodes that the
    /// air rose. No value of this parameter fixes that.
    ///
    /// Root cause: `−∇·v` by finite differences over ~130 m cells, fed a
    /// wind noisy at the cell scale (deflection, breeze), measures grid
    /// noise rather than synoptic convergence; × `h_column` = 1500 m gives
    /// the absurd magnitudes.
    ///
    /// # The estimator is repaired (2026-09-30), the trigger stays off
    ///
    /// `fill_updraft_into` now derives the **ambient** wind, the synoptic
    /// base interpolated from the coarse mesh (`AtmoForcing::synoptic_wind`,
    /// smooth at `L_d`), for the convergence AND the barrier lift, never
    /// the deflected composite. Measured by `diag_updraft_field` (r30,
    /// seed 42, 2 years sampled), before → after, same day, same base:
    /// `w` −118 / +147 m/s → **−3.6 / +3.2 m/s**, std 23.5 → **1.10**,
    /// mean −1.3 → 0.003; band means +5.6 m/s over the plains and −38
    /// over the summits → **−0.015 over the plains, +0.117 over the
    /// summits**, the right way up. The ±0.3-1 m/s orders of magnitude
    /// above are the ones the field now has. Same result as the orphaned
    /// July branch `fix/synoptic-zero-fronts` (±3.6, std 1.10), redone on
    /// the current structure rather than merged.
    ///
    /// Still `0.0` by default: a sane `w` is the prerequisite, not the
    /// verdict. What the trigger does to the climate and to cloud travel
    /// is measured (`diag_cloud_travel`, the r30 bench) and written in
    /// the JOURNAL of 2026-09-30 and on #110; the flip is the owner's
    /// call on those numbers.
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

    // --- Imposed weather regime (#63), boundary condition, ON by default ---
    /// Master switch of the imposed synoptic weather regime
    /// (`atmosphere::regime`), dimensionless: `0` = off (the pass is
    /// skipped entirely and the build is bit-identical to one without
    /// it), anything else = on. Same "0 disables" shape as
    /// `updraft_ref_ms` above rather than a `bool`, so the whole struct
    /// stays f32 and the runtime key `atmosphere.regime_enabled` behaves
    /// like every other one.
    ///
    /// The regime is a boundary condition, not a phenomenon internal to a
    /// cell: the synoptic scale (the weather systems that decide whether a
    /// region is under a dry high or a moist frontal passage) lives at
    /// hundreds of kilometres, well outside the ~16 km box the map covers,
    /// exactly like the sun's position is a forcing the map cannot compute
    /// from its own cells. Without it, every cell resolves its own
    /// saturation locally and the map never runs dry all at once, because
    /// nothing sub-synoptic desaturates every column together. A dry spell
    /// is large-scale subsidence: descending, adiabatically warmed air that
    /// suppresses convection over the whole box for days, not a local
    /// depletion any cell-level rule can produce on its own.
    ///
    /// **Default 1.0 (flipped 2026-09-06, #63/#146).** Measured on 3 seeds
    /// (r30, 730 days, combined with the L2b saturation-adjustment cloud
    /// evaporation it now ships alongside): with the regime ON, 131 / 90 /
    /// 239 fully rain-free days per 2 years (0 with it off), the wettest
    /// cell drops from ~355 to 169-240 rain days/year, and
    /// `summer/winter`, lapse rate and water-budget drift all hold or
    /// improve. With the regime off, the new evaporation alone
    /// concentrates rain on relief and dries the plains instead (56 → 26
    /// rain days/yr), so the coherent package is evaporation *and* regime
    /// together, not evaporation alone. Get the old always-saturating,
    /// stationary atmosphere back with `just param
    /// "atmosphere.regime_enabled" 0`.
    ///
    /// `#[serde(default)]` stays bare (`f32::default()` = 0.0), not the
    /// new 1.0: a checkpoint or params file predating the regime (no
    /// `regime_enabled` key at all) is data from before the mechanism
    /// existed and should not wake up under it uninvited. Only a freshly
    /// generated world (`AtmosphereParams::default()` below) gets the new
    /// default; anything old that already carries the key keeps whatever
    /// value it was saved with.
    #[serde(default)]
    pub regime_enabled: f32,
    /// Mean length (days) of a dry episode of the two-state Markov chain,
    /// i.e. `p(dry→wet) = 1 / regime_dry_mean_days` evaluated once per
    /// simulated day. With `regime_wet_mean_days` below, the stationary
    /// wet fraction is `(1/dry) / (1/dry + 1/wet)` = 0.31, i.e. ≈112
    /// days a year with a moist air mass over the box.
    ///
    /// 4.5 d is chosen to land that wet fraction on the ~80-110 rainy
    /// days a year of a lowland Rhône-valley climate (the Drôme the map
    /// is calibrated on). **Not verified against the published
    /// Météo-France 1991-2020 normals in this session** — the figure is
    /// an order of magnitude taken from the design note (JOURNAL
    /// 2026-09-03), not a sourced constant. Re-anchor it on the real
    /// normals before the lever is ever turned on by default.
    #[serde(default = "default_regime_dry_mean_days")]
    pub regime_dry_mean_days: f32,
    /// Mean length (days) of a wet episode, `p(wet→dry) = 1 /
    /// regime_wet_mean_days`. 2.0 d = the passage of one frontal system,
    /// same caveat on the source as `regime_dry_mean_days`.
    #[serde(default = "default_regime_wet_mean_days")]
    pub regime_wet_mean_days: f32,
    /// Relative humidity (dimensionless, 0-1) the upper layer relaxes
    /// toward during a dry episode. A physical statement about the air a
    /// large-scale subsidence brings down: free-tropospheric air that has
    /// descended is dry, RH 0.3-0.6 is its usual range (Sherwood et al.
    /// 2010, *Tropospheric water vapor, convection and climate*, Rev.
    /// Geophys. 48, on subsidence drying). 0.5 = the middle of it. Kept
    /// at 0.5 by #63 L2b: the ablation that produced rain-free days used
    /// 0.25, and free-tropospheric subsidence really is drier than 0.5
    /// on the dry branches, but no radiosonde climatology could be
    /// sourced offline in that session to justify a specific number, and
    /// the value that makes the bench prettiest is not a source.
    ///
    /// The coupling with the vapour ↔ droplet transition inverted at
    /// L2b. It used to be inert: 0.5 sat above the old
    /// `cloud_evap_hr_threshold` (0.4), so the export left the droplets
    /// alone and they had to rain out or travel. With the saturation
    /// adjustment that replaced that threshold, an export down to
    /// `0.5 × sat` opens a deficit of the same `0.5 × sat`, and the
    /// transition pass that runs immediately after evaporates droplets
    /// straight into it. Lowering this target now dissolves clouds
    /// faster as well as drying the vapour — one number, two effects,
    /// which is a reason to source it rather than tune it.
    #[serde(default = "default_regime_dry_rh_target")]
    pub regime_dry_rh_target: f32,
    /// Time constant τ (hours) of the first-order export of the vapour
    /// surplus toward the sky reservoir during a dry episode: one hourly
    /// step moves `excess × (1 − exp(−1 h / τ))`. 6 h = the order of
    /// magnitude of a subsidence drying out a 1500 m column at the
    /// ~1 cm/s typical of large-scale descent.
    #[serde(default = "default_regime_export_hours")]
    pub regime_export_hours: f32,
    /// Time constant τ (hours) of the return of the sky reservoir to the
    /// map during a wet episode, same first-order form. 12 h = a moist
    /// air mass takes about half a day to fill the box it advects into.
    #[serde(default = "default_regime_return_hours")]
    pub regime_return_hours: f32,
}

/// `serde(default)` value for [`AtmosphereParams::regime_dry_mean_days`]:
/// a checkpoint or a params file predating the weather regime loads the
/// shipped default rather than being refused (same precedent as
/// `TemperatureParams::terrain_insolation_factor`). Harmless whatever the
/// value: a file that old is also missing `regime_enabled`, which keeps
/// its own bare `#[serde(default)]` of 0 (off) for exactly that reason.
fn default_regime_dry_mean_days() -> f32 {
    4.5
}

/// See [`default_regime_dry_mean_days`].
fn default_regime_wet_mean_days() -> f32 {
    2.0
}

/// See [`default_regime_dry_mean_days`].
fn default_regime_dry_rh_target() -> f32 {
    0.5
}

/// See [`default_regime_dry_mean_days`].
fn default_regime_export_hours() -> f32 {
    6.0
}

/// See [`default_regime_dry_mean_days`].
fn default_regime_return_hours() -> f32 {
    12.0
}

/// `serde(default)` value for
/// [`AtmosphereParams::precip_spread_radius`]: the pre-existing single-ring
/// footprint (`1` = today's behavior, see `precip_spread_radius`'s doc), so
/// a params file or checkpoint predating this radius keeps the footprint
/// it always had rather than jumping to the new default of `3`.
fn default_precip_spread_radius_legacy() -> f32 {
    1.0
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
            // A relative humidity since #152 (it was 10 mm, i.e. RH ≈ 0.5
            // at 15 °C and ×2 saturation over a winter summit): 0.6 of the
            // layer's saturation at its own temperature, ~1 mm per cell in
            // a January world, the order the engine holds by itself by its
            // sixth January (0.6-1.1 mm per cell, 2026-10-01).
            initial_humidity_floor: 0.6,
            // 0.05 /m/day, i.e. 0.00208 /m per hourly tick. Strong by
            // design: isolated summits (>1000 m) are far from humidity
            // sources (lowland lakes) and cascading propagation must
            // cross several neighbour levels before arriving. Its two
            // consumers read it differently, see the field's doc.
            // For the isotropic pump it is `U_slope·dt/(H·L)`, an upslope
            // venting speed of ~0.11 m/s at H = 1500 m and L = 130 m; on
            // the median hillside of a radius-30 map (Σ Δz⁺ ≈ 190 m) the
            // pump exports ~33 % of the surface layer per hour.
            // Sensitivity re-measured at x3 and /3 on 3 seeds when the
            // 0.30 cap was removed (#156): mountain rain days move
            // monotonically, 66 -> 92 -> 107 -> 264 days/year above
            // 1500 m on seed 42. Lower it (0.005-0.02) if rainfall
            // becomes too concentrated on relief.
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
            // 35% of rain leaves the source, spread over
            // `precip_spread_radius` rings below. Physically realistic: a
            // real storm is 5-20 km wide (Byers & Braham 1949), so it
            // covers several hex cells at the map's 130 m spacing.
            precip_neighbor_share: 0.35,
            // 3 rings (390 m): the cheap, conservative end of the 3-25
            // hex physical range derived in `precip_neighbor_share`'s
            // doc. Fixes the "salt and pepper" rain pattern measured on a
            // live r120 world (2026-09-06): a footprint of ~140 rained-on
            // islets/hour averaging 19 cells, 47% just 1-2 cells. `1`
            // reverts to the historical single-ring footprint.
            precip_spread_radius: 3.0,
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
            // texture lever. The measurement, the root cause and what
            // turning it on would take are on `updraft_ref_ms` itself.
            precip_crit_mm: 0.15,
            // Imposed weather regime (#63): ON by default since 2026-09-06
            // (#146), shipped together with the L2b saturation-adjustment
            // cloud evaporation. See `regime_enabled` for the measurement
            // and the one-line way back to the old stationary atmosphere.
            regime_enabled: 1.0,
            regime_dry_mean_days: default_regime_dry_mean_days(),
            regime_wet_mean_days: default_regime_wet_mean_days(),
            regime_dry_rh_target: default_regime_dry_rh_target(),
            regime_export_hours: default_regime_export_hours(),
            regime_return_hours: default_regime_return_hours(),
        }
    }
}
