//! Per-phase timing of the real tick (`Simulation::step_hour`).
//!
//! Unlike the `perf_phase_breakdown` bench (which REPLAYS the tick phase
//! by phase and must be kept as a mirror, a blind spot documented in its
//! header), these counters live INSIDE the orchestrator: every phase
//! that runs is measured, at production cadences, including the Tier 3
//! phases (vegetation, fire, lakes, erosion, normals) that the mirror
//! bench ignores.
//!
//! Cost: 2 clock reads per phase per tick (~15 ns each on Apple
//! Silicon), negligible next to phases running in the hundreds of µs.
//! On wasm32 (`Instant` unavailable), the clock is a no-op and all
//! counters stay at zero.

/// Wall-clock cumulative totals (seconds) per tick phase, since the
/// [`crate::simulation::Simulation`] was created or the last
/// [`crate::simulation::Simulation::reset_phase_timings`].
#[derive(Debug, Default, Clone, Copy)]
pub struct PhaseTimings {
    /// `compute_illumination` (shadow raymarch + cloud shadow), hourly.
    pub illumination: f64,
    /// `step_temperature`, hourly.
    pub temperature: f64,
    /// Synoptic dynamics (aggregate + ODE + interpolation), 1 h out of M.
    pub synoptic: f64,
    /// `compute_wind_field_into` + magnitudes, 1 h out of N.
    pub wind: f64,
    /// `step_snow`, hourly.
    pub snow: f64,
    /// `step_atmosphere_into` (evap, uplift, advection, condensation, precip).
    pub atmosphere: f64,
    /// `ClimateNormalsAccumulator::record_tick`, hourly.
    pub normals: f64,
    /// `ClimateHistory::record_tick`, daily.
    pub history: f64,
    /// `step_groundwater`, daily.
    pub groundwater: f64,
    /// MFD slice (8 passes + map accumulation), daily.
    pub hydro: f64,
    /// EMA discharge + edge flux (#105), daily.
    pub ema: f64,
    /// `step_lake_leveling` (#106), daily.
    pub lakes: f64,
    /// `step_erosion` + surface normals recompute, daily.
    pub erosion: f64,
    /// `step_vegetation`, daily.
    pub vegetation: f64,
    /// `step_fire`, daily.
    pub fire: f64,
    /// Number of simulated hours covered by the cumulative totals.
    pub hours: u64,
}

impl PhaseTimings {
    /// Sum of the measured phases (s). Slightly lower than the full
    /// tick's wall clock: the inter-phase glue (swaps, precip
    /// accumulation) is not counted.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.rows().iter().map(|&(_, s)| s).sum()
    }

    /// `(name, seconds)` rows in tick execution order.
    #[must_use]
    pub fn rows(&self) -> [(&'static str, f64); 15] {
        [
            ("illumination", self.illumination),
            ("temperature", self.temperature),
            ("synoptic", self.synoptic),
            ("wind", self.wind),
            ("snow", self.snow),
            ("atmosphere", self.atmosphere),
            ("normals", self.normals),
            ("history", self.history),
            ("groundwater", self.groundwater),
            ("hydro", self.hydro),
            ("ema", self.ema),
            ("lakes", self.lakes),
            ("erosion", self.erosion),
            ("vegetation", self.vegetation),
            ("fire", self.fire),
        ]
    }
}

/// Opaque clock mark. `Instant` on native, unit on wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub type Mark = std::time::Instant;
/// Opaque clock mark. `Instant` on native, unit on wasm32.
#[cfg(target_arch = "wasm32")]
pub type Mark = ();

/// Start a phase measurement.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
#[must_use]
pub fn mark() -> Mark {
    std::time::Instant::now()
}

/// Start a phase measurement (wasm no-op).
#[cfg(target_arch = "wasm32")]
#[inline]
#[must_use]
pub fn mark() -> Mark {}

/// Seconds elapsed since `m`.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
#[must_use]
pub fn elapsed_s(m: Mark) -> f64 {
    m.elapsed().as_secs_f64()
}

/// Seconds elapsed since `m` (always 0 on wasm).
#[cfg(target_arch = "wasm32")]
#[inline]
#[must_use]
pub fn elapsed_s(m: Mark) -> f64 {
    let () = m;
    0.0
}
