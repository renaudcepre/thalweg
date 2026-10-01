//! Centralizes the environment-variable ablation switches (A/B knobs) read
//! by the engine.
//!
//! # Why
//! Each switch is a legitimate, measured A/B lever (see the doc-comments at
//! their call sites: `simulation::wind_subsample`, `simulation::synoptic_subsample`,
//! the coarse synoptic mesh toggle in `Simulation::new`,
//! the moist-layer coarse mesh toggle in `Simulation::new`
//! (`moist_coarse`, coarse upper layer step 1),
//! `atmosphere::scaling::transport_subsample`, `atmosphere::scaling::oro_subsample`,
//! `atmosphere::scaling::temp_advection_subsample`,
//! `atmosphere::scaling::precip_subsample`, `temperature::illum_ko`).
//! None of them should be deleted.
//!
//! But every one of them is process-global state read from the environment,
//! entirely outside the seed. `Simulation::save_state`'s doc-comment promises
//! "bit-identical resumption is proven by test", and that promise is false
//! the moment a checkpoint saved under one ablation config is reloaded under
//! another: the synoptic subsample in particular is documented at its call
//! site as "a real physics change (systems slowed to 1/M)", not a cosmetic
//! one. Reloading with a different `HEXSIM_SYNOPTIC_SUBSAMPLE` silently
//! resumes the same seed in different physics.
//!
//! So the ablation config is captured by `Checkpoint`
//! alongside the grid and the clock, and checked on load: a mismatch is
//! refused ([`crate::checkpoint::CheckpointError::Ablation`]), not silently
//! applied.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::atmosphere::MoistCoarseMode;
use crate::atmosphere::{
    MOIST_COARSE_DEFAULT, ORO_SUBSAMPLE_HOURS, PRECIP_SUBSAMPLE_HOURS,
    TEMP_ADVECTION_SUBSAMPLE_HOURS, TRANSPORT_SUBSAMPLE_HOURS,
};
use crate::simulation::{SYNOPTIC_COARSE_DEFAULT, SYNOPTIC_SUBSAMPLE_HOURS, WIND_SUBSAMPLE_HOURS};
use crate::temperature::ILLUM_KO_DEFAULT;

/// Environment variable names, one per switch. Read exclusively by
/// [`Ablation::from_env`], the only function in the crate calling
/// `std::env::var`.
const ENV_WIND_SUBSAMPLE: &str = "HEXSIM_WIND_SUBSAMPLE";
const ENV_SYNOPTIC_SUBSAMPLE: &str = "HEXSIM_SYNOPTIC_SUBSAMPLE";
const ENV_SYNOPTIC_COARSE: &str = "HEXSIM_SYNOPTIC_COARSE";
const ENV_MOIST_COARSE: &str = "HEXSIM_MOIST_COARSE";
const ENV_TRANSPORT_SUBSAMPLE: &str = "HEXSIM_TRANSPORT_SUBSAMPLE";
const ENV_ORO_SUBSAMPLE: &str = "HEXSIM_ORO_SUBSAMPLE";
const ENV_TEMP_ADVECTION_SUBSAMPLE: &str = "HEXSIM_TEMP_ADVECTION_SUBSAMPLE";
const ENV_PRECIP_SUBSAMPLE: &str = "HEXSIM_PRECIP_SUBSAMPLE";
const ENV_ILLUM_KO: &str = "HEXSIM_ILLUM_KO";

/// Snapshot of every ablation switch the engine reads from the environment.
///
/// Captured once per process by [`Ablation::effective`] and persisted in the
/// `Checkpoint` so a reload can detect (and
/// refuse) a mismatch instead of silently resuming in a different physics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ablation {
    /// See `simulation::wind_subsample`. Wind field recompute cadence
    /// (hours); `1` = historical hourly recompute.
    pub wind_subsample: u64,
    /// See `simulation::synoptic_subsample`. Synoptic ODE integration
    /// cadence (hours); `1` = historical behavior.
    pub synoptic_subsample: u64,
    /// See the coarse synoptic mesh toggle in `Simulation::new`. `false`
    /// forces the identity (fine-grid) mesh, historical bit-for-bit
    /// behavior.
    pub synoptic_coarse: bool,
    /// Coarse upper layer: the ~1 km torus owns the precipitation
    /// (`atmosphere::coarse`, `MoistCoarseMode::precipitates_coarse`).
    /// `false` forces the identity mesh (`Rc = R`) and the historical
    /// fine pipeline, bit for bit.
    pub moist_coarse: bool,
    /// See `atmosphere::scaling::transport_subsample`. Horizontal transport
    /// pass cadence (hours).
    pub transport_subsample: u16,
    /// See `atmosphere::scaling::oro_subsample`. Orographic convection
    /// pump cadence (hours); `1` = historical hourly behavior.
    pub oro_subsample: u16,
    /// See `atmosphere::scaling::temp_advection_subsample`. Temperature
    /// advection gather cadence (hours); `1` = historical hourly
    /// behavior.
    pub temp_advection_subsample: u16,
    /// See `atmosphere::scaling::precip_subsample`. Precipitation pass
    /// cadence (hours); `1` = historical hourly behavior.
    pub precip_subsample: u16,
    /// See `temperature::illum_ko`. Raymarch ablation switch, perf
    /// measurement only, never active by default.
    pub illum_ko: bool,
}

impl Default for Ablation {
    /// The compiled-in configuration. Exists so `Checkpoint` can carry
    /// `#[serde(default)]` on its `ablation` field: a file saved before the
    /// field existed is read as the defaults, which is the only
    /// configuration it can have been produced under — with one dated
    /// exception. `moist_coarse` flipped to `true` on 2026-09-30, so a
    /// file from before the ablation key existed was produced on the
    /// fine path yet reads back as coarse. Accepted, because resuming a
    /// fine-path world on the coarse mode is well defined (the fine
    /// fields are the state on both modes): see
    /// `atmosphere::coarse::MOIST_COARSE_DEFAULT`'s doc, "The flip".
    fn default() -> Self {
        Self::defaults()
    }
}

impl Ablation {
    /// Compiled-in defaults, ignoring the environment entirely.
    #[must_use]
    pub fn defaults() -> Self {
        Self {
            wind_subsample: WIND_SUBSAMPLE_HOURS,
            synoptic_subsample: SYNOPTIC_SUBSAMPLE_HOURS,
            synoptic_coarse: SYNOPTIC_COARSE_DEFAULT,
            moist_coarse: MOIST_COARSE_DEFAULT,
            transport_subsample: TRANSPORT_SUBSAMPLE_HOURS,
            oro_subsample: ORO_SUBSAMPLE_HOURS,
            temp_advection_subsample: TEMP_ADVECTION_SUBSAMPLE_HOURS,
            precip_subsample: PRECIP_SUBSAMPLE_HOURS,
            illum_ko: ILLUM_KO_DEFAULT,
        }
    }

    /// Reads every switch from the environment, falling back to
    /// [`Ablation::defaults`] field by field. The only place in the crate
    /// that calls `std::env::var`.
    fn from_env() -> Self {
        let defaults = Self::defaults();
        Self {
            wind_subsample: std::env::var(ENV_WIND_SUBSAMPLE)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(defaults.wind_subsample),
            synoptic_subsample: std::env::var(ENV_SYNOPTIC_SUBSAMPLE)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(defaults.synoptic_subsample),
            synoptic_coarse: std::env::var(ENV_SYNOPTIC_COARSE)
                .map_or(defaults.synoptic_coarse, |v| {
                    v != "0" && !v.eq_ignore_ascii_case("false")
                }),
            // One variable, two modes: `0` fine, anything else the coarse
            // precipitation unit (variant M). `HEXSIM_MOIST_COARSE=2` used
            // to select a third mode (the legacy coarse stock,
            // `MoistCoarseMode::CoarseStock`, retired 2026-09-07: it
            // measured worse than M and M replaced it, see
            // `atmosphere::coarse`'s module doc). `2` is still truthy here,
            // same as `1` or any other non-`0`/`false` value, so a script
            // still setting `=2` keeps running, on M rather than erroring
            // out.
            moist_coarse: std::env::var(ENV_MOIST_COARSE).map_or(defaults.moist_coarse, |v| {
                v != "0" && !v.eq_ignore_ascii_case("false")
            }),
            transport_subsample: std::env::var(ENV_TRANSPORT_SUBSAMPLE)
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(defaults.transport_subsample),
            oro_subsample: std::env::var(ENV_ORO_SUBSAMPLE)
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(defaults.oro_subsample),
            temp_advection_subsample: std::env::var(ENV_TEMP_ADVECTION_SUBSAMPLE)
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(defaults.temp_advection_subsample),
            precip_subsample: std::env::var(ENV_PRECIP_SUBSAMPLE)
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(defaults.precip_subsample),
            illum_ko: std::env::var(ENV_ILLUM_KO).map_or(defaults.illum_ko, |v| v == "1"),
        }
    }

    /// Effective ablation config for this process: the environment is read
    /// once, on first call, and cached for every later call (including the
    /// per-switch accessors in `simulation`, `atmosphere::scaling` and
    /// `temperature`).
    #[must_use]
    pub fn effective() -> &'static Self {
        static ABLATION: OnceLock<Ablation> = OnceLock::new();
        ABLATION.get_or_init(Self::from_env)
    }

    /// The moist-layer mode this switch names
    /// (`atmosphere::MoistCoarseMode`), the form every consumer wants.
    #[must_use]
    pub fn moist_coarse_mode(&self) -> MoistCoarseMode {
        MoistCoarseMode::of(self.moist_coarse)
    }

    /// Names the fields that differ from `other`, each formatted as
    /// `"field: self=<value> other=<value>"`. Empty when the two configs
    /// match. Used to build an actionable refusal message when a
    /// checkpoint's ablation doesn't match the running process.
    #[must_use]
    pub fn differences(&self, other: &Self) -> Vec<String> {
        let mut diffs = Vec::new();
        if self.wind_subsample != other.wind_subsample {
            diffs.push(format!(
                "wind_subsample: self={} other={}",
                self.wind_subsample, other.wind_subsample
            ));
        }
        if self.synoptic_subsample != other.synoptic_subsample {
            diffs.push(format!(
                "synoptic_subsample: self={} other={}",
                self.synoptic_subsample, other.synoptic_subsample
            ));
        }
        if self.synoptic_coarse != other.synoptic_coarse {
            diffs.push(format!(
                "synoptic_coarse: self={} other={}",
                self.synoptic_coarse, other.synoptic_coarse
            ));
        }
        if self.moist_coarse != other.moist_coarse {
            diffs.push(format!(
                "moist_coarse: self={} other={}",
                self.moist_coarse, other.moist_coarse
            ));
        }
        if self.transport_subsample != other.transport_subsample {
            diffs.push(format!(
                "transport_subsample: self={} other={}",
                self.transport_subsample, other.transport_subsample
            ));
        }
        if self.oro_subsample != other.oro_subsample {
            diffs.push(format!(
                "oro_subsample: self={} other={}",
                self.oro_subsample, other.oro_subsample
            ));
        }
        if self.temp_advection_subsample != other.temp_advection_subsample {
            diffs.push(format!(
                "temp_advection_subsample: self={} other={}",
                self.temp_advection_subsample, other.temp_advection_subsample
            ));
        }
        if self.precip_subsample != other.precip_subsample {
            diffs.push(format!(
                "precip_subsample: self={} other={}",
                self.precip_subsample, other.precip_subsample
            ));
        }
        if self.illum_ko != other.illum_ko {
            diffs.push(format!(
                "illum_ko: self={} other={}",
                self.illum_ko, other.illum_ko
            ));
        }
        diffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_matches_compiled_in_constants() {
        let defaults = Ablation::defaults();
        assert_eq!(defaults.wind_subsample, WIND_SUBSAMPLE_HOURS);
        assert_eq!(defaults.synoptic_subsample, SYNOPTIC_SUBSAMPLE_HOURS);
        assert_eq!(defaults.synoptic_coarse, SYNOPTIC_COARSE_DEFAULT);
        assert_eq!(defaults.moist_coarse, MOIST_COARSE_DEFAULT);
        assert_eq!(defaults.transport_subsample, TRANSPORT_SUBSAMPLE_HOURS);
        assert_eq!(defaults.oro_subsample, ORO_SUBSAMPLE_HOURS);
        assert_eq!(
            defaults.temp_advection_subsample,
            TEMP_ADVECTION_SUBSAMPLE_HOURS
        );
        assert_eq!(defaults.precip_subsample, PRECIP_SUBSAMPLE_HOURS);
        assert_eq!(defaults.illum_ko, ILLUM_KO_DEFAULT);
    }

    #[test]
    fn differences_is_empty_for_equal_configs() {
        let a = Ablation::defaults();
        let b = a.clone();
        assert_eq!(a.differences(&b), Vec::<String>::new());
    }

    #[test]
    fn differences_names_the_diverging_field() {
        let a = Ablation::defaults();
        let mut b = a.clone();
        b.illum_ko = !b.illum_ko;
        let diffs = a.differences(&b);
        assert_eq!(diffs.len(), 1, "exactly one field diverges: {diffs:?}");
        assert!(
            diffs[0].contains("illum_ko"),
            "message must name the diverging field, got: {}",
            diffs[0]
        );
    }

    /// Coarse upper layer, step 1: `differences()` must see `moist_coarse`
    /// diverge, same contract as every other switch (a checkpoint saved
    /// under one mirror config must be refused when reloaded under
    /// another — see the module doc).
    #[test]
    fn differences_names_moist_coarse_when_it_diverges() {
        let a = Ablation::defaults();
        let mut b = a.clone();
        b.moist_coarse = !b.moist_coarse;
        let diffs = a.differences(&b);
        assert_eq!(diffs.len(), 1, "exactly one field diverges: {diffs:?}");
        assert!(
            diffs[0].contains("moist_coarse"),
            "message must name the diverging field, got: {}",
            diffs[0]
        );
    }

    #[test]
    fn differences_names_every_diverging_field() {
        let a = Ablation::defaults();
        let b = Ablation {
            wind_subsample: a.wind_subsample + 1,
            synoptic_coarse: !a.synoptic_coarse,
            ..a.clone()
        };
        let diffs = a.differences(&b);
        assert_eq!(diffs.len(), 2, "two fields diverge: {diffs:?}");
        assert!(diffs.iter().any(|d| d.contains("wind_subsample")));
        assert!(diffs.iter().any(|d| d.contains("synoptic_coarse")));
    }

    /// The checkpoint vigilance point of the 2026-09-07 effort
    /// ("settle the off switches", #159 follow-up): `Ablation` has no
    /// `#[serde(deny_unknown_fields)]`, so a checkpoint written by an
    /// engine that still had `oro_legacy_clamp`/`moist_coarse_stock`
    /// (both retired this same effort) with either forced on decodes
    /// here **without error**, the two keys silently dropped — unlike a
    /// switch that still exists and merely differs, which
    /// `Checkpoint::decode`'s `differences()` refusal would catch.
    ///
    /// Measured, not assumed: this builds the exact bytes a pre-removal
    /// engine would have written — a hand-built map rather than a
    /// `#[derive(Serialize)]` struct (which would need 5 `bool` fields to
    /// mirror the pre-removal shape, past clippy's
    /// `struct_excessive_bools`) — and decodes them with today's
    /// `Ablation`.
    ///
    /// Judged negligible, not fixed — see [`crate::checkpoint::Checkpoint`]'s
    /// `ablation` field doc for the reasoning.
    #[test]
    fn checkpoint_with_a_retired_switch_decodes_silently_as_if_it_were_absent() {
        /// One msgpack scalar, untagged so it serializes as its bare
        /// value: lets the map below mix `u64`/`u16`/`bool` fields the
        /// way the real (struct-encoded) `Ablation` map does, without a
        /// second struct carrying the same bool count as the first.
        #[derive(Serialize)]
        #[serde(untagged)]
        enum Scalar {
            U64(u64),
            U16(u16),
            Bool(bool),
        }

        let defaults = Ablation::defaults();
        // A process that had benched both retired levers at once, forced
        // on: the checkpoint the brief's vigilance point worries about.
        let old: std::collections::BTreeMap<&str, Scalar> = [
            ("wind_subsample", Scalar::U64(defaults.wind_subsample)),
            (
                "synoptic_subsample",
                Scalar::U64(defaults.synoptic_subsample),
            ),
            ("synoptic_coarse", Scalar::Bool(defaults.synoptic_coarse)),
            ("moist_coarse", Scalar::Bool(defaults.moist_coarse)),
            ("moist_coarse_stock", Scalar::Bool(true)),
            (
                "transport_subsample",
                Scalar::U16(defaults.transport_subsample),
            ),
            ("oro_subsample", Scalar::U16(defaults.oro_subsample)),
            ("oro_legacy_clamp", Scalar::Bool(true)),
            (
                "temp_advection_subsample",
                Scalar::U16(defaults.temp_advection_subsample),
            ),
            ("precip_subsample", Scalar::U16(defaults.precip_subsample)),
            ("illum_ko", Scalar::Bool(defaults.illum_ko)),
        ]
        .into_iter()
        .collect();
        let bytes = rmp_serde::to_vec_named(&old).expect("encode the pre-removal shape");

        let decoded: Ablation = rmp_serde::from_slice(&bytes).expect(
            "decode must succeed without deny_unknown_fields: this is exactly \
             the silent-drop this test measures, not a hoped-for refusal",
        );
        assert_eq!(
            decoded, defaults,
            "the two retired keys are dropped, every other field survives untouched: the file \
             reads back as plain defaults, with no trace that it was saved with the legacy pump \
             or the legacy coarse stock forced on"
        );
    }
}
