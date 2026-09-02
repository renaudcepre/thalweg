//! Centralizes the environment-variable ablation switches (A/B knobs) read
//! by the engine.
//!
//! # Why
//! Each switch is a legitimate, measured A/B lever (see the doc-comments at
//! their call sites: `simulation::wind_subsample`, `simulation::synoptic_subsample`,
//! the coarse synoptic mesh toggle in `Simulation::new`,
//! `atmosphere::scaling::transport_subsample`, `temperature::illum_ko`).
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

use crate::atmosphere::TRANSPORT_SUBSAMPLE_HOURS;
use crate::simulation::{SYNOPTIC_COARSE_DEFAULT, SYNOPTIC_SUBSAMPLE_HOURS, WIND_SUBSAMPLE_HOURS};
use crate::temperature::ILLUM_KO_DEFAULT;

/// Environment variable names, one per switch. Read exclusively by
/// [`Ablation::from_env`], the only function in the crate calling
/// `std::env::var`.
const ENV_WIND_SUBSAMPLE: &str = "HEXSIM_WIND_SUBSAMPLE";
const ENV_SYNOPTIC_SUBSAMPLE: &str = "HEXSIM_SYNOPTIC_SUBSAMPLE";
const ENV_SYNOPTIC_COARSE: &str = "HEXSIM_SYNOPTIC_COARSE";
const ENV_TRANSPORT_SUBSAMPLE: &str = "HEXSIM_TRANSPORT_SUBSAMPLE";
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
    /// See `atmosphere::scaling::transport_subsample`. Horizontal transport
    /// pass cadence (hours).
    pub transport_subsample: u16,
    /// See `temperature::illum_ko`. Raymarch ablation switch, perf
    /// measurement only, never active by default.
    pub illum_ko: bool,
}

impl Default for Ablation {
    /// The compiled-in configuration. Exists so `Checkpoint` can carry
    /// `#[serde(default)]` on its `ablation` field: a file saved before the
    /// field existed is read as the defaults, which is the only
    /// configuration it can have been produced under.
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
            transport_subsample: TRANSPORT_SUBSAMPLE_HOURS,
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
            transport_subsample: std::env::var(ENV_TRANSPORT_SUBSAMPLE)
                .ok()
                .and_then(|v| v.parse::<u16>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(defaults.transport_subsample),
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
        if self.transport_subsample != other.transport_subsample {
            diffs.push(format!(
                "transport_subsample: self={} other={}",
                self.transport_subsample, other.transport_subsample
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
        assert_eq!(defaults.transport_subsample, TRANSPORT_SUBSAMPLE_HOURS);
        assert_eq!(defaults.illum_ko, ILLUM_KO_DEFAULT);
    }

    #[test]
    fn differences_is_empty_for_equal_configs() {
        let a = Ablation::defaults();
        let b = a.clone();
        assert!(a.differences(&b).is_empty());
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
}
