//! Vocabulary of client -> engine commands.
//!
//! A command arrives as JSON (`{"cmd": "step", "n": 30}`) and parses into
//! [`Command`]: this is where, and nowhere else, the command names, field
//! names, and **bounds** (`n` max, `speed` clamped) live. The WebSocket
//! server and the WASM module parse with this same code, so they cannot
//! diverge on what counts as a valid command.
//!
//! Parsing is **silent**: it logs nothing and has no side effects.
//! Discrepancies (missing field, out-of-bounds value) bubble up to the
//! shell, which decides how to log them: `tracing` on the server side,
//! console in the browser.

use std::ops::RangeInclusive;

use hexsim_core::coord::HexCoord;
use serde_json::Value;
use thiserror::Error;

/// Upper bound of `step_hour`: 8760 h = 1 year, hourly.
pub const MAX_STEP_HOURS: u64 = 8760;
/// Upper bound of `step`: 3650 days = 10 years.
pub const MAX_STEP_DAYS: u64 = 3650;
/// Minimum auto-tick period (ms), below this the server saturates the
/// channel.
pub const MIN_TICK_MS: u64 = 1;
/// Maximum auto-tick period (ms).
pub const MAX_TICK_MS: u64 = 2000;
/// Maximum radius of a `region` query, in cells.
pub const MAX_REGION_RADIUS: i32 = 50;
/// Accepted range for an explicit `radius` on `reset`. Below 5, the coarse
/// synoptic torus degenerates (the micro-test rule "no transport at radius
/// 0" generalizes: too few cells and a hex is effectively its own
/// neighbor); above 300, we're past the r250 target that the perf work
/// (#63, JOURNAL 2026-09-06) already treats as its ceiling.
pub const RESET_RADIUS_RANGE: RangeInclusive<i32> = 5..=300;

/// A validated client command. Bounds have already been applied: a
/// constructed [`Command`] is executable as-is.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Resumes auto-tick.
    Play,
    /// Suspends auto-tick.
    Pause,
    /// Changes the auto-tick period.
    ///
    /// `tick_ms` is already bounded to `[MIN_TICK_MS, MAX_TICK_MS]`;
    /// `requested` keeps the client's raw value so the shell can report
    /// the discrepancy (the clamp itself is silent).
    Speed { tick_ms: u64, requested: u64 },
    /// Advances by `n` units: hours if `hourly`, days otherwise.
    Step { n: u64, hourly: bool },
    /// Regenerates the world. `seed: None` = reuse the current seed,
    /// `radius: None` = keep the current radius (today's behaviour,
    /// byte-for-byte). An explicit `radius` is already bounded to
    /// [`RESET_RADIUS_RANGE`]: whether it can actually be honored (a DEM
    /// world locks its radius) is a `World::apply` concern, not parsing's.
    Reset {
        seed: Option<u32>,
        radius: Option<i32>,
    },
    /// Hot-reloads a parameter (`"atmosphere.condensation_rate"`).
    SetParam { key: String, value: f32 },
    /// Read-only, never mutates the world.
    Query(Query),
}

/// Read commands. Response targeted to the requester, never broadcast.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Query {
    Diagnostics,
    Climate,
    Cell { coord: HexCoord },
    Region { center: HexCoord, radius: i32 },
    Params,
    Meta,
}

/// Why a client payload is not a command.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("invalid JSON")]
    InvalidJson,
    #[error("missing `cmd` field")]
    MissingCmd,
    #[error("unknown command: `{0}`")]
    UnknownCommand(String),
    #[error("command `{cmd}`: missing or invalid field `{field}`")]
    BadField {
        cmd: &'static str,
        field: &'static str,
    },
}

impl Command {
    /// Parses a JSON text payload.
    ///
    /// # Errors
    ///
    /// [`ParseError`] if the JSON is unreadable, if `cmd` is missing or
    /// unknown, or if a required field is absent/out of bounds.
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let value: Value = serde_json::from_str(text).map_err(|_| ParseError::InvalidJson)?;
        Self::from_value(&value)
    }

    /// Variant of [`Command::parse`] for JSON that's already deserialized.
    ///
    /// # Errors
    ///
    /// See [`Command::parse`].
    pub fn from_value(value: &Value) -> Result<Self, ParseError> {
        let name = value
            .get("cmd")
            .and_then(Value::as_str)
            .ok_or(ParseError::MissingCmd)?;

        match name {
            "play" => Ok(Self::Play),
            "pause" => Ok(Self::Pause),
            "speed" => {
                let requested =
                    value
                        .get("value")
                        .and_then(Value::as_u64)
                        .ok_or(ParseError::BadField {
                            cmd: "speed",
                            field: "value",
                        })?;
                Ok(Self::Speed {
                    tick_ms: requested.clamp(MIN_TICK_MS, MAX_TICK_MS),
                    requested,
                })
            }
            // `n` absent = 1 step, the common use case from the front-end
            // (the "+1" button). An outlandish value is clamped to the
            // bound rather than rejected: the client gets a big step, not
            // an error.
            "step_hour" => Ok(Self::Step {
                n: step_count(value, MAX_STEP_HOURS),
                hourly: true,
            }),
            "step" => Ok(Self::Step {
                n: step_count(value, MAX_STEP_DAYS),
                hourly: false,
            }),
            // Seed out of u32 bounds: rejected rather than silently
            // truncated (a truncated seed gives a different world than the
            // one requested).
            "reset" => {
                let seed = match value.get("seed") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(v.as_u64().and_then(|s| u32::try_from(s).ok()).ok_or(
                        ParseError::BadField {
                            cmd: "reset",
                            field: "seed",
                        },
                    )?),
                };
                // Same rationale as the seed: an out-of-range radius is
                // rejected outright rather than clamped, silently shrinking
                // a request for r400 down to r300 gives a world the client
                // didn't ask for.
                let radius = match value.get("radius") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(
                        v.as_i64()
                            .and_then(|r| i32::try_from(r).ok())
                            .filter(|r| RESET_RADIUS_RANGE.contains(r))
                            .ok_or(ParseError::BadField {
                                cmd: "reset",
                                field: "radius",
                            })?,
                    ),
                };
                Ok(Self::Reset { seed, radius })
            }
            "set_param" => {
                let key = value
                    .get("key")
                    .and_then(Value::as_str)
                    .ok_or(ParseError::BadField {
                        cmd: "set_param",
                        field: "key",
                    })?;
                // Deserialized directly as f32 (simulation params are f32)
                // rather than `as_f64() as f32`: the conversion is serde's,
                // not a hand-written truncating cast.
                let value_f32 = value
                    .get("value")
                    .and_then(|v| serde_json::from_value::<f32>(v.clone()).ok())
                    .ok_or(ParseError::BadField {
                        cmd: "set_param",
                        field: "value",
                    })?;
                Ok(Self::SetParam {
                    key: key.to_owned(),
                    value: value_f32,
                })
            }
            "diagnostics" => Ok(Self::Query(Query::Diagnostics)),
            "climate" => Ok(Self::Query(Query::Climate)),
            "params" => Ok(Self::Query(Query::Params)),
            "meta" => Ok(Self::Query(Query::Meta)),
            "cell" => {
                let coord = parse_coord(value).ok_or(ParseError::BadField {
                    cmd: "cell",
                    field: "q/r",
                })?;
                Ok(Self::Query(Query::Cell { coord }))
            }
            "region" => {
                let center = parse_coord(value).ok_or(ParseError::BadField {
                    cmd: "region",
                    field: "q/r",
                })?;
                let radius = value
                    .get("radius")
                    .and_then(Value::as_i64)
                    .unwrap_or(3)
                    .clamp(0, i64::from(MAX_REGION_RADIUS));
                let radius = i32::try_from(radius).unwrap_or(MAX_REGION_RADIUS);
                Ok(Self::Query(Query::Region { center, radius }))
            }
            other => Err(ParseError::UnknownCommand(other.to_owned())),
        }
    }

    /// `true` if the command cannot modify the world.
    #[must_use]
    pub fn is_query(&self) -> bool {
        matches!(self, Self::Query(_))
    }
}

/// Number of steps of a `step`/`step_hour`: default 1, bounded to `max`.
fn step_count(value: &Value, max: u64) -> u64 {
    value.get("n").and_then(Value::as_u64).unwrap_or(1).min(max)
}

/// Reads the axial coordinates `q`/`r` of a command.
fn parse_coord(value: &Value) -> Option<HexCoord> {
    let q = i32::try_from(value.get("q").and_then(Value::as_i64)?).ok()?;
    let r = i32::try_from(value.get("r").and_then(Value::as_i64)?).ok()?;
    Some(HexCoord::new(q, r))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_pause_without_a_field() {
        assert_eq!(Command::parse(r#"{"cmd":"play"}"#), Ok(Command::Play));
        assert_eq!(Command::parse(r#"{"cmd":"pause"}"#), Ok(Command::Pause));
    }

    /// The `speed` bound lives in the protocol, not in the server: the WASM
    /// build is subject to it identically. `requested` survives the clamp
    /// so the shell can report it.
    #[test]
    fn speed_is_clamped_but_raw_value_is_kept() {
        assert_eq!(
            Command::parse(r#"{"cmd":"speed","value":99999}"#),
            Ok(Command::Speed {
                tick_ms: MAX_TICK_MS,
                requested: 99999
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"speed","value":0}"#),
            Ok(Command::Speed {
                tick_ms: MIN_TICK_MS,
                requested: 0
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"speed","value":30}"#),
            Ok(Command::Speed {
                tick_ms: 30,
                requested: 30
            })
        );
    }

    #[test]
    fn step_defaults_to_one_and_is_bounded() {
        assert_eq!(
            Command::parse(r#"{"cmd":"step_hour"}"#),
            Ok(Command::Step { n: 1, hourly: true })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"step_hour","n":100000}"#),
            Ok(Command::Step {
                n: MAX_STEP_HOURS,
                hourly: true
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"step","n":100000}"#),
            Ok(Command::Step {
                n: MAX_STEP_DAYS,
                hourly: false
            })
        );
    }

    /// A seed that doesn't fit in a `u32` would give a different world if
    /// truncated, so it's rejected.
    #[test]
    fn reset_seed_is_optional_and_bounded() {
        assert_eq!(
            Command::parse(r#"{"cmd":"reset"}"#),
            Ok(Command::Reset {
                seed: None,
                radius: None
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","seed":42}"#),
            Ok(Command::Reset {
                seed: Some(42),
                radius: None
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","seed":4294967296}"#),
            Err(ParseError::BadField {
                cmd: "reset",
                field: "seed"
            })
        );
    }

    /// `radius` is optional (absent = keep the current radius, today's
    /// behaviour byte-for-byte) and bounded to [`RESET_RADIUS_RANGE`]: an
    /// out-of-range value is rejected, not clamped, same rationale as the
    /// seed above.
    #[test]
    fn reset_radius_is_optional_and_bounded() {
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","radius":30}"#),
            Ok(Command::Reset {
                seed: None,
                radius: Some(30)
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","seed":7,"radius":120}"#),
            Ok(Command::Reset {
                seed: Some(7),
                radius: Some(120)
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","radius":5}"#),
            Ok(Command::Reset {
                seed: None,
                radius: Some(5)
            }),
            "the lower bound itself is accepted"
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","radius":300}"#),
            Ok(Command::Reset {
                seed: None,
                radius: Some(300)
            }),
            "the upper bound itself is accepted"
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","radius":4}"#),
            Err(ParseError::BadField {
                cmd: "reset",
                field: "radius"
            }),
            "below RESET_RADIUS_RANGE"
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","radius":301}"#),
            Err(ParseError::BadField {
                cmd: "reset",
                field: "radius"
            }),
            "above RESET_RADIUS_RANGE"
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"reset","radius":-5}"#),
            Err(ParseError::BadField {
                cmd: "reset",
                field: "radius"
            }),
            "negative radius"
        );
    }

    #[test]
    fn set_param_requires_key_and_value() {
        assert_eq!(
            Command::parse(
                r#"{"cmd":"set_param","key":"atmosphere.condensation_rate","value":0.02}"#
            ),
            Ok(Command::SetParam {
                key: "atmosphere.condensation_rate".to_owned(),
                value: 0.02
            })
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"set_param","key":"a.b"}"#),
            Err(ParseError::BadField {
                cmd: "set_param",
                field: "value"
            })
        );
    }

    #[test]
    fn queries_reconnues() {
        for (json, expected) in [
            (r#"{"cmd":"diagnostics"}"#, Query::Diagnostics),
            (r#"{"cmd":"climate"}"#, Query::Climate),
            (r#"{"cmd":"params"}"#, Query::Params),
            (r#"{"cmd":"meta"}"#, Query::Meta),
        ] {
            let cmd = Command::parse(json).expect("query valide");
            assert_eq!(cmd, Command::Query(expected));
            assert!(cmd.is_query());
        }
    }

    #[test]
    fn cell_and_region_read_coordinates() {
        assert_eq!(
            Command::parse(r#"{"cmd":"cell","q":3,"r":-7}"#),
            Ok(Command::Query(Query::Cell {
                coord: HexCoord::new(3, -7)
            }))
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"region","q":0,"r":0}"#),
            Ok(Command::Query(Query::Region {
                center: HexCoord::new(0, 0),
                radius: 3
            }))
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"region","q":0,"r":0,"radius":9000}"#),
            Ok(Command::Query(Query::Region {
                center: HexCoord::new(0, 0),
                radius: MAX_REGION_RADIUS
            }))
        );
        assert_eq!(
            Command::parse(r#"{"cmd":"cell"}"#),
            Err(ParseError::BadField {
                cmd: "cell",
                field: "q/r"
            })
        );
    }

    #[test]
    fn payloads_invalides() {
        assert_eq!(Command::parse("not json"), Err(ParseError::InvalidJson));
        assert_eq!(Command::parse("{}"), Err(ParseError::MissingCmd));
        assert_eq!(
            Command::parse(r#"{"cmd":"danser"}"#),
            Err(ParseError::UnknownCommand("danser".to_owned()))
        );
    }
}
