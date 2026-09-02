//! Startup configuration read from TOML file.
//!
//! Scope intentionally restricted to *bootstrap* parameters, those set once at
//! world generation: server port, grid radius, seed. Physics parameters
//! (`temperature.base_temp`, `atmosphere.evap_rate`, …) don't go here, stay
//! hot-tunable via `update_param` / front sliders.
//!
//! Resolution priority: **defaults < file < environment variables**. So
//! `HEXSIM_RADIUS=200 just run` still overrides file for quick iteration
//! without editing `hexsim.toml`.
//!
//! File search: `$HEXSIM_CONFIG` if set, else first existing among `hexsim.toml`
//! (cwd) and `../hexsim.toml` (server runs from `simulation/`, so `../` = repo
//! root). Missing => defaults.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

/// Default server port. 8355 = "TELL" on phone keyboard, nod to `tellhex` repo.
/// Server binds dual-stack `[::]:{port}`.
pub const DEFAULT_PORT: u16 = 8355;
/// Default grid radius (~6211 hex).
pub const DEFAULT_RADIUS: i32 = 45;

/// Typed mirror of `hexsim.toml` file. Each field optional: missing section or
/// key falls back to default/env. `deny_unknown_fields` turns typo (`prot = …`)
/// into startup error rather than silently ignored setting.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub world: WorldConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub port: Option<u16>,
    pub radius: Option<i32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldConfig {
    pub seed: Option<u32>,
}

/// Resolved startup configuration: no more `Option` on port/radius (they
/// have a default), `seed` stays optional since its default lives in
/// `TerrainParams::default()`, `None` means "do not override".
#[derive(Debug)]
pub struct BootConfig {
    pub port: u16,
    pub radius: i32,
    pub seed: Option<u32>,
    /// JSON path to an external elevation override (real DEM), env var
    /// only, a validation tool (`scripts/dem_import/`), not a durable
    /// playable-world parameter. See `terrain::apply_dem_override`.
    pub dem_override_path: Option<String>,
}

/// Reads an integer from an environment variable, silently ignoring
/// a missing or unparsable value (falls back to the next priority level).
fn env_parse<T: FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Locates the config file, or `None` if no candidate exists.
fn config_path() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("HEXSIM_CONFIG") {
        return Some(PathBuf::from(explicit));
    }
    ["hexsim.toml", "../hexsim.toml"]
        .into_iter()
        .map(Path::new)
        .find(|p| p.exists())
        .map(Path::to_path_buf)
}

/// Loads the file (if it exists) and applies the defaults < file < env priority.
///
/// # Errors
/// Returns an error if the file exists but is unreadable or malformed,
/// since a loud failure at startup beats a silently ignored config.
pub fn resolve() -> Result<BootConfig> {
    let file = if let Some(path) = config_path() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: FileConfig =
            toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))?;
        info!(config = %path.display(), "configuration loaded");
        cfg
    } else {
        info!("no config file (hexsim.toml), using defaults");
        FileConfig::default()
    };

    Ok(BootConfig {
        port: env_parse("HEXSIM_PORT")
            .or(file.server.port)
            .unwrap_or(DEFAULT_PORT),
        radius: env_parse("HEXSIM_RADIUS")
            .or(file.server.radius)
            .unwrap_or(DEFAULT_RADIUS),
        seed: env_parse("HEXSIM_SEED").or(file.world.seed),
        dem_override_path: env_parse("HEXSIM_DEM_OVERRIDE"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_yields_defaults() {
        let cfg: FileConfig = toml::from_str("").unwrap();
        assert_eq!(cfg.server.port, None);
        assert_eq!(cfg.server.radius, None);
        assert_eq!(cfg.world.seed, None);
    }

    #[test]
    fn parses_full_file() {
        let cfg: FileConfig =
            toml::from_str("[server]\nport = 9000\nradius = 200\n[world]\nseed = 7\n").unwrap();
        assert_eq!(cfg.server.port, Some(9000));
        assert_eq!(cfg.server.radius, Some(200));
        assert_eq!(cfg.world.seed, Some(7));
    }

    #[test]
    fn unknown_field_is_rejected() {
        let err = toml::from_str::<FileConfig>("[server]\nprot = 9000\n");
        assert!(err.is_err(), "an unknown key must be rejected");
    }

    #[test]
    fn partial_file_leaves_other_fields_none() {
        let cfg: FileConfig = toml::from_str("[server]\nradius = 120\n").unwrap();
        assert_eq!(cfg.server.radius, Some(120));
        assert_eq!(cfg.server.port, None);
        assert_eq!(cfg.world.seed, None);
    }
}
