/// hexsim-ctl, WebSocket client to control hexsim from the terminal.
///
/// Usage: hexsim-ctl <COMMAND> [ARGS...]
///   play                     → runs the simulation continuously
///   pause                    → pauses it
///   step [N]                 → advances N ticks (default: 1)
///   step-hour [N]            → advances N hours (diurnal cycle, synoptic drift)
///   reset [SEED]             → resets (seed optional)
///   diag                     → current diagnostics
///   climate                  → climate aggregates
///   param <KEY> <VALUE>      → hot-changes a parameter
use std::process;

use anyhow::Result;
use clap::{Parser, Subcommand};
use hexsim_wsclient::{DEFAULT_WS_URL, WsClient};

#[derive(Parser)]
#[command(name = "hexsim-ctl", about = "Pilote hexsim via WebSocket")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the simulation continuously
    Play,
    /// Pause it
    Pause,
    /// Advance N ticks (default: 1)
    Step {
        #[arg(default_value = "1")]
        n: u64,
    },
    /// Advance N hours (default: 1), to observe the diurnal cycle or the
    /// synoptic drift below the day scale
    StepHour {
        #[arg(default_value = "1")]
        n: u64,
    },
    /// Reset (seed optional)
    Reset { seed: Option<u32> },
    /// Current diagnostics
    Diag,
    /// Climate aggregates
    Climate,
    /// Hot-change a parameter
    Param { key: String, value: f64 },
    /// Snapshot of a hexagonal region centered on (q,r) with a radius
    Region {
        q: i32,
        r: i32,
        #[arg(default_value = "15")]
        radius: i32,
    },
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Erreur : {e}");
        process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    let ws = WsClient::connect(DEFAULT_WS_URL).await?;

    match cli.cmd {
        // Fire-and-forget commands (no response expected)
        Cmd::Play => {
            ws.send(serde_json::json!({"cmd": "play"})).await?;
            eprintln!("OK");
        }
        Cmd::Pause => {
            ws.send(serde_json::json!({"cmd": "pause"})).await?;
            eprintln!("OK");
        }
        Cmd::Param { key, value } => {
            ws.send(serde_json::json!({"cmd": "set_param", "key": key, "value": value}))
                .await?;
            eprintln!("OK");
        }

        // Commands with a tagged response
        Cmd::Diag => {
            let response = ws
                .request(serde_json::json!({"cmd": "diagnostics"}), "diagnostics")
                .await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Cmd::Climate => {
            let response = ws
                .request(serde_json::json!({"cmd": "climate"}), "climate")
                .await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Cmd::Region { q, r, radius } => {
            let response = ws
                .request(
                    serde_json::json!({"cmd": "region", "q": q, "r": r, "radius": radius}),
                    "region",
                )
                .await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        // step / reset: sends the command, then requests diagnostics
        Cmd::StepHour { n } => {
            ws.send(serde_json::json!({"cmd": "step_hour", "n": n}))
                .await?;
            let response = ws
                .request(serde_json::json!({"cmd": "diagnostics"}), "diagnostics")
                .await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Cmd::Step { n } => {
            ws.send(serde_json::json!({"cmd": "step", "n": n})).await?;
            let response = ws
                .request(serde_json::json!({"cmd": "diagnostics"}), "diagnostics")
                .await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Cmd::Reset { seed } => {
            let cmd = match seed {
                Some(s) => serde_json::json!({"cmd": "reset", "seed": s}),
                None => serde_json::json!({"cmd": "reset"}),
            };
            ws.send(cmd).await?;
            // Let the reset take effect, then request diagnostics
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            let response = ws
                .request(serde_json::json!({"cmd": "diagnostics"}), "diagnostics")
                .await?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
    }

    Ok(())
}
