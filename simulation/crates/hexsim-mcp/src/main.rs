//! MCP stdio <-> WS bridge for the simulation.
//!
//! `clippy::unused_async_trait_impl` is allowed crate-wide: it fires on the
//! `ServerHandler` impl that rmcp's `#[tool_router]` macro emits as a sibling
//! item, so no attribute on our own code can reach it, and there is no line of
//! ours to correct. rmcp is pinned at 1.5 while 3.x is out; revisit on upgrade.
#![allow(clippy::unused_async_trait_impl)]

use anyhow::Result;
use hexsim_wsclient::WsClient;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{ServiceExt, tool, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// --- Tool parameter structs ---

#[derive(Serialize, Deserialize, JsonSchema)]
struct StepParams {
    /// Number of ticks (default: 1, max: 3650)
    n: Option<u64>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct ResetParams {
    /// Random seed (optional)
    seed: Option<u32>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct SetParamParams {
    /// Parameter path (e.g. `atmosphere.evap_rate`)
    key: String,
    /// New value
    value: f64,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct CellParams {
    /// Axial coordinate q
    q: i32,
    /// Axial coordinate r
    r: i32,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct RegionParams {
    /// Axial coordinate q of center
    q: i32,
    /// Axial coordinate r of center
    r: i32,
    /// Hex radius around center (default: 3, max: 15)
    radius: Option<i32>,
}

// --- MCP Server ---

#[derive(Clone)]
struct SimServer {
    ws: Arc<WsClient>,
}

#[tool_router(server_handler)]
impl SimServer {
    #[tool(description = "Advance simulation N ticks and return diagnostics")]
    async fn sim_step(&self, p: Parameters<StepParams>) -> String {
        let n = p.0.n.unwrap_or(1).min(3650);
        let _ = self
            .ws
            .send(serde_json::json!({"cmd": "step", "n": n}))
            .await
            .map_err(|e| format!("Error: {e}"));
        match self
            .ws
            .request(serde_json::json!({"cmd": "diagnostics"}), "diagnostics")
            .await
        {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Run simulation continuously")]
    async fn sim_play(&self) -> String {
        match self.ws.send(serde_json::json!({"cmd": "play"})).await {
            Ok(()) => "Simulation playing".to_string(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Pause simulation")]
    async fn sim_pause(&self) -> String {
        match self.ws.send(serde_json::json!({"cmd": "pause"})).await {
            Ok(()) => "Simulation paused".to_string(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Reset simulation (optional seed)")]
    async fn sim_reset(&self, p: Parameters<ResetParams>) -> String {
        let cmd = match p.0.seed {
            Some(s) => serde_json::json!({"cmd": "reset", "seed": s}),
            None => serde_json::json!({"cmd": "reset"}),
        };
        let _ = self.ws.send(cmd).await.map_err(|e| format!("Error: {e}"));
        match self
            .ws
            .request(serde_json::json!({"cmd": "diagnostics"}), "diagnostics")
            .await
        {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Get current simulation diagnostics")]
    async fn sim_diagnostics(&self) -> String {
        match self
            .ws
            .request(serde_json::json!({"cmd": "diagnostics"}), "diagnostics")
            .await
        {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Get climate aggregates by altitude band (30d/180d/365d)")]
    async fn sim_climate(&self) -> String {
        match self
            .ws
            .request(serde_json::json!({"cmd": "climate"}), "climate")
            .await
        {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Full cell state (q, r): properties, wind, fluxes, neighbors")]
    async fn sim_cell(&self, p: Parameters<CellParams>) -> String {
        match self
            .ws
            .request(
                serde_json::json!({"cmd": "cell", "q": p.0.q, "r": p.0.r}),
                "cell",
            )
            .await
        {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Compact list of cells in radius around (q, r)")]
    async fn sim_region(&self, p: Parameters<RegionParams>) -> String {
        let radius = p.0.radius.unwrap_or(3).clamp(0, 15);
        match self
            .ws
            .request(
                serde_json::json!({"cmd": "region", "q": p.0.q, "r": p.0.r, "radius": radius}),
                "region",
            )
            .await
        {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Read current simulation parameters (all groups)")]
    async fn sim_params(&self) -> String {
        match self
            .ws
            .request(serde_json::json!({"cmd": "params"}), "params")
            .await
        {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("Error: {e}"),
        }
    }

    #[tool(description = "Change a parameter live (e.g. atmosphere.evap_rate 0.05)")]
    async fn sim_set_param(&self, p: Parameters<SetParamParams>) -> String {
        match self
            .ws
            .send(serde_json::json!({"cmd": "set_param", "key": p.0.key, "value": p.0.value}))
            .await
        {
            Ok(()) => format!("Parameter {} = {} applied", p.0.key, p.0.value),
            Err(e) => format!("Error: {e}"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    use hexsim_wsclient::DEFAULT_WS_URL;

    let url = DEFAULT_WS_URL;
    let ws = WsClient::connect(url)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let ws = Arc::new(ws);

    let server = SimServer { ws };

    eprintln!("hexsim-mcp connected to {url}");
    let peer = server
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    peer.waiting().await?;

    Ok(())
}
