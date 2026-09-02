//! `HexSim` server: WebSocket + static file serving for the front end.
//!
//! This binary is a **transport shell**. It doesn't know what a valid
//! command is or what a response looks like, that's `hexsim-proto`. Here
//! we open sockets, hold a lock, broadcast frames and log. The WASM module
//! is the other shell of the same protocol, it does the same job with a
//! Web Worker instead of tokio.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::{
    Router,
    extract::{
        DefaultBodyLimit, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::{StatusCode, header},
    response::IntoResponse,
    routing::get,
};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{Mutex, broadcast, mpsc};
use tower_http::services::ServeDir;
use tracing::{debug, info, warn};

use hexsim_core::dynamics::CELL_SPACING_M;
use hexsim_core::grid::HexGrid;
use hexsim_core::simulation::Simulation;
use hexsim_core::terrain::{DemOverrideFile, TerrainParams, apply_dem_override, generate_terrain};
use hexsim_proto::command::Command;
use hexsim_proto::query::BuildInfo;
use hexsim_proto::world::{Outcome, Schedule, World, batch_stride};

mod config;
mod log_bridge;
mod perf;

// Shared state between the tick loop, the WS handlers, and the controls.
//
// AtomicBool/AtomicU64 = atomic variables, no need for a Mutex to read or
// write them. It's like a volatile in C but with formal memory guarantees.
// Ordering::Relaxed is enough here since we don't need to synchronize other
// data with these flags.
struct AppState {
    /// The simulated world and its protocol. Everything that is *physical*
    /// lives behind this lock; what surrounds it below is scheduling,
    /// specific to this shell.
    world: Mutex<World>,
    /// Binary msgpack frames (`wire` snapshots, perf), broadcast to all
    /// clients. Targeted responses stay in text JSON.
    tx: broadcast::Sender<Bytes>,
    /// Text JSON broadcast to everyone: tracing events (via `WsLogLayer`)
    /// and `params` notifications post-`set_param` (#2, UI sync).
    text_tx: broadcast::Sender<String>,
    paused: AtomicBool,
    tick_ms: AtomicU64,
    next_client_id: AtomicU64,
    last_tick_us: AtomicU64,
    last_tick_num: AtomicU64,
    /// `true` while a multi-tick `step`/`step_hour` is in progress. Used to
    /// (1) prevent the auto-tick from doubling the advance if the world was
    /// already running, (2) expose a clear "busy" state. The job releases
    /// the lock between each batch, so this flag is the single source of
    /// truth on whether a long computation is active.
    stepping: AtomicBool,
}

/// Identity of this build, surfaced by the `meta` query. Injected by
/// `build.rs`; the WASM module will supply its own.
fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        hash: env!("HEXSIM_BUILD_HASH"),
        unix: env!("HEXSIM_BUILD_UNIX").parse().unwrap_or(0),
    }
}

/// Loads an external DEM survey onto the grid and **makes loud** anything
/// that, if silent, would pass for a complete world: cells falling outside
/// the domain, radius or spacing different from the engine's, missing
/// provenance.
fn load_dem_override(
    path: &str,
    grid: &mut HexGrid,
    terrain_params: &TerrainParams,
) -> anyhow::Result<()> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading DEM override {path}"))?;
    let file: DemOverrideFile =
        serde_json::from_str(&text).with_context(|| format!("parsing DEM override {path}"))?;

    if let Some(meta) = file.meta() {
        info!(
            path,
            center_lat = meta.center_lat,
            center_lon = meta.center_lon,
            radius = meta.radius,
            cell_spacing_m = meta.cell_spacing_m,
            dataset = meta.dataset,
            "DEM override provenance"
        );
        if meta.radius != grid.radius() {
            warn!(
                dem_radius = meta.radius,
                grid_radius = grid.radius(),
                "DEM and grid radii differ: the world will be a crop of the survey"
            );
        }
        if (meta.cell_spacing_m - CELL_SPACING_M).abs() > 1e-3 {
            warn!(
                dem_cell_spacing_m = meta.cell_spacing_m,
                engine_cell_spacing_m = CELL_SPACING_M,
                "DEM sampled at a different spacing than the engine: relief stretched or compressed"
            );
        }
    } else {
        warn!(
            path,
            "DEM override without provenance (legacy format): center and spacing unknown"
        );
    }

    let report = apply_dem_override(grid, file.cells(), terrain_params);
    if report.skipped > 0 {
        warn!(
            applied = report.applied,
            skipped = report.skipped,
            grid_radius = grid.radius(),
            "DEM cells outside domain ignored: loaded relief is truncated"
        );
    }
    info!(
        path,
        cells = report.applied,
        "relief DEM applique (validation)"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Broadcast channel shared between the tracing Layer and the WS handlers.
    // Each tracing event => a JSON push here => forwarded to every client.
    // Also used for `params` notifications post-set_param (#2).
    let (text_tx, _) = broadcast::channel::<String>(256);
    init_tracing(text_tx.clone());

    // Startup config (port, radius, seed): defaults < hexsim.toml < env.
    // `HEXSIM_RADIUS=200 just run` therefore always overrides the file, for
    // quick iteration. See `config.rs`.
    let boot = config::resolve()?;
    let mut terrain_params = TerrainParams::default();
    if let Some(seed) = boot.seed {
        terrain_params.seed = seed;
    }

    // The grid is built here rather than by `World::generate`: the DEM
    // override must be applied between terrain generation and simulation
    // creation.
    let mut grid = HexGrid::from_radius(boot.radius);
    generate_terrain(&mut grid, &terrain_params);
    info!(
        cells = grid.len(),
        seed = terrain_params.seed,
        "grille generee"
    );

    if let Some(path) = &boot.dem_override_path {
        load_dem_override(path, &mut grid, &terrain_params)?;
    }

    let world = World::from_grid(grid, boot.radius, terrain_params, build_info());
    let (tx, _) = broadcast::channel(16);

    let state = Arc::new(AppState {
        world: Mutex::new(world),
        tx,
        text_tx,
        paused: AtomicBool::new(true),
        // 30 ms by default since v0.4.1: the tick went from ~140 ms to ~42 ms
        // at radius 30 (issue #41), the 100 ms throttle had become the
        // limiting factor for the play feel. 30 ms = ~24 sim-days/sec under
        // CPU-bound conditions.
        tick_ms: AtomicU64::new(30),
        next_client_id: AtomicU64::new(1),
        last_tick_us: AtomicU64::new(0),
        last_tick_num: AtomicU64::new(0),
        stepping: AtomicBool::new(false),
    });

    spawn_tick_loop(Arc::clone(&state));
    perf::spawn_perf_task(Arc::clone(&state));

    let frontend_dir = std::env::current_dir()?.join("../frontend");
    let app = Router::new()
        .route("/ws", get(ws_handler))
        // Export/import of complete state (checkpoint) over HTTP: natural
        // transport for a binary file (download via Content-Disposition,
        // upload as request body). Body limit raised for large worlds.
        .route(
            "/checkpoint",
            get(export_checkpoint)
                .post(import_checkpoint)
                .layer(DefaultBodyLimit::max(256 * 1024 * 1024)),
        )
        .fallback_service(ServeDir::new(frontend_dir))
        .with_state(state);

    // Dual-stack bind (`[::]` = IPv6 unspecified, IPV6_V6ONLY=0 by default on
    // macOS/Linux): serves IPv4 and IPv6 so `localhost` works whichever way
    // it resolves (`::1` takes priority on macOS). Pure IPv4 fallback for
    // environments without IPv6 (containers/CI: binding `[::]` → EAFNOSUPPORT).
    let addr = format!("[::]:{}", boot.port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            let v4 = format!("0.0.0.0:{}", boot.port);
            warn!(error = %e, addr = %addr, fallback = %v4, "bind IPv6 impossible, repli IPv4");
            tokio::net::TcpListener::bind(&v4).await?
        }
    };
    info!(
        addr = %listener.local_addr()?,
        version = env!("CARGO_PKG_VERSION"),
        build_hash = env!("HEXSIM_BUILD_HASH"),
        "serveur demarre"
    );
    axum::serve(listener, app).await?;

    Ok(())
}

/// Auto-tick loop: advances the sim by 1 hour per cycle and broadcasts the
/// snapshot. Detached into a tokio task (mirror pattern of
/// `perf::spawn_perf_task`), only captures the shared state.
fn spawn_tick_loop(state: Arc<AppState>) {
    tokio::spawn(async move {
        loop {
            let ms = state.tick_ms.load(Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(ms)).await;

            // During a step/step_hour job (`stepping`), the job is already
            // advancing the sim batch by batch: the auto-tick must stand
            // down, otherwise it doubles the advance and fights for the
            // lock between each batch.
            if state.paused.load(Ordering::Relaxed) || state.stepping.load(Ordering::Relaxed) {
                continue;
            }

            // Issue #47 (fix day/night visual bug): we advance by 1 simulated
            // *hour* per auto-tick cycle (instead of a full day = 24
            // step_hour). The snapshot is emitted every hour → the frontend
            // sees the diurnal cycle.
            // Before: sim.step() = 24 silent step_hour → snapshot always at
            // midnight, sin_elev always negative, "eternal night" visually.
            // Cost: 24× more WS snapshots, but sim.step_hour() is ~24× faster
            // than sim.step() so the CPU-bound throughput is unchanged.
            let frame = {
                let mut world = state.world.lock().await;
                let t0 = Instant::now();
                world.sim_mut().step_hour();
                let elapsed_us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
                let cur_tick = world.sim().tick();
                state.last_tick_us.store(elapsed_us, Ordering::Relaxed);
                state.last_tick_num.store(cur_tick, Ordering::Relaxed);
                world.snapshot_bytes()
            };

            let _ = state.tx.send(frame);
        }
    });
}

/// Installs the global subscriber: stdout (fmt) + WebSocket (`log_bridge`),
/// filterable via `RUST_LOG`. Default: `hexsim=debug,info` to see project
/// events in debug without polluting with third-party crates.
fn init_tracing(log_tx: broadcast::Sender<String>) {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("hexsim_cli=debug,hexsim_core=info,info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().with_target(true).compact())
        .with(log_bridge::WsLogLayer::new(log_tx))
        .init();
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let client_id = state.next_client_id.fetch_add(1, Ordering::Relaxed);
    debug!(client_id, "ws open");

    // Splits the socket into sender/receiver to read and write in parallel.
    // In Python, you'd have two coroutines on the same WS. In Rust, split()
    // gives two independent halves, the compiler guarantees you can't write
    // from two places at once.
    let (mut sender, mut receiver) = socket.split();

    // Sends the current state immediately
    {
        let frame = state.world.lock().await.snapshot_bytes();
        if sender.send(Message::Binary(frame)).await.is_err() {
            debug!(client_id, "ws close (send initial snapshot failed)");
            return;
        }
    }

    let mut rx = state.tx.subscribe();
    let mut text_rx = state.text_tx.subscribe();

    // Channel for targeted responses (diagnostics, rivers, climate):
    // handle_command sends via reply_tx, send_task routes them to this client.
    let (reply_tx, mut reply_rx) = mpsc::channel::<String>(4);

    // Two tasks in parallel:
    // 1. Listens to the broadcast snapshots (binary), the targeted
    //    responses, and the text (logs, params)
    // 2. Listens to the client's commands
    let mut send_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                result = rx.recv() => {
                    match result {
                        Ok(frame) => {
                            if sender.send(Message::Binary(frame)).await.is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                reply = reply_rx.recv() => {
                    match reply {
                        Some(json) => {
                            if sender.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                }
                text = text_rx.recv() => {
                    match text {
                        Ok(json) => {
                            if sender.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                        // Lagged: the client didn't consume fast enough.
                        // We absorb the error, old messages are lost but the
                        // connection survives.
                        Err(broadcast::error::RecvError::Lagged(_)) => {}
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        }
    });

    let recv_state = Arc::clone(&state);
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                handle_command(&text, &recv_state, &reply_tx, client_id).await;
            }
        }
    });

    // If either of the two tasks finishes, stop the other
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
    debug!(client_id, "ws close");
}

/// Parses a command, applies it to the world, then executes what the
/// protocol asks for: broadcast, reply, schedule.
///
/// The world lock is taken for `apply` **and released immediately**: an
/// `Outcome::Advance` (long step) reacquires it batch by batch, otherwise
/// the whole server would stay blocked for the duration of the advance.
async fn handle_command(
    text: &str,
    state: &AppState,
    reply: &mpsc::Sender<String>,
    client_id: u64,
) {
    let cmd = match Command::parse(text) {
        Ok(cmd) => cmd,
        Err(e) => {
            warn!(client_id, payload = %text, error = %e, "commande rejetee");
            return;
        }
    };
    log_started(client_id, &cmd);

    let outcome = {
        let mut world = state.world.lock().await;
        world.apply(&cmd)
    };
    match outcome {
        Ok(outcome) => {
            // `set_param` is only logged once the key is accepted: announcing
            // it beforehand would produce a misleading INFO immediately
            // followed by the "unknown parameter" WARN.
            if let Command::SetParam { key, value } = &cmd {
                info!(client_id, key, value, "set_param");
            }
            run_outcome(outcome, state, reply, client_id).await;
        }
        Err(e) => warn!(client_id, error = %e, "commande inapplicable"),
    }
}

/// Announces **long** commands before their execution: a `reset` of a large
/// world or a `+1 year` take several seconds, and a log that only arrives at
/// the end doesn't say the server is working.
///
/// The others are logged after the fact, where we know what was actually
/// applied: `apply_schedule` for the clamped speed, `handle_command` for
/// `set_param`.
fn log_started(client_id: u64, cmd: &Command) {
    match cmd {
        Command::Reset { seed } => info!(client_id, seed = ?seed, "reset"),
        Command::Step { n, hourly } => {
            if *hourly {
                info!(client_id, n, "step_hour");
            } else {
                info!(client_id, n, "step");
            }
        }
        Command::Play
        | Command::Pause
        | Command::Speed { .. }
        | Command::SetParam { .. }
        | Command::Query(_) => {}
    }
}

/// Executes the follow-up of a command: this is where, and only where, the
/// protocol meets the network.
async fn run_outcome(
    outcome: Outcome,
    state: &AppState,
    reply: &mpsc::Sender<String>,
    client_id: u64,
) {
    match outcome {
        Outcome::Nothing => {}
        Outcome::Snapshot => {
            let frame = state.world.lock().await.snapshot_bytes();
            let _ = state.tx.send(frame);
        }
        // Targeted response: this client requested it, the others have no
        // use for it.
        Outcome::Reply(value) => {
            if let Ok(json) = serde_json::to_string(&value) {
                let _ = reply.send(json).await;
            }
        }
        // Broadcast to everyone: the change may come from another client or
        // from the CLI (`just param`), each front realigns its sliders on it
        // (#2).
        Outcome::Broadcast(value) => {
            if let Ok(json) = serde_json::to_string(&value) {
                let _ = state.text_tx.send(json);
            }
        }
        Outcome::Schedule(schedule) => apply_schedule(schedule, state, client_id),
        Outcome::Advance { n, hourly } => run_stepped(state, reply, n, hourly).await,
    }
}

/// Applies a scheduling change. These flags are specific to the server:
/// they don't describe the world, but the way this shell makes it advance.
fn apply_schedule(schedule: Schedule, state: &AppState, client_id: u64) {
    match schedule {
        Schedule::Play => {
            state.paused.store(false, Ordering::Relaxed);
            info!(client_id, "play");
        }
        Schedule::Pause => {
            state.paused.store(true, Ordering::Relaxed);
            info!(client_id, "pause");
        }
        Schedule::Speed { tick_ms, requested } => {
            if tick_ms != requested {
                warn!(
                    client_id,
                    requested,
                    applied = tick_ms,
                    "speed: value out of bounds, clamped"
                );
            }
            state.tick_ms.store(tick_ms, Ordering::Relaxed);
            info!(client_id, tick_ms, "speed");
        }
    }
}

/// Advances the simulation by `n` units (days if `hourly == false`, hours
/// otherwise) in several batches rather than in one locked block.
///
/// Motivation: a `+1 year` (365 days, ~15 s at R=30) held the lock from
/// start to finish and only emitted a single final snapshot, the front
/// stayed frozen with no feedback at all. Here we split into batches (see
/// `batch_stride`): at each batch we broadcast an intermediate snapshot
/// (the map animates live) and a `progress` message targeted at the
/// requesting client (current date + progress for its loader). We yield
/// (`yield_now`) between each batch so frames can drain and the rest of the
/// server (queries, other clients) gets to breathe.
///
/// `stepping` neutralizes the auto-tick for the duration of the job (see
/// the tick loop).
async fn run_stepped(state: &AppState, reply: &mpsc::Sender<String>, n: u64, hourly: bool) {
    state.stepping.store(true, Ordering::Relaxed);

    // Acknowledges receipt before the first batch. Without this, the first
    // sign of life only arrives once a whole batch is computed, 182 days for
    // a `step 3650`, i.e. ~5 s at R=45 and much more at larger radius. A
    // client watching for inactivity to avoid abandoning a long job
    // therefore had nothing to chew on for all that time, and would abandon
    // a computation that was still running (#145).
    let tick0 = state.world.lock().await.sim().tick();
    if let Ok(json) = serde_json::to_string(&serde_json::json!({
        "type": "progress",
        "done": 0,
        "total": n,
        "tick": tick0,
        "finished": false,
    })) {
        let _ = reply.try_send(json);
    }

    let stride = batch_stride(n);
    let mut done = 0u64;
    while done < n {
        let batch = stride.min(n - done);
        let (frame, tick) = {
            let mut world = state.world.lock().await;
            world.advance(batch, hourly);
            done += batch;
            (world.snapshot_bytes(), world.sim().tick())
        };
        // Snapshot broadcast to everyone (each client's map follows the
        // computation).
        let _ = state.tx.send(frame);
        // Progress targeted at the requester only (the others have no job
        // in progress on the UI side). Best-effort: if the channel is full,
        // we skip the step rather than block the sim on client consumption.
        let finished = done >= n;
        let progress = serde_json::json!({
            "type": "progress",
            "done": done,
            "total": n,
            "tick": tick,
            "finished": finished,
        });
        if let Ok(json) = serde_json::to_string(&progress) {
            if finished {
                // Final step: guarantees delivery (the front waits for it to
                // exit the "busy" state). Intermediate steps are
                // best-effort, but this one must never be dropped.
                let _ = reply.send(json).await;
            } else {
                let _ = reply.try_send(json);
            }
        }
        // Lets frames drain and the lock circulate before the next batch.
        tokio::task::yield_now().await;
    }

    state.stepping.store(false, Ordering::Relaxed);
}

/// `GET /checkpoint`: exports the complete world state in `MessagePack`,
/// served as an attachment (`Content-Disposition: attachment`), the browser
/// downloads a `.ckpt` file reloadable via `POST /checkpoint`. See
/// `hexsim_core::checkpoint`.
async fn export_checkpoint(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (result, tick) = {
        let world = state.world.lock().await;
        (world.sim().save_state(), world.sim().tick())
    };
    match result {
        Ok(bytes) => {
            info!(tick, size = bytes.len(), "export checkpoint");
            let disposition = format!("attachment; filename=\"hexsim-t{tick}.ckpt\"");
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, "application/octet-stream".to_owned()),
                    (header::CONTENT_DISPOSITION, disposition),
                ],
                bytes,
            )
                .into_response()
        }
        Err(e) => {
            warn!(error = %e, "export checkpoint impossible");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("save_state: {e}"),
            )
                .into_response()
        }
    }
}

/// `POST /checkpoint`: loads an exported state (body = `.ckpt` bytes) and
/// replaces the current world, then broadcasts a fresh snapshot to refresh
/// all clients. An invalid blob or one with an incompatible version is
/// rejected (`400`) without touching the world in progress.
async fn import_checkpoint(State(state): State<Arc<AppState>>, body: Bytes) -> impl IntoResponse {
    match Simulation::load_state(body.as_ref()) {
        Ok(new_sim) => {
            let (frame, tick) = {
                let mut world = state.world.lock().await;
                world.replace_simulation(new_sim);
                (world.snapshot_bytes(), world.sim().tick())
            };
            let _ = state.tx.send(frame);
            info!(tick, size = body.len(), "import checkpoint");
            (StatusCode::OK, format!("loaded (tick {tick})")).into_response()
        }
        Err(e) => {
            warn!(error = %e, "import checkpoint rejected");
            (StatusCode::BAD_REQUEST, format!("load_state: {e}")).into_response()
        }
    }
}
