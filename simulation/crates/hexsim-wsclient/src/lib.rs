//! Shared WebSocket client for `HexSim`.
//!
//! Provides a low-level connection to a `HexSim` server over WebSocket,
//! with support for sending commands and receiving typed responses.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex, oneshot};
use tokio::time::timeout;
use tokio_tungstenite::connect_async_with_config;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;

/// Default URL of the `HexSim` WebSocket server.
pub const DEFAULT_WS_URL: &str = "ws://localhost:8355/ws";

/// Mirrors the server-side limit (hexsim-cli `src/main.rs::ws_handler`).
/// Binary snapshots of large worlds (msgpack ≈ 0.17 KB/cell,
/// R=300 ≈ 46 MB) can exceed tungstenite's default frame size (16 MB).
/// We don't decode them, but we still need to be able to receive them.
pub const MAX_WS_MESSAGE_BYTES: usize = 256 * 1024 * 1024;

/// Silence tolerance before abandoning request.
///
/// An **inactivity** timeout, not maximum duration: as long as server pushes
/// `progress`, wait is re-armed. A `step` of years works much longer while
/// showing signs of life.
///
/// Thirty seconds and not five: this delay only serves to spot a truly mute
/// server, and it has to cover the interval between two steps. A job is split
/// into `STEP_BATCHES` batches, so one batch of a
/// ten-year `step` is worth six simulated months: five seconds at R=45, and
/// far more at a large radius (#145).
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// WebSocket client for communicating with a `HexSim` server.
pub struct WsClient {
    sink: Mutex<
        futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            Message,
        >,
    >,
    /// Map of pending requests, keyed by the expected response type.
    ///
    /// # Known limitation
    /// Two concurrent requests of the same type overwrite each other.
    /// This behavior already existed in the original implementation.
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    /// Number of `progress` frames received since connecting. Serves as
    /// proof of life: a value that keeps moving means the server is still
    /// working.
    progress: Arc<AtomicU64>,
}

/// WebSocket client errors.
#[derive(Debug, Error)]
pub enum WsClientError {
    /// Failed to connect to the given URL.
    /// The server may not be running.
    #[error("Could not connect to {url} (is the server running?)")]
    Connect { url: String },

    /// Failed to send a message on the socket.
    ///
    /// Boxed: `tungstenite::Error` weighs 136 bytes and would inflate every
    /// `Result` of this crate (clippy `result_large_err`). The `From` impl
    /// below keeps `?` working on socket calls.
    #[error("Failed to send the message")]
    Send(#[source] Box<tokio_tungstenite::tungstenite::Error>),

    /// No sign of life from the server while waiting for a `{expected}`
    /// response.
    #[error(
        "Timeout: no {expected} response and no progress for {}s",
        DEFAULT_REQUEST_TIMEOUT.as_secs()
    )]
    Timeout { expected: String },

    /// The response channel was closed.
    #[error("Channel closed while waiting for {expected}")]
    ChannelClosed { expected: String },

    /// Failed to deserialize JSON.
    #[error("Failed to deserialize JSON")]
    Json(#[from] serde_json::Error),
}

impl From<tokio_tungstenite::tungstenite::Error> for WsClientError {
    fn from(err: tokio_tungstenite::tungstenite::Error) -> Self {
        Self::Send(Box::new(err))
    }
}

impl WsClient {
    /// Connects to the given URL, splits the stream, and spawns the
    /// background reader task.
    ///
    /// # Errors
    /// Returns [`WsClientError::Connect`] if the connection fails.
    pub async fn connect(url: &str) -> Result<Self, WsClientError> {
        let ws_config = WebSocketConfig::default()
            .max_message_size(Some(MAX_WS_MESSAGE_BYTES))
            .max_frame_size(Some(MAX_WS_MESSAGE_BYTES));

        let (ws_stream, _) = connect_async_with_config(url, Some(ws_config), false)
            .await
            .map_err(|_| WsClientError::Connect {
                url: url.to_string(),
            })?;
        let (sink, stream) = ws_stream.split();

        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let progress = Arc::new(AtomicU64::new(0));

        tokio::spawn(ws_reader_task(
            stream,
            Arc::clone(&pending),
            Arc::clone(&progress),
        ));

        Ok(Self {
            sink: Mutex::new(sink),
            pending,
            progress,
        })
    }

    /// Sends a command without waiting for a response.
    ///
    /// Performs send + flush to guarantee that the message goes out.
    ///
    /// # Errors
    /// Returns [`WsClientError::Send`] if sending fails.
    pub async fn send(&self, cmd: Value) -> Result<(), WsClientError> {
        let msg = serde_json::to_string(&cmd)?;
        self.sink
            .lock()
            .await
            .send(Message::Text(msg.into()))
            .await?;
        Ok(())
    }

    /// Sends a command and waits for the response whose `type` ==
    /// `expected_type`.
    ///
    /// # Errors
    /// Returns:
    /// - [`WsClientError::Send`] if sending fails
    /// - [`WsClientError::Timeout`] if no response is received in time
    /// - [`WsClientError::ChannelClosed`] if the response channel is closed
    /// - [`WsClientError::Json`] if the response is not valid JSON
    pub async fn request(&self, cmd: Value, expected_type: &str) -> Result<Value, WsClientError> {
        let (tx, mut rx) = oneshot::channel();
        self.pending
            .lock()
            .await
            .insert(expected_type.to_string(), tx);

        let msg = serde_json::to_string(&cmd)?;
        self.sink
            .lock()
            .await
            .send(Message::Text(msg.into()))
            .await?;

        // Inactivity timeout, not a maximum duration: the server splits long
        // jobs into batches and pushes a `progress` between each one. With a
        // hard limit of 5 s, `hexsim-ctl step 3650` would abandon a job that
        // was still running (and the client disconnecting would kill it),
        // leaving the world at 182 days out of the 3650 requested. `just
        // year` (~10 s) and `just monitor` were in the same situation (#145).
        let mut vues = self.progress.load(Ordering::Relaxed);
        loop {
            match timeout(DEFAULT_REQUEST_TIMEOUT, &mut rx).await {
                Ok(Ok(val)) => return Ok(val),
                Ok(Err(_)) => {
                    return Err(WsClientError::ChannelClosed {
                        expected: expected_type.to_string(),
                    });
                }
                Err(_) => {
                    let maintenant = self.progress.load(Ordering::Relaxed);
                    if maintenant == vues {
                        return Err(WsClientError::Timeout {
                            expected: expected_type.to_string(),
                        });
                    }
                    vues = maintenant;
                }
            }
        }
    }
}

/// Background task: reads the WS stream and routes tagged responses to
/// callers. Broadcast frames (grid snapshots, perf) are binary (msgpack),
/// ignored by the `Message::Text` match.
async fn ws_reader_task(
    mut stream: impl StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    progress: Arc<AtomicU64>,
) {
    while let Some(Ok(msg)) = stream.next().await {
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };
        let Ok(val) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Some(type_str) = val.get("type").and_then(Value::as_str).map(str::to_owned) else {
            continue;
        };
        // No caller waits for a `progress`: it only serves to prove the
        // server is still advancing, to re-arm the wait on the `request`
        // side.
        if type_str == "progress" {
            progress.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(tx) = pending.lock().await.remove(&type_str) {
            let _ = tx.send(val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Arc, AtomicU64, HashMap, Message, Mutex, Ordering, Value, oneshot, ws_reader_task,
    };
    use futures_util::stream;

    /// The reader task routes the tagged response and lets the rest through
    /// without breaking the loop: binary snapshots (msgpack) and text frames
    /// without a `type` field are silently ignored.
    #[tokio::test]
    async fn reader_routes_the_tagged_frame_and_ignores_the_rest() {
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert("diagnostics".to_string(), tx);

        let frames = vec![
            // Grid snapshot: binary, never decoded on the client side.
            Ok(Message::Binary(vec![0x93, 0x01, 0x02].into())),
            // Valid JSON but without a routing tag.
            Ok(Message::Text(r#"{"without":"type"}"#.into())),
            // The expected response.
            Ok(Message::Text(r#"{"type":"diagnostics","tick":42}"#.into())),
        ];
        let progress = Arc::new(AtomicU64::new(0));
        ws_reader_task(stream::iter(frames), Arc::clone(&pending), progress).await;

        let val = rx.await.expect("the oneshot must have been woken");
        assert_eq!(val["tick"], 42);
        assert!(
            pending.lock().await.is_empty(),
            "the pending entry must be consumed by routing"
        );
    }

    /// Each `progress` counts as proof of life, and doesn't prevent the final
    /// response from being routed. This is the counter that `request`
    /// watches to re-arm its wait instead of abandoning a job that's still
    /// running (#145).
    #[tokio::test]
    async fn reader_counts_progress_without_disturbing_routing() {
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert("finished".to_string(), tx);

        let frames = vec![
            Ok(Message::Text(r#"{"type":"progress","done":1}"#.into())),
            Ok(Message::Text(r#"{"type":"progress","done":2}"#.into())),
            Ok(Message::Text(r#"{"type":"progress","done":3}"#.into())),
            Ok(Message::Text(r#"{"type":"finished","tick":3650}"#.into())),
        ];
        let progress = Arc::new(AtomicU64::new(0));
        ws_reader_task(
            stream::iter(frames),
            Arc::clone(&pending),
            Arc::clone(&progress),
        )
        .await;

        assert_eq!(
            progress.load(Ordering::Relaxed),
            3,
            "one count per progress"
        );
        let val = rx.await.expect("the final response must be routed");
        assert_eq!(val["tick"], 3650);
    }
}
