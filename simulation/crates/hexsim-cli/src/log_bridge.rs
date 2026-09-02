//! Bridge between `tracing` and the WebSocket.
//!
//! A custom `Layer` captures every event, extracts `message` + structured
//! fields via `Visit`, and pushes a JSON `{"type":"log", ...}` into a
//! tokio broadcast. Connected WS clients subscribe to this broadcast to
//! display the logs in real time in the frontend.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};
use tokio::sync::broadcast;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

static SEQ: AtomicU64 = AtomicU64::new(0);

pub struct WsLogLayer {
    tx: broadcast::Sender<String>,
}

impl WsLogLayer {
    pub fn new(tx: broadcast::Sender<String>) -> Self {
        Self { tx }
    }
}

#[derive(Default)]
struct FieldCollector {
    message: Option<String>,
    fields: Map<String, Value>,
}

impl FieldCollector {
    fn store(&mut self, name: &str, value: Value) {
        if name == "message" {
            if let Value::String(s) = value {
                self.message = Some(s);
            }
        } else {
            self.fields.insert(name.to_owned(), value);
        }
    }
}

impl Visit for FieldCollector {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.store(field.name(), Value::String(value.to_owned()));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.store(field.name(), Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.store(field.name(), Value::from(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.store(field.name(), Value::from(value));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.store(field.name(), Value::from(value));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.store(field.name(), Value::String(format!("{value:?}")));
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.store(field.name(), Value::String(value.to_string()));
    }
}

impl<S: Subscriber> Layer<S> for WsLogLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        // No subscriber: no point paying for JSON serialization.
        if self.tx.receiver_count() == 0 {
            return;
        }

        let mut visitor = FieldCollector::default();
        event.record(&mut visitor);

        let meta = event.metadata();
        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);

        let payload = serde_json::json!({
            "type": "log",
            "seq": seq,
            "ts_ms": ts_ms,
            "level": meta.level().as_str(),
            "target": meta.target(),
            "message": visitor.message.unwrap_or_default(),
            "fields": Value::Object(visitor.fields),
        });

        if let Ok(json) = serde_json::to_string(&payload) {
            // Error ignored: no receiver => not a problem, the short-circuit
            // already covered it, or a client just disconnected.
            let _ = self.tx.send(json);
        }
    }
}
