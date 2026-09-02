use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use serde::Serialize;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};
use tokio::time;

use crate::AppState;

#[derive(Serialize)]
#[serde(tag = "type", rename = "perf")]
struct PerfSample {
    cpu_percent: f32,
    rss_mb: f32,
    tick_us: u64,
    tick: u64,
}

pub fn spawn_perf_task(state: Arc<AppState>) {
    tokio::spawn(async move {
        let mut sys = System::new();
        let pid = match sysinfo::get_current_pid() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("perf: get_current_pid failed: {e}");
                return;
            }
        };
        let refresh_kind = ProcessRefreshKind::nothing().with_cpu().with_memory();
        let mut interval = time::interval(Duration::from_secs(1));

        // Immediate first tick: refresh without reading, to prime the CPU delta.
        interval.tick().await;
        sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), false, refresh_kind);

        loop {
            interval.tick().await;
            sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), false, refresh_kind);

            let (cpu_percent, rss_mb) = sys.process(pid).map_or((0.0, 0.0), |p| {
                #[allow(clippy::cast_precision_loss)]
                let rss = p.memory() as f32 / 1_048_576.0;
                (p.cpu_usage(), rss)
            });

            let tick_us = state.last_tick_us.load(Ordering::Relaxed);
            let tick = state.last_tick_num.load(Ordering::Relaxed);

            let sample = PerfSample {
                cpu_percent,
                rss_mb,
                tick_us,
                tick,
            };
            // Same broadcast channel as the snapshots → same named msgpack
            // encoding (the front dispatches on the `type` field after decoding).
            if let Ok(frame) = rmp_serde::to_vec_named(&sample) {
                let _ = state.tx.send(frame.into());
            }
        }
    });
}
