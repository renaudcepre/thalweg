//! Structural diagnostic: anatomy of the rain hotspots.
//!
//! `diag_rain_attractors` tells WHERE it rains all the time; this one
//! tells WHY: for the cells with the most rainy days it prints their
//! surface state averaged over the measurement year (temperature
//! anomaly against their elevation band, upper and surface humidity,
//! cloud water, open water, snow days, slope and aspect) next to the
//! band means, so a permanent condenser shows its mechanism (cold
//! surface, lake, fog, orographic pump) in one table.
//!
//! Built for the 2026-09-02 investigation (0 rain-free day per year,
//! one cell raining 365 days). Eval style: `#[ignore]`, `eprintln!`,
//! no assert. Env: `HEXSIM_DIAG_SEED`, `HEXSIM_DIAG_RADIUS`.
//!
//! ```text
//! cargo test --release -p hexsim-core --test diag_rain_hotspot_anatomy \
//!     -- --ignored --nocapture
//! ```

mod common;

use common::build_prod_sim;
use hexsim_core::simulation::Simulation;

const WARMUP_DAYS: u64 = 365;
const MEASURE_DAYS: u64 = 365;
const TOP_N: usize = 8;
const BAND_EDGES: [f32; 4] = [200.0, 500.0, 900.0, 1400.0];

#[derive(Default, Clone)]
struct CellAcc {
    rain_days: u32,
    snow_days: u32,
    t_sum: f64,
    hu_sum: f64,
    hs_sum: f64,
    cw_sum: f64,
    water_sum: f64,
    rain_mm: f64,
}

fn band_of(elevation: f32) -> usize {
    BAND_EDGES
        .iter()
        .position(|e| elevation < *e)
        .unwrap_or(BAND_EDGES.len())
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn observe(sim: &Simulation, acc: &mut [CellAcc]) {
    let precip = sim.last_precipitation();
    for (i, (_, cell)) in sim.grid().iter().enumerate() {
        let a = &mut acc[i];
        let p = &precip[i];
        if p.rained() {
            a.rain_days += 1;
        }
        if p.snow > 1e-4 {
            a.snow_days += 1;
        }
        a.rain_mm += f64::from(p.rain);
        a.t_sum += f64::from(cell.temperature);
        a.hu_sum += f64::from(cell.humidity_upper);
        a.hs_sum += f64::from(cell.humidity_surface);
        a.cw_sum += f64::from(cell.cloud_water);
        a.water_sum += f64::from(cell.water_level);
    }
}

#[test]
#[ignore = "diagnostic tool, run on demand (see module doc)"]
fn rain_hotspot_anatomy() {
    let seed: u32 = env_or("HEXSIM_DIAG_SEED", 42);
    let radius: i32 = env_or("HEXSIM_DIAG_RADIUS", 30);
    let mut sim = build_prod_sim(seed, radius);
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }
    let n = sim.grid().len();
    let mut acc = vec![CellAcc::default(); n];
    for _ in 0..MEASURE_DAYS {
        sim.step();
        observe(&sim, &mut acc);
    }
    let days = f64::from(u32::try_from(MEASURE_DAYS).expect("fits u32"));

    // Band means for the anomaly columns.
    let mut band_t = [0.0_f64; 5];
    let mut band_hu = [0.0_f64; 5];
    let mut band_n = [0_u32; 5];
    for (i, (_, cell)) in sim.grid().iter().enumerate() {
        let b = band_of(cell.elevation);
        band_t[b] += acc[i].t_sum / days;
        band_hu[b] += acc[i].hu_sum / days;
        band_n[b] += 1;
    }
    for b in 0..5 {
        if band_n[b] > 0 {
            band_t[b] /= f64::from(band_n[b]);
            band_hu[b] /= f64::from(band_n[b]);
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| acc[b].rain_days.cmp(&acc[a].rain_days));

    eprintln!(
        "\n== Rain hotspot anatomy, seed {seed}, radius {radius}, {MEASURE_DAYS} measured days =="
    );
    eprintln!(
        "  {:<12} {:>6} {:>5} {:>5} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}",
        "cell",
        "elev",
        "rain",
        "snow",
        "T",
        "dT_band",
        "hu",
        "dhu",
        "hs",
        "cw",
        "water",
        "slope",
        "aspN"
    );
    let coords = sim.grid().coords_slice();
    for &i in order.iter().take(TOP_N) {
        let cell = sim.grid().get(coords[i]).unwrap();
        let a = &acc[i];
        let b = band_of(cell.elevation);
        let (ne, nn) = (cell.normal_east, cell.normal_north);
        let n_u = (1.0 - ne * ne - nn * nn).max(0.0).sqrt();
        eprintln!(
            "  ({:>3},{:>3})    {:>6.0} {:>5} {:>5} {:>7.1} {:>7.1} {:>6.1} {:>6.1} {:>6.2} {:>6.2} {:>7.1} {:>5.0}° {:>6.2}",
            coords[i].q,
            coords[i].r,
            cell.elevation,
            a.rain_days,
            a.snow_days,
            a.t_sum / days,
            a.t_sum / days - band_t[b],
            a.hu_sum / days,
            a.hu_sum / days - band_hu[b],
            a.hs_sum / days,
            a.cw_sum / days,
            a.water_sum / days,
            n_u.acos().to_degrees(),
            nn
        );
    }
    eprintln!("  band means (T, hu) by elevation band <200/200-500/500-900/900-1400/>=1400:");
    for b in 0..5 {
        if band_n[b] > 0 {
            eprintln!(
                "    band {b}: n={:<5} T={:>6.1}  hu={:>6.1}",
                band_n[b], band_t[b], band_hu[b]
            );
        }
    }
    let median_rain = {
        let mut v: Vec<u32> = acc.iter().map(|a| a.rain_days).collect();
        v.sort_unstable();
        v[v.len() / 2]
    };
    eprintln!(
        "  median rainy days per cell = {median_rain}, top = {}",
        acc[order[0]].rain_days
    );
}
