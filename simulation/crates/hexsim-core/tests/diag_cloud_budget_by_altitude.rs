//! Where does the cloud water go, band of altitude by band of altitude,
//! on the fine path against the coarse one (coarse upper layer, lever 2c)?
//!
//! # History: the hypothesis this instrument confirmed
//!
//! Step 2b gave the coarse path back its condensation trigger and its
//! sub-grid autoconversion, and the plains still lost 19 to 46 % of their
//! rain (JOURNAL 2026-09-06). The residual was therefore neither the
//! trigger nor the drain, and the hypothesis this instrument was written
//! to test was the **broadcast exposing the cloud to dry columns**, a
//! property of the legacy `CoarseStock` mode (retired 2026-09-07, see
//! `atmosphere::coarse`'s module doc for the numbers this instrument
//! produced against it):
//!
//! - on the fine path a plains column saturated by the pump condenses and
//!   **keeps** its cloud (zero deficit, nothing to evaporate), so the
//!   cloud persists for hours and KK2000 drains it slowly;
//! - on `CoarseStock` that condensate was pooled and rebroadcast over
//!   every fine column of its ~1 km cell, most of which are subsaturated,
//!   and next hour each of them evaporated `min(cw_view, deficit_i)` back
//!   to vapour through the saturation adjustment.
//!
//! The hypothesis was confirmed: below 300 m, `CoarseStock`'s evaporation
//! ran several times the fine path's for a comparable condensation, and
//! the cloud's residence time was divided by as much.
//!
//! # What the instrument still measures
//!
//! Both branches of the vapour ↔ droplet adjustment are one number: the
//! signed transfer `condensation::cloud_dynamics` returns, recorded per
//! column by the fused sweep (`Simulation::set_cloud_transfer_probe`).
//! Split it by sign and by the cell's elevation:
//!
//! - **condensation** (transfer > 0), **evaporation by adjustment**
//!   (transfer < 0), the two branches, summed over the window;
//! - the **KK2000 sheet** that actually fell (`precip_this_tick`);
//! - the mean **cloud stock** carried;
//! - the **residence time** of a millimetre of cloud water, `stock /
//!   (evaporation + precipitation)`, and the mean **lifetime** in
//!   consecutive hours a column holds `cloud_water > 0` — the second one
//!   would have read the broadcast on the now-retired coarse-stock path,
//!   see [`BandBudget::residence_h`]; on the surviving [`CoarsePrecip`]
//!   there is no broadcast to bias it.
//!
//! Now run against [`MoistCoarseMode::CoarsePrecip`] (the only surviving
//! coarse mode), this instrument is a live seam for its own open
//! question: `plains_precip_mm_per_day` is the one blocker on the flip to
//! on by default (see `MOIST_COARSE_DEFAULT`'s doc), and this by-altitude
//! breakdown is where to look first if that gap ever needs diagnosing.
//!
//! Ignored, like every `diag_*`: an instrument, not a gate.
//!
//! [`CoarsePrecip`]: hexsim_core::atmosphere::MoistCoarseMode::CoarsePrecip

mod common;

use common::build_prod_sim;
use hexsim_core::atmosphere::MoistCoarseMode;
use hexsim_core::simulation::Simulation;

/// Warm-up before the window opens, in days: the same one year every
/// measurement of this effort uses.
const WARMUP_DAYS: u32 = 365;
/// Measured window, in hours: 30 days.
const MEASURED_HOURS: u32 = 24 * 30;

/// Upper edges of the first three altitude bands (m); everything above
/// the last one is the fourth.
const BAND_EDGES: [f32; 3] = [300.0, 800.0, 1500.0];
const BAND_LABELS: [&str; 4] = ["  < 300 m", " 300-800 m", "800-1500 m", " >= 1500 m"];
const BANDS: usize = 4;

fn band_of(elevation: f32) -> usize {
    BAND_EDGES.iter().filter(|&&edge| elevation >= edge).count()
}

/// One altitude band's cloud budget over the window. Everything is a
/// **sum over the band's cells and over the window's hours**, in mm, so
/// two bands of different size are compared through the per-cell-per-day
/// normalisation the report does, not here.
#[derive(Default, Clone)]
struct BandBudget {
    /// Counters kept as `f64` rather than integers: every one of them is
    /// only ever read as the denominator of a mean, and an f64 holds a
    /// cell or hour count exactly far past anything this engine builds.
    cells: f64,
    /// Σ of the positive transfers: vapour turned into droplets.
    condensed_mm: f64,
    /// Σ of the negative transfers, as a positive number: droplets given
    /// back to vapour by the saturation adjustment.
    evaporated_mm: f64,
    /// Σ of the rain and snow that actually fell.
    precip_mm: f64,
    /// Σ over hours of Σ over cells of `cloud_water`: divided by
    /// `cells × hours` it is the mean stock a column carries.
    cloud_stock_mm: f64,
    /// Column-hours with a cloud, and the runs they form.
    cloudy_cell_hours: f64,
    runs: f64,
    run_hours: f64,
}

impl BandBudget {
    /// Mean stock of a column (mm).
    fn mean_stock(&self, hours: u32) -> f64 {
        let denom = self.cells * f64::from(hours);
        if denom > 0.0 {
            self.cloud_stock_mm / denom
        } else {
            0.0
        }
    }

    /// Mean length of a cloud's life, in consecutive hours.
    fn mean_lifetime_h(&self) -> f64 {
        if self.runs > 0.0 {
            self.run_hours / self.runs
        } else {
            0.0
        }
    }

    /// Share of the window a column of this band holds a cloud.
    fn cloudy_share(&self, hours: u32) -> f64 {
        let denom = self.cells * f64::from(hours);
        if denom > 0.0 {
            self.cloudy_cell_hours / denom
        } else {
            0.0
        }
    }

    /// Per cell and per day, for a quantity summed over cells and hours.
    fn per_cell_day(&self, total: f64, hours: u32) -> f64 {
        let days = f64::from(hours) / 24.0;
        let denom = self.cells * days;
        if denom > 0.0 { total / denom } else { 0.0 }
    }

    /// **Residence time of a millimetre of cloud water**, in hours:
    /// `stock / (evaporation + precipitation) per hour`, the mass form of
    /// "how long does a cloud live here".
    ///
    /// The reason it is reported next to [`Self::mean_lifetime_h`] and not
    /// instead of it: on the now-retired coarse-stock path, the fine
    /// `cloud_water` was the stock **broadcast**, so "this column holds a
    /// cloud" really meant "this column's 1 km cell holds a cloud" and
    /// the occupancy measure read the broadcast, exactly like
    /// `cloud_cover_mountain_pct` did (JOURNAL 2026-09-06, step 2b). The
    /// mass ratio has no such bias — both numerator and denominator are
    /// mass, and a broadcast conserves mass — which is why it is kept
    /// even now that the only surviving coarse mode, `CoarsePrecip`, has
    /// no broadcast to bias `mean_lifetime_h` either.
    fn residence_h(&self, hours: u32) -> f64 {
        let stock = self.mean_stock(hours);
        let loss_per_hour = self.per_cell_day(self.evaporated_mm + self.precip_mm, hours) / 24.0;
        if loss_per_hour > 0.0 {
            stock / loss_per_hour
        } else {
            f64::NAN
        }
    }
}

/// Runs one world for the window and returns its four bands.
fn measure(seed: u32, radius: i32, mode: MoistCoarseMode) -> [BandBudget; BANDS] {
    let mut sim = build_prod_sim(seed, radius);
    sim.set_moist_coarse_mode(mode);
    assert_eq!(sim.moist_coarse_mode(), mode, "the seam must have taken");
    sim.set_cloud_transfer_probe(true);
    for _ in 0..WARMUP_DAYS {
        sim.step();
    }

    // Elevation is frozen after generation, so the banding is done once.
    let bands: Vec<usize> = sim
        .grid()
        .cells_slice()
        .iter()
        .map(|c| band_of(c.elevation))
        .collect();
    let mut budgets: [BandBudget; BANDS] = std::array::from_fn(|_| BandBudget::default());
    for &b in &bands {
        budgets[b].cells += 1.0;
    }

    // Consecutive hours the column has been holding a cloud.
    let mut run = vec![0_u32; bands.len()];
    for _ in 0..MEASURED_HOURS {
        sim.step_hour();
        accumulate_hour(&sim, &bands, &mut budgets, &mut run);
    }
    // Right-censored runs still open when the window closes: counted as
    // they stand rather than dropped, which would throw away exactly the
    // long-lived clouds the measurement is about.
    for (i, &r) in run.iter().enumerate() {
        if r > 0 {
            let b = &mut budgets[bands[i]];
            b.runs += 1.0;
            b.run_hours += f64::from(r);
        }
    }
    budgets
}

fn accumulate_hour(
    sim: &Simulation,
    bands: &[usize],
    budgets: &mut [BandBudget; BANDS],
    run: &mut [u32],
) {
    let transfer = sim.cloud_transfer();
    let events = sim.precip_this_tick();
    for (i, cell) in sim.grid().cells_slice().iter().enumerate() {
        let b = &mut budgets[bands[i]];
        let t = f64::from(transfer[i]);
        if t > 0.0 {
            b.condensed_mm += t;
        } else {
            b.evaporated_mm += -t;
        }
        b.precip_mm += f64::from(events[i].rain) + f64::from(events[i].snow);
        b.cloud_stock_mm += f64::from(cell.cloud_water);
        if cell.cloud_water > 0.0 {
            b.cloudy_cell_hours += 1.0;
            run[i] += 1;
        } else if run[i] > 0 {
            b.runs += 1.0;
            b.run_hours += f64::from(run[i]);
            run[i] = 0;
        }
    }
}

fn ratio(coarse: f64, fine: f64) -> String {
    if fine.abs() > 0.0 {
        format!("{:>6.2}x", coarse / fine)
    } else if coarse.abs() > 0.0 {
        "   inf".to_string()
    } else {
        "     -".to_string()
    }
}

fn report(seed: u32, radius: i32, mode: MoistCoarseMode) {
    let fine = measure(seed, radius, MoistCoarseMode::Fine);
    let coarse = measure(seed, radius, mode);
    println!(
        "\n--- seed {seed}, r{radius}, {MEASURED_HOURS} h after {WARMUP_DAYS} d, \
         Fine against {mode:?} (mm per cell per day unless stated) ---"
    );
    fluxes_table(&fine, &coarse);
    lifetimes_table(&fine, &coarse);
    shares_table(&fine, &coarse);
}

/// The three fluxes that move cloud water: what condensed, what the
/// saturation adjustment gave back to vapour, what fell.
fn fluxes_table(fine: &[BandBudget; BANDS], coarse: &[BandBudget; BANDS]) {
    println!(
        "{:<11} {:>5} | {:>9} {:>9} {:>7} | {:>9} {:>9} {:>7} | {:>9} {:>9} {:>7}",
        "band",
        "cells",
        "cond fine",
        "cond crs",
        "ratio",
        "evap fine",
        "evap crs",
        "ratio",
        "prec fine",
        "prec crs",
        "ratio"
    );
    let per_day = |b: &BandBudget, v: f64| b.per_cell_day(v, MEASURED_HOURS);
    for (b, (f, c)) in fine.iter().zip(coarse).enumerate() {
        let (cf, cc) = (per_day(f, f.condensed_mm), per_day(c, c.condensed_mm));
        let (ef, ec) = (per_day(f, f.evaporated_mm), per_day(c, c.evaporated_mm));
        let (pf, pc) = (per_day(f, f.precip_mm), per_day(c, c.precip_mm));
        println!(
            "{:<11} {:>5.0} | {cf:>9.4} {cc:>9.4} {} | {ef:>9.4} {ec:>9.4} {} | {pf:>9.4} {pc:>9.4} {}",
            BAND_LABELS[b],
            f.cells,
            ratio(cc, cf),
            ratio(ec, ef),
            ratio(pc, pf),
        );
    }
}

/// How long the cloud water stays: the stock a column carries, the mass
/// residence time, and the occupancy-based lifetime (biased by the
/// broadcast on the coarse-stock mode, see [`BandBudget::residence_h`]).
fn lifetimes_table(fine: &[BandBudget; BANDS], coarse: &[BandBudget; BANDS]) {
    println!(
        "{:<11} {:>5} | {:>9} {:>9} {:>7} | {:>9} {:>9} {:>7} | {:>9} {:>9} {:>7}",
        "band",
        "cells",
        "stock fin",
        "stock crs",
        "ratio",
        "resid fin",
        "resid crs",
        "ratio",
        "life fine",
        "life crs",
        "ratio"
    );
    for (b, (f, c)) in fine.iter().zip(coarse).enumerate() {
        let (sf, sc) = (f.mean_stock(MEASURED_HOURS), c.mean_stock(MEASURED_HOURS));
        let (rf, rc) = (f.residence_h(MEASURED_HOURS), c.residence_h(MEASURED_HOURS));
        let (lf, lc) = (f.mean_lifetime_h(), c.mean_lifetime_h());
        println!(
            "{:<11} {:>5.0} | {sf:>9.5} {sc:>9.5} {} | {rf:>9.2} {rc:>9.2} {} | {lf:>9.2} {lc:>9.2} {}",
            BAND_LABELS[b],
            f.cells,
            ratio(sc, sf),
            ratio(rc, rf),
            ratio(lc, lf),
        );
    }
}

/// The two dimensionless shares the hypothesis is about: of what a band
/// condensed, how much went back to vapour and how much reached the
/// ground. Plus the raw occupancy, for the record.
fn shares_table(fine: &[BandBudget; BANDS], coarse: &[BandBudget; BANDS]) {
    println!(
        "{:<11} {:>5} | {:>9} {:>9} {:>7} | {:>15} {:>15} | {:>15} {:>15}",
        "band",
        "cells",
        "cloudy f",
        "cloudy c",
        "ratio",
        "evap/cond fin",
        "evap/cond crs",
        "prec/cond fin",
        "prec/cond crs"
    );
    for (b, (f, c)) in fine.iter().zip(coarse).enumerate() {
        let (yf, yc) = (
            f.cloudy_share(MEASURED_HOURS),
            c.cloudy_share(MEASURED_HOURS),
        );
        let share = |num: f64, den: f64| if den > 0.0 { num / den } else { f64::NAN };
        println!(
            "{:<11} {:>5.0} | {yf:>9.3} {yc:>9.3} {} | {:>15.3} {:>15.3} | {:>15.3} {:>15.3}",
            BAND_LABELS[b],
            f.cells,
            ratio(yc, yf),
            share(f.evaporated_mm, f.condensed_mm),
            share(c.evaporated_mm, c.condensed_mm),
            share(f.precip_mm, f.condensed_mm),
            share(c.precip_mm, c.condensed_mm),
        );
    }
}

/// r30, the three seeds of the effort's bench protocol.
#[test]
#[ignore = "diagnostic: the cloud budget by altitude band, fine path against coarse (lever 2c)"]
fn cloud_budget_by_altitude_fine_against_coarse() {
    println!(
        "\n=== cloud budget by altitude band, fine path against coarse ===\n\
         cond = vapour -> droplets (positive transfer)\n\
         evap = droplets -> vapour, saturation adjustment (negative transfer)\n\
         prec = rain + snow fallen; stock = mean cloud_water of a column (mm)\n\
         life = mean consecutive hours a column holds a cloud\n\
         cloudy = share of the window a column holds a cloud"
    );
    // The mode the hypothesis was originally about, `CoarseStock`
    // (broadcasting the pooled condensate over every column of its 1 km
    // cell), was retired 2026-09-07 — see `atmosphere::coarse`'s module
    // doc for the numbers this instrument produced against it. The
    // instrument itself stays, run against the surviving coarse mode: a
    // live seam for any future plains-bias investigation of
    // `CoarsePrecip` (see its own doc, the one open blocker on the flip
    // to on by default).
    let mode = MoistCoarseMode::CoarsePrecip;
    for seed in [42, 7, 123] {
        report(seed, 30, mode);
    }

    // Not an assertion on the physics, only on the instrument: a run that
    // recorded nothing would print zeros and read like "nothing condenses".
    let fine = measure(42, 6, MoistCoarseMode::Fine);
    let total: f64 = fine.iter().map(|b| b.condensed_mm).sum();
    assert!(
        total > 0.0,
        "the probe recorded no transfer at all, the instrument is not wired"
    );
}
