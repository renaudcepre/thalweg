//! Per-cell parallel iteration, gated by the `parallel` feature (default
//! on, off for `hexsim-proto`/`hexsim-wasm`'s wasm32 build). Every hot loop
//! in the crate that is a pure per-cell map — cell `i` of the output
//! depends only on read-only inputs and on cell `i` itself, never on
//! another output cell — goes through [`for_each_chunk_mut`] or
//! [`for_each_chunk_mut2`] instead of sprinkling `#[cfg(feature =
//! "parallel")]` at call sites. Anything that scatters into a neighbor's
//! output (`deltas[j] += …`) stays a plain serial loop or is rewritten as
//! a per-source outflow + fixed-order gather (the atmosphere's and the
//! hydro's route since the r250 effort); a reduction across cells (a sum,
//! a mean) goes through [`reduce_blocks`], whose fixed-size blocks keep it
//! deterministic whatever the thread count.
//!
//! The whole hourly tick runs inside the crate's own thread pool
//! ([`install`], called once per tick by `Simulation::step_hour`), so the
//! serial passes and the parallel passes execute on pool workers and the
//! per-pass dispatch below hands each worker its own region of a pool
//! worker's own deque. Injecting each pass from a foreign thread instead
//! costs a cross-thread hand-off and a cache migration per pass, ~500 per
//! simulated day: measured +70 % on the single-thread path at r45 (58.9 vs
//! 33.8 ms/day) before this design was adopted.
//!
//! `HEXSIM_THREADS`: number of workers in that pool, read once per process
//! on first use. Unset (the default) means one worker per core. Irrelevant
//! to physics: every parallelized loop is proven bit-identical across
//! thread counts by
//! `simulation::persistence::tests::parallel_tick_is_bit_identical_to_serial`.
//!
//! ## Sticky region dispatch (r250 perf effort, chunk B1)
//!
//! `par_chunks_mut`'s fixed-size chunks (formerly `CHUNK_CELLS` cells
//! each, now removed) are handed out through rayon's work-stealing queues, so which
//! worker processes a given cell varies from pass to pass — and ~30 passes
//! run per simulated hour. A cell's data then migrates between cores'
//! caches on almost every pass instead of staying resident in the core
//! that last touched it, which measurably capped the scaling of the
//! parallel passes (×2–2.5 instead of the ×4 Amdahl's law allows at 85 %
//! parallel fraction on 4 threads).
//!
//! The helpers below fix that: each worker gets one contiguous region of
//! the slice (computed by [`region_ranges`]) for the whole pass, and
//! [`rayon::broadcast`] runs the per-cell closure once per worker instead
//! of once per chunk. The hand-off from the dispatching thread to the `T`
//! workers is a `Vec<Mutex<Option<&mut [T]>>>`: one region per worker, each
//! lock taken exactly once (by the worker whose [`rayon::BroadcastContext::index`]
//! matches the region's position) and never contended, so it costs a cheap
//! uncontended lock rather than real synchronization. No `unsafe`: the
//! borrow checker, not a manual pointer split, proves the regions are
//! disjoint (they come from chaining `split_at_mut` down the original
//! slice).
//!
//! `rayon::broadcast` runs on the registry of the thread that calls it —
//! the *global* pool if that thread isn't a rayon worker — so every helper
//! below only takes this path when `rayon::current_thread_index()` is
//! `Some`, i.e. when it is already running on a worker of `pool()` (large
//! grids: `install` above) or of a test's own pinned pool
//! (`parallel_tick_is_bit_identical_to_serial`). Called from a non-pool
//! thread it falls back to the plain serial call, which is still correct,
//! just not parallel — a bare call from outside any pool has no thread
//! count to size the regions with anyway.
//!
//! Not solved here: the owner's machine mixes performance and efficiency
//! cores, and an equal-size split makes every pass as slow as its slowest
//! worker. [`region_ranges`] is the single place that decides the split,
//! so a later chunk can bias it by core throughput instead of by cell
//! count alone; this chunk only fixes *which* cells stay on which worker
//! across passes, not how many each worker gets.
//!
//! ## Sweep fusion (r250 perf effort, chunk B2)
//!
//! Five phases (`step_temperature`, `step_snow`, `step_atmosphere_into`,
//! `step_hydro_mfd_into`, `step_groundwater_into`) used to open with a
//! `current → next` full-grid copy (formerly a `clone_from_slice` helper
//! here) before their own per-cell writes. Each copy folded into the
//! phase's own first per-cell pass instead (`*next_cell = cur_cell.clone();`
//! ahead of that pass's writes) — same bytes end up in `next`, same order
//! of operations per field, one full-grid stream fewer per tick per
//! phase. `clone_from_slice` had no callers left afterward and was
//! removed rather than kept as dead code.

#[cfg(feature = "parallel")]
use std::sync::{Mutex, OnceLock};

/// Below this many cells a pass runs serially even with the feature on.
/// Measured on a 4-vCPU container (2026-09-04, `scale_perf_radius_60`,
/// ms per simulated day, serial build / 1 thread / 4 threads): r45
/// (6 211 cells) 33.8 / 37.3 / 51.0, r120 (43 561) 298.6 / 327.2 / 302.2,
/// r250 (188 251) 1488 / 1515 / 1249. Under ~50 000 cells the fork-join
/// barrier and the cache migration of every pass cost more than the split
/// earns, so those grids keep the exact serial path.
#[cfg(feature = "parallel")]
pub(crate) const PAR_MIN_CELLS: usize = 50_000;

/// The crate's thread pool, built once on first use with `HEXSIM_THREADS`
/// workers (rayon's default, one per core, when unset or unparsable).
#[cfg(feature = "parallel")]
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("HEXSIM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0);
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = threads {
            builder = builder.num_threads(n);
        }
        builder
            .build()
            .expect("rayon thread pool for the simulation tick")
    })
}

/// Runs `f`, the whole hourly tick, on a pool worker when the grid is
/// large enough for any pass to split (`cells >= PAR_MIN_CELLS`); below
/// that every pass is serial anyway and running on the calling thread
/// avoids the hand-off (measured +10 % on r45 otherwise, 37.4 vs 33.8
/// ms/day). Already on a rayon worker (any pool, e.g. a test pinning its
/// own thread count around the tick): runs in place so the caller's pool
/// keeps control. Without the feature: plain call.
pub(crate) fn install<R, F>(cells: usize, f: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    #[cfg(feature = "parallel")]
    {
        if cells < PAR_MIN_CELLS || rayon::current_thread_index().is_some() {
            f()
        } else {
            pool().install(f)
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = cells;
        f()
    }
}

/// Splits `total_len` cells into `regions` contiguous, near-equal-size
/// spans and returns each span's `(start, len)`: the leading
/// `total_len % regions` spans get one extra cell so every cell is
/// assigned to exactly one region and the spans are within one cell of
/// each other in size. The one place a later chunk would change to weight
/// the split by a worker's actual throughput (performance vs. efficiency
/// cores) instead of splitting purely by cell count — an equal split
/// today makes a pass only as fast as its slowest worker.
#[cfg(feature = "parallel")]
fn region_ranges(total_len: usize, regions: usize) -> Vec<(usize, usize)> {
    if regions == 0 {
        return Vec::new();
    }
    let base = total_len / regions;
    let rem = total_len % regions;
    let mut start = 0;
    (0..regions)
        .map(|i| {
            let len = base + usize::from(i < rem);
            let range = (start, len);
            start += len;
            range
        })
        .collect()
}

/// Chains `split_at_mut` down `slice` at the boundaries in `ranges`,
/// returning one `&mut` sub-slice per range in order. The borrow checker
/// proves the returned slices are disjoint — no `unsafe` pointer splitting
/// needed to hand each region to a different worker.
#[cfg(feature = "parallel")]
fn split_into_regions<'a, T>(slice: &'a mut [T], ranges: &[(usize, usize)]) -> Vec<&'a mut [T]> {
    let mut rest = slice;
    let mut regions = Vec::with_capacity(ranges.len());
    for &(_, len) in ranges {
        let (head, tail) = rest.split_at_mut(len);
        regions.push(head);
        rest = tail;
    }
    regions
}

pub(crate) fn for_each_chunk_mut<T, F>(out: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    let cells = out.len();
    for_each_chunk_mut_over(out, cells, f);
}

/// [`for_each_chunk_mut`] for a pass whose work is not proportional to
/// `out.len()`: `cells` is the number of grid cells the pass streams,
/// the quantity the [`PAR_MIN_CELLS`] threshold was measured against. A
/// reduction that emits one partial per block ([`reduce_blocks`]) or a
/// gather onto the synoptic coarse grid (`SynopticMesh::aggregate_temperature`)
/// writes a few dozen to a few thousand outputs while reading the whole
/// fine grid; sized on its output alone it would never leave the serial
/// path.
pub(crate) fn for_each_chunk_mut_over<T, F>(out: &mut [T], cells: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        if cells < PAR_MIN_CELLS || rayon::current_thread_index().is_none() {
            return f(0, out);
        }
        let ranges = region_ranges(out.len(), rayon::current_num_threads());
        let slots: Vec<Mutex<Option<&mut [T]>>> = split_into_regions(out, &ranges)
            .into_iter()
            .map(|region| Mutex::new(Some(region)))
            .collect();
        rayon::broadcast(|ctx| {
            let t = ctx.index();
            let region = slots[t].lock().expect("region mutex poisoned").take();
            if let Some(region) = region {
                f(ranges[t].0, region);
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = cells;
        f(0, out);
    }
}

/// Cells per block of a deterministic block reduction ([`reduce_blocks`]).
/// Fixed whatever the grid size and the thread count: that is what makes
/// every partial, hence the folded result, independent of how many
/// workers computed them. 4 096 cells of 84 bytes stream in ~350 KB, a
/// block per L2, and r250 (188 251 cells) splits into 46 blocks, enough
/// to keep 10 workers busy.
pub(crate) const REDUCE_BLOCK_CELLS: usize = 4096;

/// Deterministic parallel reduction over `0..n` (r250 perf effort, the
/// "reductions" chunk after B2): `block(range)` runs once per fixed-size
/// block of [`REDUCE_BLOCK_CELLS`] cells (the last one shorter) and its
/// results land in `partials` in block order, for the caller to fold
/// sequentially (the fold stays with the caller so the partial type is
/// free: a sum, a pair of sums, a min/max/count triple).
///
/// Bit-identical whatever the thread count, and identical to the serial
/// path below [`PAR_MIN_CELLS`]: a block's partial is one sequential loop
/// over a fixed range, the same wherever it runs, and the caller's fold
/// order is fixed. The alternative, one partial per worker region, would
/// group the terms by thread count and move the rounding with it. Not
/// bit-identical to the single serial loop it replaces (a different
/// association of the same terms): a change gated, like the scatter →
/// gather rewrites, by the 3-seed climate ablation, not by a golden.
///
/// `partials` is cleared and resized by the call: pass a reused buffer
/// from the hot path, or a fresh `Vec` where one allocation of
/// `n / 4096` entries is in the noise.
pub(crate) fn reduce_blocks<T, F>(n: usize, partials: &mut Vec<T>, block: F)
where
    T: Send + Default + Clone,
    F: Fn(std::ops::Range<usize>) -> T + Sync + Send,
{
    let blocks = n.div_ceil(REDUCE_BLOCK_CELLS);
    partials.clear();
    partials.resize(blocks, T::default());
    for_each_chunk_mut_over(partials, n, |start, chunk| {
        for (local, partial) in chunk.iter_mut().enumerate() {
            let lo = (start + local) * REDUCE_BLOCK_CELLS;
            let hi = (lo + REDUCE_BLOCK_CELLS).min(n);
            *partial = block(lo..hi);
        }
    });
}

/// Two-output variant of [`for_each_chunk_mut`]: `a` and `b` (same length)
/// are split at the same region boundaries and their matching regions
/// handed to `f` together, for a phase that writes two parallel output
/// buffers from the same per-cell computation (e.g. `flux_factor` +
/// `illumination`).
///
/// # Panics
/// Debug builds only: `a.len() != b.len()`, a caller bug (the two outputs
/// are meant to be indexed identically).
pub(crate) fn for_each_chunk_mut2<T, U, F>(a: &mut [T], b: &mut [U], f: F)
where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Sync + Send,
{
    debug_assert_eq!(a.len(), b.len(), "for_each_chunk_mut2: length mismatch");
    #[cfg(feature = "parallel")]
    {
        /// One worker's still-unclaimed `(a, b)` region pair.
        type PairSlot<'a, T, U> = Mutex<Option<(&'a mut [T], &'a mut [U])>>;

        if a.len() < PAR_MIN_CELLS || rayon::current_thread_index().is_none() {
            return f(0, a, b);
        }
        let ranges = region_ranges(a.len(), rayon::current_num_threads());
        let slots: Vec<PairSlot<'_, T, U>> = split_into_regions(a, &ranges)
            .into_iter()
            .zip(split_into_regions(b, &ranges))
            .map(|(ra, rb)| Mutex::new(Some((ra, rb))))
            .collect();
        rayon::broadcast(|ctx| {
            let t = ctx.index();
            let region = slots[t].lock().expect("region mutex poisoned").take();
            if let Some((ra, rb)) = region {
                f(ranges[t].0, ra, rb);
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        f(0, a, b);
    }
}

/// Six-output variant of [`for_each_chunk_mut2`]: writes six equally-sized
/// buffers (`out[0..6]`, one per hex direction) at the same region
/// boundaries. Outflow phase of a scatter → gather split (r250 perf
/// effort, `atmosphere::uplift`/`advection`/`condensation`): a source
/// cell's outgoing amount toward each of its 6 neighbors is computed
/// together from the same per-cell state (wind weights, elevation
/// gradient), so six separate single-output passes would redo that
/// shared work six times over.
///
/// # Panics
/// Debug builds only: the six buffers don't all have the same length.
pub(crate) fn for_each_chunk_mut6<F>(out: &mut [Vec<f32>; 6], f: F)
where
    F: Fn(usize, &mut [&mut [f32]; 6]) + Sync + Send,
{
    let n = out[0].len();
    debug_assert!(
        out.iter().all(|v| v.len() == n),
        "for_each_chunk_mut6: length mismatch"
    );
    #[cfg(feature = "parallel")]
    {
        if n < PAR_MIN_CELLS || rayon::current_thread_index().is_none() {
            let mut whole = out.each_mut().map(Vec::as_mut_slice);
            f(0, &mut whole);
            return;
        }
        let ranges = region_ranges(n, rayon::current_num_threads());
        let [dir0, dir1, dir2, dir3, dir4, dir5] = out;
        let slots: Vec<Mutex<Option<[&mut [f32]; 6]>>> = split_into_regions(dir0, &ranges)
            .into_iter()
            .zip(split_into_regions(dir1, &ranges))
            .zip(split_into_regions(dir2, &ranges))
            .zip(split_into_regions(dir3, &ranges))
            .zip(split_into_regions(dir4, &ranges))
            .zip(split_into_regions(dir5, &ranges))
            .map(|(((((r0, r1), r2), r3), r4), r5)| Mutex::new(Some([r0, r1, r2, r3, r4, r5])))
            .collect();
        rayon::broadcast(|ctx| {
            let t = ctx.index();
            let region = slots[t].lock().expect("region mutex poisoned").take();
            if let Some(mut region) = region {
                f(ranges[t].0, &mut region);
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut whole = out.each_mut().map(Vec::as_mut_slice);
        f(0, &mut whole);
    }
}

/// Sum of one cell's row across the six per-direction outflow buffers a
/// scatter -> gather pass fills in its outflow phase (`dir_out[d][idx]`
/// for `d` in `0..6`): exactly what cell `idx` sent out in total, across
/// every direction, in the historical serial scatter this replaces (its
/// self-decrement, `deltas[i] -= …`, always subtracted the FULL amount
/// regardless of where each share landed).
pub(crate) fn sum_dir_out(dir_out: &[Vec<f32>; 6], idx: usize) -> f32 {
    dir_out.iter().map(|row| row[idx]).sum()
}

#[cfg(test)]
mod tests {
    use super::{
        REDUCE_BLOCK_CELLS, for_each_chunk_mut, for_each_chunk_mut2, for_each_chunk_mut6,
        reduce_blocks,
    };

    /// Pseudo-random f32 series (LCG), enough spread for f32 rounding to
    /// depend on the association of the terms.
    fn noisy_series(n: usize) -> Vec<f32> {
        let mut state = 0x9E37_79B9_u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = f32::from(u16::try_from(state >> 16).unwrap()) / 65_535.0;
                unit * 40.0 - 20.0
            })
            .collect()
    }

    fn block_sums(values: &[f32]) -> Vec<f32> {
        let mut partials = Vec::new();
        reduce_blocks(values.len(), &mut partials, |range| {
            values[range].iter().sum::<f32>()
        });
        partials
    }

    #[test]
    fn reduce_blocks_covers_every_index_exactly_once() {
        for n in [
            0_usize,
            1,
            REDUCE_BLOCK_CELLS - 1,
            REDUCE_BLOCK_CELLS,
            3 * REDUCE_BLOCK_CELLS + 7,
        ] {
            let mut partials: Vec<Vec<usize>> = Vec::new();
            reduce_blocks(n, &mut partials, Iterator::collect);
            assert_eq!(partials.len(), n.div_ceil(REDUCE_BLOCK_CELLS), "n={n}");
            let flat: Vec<usize> = partials.into_iter().flatten().collect();
            assert_eq!(flat, (0..n).collect::<Vec<_>>(), "n={n}");
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn reduce_blocks_partials_are_bit_identical_across_thread_counts() {
        // > PAR_MIN_CELLS so the pooled runs take the broadcast path.
        let values = noisy_series(60_000);
        let outside_pool = block_sums(&values);
        assert!(rayon::current_thread_index().is_none());
        for threads in [1, 3, 4] {
            let pooled = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build pool")
                .install(|| block_sums(&values));
            let same_bits = pooled
                .iter()
                .zip(&outside_pool)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same_bits, "{threads} threads: partials diverged");
        }
    }

    #[test]
    fn for_each_chunk_mut_covers_every_index_exactly_once() {
        let mut data = vec![0_i64; 10_000];
        for_each_chunk_mut(&mut data, |start, chunk| {
            for (local, v) in chunk.iter_mut().enumerate() {
                *v = i64::try_from(start + local).unwrap();
            }
        });
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i64::try_from(i).unwrap());
        }
    }

    #[test]
    fn for_each_chunk_mut2_keeps_both_outputs_aligned() {
        let mut a = vec![0_i32; 5_000];
        let mut b = vec![0_i32; 5_000];
        for_each_chunk_mut2(&mut a, &mut b, |start, ca, cb| {
            for local in 0..ca.len() {
                let i = i32::try_from(start + local).unwrap();
                ca[local] = i;
                cb[local] = i * 2;
            }
        });
        for i in 0..a.len() {
            assert_eq!(a[i], i32::try_from(i).unwrap());
            assert_eq!(b[i], 2 * i32::try_from(i).unwrap());
        }
    }

    #[test]
    fn for_each_chunk_mut6_covers_every_index_and_direction_exactly_once() {
        // > PAR_MIN_CELLS: exercises the real multi-region parallel path,
        // not just the small-slice fallback.
        let n = 55_000;
        let mut out: [Vec<f32>; 6] = std::array::from_fn(|_| vec![-1.0; n]);
        for_each_chunk_mut6(&mut out, |start, chunks| {
            for (d, chunk) in chunks.iter_mut().enumerate() {
                let d_val = f32::from(u16::try_from(d).unwrap());
                for (local, slot) in chunk.iter_mut().enumerate() {
                    let i_val = f32::from(u16::try_from(start + local).unwrap());
                    *slot = d_val * 100_000.0 + i_val;
                }
            }
        });
        for (d, plane) in out.iter().enumerate() {
            let d_val = f32::from(u16::try_from(d).unwrap());
            for (i, &got) in plane.iter().enumerate() {
                let i_val = f32::from(u16::try_from(i).unwrap());
                let expected = d_val * 100_000.0 + i_val;
                assert_eq!(got.to_bits(), expected.to_bits(), "cell {i} dir {d}");
            }
        }
    }

    #[test]
    fn small_slice_is_a_single_chunk() {
        let mut data = vec![0_u32; 3];
        for_each_chunk_mut(&mut data, |start, chunk| {
            for (local, v) in chunk.iter_mut().enumerate() {
                *v = u32::try_from(start + local).unwrap();
            }
        });
        assert_eq!(data, vec![0, 1, 2]);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn region_ranges_partition_every_cell_exactly_once() {
        for total_len in [0_usize, 1, 7, 4096, 50_000, 188_251] {
            for regions in [1_usize, 2, 3, 4, 8] {
                let ranges = super::region_ranges(total_len, regions);
                assert_eq!(ranges.len(), regions);
                let mut expected_start = 0;
                for &(start, len) in &ranges {
                    assert_eq!(start, expected_start, "region start not contiguous");
                    expected_start += len;
                }
                assert_eq!(expected_start, total_len, "regions don't cover total_len");
                let lens: Vec<usize> = ranges.iter().map(|&(_, len)| len).collect();
                let max = lens.iter().copied().max().unwrap_or(0);
                let min = lens.iter().copied().min().unwrap_or(0);
                assert!(max - min <= 1, "region sizes differ by more than one cell");
            }
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn broadcast_dispatch_matches_serial_dispatch_outside_a_pool() {
        // No pool on this thread: `current_thread_index()` is `None`, so
        // this exercises the serial fallback, not `rayon::broadcast`
        // (which would otherwise silently fall back to the *global* pool).
        assert!(rayon::current_thread_index().is_none());
        let n = 60_000;
        let mut serial = vec![0_i64; n];
        for_each_chunk_mut(&mut serial, |start, chunk| {
            for (local, v) in chunk.iter_mut().enumerate() {
                *v = i64::try_from(start + local).unwrap();
            }
        });

        let mut pooled = vec![0_i64; n];
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build 4-thread pool")
            .install(|| {
                for_each_chunk_mut(&mut pooled, |start, chunk| {
                    for (local, v) in chunk.iter_mut().enumerate() {
                        *v = i64::try_from(start + local).unwrap();
                    }
                });
            });

        assert_eq!(serial, pooled);
    }
}
