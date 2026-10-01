//! Deterministic value hashing: the one place a phenomenon turns world
//! state (seed, day, cell) into a reproducible draw in `[0, 1)`.
//!
//! A stateful generator would be hidden state to checkpoint and would make
//! the same world diverge on restart; hashing the coordinates of the draw
//! instead keeps "one seed = one world" true down to the fires and the
//! weather regime, with nothing to serialise. Two phenomena needed the
//! same function (`fire`, `atmosphere::regime`), so it lives here rather
//! than being written twice with the same constants.

/// FNV-1a over the supplied 64-bit words, then a splitmix64 finalizer,
/// mapped to `[0, 1)`.
///
/// The caller passes the *coordinates of the draw* as words: a seed, a
/// day, cell coordinates, a salt separating independent streams. Order
/// matters (FNV-1a is sequential), so a caller must keep its word order
/// stable or its stream changes.
///
/// The `[0, 1)` conversion goes through the f32 bit pattern (23 bits of
/// the finalized hash into the mantissa of a float in `[1, 2)`, minus
/// one): integer work only, no lossy cast to silence.
#[must_use]
pub(crate) fn hash01(words: &[u64]) -> f32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in words {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    let mantissa = u32::try_from((h >> 41) & 0x007F_FFFF).unwrap_or(0);
    f32::from_bits(0x3F80_0000 | mantissa) - 1.0
}

/// Bit reinterpretation of a signed coordinate as a hash word: the two's
/// complement pattern widened, so `-1` and `4294967295` are the same word
/// and no sign is lost on the way in.
#[must_use]
pub(crate) fn coord_word(v: i32) -> u64 {
    u64::from(u32::from_ne_bytes(v.to_ne_bytes()))
}

#[cfg(test)]
mod tests {
    use super::{coord_word, hash01};

    /// The output is a real `[0, 1)`, never 1.0 and never negative, and
    /// the same words always give the same bits (that is the whole point:
    /// a checkpoint restart replays the same draws).
    #[test]
    fn hash01_stays_in_the_unit_interval_and_is_reproducible() {
        for seed in 0..8_u64 {
            for day in 0..64_u64 {
                let x = hash01(&[seed, day, 7]);
                assert!((0.0..1.0).contains(&x), "seed={seed} day={day} -> {x}");
                assert_eq!(x.to_bits(), hash01(&[seed, day, 7]).to_bits());
            }
        }
    }

    /// A different salt is a different stream: two draws sharing seed and
    /// day must not move together.
    #[test]
    fn salt_separates_streams() {
        let a: Vec<f32> = (0..256).map(|d| hash01(&[42, d, 0])).collect();
        let b: Vec<f32> = (0..256).map(|d| hash01(&[42, d, 1])).collect();
        let equal = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count();
        assert!(equal <= 1, "{equal} of 256 draws collided across salts");
    }

    /// Uniformity, coarse but enough to catch a broken finalizer: 4 096
    /// draws split into 8 buckets stay within ±30 % of an eighth each.
    #[test]
    fn draws_are_roughly_uniform() {
        let mut buckets = [0_u32; 8];
        for day in 0..4096_u64 {
            let x = hash01(&[42, day, 0]);
            // No float→int cast (clippy pedantic, and none is needed): the
            // bucket is found by walking the eight boundaries.
            let bucket = buckets
                .iter()
                .enumerate()
                .map(|(i, _)| i)
                .find(|&i| x < f32::from(u8::try_from(i + 1).unwrap()) / 8.0)
                .unwrap_or(7);
            buckets[bucket] += 1;
        }
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                (358..=666).contains(&count),
                "bucket {i}: {count} draws, expected ≈512"
            );
        }
    }

    #[test]
    fn coord_word_keeps_negative_coordinates_distinct() {
        assert_eq!(coord_word(0), 0);
        assert_eq!(coord_word(-1), u64::from(u32::MAX));
        assert_ne!(coord_word(-3), coord_word(3));
    }
}
