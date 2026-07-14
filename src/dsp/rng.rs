// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Crate-internal seeded PRNG primitives.
//!
//! Shared by the seeded processors (`WhiteNoise`, `Dither`). Both the generator
//! and the per-channel seed derivation are fixed algorithms with pinned
//! constants: output is bit-exact across platforms and releases, which the
//! committed snapshots rely on. This module is not public API.

/// The `xorshift64` fixed-point escape and `splitmix64` golden-ratio constant.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// A seeded `xorshift64` generator.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rng {
    pub(crate) state: u64,
}

impl Rng {
    /// Seed the generator. Zero is replaced with a fixed non-zero value, since
    /// zero is the `xorshift64` fixed point.
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { GOLDEN } else { seed },
        }
    }

    /// Next 64-bit output, advancing the state.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// A uniform value in `[0, 1)` from the top 53 bits.
    pub(crate) fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform value in `[-1, 1)` from the top 53 bits.
    ///
    /// Used by the `generators` noise source; `mastering`-only builds compile
    /// the module without a caller.
    #[cfg_attr(not(feature = "generators"), allow(dead_code))]
    pub(crate) fn next_bipolar(&mut self) -> f64 {
        self.next_unit() * 2.0 - 1.0
    }
}

/// Derive a per-channel seed from the base seed via one `splitmix64` round.
///
/// Zero results are remapped like [`Rng::new`] so no channel can land on the
/// `xorshift64` fixed point.
pub(crate) fn channel_seed(base: u64, ch: usize) -> u64 {
    let mut z = base.wrapping_add(GOLDEN.wrapping_mul(ch as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 {
        GOLDEN
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::{channel_seed, Rng};

    /// Reference calculation for the `splitmix64` round used by `channel_seed`.
    fn splitmix64_ref(base: u64, ch: usize) -> u64 {
        const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
        fn mix(z: u64, shift: u32, mul: u64) -> u64 {
            (z ^ (z >> shift)).wrapping_mul(mul)
        }
        let seed = base.wrapping_add(GOLDEN.wrapping_mul((ch as u64).wrapping_add(1)));
        let a = mix(seed, 30, 0xBF58_476D_1CE4_E5B9);
        let b = mix(a, 27, 0x94D0_49BB_1331_11EB);
        let out = b ^ (b >> 31);
        if out == 0 {
            GOLDEN
        } else {
            out
        }
    }

    #[test]
    fn channel_seed_matches_independent_splitmix64() {
        // Three channels from a non-zero base match the reference values.
        let base = 0xABCD_1234_5678u64;
        for ch in 0..3 {
            assert_eq!(
                channel_seed(base, ch),
                splitmix64_ref(base, ch),
                "channel_seed disagrees with the reference at ch{ch}"
            );
        }
        assert_eq!(channel_seed(base, 0), 0xC332_6DED_851F_51AD);
        assert_eq!(channel_seed(base, 1), 0x13F0_5B3A_52B5_D42A);
        assert_eq!(channel_seed(base, 2), 0x5ABB_5F2D_7498_01C1);
    }

    #[test]
    fn rng_next_u64_is_the_exact_xorshift_sequence() {
        // xorshift64 with the documented (13, 7, 17) shifts.
        let mut r = Rng::new(0x1234_5678_9ABC_DEF0);
        let one = r.next_u64();
        let mut x = 0x1234_5678_9ABC_DEF0u64;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        assert_eq!(one, x, "first xorshift output");
        assert_eq!(one, 0xFE80_0D65_69FA_1B4D, "first xorshift output (pinned)");
        // A second step advances state.
        assert_ne!(r.next_u64(), one, "the generator must advance");
    }

    #[test]
    fn rng_seed_zero_is_remapped() {
        // A zero seed is remapped away from the xorshift64 fixed point.
        let mut zero = Rng::new(0);
        assert_ne!(zero.next_u64(), 0, "a 0 seed must be remapped, not frozen");
    }
}
