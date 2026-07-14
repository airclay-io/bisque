// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `Dither`.
//!
//! Covers quantization grid alignment, full-scale bounds, half-LSB unbiasedness,
//! seed reproducibility, channel decorrelation, validation, memory budget, and
//! memory footprint.

use bisque::mastering::{Dither, DitherSettings};
use bisque::processor::KernelProcessor;
use bisque::processor::{DspError, ProcessSpec, Processor};
use bisque::testing::{bits_eq, sine, Buffers, Contract};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// A fresh, unprepared 16-bit dither at the test seed, as `KernelProcessor`.
fn dither16() -> KernelProcessor<Dither> {
    KernelProcessor::new(Dither::with_settings(
        DitherSettings::new().bits(16).seed(SEED),
    ))
}

/// Prepare a dither with `T` fixed to f32 (so later method calls are
/// unambiguous), returning it behind `impl Processor<f32>` or the `prepare` error.
fn prepared(bits: u32, seed: u64, spec: ProcessSpec) -> Result<impl Processor<f32>, DspError> {
    let mut p = KernelProcessor::new(Dither::with_settings(
        DitherSettings::new().bits(bits).seed(seed),
    ));
    Processor::<f32>::prepare(&mut p, spec).map(|()| p)
}

/// The quantizer LSB for `bits`: `2^-(bits-1)`, exact.
fn step(bits: u32) -> f64 {
    1.0 / (1u64 << (bits - 1)) as f64
}

/// Audio behavior checks.
mod audio {
    use super::*;

    #[test]
    fn output_is_quantized_to_the_grid() {
        // Every output sample is an exact integer multiple of the LSB.
        let input = sine(2, 1500);
        let out = Contract::default().run(dither16, &input, &[], 64);
        let inv = (1u64 << 15) as f64;
        for (ch, plane) in out.iter().enumerate() {
            for &s in plane {
                let scaled = f64::from(s) * inv;
                assert!(
                    (scaled - scaled.round()).abs() == 0.0,
                    "ch{ch}: {s} is not on the 16-bit grid (scaled = {scaled})"
                );
            }
        }
    }

    #[test]
    fn output_is_bounded_at_and_beyond_full_scale() {
        // Over-range input clamps to the 16-bit full-scale interval.
        let n = 2000;
        let input: Buffers = (0..2)
            .map(|ch| {
                let w = 0.03 + 0.01 * ch as f32;
                (0..n).map(|i| 4.0 * (i as f32 * w).sin()).collect()
            })
            .collect();
        let out = Contract::default().run(dither16, &input, &[], 128);
        let lo = -1.0_f32;
        let hi = (((1u64 << 15) - 1) as f32) / ((1u64 << 15) as f32); // 32767/32768
        for (ch, plane) in out.iter().enumerate() {
            for &s in plane {
                assert!(
                    (lo..=hi).contains(&s),
                    "ch{ch}: {s} escaped [{lo}, {hi}] at/beyond full scale"
                );
            }
        }
    }

    #[test]
    fn dither_is_unbiased_at_half_lsb() {
        // A half-LSB DC input is unbiased over a long run.
        let s = step(16);
        let d = (0.5 * s) as f32; // = 2^-16, exactly representable
        let n = 200_000;
        let input: Buffers = vec![vec![d; n], vec![d; n]];
        let out = Contract::default().run(dither16, &input, &[], 512);
        for (ch, plane) in out.iter().enumerate() {
            let mean = plane.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
            // The tolerance is 0.1 LSB.
            assert!(
                (mean - f64::from(d)).abs() < 0.1 * s,
                "ch{ch}: dithered mean {mean} strayed from input {d} by > 0.1 LSB"
            );
        }
    }

    #[test]
    fn minimum_bit_depth_is_unbiased_at_silence() {
        let bits = 2;
        let n = 200_000;
        let input: Buffers = vec![vec![0.0; n], vec![0.0; n]];
        let out = Contract::default().run(
            || {
                KernelProcessor::new(Dither::with_settings(
                    DitherSettings::new().bits(bits).seed(SEED),
                ))
            },
            &input,
            &[],
            512,
        );
        let tolerance = 0.01 * step(bits);
        for (ch, plane) in out.iter().enumerate() {
            let mean = plane.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
            assert!(
                mean.abs() < tolerance,
                "ch{ch}: dithered silence has mean {mean}, tolerance {tolerance}"
            );
        }
    }
}

#[test]
fn same_seed_is_bit_exact() {
    // Two fresh instances at the same seed produce byte-identical output.
    let input = sine(2, 900);
    let c = Contract::default();
    let a = c.run(dither16, &input, &[], 64);
    let b = c.run(dither16, &input, &[], 64);
    assert!(bits_eq(&a, &b), "same seed must give bit-identical output");
}

#[test]
fn different_seed_differs() {
    // A different seed changes the output.
    let input = sine(2, 900);
    let c = Contract::default();
    let a = c.run(dither16, &input, &[], 64);
    let b = c.run(
        || {
            KernelProcessor::new(Dither::with_settings(
                DitherSettings::new().bits(16).seed(SEED ^ 0x1),
            ))
        },
        &input,
        &[],
        64,
    );
    assert!(!bits_eq(&a, &b), "a different seed must change the dither");
}

#[test]
fn block_size_invariance_is_bit_exact() {
    // Per-channel RNG state advances with samples, independent of host block
    // boundaries.
    let input = sine(2, 1000);
    Contract::default().assert_block_size_invariant(dither16, &input, &[]);
}

#[test]
fn reset_equivalence_no_state_leak() {
    // Reset reseeds every generator.
    let input = sine(2, 800);
    Contract::default().assert_reset_equivalence(dither16, &input, &[]);
}

#[test]
fn channels_are_decorrelated() {
    // Identical input channels receive independent per-channel dither streams.
    let mono = sine(1, 600);
    let input: Buffers = vec![mono[0].clone(), mono[0].clone()];
    let out = Contract::default().run(dither16, &input, &[], 64);
    let differ = out[0]
        .iter()
        .zip(&out[1])
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(differ, "per-channel dither streams must be decorrelated");
}

mod validation {
    use super::*;

    #[test]
    fn invalid_bit_depth_is_rejected() {
        // Bit depth is validated in prepare.
        let spec = Contract::default().spec;
        assert!(matches!(
            prepared(0, SEED, spec),
            Err(DspError::InvalidParam(_))
        ));
        assert!(matches!(
            prepared(1, SEED, spec),
            Err(DspError::InvalidParam(_))
        ));
        assert!(matches!(
            prepared(25, SEED, spec),
            Err(DspError::InvalidParam(_))
        ));
        assert!(prepared(2, SEED, spec).is_ok());
        assert!(prepared(16, SEED, spec).is_ok());
        assert!(prepared(24, SEED, spec).is_ok());
    }

    #[test]
    fn over_budget_is_rejected() {
        // A 1-byte memory cap is below the per-channel generator footprint.
        let tight = ProcessSpec {
            max_memory: Some(1),
            ..Contract::default().spec
        };
        assert!(matches!(
            prepared(16, SEED, tight),
            Err(DspError::OverBudget { .. })
        ));
        // A generous cap fits.
        let roomy = ProcessSpec {
            max_memory: Some(1 << 20),
            ..Contract::default().spec
        };
        assert!(prepared(16, SEED, roomy).is_ok());
    }
}

#[test]
fn memory_footprint_scales_with_channels() {
    // One generator is allocated per channel and counted in the footprint.
    let spec2 = ProcessSpec {
        channels: 2,
        ..Contract::default().spec
    };
    let spec4 = ProcessSpec {
        channels: 4,
        ..Contract::default().spec
    };
    let f2 = prepared(16, SEED, spec2).unwrap().memory_footprint();
    let f4 = prepared(16, SEED, spec4).unwrap().memory_footprint();
    assert!(f2 > 0, "dither has per-channel state");
    assert_eq!(f4, 2 * f2, "footprint must scale linearly with channels");
}
