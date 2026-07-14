// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `Scale`.
//!
//! Covers unity identity, linear and dB factors, polarity inversion, non-finite
//! input sanitization, and non-finite factor rejection.

use bisque::mastering::Scale;
use bisque::processor::{DspError, KernelProcessor, ProcessSpec, Processor};
use bisque::testing::{bits_eq, sine, Contract};

/// A fresh, unprepared scale processor at a linear factor.
fn scale(factor: f64) -> KernelProcessor<Scale> {
    KernelProcessor::new(Scale::new(factor))
}

fn spec() -> ProcessSpec {
    ProcessSpec {
        sample_rate: 48_000,
        channels: 2,
        max_block: 512,
        max_memory: None,
    }
}

/// Audio behavior checks.
mod audio {
    use super::*;

    #[test]
    fn unity_factor_is_identity() {
        // Factor 1.0 is a bit-exact no-op.
        let input = sine(2, 777);
        let out = Contract::default().run(|| scale(1.0), &input, &[], 64);
        assert!(
            bits_eq(&out, &input),
            "factor 1.0 must be a bit-exact no-op"
        );
    }

    #[test]
    fn factor_scales_every_sample() {
        // 0.5 is exact in binary, so the multiply is exact too.
        let input = sine(2, 500);
        let out = Contract::default().run(|| scale(0.5), &input, &[], 64);
        for ch in 0..input.len() {
            for i in 0..input[ch].len() {
                let want = (f64::from(input[ch][i]) * 0.5) as f32;
                assert_eq!(out[ch][i], want, "ch{ch}[{i}]");
            }
        }
    }

    #[test]
    fn inverted_negates_every_sample() {
        // Factor -1.0 flips polarity exactly.
        let input = sine(2, 500);
        let out =
            Contract::default().run(|| KernelProcessor::new(Scale::inverted()), &input, &[], 64);
        for ch in 0..input.len() {
            for i in 0..input[ch].len() {
                assert_eq!(out[ch][i], -input[ch][i], "ch{ch}[{i}] polarity");
            }
        }
    }

    #[test]
    fn from_db_matches_independent_factor() {
        // -6 dB equals input * 10^(-6/20), independent of the impl's exp path.
        let input = sine(2, 500);
        let out = Contract::default().run(
            || KernelProcessor::new(Scale::from_db(-6.0)),
            &input,
            &[],
            64,
        );
        let expected = 10f64.powf(-6.0 / 20.0);
        for ch in 0..input.len() {
            for i in 0..input[ch].len() {
                let want = (f64::from(input[ch][i]) * expected) as f32;
                let tol = 1e-6 * want.abs().max(1e-6);
                assert!((out[ch][i] - want).abs() <= tol, "ch{ch}[{i}]");
            }
        }
    }

    #[test]
    fn non_finite_input_reads_as_silence() {
        // NaN and infinities are sanitized to 0 before the multiply.
        let input: Vec<Vec<f32>> = vec![
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25],
            vec![0.5, f32::NAN, 1.0, f32::INFINITY],
        ];
        let out = Contract::default().run(|| scale(2.0), &input, &[], 4);
        assert_eq!(out[0], vec![0.0, 0.0, 0.0, 0.5]);
        assert_eq!(out[1], vec![1.0, 0.0, 2.0, 0.0]);
    }
}

/// Rejected settings and specs.
mod validation {
    use super::*;

    #[test]
    fn non_finite_factor_is_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut p = scale(bad);
            assert!(
                matches!(
                    Processor::<f32>::prepare(&mut p, spec()),
                    Err(DspError::InvalidParam(_))
                ),
                "linear factor {bad} must be rejected"
            );
        }
    }

    #[test]
    fn from_db_non_finite_is_rejected_but_neg_inf_is_silence() {
        // +inf and NaN dB resolve to non-finite factors and are rejected.
        for bad_db in [f64::INFINITY, f64::NAN] {
            let mut p = KernelProcessor::new(Scale::from_db(bad_db));
            assert!(
                matches!(
                    Processor::<f32>::prepare(&mut p, spec()),
                    Err(DspError::InvalidParam(_))
                ),
                "from_db({bad_db}) must be rejected"
            );
        }
        // -inf dB resolves to a finite 0.0 (silence) and is allowed.
        let mut ok = KernelProcessor::new(Scale::from_db(f64::NEG_INFINITY));
        assert!(
            Processor::<f32>::prepare(&mut ok, spec()).is_ok(),
            "from_db(-inf) is silence, not an error"
        );
    }
}
