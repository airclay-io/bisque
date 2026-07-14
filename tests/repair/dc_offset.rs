// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `DcOffset`.
//!
//! Covers per-channel and broadcast application, non-finite input, and the
//! length, finiteness, and memory-budget validation done in `prepare`.

use bisque::processor::{DspError, KernelProcessor, ProcessSpec, Processor};
use bisque::repair::DcOffset;
use bisque::testing::{sine, Contract};

fn per_channel(offsets: &[f64]) -> KernelProcessor<DcOffset> {
    KernelProcessor::new(DcOffset::per_channel_from_slice(offsets))
}

fn broadcast(offset: f64) -> KernelProcessor<DcOffset> {
    KernelProcessor::new(DcOffset::broadcast(offset))
}

fn spec_channels(channels: usize) -> ProcessSpec {
    ProcessSpec {
        sample_rate: 48_000,
        channels,
        max_block: 512,
        max_memory: None,
    }
}

/// Audio behavior checks.
mod audio {
    use super::*;

    #[test]
    fn per_channel_offsets_apply_exactly() {
        let input = sine(2, 500);
        let offs = [0.05, -0.05];
        let out = Contract::default().run(|| per_channel(&offs), &input, &[], 64);
        for ch in 0..input.len() {
            for i in 0..input[ch].len() {
                let want = (f64::from(input[ch][i]) + offs[ch]) as f32;
                assert_eq!(out[ch][i], want, "ch{ch}[{i}]");
            }
        }
    }

    #[test]
    fn single_offset_broadcasts_to_all_channels() {
        let input = sine(2, 500);
        let out = Contract::default().run(|| broadcast(0.1), &input, &[], 64);
        for ch in 0..input.len() {
            for i in 0..input[ch].len() {
                let want = (f64::from(input[ch][i]) + 0.1) as f32;
                assert_eq!(out[ch][i], want, "ch{ch}[{i}]");
            }
        }
    }

    #[test]
    fn non_finite_input_becomes_offset_not_nan() {
        // NaN and infinities are sanitized to 0, so the output is the offset
        // value, never a propagated NaN.
        let input: Vec<Vec<f32>> = vec![
            vec![f32::NAN, 0.25, f32::INFINITY, -0.25],
            vec![f32::NEG_INFINITY, 0.5, f32::NAN, 1.0],
        ];
        let out = Contract::default().run(|| per_channel(&[0.1, -0.2]), &input, &[], 4);
        assert_eq!(
            out[0],
            vec![0.1, (0.25f64 + 0.1) as f32, 0.1, (-0.25f64 + 0.1) as f32]
        );
        assert_eq!(
            out[1],
            vec![-0.2, (0.5f64 - 0.2) as f32, -0.2, (1.0f64 - 0.2) as f32]
        );
    }
}

/// Rejected settings and specs.
mod validation {
    use super::*;

    #[test]
    fn wrong_length_is_rejected() {
        // Three per-channel offsets do not match a stereo spec.
        let mut p = per_channel(&[0.1, 0.2, 0.3]);
        assert!(matches!(
            Processor::<f32>::prepare(&mut p, spec_channels(2)),
            Err(DspError::InvalidParam(_))
        ));
    }

    #[test]
    fn non_finite_offset_is_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut broadcast = broadcast(bad);
            assert!(matches!(
                Processor::<f32>::prepare(&mut broadcast, spec_channels(2)),
                Err(DspError::InvalidParam(_))
            ));

            let mut p = per_channel(&[0.1, bad]);
            assert!(
                matches!(
                    Processor::<f32>::prepare(&mut p, spec_channels(2)),
                    Err(DspError::InvalidParam(_))
                ),
                "offset {bad} must be rejected"
            );
        }
    }

    #[test]
    fn over_tight_max_memory_is_over_budget_per_channel() {
        // Broadcast state is inline, so cover the owned per-channel buffer.
        let footprint = 2 * std::mem::size_of::<f64>(); // 16 for stereo

        let mut fits = per_channel(&[0.1, -0.1]);
        let mut spec = spec_channels(2);
        spec.max_memory = Some(footprint);
        assert!(
            Processor::<f32>::prepare(&mut fits, spec).is_ok(),
            "a cap of the exact footprint must fit"
        );
        assert_eq!(
            Processor::<f32>::memory_footprint(&fits),
            footprint,
            "reported footprint must match the budgeted offset storage"
        );

        let mut over = per_channel(&[0.1, -0.1]);
        spec.max_memory = Some(footprint - 1);
        assert!(
            matches!(
                Processor::<f32>::prepare(&mut over, spec),
                Err(DspError::OverBudget { .. })
            ),
            "one byte under the footprint must be OverBudget"
        );
    }

    #[test]
    fn broadcast_prepares_at_any_channel_count() {
        for ch in [1usize, 2, 4, 6] {
            let mut p = broadcast(0.1);
            assert!(
                Processor::<f32>::prepare(&mut p, spec_channels(ch)).is_ok(),
                "broadcast must prepare at {ch} channels"
            );
            assert_eq!(p.memory_footprint(), 0, "broadcast state is inline");
        }
    }

    #[test]
    fn one_per_channel_offset_does_not_implicitly_broadcast() {
        let mut p = per_channel(&[0.1]);
        assert!(Processor::<f32>::prepare(&mut p, spec_channels(1)).is_ok());
        assert!(matches!(
            Processor::<f32>::prepare(&mut p, spec_channels(2)),
            Err(DspError::InvalidParam(_))
        ));
    }
}
