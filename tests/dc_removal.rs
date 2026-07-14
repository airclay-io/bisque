// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Integration proof for the measure-then-subtract DC-removal path.
//!
//! `MeanMeter` measures a per-channel mean; `DcOffset` applies its negation.
//! This is the exact, spectrum-preserving alternative to `DcBlocker` that a
//! consumer's offline DC removal uses. No per-processor test covers the pairing.

#![cfg(all(feature = "analysis", feature = "repair", feature = "test-support"))]

use bisque::analysis::MeanMeter;
use bisque::processor::{KernelProcessor, Measurer};
use bisque::repair::DcOffset;
use bisque::testing::{observe_blocks, Contract};

mod audio {
    use super::*;

    #[test]
    fn measure_then_subtract_removes_per_channel_dc() {
        let frames = 4000;
        // A DC-biased stereo signal: distinct AC content and DC offset per
        // channel, so a single global offset would not remove both.
        let left: Vec<f32> = (0..frames)
            .map(|i| 0.5 * (i as f32 * 0.03).sin() + 0.3)
            .collect();
        let right: Vec<f32> = (0..frames)
            .map(|i| 0.4 * (i as f32 * 0.07).cos() - 0.2)
            .collect();
        let input = vec![left, right];
        let spec = Contract::default().spec; // 48 kHz stereo

        // Pass 1: measure per-channel means. The measurement is a separate pass
        // over the whole signal, before DcOffset is constructed.
        let mut meter = MeanMeter::new();
        Measurer::<f32>::prepare(&mut meter, spec).expect("prepare meter");
        observe_blocks(&mut meter, &input, 128);
        let offsets = [-meter.channel_mean(0), -meter.channel_mean(1)];

        // Pass 2: apply the negated means as a fixed per-channel offset.
        let out = Contract::default().run(
            || KernelProcessor::new(DcOffset::per_channel(offsets.to_vec())),
            &input,
            &[],
            128,
        );

        // Re-measure: each residual per-channel mean sits at the f32 floor, a
        // small multiple of f32 EPSILON (~1.19e-7) at unit signal scale.
        let mut check = MeanMeter::new();
        Measurer::<f32>::prepare(&mut check, spec).expect("prepare check");
        observe_blocks(&mut check, &out, 128);
        for ch in 0..2 {
            let residual = check.channel_mean(ch);
            assert!(
                residual.abs() < 1e-6,
                "ch{ch} residual DC {residual} exceeds the f32 floor"
            );
        }
    }
}
