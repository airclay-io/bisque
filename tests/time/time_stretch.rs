// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `TimeStretch`.
//!
//! Covers DC reconstruction, 1.0x identity, duration ratio, output block-size
//! invariance, reset behavior, underrun resilience, full consumption, and
//! validation.

use bisque::processor::{AudioBlockMut, DspError, ProcessSpec, RingSource, VariableRate};
use bisque::testing::{bits_eq, tone_stereo, Buffers, Contract};
use bisque::time::{TimeStretch, TimeStretchSettings};

/// Test-side copy of the window and hop sizes.
const W: usize = 1024;
const HS: usize = W / 2;

fn ts(stretch: f64) -> TimeStretch<f32> {
    TimeStretch::with_settings(TimeStretchSettings::new().stretch(stretch))
}

mod audio {
    use super::*;

    #[test]
    fn dc_is_reconstructed_exactly() {
        // Boundary normalization keeps a constant signal flat from the first
        // output frame through the last for every ratio.
        let c = 0.5f32;
        let n_in = 40 * 1024;
        let input: Buffers = vec![vec![c; n_in]; 2];
        for &stretch in &[0.5, 1.0, 1.5, 2.0] {
            let out = Contract::default().stretch(|| ts(stretch), &input, 333, usize::MAX);
            for plane in &out {
                for (i, &v) in plane.iter().enumerate() {
                    assert!(
                        (v - c).abs() < 1e-5,
                        "DC must reconstruct at {stretch}x: got {v} at {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn unity_ratio_is_identity_no_dup_or_skip() {
        // At 1.0x, Ha == HS and the complete output equals the input.
        let n = 8 * 1024;
        let ramp: Buffers = (0..2)
            .map(|ch| {
                (0..n)
                    .map(|i| (i as f32 * 1e-4) + ch as f32 * 0.25)
                    .collect()
            })
            .collect();
        let out = Contract::default().stretch(|| ts(1.0), &ramp, 256, usize::MAX);
        assert!(bits_eq(&out, &ramp), "unity stretch must be bit-exact");
        for (out_ch, ramp_ch) in out.iter().zip(&ramp) {
            assert_eq!(out_ch.len(), n, "unity stretch must preserve duration");
            for (k, (&o, &r)) in out_ch.iter().zip(ramp_ch).enumerate() {
                assert!(
                    (o - r).abs() < 1e-5,
                    "1.0x must be identity at {k}: out {o} vs in {r}"
                );
            }
        }
    }

    #[test]
    fn a_single_leading_sample_survives_unity_stretch() {
        let input: Buffers = vec![vec![0.75], vec![-0.25]];
        let out = Contract::default().stretch(|| ts(1.0), &input, 64, usize::MAX);
        assert_eq!(out, input);
    }

    #[test]
    fn duration_scales_by_the_ratio() {
        // Output length is the input length times the effective ratio, rounded
        // to the nearest frame.
        let n = 48_000;
        let input = tone_stereo(n);
        for &stretch in &[0.5, 1.5, 2.0] {
            let out = Contract::default().stretch(|| ts(stretch), &input, 512, usize::MAX);
            let out_len = out[0].len();
            let ha = (HS as f64 / stretch).round() as usize;
            let expected = (n * HS + ha / 2) / ha;
            assert_eq!(out_len, expected, "{stretch}x output duration");
            assert!(
                (out_len as f64 / n as f64 - stretch).abs() < 0.05,
                "{stretch}x: measured ratio {} off target",
                out_len as f64 / n as f64
            );
        }
    }

    #[test]
    fn short_streams_preserve_dc_and_have_exact_duration() {
        for &n in &[0usize, 1, 2, 255, 511, 512, 513, 1023, 1024, 1025] {
            let input: Buffers = vec![vec![0.25; n]; 2];
            for &stretch in &[0.5, 0.75, 1.0, 1.3, 2.0] {
                let out = Contract::default().stretch(|| ts(stretch), &input, 73, 17);
                let ha = (HS as f64 / stretch).round() as usize;
                let expected = (n * HS + ha / 2) / ha;
                assert_eq!(out[0].len(), expected, "{n} frames at {stretch}x");
                assert!(
                    out.iter().flatten().all(|&sample| sample == 0.25),
                    "DC changed for {n} frames at {stretch}x"
                );
            }
        }
    }

    // Plain OLA does not preserve tone amplitude. DC is the amplitude check.
}

#[test]
fn block_size_invariance_is_bit_exact_on_output_timeline() {
    let input = tone_stereo(20 * 1024);
    Contract::default().assert_stretch_block_size_invariant(|| ts(1.7), &input);
}

#[test]
fn reset_equivalence_no_state_leak() {
    let input = tone_stereo(12 * 1024);
    let c = Contract::default();
    let fresh = c.stretch(|| ts(1.5), &input, 64, usize::MAX);

    let mut v = ts(1.5);
    v.prepare(c.spec).expect("prepare");
    // Advance state with an odd output block and capped source.
    let _ = c.stretch_reusing(&mut v, &input, 50, 11);
    v.reset();
    let after = c.stretch_reusing(&mut v, &input, 64, usize::MAX);

    assert!(
        bits_eq(&after, &fresh),
        "reset must reproduce a fresh stretcher"
    );
}

#[test]
fn underrun_resilience_matches_unlimited() {
    // A capped source yields the same stream as an unlimited source.
    let input = tone_stereo(16 * 1024);
    let c = Contract::default();
    let unlimited = c.stretch(|| ts(1.3), &input, 256, usize::MAX);
    let dribbled = c.stretch(|| ts(1.3), &input, 256, 13);
    assert!(
        bits_eq(&unlimited, &dribbled),
        "a partial-pull source must not change the output"
    );
}

#[test]
fn mono_reconstructs_dc_and_stays_invariant() {
    // Mono exercises FIFO frame counting independently of channel count.
    let mut c = Contract::default();
    c.spec.channels = 1;
    let cval = 0.4f32;
    let n_in = 30 * 1024;
    let dc: Buffers = vec![vec![cval; n_in]];
    let stretch = 1.5;
    let out = c.stretch(|| ts(stretch), &dc, 257, usize::MAX);
    let ha = (HS as f64 / stretch).round() as usize;
    let safe_end = ((n_in - W) / ha) * HS;
    for (i, &v) in out[0][..safe_end].iter().enumerate().skip(W) {
        assert!(
            (v - cval).abs() < 1e-5,
            "mono DC must reconstruct: {v} at {i}"
        );
    }
    let tone: Buffers = vec![(0..n_in).map(|i| (i as f32 * 0.01).sin() * 0.5).collect()];
    c.assert_stretch_block_size_invariant(|| ts(1.3), &tone);
}

#[test]
fn sub_window_and_empty_input_terminate_cleanly() {
    // Input at or below one window terminates and produces bounded output.
    let c = Contract::default();
    for &n_in in &[0usize, 1, 7, 500, 1023, 1024] {
        let input = tone_stereo(n_in);
        let out = c.stretch(|| ts(1.5), &input, 64, usize::MAX);
        assert!(
            out[0].len() <= 4 * W,
            "n_in {n_in}: output {} should be bounded by a few windows",
            out[0].len()
        );
    }
}

#[test]
fn input_is_fully_consumed() {
    // Drive manually so the source can be inspected afterwards.
    let input = tone_stereo(10 * 1024);
    let c = Contract::default();
    let mut v = ts(2.0);
    v.prepare(c.spec).expect("prepare");
    let mut src = RingSource::new(input.clone());

    let mut stage: Buffers = vec![vec![0.0f32; 300]; 2];
    let mut done = false;
    let mut guard = 0;
    while !done {
        let produced = {
            let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
            let mut blk = AudioBlockMut::new(&mut planes);
            v.process(&mut src, &mut blk)
        };
        done = produced.done;
        guard += 1;
        assert!(guard < 100_000, "stretch did not terminate");
    }
    assert_eq!(src.remaining(), 0, "every input frame must be pulled");
}

mod validation {
    use super::*;

    #[test]
    fn channel_count_bounds_are_enforced() {
        // `prepare` accepts 1..=16 channels and rejects 0 and 17.
        let spec = |channels: usize| ProcessSpec {
            channels,
            ..Contract::default().spec
        };
        assert!(
            matches!(ts(1.0).prepare(spec(0)), Err(DspError::UnsupportedSpec(_))),
            "zero channels must be rejected"
        );
        ts(1.0)
            .prepare(spec(TimeStretch::<f32>::MAX_CHANNELS))
            .expect("the documented maximum channel count must be accepted");
        assert!(
            matches!(ts(1.0).prepare(spec(17)), Err(DspError::UnsupportedSpec(_))),
            "17 channels (one past the maximum) must be rejected"
        );
    }

    #[test]
    fn effective_ratio_is_available_after_prepare() {
        let requested = 1.3;
        let mut stretch = ts(requested);
        assert_eq!(stretch.stretch(), requested);
        assert_eq!(stretch.effective_stretch(), None);
        stretch
            .prepare(Contract::default().spec)
            .expect("valid preparation");
        let ha = (HS as f64 / requested).round();
        assert_eq!(stretch.effective_stretch(), Some(HS as f64 / ha));
    }
}
