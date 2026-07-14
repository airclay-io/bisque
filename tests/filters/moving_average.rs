// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `MovingAverage`.
//!
//! Covers split I/O, rectangular impulse response, unity DC gain, latency,
//! tail declaration, flush draining, block-size invariance, reset behavior,
//! and memory footprint.

use bisque::filters::MovingAverage;
use bisque::processor::KernelProcessor;
use bisque::processor::{AudioBlockMut, Kernel, Processor, Produced};
use bisque::testing::{bits_eq, sine, Buffers, Contract};

/// A wrapped moving-average FIR over `taps` samples.
fn ma(taps: usize) -> KernelProcessor<MovingAverage> {
    KernelProcessor::new(MovingAverage::new(taps))
}

mod validation {
    use super::*;
    use bisque::processor::DspError;

    #[test]
    fn zero_taps_is_rejected_not_clamped() {
        // Structural policy: the constructed configuration is preserved and
        // `prepare` rejects it rather than silently clamping to one tap.
        let mut p = ma(0);
        assert!(matches!(
            Processor::<f32>::prepare(&mut p, Contract::default().spec),
            Err(DspError::InvalidParam(_))
        ));
    }
}

mod audio {
    use super::*;

    #[test]
    fn impulse_response_is_rectangular() {
        // A length-N moving average has impulse response 1/N for the first N
        // samples, then zero.
        let taps = 9;
        let mut input: Buffers = vec![vec![0.0f32; 100]; 2];
        input[0][0] = 1.0;
        input[1][0] = 1.0;
        let out = Contract::default().run_split(|| ma(taps), &input, 16);
        let expected = 1.0 / taps as f32;
        for (i, &v) in out[0].iter().take(taps).enumerate() {
            assert!(
                (v - expected).abs() < 1e-6,
                "tap {i} should be 1/N = {expected}, got {v}"
            );
        }
        assert!(
            out[0][taps].abs() < 1e-6,
            "after the window the response is zero"
        );
    }

    #[test]
    fn dc_gain_is_unity() {
        // The average of a constant is the constant once the ring has filled.
        let taps = 8;
        let input: Buffers = vec![vec![0.5f32; 100]; 2];
        let out = Contract::default().run_split(|| ma(taps), &input, 16);
        for (i, &v) in out[0].iter().enumerate().skip(taps) {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "settled DC gain should be 1 (got {v} at {i})"
            );
        }
    }

    #[test]
    fn non_finite_input_is_treated_as_silence() {
        let mut bad = sine(2, 512);
        let mut clean = bad.clone();
        bad[0][17] = f32::NAN;
        bad[1][300] = f32::INFINITY;
        clean[0][17] = 0.0;
        clean[1][300] = 0.0;
        let bad_out = Contract::default().run_split(|| ma(16), &bad, 64);
        let clean_out = Contract::default().run_split(|| ma(16), &clean, 64);
        assert!(bits_eq(&bad_out, &clean_out));
    }
}

mod contract {
    use super::*;
    use bisque::processor::Tail;

    /// Flush a prepared processor into a `frames`-frame stereo stage.
    fn flush_stage(proc: &mut impl Processor<f32>, frames: usize) -> (Buffers, Produced) {
        let mut stage: Buffers = vec![vec![0.0f32; frames]; 2];
        let produced = {
            let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
            let mut out = AudioBlockMut::new(&mut planes);
            proc.flush(&mut out)
        };
        (stage, produced)
    }

    #[test]
    fn tail_is_taps_minus_one_and_flush_completes_the_response() {
        // An impulse on the final body frame still has taps - 1 frames of FIR
        // response in the ring. Flushing recovers exactly that continuation,
        // then reports done.
        let taps = 9;
        let n = 64;
        let mut input: Buffers = vec![vec![0.0f32; n]; 2];
        input[0][n - 1] = 1.0;
        input[1][n - 1] = 1.0;
        let mut proc = ma(taps);
        let c = Contract::default();
        Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
        assert_eq!(
            Processor::<f32>::tail(&proc),
            Tail::Frames(taps - 1),
            "the tail is the remaining FIR response"
        );
        let body = c.run_split_reusing(&mut proc, &input, 16);
        let expected = body[0][n - 1]; // 1/taps, as the body rendered it

        let (tail, produced) = flush_stage(&mut proc, taps);
        assert_eq!(produced.frames, taps - 1, "the drain is exactly taps - 1");
        assert!(produced.done, "a fully drained FIR tail reports done");
        for (ch, plane) in tail.iter().enumerate() {
            for (i, &v) in plane.iter().take(taps - 1).enumerate() {
                assert_eq!(
                    v, expected,
                    "ch{ch}: flushed frame {i} continues the 1/N response"
                );
            }
        }
    }

    #[test]
    fn flush_equals_processing_zeros() {
        // Draining the tail must be byte-identical to feeding explicit
        // silence: flush is the same convolution with the same ring.
        let taps = 16;
        let input = sine(2, 500);
        let c = Contract::default();

        let mut flushed = ma(taps);
        Processor::<f32>::prepare(&mut flushed, c.spec).expect("prepare");
        let _ = c.run_split_reusing(&mut flushed, &input, 64);
        let (tail, _) = flush_stage(&mut flushed, taps - 1);

        let mut zero_fed = ma(taps);
        Processor::<f32>::prepare(&mut zero_fed, c.spec).expect("prepare");
        let _ = c.run_split_reusing(&mut zero_fed, &input, 64);
        let zeros: Buffers = vec![vec![0.0f32; taps - 1]; 2];
        let out = c.run_split_reusing(&mut zero_fed, &zeros, 64);

        assert!(
            bits_eq(&tail, &out),
            "the drained tail must match zero-fed processing bit for bit"
        );
    }
}

#[test]
fn latency_is_the_group_delay() {
    // Integer host latency is the floor of the exact linear-phase group delay.
    for (taps, want) in [(9usize, 4usize), (8, 3), (6, 2)] {
        let mut p = ma(taps);
        Processor::<f32>::prepare(&mut p, Contract::default().spec).expect("prepare");
        assert_eq!(
            Processor::<f32>::latency(&p),
            want,
            "taps={taps}: latency is floor((taps - 1) / 2)"
        );
    }

    assert_eq!(MovingAverage::new(9).group_delay_frames(), 4.0);
    assert_eq!(MovingAverage::new(16).group_delay_frames(), 7.5);
}

#[test]
fn memory_footprint_is_exactly_the_state_bytes() {
    // The kernel state is one length-`taps` f64 ring plus a compensated sum per
    // channel.
    let taps = 16;
    let mut k = MovingAverage::new(taps);
    let spec = Contract::default().spec;
    Kernel::<f32>::prepare(&mut k, spec).expect("prepare");
    let expected = spec.channels * (taps + 2) * std::mem::size_of::<f64>();
    assert_eq!(
        Kernel::<f32>::memory_footprint(&k),
        expected,
        "footprint includes each ring and its compensated sum"
    );
}

#[test]
fn block_size_invariance_is_bit_exact() {
    let input = sine(2, 1000);
    let c = Contract::default();
    let reference = c.run_split(|| ma(16), &input, 1000);
    for &block in &[1usize, 7, 32, 64, 128, 257] {
        let out = c.run_split(|| ma(16), &input, block);
        assert!(bits_eq(&out, &reference), "block size {block} diverged");
    }
}

#[test]
fn reset_equivalence_no_state_leak() {
    let input = sine(2, 800);
    let c = Contract::default();
    let fresh = c.run_split(|| ma(16), &input, 64);

    let mut proc = ma(16);
    Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
    let _ = c.run_split_reusing(&mut proc, &input, 50); // dirty the history ring
    Processor::<f32>::reset(&mut proc);
    let after = c.run_split_reusing(&mut proc, &input, 64);

    assert!(bits_eq(&after, &fresh), "reset must reproduce a fresh FIR");
}
