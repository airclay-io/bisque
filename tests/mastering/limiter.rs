// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `Limiter`.
//!
//! Covers ceiling behavior, transparency below threshold, true-peak limiting,
//! latency, flush behavior, block-size invariance, reset behavior, validation,
//! memory footprint, tail shape, and snapshots.

use std::f64::consts::LN_10;

use bisque::analysis::TruePeakMeter;
use bisque::mastering::{Limiter, LimiterSettings};
use bisque::processor::KernelProcessor;
use bisque::processor::{
    AudioBlockMut, DspError, Measurer, ProcessSpec, Processor, Produced, Tail,
};
use bisque::testing::{bits_eq, ev, observe_blocks, sine, Buffers, Contract};

const THRESHOLD: bisque::parameter::ParamId = Limiter::THRESHOLD_DB;

/// A fresh, unprepared limiter at the mastering defaults.
fn limiter() -> KernelProcessor<Limiter> {
    KernelProcessor::new(Limiter::new())
}

/// A fresh, unprepared limiter with explicit settings.
fn limiter_with(settings: LimiterSettings) -> KernelProcessor<Limiter> {
    KernelProcessor::new(Limiter::with_settings(settings))
}

/// Prepare a limiter behind `impl Processor<f32>`, or surface the `prepare` error.
fn prepared(spec: ProcessSpec) -> Result<impl Processor<f32>, DspError> {
    let mut p = KernelProcessor::new(Limiter::new());
    Processor::<f32>::prepare(&mut p, spec).map(|()| p)
}

/// The default contract spec with the channel count overridden.
fn spec_ch(channels: usize) -> ProcessSpec {
    ProcessSpec {
        channels,
        ..Contract::default().spec
    }
}

/// Convert dBFS to linear amplitude.
fn thresh_lin(db: f64) -> f64 {
    (db * (LN_10 / 20.0)).exp()
}

/// A loud two-channel sine at `amp` (above the ceiling, to exercise limiting).
fn loud(frames: usize, amp: f32) -> Buffers {
    (0..2)
        .map(|ch| {
            let w = 0.02 + 0.005 * ch as f32;
            (0..frames).map(|i| amp * (i as f32 * w).sin()).collect()
        })
        .collect()
}

mod audio {
    use super::*;

    #[test]
    fn transparent_below_threshold() {
        // A signal under the ceiling is delayed by lookahead and otherwise unchanged.
        let input = sine(2, 1000); // amplitude < the -1 dBFS ceiling
        let out = Contract::default().run(limiter, &input, &[], 64);
        let l = prepared(Contract::default().spec).unwrap().latency();
        for (ch, plane) in out.iter().enumerate() {
            for &s in &plane[..l] {
                assert!(s.abs() < 1e-6, "ch{ch}: pre-roll must be silent");
            }
        }
        for ch in 0..input.len() {
            for k in 0..(1000 - l) {
                assert!(
                    (out[ch][l + k] - input[ch][k]).abs() < 1e-6,
                    "ch{ch}: below threshold must pass through, delayed by {l}"
                );
            }
        }
    }

    #[test]
    fn output_stays_under_threshold() {
        // Output samples stay under the ceiling.
        let input = loud(4000, 2.0);
        let out = Contract::default().run(limiter, &input, &[], 128);
        let ceiling = thresh_lin(-1.0) as f32;
        for (ch, plane) in out.iter().enumerate() {
            for &s in plane {
                assert!(
                    s.abs() <= ceiling + 1e-4,
                    "ch{ch}: output {s} exceeded ceiling {ceiling}"
                );
            }
        }
    }

    #[test]
    fn default_margin_lowers_the_internal_gain_target() {
        let input = loud(4000, 2.0);
        let out = Contract::default().run(limiter, &input, &[], 128);
        let target = thresh_lin(-1.1) as f32;
        for (ch, plane) in out.iter().enumerate() {
            for &s in plane {
                assert!(
                    s.abs() <= target + 1e-4,
                    "ch{ch}: output {s} exceeded default margin target {target}"
                );
            }
        }
    }

    #[test]
    fn zero_margin_uses_the_requested_threshold_as_the_target() {
        let input = loud(4000, 2.0);
        let make = || limiter_with(LimiterSettings::new().true_peak_margin_db(0.0));
        let out = Contract::default().run(make, &input, &[], 128);
        let target = thresh_lin(-1.0) as f32;
        let saw_near_target = out
            .iter()
            .flatten()
            .any(|sample| (sample.abs() - target).abs() < 1e-3);
        assert!(
            saw_near_target,
            "zero margin should allow limiting near the requested threshold"
        );
    }

    #[test]
    fn output_true_peak_is_held_near_the_ceiling() {
        // Inter-sample peak is bounded near the ceiling.
        let v = std::f32::consts::FRAC_1_SQRT_2 * 1.4; // sample ~0.99, true peak ~1.4
        let pat = [v, v, -v, -v];
        let ch: Vec<f32> = (0..4000).map(|i| pat[i % 4]).collect();
        let input: Buffers = vec![ch.clone(), ch];
        let out = Contract::default().run(limiter, &input, &[], 128);

        let mut tpm = TruePeakMeter::new();
        Measurer::<f32>::prepare(&mut tpm, Contract::default().spec).expect("prepare");
        observe_blocks(&mut tpm, &out, 128);
        let out_tp = Measurer::<f32>::read(&tpm);

        let ceiling = thresh_lin(-1.0);
        assert!(
            out_tp <= ceiling + 0.02,
            "output true peak {out_tp} should stay under the ceiling {ceiling} with the default margin"
        );
    }

    #[test]
    fn flushed_tail_holds_the_true_peak_ceiling() {
        // An inter-sample-peak burst inside the detector's group delay (6
        // frames) of end-of-input: the sample peak stays under the ceiling,
        // so only the true-peak detector can catch it, and its report can
        // only arrive after the final render push. The burst itself is still
        // in the delay line, so the drained tail must carry the reduction.
        let n = 600;
        let v = 0.85f32; // sample peak 0.85, inter-sample true peak ~1.2
        let pat = [v, v, -v, -v];
        let mut ch = vec![0.0f32; n];
        for (i, s) in ch.iter_mut().enumerate().skip(n - 6) {
            *s = pat[i % 4];
        }
        let input: Buffers = vec![ch.clone(), ch];
        let minimum_lookahead_ms = 11.0 * 1000.0 / 48_000.0;
        let mut proc = limiter_with(LimiterSettings::new().lookahead_ms(minimum_lookahead_ms));
        Processor::<f32>::prepare(&mut proc, Contract::default().spec).unwrap();
        let l = proc.latency();
        assert_eq!(l, 11, "test must exercise the minimum lookahead");
        let body = Contract::default().run_reusing(&mut proc, &input, &[], 64);

        let mut tail: Buffers = vec![vec![0.0f32; l]; 2];
        let produced = {
            let mut planes: Vec<&mut [f32]> = tail.iter_mut().map(Vec::as_mut_slice).collect();
            let mut out = AudioBlockMut::new(&mut planes);
            proc.flush(&mut out)
        };
        assert_eq!(produced.frames, l, "the whole tail drains in one call");

        // Measure body + tail + trailing silence: the meter shares the
        // detector's group delay, so its window must sweep past the end.
        let mut full = body;
        for (plane, t) in full.iter_mut().zip(&tail) {
            plane.extend_from_slice(t);
            plane.resize(plane.len() + 64, 0.0);
        }
        let mut tpm = TruePeakMeter::new();
        Measurer::<f32>::prepare(&mut tpm, Contract::default().spec).expect("prepare");
        observe_blocks(&mut tpm, &full, 128);
        let out_tp = Measurer::<f32>::read(&tpm);

        let ceiling = thresh_lin(-1.0);
        assert!(
            out_tp <= ceiling + 0.02,
            "flushed-tail true peak {out_tp} must stay under the ceiling {ceiling}"
        );
    }

    #[test]
    fn latency_matches_lookahead_and_delays_the_impulse() {
        // At 48 kHz, 1.5 ms of lookahead is 72 samples.
        let l = prepared(Contract::default().spec).unwrap().latency();
        assert_eq!(l, 72, "1.5 ms lookahead at 48 kHz");

        let mut imp = vec![0.0f32; 500];
        imp[0] = 0.5; // below the ceiling, so it passes through
        let input = vec![imp.clone(), imp];
        let out = Contract::default().run(limiter, &input, &[], 64);
        assert!(
            (out[0][l] - 0.5).abs() < 1e-6,
            "impulse must appear at frame {l}"
        );
        for (i, &s) in out[0].iter().enumerate() {
            if i != l {
                assert!(
                    s.abs() < 1e-6,
                    "only frame {l} should be non-zero (saw {s} at {i})"
                );
            }
        }
    }

    #[test]
    fn attack_has_no_step_discontinuity() {
        // A quiet-then-loud DC step. The applied gain (output over delayed
        // input) descends in raised-cosine increments spread across the
        // available attack interval, not in a single instant-attack step.
        let n = 1000;
        let step = 400;
        let ch: Vec<f32> = (0..n).map(|i| if i < step { 0.5 } else { 2.0 }).collect();
        let input: Buffers = vec![ch.clone(), ch];
        let out = Contract::default().run(limiter, &input, &[], 64);
        let l = prepared(Contract::default().spec).unwrap().latency();

        // Recover the per-frame applied gain where the delayed input is known
        // and nonzero (both DC levels are nonzero, so every frame qualifies).
        let gains: Vec<f64> = (l..n)
            .map(|i| f64::from(out[0][i]) / f64::from(input[0][i - l]))
            .collect();
        let max_gain = gains.iter().copied().fold(f64::MIN, f64::max);
        let min_gain = gains.iter().copied().fold(f64::MAX, f64::min);
        let drop = max_gain - min_gain;
        assert!(
            drop > 0.3,
            "the step must drive substantial gain reduction, got a drop of {drop}"
        );
        let max_delta = gains
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_delta < drop / 4.0,
            "attack must be spread over the available interval: max per-frame gain step \
             {max_delta} vs total gain drop {drop}"
        );
    }

    #[test]
    fn gain_reaches_minimum_when_the_peak_plays() {
        // An isolated over-threshold impulse: at the exact frame the delayed
        // peak emerges, the raised-cosine attack has fully reached the required
        // gain, so the sample respects the internal target.
        let n = 500;
        let at = 200;
        let mut ch = vec![0.0f32; n];
        ch[at] = 1.5;
        let input: Buffers = vec![ch.clone(), ch];
        let out = Contract::default().run(limiter, &input, &[], 64);
        let l = prepared(Contract::default().spec).unwrap().latency();
        let target = thresh_lin(-1.1) as f32; // ceiling minus the default margin
        let peak = out[0][at + l].abs();
        assert!(
            peak <= target + 1e-4,
            "the delayed peak {peak} must respect the target {target} at frame {}",
            at + l
        );
        assert!(
            peak > 0.1,
            "the limited impulse must still play, got {peak}"
        );
    }

    #[test]
    fn flush_drains_tail_and_conserves_signal() {
        // process plus flush reconstructs the delayed signal.
        let n = 600;
        let input = sine(2, n); // below threshold, delayed identity
        let mut proc = prepared(Contract::default().spec).unwrap();
        let l = proc.latency();

        let body = Contract::default().run_reusing(&mut proc, &input, &[], 64);

        let mut tail: Buffers = vec![vec![0.0f32; l], vec![0.0f32; l]];
        let produced = {
            let mut planes: Vec<&mut [f32]> = tail.iter_mut().map(Vec::as_mut_slice).collect();
            let mut out = AudioBlockMut::new(&mut planes);
            proc.flush(&mut out)
        };
        assert_eq!(produced.frames, l, "flush drains exactly the lookahead");
        assert!(produced.done, "the tail is finite and fully drained");

        for ch in 0..input.len() {
            for k in 0..(n - l) {
                assert!(
                    (body[ch][l + k] - input[ch][k]).abs() < 1e-6,
                    "ch{ch}: body must be the delayed input"
                );
            }
            for k in 0..l {
                assert!(
                    (tail[ch][k] - input[ch][n - l + k]).abs() < 1e-6,
                    "ch{ch}: flush must yield the final {l} samples"
                );
            }
        }
    }
}

mod contract {
    use super::*;

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
    fn flush_equals_processing_zeros() {
        // Draining the tail must be byte-identical to feeding explicit
        // silence: flush pops the same delay line, runs the same true-peak
        // detectors over zero input, and uses the same gain target.
        let input = loud(600, 1.8); // above the ceiling, so the drain limits
        let c = Contract::default();

        let mut flushed = prepared(c.spec).unwrap();
        let l = flushed.latency();
        let _ = c.run_reusing(&mut flushed, &input, &[], 64);
        let (tail, _) = flush_stage(&mut flushed, l);

        let mut zero_fed = prepared(c.spec).unwrap();
        let _ = c.run_reusing(&mut zero_fed, &input, &[], 64);
        let zeros: Buffers = vec![vec![0.0f32; l]; 2];
        let out = c.run_reusing(&mut zero_fed, &zeros, &[], 64);

        assert!(
            bits_eq(&tail, &out),
            "the drained tail must match zero-fed processing bit for bit"
        );
    }

    #[test]
    fn new_input_starts_a_fresh_drain() {
        // A drain delivers the remaining declared tail across calls, and
        // processing new input starts a new drain of the full tail.
        let input = sine(2, 600);
        let mut proc = prepared(Contract::default().spec).unwrap();
        let l = proc.latency();
        let part = 30usize; // under the 72-frame tail
        let _ = Contract::default().run_reusing(&mut proc, &input, &[], 64);
        let (_, first) = flush_stage(&mut proc, part);
        assert_eq!(first.frames, part, "a partial request drains partially");
        assert!(!first.done, "tail frames remain after a partial drain");
        let (_, rest) = flush_stage(&mut proc, l);
        assert_eq!(rest.frames, l - part, "the drain resumes where it left off");
        assert!(rest.done, "the declared tail is now fully delivered");
        let (_, empty) = flush_stage(&mut proc, l);
        assert_eq!(empty.frames, 0, "an exhausted drain writes nothing");

        let _ = Contract::default().run_reusing(&mut proc, &input, &[], 64);
        let (_, second) = flush_stage(&mut proc, l);
        assert_eq!(
            second.frames, l,
            "a new process block must start a fresh full-tail drain"
        );
    }
}

#[test]
fn flush_in_small_chunks_drains_exactly_the_lookahead() {
    // Draining the tail across undersized output buffers yields `look` frames
    // total.
    let n = 600;
    let input = sine(2, n); // below threshold, delayed identity
    let mut proc = prepared(Contract::default().spec).unwrap();
    let l = proc.latency();

    let _ = Contract::default().run_reusing(&mut proc, &input, &[], 64);

    // Drain in 30-frame chunks (l = 72, so this needs 30 + 30 + 12 = 3 calls) and
    // reconstruct the tail to compare against the final `l` input samples.
    let chunk = 30;
    let mut drained: Buffers = vec![Vec::new(), Vec::new()];
    let mut total = 0;
    loop {
        let mut stage: Buffers = vec![vec![0.0f32; chunk], vec![0.0f32; chunk]];
        let produced = {
            let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
            let mut out = AudioBlockMut::new(&mut planes);
            proc.flush(&mut out)
        };
        for (acc, st) in drained.iter_mut().zip(&stage) {
            acc.extend_from_slice(&st[..produced.frames]);
        }
        total += produced.frames;
        if produced.done {
            break;
        }
        assert!(
            total <= l,
            "flush over-produced: {total} frames exceeds the {l}-frame tail"
        );
    }
    assert_eq!(total, l, "the whole tail is exactly the lookahead, no more");

    for (ch, plane) in input.iter().enumerate() {
        for k in 0..l {
            assert!(
                (drained[ch][k] - plane[n - l + k]).abs() < 1e-6,
                "ch{ch}: chunked flush must yield the final {l} samples"
            );
        }
    }
}

#[test]
fn block_size_invariance_is_bit_exact() {
    // Active limiting with threshold events stays split-invariant.
    let input = loud(1000, 1.6);
    let events = [ev(0, THRESHOLD, -3.0), ev(500, THRESHOLD, -1.0)];
    Contract::default().assert_block_size_invariant(limiter, &input, &events);
}

#[test]
fn reset_equivalence_no_state_leak() {
    // Reset clears the delay line, envelope, and tail counter.
    let input = loud(800, 1.6);
    let events = [ev(0, THRESHOLD, -2.0), ev(300, THRESHOLD, -6.0)];
    Contract::default().assert_reset_equivalence(limiter, &input, &events);
}

#[test]
fn memory_footprint_grows_per_channel() {
    // Per-channel delay lines have a constant marginal cost per channel.
    let f1 = prepared(spec_ch(1)).unwrap().memory_footprint();
    let f2 = prepared(spec_ch(2)).unwrap().memory_footprint();
    let f3 = prepared(spec_ch(3)).unwrap().memory_footprint();
    assert!(f2 > f1, "footprint grows with channels");
    assert_eq!(f3 - f2, f2 - f1, "per-channel cost is constant");
}

#[test]
fn tail_is_the_lookahead() {
    let p = prepared(Contract::default().spec).unwrap();
    assert_eq!(p.tail(), Tail::Frames(p.latency()));
}

mod validation {
    use super::*;

    #[test]
    fn zero_sample_rate_is_rejected() {
        let bad = ProcessSpec {
            sample_rate: 0,
            ..Contract::default().spec
        };
        assert!(matches!(prepared(bad), Err(DspError::UnsupportedSpec(_))));
    }

    #[test]
    fn negative_true_peak_margin_is_rejected() {
        let mut p = limiter_with(LimiterSettings::new().true_peak_margin_db(-0.1));
        assert!(matches!(
            Processor::<f32>::prepare(&mut p, Contract::default().spec),
            Err(DspError::InvalidParam(_))
        ));
    }

    #[test]
    fn lookahead_must_retain_the_detector_tail() {
        let spec = Contract::default().spec;
        let mut short = limiter_with(LimiterSettings::new().lookahead_ms(0.2));
        assert!(matches!(
            Processor::<f32>::prepare(&mut short, spec),
            Err(DspError::InvalidParam(_))
        ));

        let minimum_ms = 11.0 * 1000.0 / f64::from(spec.sample_rate);
        let mut minimum = limiter_with(LimiterSettings::new().lookahead_ms(minimum_ms));
        Processor::<f32>::prepare(&mut minimum, spec).expect("11-frame lookahead");
        assert_eq!(minimum.latency(), 11);
    }

    #[test]
    fn zero_release_is_accepted_and_negative_release_is_rejected() {
        let spec = Contract::default().spec;
        let mut zero = limiter_with(LimiterSettings::new().release_ms(0.0));
        Processor::<f32>::prepare(&mut zero, spec).expect("zero release");

        let mut negative = limiter_with(LimiterSettings::new().release_ms(-0.1));
        assert!(matches!(
            Processor::<f32>::prepare(&mut negative, spec),
            Err(DspError::InvalidParam(_))
        ));
    }
}

mod snapshots {
    use super::*;

    #[test]
    fn output_is_byte_reproducible() {
        // Threshold conversion uses vendored libm and the hot loop is FMA-free.
        let input = loud(1000, 1.8);
        let events = [ev(0, THRESHOLD, -2.0)];
        let c = Contract::default();
        let make = || KernelProcessor::new(Limiter::new());
        let a = c.run(make, &input, &events, 128);
        let b = c.run(make, &input, &events, 128);
        assert!(bits_eq(&a, &b), "limiter output must be byte-reproducible");
    }
}
