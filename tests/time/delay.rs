// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `Delay`.
//!
//! Covers echo timing and feedback decay, dry passthrough, block-size invariance,
//! reset behavior, memory footprint, tail declaration, flush draining, and
//! reference rendering.

use bisque::processor::KernelProcessor;
use bisque::processor::{AudioBlockMut, DspError, Kernel, ProcessSpec, Processor, Produced};
use bisque::testing::{bits_eq, ev, sine, Buffers, Contract};
use bisque::time::{Delay, DelaySettings};

const DELAY: bisque::parameter::ParamId = Delay::DELAY_MS;
const FEEDBACK: bisque::parameter::ParamId = Delay::FEEDBACK;
const MIX: bisque::parameter::ParamId = Delay::MIX;

/// Zigzag event schedule sweeping `DELAY_MS` across `1.0..=max_ms` and back,
/// one event every `step` frames.
fn sweep_events(frames: usize, step: usize, max_ms: f64) -> Vec<bisque::parameter::ParamEvent> {
    (0..frames as u32)
        .step_by(step)
        .enumerate()
        .map(|(k, at)| {
            let t = (k % 16) as f64 / 15.0;
            let up = (k / 16) % 2 == 0;
            let x = if up { t } else { 1.0 - t };
            ev(at, DELAY, 1.0 + x * (max_ms - 1.0))
        })
        .collect()
}

fn delay_settings(delay_ms: f64, feedback: f64, mix: f64, max_delay_ms: f64) -> DelaySettings {
    DelaySettings::new()
        .delay_ms(delay_ms)
        .feedback(feedback)
        .mix(mix)
        .max_delay_ms(max_delay_ms)
}

/// Prepare a delay behind `impl Processor<f32>`.
fn prepared(spec: ProcessSpec) -> impl Processor<f32> {
    let mut p = KernelProcessor::new(Delay::with_settings(delay_settings(100.0, 0.3, 0.5, 200.0)));
    Processor::<f32>::prepare(&mut p, spec).expect("prepare");
    p
}

/// The default contract spec with the channel count overridden.
fn spec_ch(channels: usize) -> ProcessSpec {
    ProcessSpec {
        channels,
        ..Contract::default().spec
    }
}

/// Flush a prepared processor into a `frames`-frame stereo stage.
///
/// Returns the drained buffers and the `Produced` status.
fn flush_stage(proc: &mut impl Processor<f32>, frames: usize) -> (Buffers, Produced) {
    let mut stage: Buffers = vec![vec![0.0f32; frames]; 2];
    let produced = {
        let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
        let mut out = AudioBlockMut::new(&mut planes);
        proc.flush(&mut out)
    };
    (stage, produced)
}

mod audio {
    use super::*;

    #[test]
    fn delayed_impulse_echoes_decay_by_feedback() {
        // Fully wet output repeats an impulse every delay period, scaled by
        // feedback^(k-1).
        let (delay_ms, fb, fs) = (10.0f64, 0.5f64, 48_000.0f64);
        let d = (delay_ms * 1e-3 * fs).round() as usize; // 480 samples
        let n = d * 5;
        let mut input: Buffers = vec![vec![0.0f32; n]; 2];
        input[0][0] = 1.0;
        input[1][0] = 1.0;
        let make = || {
            KernelProcessor::new(Delay::with_settings(delay_settings(
                delay_ms, fb, 1.0, 100.0,
            )))
        };
        let out = Contract::default().run(make, &input, &[], 64);

        for k in 1..=4 {
            let expected = fb.powi(k as i32 - 1) as f32; // 1, 0.5, 0.25, 0.125
            assert!(
                (out[0][k * d] - expected).abs() < 1e-5,
                "echo {k} at frame {} should be {expected}, got {}",
                k * d,
                out[0][k * d]
            );
        }
        assert!(out[0][d / 2].abs() < 1e-5, "between taps should be silent");
    }

    #[test]
    fn dry_mix_is_transparent() {
        // mix = 0: the delay runs internally but only the dry input reaches the
        // output, bit for bit.
        let input = sine(2, 1000);
        let make =
            || KernelProcessor::new(Delay::with_settings(delay_settings(50.0, 0.5, 0.0, 100.0)));
        let out = Contract::default().run(make, &input, &[], 64);
        assert!(bits_eq(&out, &input), "mix = 0 must be a dry passthrough");
    }

    #[test]
    fn non_finite_input_does_not_poison_feedback() {
        let mut bad: Buffers = vec![vec![0.0f32; 512]; 2];
        bad[0][0] = f32::NAN;
        bad[1][32] = f32::INFINITY;
        bad[0][64] = 1.0;
        bad[1][64] = 1.0;
        let mut sanitized = bad.clone();
        sanitized[0][0] = 0.0;
        sanitized[1][32] = 0.0;

        let make =
            || KernelProcessor::new(Delay::with_settings(delay_settings(2.0, 0.7, 1.0, 10.0)));
        let got = Contract::default().run(make, &bad, &[], 32);
        let want = Contract::default().run(make, &sanitized, &[], 32);

        assert!(
            got.iter().flatten().all(|sample| sample.is_finite()),
            "non-finite input must not poison delay output"
        );
        assert!(bits_eq(&got, &want), "non-finite samples are silence");
    }

    #[test]
    fn full_range_delay_time_sweep_is_click_safe() {
        // Adversarial sweep: DELAY_MS zigzags over its whole declared range,
        // fully wet with no feedback, driven by the sine() multi-tone. The
        // analytic per-frame jump bound is the source's own worst slope
        // (amplitude * angular frequency) plus the crossfade's worst step:
        // the blend weight moves 1/32 per frame between taps at most
        // 2 * amplitude apart, so `amp * w + 2 * amp / 32`.
        let frames = 6000;
        let max_ms = 50.0;
        let input = sine(2, frames);
        let events = sweep_events(frames, 64, max_ms);
        let make =
            || KernelProcessor::new(Delay::with_settings(delay_settings(25.0, 0.0, 1.0, max_ms)));
        let out = Contract::default().run(make, &input, &events, 128);
        for (ch, plane) in out.iter().enumerate() {
            let amp = 0.7 - 0.08 * ch as f64;
            let w = 0.011 + 0.006 * ch as f64;
            let bound = (amp * w + 2.0 * amp / 32.0) as f32 + 1e-6;
            for i in 1..plane.len() {
                let jump = (plane[i] - plane[i - 1]).abs();
                assert!(
                    jump <= bound,
                    "ch{ch}: frame {i} jumps {jump}, above the crossfade bound {bound}"
                );
            }
        }
    }

    #[test]
    fn feedback_stays_stable_while_delay_time_sweeps() {
        // Maximum feedback under a continuous full-range delay-time sweep.
        // The crossfade blend is convex, so the loop gain never exceeds the
        // feedback value and the wet output stays inside the geometric bound
        // `amp / (1 - 0.95)`.
        let frames = 48_000;
        let max_ms = 50.0;
        let input = sine(2, frames);
        let events = sweep_events(frames, 96, max_ms);
        let make = || {
            KernelProcessor::new(Delay::with_settings(delay_settings(
                10.0, 0.95, 1.0, max_ms,
            )))
        };
        let out = Contract::default().run(make, &input, &events, 512);
        for (ch, plane) in out.iter().enumerate() {
            let amp = 0.7 - 0.08 * ch as f64;
            let bound = (amp / (1.0 - 0.95)) as f32 + 1e-3;
            for (i, &s) in plane.iter().enumerate() {
                assert!(s.is_finite(), "ch{ch}: frame {i} must be finite");
                assert!(
                    s.abs() <= bound,
                    "ch{ch}: frame {i} = {s} exceeds the stability bound {bound}"
                );
            }
        }
    }

    #[test]
    fn flush_continues_an_active_crossfade_click_safe() {
        // End input while a delay-time fade is in flight: the drain keeps
        // fading. The input is tapered to silence over its last 256 frames so
        // the recorded material ends smoothly, isolating the crossfade as the
        // only jump source across the body/flush seam. The bound adds the
        // taper's own slope (amp / 256) to the sweep bound.
        let frames = 2048usize;
        let taper = 256usize;
        let mut input = sine(2, frames);
        for plane in &mut input {
            for i in 0..taper {
                let g = (taper - 1 - i) as f32 / taper as f32;
                plane[frames - taper + i] *= g;
            }
        }
        // The late event leaves the fade mid-flight at end of input.
        let events = [ev(0, DELAY, 5.0), ev((frames - 40) as u32, DELAY, 45.0)];
        let mut proc =
            KernelProcessor::new(Delay::with_settings(delay_settings(5.0, 0.0, 1.0, 50.0)));
        Processor::<f32>::prepare(&mut proc, Contract::default().spec).expect("prepare");
        let body = Contract::default().run_reusing(&mut proc, &input, &events, 128);
        let (tail, produced) = flush_stage(&mut proc, 512);
        assert!(produced.frames > 0, "the drain continues past end of input");

        for ch in 0..2usize {
            let amp = 0.7 - 0.08 * ch as f64;
            let w = 0.011 + 0.006 * ch as f64;
            let bound = (amp * w + amp / taper as f64 + 2.0 * amp / 32.0) as f32 + 1e-6;
            // The last body frames plus the drained tail form one seamless
            // wet stream.
            let mut stream: Vec<f32> = body[ch][frames - 64..].to_vec();
            stream.extend_from_slice(&tail[ch][..produced.frames]);
            for i in 1..stream.len() {
                let jump = (stream[i] - stream[i - 1]).abs();
                assert!(
                    jump <= bound,
                    "ch{ch}: seam frame {i} jumps {jump}, above the bound {bound}"
                );
            }
        }
    }

    #[test]
    fn settled_delay_output_is_identical_to_the_single_tap_read() {
        // At settled values the crossfade is inactive and the output is the
        // exact single-tap delay, bit for bit against the independent
        // reference model (which has no crossfade at all).
        let (delay_ms, fb, mix, max_delay_ms, fs) = (10.0, 0.5, 0.7, 50.0, 48_000.0);
        let input = sine(2, 1200);
        let make = || {
            KernelProcessor::new(Delay::with_settings(delay_settings(
                delay_ms,
                fb,
                mix,
                max_delay_ms,
            )))
        };
        let out = Contract::default().run(make, &input, &[], 64);
        let want = super::delay_reference(&input, delay_ms, fb, mix, max_delay_ms, fs);
        assert!(
            bits_eq(&out, &want),
            "a never-automated delay must be bit-identical to the single-tap reference"
        );
    }

    #[test]
    fn flush_continues_the_echo_train() {
        // Fully wet, an impulse echoes at k*d with amplitude fb^(k-1). Flushing
        // after 2.5 delay periods must continue that closed-form train with
        // silent input.
        let (delay_ms, fb, fs) = (10.0f64, 0.5f64, 48_000.0f64);
        let d = (delay_ms * 1e-3 * fs).round() as usize; // 480 samples
        let n = 2 * d + d / 2; // the body ends between echoes 2 and 3
        let mut input: Buffers = vec![vec![0.0f32; n]; 2];
        input[0][0] = 1.0;
        input[1][0] = 1.0;
        let mut proc = KernelProcessor::new(Delay::with_settings(delay_settings(
            delay_ms, fb, 1.0, 100.0,
        )));
        Processor::<f32>::prepare(&mut proc, Contract::default().spec).expect("prepare");
        let body = Contract::default().run_reusing(&mut proc, &input, &[], 64);
        assert!((body[0][d] - 1.0).abs() < 1e-5, "echo 1 in the body");
        assert!(
            (body[0][2 * d] - fb as f32).abs() < 1e-5,
            "echo 2 in the body"
        );

        // Drain enough frames for echoes 3, 4, and 5.
        let (tail, produced) = flush_stage(&mut proc, 3 * d);
        assert_eq!(produced.frames, 3 * d, "the tail is still live here");
        assert!(
            !produced.done,
            "louder-than-floor echoes remain in the ring"
        );
        for (ch, plane) in tail.iter().enumerate() {
            for k in 3..=5usize {
                let at = k * d - n; // echo position on the flush timeline
                let expected = fb.powi(k as i32 - 1) as f32;
                assert!(
                    (plane[at] - expected).abs() < 1e-5,
                    "ch{ch}: flushed echo {k} at tail frame {at} should be {expected}, got {}",
                    plane[at]
                );
            }
            assert!(
                plane[d / 4].abs() < 1e-6,
                "ch{ch}: between echoes the flushed tail is silent"
            );
        }
    }
}

mod contract {
    use super::*;
    use bisque::processor::Tail;

    /// Test-side copy of the tail bound's feedback-pass count from retained
    /// headroom `1e3` to the `1e-6` decay floor at maximum feedback.
    const K_MAX: usize = 405;

    /// Ring length for `max_delay_ms` at 48 kHz (max delay samples plus one).
    fn ring_len(max_delay_ms: f64) -> usize {
        (max_delay_ms * 1e-3 * 48_000.0).round() as usize + 1
    }

    /// A prepared delay processor with a 100 ms max ring and 0.5 feedback.
    fn prepared_impulse_delay() -> impl Processor<f32> {
        let mut proc =
            KernelProcessor::new(Delay::with_settings(delay_settings(10.0, 0.5, 1.0, 100.0)));
        Processor::<f32>::prepare(&mut proc, Contract::default().spec).expect("prepare");
        proc
    }

    /// Drive one stereo impulse through `proc` so the ring holds a live tail.
    fn feed_impulse(proc: &mut impl Processor<f32>) {
        let mut input: Buffers = vec![vec![0.0f32; 600]; 2];
        input[0][0] = 1.0;
        input[1][0] = 1.0;
        let _ = Contract::default().run_reusing(proc, &input, &[], 64);
    }

    #[test]
    fn tail_is_the_documented_worst_case_bound() {
        // (1 + K_MAX) * ring capacity, constant after prepare, from the
        // declared range maxima (delay up to the ring, feedback up to 0.95).
        let max_delay_ms = 200.0;
        let mut proc = KernelProcessor::new(Delay::with_settings(delay_settings(
            100.0,
            0.3,
            0.5,
            max_delay_ms,
        )));
        Processor::<f32>::prepare(&mut proc, Contract::default().spec).expect("prepare");
        assert_eq!(
            Processor::<f32>::tail(&proc),
            Tail::Frames((1 + K_MAX) * ring_len(max_delay_ms)),
            "tail must be the documented conservative bound"
        );
    }

    #[test]
    fn full_drain_ends_done_well_before_the_declared_bound() {
        // 0.5 feedback decays below the 1e-6 floor after ~20 echoes, so a real
        // drain ends orders of magnitude before the worst-case bound.
        let mut proc = prepared_impulse_delay();
        feed_impulse(&mut proc);
        let bound = match proc.tail() {
            Tail::Frames(n) => n,
            other => panic!("expected Tail::Frames, got {other:?}"),
        };
        let mut total = 0usize;
        loop {
            let (_, produced) = flush_stage(&mut proc, 512);
            total += produced.frames;
            if produced.done {
                break;
            }
            assert!(
                total < 32_768,
                "feedback 0.5 must decay within the analytical echo budget"
            );
        }
        assert!(
            total < bound / 10,
            "a real drain ({total} frames) must end far before the bound ({bound})"
        );
    }

    #[test]
    fn a_host_caps_a_drain_by_bounding_requested_frames() {
        // `flush` writes at most `out.frames()` per call, so a host caps a
        // drain by limiting the total frames it requests. The uncollected
        // tail stays live in the ring.
        let mut proc = prepared_impulse_delay();
        feed_impulse(&mut proc);
        let cap = 1000usize;
        let mut total = 0usize;
        while total < cap {
            let want = 300.min(cap - total);
            let (_, produced) = flush_stage(&mut proc, want);
            assert_eq!(
                produced.frames, want,
                "a live tail fills every requested frame"
            );
            assert!(
                !produced.done,
                "the capped drain leaves audible tail in the ring"
            );
            total += produced.frames;
        }
        assert_eq!(total, cap, "the host's cap bounds the whole drain");
    }

    #[test]
    fn new_input_starts_a_new_drain() {
        // Drain the echo train until the decay early-exit reports done, then
        // feed a fresh impulse: the next flush must deliver a live tail again.
        let mut proc = prepared_impulse_delay();
        feed_impulse(&mut proc);
        let mut guard = 0usize;
        loop {
            let (_, produced) = flush_stage(&mut proc, 512);
            if produced.done {
                break;
            }
            guard += 1;
            assert!(guard < 64, "the decayed drain must report done");
        }

        feed_impulse(&mut proc);
        let (stage, produced) = flush_stage(&mut proc, 600);
        assert_eq!(produced.frames, 600, "a new drain delivers frames again");
        assert!(!produced.done, "the fresh impulse re-arms the tail");
        assert!(
            stage.iter().flatten().any(|&s| s.abs() > 1e-3),
            "the new drain carries the fresh echo train"
        );
    }

    #[test]
    fn reset_clears_the_ring_and_the_drain() {
        // After reset the ring is silent: the drain writes zeros and the decay
        // early-exit reports done immediately.
        let mut proc = prepared_impulse_delay();
        feed_impulse(&mut proc);
        let (_, live) = flush_stage(&mut proc, 200);
        assert!(!live.done, "the tail is live before reset");

        proc.reset();
        let (stage, again) = flush_stage(&mut proc, 200);
        assert!(again.done, "the reset ring is silent, so the drain is done");
        assert!(
            stage.iter().flatten().all(|&s| s == 0.0),
            "a reset ring drains silence"
        );
    }
}

#[test]
fn block_size_invariance_is_bit_exact() {
    // Feedback and mix automation stay split-invariant.
    let input = sine(2, 1000);
    let events = [ev(0, FEEDBACK, 0.6), ev(300, MIX, 0.7)];
    Contract::default().assert_block_size_invariant(
        || KernelProcessor::new(Delay::new()),
        &input,
        &events,
    );
}

#[test]
fn reset_equivalence_no_state_leak() {
    let input = sine(2, 800);
    Contract::default().assert_reset_equivalence(
        || KernelProcessor::new(Delay::new()),
        &input,
        &[],
    );
}

#[test]
fn memory_footprint_grows_per_channel() {
    // A ring buffer per channel plus a fixed bank: the marginal cost per channel
    // is constant.
    let f1 = prepared(spec_ch(1)).memory_footprint();
    let f2 = prepared(spec_ch(2)).memory_footprint();
    let f3 = prepared(spec_ch(3)).memory_footprint();
    assert!(f2 > f1, "footprint grows with channels");
    assert_eq!(f3 - f2, f2 - f1, "per-channel cost is constant");
}

mod validation {
    use super::*;

    #[test]
    fn zero_sample_rate_is_rejected() {
        let mut p = KernelProcessor::new(Delay::new());
        let bad = ProcessSpec {
            sample_rate: 0,
            ..Contract::default().spec
        };
        assert!(matches!(
            Processor::<f32>::prepare(&mut p, bad),
            Err(DspError::UnsupportedSpec(_))
        ));
    }

    #[test]
    fn out_of_range_settings_are_rejected_not_clamped() {
        // Structural policy: settings are preserved as constructed, and
        // `prepare` rejects them instead of silently changing them. A
        // `delay_ms` longer than `max_delay_ms` is an out-of-range parameter
        // default.
        let spec = Contract::default().spec;
        let mut long = KernelProcessor::new(Delay::with_settings(delay_settings(
            5.0, 0.4, 1.0, 2.0, // delay_ms > max_delay_ms
        )));
        assert!(matches!(
            Processor::<f32>::prepare(&mut long, spec),
            Err(DspError::InvalidParam(_))
        ));
        // Out-of-range feedback and mix defaults are rejected the same way.
        let mut hot =
            KernelProcessor::new(Delay::with_settings(delay_settings(50.0, 1.5, 0.5, 100.0)));
        assert!(matches!(
            Processor::<f32>::prepare(&mut hot, spec),
            Err(DspError::InvalidParam(_))
        ));
        // A non-finite or sub-1 ms maximum is rejected too.
        for bad_max in [0.5, f64::NAN] {
            let mut p =
                KernelProcessor::new(Delay::with_settings(delay_settings(1.0, 0.3, 0.5, bad_max)));
            assert!(matches!(
                Processor::<f32>::prepare(&mut p, spec),
                Err(DspError::InvalidParam(_))
            ));
        }
    }
}

#[test]
fn memory_footprint_is_exact_byte_count() {
    // The kernel footprint is one f64 ring of max_samples + 1 per channel.
    let (max_delay_ms, fs) = (200.0f64, 48_000.0f64);
    let max_samples = (max_delay_ms * 1e-3 * fs).round().max(1.0) as usize; // 9600
    let f64_bytes = std::mem::size_of::<f64>();
    for ch in 1usize..=4 {
        let mut k = Delay::with_settings(delay_settings(100.0, 0.3, 0.5, max_delay_ms));
        Kernel::<f32>::prepare(&mut k, spec_ch(ch)).expect("prepare");
        let expected = ch * (max_samples + 1) * f64_bytes;
        assert_eq!(
            Kernel::<f32>::memory_footprint(&k),
            expected,
            "{ch}-channel ring must be exactly {expected} bytes"
        );
    }
}

/// Independent feedback-delay reference in `f64`.
///
/// Uses the same operation order as the implementation.
fn delay_reference(
    input: &[Vec<f32>],
    delay_ms: f64,
    feedback: f64,
    mix: f64,
    max_delay_ms: f64,
    fs: f64,
) -> Buffers {
    let max_samples = (max_delay_ms * 1e-3 * fs).round().max(1.0) as usize;
    let ring_len = max_samples + 1;
    let d = ((delay_ms * 1e-3 * fs).round() as usize).clamp(1, ring_len - 1);
    let fb = feedback.clamp(0.0, 0.95);
    let mix = mix.clamp(0.0, 1.0);
    let dry = 1.0 - mix;
    input
        .iter()
        .map(|chan| {
            let mut ring = vec![0.0f64; ring_len];
            let mut p = 0usize;
            chan.iter()
                .map(|&s| {
                    let read = if p >= d { p - d } else { p + ring_len - d };
                    let delayed = ring[read];
                    let x = f64::from(s);
                    ring[p] = x + fb * delayed;
                    p = if p + 1 == ring_len { 0 } else { p + 1 };
                    (x * dry + delayed * mix) as f32
                })
                .collect()
        })
        .collect()
}

#[test]
fn render_matches_independent_reference_when_ring_wraps() {
    // A small ring driven past its length exercises wrapped reads with feedback
    // and partial mix.
    let (delay_ms, fb, mix, max_delay_ms, fs) = (2.0, 0.5, 0.7, 10.0, 48_000.0);
    let input = sine(2, 1500); // 1500 frames >> 481-slot ring => repeated wraps
    let make = || {
        KernelProcessor::new(Delay::with_settings(delay_settings(
            delay_ms,
            fb,
            mix,
            max_delay_ms,
        )))
    };
    let out = Contract::default().run(make, &input, &[], 64);
    let want = delay_reference(&input, delay_ms, fb, mix, max_delay_ms, fs);
    for (oc, wc) in out.iter().zip(&want) {
        for (i, (&o, &w)) in oc.iter().zip(wc).enumerate() {
            assert!(
                (o - w).abs() < 1e-4,
                "wrapped delay must match the reference at {i}: got {o}, want {w}"
            );
        }
    }
}

#[test]
fn render_matches_reference_at_the_full_ring_delay() {
    // The longest valid delay (delay_ms == max_delay_ms) reads the oldest
    // ring slot, `ring_len - 1` frames back.
    let (delay_ms, fb, mix, max_delay_ms, fs) = (2.0, 0.4, 1.0, 2.0, 48_000.0);
    let input = sine(2, 600);
    let make = || {
        KernelProcessor::new(Delay::with_settings(delay_settings(
            delay_ms,
            fb,
            mix,
            max_delay_ms,
        )))
    };
    let out = Contract::default().run(make, &input, &[], 64);
    let want = delay_reference(&input, delay_ms, fb, mix, max_delay_ms, fs);
    for (oc, wc) in out.iter().zip(&want) {
        for (i, (&o, &w)) in oc.iter().zip(wc).enumerate() {
            assert!(
                (o - w).abs() < 1e-4,
                "full-ring delay must match the reference at {i}: got {o}, want {w}"
            );
        }
    }
}

#[test]
fn partial_mix_blends_dry_and_wet() {
    // With mix = 0.5 and delayed = 0, the first output frame is the dry half.
    let (delay_ms, fb, mix, max_delay_ms) = (5.0, 0.3, 0.5, 50.0);
    let mut input: Buffers = vec![vec![0.0f32; 64]; 2];
    input[0][0] = 1.0;
    input[1][0] = 1.0;
    let make = || {
        KernelProcessor::new(Delay::with_settings(delay_settings(
            delay_ms,
            fb,
            mix,
            max_delay_ms,
        )))
    };
    let out = Contract::default().run(make, &input, &[], 64);
    for (ch, plane) in out.iter().enumerate() {
        assert!(
            (plane[0] - 0.5).abs() < 1e-6,
            "ch {ch}: first frame must be the dry half (0.5), got {}",
            plane[0]
        );
    }
}

#[test]
fn reset_clears_a_wrapped_ring() {
    // A small ring driven by a long input wraps several times before reset.
    let input = sine(2, 1500);
    Contract::default().assert_reset_equivalence(
        || KernelProcessor::new(Delay::with_settings(delay_settings(2.0, 0.5, 0.7, 10.0))),
        &input,
        &[ev(0, FEEDBACK, 0.6), ev(700, MIX, 0.8)],
    );
}
