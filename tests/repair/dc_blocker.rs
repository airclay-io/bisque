// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `DcBlocker`.
//!
//! Covers DC removal, audio-band gain, tail declaration, flush draining,
//! block-size invariance, memory footprint, and reset behavior.

use std::f64::consts::TAU;

use bisque::processor::Kernel;
use bisque::processor::KernelProcessor;
use bisque::processor::{AudioBlockMut, DspError, ProcessSpec, Processor, Produced, Tail};
use bisque::repair::{DcBlocker, DcBlockerSettings};
use bisque::testing::{bits_eq, ev, sine, Buffers, Contract};

const CUTOFF: bisque::parameter::ParamId = DcBlocker::CUTOFF_HZ;
const FS: f64 = 48_000.0;

fn rms(s: &[f32]) -> f64 {
    (s.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
}

#[test]
fn settings_builder_stores_the_cutoff() {
    // Pins the builder itself: both audio tests would still pass at the
    // 20 Hz default, so this is what proves a custom cutoff is kept.
    let s = DcBlockerSettings::new().cutoff_hz(150.0);
    assert!(
        (s.cutoff_hz - 150.0).abs() < 1e-12,
        "cutoff_hz builder must store the value, got {}",
        s.cutoff_hz
    );
}

#[test]
fn drives_a_dc_offset_to_zero() {
    // A constant offset has all its energy at DC, where the blocker has zero gain.
    let level = 0.5f32;
    let len = 8_000;
    let input = vec![vec![level; len]; 2];
    let out = Contract::default().run(
        || {
            KernelProcessor::new(DcBlocker::with_settings(
                DcBlockerSettings::new().cutoff_hz(100.0),
            ))
        },
        &input,
        &[],
        64,
    );
    for &v in &out[0][len - 500..] {
        assert!(v.abs() < 1e-3, "DC must be removed, settled tail {v}");
    }
}

#[test]
fn passes_the_audio_band_at_about_unity() {
    // A 2 kHz tone is far above the 100 Hz corner, so it passes essentially
    // unchanged (a DC-blocker is ~unity across the band).
    let (freq, amp, len) = (2_000.0, 0.5f32, 20_000);
    let ch: Vec<f32> = (0..len)
        .map(|i| (TAU * freq * i as f64 / FS).sin() as f32 * amp)
        .collect();
    let input = vec![ch.clone(), ch];
    let out = Contract::default().run(
        || {
            KernelProcessor::new(DcBlocker::with_settings(
                DcBlockerSettings::new().cutoff_hz(100.0),
            ))
        },
        &input,
        &[],
        64,
    );
    // Compare the settled tail (past the brief transient).
    let in_rms = rms(&input[0][len - 8_000..]);
    let out_rms = rms(&out[0][len - 8_000..]);
    assert!(
        (out_rms / in_rms - 1.0).abs() < 0.02,
        "audio band must pass ~unity: out {out_rms} vs in {in_rms}"
    );
}

#[test]
fn non_finite_input_matches_silence() {
    let input: Buffers = vec![
        vec![f32::NAN, 0.25, f32::INFINITY, -0.25],
        vec![f32::NEG_INFINITY, 0.5, f32::NAN, 1.0],
    ];
    let sanitized: Buffers = vec![vec![0.0, 0.25, 0.0, -0.25], vec![0.0, 0.5, 0.0, 1.0]];
    let contract = Contract::default();
    let actual = contract.run(|| KernelProcessor::new(DcBlocker::new()), &input, &[], 4);
    let expected = contract.run(
        || KernelProcessor::new(DcBlocker::new()),
        &sanitized,
        &[],
        4,
    );
    assert!(bits_eq(&actual, &expected));
}

#[test]
fn full_range_cutoff_automation_has_no_large_discontinuity() {
    let dwell = 1_024u32;
    let frames = 2_048usize;
    let w = TAU * 997.0 / FS;
    let channel: Vec<f32> = (0..frames)
        .map(|i| (bisque::dsp::math::sin(i as f64 * w) * 0.5) as f32)
        .collect();
    let input = vec![channel.clone(), channel];
    let events = [ev(dwell, CUTOFF, 1_000.0)];
    let output = Contract::default().run(
        || {
            KernelProcessor::new(DcBlocker::with_settings(
                DcBlockerSettings::new().cutoff_hz(1.0),
            ))
        },
        &input,
        &events,
        64,
    );
    let max_delta = output[0]
        .windows(2)
        .skip(dwell as usize - 1)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta < 0.2,
        "smoothed cutoff sweep produced a sample jump of {max_delta}"
    );
}

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
    // Draining the settle-out must be byte-identical to feeding explicit
    // silence: flush continues the recurrence with the pole radius last seen
    // by render.
    let input: Buffers = vec![vec![0.5f32; 1_000]; 2]; // a DC step charges the state
    let c = Contract::default();
    let make = || KernelProcessor::new(DcBlocker::new());

    let mut flushed = make();
    Processor::<f32>::prepare(&mut flushed, c.spec).expect("prepare");
    let _ = c.run_reusing(&mut flushed, &input, &[], 64);
    let (tail, _) = flush_stage(&mut flushed, 256);

    let mut zero_fed = make();
    Processor::<f32>::prepare(&mut zero_fed, c.spec).expect("prepare");
    let _ = c.run_reusing(&mut zero_fed, &input, &[], 64);
    let zeros: Buffers = vec![vec![0.0f32; 256]; 2];
    let out = c.run_reusing(&mut zero_fed, &zeros, &[], 64);

    assert!(
        bits_eq(&tail, &out),
        "the drained settle-out must match zero-fed processing bit for bit"
    );
}

#[test]
fn tail_bound_matches_the_declared_range_worst_case() {
    // Independently recompute the conservative bound at 48 kHz: the slowest
    // declared pole is at the 1 Hz range minimum. The first silent recurrence
    // can combine both retained state values, so it budgets 2e3 down to 1e-6
    // and includes the first emitted frame.
    let mut proc = KernelProcessor::new(DcBlocker::new());
    Processor::<f32>::prepare(&mut proc, Contract::default().spec).expect("prepare");
    let Tail::Frames(bound) = Processor::<f32>::tail(&proc) else {
        panic!("a prepared DC blocker declares a finite tail");
    };
    let expected = (2e9_f64.ln() * 48_000.0 / (TAU * 1.0)).ceil() + 1.0;
    assert!(
        ((bound as f64) - expected).abs() <= 1.0,
        "declared tail bound {bound} must match the range worst case {expected}"
    );
}

#[test]
fn declared_tail_reaches_the_floor_from_opposing_retained_state() {
    let settings = DcBlockerSettings::new().cutoff_hz(1.0);
    let mut proc = KernelProcessor::new(DcBlocker::with_settings(settings));
    let contract = Contract::default();
    Processor::<f32>::prepare(&mut proc, contract.spec).expect("prepare");

    let r = bisque::dsp::math::exp(-TAU / FS);
    let first = (2_000.0 / (1.0 - r)) as f32;
    let input = vec![vec![first, 1_000.0]; 2];
    let _ = contract.run_reusing(&mut proc, &input, &[], 2);

    let Tail::Frames(bound) = proc.tail() else {
        panic!("a DC blocker declares a finite tail");
    };
    let mut total = 0usize;
    let mut last = 0.0f32;
    loop {
        let (stage, produced) = flush_stage(&mut proc, 4_096);
        if produced.frames > 0 {
            last = stage[0][produced.frames - 1];
        }
        total += produced.frames;
        if produced.done {
            break;
        }
    }
    assert!(total <= bound);
    assert!(
        last.abs() < 1e-6,
        "last declared tail sample {last} must be below the decay floor"
    );
}

#[test]
fn unbudgeted_drain_ends_done_within_the_declared_tail() {
    // Cutting a DC step leaves the recurrence settling back toward zero. The
    // drain ends at the decay floor, well inside the conservative bound.
    let input: Buffers = vec![vec![0.5f32; 1_000]; 2];
    let c = Contract::default();
    let mut proc = KernelProcessor::new(DcBlocker::new());
    Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
    let Tail::Frames(bound) = Processor::<f32>::tail(&proc) else {
        panic!("a prepared DC blocker declares a finite tail");
    };
    assert!(bound > 0, "the declared tail bound is nonzero");
    let _ = c.run_reusing(&mut proc, &input, &[], 64);

    let mut total = 0usize;
    loop {
        let (_, produced) = flush_stage(&mut proc, 4_096);
        total += produced.frames;
        if produced.done {
            break;
        }
        assert!(total <= bound, "a finite tail must not over-produce");
    }
    assert!(total > 0, "a charged blocker has a nonempty drain");
    assert!(
        total < bound,
        "the decay floor ends the drain before the conservative bound \
         ({total} of {bound})"
    );
}

#[test]
fn pathological_input_drains_by_the_documented_decay_ratio() {
    // The declared finite tail is a decay guarantee, not a silence guarantee.
    // the drain covers about 186 dB of decay relative to the largest retained
    // state value. A +120 dBFS DC charge therefore
    // ends near 1e-3 when the bound is reached, above the absolute 1e-6 floor
    // reserved for in-headroom signals. `done` still reports the declared tail
    // as delivered.
    let input: Buffers = vec![vec![1.0e6f32; 2_000]; 2];
    let c = Contract::default();
    let mut proc = KernelProcessor::new(DcBlocker::with_settings(
        DcBlockerSettings::new().cutoff_hz(1.0), // the slowest declared pole
    ));
    Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
    let _ = c.run_reusing(&mut proc, &input, &[], 64);

    let mut first = 0.0f32;
    let mut last = 0.0f32;
    let mut total = 0usize;
    loop {
        let (stage, produced) = flush_stage(&mut proc, 4_096);
        if total == 0 && produced.frames > 0 {
            first = stage[0][0];
        }
        if produced.frames > 0 {
            last = stage[0][produced.frames - 1];
        }
        total += produced.frames;
        if produced.done {
            break;
        }
    }
    assert!(
        first.abs() > 1e5,
        "the cut DC step enters the tail near full charge, got {first}"
    );
    assert!(
        last.abs() <= first.abs() * 6e-10,
        "the drain covers about 186 dB of relative decay: {first} -> {last}"
    );
    assert!(
        last.abs() > 0.0,
        "over-headroom input is honestly still above the absolute floor at done"
    );
}

#[test]
fn block_size_invariance_is_bit_exact() {
    // The recurrence and cutoff automation stay split-invariant.
    let input = sine(2, 1000);
    let events = [ev(0, CUTOFF, 30.0), ev(384, CUTOFF, 200.0)];
    Contract::default().assert_block_size_invariant(
        || KernelProcessor::new(DcBlocker::new()),
        &input,
        &events,
    );
}

#[test]
fn reports_its_memory_footprint() {
    // Per-channel state is two f64 values: last input and last output.
    let c = Contract::default(); // 48 kHz stereo
    let mut bq = DcBlocker::new();
    Kernel::<f32>::prepare(&mut bq, c.spec).expect("prepare");
    let expected = c.spec.channels * 2 * std::mem::size_of::<f64>();
    assert_eq!(Kernel::<f32>::memory_footprint(&bq), expected);
}

#[test]
fn reset_equivalence_no_state_leak() {
    let input = sine(2, 800);
    let events = [ev(0, CUTOFF, 50.0), ev(300, CUTOFF, 15.0)];
    Contract::default().assert_reset_equivalence(
        || KernelProcessor::new(DcBlocker::new()),
        &input,
        &events,
    );
}

#[test]
fn invalid_settings_and_specs_are_rejected() {
    let spec = Contract::default().spec;
    for cutoff in [f64::NAN, f64::INFINITY, 0.0, 1_001.0] {
        let mut proc = KernelProcessor::new(DcBlocker::with_settings(
            DcBlockerSettings::new().cutoff_hz(cutoff),
        ));
        assert!(matches!(
            Processor::<f32>::prepare(&mut proc, spec),
            Err(DspError::InvalidParam(_))
        ));
    }

    let mut proc = DcBlocker::new();
    let zero_rate = ProcessSpec {
        sample_rate: 0,
        ..spec
    };
    assert!(matches!(
        Kernel::<f32>::prepare(&mut proc, zero_rate),
        Err(DspError::UnsupportedSpec(_))
    ));
}

#[test]
fn state_memory_is_preflighted_before_allocation() {
    let state_bytes = Contract::default().spec.channels * 2 * std::mem::size_of::<f64>();
    let exact = ProcessSpec {
        max_memory: Some(state_bytes),
        ..Contract::default().spec
    };
    let mut fits = DcBlocker::new();
    Kernel::<f32>::prepare(&mut fits, exact).expect("exact state budget");

    let mut over = DcBlocker::new();
    let tight = ProcessSpec {
        max_memory: Some(state_bytes - 1),
        ..exact
    };
    assert!(matches!(
        Kernel::<f32>::prepare(&mut over, tight),
        Err(DspError::OverBudget { .. })
    ));
}
