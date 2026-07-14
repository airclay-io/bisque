// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for `Biquad`.
//!
//! Covers streaming impulse response, closed-form transfer function readouts,
//! DC, Nyquist and cutoff gains, pole stability, linearity, metadata, tail
//! declaration, flush draining, and snapshots.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use bisque::filters::{Biquad, BiquadCoeffs, BiquadKind, BiquadSettings};
use bisque::processor::KernelProcessor;
use bisque::processor::{AudioBlockMut, DspError, Kernel, ProcessSpec, Processor, Produced, Tail};
use bisque::testing::{bits_eq, ev, sine, Buffers, Contract};

const CUTOFF: bisque::parameter::ParamId = Biquad::CUTOFF_HZ;
const Q: bisque::parameter::ParamId = Biquad::Q;
const GAIN: bisque::parameter::ParamId = Biquad::GAIN_DB;

/// A mono `Biquad` prepared at `fs` so its `coeffs` and `omega` readouts are
/// valid.
fn prepared_biquad(kind: BiquadKind, fs: u32) -> Biquad {
    let mut bq = Biquad::with_settings(BiquadSettings::new().kind(kind));
    let spec = ProcessSpec {
        sample_rate: fs,
        channels: 1,
        max_block: 8192,
        max_memory: None,
    };
    Kernel::<f32>::prepare(&mut bq, spec).expect("prepare");
    bq
}

fn coeffs(biquad: &Biquad, f0: f64, q: f64, gain_db: f64) -> BiquadCoeffs {
    biquad
        .try_coeffs(f0, q, gain_db)
        .expect("valid coefficient readout")
}

/// A bare `Biquad` kernel prepared for `channels` at the contract sample rate.
fn prepared_kernel(kind: BiquadKind, channels: usize) -> Biquad {
    let mut bq = Biquad::with_settings(BiquadSettings::new().kind(kind));
    let spec = ProcessSpec {
        channels,
        ..Contract::default().spec
    };
    Kernel::<f32>::prepare(&mut bq, spec).expect("prepare");
    bq
}

/// Prepare a biquad behind `impl Processor<f32>`, or return the `prepare` error.
fn prepared(kind: BiquadKind, spec: ProcessSpec) -> Result<impl Processor<f32>, DspError> {
    let mut p = KernelProcessor::new(Biquad::with_settings(BiquadSettings::new().kind(kind)));
    Processor::<f32>::prepare(&mut p, spec).map(|()| p)
}

/// The default contract spec with the channel count overridden.
fn spec_ch(channels: usize) -> ProcessSpec {
    ProcessSpec {
        channels,
        ..Contract::default().spec
    }
}

#[test]
fn settings_cover_shape_and_every_startup_parameter() {
    let settings = BiquadSettings::peaking()
        .cutoff_hz(2_000.0)
        .q(2.0)
        .gain_db(6.0);
    let filter = Biquad::with_settings(settings);
    let info = Kernel::<f32>::param_info(&filter);
    assert_eq!(info[0].default, 2_000.0);
    assert_eq!(info[1].default, 2.0);
    assert_eq!(info[2].default, 6.0);
}

#[test]
fn new_matches_default_settings_and_invalid_startup_values_fail_prepare() {
    let direct = Biquad::new();
    let configured = Biquad::with_settings(BiquadSettings::default());
    assert_eq!(
        Kernel::<f32>::param_info(&direct)[0].default,
        Kernel::<f32>::param_info(&configured)[0].default
    );

    let mut invalid = KernelProcessor::new(Biquad::with_settings(
        BiquadSettings::lowpass().cutoff_hz(f64::NAN),
    ));
    assert!(matches!(
        invalid.prepare(Contract::default().spec),
        Err(DspError::InvalidParam(_))
    ));

    for settings in [
        BiquadSettings::lowpass().q(f64::NAN),
        BiquadSettings::lowpass().q(0.0),
        BiquadSettings::peaking().gain_db(f64::INFINITY),
        BiquadSettings::peaking().gain_db(25.0),
    ] {
        let mut invalid = KernelProcessor::new(Biquad::with_settings(settings));
        assert!(matches!(
            invalid.prepare(Contract::default().spec),
            Err(DspError::InvalidParam(_))
        ));
    }
}

#[test]
fn checked_coefficient_construction_rejects_invalid_inputs() {
    let valid = BiquadCoeffs::try_rbj(BiquadKind::Lowpass, 48_000.0, 1_000.0, FRAC_1_SQRT_2, 0.0);
    assert!(valid.is_ok());

    for (fs, f0, q, gain) in [
        (f64::NAN, 1_000.0, 0.707, 0.0),
        (48_000.0, f64::NAN, 0.707, 0.0),
        (48_000.0, 1_000.0, f64::NAN, 0.0),
        (48_000.0, 1_000.0, 0.707, f64::NAN),
        (0.0, 1_000.0, 0.707, 0.0),
        (48_000.0, 0.0, 0.707, 0.0),
        (48_000.0, 24_000.0, 0.707, 0.0),
        // This is above Nyquist but wraps to a stable digital pole pair if the
        // explicit frequency-domain check is weakened.
        (48_000.0, 48_000.25, 0.707, 0.0),
        (48_000.0, 1_000.0, 0.0, 0.0),
    ] {
        assert_eq!(
            BiquadCoeffs::try_rbj(BiquadKind::Lowpass, fs, f0, q, gain),
            Err(DspError::InvalidParam(
                "biquad coefficient inputs must be finite and inside the filter domain"
            ))
        );
    }

    assert_eq!(
        BiquadCoeffs::try_rbj(BiquadKind::Peaking, 48_000.0, 4_800.0, 1.0e-161, 6_000.0),
        Err(DspError::InvalidParam(
            "biquad inputs must produce finite stable coefficients"
        )),
        "finite inputs that overflow the coefficient calculation are rejected"
    );
}

#[test]
fn prepared_coefficient_readout_matches_runtime_clamping() {
    let unprepared = Biquad::peaking();
    assert!(matches!(
        unprepared.try_coeffs(1_000.0, 1.0, 0.0),
        Err(DspError::UnsupportedSpec(_))
    ));

    let bq = prepared_biquad(BiquadKind::Peaking, 48_000);
    let low = bq.try_coeffs(-100.0, -1.0, 100.0).expect("clamped");
    let low_limit = bq.try_coeffs(10.0, 0.1, 24.0).expect("limits");
    assert_eq!(low, low_limit);

    let high = bq.try_coeffs(100_000.0, 100.0, -100.0).expect("clamped");
    let high_limit = bq.try_coeffs(24_000.0, 16.0, -24.0).expect("limits");
    assert_eq!(high, high_limit);

    for (f0, q, gain_db) in [
        (f64::NAN, 1.0, 0.0),
        (1_000.0, f64::NAN, 0.0),
        (1_000.0, 1.0, f64::NAN),
    ] {
        assert_eq!(
            bq.try_coeffs(f0, q, gain_db),
            Err(DspError::InvalidParam(
                "biquad coefficient readout inputs must be finite"
            ))
        );
    }

    let minimum_rate = prepared_biquad(BiquadKind::Lowpass, 3);
    assert!(
        minimum_rate.try_coeffs(1.0, 1.0, 0.0).is_ok(),
        "the minimum supported prepared sample rate has coefficient readouts"
    );
}

mod audio {
    use super::*;

    #[test]
    fn impulse_response_matches_transfer_function() {
        // Compare the streaming impulse response to the closed-form transfer
        // function from the coefficients.
        let fs = 48_000.0;
        let bq = prepared_biquad(BiquadKind::Lowpass, 48_000);
        let coeffs = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, 0.0); // the kernel's defaults

        let n = 8192;
        let mut impulse = vec![0.0f32; n];
        impulse[0] = 1.0;
        let mut mono = Contract::default();
        mono.spec.channels = 1;
        let captured = mono.run(
            || KernelProcessor::new(Biquad::lowpass()),
            &[impulse],
            &[],
            512,
        );
        let ir = &captured[0];

        for &f in &[50.0, 200.0, 1_000.0, 4_000.0, 12_000.0] {
            let w = 2.0 * PI * f / fs;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (idx, &s) in ir.iter().enumerate() {
                let ang = w * idx as f64;
                re += f64::from(s) * ang.cos();
                im -= f64::from(s) * ang.sin();
            }
            let mag_meas = (re * re + im * im).sqrt();
            let mag_ref = coeffs.magnitude(w);
            assert!(
                (mag_meas - mag_ref).abs() < 1e-3,
                "f={f}: |H| measured {mag_meas} vs closed-form {mag_ref}"
            );
            let phase_meas = im.atan2(re);
            let phase_ref = coeffs.phase(w);
            assert!(
                (phase_meas - phase_ref).abs() < 1e-2,
                "f={f}: phase measured {phase_meas} vs closed-form {phase_ref}"
            );
        }
    }

    #[test]
    fn non_finite_input_is_treated_as_silence() {
        let mut bad = vec![0.0f32; 256];
        bad[0] = f32::NAN;
        bad[1] = 1.0;
        bad[64] = f32::INFINITY;
        let mut sanitized = bad.clone();
        sanitized[0] = 0.0;
        sanitized[64] = 0.0;

        let mut mono = Contract::default();
        mono.spec.channels = 1;
        let got = mono.run(|| KernelProcessor::new(Biquad::lowpass()), &[bad], &[], 32);
        let want = mono.run(
            || KernelProcessor::new(Biquad::lowpass()),
            &[sanitized],
            &[],
            32,
        );

        assert!(
            got.iter().flatten().all(|sample| sample.is_finite()),
            "non-finite input must not poison biquad output"
        );
        assert!(bits_eq(&got, &want), "non-finite samples are silence");
    }

    #[test]
    fn dc_and_nyquist_gains_are_correct() {
        // Lowpass passes DC and rejects Nyquist. Highpass is the mirror.
        let bq = prepared_biquad(BiquadKind::Lowpass, 48_000);
        let c_lp = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, 0.0);
        assert!((c_lp.magnitude(0.0) - 1.0).abs() < 1e-9, "LP DC gain ~ 1");
        assert!(c_lp.magnitude(PI) < 0.05, "LP Nyquist gain ~ 0");

        let bq = prepared_biquad(BiquadKind::Highpass, 48_000);
        let c_hp = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, 0.0);
        assert!(c_hp.magnitude(0.0) < 1e-9, "HP DC gain ~ 0");
        assert!(
            (c_hp.magnitude(PI) - 1.0).abs() < 0.05,
            "HP Nyquist gain ~ 1"
        );
    }

    #[test]
    fn cutoff_is_near_minus_3db() {
        // At Q = 1/sqrt(2), the cutoff is near -3 dB for both shapes.
        for kind in [BiquadKind::Lowpass, BiquadKind::Highpass] {
            let bq = prepared_biquad(kind, 48_000);
            let mag = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, 0.0).magnitude(bq.omega(1_000.0));
            let db = 20.0 * mag.log10();
            assert!(
                (db + 3.0).abs() < 0.5,
                "{kind:?}: cutoff gain {db} dB, expected ~ -3 dB"
            );
        }
    }

    #[test]
    fn coefficients_are_stable_across_settings() {
        // Every reachable (cutoff, Q) yields poles inside the unit circle.
        for kind in [BiquadKind::Lowpass, BiquadKind::Highpass] {
            let bq = prepared_biquad(kind, 48_000);
            for &f in &[20.0, 100.0, 1_000.0, 10_000.0, 20_000.0] {
                for &q in &[0.2, 0.707, 2.0, 8.0] {
                    assert!(
                        coeffs(&bq, f, q, 0.0).is_stable(),
                        "{kind:?} f={f} q={q} unstable"
                    );
                }
            }
        }
    }

    #[test]
    fn is_linear() {
        // Scaling the input scales the output.
        let x1 = sine(2, 1000);
        let x2: Vec<Vec<f32>> = x1
            .iter()
            .map(|ch| ch.iter().map(|&s| 2.0 * s).collect())
            .collect();
        let c = Contract::default();
        let make = || KernelProcessor::new(Biquad::lowpass());
        let y1 = c.run(make, &x1, &[], 64);
        let y2 = c.run(make, &x2, &[], 64);
        for (yc1, yc2) in y1.iter().zip(&y2) {
            for (a, b) in yc1.iter().zip(yc2) {
                assert!((b - 2.0 * a).abs() < 1e-5, "scaling broke linearity");
            }
        }
    }

    // --- Shelving / peaking EQ (the gain-bearing shapes) ---

    #[test]
    fn peaking_gain_at_center_equals_the_setting() {
        // At the center frequency, peaking EQ gain equals the dB setting.
        let bq = prepared_biquad(BiquadKind::Peaking, 48_000);
        let w0 = bq.omega(1_000.0);
        for &gain in &[-12.0, -6.0, 3.0, 12.0] {
            let mag = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, gain).magnitude(w0);
            let expected = 10f64.powf(gain / 20.0);
            assert!(
                (mag / expected - 1.0).abs() < 1e-6,
                "peaking {gain} dB: |H(w0)| {mag} vs {expected}"
            );
        }
    }

    #[test]
    fn low_shelf_hits_its_gain_at_dc_and_unity_at_nyquist() {
        let bq = prepared_biquad(BiquadKind::LowShelf, 48_000);
        for &gain in &[-9.0, 9.0] {
            let c = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, gain);
            let expected = 10f64.powf(gain / 20.0);
            assert!(
                (c.magnitude(0.0) / expected - 1.0).abs() < 1e-6,
                "low shelf {gain} dB DC gain {} vs {expected}",
                c.magnitude(0.0)
            );
            assert!(
                (c.magnitude(PI) - 1.0).abs() < 1e-6,
                "low shelf is flat (unity) at Nyquist"
            );
        }
    }

    #[test]
    fn high_shelf_hits_its_gain_at_nyquist_and_unity_at_dc() {
        let bq = prepared_biquad(BiquadKind::HighShelf, 48_000);
        for &gain in &[-9.0, 9.0] {
            let c = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, gain);
            let expected = 10f64.powf(gain / 20.0);
            assert!(
                (c.magnitude(PI) / expected - 1.0).abs() < 1e-6,
                "high shelf {gain} dB Nyquist gain {} vs {expected}",
                c.magnitude(PI)
            );
            assert!(
                (c.magnitude(0.0) - 1.0).abs() < 1e-6,
                "high shelf is flat (unity) at DC"
            );
        }
    }

    #[test]
    fn shelving_and_peaking_are_stable() {
        // Poles inside the unit circle across the gain/frequency envelope.
        for kind in [
            BiquadKind::LowShelf,
            BiquadKind::HighShelf,
            BiquadKind::Peaking,
        ] {
            let bq = prepared_biquad(kind, 48_000);
            for &f in &[50.0, 1_000.0, 10_000.0] {
                for &gain in &[-18.0, -6.0, 6.0, 18.0] {
                    assert!(
                        coeffs(&bq, f, FRAC_1_SQRT_2, gain).is_stable(),
                        "{kind:?} f={f} gain={gain} unstable"
                    );
                }
            }
        }
    }

    /// Recompute the RBJ low/high-shelf coefficients independently.
    ///
    /// The shelf term `2*sqrt(A)*alpha` affects b0, b2, a0, and a2, but cancels
    /// at DC and Nyquist.
    fn shelf_reference(kind: BiquadKind, fs: f64, f0: f64, q: f64, gain_db: f64) -> BiquadCoeffs {
        let w0 = 2.0 * PI * f0 / fs;
        let (cos_w0, sin_w0) = (w0.cos(), w0.sin());
        let alpha = sin_w0 / (2.0 * q);
        let a = 10f64.powf(gain_db / 40.0);
        let am1 = a - 1.0;
        let ap1 = a + 1.0;
        let tsa = 2.0 * a.sqrt() * alpha;
        let (b0, b1, b2, a0, a1, a2) = match kind {
            BiquadKind::LowShelf => (
                a * (ap1 - am1 * cos_w0 + tsa),
                2.0 * a * (am1 - ap1 * cos_w0),
                a * (ap1 - am1 * cos_w0 - tsa),
                ap1 + am1 * cos_w0 + tsa,
                -2.0 * (am1 + ap1 * cos_w0),
                ap1 + am1 * cos_w0 - tsa,
            ),
            _ => (
                a * (ap1 + am1 * cos_w0 + tsa),
                -2.0 * a * (am1 + ap1 * cos_w0),
                a * (ap1 + am1 * cos_w0 - tsa),
                ap1 - am1 * cos_w0 + tsa,
                2.0 * (am1 - ap1 * cos_w0),
                ap1 - am1 * cos_w0 - tsa,
            ),
        };
        let inv = 1.0 / a0;
        BiquadCoeffs {
            b0: b0 * inv,
            b1: b1 * inv,
            b2: b2 * inv,
            a1: a1 * inv,
            a2: a2 * inv,
        }
    }

    #[test]
    fn shelf_coefficients_match_the_textbook_formula() {
        // Every coefficient of both shelves matches the independent recomputation.
        for kind in [BiquadKind::LowShelf, BiquadKind::HighShelf] {
            let bq = prepared_biquad(kind, 48_000);
            for &gain in &[-9.0, 6.0] {
                let got = coeffs(&bq, 1_000.0, FRAC_1_SQRT_2, gain);
                let want = shelf_reference(kind, 48_000.0, 1_000.0, FRAC_1_SQRT_2, gain);
                for (g, w, name) in [
                    (got.b0, want.b0, "b0"),
                    (got.b1, want.b1, "b1"),
                    (got.b2, want.b2, "b2"),
                    (got.a1, want.a1, "a1"),
                    (got.a2, want.a2, "a2"),
                ] {
                    assert!(
                        (g - w).abs() < 1e-9,
                        "{kind:?} gain={gain}: {name} {g} vs textbook {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn group_delay_matches_a_phase_difference() {
        // Group delay is `-d(arg H)/dw`. Compare the implementation to a central
        // difference of the public `phase()` readout in the smooth passband.
        let bq = prepared_biquad(BiquadKind::Lowpass, 48_000);
        let c = coeffs(&bq, 1_000.0, 2.0, 0.0);
        let step = 1e-4;
        for &w in &[0.02, 0.08, 0.15] {
            let gd = c.group_delay(w);
            let expected = -(c.phase(w + step) - c.phase(w - step)) / (2.0 * step);
            assert!(
                (gd - expected).abs() < 1e-6,
                "w={w}: group delay {gd} vs phase-difference {expected}"
            );
            assert!(
                gd > 0.5,
                "w={w}: a real filter has positive group delay {gd}"
            );
        }
    }

    #[test]
    fn swept_cutoff_over_half_scale_sine_is_click_free() {
        // Sweep the low-pass cutoff from 100 Hz to 10 kHz over a steady
        // half-scale 997 Hz sine and assert that no sample-to-sample jump
        // exceeds a conservative musical bound.
        //
        // Bound derivation (empirical, deterministic). The pure signal slope
        // of a 0.5-amplitude sine at 997 Hz is A * w = 0.5 * 2*pi*997/48000
        // ~= 0.0652 per sample. With coefficients recomputed at each 32-frame
        // control-rate boundary along the smoothed cutoff ramp, the measured
        // maximum jump for this exact sweep is ~0.0659, barely above the
        // signal slope, because the Direct Form I state carries across each
        // coefficient change. The signal, events, and
        // vendored math are all deterministic, so that measurement is stable
        // across platforms; the assertion allows ~3x margin. A click (state
        // discontinuity) shows up as a jump on the order of the signal
        // amplitude (0.5), far above this bound.
        let fs = 48_000.0;
        let f = 997.0;
        let w = 2.0 * PI * f / fs;
        let hz_step = 66.0;
        let steps = 150u32;
        // Dwell at 100 Hz for 1024 frames so the sanity window below sees the
        // settled low-cutoff response, then sweep one 66 Hz step per
        // control-rate boundary up to 10 kHz.
        let dwell = 1024u32;
        let mut events = vec![ev(0, CUTOFF, 100.0)];
        events.extend(
            (1..=steps).map(|k| ev(dwell + 32 * (k - 1), CUTOFF, 100.0 + hz_step * f64::from(k))),
        );
        let frames = dwell as usize + 32 * (steps as usize + 1) + 128;
        let input: Vec<Vec<f32>> = vec![
            (0..frames)
                .map(|i| (bisque::dsp::math::sin(i as f64 * w) * 0.5) as f32)
                .collect();
            2
        ];
        let out = Contract::default().run(
            || KernelProcessor::new(Biquad::lowpass()),
            &input,
            &events,
            128,
        );
        let mut max_jump = 0.0f64;
        for ch in &out {
            for pair in ch.windows(2) {
                max_jump = max_jump.max((f64::from(pair[1]) - f64::from(pair[0])).abs());
            }
        }
        assert!(
            max_jump <= 0.2,
            "swept cutoff clicked: max sample-to-sample jump {max_jump} exceeds the              empirically derived bound with margin (0.2)"
        );
        // Sanity: the sweep engaged. At a 100 Hz cutoff the 997 Hz tone is
        // heavily attenuated; at 10 kHz it passes nearly untouched.
        let peak = |s: &[f32]| s.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let early = peak(&out[0][512..1024]);
        let late = peak(&out[0][frames - 512..]);
        assert!(
            early < 0.1 && late > 0.4,
            "sweep did not traverse the filter response (early peak {early}, late peak {late})"
        );
    }
}

#[test]
fn block_size_invariance_is_bit_exact() {
    // Cutoff and Q events change coefficients mid-stream.
    let input = sine(2, 1000);
    let events = [
        ev(0, CUTOFF, 800.0),
        ev(0, Q, 1.5),
        ev(256, CUTOFF, 5_000.0),
        ev(640, CUTOFF, 300.0),
    ];
    Contract::default().assert_block_size_invariant(
        || KernelProcessor::new(Biquad::lowpass()),
        &input,
        &events,
    );
}

#[test]
fn reset_equivalence_no_state_leak() {
    // Reset clears filter history.
    let input = sine(2, 800);
    let events = [ev(0, CUTOFF, 2_000.0), ev(200, CUTOFF, 600.0)];
    Contract::default().assert_reset_equivalence(
        || KernelProcessor::new(Biquad::highpass()),
        &input,
        &events,
    );
}

#[test]
fn peaking_render_boosts_a_tone_at_center() {
    // The render path reads and applies the gain parameter.
    let n = 16_000;
    let f0 = 1_000.0;
    let tone: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * f0 * i as f64 / 48_000.0).sin() as f32 * 0.3)
        .collect();
    let input = vec![tone.clone(), tone];
    let gain = 12.0;
    let out = Contract::default().run(
        || KernelProcessor::new(Biquad::peaking()),
        &input,
        &[ev(0, GAIN, gain)],
        256,
    );
    let rms =
        |s: &[f32]| (s.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
    let tail = n - 2_000..;
    let ratio = rms(&out[0][tail.clone()]) / rms(&input[0][tail]);
    let expected = 10f64.powf(gain / 20.0);
    assert!(
        (ratio / expected - 1.0).abs() < 0.05,
        "center boost {ratio}x vs expected {expected}x"
    );
}

#[test]
fn peaking_block_size_invariance_with_gain_events() {
    // Cutoff, Q, and gain automation stay split-invariant.
    let input = sine(2, 1000);
    let events = [
        ev(0, CUTOFF, 2_000.0),
        ev(0, GAIN, 9.0),
        ev(300, GAIN, -9.0),
        ev(640, CUTOFF, 500.0),
    ];
    Contract::default().assert_block_size_invariant(
        || KernelProcessor::new(Biquad::peaking()),
        &input,
        &events,
    );
}

#[test]
fn memory_footprint_grows_per_channel() {
    // Direct Form I state is per channel. The smoother bank is fixed overhead.
    let f1 = prepared(BiquadKind::Lowpass, spec_ch(1))
        .unwrap()
        .memory_footprint();
    let f2 = prepared(BiquadKind::Lowpass, spec_ch(2))
        .unwrap()
        .memory_footprint();
    let f3 = prepared(BiquadKind::Lowpass, spec_ch(3))
        .unwrap()
        .memory_footprint();
    assert!(f2 > f1, "footprint grows with channels");
    assert_eq!(f3 - f2, f2 - f1, "per-channel cost is constant");
}

#[test]
fn memory_footprint_is_exactly_the_state_bytes() {
    // The Direct Form I state is four f64 values per channel.
    const STATE_BYTES: usize = 4 * std::mem::size_of::<f64>(); // one State
    for ch in [1usize, 2, 4] {
        let got = Kernel::<f32>::memory_footprint(&prepared_kernel(BiquadKind::Lowpass, ch));
        assert_eq!(
            got,
            ch * STATE_BYTES,
            "{ch}-channel footprint must be channels * sizeof(State)"
        );
    }
}

#[test]
fn is_stable_decides_at_the_unit_circle_boundary() {
    // Schur-Cohn for a normalized biquad: stable iff |a2| < 1 and
    // |a1| < 1 + a2.
    let coeffs = |a1: f64, a2: f64| BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1,
        a2,
    };
    // Inside the stability triangle.
    assert!(coeffs(0.5, 0.25).is_stable(), "a pole pair well inside");
    // Outside the |a2| boundary.
    assert!(!coeffs(0.0, 1.5).is_stable(), "|a2| >= 1 is unstable");
    // Outside the |a1| boundary.
    assert!(!coeffs(2.0, 0.0).is_stable(), "|a1| >= 1 + a2 is unstable");
    // Exactly on the |a2| = 1 boundary.
    assert!(
        !coeffs(0.0, 1.0).is_stable(),
        "a2 on the unit circle is unstable"
    );
    // Exactly on the |a1| = 1 + a2 boundary.
    assert!(
        !coeffs(1.5, 0.5).is_stable(),
        "a1 on the stability edge is unstable"
    );
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
        // Draining the ring-out must be byte-identical to feeding explicit
        // silence: flush continues the recursion with the coefficients last
        // seen by render. The automation settles well inside the body, so
        // the cached coefficients match the zero-fed run's settled ones.
        let input = sine(2, 2_000);
        let events = [ev(0, CUTOFF, 300.0), ev(0, Q, 8.0), ev(0, GAIN, 6.0)];
        let c = Contract::default();
        let make = || KernelProcessor::new(Biquad::peaking());

        let mut flushed = make();
        Processor::<f32>::prepare(&mut flushed, c.spec).expect("prepare");
        let _ = c.run_reusing(&mut flushed, &input, &events, 64);
        let (tail, _) = flush_stage(&mut flushed, 512);

        let mut zero_fed = make();
        Processor::<f32>::prepare(&mut zero_fed, c.spec).expect("prepare");
        let _ = c.run_reusing(&mut zero_fed, &input, &events, 64);
        let zeros: Buffers = vec![vec![0.0f32; 512]; 2];
        let out = c.run_reusing(&mut zero_fed, &zeros, &[], 64);

        assert!(
            bits_eq(&tail, &out),
            "the drained ring-out must match zero-fed processing bit for bit"
        );
    }

    #[test]
    fn tail_bound_matches_the_declared_range_worst_case() {
        // Independently recompute the conservative bound: the slowest pole
        // pair sits at whichever cutoff extreme has the smaller sin(w0), at
        // maximum Q and gain, and the drain budgets 1e3 -> 1e-6 of decay.
        // At 48 kHz the low extreme binds; at 8 kHz the Nyquist-side extreme
        // does, so both arms of the worst-case selection are pinned.
        // Tolerate one frame of libm-vs-std rounding.
        for fs_hz in [48_000_u32, 8_000] {
            let spec = ProcessSpec {
                sample_rate: fs_hz,
                ..Contract::default().spec
            };
            let mut p = KernelProcessor::new(Biquad::peaking());
            Processor::<f32>::prepare(&mut p, spec).expect("prepare");
            let Tail::Frames(bound) = Processor::<f32>::tail(&p) else {
                panic!("a prepared biquad declares a finite tail");
            };
            let fs = f64::from(fs_hz);
            let tau = std::f64::consts::TAU;
            let clamp = |f: f64| f.clamp(1.0, fs * 0.5 * 0.999);
            let w_lo = tau * clamp(10.0) / fs;
            let w_hi = tau * clamp(24_000.0) / fs;
            let sin_min = w_lo.sin().min(w_hi.sin());
            let a_max = 10.0_f64.powf(24.0 / 40.0);
            let alpha = sin_min / (2.0 * 16.0 * a_max);
            let r = ((1.0 - alpha) / (1.0 + alpha)).sqrt();
            let expected = (1e9_f64.ln() / -r.ln()).ceil();
            assert!(
                ((bound as f64) - expected).abs() <= 1.0,
                "at {fs_hz} Hz the declared tail bound {bound} must match the \
                 range worst case {expected}"
            );
        }
    }

    #[test]
    fn full_drain_ends_done_within_the_declared_tail() {
        // A resonant ring-out drains until the state decays below the floor,
        // long before the conservative declared bound. A peaking filter at
        // 0 dB gain is transparent no matter the Q, so the boost is what
        // creates the resonance; the impulse sits on the final body frame,
        // after the ramps settle, so the ring-out happens inside the drain.
        let mut input: Buffers = vec![vec![0.0f32; 256]; 2];
        input[0][255] = 1.0;
        input[1][255] = 1.0;
        let events = [ev(0, Q, 16.0), ev(0, GAIN, 24.0)];
        let c = Contract::default();
        let mut proc = KernelProcessor::new(Biquad::peaking());
        Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");
        let Tail::Frames(bound) = Processor::<f32>::tail(&proc) else {
            panic!("a prepared biquad declares a finite tail");
        };
        assert!(bound > 0, "the declared tail bound is nonzero");
        let _ = c.run_reusing(&mut proc, &input, &events, 64);

        let mut total = 0usize;
        loop {
            // Chunks well inside the ring-out (Q = 16 at 1 kHz decays to the
            // floor in ~3400 frames), so the first chunk leaves the ring hot.
            let (_, produced) = flush_stage(&mut proc, 1_024);
            if total == 0 {
                // `done` comes from the decay floor or the bound, not from
                // merely having drained one chunk.
                assert!(!produced.done, "a hot ring is not done after one chunk");
            }
            total += produced.frames;
            if produced.done {
                break;
            }
            assert!(total <= bound, "a finite tail must not over-produce");
        }
        assert!(total > 0, "a rung filter has a nonempty drain");
        assert!(
            total < bound,
            "the decay floor ends the drain before the conservative bound \
             ({total} of {bound})"
        );
    }

    #[test]
    fn new_input_starts_a_new_drain() {
        // Drain a rung filter until the decay early-exit reports done, then
        // process new input: the next flush must deliver a live tail again.
        let input = sine(2, 512);
        let c = Contract::default();
        let mut proc = KernelProcessor::new(Biquad::peaking());
        Processor::<f32>::prepare(&mut proc, c.spec).expect("prepare");

        let _ = c.run_reusing(&mut proc, &input, &[], 64);
        let mut guard = 0usize;
        loop {
            let (_, produced) = flush_stage(&mut proc, 1_024);
            if produced.done {
                break;
            }
            guard += 1;
            assert!(guard < 10_000, "the decayed drain must report done");
        }

        let _ = c.run_reusing(&mut proc, &input, &[], 64);
        let (stage, produced) = flush_stage(&mut proc, 100);
        assert_eq!(produced.frames, 100, "a new drain delivers frames again");
        assert!(
            stage.iter().flatten().any(|&s| s != 0.0),
            "the new drain carries live ring-out"
        );
    }
}

mod validation {
    use super::*;

    #[test]
    fn zero_sample_rate_is_rejected() {
        let bad = ProcessSpec {
            sample_rate: 0,
            ..Contract::default().spec
        };
        assert!(matches!(
            prepared(BiquadKind::Lowpass, bad),
            Err(DspError::UnsupportedSpec(_))
        ));
    }

    #[test]
    fn too_low_sample_rates_are_rejected_not_deferred_to_a_render_panic() {
        // At 1 or 2 Hz the cutoff clamp band (1.0, fs/2 * 0.999) inverts and
        // `f64::clamp` would panic in render. `prepare` rejects the spec before
        // render can reach that path.
        for fs in [1u32, 2] {
            let bad = ProcessSpec {
                sample_rate: fs,
                ..Contract::default().spec
            };
            assert!(
                matches!(
                    prepared(BiquadKind::Lowpass, bad),
                    Err(DspError::UnsupportedSpec(_))
                ),
                "{fs} Hz must be rejected in prepare"
            );
        }
        // The lowest rate with a non-inverted clamp band prepares fine.
        let ok = ProcessSpec {
            sample_rate: 3,
            ..Contract::default().spec
        };
        assert!(prepared(BiquadKind::Lowpass, ok).is_ok());
    }
}

mod snapshots {
    use super::*;

    #[test]
    fn output_is_byte_reproducible() {
        // Coefficients use vendored libm and the f64 recurrence is FMA-free.
        let input = sine(2, 1000);
        let events = [ev(0, CUTOFF, 700.0), ev(400, CUTOFF, 3_000.0)];
        let c = Contract::default();
        let make = || KernelProcessor::new(Biquad::lowpass());
        let a = c.run(make, &input, &events, 128);
        let b = c.run(make, &input, &events, 128);
        assert!(bits_eq(&a, &b), "biquad output must be byte-reproducible");
    }
}
