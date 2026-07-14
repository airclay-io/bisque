// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for the `Measurer` family.
//!
//! Covers `PeakMeter`, `RmsMeter`, `CrestMeter`, `TruePeakMeter`,
//! `WindowedRmsMeter`, `LoudnessMeter`, and `ClipMeter`. Checks include DC,
//! sine RMS and crest factor, LUFS, sliding-window RMS, clip counts, reset
//! behavior, and block-size invariance.

use std::f64::consts::SQRT_2;

use bisque::analysis::{
    linear_to_dbfs, ClipMeter, ClipMeterSettings, CrestMeter, LoudnessMeter, LoudnessMeterSettings,
    LoudnessReading, MeanMeter, PeakMeter, RmsMeter, TruePeakMeter, WindowedRmsMeter,
    WindowedRmsMeterSettings,
};
use bisque::processor::{DspError, Measurer, ProcessSpec};
use bisque::testing::{observe_blocks, tone_stereo, Buffers, Contract};

fn spec() -> ProcessSpec {
    Contract::default().spec // 48 kHz stereo
}

/// Prepare a meter, observe `signal` in `block`-frame chunks, and read it.
///
/// The `Reading = f64` bound fixes `T = f32`.
fn measure<M: Measurer<f32, Reading = f64>>(
    mut meter: M,
    signal: &[Vec<f32>],
    block: usize,
) -> f64 {
    meter.prepare(spec()).expect("prepare");
    observe_blocks(&mut meter, signal, block);
    meter.read()
}

/// Prepare, observe, and read a `u64`-reading meter.
fn measure_u64<M: Measurer<f32, Reading = u64>>(
    mut meter: M,
    signal: &[Vec<f32>],
    block: usize,
) -> u64 {
    meter.prepare(spec()).expect("prepare");
    observe_blocks(&mut meter, signal, block);
    meter.read()
}

/// Prepare, observe, and read a loudness meter.
fn measure_loudness(
    mut meter: LoudnessMeter,
    signal: &[Vec<f32>],
    block: usize,
) -> LoudnessReading {
    Measurer::<f32>::prepare(&mut meter, spec()).expect("prepare");
    observe_blocks(&mut meter, signal, block);
    Measurer::<f32>::read(&meter)
}

/// A two-channel DC signal at `level`.
fn dc(frames: usize, level: f32) -> Buffers {
    vec![vec![level; frames]; 2]
}

/// A two-channel sine of equal amplitude `amp` in both channels.
fn tone(frames: usize, amp: f32) -> Buffers {
    let ch: Vec<f32> = (0..frames).map(|i| amp * (i as f32 * 0.05).sin()).collect();
    vec![ch.clone(), ch]
}

/// A two-channel sine at `hz`.
fn sine_hz(frames: usize, hz: f32, amp: f32) -> Buffers {
    let phase_inc = std::f32::consts::TAU * hz / 48_000.0;
    let ch: Vec<f32> = (0..frames)
        .map(|i| amp * (i as f32 * phase_inc).sin())
        .collect();
    vec![ch.clone(), ch]
}

/// Concatenate two equally-shaped multi-channel signals along time.
fn concat(head: &[Vec<f32>], tail: &[Vec<f32>]) -> Buffers {
    head.iter()
        .zip(tail)
        .map(|(a, b)| {
            let mut ch = a.clone();
            ch.extend_from_slice(b);
            ch
        })
        .collect()
}

/// An `f_s/4` sine sampled at the +pi/4 phase.
///
/// Each sample is +/-0.707, while the continuous waveform peaks at 1.0 between
/// samples.
fn inter_sample(frames: usize) -> Buffers {
    let v = std::f32::consts::FRAC_1_SQRT_2;
    let pat = [v, v, -v, -v];
    let ch: Vec<f32> = (0..frames).map(|i| pat[i % 4]).collect();
    vec![ch.clone(), ch]
}

fn assert_negative_infinity(value: f64) {
    assert!(
        value.is_infinite() && value.is_sign_negative(),
        "expected -inf, got {value}"
    );
}

fn assert_near(got: f64, want: f64, tolerance: f64) {
    assert!(
        (got - want).abs() <= tolerance,
        "got {got}, wanted {want} +/- {tolerance}"
    );
}

mod audio {
    use super::*;

    #[test]
    fn peak_is_the_max_abs_sample() {
        // Cross-channel maximum absolute value, exact.
        let signal: Buffers = vec![
            vec![0.1, -0.8, 0.3, 0.5, -0.2],
            vec![0.2, 0.4, -0.6, 0.1, 0.7],
        ];
        let peak = measure(PeakMeter::new(), &signal, 2);
        assert_eq!(peak, f64::from(0.8f32), "peak is the largest |sample|");
    }

    #[test]
    fn mean_of_dc_is_the_dc_level_per_channel() {
        // 0.25 is exactly representable, so the per-channel mean is exact.
        let signal = dc(500, 0.25);
        let mut m = MeanMeter::new();
        Measurer::<f32>::prepare(&mut m, spec()).expect("prepare");
        observe_blocks(&mut m, &signal, 64);
        assert_eq!(m.channel_mean(0), 0.25);
        assert_eq!(m.channel_mean(1), 0.25);
        assert_eq!(
            Measurer::<f32>::read(&m),
            0.25,
            "headline is the max abs mean"
        );
    }

    #[test]
    fn mean_read_is_max_abs_not_pooled() {
        // Opposite-signed DC would cancel in a pooled mean; max-abs does not.
        let signal: Buffers = vec![vec![0.5; 400], vec![-0.5; 400]];
        let mut m = MeanMeter::new();
        Measurer::<f32>::prepare(&mut m, spec()).expect("prepare");
        observe_blocks(&mut m, &signal, 64);
        assert_eq!(m.channel_mean(0), 0.5);
        assert_eq!(m.channel_mean(1), -0.5);
        assert_eq!(Measurer::<f32>::read(&m), 0.5, "max abs, not pooled 0");
    }

    #[test]
    fn mean_ignores_non_finite_input() {
        // NaN and infinities are sanitized to 0 and do not corrupt the mean.
        let signal: Buffers = vec![
            vec![f32::NAN, 0.5, f32::INFINITY, 0.5],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let mut m = MeanMeter::new();
        Measurer::<f32>::prepare(&mut m, spec()).expect("prepare");
        observe_blocks(&mut m, &signal, 2);
        assert_eq!(m.channel_mean(0), 0.25, "(0 + 0.5 + 0 + 0.5) / 4");
        assert_eq!(m.channel_mean(1), 0.25);
    }

    #[test]
    fn rms_of_dc_is_the_dc_level() {
        // RMS of a constant c is |c|.
        let level = 0.3f32;
        let rms = measure(RmsMeter::new(), &dc(1000, level), 64);
        assert!(
            (rms - f64::from(level)).abs() < 1e-9,
            "RMS of DC {level} should be {level}, got {rms}"
        );
    }

    #[test]
    fn rms_of_sine_is_amplitude_over_sqrt2() {
        let amp = 0.6f32;
        let rms = measure(RmsMeter::new(), &tone(20_000, amp), 128);
        let want = f64::from(amp) / SQRT_2;
        assert!(
            (rms - want).abs() < 1e-3,
            "RMS of a sine should be A/sqrt(2) = {want}, got {rms}"
        );
    }

    #[test]
    fn crest_of_sine_is_sqrt2() {
        let crest = measure(CrestMeter::new(), &tone(20_000, 0.6), 128);
        assert!(
            (crest - SQRT_2).abs() < 1e-2,
            "crest factor of a sine should be sqrt(2), got {crest}"
        );
    }

    #[test]
    fn crest_of_dc_is_one() {
        // Peak == RMS == |c| for DC, so the crest factor is exactly 1.
        let crest = measure(CrestMeter::new(), &dc(1000, 0.3), 64);
        assert!(
            (crest - 1.0).abs() < 1e-9,
            "crest factor of DC should be 1, got {crest}"
        );
    }

    #[test]
    fn true_peak_catches_the_inter_sample_peak() {
        // The sample peak is 0.707. The true peak is 1.0 between samples.
        let signal = inter_sample(512);
        let sample = measure(PeakMeter::new(), &signal, 64);
        let truepeak = measure(TruePeakMeter::new(), &signal, 64);
        assert!(
            (sample - f64::from(std::f32::consts::FRAC_1_SQRT_2)).abs() < 1e-6,
            "sample peak is 1/sqrt(2), got {sample}"
        );
        assert!(
            truepeak > 0.97 && truepeak > sample,
            "true peak recovers ~1.0 (> the 0.707 sample peak), got {truepeak}"
        );
    }

    #[test]
    fn true_peak_is_at_least_the_sample_peak() {
        // The oversampler reproduces the samples at phase 0.
        let signal = tone(8000, 0.8);
        let sample = measure(PeakMeter::new(), &signal, 128);
        let truepeak = measure(TruePeakMeter::new(), &signal, 128);
        assert!(
            truepeak >= sample - 1e-9,
            "true peak {truepeak} must be >= sample peak {sample}"
        );
    }

    // --- LoudnessMeter ---

    #[test]
    fn loudness_of_silence_is_negative_infinity() {
        let reading = measure_loudness(LoudnessMeter::new(), &dc(48_000, 0.0), 256);
        assert_negative_infinity(reading.momentary_lufs);
        assert_negative_infinity(reading.short_term_lufs);
        assert_negative_infinity(reading.integrated_lufs);
    }

    #[test]
    fn loudness_tracks_amplitude_ratio_in_lu() {
        let loud = measure_loudness(LoudnessMeter::new(), &sine_hz(192_000, 997.0, 0.2), 256);
        let quiet = measure_loudness(LoudnessMeter::new(), &sine_hz(192_000, 997.0, 0.02), 256);

        assert_near(loud.momentary_lufs - quiet.momentary_lufs, 20.0, 0.02);
        assert_near(loud.short_term_lufs - quiet.short_term_lufs, 20.0, 0.02);
        assert_near(loud.integrated_lufs - quiet.integrated_lufs, 20.0, 0.02);
    }

    #[test]
    fn loudness_treats_non_finite_samples_as_silence() {
        let mut bad = sine_hz(192_000, 997.0, 0.2);
        bad[0][123] = f32::NAN;
        bad[1][456] = f32::INFINITY;
        let mut sanitized = bad.clone();
        sanitized[0][123] = 0.0;
        sanitized[1][456] = 0.0;

        let got = measure_loudness(LoudnessMeter::new(), &bad, 256);
        let want = measure_loudness(LoudnessMeter::new(), &sanitized, 256);

        assert!(got.momentary_lufs.is_finite());
        assert!(got.short_term_lufs.is_finite());
        assert!(got.integrated_lufs.is_finite());
        assert_near(got.momentary_lufs, want.momentary_lufs, 1e-12);
        assert_near(got.short_term_lufs, want.short_term_lufs, 1e-12);
        assert_near(got.integrated_lufs, want.integrated_lufs, 1e-12);
    }

    #[test]
    fn loudness_honors_explicit_channel_weights() {
        let signal = sine_hz(192_000, 997.0, 0.2);
        let stereo = measure_loudness(LoudnessMeter::new(), &signal, 512);
        let left_only = measure_loudness(
            LoudnessMeter::with_settings(LoudnessMeterSettings::with_channel_weights(vec![
                1.0, 0.0,
            ])),
            &signal,
            512,
        );

        let expected = 10.0 * 2.0f64.log10();
        assert_near(
            stereo.integrated_lufs - left_only.integrated_lufs,
            expected,
            0.02,
        );
    }

    #[test]
    fn loudness_requires_explicit_weights_for_ambiguous_surround_layouts() {
        let surround_spec = ProcessSpec {
            channels: 6,
            ..spec()
        };

        let mut inferred = LoudnessMeter::new();
        assert!(matches!(
            Measurer::<f32>::prepare(&mut inferred, surround_spec),
            Err(DspError::InvalidParam(_))
        ));

        let mut five_point_one =
            LoudnessMeter::with_settings(LoudnessMeterSettings::five_point_one());
        Measurer::<f32>::prepare(&mut five_point_one, surround_spec).expect("5.1 prepare");
    }

    #[test]
    fn loudness_rejects_bad_settings() {
        let mut wrong_count =
            LoudnessMeter::with_settings(LoudnessMeterSettings::with_channel_weights(vec![1.0]));
        assert!(matches!(
            Measurer::<f32>::prepare(&mut wrong_count, spec()),
            Err(DspError::InvalidParam(_))
        ));

        let mut negative =
            LoudnessMeter::with_settings(LoudnessMeterSettings::with_channel_weights(vec![
                1.0, -1.0,
            ]));
        assert!(matches!(
            Measurer::<f32>::prepare(&mut negative, spec()),
            Err(DspError::InvalidParam(_))
        ));

        let mut zero_rate = LoudnessMeter::new();
        let bad_spec = ProcessSpec {
            sample_rate: 0,
            ..spec()
        };
        assert!(matches!(
            Measurer::<f32>::prepare(&mut zero_rate, bad_spec),
            Err(DspError::UnsupportedSpec(_))
        ));

        let mut low_rate = LoudnessMeter::new();
        let bad_spec = ProcessSpec {
            sample_rate: 3_363,
            ..spec()
        };
        assert!(matches!(
            Measurer::<f32>::prepare(&mut low_rate, bad_spec),
            Err(DspError::UnsupportedSpec(_))
        ));

        let mut minimum_rate = LoudnessMeter::new();
        let minimum_spec = ProcessSpec {
            sample_rate: 3_364,
            ..spec()
        };
        Measurer::<f32>::prepare(&mut minimum_rate, minimum_spec)
            .expect("the minimum K-weighting sample rate is supported");

        let mut too_short =
            LoudnessMeter::with_settings(LoudnessMeterSettings::with_max_integrated_seconds(0.1));
        assert!(matches!(
            Measurer::<f32>::prepare(&mut too_short, spec()),
            Err(DspError::InvalidParam(_))
        ));

        let mut non_finite = LoudnessMeter::with_settings(
            LoudnessMeterSettings::with_max_integrated_seconds(f64::NAN),
        );
        assert!(matches!(
            Measurer::<f32>::prepare(&mut non_finite, spec()),
            Err(DspError::InvalidParam(_))
        ));
    }

    #[test]
    fn integrated_loudness_reports_when_history_is_incomplete() {
        let signal = sine_hz(48_000, 997.0, 0.2);
        let limited =
            LoudnessMeter::with_settings(LoudnessMeterSettings::with_max_integrated_seconds(0.5));

        let reading = measure_loudness(limited, &signal, 256);

        assert!(reading.integrated_lufs.is_finite());
        assert!(
            !reading.integrated_complete,
            "integrated history should report overflow after the configured duration"
        );
    }

    #[test]
    fn integrated_loudness_absolute_gate_excludes_silence() {
        // EBU Tech 3341 gating excludes blocks below the -70 LUFS absolute gate.
        // Here, trailing silence leaves the integrated measurement unchanged.
        let loud = sine_hz(20 * 48_000, 997.0, 0.5);
        let silence: Buffers = vec![vec![0.0f32; 20 * 48_000]; 2];
        let with_silence = concat(&loud, &silence);

        let i_loud = measure_loudness(LoudnessMeter::new(), &loud, 1024).integrated_lufs;
        let i_gated = measure_loudness(LoudnessMeter::new(), &with_silence, 1024).integrated_lufs;

        assert!(
            i_loud.is_finite(),
            "loud-only integrated loudness is finite"
        );
        assert_near(i_gated, i_loud, 0.1);
    }

    #[test]
    fn loudness_and_true_peak_match_the_ebur128_oracle() {
        // The `ebur128` crate is the EBU R128 conformance oracle: on a
        // deterministic multi-tone stereo signal, integrated LUFS must agree
        // within +/-0.1 LU and true peak within +/-0.2 dB.
        let frames = 10 * 48_000;
        let signal = tone_stereo(frames);

        // bisque meters.
        let integrated = measure_loudness(LoudnessMeter::new(), &signal, 1024).integrated_lufs;
        let mut tpm = TruePeakMeter::new();
        Measurer::<f32>::prepare(&mut tpm, spec()).expect("prepare");
        observe_blocks(&mut tpm, &signal, 1024);
        let tp_db = linear_to_dbfs(Measurer::<f32>::read(&tpm));

        // ebur128 oracle (MODE_I plus true peak).
        let mut oracle =
            ebur128::EbuR128::new(2, 48_000, ebur128::Mode::I | ebur128::Mode::TRUE_PEAK)
                .expect("oracle setup");
        let mut interleaved = vec![0.0f32; frames * 2];
        for i in 0..frames {
            interleaved[2 * i] = signal[0][i];
            interleaved[2 * i + 1] = signal[1][i];
        }
        oracle.add_frames_f32(&interleaved).expect("oracle frames");
        let oracle_lufs = oracle.loudness_global().expect("oracle integrated");
        let oracle_tp_lin = oracle
            .true_peak(0)
            .expect("oracle true peak L")
            .max(oracle.true_peak(1).expect("oracle true peak R"));
        let oracle_tp_db = linear_to_dbfs(oracle_tp_lin);

        assert!(
            integrated.is_finite() && oracle_lufs.is_finite(),
            "both integrated readings must be finite"
        );
        assert!(
            (integrated - oracle_lufs).abs() <= 0.1,
            "integrated {integrated} LUFS vs ebur128 {oracle_lufs} LUFS must agree within 0.1 LU"
        );
        assert!(
            (tp_db - oracle_tp_db).abs() <= 0.2,
            "true peak {tp_db} dB vs ebur128 {oracle_tp_db} dB must agree within 0.2 dB"
        );
    }

    #[test]
    fn integrated_loudness_relative_gate_excludes_quiet_section() {
        // EBU Tech 3341 gating: a section 20 dB below program loudness falls
        // under the -10 LU relative gate and must be excluded. Without the gate
        // the quiet half would drag the integrated value about 3 LU lower.
        let loud = sine_hz(20 * 48_000, 997.0, 0.5);
        let quiet = sine_hz(20 * 48_000, 997.0, 0.05); // 20 dB down
        let combined = concat(&loud, &quiet);

        let i_loud = measure_loudness(LoudnessMeter::new(), &loud, 1024).integrated_lufs;
        let i_combined = measure_loudness(LoudnessMeter::new(), &combined, 1024).integrated_lufs;

        assert_near(i_combined, i_loud, 0.15);
    }

    // --- WindowedRmsMeter ---

    #[test]
    fn windowed_rms_of_dc_is_the_level() {
        // Over a fully-filled window of constant `c`, the RMS is exactly |c|.
        let c = 0.3f32;
        let rms = measure(WindowedRmsMeter::new(), &dc(2000, c), 64);
        assert!(
            (rms - f64::from(c)).abs() < 1e-9,
            "windowed RMS of DC {c} should be {c}, got {rms}"
        );
    }

    #[test]
    fn windowed_rms_forgets_samples_outside_the_window() {
        // After a loud section followed by a quiet section, the window contains
        // only the quiet section.
        let window = 600;
        let mut ch = vec![0.8f32; window];
        ch.extend(std::iter::repeat_n(0.1f32, window));
        let signal: Buffers = vec![ch.clone(), ch];

        let windowed = measure(
            WindowedRmsMeter::with_settings(WindowedRmsMeterSettings::new().window_frames(window)),
            &signal,
            64,
        );
        assert!(
            (windowed - 0.1).abs() < 1e-6,
            "windowed RMS should forget the loud past, got {windowed}"
        );
        let whole = measure(RmsMeter::new(), &signal, 64);
        assert!(
            whole > 0.5,
            "the whole-stream RMS still includes the loud part, got {whole}"
        );
    }

    #[test]
    fn windowed_rms_before_the_window_fills_uses_what_it_has() {
        // Fewer frames than the window: the RMS is over the frames seen, not
        // diluted by the empty remainder.
        let c = 0.5f32;
        let rms = measure(
            WindowedRmsMeter::with_settings(WindowedRmsMeterSettings::new().window_frames(4096)),
            &dc(500, c),
            64,
        );
        assert!(
            (rms - f64::from(c)).abs() < 1e-9,
            "partial-window RMS of DC {c} should be {c}, got {rms}"
        );
    }

    // --- ClipMeter ---

    #[test]
    fn clip_counts_samples_at_or_above_full_scale() {
        // 7 of each channel's samples reach full scale; nothing else does.
        let mut ch = vec![0.5f32; 100];
        for slot in ch.iter_mut().take(7) {
            *slot = 1.25;
        }
        let signal: Buffers = vec![ch.clone(), ch];
        let count = measure_u64(ClipMeter::new(), &signal, 16);
        assert_eq!(count, 14, "7 clipped per channel over 2 channels");
    }

    #[test]
    fn clip_ignores_below_threshold_and_honors_a_custom_one() {
        // Everything below full scale: no clips. A lower threshold then catches them.
        let signal = dc(200, 0.6);
        assert_eq!(measure_u64(ClipMeter::new(), &signal, 32), 0, "0.6 < 1.0");
        assert_eq!(
            measure_u64(
                ClipMeter::with_settings(ClipMeterSettings::new().threshold(0.5)),
                &signal,
                32
            ),
            400,
            "all 200x2 samples clear a 0.5 threshold"
        );
    }

    // --- linear_to_dbfs ---

    #[test]
    fn dbfs_of_full_scale_is_zero() {
        // Full scale (1.0) is the 0 dB reference, exactly.
        assert_eq!(linear_to_dbfs(1.0), 0.0, "0 dBFS at full scale");
    }

    #[test]
    fn dbfs_of_half_scale_is_minus_six_db() {
        // 20 * log10(0.5) = -6.0206 dB.
        let got = linear_to_dbfs(0.5);
        assert!(
            (got - (-6.020_599_913_279_624)).abs() < 1e-12,
            "20*log10(0.5) should be -6.0206 dB, got {got}"
        );
    }

    #[test]
    fn dbfs_of_silence_and_negative_is_negative_infinity() {
        // Zero and negative amplitudes map to negative infinity.
        let zero = linear_to_dbfs(0.0);
        assert!(
            zero.is_infinite() && zero.is_sign_negative(),
            "silence maps to -inf, got {zero}"
        );
        let neg = linear_to_dbfs(-0.5);
        assert!(
            neg.is_infinite() && neg.is_sign_negative(),
            "a negative amplitude maps to -inf (not NaN), got {neg}"
        );
    }
}

#[test]
fn readings_are_block_size_invariant() {
    // Meter readings are bit-exact across host block splits.
    let signal = tone(4096, 0.5);
    let peak = measure(PeakMeter::new(), &signal, 4096);
    let rms = measure(RmsMeter::new(), &signal, 4096);
    let crest = measure(CrestMeter::new(), &signal, 4096);
    let truepeak = measure(TruePeakMeter::new(), &signal, 4096);
    for &block in &[1usize, 7, 32, 64, 100, 1000] {
        assert_eq!(
            measure(PeakMeter::new(), &signal, block).to_bits(),
            peak.to_bits(),
            "peak diverged at block {block}"
        );
        assert_eq!(
            measure(RmsMeter::new(), &signal, block).to_bits(),
            rms.to_bits(),
            "rms diverged at block {block}"
        );
        assert_eq!(
            measure(CrestMeter::new(), &signal, block).to_bits(),
            crest.to_bits(),
            "crest diverged at block {block}"
        );
        assert_eq!(
            measure(TruePeakMeter::new(), &signal, block).to_bits(),
            truepeak.to_bits(),
            "true-peak diverged at block {block}"
        );
    }
}

#[test]
fn loudness_readings_are_block_size_invariant() {
    let signal = sine_hz(192_000, 997.0, 0.2);
    let reference = measure_loudness(LoudnessMeter::new(), &signal, 4096);
    for &block in &[1usize, 7, 32, 64, 100, 1000] {
        let got = measure_loudness(LoudnessMeter::new(), &signal, block);
        assert_eq!(
            got.momentary_lufs.to_bits(),
            reference.momentary_lufs.to_bits(),
            "momentary loudness diverged at block {block}"
        );
        assert_eq!(
            got.short_term_lufs.to_bits(),
            reference.short_term_lufs.to_bits(),
            "short-term loudness diverged at block {block}"
        );
        assert_eq!(
            got.integrated_lufs.to_bits(),
            reference.integrated_lufs.to_bits(),
            "integrated loudness diverged at block {block}"
        );
        assert_eq!(
            got.integrated_complete, reference.integrated_complete,
            "integrated completeness diverged at block {block}"
        );
    }
}

#[test]
fn reset_clears_accumulated_state() {
    // Qualify calls to f32 because `RmsMeter` implements `Measurer<T>` for every
    // sample type.
    let signal = tone(2000, 0.7);
    let mut meter = RmsMeter::new();
    Measurer::<f32>::prepare(&mut meter, spec()).unwrap();
    observe_blocks(&mut meter, &signal, 64);
    assert!(
        Measurer::<f32>::read(&meter) > 0.0,
        "observed signal gives a non-zero RMS"
    );
    Measurer::<f32>::reset(&mut meter);
    assert_eq!(
        Measurer::<f32>::read(&meter),
        0.0,
        "reset returns the meter to silence"
    );
    // Observing after reset matches a fresh instance.
    observe_blocks(&mut meter, &signal, 64);
    assert_eq!(
        Measurer::<f32>::read(&meter).to_bits(),
        measure(RmsMeter::new(), &signal, 64).to_bits(),
        "post-reset reading matches a fresh meter"
    );
}

#[test]
fn loudness_reset_returns_to_silence_and_matches_a_fresh_meter() {
    let signal = sine_hz(192_000, 997.0, 0.2);
    let mut meter = LoudnessMeter::new();
    Measurer::<f32>::prepare(&mut meter, spec()).unwrap();
    observe_blocks(&mut meter, &signal, 512);
    let before_reset = Measurer::<f32>::read(&meter);
    assert!(
        before_reset.integrated_lufs.is_finite(),
        "observed signal gives finite integrated loudness"
    );

    Measurer::<f32>::reset(&mut meter);
    let reset = Measurer::<f32>::read(&meter);
    assert_negative_infinity(reset.momentary_lufs);
    assert_negative_infinity(reset.short_term_lufs);
    assert_negative_infinity(reset.integrated_lufs);
    assert!(reset.integrated_complete);

    observe_blocks(&mut meter, &signal, 512);
    let after_reset = Measurer::<f32>::read(&meter);
    let fresh = measure_loudness(LoudnessMeter::new(), &signal, 512);
    assert_eq!(
        after_reset.integrated_lufs.to_bits(),
        fresh.integrated_lufs.to_bits(),
        "post-reset integrated loudness matches a fresh meter"
    );
}

#[test]
fn peak_crest_truepeak_reset_to_silence() {
    // Reset returns each f64 meter to its silent reading.
    let signal = tone(2000, 0.7);

    let mut peak = PeakMeter::new();
    Measurer::<f32>::prepare(&mut peak, spec()).unwrap();
    observe_blocks(&mut peak, &signal, 64);
    assert!(Measurer::<f32>::read(&peak) > 0.0, "peak is non-zero");
    Measurer::<f32>::reset(&mut peak);
    assert_eq!(Measurer::<f32>::read(&peak), 0.0, "peak resets to 0");

    let mut crest = CrestMeter::new();
    Measurer::<f32>::prepare(&mut crest, spec()).unwrap();
    observe_blocks(&mut crest, &signal, 64);
    assert!(Measurer::<f32>::read(&crest) > 0.0, "crest is non-zero");
    Measurer::<f32>::reset(&mut crest);
    assert_eq!(Measurer::<f32>::read(&crest), 0.0, "crest resets to 0");

    let mut truepeak = TruePeakMeter::new();
    Measurer::<f32>::prepare(&mut truepeak, spec()).unwrap();
    observe_blocks(&mut truepeak, &signal, 64);
    assert!(
        Measurer::<f32>::read(&truepeak) > 0.0,
        "true peak is non-zero"
    );
    Measurer::<f32>::reset(&mut truepeak);
    assert_eq!(
        Measurer::<f32>::read(&truepeak),
        0.0,
        "true peak resets to 0"
    );
}

#[test]
fn clip_clipped_accessor_returns_the_running_count() {
    // The inherent `clipped()` accessor reports the running count.
    let mut ch = vec![0.2f32; 50];
    for slot in ch.iter_mut().take(9) {
        *slot = 1.5;
    }
    let signal: Buffers = vec![ch.clone(), ch];
    let mut clip = ClipMeter::new();
    Measurer::<f32>::prepare(&mut clip, spec()).unwrap();
    observe_blocks(&mut clip, &signal, 16);
    assert_eq!(
        clip.clipped(),
        18,
        "9 clipped per channel over 2 channels, via clipped()"
    );
}

#[test]
fn clip_prepare_clears_a_dirty_count() {
    // `prepare` clears an accumulated count on a reused meter.
    let clipped = dc(100, 1.5);
    let mut clip = ClipMeter::new();
    Measurer::<f32>::prepare(&mut clip, spec()).unwrap();
    observe_blocks(&mut clip, &clipped, 16);
    assert!(clip.clipped() > 0, "dirtied to a non-zero count");
    Measurer::<f32>::prepare(&mut clip, spec()).unwrap();
    assert_eq!(clip.clipped(), 0, "re-prepare clears the count");
}

#[test]
fn windowed_and_clip_readings_are_block_size_invariant() {
    // A varying signal with a clipped patch, so both meters have non-trivial state.
    let mut ch: Vec<f32> = (0..4096).map(|i| 0.6 * (i as f32 * 0.05).sin()).collect();
    for slot in ch.iter_mut().skip(1000).take(50) {
        *slot = 1.4;
    }
    let signal: Buffers = vec![ch.clone(), ch];

    let wrms333 =
        || WindowedRmsMeter::with_settings(WindowedRmsMeterSettings::new().window_frames(333));
    let win = measure(wrms333(), &signal, 4096);
    let clip = measure_u64(ClipMeter::new(), &signal, 4096);
    for &block in &[1usize, 7, 32, 64, 100, 1000] {
        assert_eq!(
            measure(wrms333(), &signal, block).to_bits(),
            win.to_bits(),
            "windowed RMS diverged at block {block}"
        );
        assert_eq!(
            measure_u64(ClipMeter::new(), &signal, block),
            clip,
            "clip count diverged at block {block}"
        );
    }
}

#[test]
fn windowed_and_clip_reset_to_silence() {
    let signal = tone(2000, 0.9);
    let clipped = dc(100, 1.5);

    let mut win =
        WindowedRmsMeter::with_settings(WindowedRmsMeterSettings::new().window_frames(256));
    Measurer::<f32>::prepare(&mut win, spec()).unwrap();
    observe_blocks(&mut win, &signal, 64);
    assert!(
        Measurer::<f32>::read(&win) > 0.0,
        "windowed RMS is non-zero"
    );
    Measurer::<f32>::reset(&mut win);
    assert_eq!(Measurer::<f32>::read(&win), 0.0, "windowed RMS resets to 0");

    let mut clip = ClipMeter::new();
    Measurer::<f32>::prepare(&mut clip, spec()).unwrap();
    observe_blocks(&mut clip, &clipped, 16);
    assert!(Measurer::<f32>::read(&clip) > 0, "clip count is non-zero");
    Measurer::<f32>::reset(&mut clip);
    assert_eq!(Measurer::<f32>::read(&clip), 0, "clip count resets to 0");
}
