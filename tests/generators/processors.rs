// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for generators.
//!
//! Covers sine oscillator amplitude and frequency, white-noise bounds and
//! reproducibility, PolyBLEP spectra, block-size invariance, and reset behavior.

use std::f64::consts::{PI, SQRT_2};

use bisque::generators::{
    PolyBlepOsc, PolyBlepOscSettings, SineOsc, SineOscSettings, Waveform, WhiteNoise,
    WhiteNoiseSettings,
};
use bisque::processor::KernelProcessor;
use bisque::processor::{IoMode, Processor};
use bisque::testing::{bits_eq, ev, Contract};

const FREQ: bisque::parameter::ParamId = SineOsc::FREQUENCY_HZ;

/// A driven sine oscillator (the harness drives it as an f32 `Processor`).
fn osc(freq_hz: f64, amp: f64) -> KernelProcessor<SineOsc> {
    KernelProcessor::new(SineOsc::with_settings(
        SineOscSettings::new().frequency_hz(freq_hz).amplitude(amp),
    ))
}

/// A driven white-noise source.
fn noise(amp: f64, seed: u64) -> KernelProcessor<WhiteNoise> {
    KernelProcessor::new(WhiteNoise::with_settings(
        WhiteNoiseSettings::new().amplitude(amp).seed(seed),
    ))
}

/// A driven PolyBLEP sawtooth / square.
fn saw(freq: f64, amp: f64) -> KernelProcessor<PolyBlepOsc> {
    KernelProcessor::new(PolyBlepOsc::with_settings(
        PolyBlepOscSettings::new()
            .waveform(Waveform::Saw)
            .frequency_hz(freq)
            .amplitude(amp),
    ))
}
fn square(freq: f64, amp: f64) -> KernelProcessor<PolyBlepOsc> {
    KernelProcessor::new(PolyBlepOsc::with_settings(
        PolyBlepOscSettings::new()
            .waveform(Waveform::Square)
            .frequency_hz(freq)
            .amplitude(amp),
    ))
}

/// The amplitude of the component at `freq` Hz via a one-bin DTFT.
fn component_amp(signal: &[f32], freq: f64, fs: f64) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, &s) in signal.iter().enumerate() {
        let ang = 2.0 * PI * freq * n as f64 / fs;
        re += f64::from(s) * ang.cos();
        im -= f64::from(s) * ang.sin();
    }
    2.0 * (re * re + im * im).sqrt() / signal.len() as f64
}

/// Pearson correlation between equal-length signals.
fn correlation(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&sample| f64::from(sample)).sum::<f64>() / n;
    let mean_b = b.iter().map(|&sample| f64::from(sample)).sum::<f64>() / n;
    let (mut covariance, mut energy_a, mut energy_b) = (0.0, 0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        let da = f64::from(a) - mean_a;
        let db = f64::from(b) - mean_b;
        covariance += da * db;
        energy_a += da * da;
        energy_b += db * db;
    }
    covariance / (energy_a * energy_b).sqrt()
}

mod audio {
    use super::*;

    #[test]
    fn frequency_clamps_below_nyquist_without_silencing_the_oscillator() {
        // The declared range caps at 24 kHz, above the guarded ceiling for a
        // 16 kHz session. Both oscillators clamp to 0.999 * Nyquist.
        let contract = Contract {
            spec: bisque::processor::ProcessSpec {
                sample_rate: 16_000,
                channels: 2,
                max_block: 8192,
                max_memory: None,
            },
            ..Contract::default()
        };
        let guarded = 0.999 * 8_000.0;
        let above = contract.generate(|| osc(24_000.0, 0.5), 4096, &[], 256);
        let limited = contract.generate(|| osc(guarded, 0.5), 4096, &[], 256);
        assert!(
            bits_eq(&above, &limited),
            "a high sine request clamps to the guarded ceiling"
        );
        let sine_rms = (above[0]
            .iter()
            .map(|&sample| f64::from(sample).powi(2))
            .sum::<f64>()
            / above[0].len() as f64)
            .sqrt();
        assert!(sine_rms > 0.1, "the guarded sine must remain audible");

        let above = contract.generate(|| saw(24_000.0, 0.5), 4096, &[], 256);
        let limited = contract.generate(|| saw(guarded, 0.5), 4096, &[], 256);
        assert!(
            bits_eq(&above, &limited),
            "a high PolyBLEP request clamps to the guarded ceiling"
        );
        assert!(
            above[0].iter().any(|sample| sample.abs() > 0.01),
            "the guarded PolyBLEP oscillator must remain audible"
        );
    }

    #[test]
    fn generates_the_requested_tone() {
        let (freq, amp) = (440.0, 0.5);
        let n = 48_000; // 1 second at the default 48 kHz
        let out = Contract::default().generate(|| osc(freq, amp), n, &[], 64);

        // The same tone in every channel.
        assert_eq!(out[0], out[1], "a mono source fills every channel alike");

        // Peak is near A and RMS is near A/sqrt(2).
        let peak = out[0]
            .iter()
            .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        assert!(
            (peak - amp).abs() < 1e-3,
            "peak {peak} should be the amplitude {amp}"
        );
        let rms = (out[0]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(
            (rms - amp / SQRT_2).abs() < 1e-2,
            "rms {rms} should be A/sqrt(2) = {}",
            amp / SQRT_2
        );

        // Frequency: count zero crossings (2 per cycle).
        let crossings = out[0]
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        let measured = crossings as f64 / 2.0 * 48_000.0 / n as f64;
        assert!(
            (measured - freq).abs() < 1.0,
            "measured frequency {measured} should be {freq} Hz"
        );
    }
}

#[test]
fn block_size_invariance_is_bit_exact() {
    Contract::default().assert_generator_block_size_invariant(|| osc(440.0, 0.5), 1000, &[]);
}

#[test]
fn block_size_invariance_under_frequency_sweep() {
    // Frequency events mid-stream stay split-invariant.
    let events = [ev(0, FREQ, 220.0), ev(400, FREQ, 880.0)];
    Contract::default().assert_generator_block_size_invariant(|| osc(440.0, 0.5), 1000, &events);
}

#[test]
fn reset_equivalence_no_state_leak() {
    let make = || osc(440.0, 0.5);
    let c = Contract::default();
    let fresh = c.generate(make, 1000, &[], 64);

    let mut gen = make();
    Processor::<f32>::prepare(&mut gen, c.spec).expect("prepare");
    // Dirty the phase and the smoother bank with a different-frequency pass.
    let _ = c.generate_reusing(&mut gen, 1000, &[ev(0, FREQ, 880.0)], 50);
    Processor::<f32>::reset(&mut gen);
    let after = c.generate_reusing(&mut gen, 1000, &[], 64);

    assert!(
        bits_eq(&after, &fresh),
        "reset must reproduce a fresh generator"
    );
}

// --- WhiteNoise ---

mod noise_audio {
    use super::*;

    const SEED: u64 = 0xABCD_1234_5678_9EF0;

    #[test]
    fn amplitude_is_bounded() {
        let amp = 0.5;
        let out = Contract::default().generate(|| noise(amp, SEED), 4000, &[], 64);
        for plane in &out {
            for &x in plane {
                assert!(x.abs() < amp as f32 + 1e-6, "|{x}| should be < amp {amp}");
            }
        }
    }

    #[test]
    fn is_unbiased() {
        // The long-run mean of uniform noise is ~0.
        let n = 200_000;
        let out = Contract::default().generate(|| noise(0.8, SEED), n, &[], 256);
        for (ch, plane) in out.iter().enumerate() {
            let mean = plane.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
            assert!(mean.abs() < 0.01, "ch{ch} mean {mean} should be ~0");
        }
    }
}

#[test]
fn noise_same_seed_is_bit_exact() {
    let c = Contract::default();
    let a = c.generate(|| noise(0.5, 42), 1000, &[], 64);
    let b = c.generate(|| noise(0.5, 42), 1000, &[], 64);
    assert!(bits_eq(&a, &b), "same seed must give bit-identical noise");
}

#[test]
fn noise_different_seed_differs() {
    let c = Contract::default();
    let a = c.generate(|| noise(0.5, 1), 1000, &[], 64);
    let b = c.generate(|| noise(0.5, 2), 1000, &[], 64);
    assert!(!bits_eq(&a, &b), "a different seed must change the noise");
}

#[test]
fn noise_channels_are_decorrelated() {
    let out = Contract::default().generate(|| noise(0.5, 7), 100_000, &[], 256);
    let measured = correlation(&out[0], &out[1]);
    assert!(
        measured.abs() < 0.02,
        "separately seeded channels must have low correlation, got {measured}"
    );
    for (channel, samples) in out.iter().enumerate() {
        let lag_one = correlation(&samples[..samples.len() - 1], &samples[1..]);
        assert!(
            lag_one.abs() < 0.02,
            "channel {channel} must have low lag-one correlation, got {lag_one}"
        );
    }
}

#[test]
fn generators_declare_output_only_io() {
    assert_eq!(
        Processor::<f32>::io_mode(&osc(440.0, 0.5)),
        IoMode::OutputOnly
    );
    assert_eq!(
        Processor::<f32>::io_mode(&noise(0.5, 7)),
        IoMode::OutputOnly
    );
    assert_eq!(
        Processor::<f32>::io_mode(&saw(440.0, 0.5)),
        IoMode::OutputOnly
    );
    assert_eq!(
        Processor::<f32>::io_mode(&square(440.0, 0.5)),
        IoMode::OutputOnly
    );
}

#[test]
fn noise_block_size_invariance_is_bit_exact() {
    Contract::default().assert_generator_block_size_invariant(|| noise(0.5, 99), 1000, &[]);
}

// --- PolyBlepOsc ---

mod polyblep_audio {
    use super::*;

    const FS: f64 = 48_000.0;

    #[test]
    fn saw_harmonics_match_the_ideal_spectrum() {
        // A band-limited sawtooth has harmonic k at amplitude 2/(pi*k). At
        // f0 = 1 kHz, low harmonics are well below Nyquist.
        let f0 = 1_000.0;
        let n = 24_000; // 48 whole periods; f0 -> bin 500
        let out = Contract::default().generate(|| saw(f0, 1.0), n, &[], 128);
        for k in 1..=3 {
            let meas = component_amp(&out[0], f0 * f64::from(k), FS);
            let ideal = 2.0 / (PI * f64::from(k));
            assert!(
                (meas - ideal).abs() < 0.03,
                "saw H{k}: {meas} vs ideal {ideal}"
            );
        }
    }

    #[test]
    fn square_has_only_odd_harmonics() {
        // A square wave has only odd harmonics, with harmonic k at 4/(pi*k).
        let f0 = 1_000.0;
        let n = 24_000;
        let out = Contract::default().generate(|| square(f0, 1.0), n, &[], 128);
        let h1 = component_amp(&out[0], f0, FS);
        let h2 = component_amp(&out[0], 2.0 * f0, FS);
        let h3 = component_amp(&out[0], 3.0 * f0, FS);
        assert!((h1 - 4.0 / PI).abs() < 0.03, "square H1 = 4/pi, got {h1}");
        assert!(h2 < 0.02, "square has no even harmonic, H2 = {h2}");
        assert!(
            (h3 - 4.0 / (3.0 * PI)).abs() < 0.03,
            "square H3 = 4/(3*pi), got {h3}"
        );
    }

    #[test]
    fn saw_suppresses_aliasing_versus_naive() {
        // At a high fundamental, the PolyBLEP saw has less aliased energy than a
        // point-sampled ramp driven with the same phase.
        let f0 = 2_333.0; // 11th harmonic (25_663 Hz) folds to 48_000-25_663 = 22_337
        let alias = 22_337.0;
        let n = 16_384;
        let pb = Contract::default().generate(|| saw(f0, 1.0), n, &[], 64);

        let dt = f0 / FS;
        let mut p = 0.0f64;
        let naive: Vec<f32> = (0..n)
            .map(|_| {
                let v = (2.0 * p - 1.0) as f32;
                p += dt;
                if p >= 1.0 {
                    p -= 1.0;
                }
                v
            })
            .collect();

        let pb_alias = component_amp(&pb[0], alias, FS);
        let naive_alias = component_amp(&naive, alias, FS);
        assert!(
            naive_alias > 0.01,
            "the naive saw genuinely aliases (else the test is vacuous): {naive_alias}"
        );
        assert!(
            pb_alias < naive_alias * 0.5,
            "PolyBLEP must suppress the alias: pb {pb_alias} vs naive {naive_alias}"
        );
    }

    #[test]
    fn square_base_level_follows_the_phase() {
        // The square base level is `+amp` for the first half-cycle and `-amp` for
        // the second. Check stable points away from BLEP-corrected discontinuities.
        let amp = 0.8;
        let out = Contract::default().generate(|| square(100.0, amp), 480, &[], 64);
        assert!(
            out[0][120] > 0.5,
            "first half-cycle is +amp, got {} at phase 0.25",
            out[0][120]
        );
        assert!(
            out[0][360] < -0.5,
            "second half-cycle is -amp, got {} at phase 0.75",
            out[0][360]
        );
    }

    #[test]
    fn square_level_is_strict_at_the_half_cycle_boundary() {
        // At 12 kHz the phase increment is exactly 1/4 cycle, so frame 2 lands on
        // phase 0.5.
        let out = Contract::default().generate(|| square(12_000.0, 0.8), 8, &[], 8);
        assert!(
            out[0][2].abs() < 0.1,
            "phase exactly 0.5 takes the lower (strict-<) level, got {}",
            out[0][2]
        );
    }

    #[test]
    fn saw_and_square_have_no_dc() {
        // Over whole periods the mean is ~0 for both shapes.
        let n = 24_000;
        let saw_out = Contract::default().generate(|| saw(1_000.0, 0.8), n, &[], 64);
        let sq_out = Contract::default().generate(|| square(1_000.0, 0.8), n, &[], 64);
        let dc = |s: &[f32]| s.iter().map(|&x| f64::from(x)).sum::<f64>() / s.len() as f64;
        assert!(dc(&saw_out[0]).abs() < 1e-3, "saw DC {}", dc(&saw_out[0]));
        assert!(dc(&sq_out[0]).abs() < 1e-3, "square DC {}", dc(&sq_out[0]));
    }
}

#[test]
fn polyblep_block_size_invariance_is_bit_exact() {
    Contract::default().assert_generator_block_size_invariant(|| saw(440.0, 0.5), 1000, &[]);
    Contract::default().assert_generator_block_size_invariant(|| square(440.0, 0.5), 1000, &[]);
}

#[test]
fn polyblep_reset_equivalence_no_state_leak() {
    let c = Contract::default();
    let make = || saw(440.0, 0.5);
    let fresh = c.generate(make, 1000, &[], 64);

    let mut g = make();
    Processor::<f32>::prepare(&mut g, c.spec).expect("prepare");
    let _ = c.generate_reusing(&mut g, 1000, &[ev(0, FREQ, 880.0)], 50);
    Processor::<f32>::reset(&mut g);
    let after = c.generate_reusing(&mut g, 1000, &[], 64);

    assert!(
        bits_eq(&after, &fresh),
        "reset must reproduce a fresh oscillator"
    );
}
