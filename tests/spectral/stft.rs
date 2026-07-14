// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for the offline [`Stft`].
//!
//! Covers analysis/synthesis reconstruction, tone-bin location, frame counting,
//! and low-pass behavior by zeroing high bins.
#![cfg(feature = "spectral")]
// Standard transform notation: n size, hop, x/y signals, i/m indices, t time.
#![allow(clippy::many_single_char_names)]

use std::f64::consts::TAU;

use bisque::spectral::fft::Complex;
use bisque::spectral::stft::Stft;
use bisque::spectral::window::Window;

const FS: f64 = 48_000.0;

/// Component amplitude at `freq` via a one-bin DTFT.
fn amp_at(signal: &[f64], freq: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &s) in signal.iter().enumerate() {
        let ang = TAU * freq * n as f64 / FS;
        re += s * ang.cos();
        im -= s * ang.sin();
    }
    2.0 * (re * re + im * im).sqrt() / signal.len() as f64
}

#[test]
fn num_frames_counts_hops() {
    let stft = Stft::new(1024, 256, Window::Hann);
    assert_eq!(stft.num_frames(0), 0);
    assert_eq!(stft.num_frames(1), 1);
    assert_eq!(stft.num_frames(256), 1);
    assert_eq!(stft.num_frames(257), 2);
    assert_eq!(stft.num_frames(1000), 4);
}

#[test]
fn analysis_then_synthesis_reconstructs_the_interior() {
    // Analysis followed by synthesis reconstructs the interior region.
    let n = 1024;
    let mut stft = Stft::new(n, n / 2, Window::Hann);
    let len = 16_000;
    let x: Vec<f64> = (0..len)
        .map(|i| {
            let t = i as f64;
            0.5 * (t * 0.05).sin() + 0.3 * (t * 0.213).sin() - 0.2 * (t * 0.011).cos()
        })
        .collect();

    let frames = stft.analyze(&x);
    let y = stft.synthesize(&frames);

    for i in n..len - n {
        assert!(
            (y[i] - x[i]).abs() < 1e-6,
            "round-trip diverged at {i}: {} vs {}",
            y[i],
            x[i]
        );
    }
}

#[test]
fn analysis_locates_a_tone_in_the_right_bin() {
    // A 3 kHz tone at 48 kHz / 1024 lands exactly on bin 64.
    let n = 1024;
    let freq = 3_000.0;
    let mut stft = Stft::new(n, n / 2, Window::Hann);
    let x: Vec<f64> = (0..8192)
        .map(|i| (TAU * freq * i as f64 / FS).sin())
        .collect();

    let frames = stft.analyze(&x);
    let mid = &frames[frames.len() / 2];
    let peak_bin = mid
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
        .unwrap()
        .0;
    let expected = (freq * n as f64 / FS).round() as usize;
    assert_eq!(
        peak_bin, expected,
        "3 kHz tone should peak at bin {expected}"
    );
}

#[test]
fn zeroing_high_bins_is_a_low_pass() {
    // Zeroing bins above about 4.7 kHz removes 8 kHz and keeps 1 kHz.
    let n = 1024;
    let mut stft = Stft::new(n, n / 2, Window::Hann);
    let len = 16_000;
    let (low_f, high_f) = (1_000.0, 8_000.0);
    let x: Vec<f64> = (0..len)
        .map(|i| {
            let t = i as f64;
            0.5 * (TAU * low_f * t / FS).sin() + 0.5 * (TAU * high_f * t / FS).sin()
        })
        .collect();

    let mut frames = stft.analyze(&x);
    let cutoff_bin = 100; // ~4.7 kHz
    for fr in &mut frames {
        for c in &mut fr[cutoff_bin..] {
            *c = Complex::new(0.0, 0.0);
        }
    }
    let y = stft.synthesize(&frames);

    let interior = &y[n..len - n];
    let high = amp_at(interior, high_f);
    let low = amp_at(interior, low_f);
    assert!(high < 0.05, "8 kHz must be removed, got {high}");
    assert!(low > 0.4, "1 kHz must survive, got {low}");
}

#[test]
#[should_panic(expected = "STFT hop must be in 1..=size")]
fn zero_hop_is_rejected() {
    let _ = Stft::new(1024, 0, Window::Hann);
}

#[test]
#[should_panic(expected = "STFT hop must be in 1..=size")]
fn oversized_hop_is_rejected() {
    let _ = Stft::new(1024, 1025, Window::Hann);
}

#[test]
#[should_panic(expected = "STFT frame bin count must match the transform size")]
fn synthesis_rejects_the_wrong_bin_count() {
    let mut stft = Stft::new(64, 32, Window::Hann);
    let malformed = vec![vec![Complex::new(0.0, 0.0); stft.num_bins() - 1]];
    let _ = stft.synthesize(&malformed);
}
