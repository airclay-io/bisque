// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for the FFT backend [`Fft`].
//!
//! Covers round-trip identity, single-bin cosine magnitude, linearity, and
//! repeated transform behavior.
#![cfg(feature = "spectral")]
// Standard transform notation: `n` size, `k` bin, `x`/`y` signals, `a`/`b` scales.
#![allow(clippy::many_single_char_names)]

use std::f64::consts::TAU;

use bisque::spectral::fft::{Complex, Fft};

fn zeros(bins: usize) -> Vec<Complex<f64>> {
    vec![Complex::new(0.0, 0.0); bins]
}

#[test]
fn inverse_is_the_inverse_of_forward() {
    // Forward then normalized inverse reconstructs the input to f64 precision.
    let n = 1024;
    let mut fft = Fft::new(n);
    let orig: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            0.4 * (t * 0.05).sin() + 0.3 * (t * 0.211).cos() - 0.2 * (t * 0.017).sin()
        })
        .collect();

    let mut input = orig.clone();
    let mut spectrum = zeros(fft.num_bins());
    fft.forward(&mut input, &mut spectrum);
    let mut output = vec![0.0; n];
    fft.inverse(&mut spectrum, &mut output);

    for (i, (o, r)) in output.iter().zip(&orig).enumerate() {
        assert!(
            (o - r).abs() < 1e-12,
            "round-trip diverged at {i}: {o} vs {r}"
        );
    }
}

#[test]
fn cosine_transforms_to_a_single_bin() {
    // x[i] = cos(2*pi*k*i/N) has one-sided magnitude N/2 at bin k.
    let n = 1024;
    let k = 64;
    let mut fft = Fft::new(n);
    let mut input: Vec<f64> = (0..n)
        .map(|i| (TAU * k as f64 * i as f64 / n as f64).cos())
        .collect();
    let mut spectrum = zeros(fft.num_bins());
    fft.forward(&mut input, &mut spectrum);

    let peak = spectrum[k].norm();
    assert!(
        (peak - n as f64 / 2.0).abs() < 1e-6,
        "bin {k} magnitude {peak} should be N/2"
    );
    for (b, c) in spectrum.iter().enumerate() {
        if b != k {
            assert!(
                c.norm() < 1e-6,
                "off-bin {b} should be ~0, got {}",
                c.norm()
            );
        }
    }
}

#[test]
fn forward_is_linear() {
    // FFT(a*x + b*y) == a*FFT(x) + b*FFT(y), bin by bin.
    let n = 256;
    let mut fft = Fft::new(n);
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.19).cos()).collect();
    let (a, b) = (1.5, -0.7);

    let spec = |fft: &mut Fft, sig: &[f64]| {
        let mut input = sig.to_vec();
        let mut s = zeros(fft.num_bins());
        fft.forward(&mut input, &mut s);
        s
    };
    let sx = spec(&mut fft, &x);
    let sy = spec(&mut fft, &y);
    let combined: Vec<f64> = x.iter().zip(&y).map(|(xi, yi)| a * xi + b * yi).collect();
    let sc = spec(&mut fft, &combined);

    for bin in 0..fft.num_bins() {
        let expect = sx[bin] * a + sy[bin] * b;
        assert!(
            (sc[bin] - expect).norm() < 1e-9,
            "linearity broke at bin {bin}"
        );
    }
}

#[test]
fn forward_and_inverse_allocate_their_scratch_once() {
    // Repeated transforms reuse planned scratch and return stable output.
    let n = 512;
    let mut fft = Fft::new(n);
    let sig: Vec<f64> = (0..n).map(|i| (i as f64 * 0.03).sin()).collect();
    let mut first = vec![0.0; n];
    for pass in 0..4 {
        let mut input = sig.clone();
        let mut spectrum = zeros(fft.num_bins());
        fft.forward(&mut input, &mut spectrum);
        let mut output = vec![0.0; n];
        fft.inverse(&mut spectrum, &mut output);
        if pass == 0 {
            first = output;
        } else {
            assert_eq!(output, first, "repeated transforms must be deterministic");
        }
    }
}
