// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Contract tests for analysis/synthesis [`Window`]s and [`cola_sum`].
//!
//! Covers periodic endpoints and constant-overlap-add behavior.

use bisque::spectral::window::{cola_sum, Window};

fn spread(v: &[f64]) -> f64 {
    let (min, max) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(mn, mx), &x| (mn.min(x), mx.max(x)));
    max - min
}

#[test]
fn periodic_shapes_hit_their_endpoints() {
    // Periodic windows start at their analytical value and peak at the center.
    let rect = Window::Rectangular.make(8);
    assert!(
        rect.iter().all(|&v| (v - 1.0).abs() < 1e-12),
        "rectangular is all ones"
    );

    let hann = Window::Hann.make(8);
    assert!(hann[0].abs() < 1e-12, "periodic Hann starts at 0");
    assert!(
        (hann[4] - 1.0).abs() < 1e-12,
        "Hann peaks at 1 in the center"
    );

    let hamming = Window::Hamming.make(8);
    assert!((hamming[0] - 0.08).abs() < 1e-12, "Hamming starts at 0.08");

    let blackman = Window::Blackman.make(8);
    assert!(blackman[0].abs() < 1e-12, "periodic Blackman starts at 0");
}

#[test]
fn hann_is_cola_and_sums_to_one_at_fifty_percent() {
    // Hann shifted by 50% overlap sums to flat gain 1.
    let n = 1024;
    let w = Window::Hann.make(n);
    let sum = cola_sum(&w, n / 2);
    assert!(
        spread(&sum) < 1e-9,
        "Hann 50% must be flat, spread {}",
        spread(&sum)
    );
    assert!(
        (sum[0] - 1.0).abs() < 1e-9,
        "Hann 50% sums to 1, got {}",
        sum[0]
    );
}

#[test]
fn rectangular_overlap_is_flat_but_not_unity() {
    // Rectangular overlap is flat, with gain equal to the overlap count.
    let n = 512;
    let sum = cola_sum(&Window::Rectangular.make(n), n / 2);
    assert!(
        spread(&sum) < 1e-12,
        "rectangular overlap is perfectly flat"
    );
    assert!(
        (sum[0] - 2.0).abs() < 1e-12,
        "two unit windows overlap to gain 2"
    );
}

#[test]
fn sine_window_squared_is_unity_cola_at_fifty_percent() {
    // Sine squared equals Hann, so 50% overlap sums to gain 1.
    let n = 1024;
    let sine = Window::Sine.make(n);
    assert!(sine[0].abs() < 1e-12, "sine window starts at 0");
    assert!(
        (sine[n / 2] - 1.0).abs() < 1e-12,
        "sine peaks at 1 in the center"
    );
    let prod: Vec<f64> = sine.iter().map(|&s| s * s).collect();
    let sum = cola_sum(&prod, n / 2);
    assert!(
        spread(&sum) < 1e-12,
        "sine squared overlap is flat, spread {}",
        spread(&sum)
    );
    assert!(
        (sum[0] - 1.0).abs() < 1e-12,
        "sine squared sums to 1, got {}",
        sum[0]
    );
}

#[test]
fn no_overlap_ripples_for_a_tapered_window() {
    // With no overlap, the gain follows the tapered window shape.
    let n = 256;
    let sum = cola_sum(&Window::Hann.make(n), n);
    assert!(
        spread(&sum) > 0.5,
        "no-overlap Hann is not flat, spread {}",
        spread(&sum)
    );
}
