// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Analysis and synthesis windows, with a constant-overlap-add check.
//!
//! Window coefficients use [`math`]. All windows are periodic forms using
//! `2*pi*n/N`.

use crate::dsp::math;
use std::f64::consts::TAU;

/// An analysis or synthesis window shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Window {
    /// Rectangular window with no taper.
    Rectangular,
    /// Periodic Hann window.
    Hann,
    /// Periodic Hamming.
    Hamming,
    /// Periodic Blackman.
    Blackman,
    /// Sine window, `sin(pi*n/N)`.
    Sine,
}

impl Window {
    /// Fill `out` with the length-`out.len()` periodic window coefficients.
    pub fn fill(self, out: &mut [f64]) {
        let len = out.len();
        if len == 0 {
            return;
        }
        let n = len as f64;
        for (i, w) in out.iter_mut().enumerate() {
            let x = TAU * i as f64 / n;
            *w = match self {
                Window::Rectangular => 1.0,
                Window::Hann => 0.5 - 0.5 * math::cos(x),
                Window::Hamming => 0.54 - 0.46 * math::cos(x),
                Window::Blackman => 0.42 - 0.5 * math::cos(x) + 0.08 * math::cos(2.0 * x),
                // x == 2*pi*i/N, so 0.5*x == pi*i/N.
                Window::Sine => math::sin(0.5 * x),
            };
        }
    }

    /// Allocate and fill a length-`len` window vector.
    #[must_use]
    pub fn make(self, len: usize) -> Vec<f64> {
        let mut v = vec![0.0; len];
        self.fill(&mut v);
        v
    }
}

/// The overlap-add gain of `window` shifted by `hop`, sampled across one hop.
///
/// For a COLA window and hop, every entry is equal. `hop` must be at least 1.
#[must_use]
pub fn cola_sum(window: &[f64], hop: usize) -> Vec<f64> {
    assert!(hop >= 1, "hop must be >= 1");
    let n = window.len();
    let mut sum = vec![0.0; hop];
    let mut base = 0;
    while base < n {
        for (i, acc) in sum.iter_mut().enumerate() {
            let idx = base + i;
            if idx < n {
                *acc += window[idx];
            }
        }
        base += hop;
    }
    sum
}
