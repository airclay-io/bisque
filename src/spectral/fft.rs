// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Real-input FFT wrapper over `realfft`.
//!
//! A single `Fft` owns forward and inverse plans plus scratch buffers. `forward`
//! and `inverse` do not allocate after construction.

use std::sync::Arc;

use crate::dsp::sanitize::finite_or_zero;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Complex bin type produced by [`Fft::forward`], re-exported so callers do
/// not depend on `realfft` directly.
pub use realfft::num_complex::Complex;

/// A real-input FFT of fixed size `n`.
///
/// Forward maps `n` real samples to `n/2 + 1` complex bins. Inverse maps them
/// back and normalizes by `1/n`.
pub struct Fft {
    n: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    fwd_scratch: Vec<Complex<f64>>,
    inv_scratch: Vec<Complex<f64>>,
}

impl Fft {
    /// Plan a real FFT of size `n`.
    ///
    /// # Panics
    /// If `n` is zero. Larger sizes are delegated to `realfft`, which plans
    /// any positive length.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut fft = Self::plan(n);
        fft.allocate_scratch();
        fft
    }

    /// Build the opaque plans without allocating the caller-owned scratch.
    pub(crate) fn plan(n: usize) -> Self {
        assert!(n >= 1, "FFT size must be at least 1");
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(n);
        let c2r = planner.plan_fft_inverse(n);
        Self {
            n,
            r2c,
            c2r,
            fwd_scratch: Vec::new(),
            inv_scratch: Vec::new(),
        }
    }

    pub(crate) fn scratch_lengths(&self) -> (usize, usize) {
        (self.r2c.get_scratch_len(), self.c2r.get_scratch_len())
    }

    pub(crate) fn allocate_scratch(&mut self) {
        self.fwd_scratch = self.r2c.make_scratch_vec();
        self.inv_scratch = self.c2r.make_scratch_vec();
    }

    pub(crate) fn scratch_footprint(&self) -> usize {
        (self.fwd_scratch.len() + self.inv_scratch.len()) * std::mem::size_of::<Complex<f64>>()
    }

    /// The transform size (real samples per frame).
    #[must_use]
    pub fn size(&self) -> usize {
        self.n
    }

    /// The number of complex bins a forward transform produces (`n/2 + 1`).
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.n / 2 + 1
    }

    /// Transform `input` of length `n` into `spectrum` of length `n/2 + 1`.
    /// `input` is overwritten.
    ///
    /// # Panics
    /// If the slice lengths do not match the planned size.
    pub fn forward(&mut self, input: &mut [f64], spectrum: &mut [Complex<f64>]) {
        for sample in input.iter_mut() {
            *sample = finite_or_zero(*sample);
        }
        self.r2c
            .process_with_scratch(input, spectrum, &mut self.fwd_scratch)
            .expect("realfft forward: slice lengths match the plan");
    }

    /// Transform `spectrum` of length `n/2 + 1` into `output` of length `n`.
    /// `spectrum` is overwritten and output is normalized by `1/n`.
    ///
    /// # Panics
    /// If the slice lengths do not match the planned size.
    pub fn inverse(&mut self, spectrum: &mut [Complex<f64>], output: &mut [f64]) {
        for bin in spectrum.iter_mut() {
            bin.re = finite_or_zero(bin.re);
            bin.im = finite_or_zero(bin.im);
        }
        self.c2r
            .process_with_scratch(spectrum, output, &mut self.inv_scratch)
            .expect("realfft inverse: slice lengths match the plan");
        let scale = 1.0 / self.n as f64;
        for s in output.iter_mut() {
            *s = finite_or_zero(*s * scale);
        }
    }
}

impl std::fmt::Debug for Fft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Plans and scratch are opaque. Only the size is shown.
        f.debug_struct("Fft")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::Fft;

    #[test]
    fn scratch_footprint_counts_owned_scratch_vectors() {
        // Odd real transforms require staging scratch in realfft. SpectralFilter
        // currently accepts even sizes only, but Fft is a public standalone
        // utility and the accounting helper must cover either plan shape.
        let fft = Fft::new(1013);
        let (forward, inverse) = fft.scratch_lengths();
        assert!(forward > 0 && inverse > 0);
        assert_eq!(
            fft.scratch_footprint(),
            (forward + inverse) * std::mem::size_of::<super::Complex<f64>>()
        );
    }
}
