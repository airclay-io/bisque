// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Offline Short-Time Fourier Transform analysis and synthesis.
//!
//! `analyze` converts a signal into per-hop spectra. `synthesize` inverse
//! transforms and overlap-adds spectra back to a signal. These methods allocate
//! per call and are intended for offline use.
//!
//! Synthesis divides by accumulated overlap from the shared analysis and
//! synthesis window. This reconstructs the interior when every overlap sum is
//! above the normalization floor.

use super::fft::{Complex, Fft};
use super::window::Window;
use crate::dsp::sanitize::finite_or_zero;

/// An offline STFT analyzer/synthesizer at a fixed size and hop.
#[derive(Debug)]
pub struct Stft {
    fft: Fft,
    analysis: Vec<f64>,
    synthesis: Vec<f64>,
    /// Per-sample reconstruction weight, `analysis * synthesis`.
    wprod: Vec<f64>,
    hop: usize,
}

impl Stft {
    /// An STFT of frame `size` (the FFT length) at `hop` samples, using `window`
    /// for both analysis and synthesis. `hop` must be in `1..=size`.
    ///
    /// # Panics
    /// Panics if `size` is zero or `hop` is outside `1..=size`.
    #[must_use]
    pub fn new(size: usize, hop: usize, window: Window) -> Self {
        assert!(size >= 1, "STFT size must be at least 1");
        assert!((1..=size).contains(&hop), "STFT hop must be in 1..=size");
        let analysis = window.make(size);
        let synthesis = window.make(size);
        let wprod: Vec<f64> = analysis
            .iter()
            .zip(&synthesis)
            .map(|(a, s)| a * s)
            .collect();
        Self {
            fft: Fft::new(size),
            analysis,
            synthesis,
            wprod,
            hop,
        }
    }

    /// The frame size (FFT length).
    #[must_use]
    pub fn size(&self) -> usize {
        self.analysis.len()
    }

    /// The hop between frames.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Complex bins per frame (`size/2 + 1`).
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.fft.num_bins()
    }

    /// How many frames [`analyze`](Self::analyze) yields for a `len`-sample signal.
    #[must_use]
    pub fn num_frames(&self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            (len - 1) / self.hop + 1
        }
    }

    /// Analyze `signal` into one spectrum per hop. Frames are zero-padded past
    /// the end.
    #[must_use]
    pub fn analyze(&mut self, signal: &[f64]) -> Vec<Vec<Complex<f64>>> {
        let n = self.size();
        let frames = self.num_frames(signal.len());
        let mut out = Vec::with_capacity(frames);
        let mut frame = vec![0.0; n];
        for m in 0..frames {
            let start = m * self.hop;
            for (i, slot) in frame.iter_mut().enumerate() {
                let idx = start + i;
                *slot = if idx < signal.len() {
                    finite_or_zero(signal[idx]) * self.analysis[i]
                } else {
                    0.0
                };
            }
            let mut spectrum = vec![Complex::new(0.0, 0.0); self.num_bins()];
            self.fft.forward(&mut frame, &mut spectrum);
            out.push(spectrum);
        }
        out
    }

    /// Synthesize a signal from per-hop frames by inverse-transforming, applying
    /// the synthesis window, overlap-adding, and normalizing by window overlap.
    ///
    /// # Panics
    ///
    /// Panics if any frame does not contain exactly [`Self::num_bins`] bins.
    #[must_use]
    pub fn synthesize(&mut self, frames: &[Vec<Complex<f64>>]) -> Vec<f64> {
        let n = self.size();
        if frames.is_empty() {
            return Vec::new();
        }
        let total = (frames.len() - 1) * self.hop + n;
        let mut out = vec![0.0; total];
        let mut wsum = vec![0.0; total];
        let mut spectrum = vec![Complex::new(0.0, 0.0); self.num_bins()];
        let mut frame = vec![0.0; n];
        for (m, fr) in frames.iter().enumerate() {
            assert_eq!(
                fr.len(),
                self.num_bins(),
                "STFT frame bin count must match the transform size"
            );
            spectrum.copy_from_slice(fr); // inverse overwrites its input
            self.fft.inverse(&mut spectrum, &mut frame);
            let start = m * self.hop;
            for i in 0..n {
                out[start + i] += finite_or_zero(frame[i]) * self.synthesis[i];
                wsum[start + i] += self.wprod[i];
            }
        }
        for (o, w) in out.iter_mut().zip(&wsum) {
            if *w > 1e-9 {
                *o /= *w;
            }
            *o = finite_or_zero(*o);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::Stft;
    use crate::spectral::Window;

    #[test]
    #[should_panic(expected = "STFT size must be at least 1")]
    fn zero_size_panics_with_documented_precondition() {
        let _ = Stft::new(0, 1, Window::Hann);
    }
}
