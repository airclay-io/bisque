// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Polyphase FIR oversampling for inter-sample peak detection.
//!
//! A signal's true peak can occur between samples and exceed every sample value.
//! [`PolyphaseUpsampler`] upsamples by an integer factor and reports the largest
//! magnitude among the interpolated phases.
//!
//! This is a detector, not a signal-path filter. It returns peak magnitude, not
//! a reconstructed stream. The FIR is causal, so finite streams must call
//! [`PolyphaseUpsampler::drain_peak`] after the final input sample to include
//! delayed detector output. Coefficients are computed once through [`math`].

use std::f64::consts::PI;

use super::math;
use crate::dsp::sanitize::finite_or_zero;

/// A single-channel integer-factor oversampler that reports inter-sample peaks.
/// Allocate one per channel. Coefficients are immutable after construction.
#[derive(Debug, Clone)]
pub struct PolyphaseUpsampler {
    taps: usize,           // taps per phase
    phases: Vec<Vec<f64>>, // [factor][taps]
    delay: Vec<f64>,       // last `taps` inputs, delay[0] = newest
}

impl PolyphaseUpsampler {
    /// Build an oversampler with `factor` phases and `taps_per_phase` taps per
    /// phase.
    ///
    /// # Panics
    /// If `factor` or `taps_per_phase` is zero, or if their product overflows
    /// `usize` (a wrapped prototype length would misindex the phase tables).
    #[must_use]
    pub fn new(factor: usize, taps_per_phase: usize) -> Self {
        assert!(
            factor >= 1 && taps_per_phase >= 1,
            "factor/taps must be >= 1"
        );
        let m = factor;
        let p = taps_per_phase;
        let len = m
            .checked_mul(p)
            .expect("factor * taps_per_phase overflows usize");
        let center = (len - 1) as f64 / 2.0;
        let denom = (len - 1) as f64;

        let mut proto = vec![0.0f64; len];
        let mut sum = 0.0f64;
        for (i, h) in proto.iter_mut().enumerate() {
            // Sinc lowpass at the input Nyquist in the upsampled grid.
            let t = (i as f64 - center) / m as f64;
            let sinc = if t.abs() < 1e-12 {
                1.0
            } else {
                math::sin(PI * t) / (PI * t)
            };
            // Hann window.
            let hann = if denom > 0.0 {
                0.5 - 0.5 * math::cos(2.0 * PI * i as f64 / denom)
            } else {
                1.0
            };
            *h = sinc * hann;
            sum += *h;
        }
        // Normalize to unity per-phase DC gain.
        let scale = m as f64 / sum;
        for h in &mut proto {
            *h *= scale;
        }

        // Phase `ph` uses proto[ph], proto[ph + M], ...
        let phases: Vec<Vec<f64>> = (0..m)
            .map(|ph| (0..p).map(|k| proto[ph + k * m]).collect())
            .collect();

        Self {
            taps: p,
            phases,
            delay: vec![0.0; p],
        }
    }

    /// The oversampling factor.
    #[must_use]
    pub fn factor(&self) -> usize {
        self.phases.len()
    }

    /// Detector group delay in input frames.
    #[must_use]
    pub fn latency(&self) -> usize {
        (self.factor() * self.taps - 1).div_ceil(2 * self.factor())
    }

    /// Zero-input frames needed to remove every retained input contribution.
    #[must_use]
    pub fn tail_frames(&self) -> usize {
        self.taps - 1
    }

    /// Clear the delay line.
    pub fn reset(&mut self) {
        self.delay.fill(0.0);
    }

    /// Push one input sample and return the largest absolute value among the
    /// `factor` interpolated outputs.
    pub fn peak_abs(&mut self, x: f64) -> f64 {
        let x = finite_or_zero(x);
        // Shift the delay line down by one and insert the newest sample.
        self.delay.copy_within(0..self.taps - 1, 1);
        self.delay[0] = x;
        let mut peak = 0.0f64;
        for phase in &self.phases {
            let mut acc = 0.0f64;
            for (c, d) in phase.iter().zip(&self.delay) {
                acc += c * d; // separate multiply and add
            }
            let a = acc.abs();
            if a > peak {
                peak = a;
            }
        }
        peak
    }

    /// Push enough zero-input frames to complete a finite stream and return
    /// the largest absolute value produced while draining.
    ///
    /// No retained input can affect later output after this call. Calling it
    /// again without new input returns zero.
    pub fn drain_peak(&mut self) -> f64 {
        let mut peak = 0.0f64;
        for _ in 0..self.tail_frames() {
            peak = peak.max(self.peak_abs(0.0));
        }
        peak
    }

    /// State bytes for coefficients and delay.
    #[must_use]
    pub fn footprint(&self) -> usize {
        let f = std::mem::size_of::<f64>();
        (self.phases.iter().map(Vec::len).sum::<usize>() + self.delay.len()) * f
    }
}

#[cfg(test)]
mod tests {
    use super::PolyphaseUpsampler;

    #[test]
    fn dc_passes_at_unity() {
        let mut os = PolyphaseUpsampler::new(4, 12);
        let mut out = 0.0;
        for _ in 0..64 {
            out = os.peak_abs(0.5); // warm the delay line
        }
        assert!((out - 0.5).abs() < 1e-3, "DC gain ~1, got {out}");
    }

    #[test]
    fn recovers_inter_sample_peak() {
        // An f_s/4 sine sampled at the +pi/4 phase has samples at +/-1/sqrt(2)
        // and a continuous peak of 1.0.
        let v = std::f64::consts::FRAC_1_SQRT_2;
        let pattern = [v, v, -v, -v];
        let mut os = PolyphaseUpsampler::new(4, 12);
        let mut peak = 0.0f64;
        for n in 0..512 {
            peak = peak.max(os.peak_abs(pattern[n % 4]));
        }
        assert!(
            peak > 0.97 && peak < 1.05,
            "true peak ~1.0 recovered, got {peak} (sample peak is only 0.707)"
        );
    }

    #[test]
    fn reports_factor() {
        assert_eq!(PolyphaseUpsampler::new(4, 12).factor(), 4);
        assert_eq!(PolyphaseUpsampler::new(2, 8).factor(), 2);
    }

    #[test]
    fn reports_latency_and_tail_length() {
        let os = PolyphaseUpsampler::new(4, 12);
        assert_eq!(os.latency(), 6);
        assert_eq!(os.tail_frames(), 11);
    }

    #[test]
    fn drain_completes_a_finite_stream() {
        let mut os = PolyphaseUpsampler::new(4, 12);
        let immediate = os.peak_abs(1.0);
        let drained = os.drain_peak();
        assert!(drained > immediate, "the delayed impulse peak is drained");
        assert_eq!(os.drain_peak(), 0.0);
        assert_eq!(os.peak_abs(0.0), 0.0);
    }

    #[test]
    fn non_finite_input_is_silence() {
        let mut bad = PolyphaseUpsampler::new(4, 12);
        let mut sanitized = PolyphaseUpsampler::new(4, 12);
        assert_eq!(bad.peak_abs(f64::NAN), sanitized.peak_abs(0.0));
        assert_eq!(bad.peak_abs(f64::INFINITY), sanitized.peak_abs(0.0));
        assert_eq!(bad.peak_abs(0.5), sanitized.peak_abs(0.5));
    }

    #[test]
    fn reports_footprint() {
        // Phase coefficients plus the delay line.
        let os = PolyphaseUpsampler::new(4, 12);
        let expected = (4 * 12 + 12) * std::mem::size_of::<f64>();
        assert_eq!(os.footprint(), expected);
    }

    #[test]
    fn reset_clears_the_delay_line() {
        let mut os = PolyphaseUpsampler::new(4, 12);
        for _ in 0..32 {
            os.peak_abs(1.0); // fill the delay line
        }
        os.reset();
        // A zeroed delay line and zero input produce no retained peak.
        assert!(os.peak_abs(0.0) < 1e-9, "reset must clear the delay line");
    }
}
