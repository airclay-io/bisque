// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! PolyBLEP oscillators.

use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{DspError, Kernel, ProcessSpec, Sample, SubBlock};

use super::max_oscillator_frequency;

// ---------------------------------------------------------------------------
// PolyBLEP anti-aliased oscillators (saw / square)
// ---------------------------------------------------------------------------

/// The alias-reduced oscillator waveform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Waveform {
    /// Sawtooth with one downward step per cycle.
    Saw,
    /// 50% duty square with one rising and one falling step per cycle.
    Square,
}

/// Construction settings for [`PolyBlepOsc`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PolyBlepOscSettings {
    /// Oscillator waveform.
    pub waveform: Waveform,
    /// Frequency in Hz.
    pub frequency_hz: f64,
    /// Linear amplitude.
    pub amplitude: f64,
}

impl Default for PolyBlepOscSettings {
    fn default() -> Self {
        Self {
            waveform: Waveform::Saw,
            frequency_hz: 440.0,
            amplitude: 0.5,
        }
    }
}

impl PolyBlepOscSettings {
    /// Default PolyBLEP oscillator settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the oscillator waveform.
    #[must_use]
    pub fn waveform(mut self, waveform: Waveform) -> Self {
        self.waveform = waveform;
        self
    }

    /// Set the frequency in Hz.
    #[must_use]
    pub fn frequency_hz(mut self, frequency_hz: f64) -> Self {
        self.frequency_hz = frequency_hz;
        self
    }

    /// Set the linear amplitude.
    #[must_use]
    pub fn amplitude(mut self, amplitude: f64) -> Self {
        self.amplitude = amplitude;
        self
    }
}

crate::params! {
    /// Smoothed parameter values for [`PolyBlepOsc`].
    pub struct PolyBlepOscParams {
        /// Frequency in Hz.
        pub frequency_hz => FREQUENCY_HZ,
        /// Linear amplitude.
        pub amplitude => AMPLITUDE,
    }
}

/// The two-point PolyBLEP residual for a unit upward step.
///
/// `t` is normalized phase in `[0, 1)`. `dt` is phase increment per sample.
pub(super) fn poly_blep(t: f64, dt: f64) -> f64 {
    if t < dt {
        // Just after the wrap.
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        // Just before the wrap.
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// An alias-reduced sawtooth or square oscillator using PolyBLEP.
///
/// An output-only [`Kernel`]. Frequency and amplitude are automatable. Phase is
/// held in `f64` and accumulated one sample at a time. The effective frequency
/// is clamped to `0.999 * Nyquist` when the declared 24 kHz range exceeds the
/// prepared sample rate's audio band.
#[derive(Debug, Clone)]
pub struct PolyBlepOsc {
    kind: Waveform,
    params: [ParamInfo; 2],
    fs: f64,
    phase: f64, // normalized, [0, 1)
}

impl PolyBlepOsc {
    /// Frequency in Hz.
    pub const FREQUENCY_HZ: ParamId = PolyBlepOscParams::FREQUENCY_HZ;
    /// Linear amplitude.
    pub const AMPLITUDE: ParamId = PolyBlepOscParams::AMPLITUDE;

    /// An alias-reduced oscillator configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: PolyBlepOscSettings) -> Self {
        Self {
            kind: settings.waveform,
            params: [
                ParamInfo::new(
                    Self::FREQUENCY_HZ,
                    "frequency",
                    (1.0, 24_000.0),
                    settings.frequency_hz,
                    Unit::Hz,
                ),
                ParamInfo::new(
                    Self::AMPLITUDE,
                    "amplitude",
                    (0.0, 1.0),
                    settings.amplitude,
                    Unit::Linear,
                ),
            ],
            fs: 0.0,
            phase: 0.0,
        }
    }

    /// The default oscillator: a sawtooth at 440 Hz, amplitude 0.5.
    ///
    /// The default waveform is [`Waveform::Saw`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(PolyBlepOscSettings::default())
    }

    /// A sawtooth at 440 Hz, amplitude 0.5.
    #[must_use]
    pub fn saw() -> Self {
        Self::with_settings(PolyBlepOscSettings::default())
    }

    /// A square at 440 Hz, amplitude 0.5.
    #[must_use]
    pub fn square() -> Self {
        Self::with_settings(PolyBlepOscSettings::new().waveform(Waveform::Square))
    }
}

impl Default for PolyBlepOsc {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for PolyBlepOsc {
    type Params = PolyBlepOscParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        if spec.sample_rate == 0 {
            return Err(DspError::UnsupportedSpec("sample rate must be non-zero"));
        }
        self.fs = f64::from(spec.sample_rate);
        self.phase = 0.0;
        Ok(())
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn io_mode(&self) -> crate::processor::IoMode {
        crate::processor::IoMode::OutputOnly
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &PolyBlepOscParams) {
        // Normalized increment in cycles per sample for this sub-block.
        let frequency = params.frequency_hz.min(max_oscillator_frequency(self.fs));
        let dt = (frequency / self.fs).clamp(0.0, 0.5);
        let amp = params.amplitude;
        let kind = self.kind;
        let start = self.phase;
        let mut end = start;
        // Each channel carries the same tone and phase advances once per frame.
        for ch in 0..io.channels() {
            let mut p = start;
            for slot in io.output_mut(ch).iter_mut() {
                let v = match kind {
                    Waveform::Saw => (2.0 * p - 1.0) - poly_blep(p, dt),
                    Waveform::Square => {
                        let mut s = if p < 0.5 { 1.0 } else { -1.0 };
                        s += poly_blep(p, dt); // rising step at phase 0
                        let p2 = if p + 0.5 >= 1.0 { p - 0.5 } else { p + 0.5 };
                        s -= poly_blep(p2, dt); // falling step at phase 0.5
                        s
                    }
                };
                *slot = T::from_f64(amp * v);
                p += dt;
                if p >= 1.0 {
                    p -= 1.0;
                }
            }
            // Every channel starts at `start` and advances by the same `dt`
            // per frame, so each iteration lands on the same end phase.
            // Carrying the last channel's value is correct only because of
            // that per-channel identity.
            end = p;
        }
        self.phase = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_and_named_constructors_select_the_requested_waveform() {
        let settings = PolyBlepOscSettings::new()
            .waveform(Waveform::Square)
            .frequency_hz(220.0)
            .amplitude(0.25);
        assert_eq!(settings.waveform, Waveform::Square);
        assert_eq!(settings.frequency_hz, 220.0);
        assert_eq!(settings.amplitude, 0.25);
        assert_eq!(PolyBlepOsc::saw().kind, Waveform::Saw);
        assert_eq!(PolyBlepOsc::square().kind, Waveform::Square);
    }
}
