// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Sine oscillator.

use std::f64::consts::TAU;

use crate::dsp::math;
use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{DspError, Kernel, ProcessSpec, Sample, SubBlock};

use super::max_oscillator_frequency;

/// Construction settings for [`SineOsc`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SineOscSettings {
    /// Frequency in Hz.
    pub frequency_hz: f64,
    /// Linear amplitude.
    pub amplitude: f64,
}

impl Default for SineOscSettings {
    fn default() -> Self {
        Self {
            frequency_hz: 440.0,
            amplitude: 0.5,
        }
    }
}

impl SineOscSettings {
    /// Default sine oscillator settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
    /// Smoothed parameter values for [`SineOsc`].
    pub struct SineOscParams {
        /// Frequency in Hz.
        pub frequency_hz => FREQUENCY_HZ,
        /// Linear amplitude.
        pub amplitude => AMPLITUDE,
    }
}

/// Fold a radian phase back into `[0, TAU)` with a single subtract.
pub(super) fn wrap_phase(ph: f64) -> f64 {
    if ph >= TAU {
        ph - TAU
    } else {
        ph
    }
}

/// A sine oscillator. It emits `amp * sin(2*pi*f*t)` in every output channel.
///
/// An output-only [`Kernel`]. Frequency and amplitude are automatable. A pure
/// sine has no harmonics, so it is alias-free below Nyquist. The effective
/// frequency is clamped to `0.999 * Nyquist` when the declared 24 kHz range
/// exceeds the prepared sample rate's audio band. Phase is held in `f64`.
#[derive(Debug, Clone)]
pub struct SineOsc {
    params: [ParamInfo; 2],
    fs: f64,
    phase: f64,
}

impl SineOsc {
    /// Frequency in Hz.
    pub const FREQUENCY_HZ: ParamId = SineOscParams::FREQUENCY_HZ;
    /// Linear amplitude.
    pub const AMPLITUDE: ParamId = SineOscParams::AMPLITUDE;

    /// A sine oscillator configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: SineOscSettings) -> Self {
        Self {
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

    /// A sine oscillator at 440 Hz, amplitude 0.5.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(SineOscSettings::default())
    }
}

impl Default for SineOsc {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for SineOsc {
    type Params = SineOscParams;

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

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &SineOscParams) {
        // Keep the endpoint below Nyquist so the maximum setting cannot
        // collapse to phase-dependent silence.
        let f = params.frequency_hz.min(max_oscillator_frequency(self.fs));
        let inc = TAU * f / self.fs; // radians per sample
        let amp = params.amplitude;
        let start = self.phase;
        let mut end = start;
        // Each channel carries the same tone and phase advances once per frame.
        for ch in 0..io.channels() {
            let mut ph = start;
            for slot in io.output_mut(ch).iter_mut() {
                *slot = T::from_f64(amp * math::sin(ph));
                ph += inc;
                ph = wrap_phase(ph);
            }
            // Every channel starts at `start` and advances by the same `inc`
            // per frame, so each iteration lands on the same end phase.
            // Carrying the last channel's value is correct only because of
            // that per-channel identity.
            end = ph;
        }
        self.phase = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::{AudioBlockMut, Io};

    #[test]
    fn settings_builders_and_phase_advance_preserve_direction() {
        let settings = SineOscSettings::new()
            .frequency_hz(12_000.0)
            .amplitude(0.25);
        assert_eq!(settings.frequency_hz, 12_000.0);
        assert_eq!(settings.amplitude, 0.25);

        let mut oscillator = SineOsc::with_settings(settings);
        Kernel::<f32>::prepare(
            &mut oscillator,
            ProcessSpec {
                sample_rate: 48_000,
                channels: 1,
                max_block: 4,
                max_memory: None,
            },
        )
        .expect("prepare");
        let mut samples = [0.0f32; 2];
        let mut planes = [samples.as_mut_slice()];
        let mut io = Io::OutputOnly(AudioBlockMut::new(&mut planes));
        let mut block = SubBlock {
            io: &mut io,
            sc: &[],
            start: 0,
            len: 2,
        };
        Kernel::<f32>::render(
            &mut oscillator,
            &mut block,
            &SineOscParams {
                frequency_hz: 12_000.0,
                amplitude: 0.25,
            },
        );
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 0.25).abs() < 1e-6);
    }
}
