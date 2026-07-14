// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Seeded white noise.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::rng::{channel_seed, Rng};
use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{DspError, Kernel, ProcessSpec, Sample, SubBlock};

// ---------------------------------------------------------------------------
// White noise
// ---------------------------------------------------------------------------

/// The default white-noise seed.
pub(super) const DEFAULT_NOISE_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Construction settings for [`WhiteNoise`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct WhiteNoiseSettings {
    /// Linear amplitude.
    pub amplitude: f64,
    /// Base seed. Each channel derives its own seed from this value.
    pub seed: u64,
}

impl Default for WhiteNoiseSettings {
    fn default() -> Self {
        Self {
            amplitude: 0.5,
            seed: DEFAULT_NOISE_SEED,
        }
    }
}

impl WhiteNoiseSettings {
    /// Default white-noise settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the linear amplitude.
    #[must_use]
    pub fn amplitude(mut self, amplitude: f64) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Set the base seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

crate::params! {
    /// Smoothed parameter values for [`WhiteNoise`].
    pub struct WhiteNoiseParams {
        /// Linear amplitude.
        pub amplitude => AMPLITUDE,
    }
}

/// A seeded white-noise source. Each channel emits a separately seeded,
/// deterministic uniform stream in `[-amp, amp)`.
///
/// An output-only [`Kernel`].
#[derive(Debug, Clone)]
pub struct WhiteNoise {
    params: [ParamInfo; 1],
    seed: u64,
    pub(super) rngs: Vec<Rng>,
}

impl WhiteNoise {
    /// Linear amplitude.
    pub const AMPLITUDE: ParamId = WhiteNoiseParams::AMPLITUDE;

    /// A white-noise source configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: WhiteNoiseSettings) -> Self {
        Self {
            params: [ParamInfo::new(
                Self::AMPLITUDE,
                "amplitude",
                (0.0, 1.0),
                settings.amplitude,
                Unit::Linear,
            )],
            seed: settings.seed,
            rngs: Vec::new(),
        }
    }

    /// White noise at amplitude 0.5 with the default seed.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(WhiteNoiseSettings::default())
    }
}

impl Default for WhiteNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for WhiteNoise {
    type Params = WhiteNoiseParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        MemoryLayout::new()
            .array::<Rng>(spec.channels)
            .preflight(spec.max_memory)?;
        self.rngs = (0..spec.channels)
            .map(|ch| Rng::new(channel_seed(self.seed, ch)))
            .collect();
        Ok(())
    }

    fn reset(&mut self) {
        for (ch, rng) in self.rngs.iter_mut().enumerate() {
            *rng = Rng::new(channel_seed(self.seed, ch));
        }
    }

    fn memory_footprint(&self) -> usize {
        self.rngs.len() * std::mem::size_of::<Rng>()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn io_mode(&self) -> crate::processor::IoMode {
        crate::processor::IoMode::OutputOnly
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &WhiteNoiseParams) {
        let amp = params.amplitude;
        for (ch, rng) in self.rngs.iter_mut().enumerate() {
            for slot in io.output_mut(ch).iter_mut() {
                *slot = T::from_f64(amp * rng.next_bipolar());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_builders_preserve_each_requested_value() {
        let settings = WhiteNoiseSettings::new().amplitude(0.25).seed(0x1234_5678);
        assert_eq!(settings.amplitude, 0.25);
        assert_eq!(settings.seed, 0x1234_5678);
    }
}
