// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! TPDF dither and quantization.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::rng::{channel_seed, Rng};
use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{DspError, IoMode, Kernel, ProcessSpec, Sample, SubBlock};

// ---------------------------------------------------------------------------
// Dither + quantizer
// ---------------------------------------------------------------------------

/// The default dither seed used by [`DitherSettings::default`].
const DEFAULT_DITHER_SEED: u64 = 0x1234_5678_9ABC_DEF0;

/// Construction settings for [`Dither`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DitherSettings {
    /// Quantizer bit depth (e.g. 16 for CD). Validated in `prepare` (`2..=24`).
    pub bits: u32,
    /// Base seed. Each channel derives its own seed from this value.
    pub seed: u64,
}

impl Default for DitherSettings {
    fn default() -> Self {
        Self {
            bits: 16,
            seed: DEFAULT_DITHER_SEED,
        }
    }
}

impl DitherSettings {
    /// Default dither settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the quantizer bit depth.
    #[must_use]
    pub fn bits(mut self, bits: u32) -> Self {
        self.bits = bits;
        self
    }

    /// Set the base seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Seeded TPDF dither plus a mid-tread quantizer.
///
/// Each sample receives triangular-PDF dither before quantization. Each channel
/// has its own generator seeded from the base seed. `reset` restores each
/// generator to its starting state.
#[derive(Debug, Clone)]
pub struct Dither {
    bits: u32,
    seed: u64,
    rngs: Vec<Rng>,
    step: f64,     // quantizer LSB, 2^-(bits-1)
    inv_step: f64, // 2^(bits-1), the number of positive levels
}

impl Dither {
    /// A dither + quantizer configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: DitherSettings) -> Self {
        Self {
            bits: settings.bits,
            seed: settings.seed,
            rngs: Vec::new(),
            step: 0.0,
            inv_step: 0.0,
        }
    }

    /// 16-bit dither with the default seed.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(DitherSettings::default())
    }
}

impl Default for Dither {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Dither {
    type Params = crate::parameter::NoParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        if !(2..=24).contains(&self.bits) {
            return Err(DspError::InvalidParam("dither bit depth must be in 2..=24"));
        }
        // 2^(bits - 1) is exact for bits <= 24, and its reciprocal is a power of two.
        let levels = (1u64 << (self.bits - 1)) as f64;
        MemoryLayout::new()
            .array::<Rng>(spec.channels)
            .preflight(spec.max_memory)?;
        self.inv_step = levels;
        self.step = 1.0 / levels;
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

    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }

    fn memory_footprint(&self) -> usize {
        self.rngs.len() * std::mem::size_of::<Rng>()
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, _params: &crate::parameter::NoParams) {
        let step = self.step;
        let inv = self.inv_step;
        let max_idx = inv - 1.0; // top level, e.g. +32767 for 16-bit
        let min_idx = -inv; // bottom level, e.g. -32768
        for (ch, rng) in self.rngs.iter_mut().enumerate() {
            for x in io.channel_mut(ch).iter_mut() {
                let v = finite_or_zero(x.to_f64());
                // 2-LSB peak-to-peak TPDF.
                let tri = rng.next_unit() - rng.next_unit(); // in (-1, 1)
                let dithered = v + tri * step;
                // Round to the integer level and clamp to the valid range.
                let idx = (dithered * inv).round().clamp(min_idx, max_idx);
                *x = T::from_f64(idx * step);
            }
        }
    }
}
