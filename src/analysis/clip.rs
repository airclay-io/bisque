// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Clip meter.

use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{AudioBlock, DspError, Measurer, ProcessSpec, Sample};

use super::{debug_validate_meter_geometry, PreparedMeterContract};

/// Construction settings for [`ClipMeter`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ClipMeterSettings {
    /// Clip threshold as a linear amplitude. Values at or above it count as
    /// clipped. `prepare` requires a finite positive value.
    pub threshold: f64,
}

impl Default for ClipMeterSettings {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

impl ClipMeterSettings {
    /// Default clip-meter settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the clip threshold as a linear amplitude.
    #[must_use]
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

/// Clip meter. It counts samples whose magnitude reaches `threshold` across all
/// channels since reset.
#[derive(Debug, Clone)]
pub struct ClipMeter {
    threshold: f64,
    count: u64,
    prepared: Option<PreparedMeterContract>,
}

impl ClipMeter {
    /// A clip meter configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: ClipMeterSettings) -> Self {
        Self {
            threshold: settings.threshold,
            count: 0,
            prepared: None,
        }
    }

    /// A clip meter at full scale (counts `|x| >= 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(ClipMeterSettings::default())
    }

    /// The number of clipped samples observed since the last reset.
    #[must_use]
    pub fn clipped(&self) -> u64 {
        self.count
    }
}

impl Default for ClipMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Measurer<T> for ClipMeter {
    type Reading = u64;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        if !self.threshold.is_finite() || self.threshold <= 0.0 {
            return Err(DspError::InvalidParam(
                "clip threshold must be finite and positive",
            ));
        }
        self.count = 0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        self.count = 0;
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        for ch in 0..block.channels() {
            for &s in block.channel(ch) {
                if finite_or_zero(s.to_f64()).abs() >= self.threshold {
                    self.count += 1;
                }
            }
        }
    }

    fn read(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clip meter allocates nothing in `prepare`, so the trait's default
    /// zero footprint is the exact layout byte count.
    #[test]
    fn footprint_is_zero_scalar_state_only() {
        let mut meter = ClipMeter::new();
        let spec = ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 512,
            max_memory: None,
        };
        Measurer::<f32>::prepare(&mut meter, spec).expect("prepare");
        assert_eq!(Measurer::<f32>::memory_footprint(&meter), 0);
    }

    #[test]
    fn invalid_thresholds_are_rejected_during_prepare() {
        for threshold in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut meter = ClipMeter::with_settings(ClipMeterSettings::new().threshold(threshold));
            let spec = ProcessSpec {
                sample_rate: 48_000,
                channels: 2,
                max_block: 512,
                max_memory: None,
            };
            assert!(matches!(
                Measurer::<f32>::prepare(&mut meter, spec),
                Err(DspError::InvalidParam(_))
            ));
        }
    }
}
