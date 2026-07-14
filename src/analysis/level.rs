// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Peak, RMS, and crest meters.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{AudioBlock, DspError, Measurer, ProcessSpec, Sample};

use super::{debug_validate_meter_geometry, PreparedMeterContract};

/// Convert a linear amplitude to dBFS.
///
/// Full scale `1.0` maps to `0.0`. Silence maps to negative infinity.
#[must_use]
pub fn linear_to_dbfs(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        crate::dsp::amplitude_to_db(linear)
    }
}

/// Sample-peak meter.
///
/// Reports the maximum absolute sample observed since the last reset as a
/// linear amplitude. Convert with [`linear_to_dbfs`].
#[derive(Debug, Clone, Default)]
pub struct PeakMeter {
    peak: f64,
    prepared: Option<PreparedMeterContract>,
}

impl PeakMeter {
    /// A peak meter reading 0 (silence).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Sample> Measurer<T> for PeakMeter {
    type Reading = f64;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        // `prepare` establishes the post-prepare state regardless of history.
        // A reused meter starts the new stream without a stale peak.
        self.peak = 0.0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        self.peak = 0.0;
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        for ch in 0..block.channels() {
            for &s in block.channel(ch) {
                let a = finite_or_zero(s.to_f64()).abs();
                self.peak = self.peak.max(a);
            }
        }
    }

    fn read(&self) -> f64 {
        self.peak
    }
}

/// Root-mean-square meter over all samples observed since reset.
#[derive(Debug, Clone, Default)]
pub struct RmsMeter {
    sum_sq: Vec<f64>, // per-channel sum of squares
    count: u64,       // total samples
    prepared: Option<PreparedMeterContract>,
}

impl RmsMeter {
    /// An RMS meter reading 0 (silence).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Sample> Measurer<T> for RmsMeter {
    type Reading = f64;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        // Check the budget before allocating the per-channel accumulators.
        MemoryLayout::new()
            .array::<f64>(spec.channels)
            .preflight(spec.max_memory)?;
        self.sum_sq = vec![0.0; spec.channels];
        self.count = 0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        self.sum_sq.fill(0.0);
        self.count = 0;
    }

    fn memory_footprint(&self) -> usize {
        // The per-channel sum-of-squares accumulators.
        self.sum_sq.len() * std::mem::size_of::<f64>()
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        for ch in 0..block.channels() {
            let acc = &mut self.sum_sq[ch];
            for &s in block.channel(ch) {
                let x = finite_or_zero(s.to_f64());
                *acc += x * x;
            }
        }
        self.count += block.frames() as u64 * block.channels() as u64;
    }

    fn read(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let total: f64 = self.sum_sq.iter().sum();
        crate::dsp::math::sqrt(total / self.count as f64)
    }
}

/// Crest-factor meter. The reading is peak divided by RMS as a linear ratio.
#[derive(Debug, Clone, Default)]
pub struct CrestMeter {
    peak: f64,
    sum_sq: Vec<f64>,
    count: u64,
    prepared: Option<PreparedMeterContract>,
}

impl CrestMeter {
    /// A crest meter reading 0 (silence).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Sample> Measurer<T> for CrestMeter {
    type Reading = f64;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        // Check the budget before allocating the per-channel accumulators.
        MemoryLayout::new()
            .array::<f64>(spec.channels)
            .preflight(spec.max_memory)?;
        self.peak = 0.0;
        self.sum_sq = vec![0.0; spec.channels];
        self.count = 0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        self.peak = 0.0;
        self.sum_sq.fill(0.0);
        self.count = 0;
    }

    fn memory_footprint(&self) -> usize {
        // The per-channel sum-of-squares accumulators.
        self.sum_sq.len() * std::mem::size_of::<f64>()
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        for ch in 0..block.channels() {
            for &s in block.channel(ch) {
                let x = finite_or_zero(s.to_f64());
                self.sum_sq[ch] += x * x;
                let a = x.abs();
                self.peak = self.peak.max(a);
            }
        }
        self.count += block.frames() as u64 * block.channels() as u64;
    }

    fn read(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let total: f64 = self.sum_sq.iter().sum();
        let rms = crate::dsp::math::sqrt(total / self.count as f64);
        if rms <= 0.0 {
            0.0
        } else {
            self.peak / rms
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(channels: usize) -> ProcessSpec {
        ProcessSpec {
            sample_rate: 48_000,
            channels,
            max_block: 512,
            max_memory: None,
        }
    }

    /// Pin the public non-finite behavior of `linear_to_dbfs`.
    #[test]
    fn linear_to_dbfs_non_finite_behavior_is_pinned() {
        assert!(linear_to_dbfs(f64::NAN).is_nan());
        assert_eq!(linear_to_dbfs(f64::INFINITY), f64::INFINITY);
        assert_eq!(linear_to_dbfs(0.0), f64::NEG_INFINITY);
        assert_eq!(linear_to_dbfs(-1.0), f64::NEG_INFINITY);
        assert!(linear_to_dbfs(1.0).abs() < 1e-12);
    }

    /// `memory_footprint` equals the byte count derived from the allocation
    /// layout.
    #[test]
    fn footprints_are_the_exact_layout_byte_counts() {
        let f = std::mem::size_of::<f64>();
        for ch in [1usize, 2, 3] {
            let mut peak = PeakMeter::new();
            Measurer::<f32>::prepare(&mut peak, spec(ch)).expect("prepare");
            assert_eq!(
                Measurer::<f32>::memory_footprint(&peak),
                0,
                "PeakMeter holds scalar state only"
            );

            let mut rms = RmsMeter::new();
            Measurer::<f32>::prepare(&mut rms, spec(ch)).expect("prepare");
            assert_eq!(
                Measurer::<f32>::memory_footprint(&rms),
                ch * f,
                "RmsMeter holds one f64 accumulator per channel"
            );

            let mut crest = CrestMeter::new();
            Measurer::<f32>::prepare(&mut crest, spec(ch)).expect("prepare");
            assert_eq!(
                Measurer::<f32>::memory_footprint(&crest),
                ch * f,
                "CrestMeter holds one f64 accumulator per channel"
            );
        }
    }
}
