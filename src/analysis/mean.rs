// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Per-channel mean (DC offset) meter.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{AudioBlock, DspError, Measurer, ProcessSpec, Sample};

use super::{debug_validate_meter_geometry, PreparedMeterContract};

/// Per-channel running-mean meter.
///
/// Accumulates the mean of each channel since the last reset. The `Measurer`
/// reading is the maximum absolute per-channel mean: a "peak DC" headline that
/// never cancels, unlike a pooled mean where opposite-signed channels would
/// average away. The signed per-channel means are read with
/// [`channel_mean`](Self::channel_mean).
///
/// Pairs with [`DcOffset`](crate::repair::DcOffset) for fixed DC correction.
/// Read each channel's mean, then apply its negation as a per-channel offset.
///
/// The denominator is frames observed per channel, not total samples: unlike
/// [`RmsMeter`](crate::analysis::RmsMeter), which pools all samples, a mean is
/// per channel. Accumulation is a plain running sum.
#[derive(Debug, Clone, Default)]
pub struct MeanMeter {
    sums: Vec<f64>, // per-channel running sum
    frames: u64,    // frames observed since reset, equal across channels
    prepared: Option<PreparedMeterContract>,
}

impl MeanMeter {
    /// A mean meter reading 0 (silence).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The mean of channel `ch` since the last reset, or `0.0` before any frame
    /// is observed.
    ///
    /// # Panics
    /// If `ch` is out of range or the meter has not been prepared. The channel
    /// is indexed before the empty-observation check, matching
    /// [`AudioBlock::channel`](crate::processor::AudioBlock::channel).
    #[must_use]
    pub fn channel_mean(&self, ch: usize) -> f64 {
        let sum = self.sums[ch]; // index first: panics on out-of-range or unprepared
        if self.frames == 0 {
            0.0
        } else {
            sum / self.frames as f64
        }
    }
}

impl<T: Sample> Measurer<T> for MeanMeter {
    type Reading = f64;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        // Check the budget before allocating the per-channel accumulators.
        MemoryLayout::new()
            .array::<f64>(spec.channels)
            .preflight(spec.max_memory)?;
        self.sums = vec![0.0; spec.channels];
        self.frames = 0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        self.sums.fill(0.0);
        self.frames = 0;
    }

    fn memory_footprint(&self) -> usize {
        // One f64 accumulator per channel.
        self.sums.len() * std::mem::size_of::<f64>()
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        for ch in 0..block.channels() {
            let acc = &mut self.sums[ch];
            for &s in block.channel(ch) {
                *acc += finite_or_zero(s.to_f64());
            }
        }
        self.frames += block.frames() as u64;
    }

    fn read(&self) -> f64 {
        if self.frames == 0 {
            return 0.0;
        }
        let frames = self.frames as f64;
        self.sums
            .iter()
            .map(|&s| (s / frames).abs())
            .fold(0.0f64, f64::max)
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

    /// `memory_footprint` is one f64 accumulator per channel.
    #[test]
    fn footprint_is_one_f64_per_channel() {
        let f = std::mem::size_of::<f64>();
        for ch in [1usize, 2, 6] {
            let mut m = MeanMeter::new();
            Measurer::<f32>::prepare(&mut m, spec(ch)).expect("prepare");
            assert_eq!(
                Measurer::<f32>::memory_footprint(&m),
                ch * f,
                "one f64 accumulator per channel"
            );
        }
    }

    /// A cap one byte under the footprint is rejected before allocating.
    #[test]
    fn over_budget_before_allocation() {
        let mut s = spec(2);
        s.max_memory = Some(2 * std::mem::size_of::<f64>() - 1);
        let mut m = MeanMeter::new();
        assert!(matches!(
            Measurer::<f32>::prepare(&mut m, s),
            Err(DspError::OverBudget { .. })
        ));
    }

    /// `channel_mean` indexes first, so an unprepared meter panics.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn channel_mean_panics_on_unprepared() {
        let m = MeanMeter::new();
        let _ = m.channel_mean(0);
    }

    /// An out-of-range channel panics.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn channel_mean_panics_out_of_range() {
        let mut m = MeanMeter::new();
        Measurer::<f32>::prepare(&mut m, spec(2)).expect("prepare");
        let _ = m.channel_mean(5);
    }
}
