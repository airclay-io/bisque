// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! True-peak meter.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::oversample::PolyphaseUpsampler;
use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{AudioBlock, DspError, Measurer, ProcessSpec, Sample};

use super::{debug_validate_meter_geometry, PreparedMeterContract};

/// True-peak meter. It oversamples each channel and reports the largest
/// reconstructed inter-sample magnitude.
///
/// The oversampling FIR introduces a small detector group delay: the reading
/// describes the input [`latency`](Measurer::latency) frames before the most
/// recently observed one (6 frames at the default 4x, 12-tap configuration).
///
/// This type is intended for deterministic, allocation-free audio-path
/// metering. For offline conformance measurement, use a dedicated measurement
/// tool; see the [`analysis`](crate::analysis) module guidance.
#[derive(Debug, Clone)]
pub struct TruePeakMeter {
    factor: usize,
    taps: usize,
    os: Vec<PolyphaseUpsampler>, // one per channel
    peak: f64,
    prepared: Option<PreparedMeterContract>,
}

impl TruePeakMeter {
    /// A true-peak meter with the 4x, 12-tap default oversampler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factor: 4,
            taps: 12,
            os: Vec::new(),
            peak: 0.0,
            prepared: None,
        }
    }
}

impl Default for TruePeakMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Measurer<T> for TruePeakMeter {
    type Reading = f64;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        // Check the budget before allocating one oversampler (phase
        // coefficients plus a delay line) per channel.
        let per_channel = self
            .factor
            .checked_mul(self.taps)
            .and_then(|n| n.checked_add(self.taps))
            .ok_or(DspError::UnsupportedSpec(
                "true-peak oversampler layout exceeds addressable memory",
            ))?;
        MemoryLayout::new()
            .repeated_array::<f64>(spec.channels, per_channel)
            .preflight(spec.max_memory)?;
        self.os = (0..spec.channels)
            .map(|_| PolyphaseUpsampler::new(self.factor, self.taps))
            .collect();
        self.peak = 0.0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        for o in &mut self.os {
            o.reset();
        }
        self.peak = 0.0;
    }

    fn latency(&self) -> usize {
        // The prototype FIR has `factor * taps` coefficients, so its group
        // delay is `(factor * taps - 1) / 2` samples in the oversampled domain.
        // Dividing by the factor and rounding up (a half-phase rounds to the
        // next whole input frame) gives the input-domain group delay:
        // `(4 * 12 - 1).div_ceil(2 * 4) = 6` frames at the default settings.
        (self.factor * self.taps - 1).div_ceil(2 * self.factor)
    }

    fn memory_footprint(&self) -> usize {
        // One oversampler (coefficients plus delay line) per channel.
        self.os.iter().map(PolyphaseUpsampler::footprint).sum()
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        for ch in 0..block.channels() {
            let os = &mut self.os[ch];
            for &s in block.channel(ch) {
                let x = finite_or_zero(s.to_f64());
                let tp = os.peak_abs(x).max(x.abs());
                self.peak = self.peak.max(tp);
            }
        }
    }

    fn read(&self) -> f64 {
        self.peak
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

    /// `memory_footprint` equals the byte count derived from the allocation
    /// layout.
    #[test]
    fn footprint_is_the_exact_layout_byte_count() {
        let f = std::mem::size_of::<f64>();
        // Each 4x, 12-tap oversampler holds coefficients plus a delay line.
        let per_os = (4 * 12 + 12) * f;
        for ch in [1usize, 2, 3] {
            let mut meter = TruePeakMeter::new();
            Measurer::<f32>::prepare(&mut meter, spec(ch)).expect("prepare");
            assert_eq!(
                Measurer::<f32>::memory_footprint(&meter),
                ch * per_os,
                "one oversampler per channel"
            );
        }
    }

    /// The documented input-domain FIR group delay.
    #[test]
    fn latency_is_the_documented_fir_group_delay() {
        let mut meter = TruePeakMeter::new();
        Measurer::<f32>::prepare(&mut meter, spec(2)).expect("prepare");
        // (4 * 12 - 1).div_ceil(2 * 4) = 6 input frames.
        assert_eq!(Measurer::<f32>::latency(&meter), 6);
    }
}
