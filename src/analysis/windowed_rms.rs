// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Sliding-window RMS meter.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{AudioBlock, DspError, Measurer, ProcessSpec, Sample};

use super::{debug_validate_meter_geometry, PreparedMeterContract};

/// Construction settings for [`WindowedRmsMeter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct WindowedRmsMeterSettings {
    /// Sliding-window length in frames. `prepare` requires a nonzero value.
    pub window_frames: usize,
}

impl Default for WindowedRmsMeterSettings {
    fn default() -> Self {
        Self { window_frames: 512 }
    }
}

impl WindowedRmsMeterSettings {
    /// Default windowed-RMS settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sliding-window length in frames.
    #[must_use]
    pub fn window_frames(mut self, window_frames: usize) -> Self {
        self.window_frames = window_frames;
        self
    }
}

/// Sliding-window RMS meter over the most recent `window` frames.
///
/// Each channel stores a ring of the last `window` squared samples. `read` sums
/// the rings in a fixed order.
#[derive(Debug, Clone)]
pub struct WindowedRmsMeter {
    window: usize,        // frames in the sliding window
    rings: Vec<Vec<f64>>, // per channel, the last `window` squared samples
    pos: usize,           // shared frame write position into each ring
    filled: usize,        // frames seen so far, capped at `window`
    prepared: Option<PreparedMeterContract>,
}

impl WindowedRmsMeter {
    /// A windowed RMS configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: WindowedRmsMeterSettings) -> Self {
        Self {
            window: settings.window_frames,
            rings: Vec::new(),
            pos: 0,
            filled: 0,
            prepared: None,
        }
    }

    /// A windowed RMS over the most recent 512 frames.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(WindowedRmsMeterSettings::default())
    }
}

impl Default for WindowedRmsMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Measurer<T> for WindowedRmsMeter {
    type Reading = f64;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        if self.window == 0 {
            return Err(DspError::InvalidParam(
                "windowed RMS window must be nonzero",
            ));
        }
        // Check the budget before allocating the per-channel rings. This meter
        // can use large windows, so the cap matters most here.
        MemoryLayout::new()
            .repeated_array::<f64>(spec.channels, self.window)
            .preflight(spec.max_memory)?;
        self.rings = vec![vec![0.0; self.window]; spec.channels];
        self.pos = 0;
        self.filled = 0;
        self.prepared = Some(PreparedMeterContract::new(spec));
        Ok(())
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            ring.fill(0.0);
        }
        self.pos = 0;
        self.filled = 0;
    }

    fn memory_footprint(&self) -> usize {
        // One `window`-slot squared-sample ring per channel.
        self.rings.iter().map(Vec::len).sum::<usize>() * std::mem::size_of::<f64>()
    }

    fn observe(&mut self, block: AudioBlock<'_, '_, T>) {
        debug_validate_meter_geometry(self.prepared.as_ref(), &block);
        // Each channel walks its ring from the same start position.
        let start = self.pos;
        let mut end = start;
        for ch in 0..block.channels() {
            let ring = &mut self.rings[ch];
            let mut p = start;
            for &s in block.channel(ch) {
                let x = finite_or_zero(s.to_f64());
                ring[p] = x * x;
                p = if p + 1 == self.window { 0 } else { p + 1 };
            }
            // Every channel advances one slot per frame from `start`; carrying
            // the last channel's end position relies on that identity.
            end = p;
        }
        self.pos = end;
        self.filled = (self.filled + block.frames()).min(self.window);
    }

    fn read(&self) -> f64 {
        if self.filled == 0 || self.rings.is_empty() {
            return 0.0;
        }
        // Sum rings in channel-major, ring-index order. Unfilled slots are zero.
        let total: f64 = self.rings.iter().map(|r| r.iter().sum::<f64>()).sum();
        crate::dsp::math::sqrt(total / (self.rings.len() * self.filled) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `memory_footprint` equals the byte count derived from the allocation
    /// layout.
    #[test]
    fn footprint_is_the_exact_layout_byte_count() {
        let window = 512usize;
        let f = std::mem::size_of::<f64>();
        for ch in [1usize, 2, 3] {
            let mut meter = WindowedRmsMeter::with_settings(
                WindowedRmsMeterSettings::new().window_frames(window),
            );
            let spec = ProcessSpec {
                sample_rate: 48_000,
                channels: ch,
                max_block: 512,
                max_memory: None,
            };
            Measurer::<f32>::prepare(&mut meter, spec).expect("prepare");
            assert_eq!(
                Measurer::<f32>::memory_footprint(&meter),
                ch * window * f,
                "one {window}-slot f64 ring per channel"
            );
        }
    }

    #[test]
    fn zero_window_is_rejected_during_prepare() {
        let mut meter =
            WindowedRmsMeter::with_settings(WindowedRmsMeterSettings::new().window_frames(0));
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
