// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Fixed additive DC offset.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::finite_or_zero;
use crate::parameter::NoParams;
use crate::processor::{DspError, IoMode, Kernel, ProcessSpec, Sample, SubBlock};

/// A fixed additive offset applied uniformly or per channel.
///
/// Adds a constant to every sample of a channel. Paired with a per-channel mean
/// meter (`MeanMeter`), this is exact DC removal: measure the mean of each
/// channel, then apply its negation here. Unlike
/// [`DcBlocker`](crate::repair::DcBlocker) it is not a filter, so it adds no
/// transient and does not change the spectrum; it removes one constant offset,
/// not time-varying DC, and the residual sits at the numerical floor rather than
/// bitwise zero.
///
/// Choose the uniform form with [`Self::broadcast`]. Choose the per-channel form
/// with [`Self::per_channel`] or [`Self::per_channel_from_slice`]. Per-channel
/// offsets must exactly match the prepared channel count.
#[derive(Debug, Clone)]
pub struct DcOffset {
    offsets: Offsets,
}

#[derive(Debug, Clone)]
enum Offsets {
    Broadcast(f64),
    PerChannel(Vec<f64>),
}

impl DcOffset {
    /// One offset broadcast to every channel.
    #[must_use]
    pub fn broadcast(offset: f64) -> Self {
        Self {
            offsets: Offsets::Broadcast(offset),
        }
    }

    /// Per-channel offsets, taking ownership of the buffer.
    ///
    /// The number of offsets must exactly match the prepared channel count.
    #[must_use]
    pub fn per_channel(offsets: Vec<f64>) -> Self {
        Self {
            offsets: Offsets::PerChannel(offsets),
        }
    }

    /// Per-channel offsets copied from a slice.
    ///
    /// The number of offsets must exactly match the prepared channel count.
    #[must_use]
    pub fn per_channel_from_slice(offsets: &[f64]) -> Self {
        Self::per_channel(offsets.to_vec())
    }
}

impl<T: Sample> Kernel<T> for DcOffset {
    type Params = NoParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        let channels = spec.channels;
        match &self.offsets {
            Offsets::Broadcast(offset) => {
                if !offset.is_finite() {
                    return Err(DspError::InvalidParam("offset must be finite"));
                }
            }
            Offsets::PerChannel(offsets) => {
                if offsets.len() != channels {
                    return Err(DspError::InvalidParam(
                        "per-channel offset count must equal the channel count",
                    ));
                }
                if offsets.iter().any(|offset| !offset.is_finite()) {
                    return Err(DspError::InvalidParam("offsets must be finite"));
                }
                MemoryLayout::new()
                    .array::<f64>(offsets.len())
                    .preflight(spec.max_memory)?;
            }
        }
        Ok(())
    }

    fn reset(&mut self) {}

    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }

    fn memory_footprint(&self) -> usize {
        match &self.offsets {
            Offsets::Broadcast(_) => 0,
            Offsets::PerChannel(offsets) => offsets.len() * std::mem::size_of::<f64>(),
        }
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, _params: &NoParams) {
        let channels = io.channels();
        for ch in 0..channels {
            let offset = match &self.offsets {
                Offsets::Broadcast(offset) => *offset,
                Offsets::PerChannel(offsets) => offsets[ch],
            };
            for x in io.channel_mut(ch).iter_mut() {
                *x = T::from_f64(finite_or_zero(x.to_f64()) + offset);
            }
        }
    }
}
