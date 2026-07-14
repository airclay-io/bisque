// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Fixed linear-gain scaling.

use crate::dsp::sanitize::finite_or_zero;
use crate::parameter::NoParams;
use crate::processor::{DspError, IoMode, Kernel, ProcessSpec, Sample, SubBlock};

/// A fixed linear-gain scale.
///
/// Multiplies every sample by one constant factor chosen at construction. Unlike
/// [`Gain`](crate::mastering::Gain), the factor is not an automatable parameter,
/// so it is unbounded: any finite factor is allowed, including values outside
/// Gain's `-96..=24` dB range and negative factors (a polarity inversion). The
/// factor is validated in `prepare`; a non-finite factor is rejected with
/// [`DspError::InvalidParam`].
///
/// `from_db` uses the pinned deterministic
/// [`db_to_linear`](crate::dsp::db_to_linear), so byte-exact-capable chains stay
/// byte-exact.
#[derive(Debug, Clone)]
pub struct Scale {
    factor: f64,
}

impl Scale {
    /// A scale by a linear `factor` (unity is `1.0`).
    #[must_use]
    pub fn new(factor: f64) -> Self {
        Self { factor }
    }

    /// A scale by a decibel gain, converted to a linear factor.
    ///
    /// `from_db(f64::NEG_INFINITY)` resolves to a finite `0.0` (silence) and is
    /// allowed; a `+inf` or `NaN` dB resolves to a non-finite factor and is
    /// rejected in `prepare`.
    #[must_use]
    pub fn from_db(db: f64) -> Self {
        Self::new(crate::dsp::db_to_linear(db))
    }

    /// A polarity inversion (factor `-1.0`).
    #[must_use]
    pub fn inverted() -> Self {
        Self::new(-1.0)
    }
}

impl<T: Sample> Kernel<T> for Scale {
    type Params = NoParams;

    fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
        if !self.factor.is_finite() {
            return Err(DspError::InvalidParam("scale factor must be finite"));
        }
        Ok(())
    }

    fn reset(&mut self) {}

    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, _params: &NoParams) {
        let g = self.factor; // finite, validated in prepare
        let channels = io.channels();
        for ch in 0..channels {
            for x in io.channel_mut(ch).iter_mut() {
                *x = T::from_f64(finite_or_zero(x.to_f64()) * g);
            }
        }
    }
}
