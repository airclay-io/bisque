// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Automatable gain in decibels.

use std::f64::consts::LN_10;

use crate::dsp::math;
use crate::dsp::sanitize::finite_or_zero;
use crate::parameter::{ParamId, ParamInfo, Unit};
use crate::processor::{DspError, IoMode, Kernel, ProcessSpec, Sample, SubBlock};

crate::params! {
    /// Smoothed parameter values for [`Gain`].
    pub struct GainParams {
        /// Gain in dB.
        pub gain_db => GAIN_DB,
    }
}

/// Automatable gain in decibels.
///
/// dB-to-linear is computed once per sub-block (the parameter is constant within
/// that sub-block), so the per-sample work is a multiply. The conversion uses
/// the vendored deterministic math, so output is byte-exact across supported
/// platforms.
#[derive(Debug, Clone)]
pub struct Gain {
    params: [ParamInfo; 1],
}

/// Construction settings for [`Gain`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct GainSettings {
    /// Initial gain in decibels.
    pub gain_db: f64,
}

impl GainSettings {
    /// Unity-gain settings.
    #[must_use]
    pub const fn new() -> Self {
        Self { gain_db: 0.0 }
    }

    /// Set the initial gain in decibels.
    #[must_use]
    pub const fn gain_db(mut self, gain_db: f64) -> Self {
        self.gain_db = gain_db;
        self
    }
}

impl Default for GainSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl Gain {
    /// Gain in decibels.
    pub const GAIN_DB: ParamId = GainParams::GAIN_DB;

    /// A gain kernel defaulting to 0 dB (unity).
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(GainSettings::new())
    }

    /// A gain kernel configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: GainSettings) -> Self {
        Self {
            params: [ParamInfo::new(
                Self::GAIN_DB,
                "gain",
                (-96.0, 24.0),
                settings.gain_db,
                Unit::Db,
            )],
        }
    }
}

impl Default for Gain {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Gain {
    type Params = GainParams;

    fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
        Ok(())
    }

    fn reset(&mut self) {}

    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &GainParams) {
        let db = params.gain_db; // constant for this sub-block
        let g = math::exp(db * (LN_10 / 20.0));
        let channels = io.channels();
        for ch in 0..channels {
            for x in io.channel_mut(ch).iter_mut() {
                *x = T::from_f64(finite_or_zero(x.to_f64()) * g);
            }
        }
    }
}
