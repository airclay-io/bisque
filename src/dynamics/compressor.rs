// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Feed-forward downward compression.

use super::shared::{db_param, db_to_lin, ratio_param, DynamicsCore};
use crate::dsp::math;
use crate::parameter::{ParamId, ParamInfo};
use crate::processor::{DspError, IoMode, Kernel, ProcessSpec, Sample, SubBlock};

// ---------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------

/// Construction settings for [`Compressor`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct CompressorSettings {
    /// Threshold in dBFS.
    pub threshold_db: f64,
    /// Compression ratio as `ratio:1`.
    pub ratio: f64,
    /// Attack time constant in milliseconds. Zero is immediate.
    pub attack_ms: f64,
    /// Release time constant in milliseconds. Zero is immediate.
    pub release_ms: f64,
    /// Makeup gain in dB.
    pub makeup_db: f64,
    /// Whether detection reads sidechain bus 0 instead of the main input.
    pub use_sidechain: bool,
}

impl Default for CompressorSettings {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            makeup_db: 0.0,
            use_sidechain: false,
        }
    }
}

impl CompressorSettings {
    /// Default compressor settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the threshold in dBFS.
    #[must_use]
    pub fn threshold_db(mut self, threshold_db: f64) -> Self {
        self.threshold_db = threshold_db;
        self
    }

    /// Set the compression ratio.
    #[must_use]
    pub fn ratio(mut self, ratio: f64) -> Self {
        self.ratio = ratio;
        self
    }

    /// Set the attack time constant in milliseconds. Zero is immediate.
    #[must_use]
    pub fn attack_ms(mut self, attack_ms: f64) -> Self {
        self.attack_ms = attack_ms;
        self
    }

    /// Set the release time constant in milliseconds. Zero is immediate.
    #[must_use]
    pub fn release_ms(mut self, release_ms: f64) -> Self {
        self.release_ms = release_ms;
        self
    }

    /// Set the makeup gain in dB.
    #[must_use]
    pub fn makeup_db(mut self, makeup_db: f64) -> Self {
        self.makeup_db = makeup_db;
        self
    }

    /// Enable or disable sidechain detection.
    #[must_use]
    pub fn use_sidechain(mut self, use_sidechain: bool) -> Self {
        self.use_sidechain = use_sidechain;
        self
    }
}

crate::params! {
    /// Smoothed parameter values for [`Compressor`].
    pub struct CompressorParams {
        /// Threshold in dBFS.
        pub threshold_db => THRESHOLD_DB,
        /// Compression ratio as `ratio:1`.
        pub ratio => RATIO,
        /// Makeup gain in dB.
        pub makeup_db => MAKEUP_DB,
    }
}

/// A feed-forward downward compressor.
///
/// With [`with_sidechain`](Compressor::with_sidechain) the follower keys off an
/// external bus instead of the main signal. Threshold, ratio, and makeup are
/// automatable. Attack and release are fixed at construction.
///
/// Detection is linked peak detection. Attack and release smooth the detected
/// level before the hard-knee static curve is evaluated. The compressor does not
/// provide RMS detection, soft knee, program-dependent release, or gain-reduction
/// smoothing.
#[derive(Debug, Clone)]
pub struct Compressor {
    params: [ParamInfo; 3],
    core: DynamicsCore,
}

impl Compressor {
    /// Threshold in dBFS.
    pub const THRESHOLD_DB: ParamId = CompressorParams::THRESHOLD_DB;
    /// Compression ratio.
    pub const RATIO: ParamId = CompressorParams::RATIO;
    /// Makeup gain in dB.
    pub const MAKEUP_DB: ParamId = CompressorParams::MAKEUP_DB;

    /// A compressor configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: CompressorSettings) -> Self {
        Self {
            params: [
                db_param(
                    Self::THRESHOLD_DB,
                    "threshold",
                    (-60.0, 0.0),
                    settings.threshold_db,
                ),
                ratio_param(settings.ratio),
                db_param(Self::MAKEUP_DB, "makeup", (0.0, 24.0), settings.makeup_db),
            ],
            core: DynamicsCore::new(
                settings.attack_ms,
                settings.release_ms,
                settings.use_sidechain,
            ),
        }
    }

    /// Defaults to -20 dBFS, 4:1, 10 ms attack, 100 ms release, no makeup, and
    /// main detection.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(CompressorSettings::default())
    }

    /// As [`new`](Compressor::new), but keying detection off a sidechain bus.
    #[must_use]
    pub fn with_sidechain() -> Self {
        Self::with_settings(CompressorSettings::new().use_sidechain(true))
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Compressor {
    type Params = CompressorParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.core.prepare(spec)
    }
    fn reset(&mut self) {
        self.core.reset();
    }
    fn io_mode(&self) -> IoMode {
        IoMode::InPlace
    }
    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }
    fn sidechain_inputs(&self) -> usize {
        usize::from(self.core.use_sidechain)
    }
    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &CompressorParams) {
        let thresh_lin = db_to_lin(params.threshold_db);
        let ratio = params.ratio.max(1.0);
        let makeup_lin = db_to_lin(params.makeup_db);
        let slope = 1.0 - 1.0 / ratio; // log-domain compression slope
        self.core.render(io, move |env| {
            if env <= thresh_lin {
                makeup_lin
            } else {
                makeup_lin * math::exp(-slope * math::ln(env / thresh_lin))
            }
        });
    }
}
