// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Feed-forward downward expansion.

use super::shared::{db_param, db_to_lin, ratio_param, DynamicsCore, ENV_FLOOR};
use crate::dsp::math;
use crate::parameter::{ParamId, ParamInfo};
use crate::processor::{DspError, IoMode, Kernel, ProcessSpec, Sample, SubBlock};

// ---------------------------------------------------------------------------
// Expander
// ---------------------------------------------------------------------------

/// Construction settings for [`Expander`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ExpanderSettings {
    /// Threshold in dBFS.
    pub threshold_db: f64,
    /// Expansion slope below threshold. At 2.0, a level 10 dB below threshold
    /// is moved to 20 dB below threshold.
    pub ratio: f64,
    /// Attack time constant in milliseconds. Zero is immediate.
    pub attack_ms: f64,
    /// Release time constant in milliseconds. Zero is immediate.
    pub release_ms: f64,
    /// Whether detection reads sidechain bus 0 instead of the main input.
    pub use_sidechain: bool,
}

impl Default for ExpanderSettings {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            ratio: 2.0,
            attack_ms: 5.0,
            release_ms: 80.0,
            use_sidechain: false,
        }
    }
}

impl ExpanderSettings {
    /// Default expander settings.
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

    /// Set the expansion slope below threshold.
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

    /// Enable or disable sidechain detection.
    #[must_use]
    pub fn use_sidechain(mut self, use_sidechain: bool) -> Self {
        self.use_sidechain = use_sidechain;
        self
    }
}

crate::params! {
    /// Smoothed parameter values for [`Expander`].
    pub struct ExpanderParams {
        /// Threshold in dBFS.
        pub threshold_db => THRESHOLD_DB,
        /// Expansion slope below threshold.
        pub ratio => RATIO,
    }
}

/// A feed-forward downward expander.
///
/// Below the threshold the signal is reduced. Above the threshold it passes
/// unchanged.
///
/// Detection is linked peak detection. Attack and release smooth the detected
/// level before the hard-knee static curve is evaluated. The expander does not
/// provide RMS detection, soft knee, program-dependent release, or gain-reduction
/// smoothing.
#[derive(Debug, Clone)]
pub struct Expander {
    params: [ParamInfo; 2],
    core: DynamicsCore,
}

impl Expander {
    /// Threshold in dBFS.
    pub const THRESHOLD_DB: ParamId = ExpanderParams::THRESHOLD_DB;
    /// Expansion slope below threshold.
    pub const RATIO: ParamId = ExpanderParams::RATIO;

    /// An expander configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: ExpanderSettings) -> Self {
        Self {
            params: [
                db_param(
                    Self::THRESHOLD_DB,
                    "threshold",
                    (-80.0, 0.0),
                    settings.threshold_db,
                ),
                ratio_param(settings.ratio),
            ],
            core: DynamicsCore::new(
                settings.attack_ms,
                settings.release_ms,
                settings.use_sidechain,
            ),
        }
    }

    /// Defaults to -40 dBFS, 2:1, 5 ms attack, 80 ms release, and main detection.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(ExpanderSettings::default())
    }

    /// As [`new`](Expander::new), but keying detection off a sidechain bus.
    #[must_use]
    pub fn with_sidechain() -> Self {
        Self::with_settings(ExpanderSettings::new().use_sidechain(true))
    }
}

impl Default for Expander {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Expander {
    type Params = ExpanderParams;

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
    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &ExpanderParams) {
        let thresh_lin = db_to_lin(params.threshold_db);
        let ratio = params.ratio.max(1.0);
        self.core.render(io, move |env| {
            if env >= thresh_lin {
                1.0
            } else {
                let e = env.max(ENV_FLOOR);
                math::exp((ratio - 1.0) * math::ln(e / thresh_lin))
            }
        });
    }
}
