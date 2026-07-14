// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Noise gate.

use super::shared::{db_param, db_to_lin, ratio_param, DynamicsCore, ENV_FLOOR};
use crate::dsp::math;
use crate::parameter::{ParamId, ParamInfo};
use crate::processor::{DspError, IoMode, Kernel, ProcessSpec, Sample, SubBlock};

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

/// Construction settings for [`Gate`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct GateSettings {
    /// Threshold in dBFS.
    pub threshold_db: f64,
    /// Expansion slope below threshold before the range floor is applied. At
    /// 2.0, a level 10 dB below threshold is moved toward 20 dB below threshold.
    pub ratio: f64,
    /// Minimum gain below threshold in dB, from -120.0 to 0.0.
    pub range_db: f64,
    /// Attack time constant in milliseconds. Zero is immediate.
    pub attack_ms: f64,
    /// Release time constant in milliseconds. Zero is immediate.
    pub release_ms: f64,
    /// Whether detection reads sidechain bus 0 instead of the main input.
    pub use_sidechain: bool,
}

impl Default for GateSettings {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            ratio: 4.0,
            range_db: -60.0,
            attack_ms: 1.0,
            release_ms: 100.0,
            use_sidechain: false,
        }
    }
}

impl GateSettings {
    /// Default gate settings.
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

    /// Set the minimum gain below threshold in dB. Values are negative or zero.
    #[must_use]
    pub fn range_db(mut self, range_db: f64) -> Self {
        self.range_db = range_db;
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
    /// Smoothed parameter values for [`Gate`].
    pub struct GateParams {
        /// Threshold in dBFS.
        pub threshold_db => THRESHOLD_DB,
        /// Expansion slope below threshold before the range floor.
        pub ratio => RATIO,
        /// Minimum gain below threshold in dB.
        pub range_db => RANGE_DB,
    }
}

/// A noise gate.
///
/// Below the threshold the signal is attenuated toward the `range_db` gain
/// floor. Above the threshold it passes unchanged.
///
/// Detection is linked peak detection. Attack and release smooth the detected
/// level before the hard-knee static curve is evaluated. The gate does not
/// provide RMS detection, soft knee, program-dependent release, or gain-reduction
/// smoothing.
#[derive(Debug, Clone)]
pub struct Gate {
    params: [ParamInfo; 3],
    core: DynamicsCore,
}

impl Gate {
    /// Threshold in dBFS.
    pub const THRESHOLD_DB: ParamId = GateParams::THRESHOLD_DB;
    /// Expansion slope below threshold before the range floor.
    pub const RATIO: ParamId = GateParams::RATIO;
    /// Minimum gain below threshold in dB.
    pub const RANGE_DB: ParamId = GateParams::RANGE_DB;

    /// A gate configured from `settings`.
    #[must_use]
    pub fn with_settings(settings: GateSettings) -> Self {
        Self {
            params: [
                db_param(
                    Self::THRESHOLD_DB,
                    "threshold",
                    (-80.0, 0.0),
                    settings.threshold_db,
                ),
                ratio_param(settings.ratio),
                db_param(Self::RANGE_DB, "range", (-120.0, 0.0), settings.range_db),
            ],
            core: DynamicsCore::new(
                settings.attack_ms,
                settings.release_ms,
                settings.use_sidechain,
            ),
        }
    }

    /// Defaults to -40 dBFS, 4:1 steepness, -60 dB range, 1 ms attack, and
    /// 100 ms release.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(GateSettings::default())
    }

    /// As [`new`](Gate::new), but keying detection off a sidechain bus.
    #[must_use]
    pub fn with_sidechain() -> Self {
        Self::with_settings(GateSettings::new().use_sidechain(true))
    }
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> Kernel<T> for Gate {
    type Params = GateParams;

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
    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &GateParams) {
        let thresh_lin = db_to_lin(params.threshold_db);
        let ratio = params.ratio.max(1.0);
        let range_lin = db_to_lin(params.range_db);
        self.core.render(io, move |env| {
            if env >= thresh_lin {
                1.0
            } else {
                let e = env.max(ENV_FLOOR);
                let g = math::exp((ratio - 1.0) * math::ln(e / thresh_lin));
                g.max(range_lin)
            }
        });
    }
}
