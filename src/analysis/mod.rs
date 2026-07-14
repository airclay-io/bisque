// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Level meters implemented as `Measurer`s.
//!
//! For offline file measurement, use a full loudness tool. The `ebur128` crate
//! is the conformance reference in bisque's tests. bisque's meters are for
//! audio-path use with allocation-free `observe`, block-size invariance, and
//! deterministic output.
//!
//! # Public API
//!
//! - [`linear_to_dbfs`](crate::analysis::linear_to_dbfs) converts a linear level
//!   to dBFS.
//! - [`PeakMeter`](crate::analysis::PeakMeter),
//!   [`RmsMeter`](crate::analysis::RmsMeter),
//!   [`CrestMeter`](crate::analysis::CrestMeter), and
//!   [`TruePeakMeter`](crate::analysis::TruePeakMeter) report stream-level
//!   measurements.
//! - [`MeanMeter`](crate::analysis::MeanMeter) reports the per-channel mean (DC
//!   offset) for fixed correction with `DcOffset`.
//! - [`WindowedRmsMeter`](crate::analysis::WindowedRmsMeter) and
//!   [`WindowedRmsMeterSettings`](crate::analysis::WindowedRmsMeterSettings)
//!   report RMS over a rolling window.
//! - [`LoudnessMeter`](crate::analysis::LoudnessMeter) reports momentary,
//!   short-term, and integrated LUFS.
//! - [`DEFAULT_MAX_INTEGRATED_SECONDS`](crate::analysis::DEFAULT_MAX_INTEGRATED_SECONDS)
//!   is the default integrated-loudness history duration.
//! - [`ClipMeter`](crate::analysis::ClipMeter) and
//!   [`ClipMeterSettings`](crate::analysis::ClipMeterSettings) count samples at
//!   or above the clip threshold.

mod clip;
mod level;
mod loudness;
mod mean;
mod true_peak;
mod windowed_rms;

use crate::processor::{AudioBlock, ProcessSpec, Sample};

/// Prepared geometry retained for debug-time meter validation.
#[derive(Clone, Copy, Debug)]
struct PreparedMeterContract {
    channels: usize,
    max_block: usize,
}

impl PreparedMeterContract {
    const fn new(spec: ProcessSpec) -> Self {
        Self {
            channels: spec.channels,
            max_block: spec.max_block,
        }
    }
}

/// Check built-in meter geometry before indexing prepared state.
#[inline]
fn debug_validate_meter_geometry<T: Sample>(
    prepared: Option<&PreparedMeterContract>,
    block: &AudioBlock<'_, '_, T>,
) {
    #[cfg(debug_assertions)]
    {
        let prepared = prepared.expect("observe requires a successful prepare");
        assert!(
            block.channels() == prepared.channels,
            "meter channel count ({}) must equal the prepared channel count ({})",
            block.channels(),
            prepared.channels
        );
        assert!(
            block.frames() <= prepared.max_block,
            "meter block frames ({}) must not exceed the prepared max_block ({})",
            block.frames(),
            prepared.max_block
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (prepared, block);
    }
}

pub use clip::{ClipMeter, ClipMeterSettings};
pub use level::{linear_to_dbfs, CrestMeter, PeakMeter, RmsMeter};
pub use loudness::{
    LoudnessMeter, LoudnessMeterSettings, LoudnessReading, DEFAULT_MAX_INTEGRATED_SECONDS,
};
pub use mean::MeanMeter;
pub use true_peak::TruePeakMeter;
pub use windowed_rms::{WindowedRmsMeter, WindowedRmsMeterSettings};
