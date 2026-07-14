// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

#![cfg_attr(docsrs, feature(doc_cfg))]

//! Contract-tested audio DSP processors for realtime and offline hosts.
//!
//! # Start With One Processor
//!
//! Configure ordinary fixed values through a domain settings type, then use
//! [`host::PreparedProcessor`] to prepare and drive one processor while it owns
//! the absolute processing timeline.
//!
//! ```no_run
//! use bisque::host::PreparedProcessor;
//! use bisque::mastering::{Gain, GainSettings};
//! use bisque::processor::ProcessSpec;
//!
//! # fn run(samples: &mut [f32]) -> Result<(), bisque::processor::DspError> {
//! let spec = ProcessSpec {
//!     sample_rate: 48_000,
//!     channels: 1,
//!     max_block: samples.len().max(1),
//!     max_memory: None,
//! };
//! let gain = Gain::with_settings(GainSettings::new().gain_db(-6.0));
//! let mut gain = PreparedProcessor::prepare_kernel(gain, spec)?;
//! let mut planes = [samples];
//! gain.process_in_place(&mut planes, &[]);
//! # Ok(())
//! # }
//! ```
//!
//! # Public Shape
//!
//! - Domain modules such as [`filters`], [`dynamics`], and [`mastering`] contain
//!   concrete processors and settings.
//! - [`host`] contains the optional single-processor lifecycle helper.
//! - [`processor`] contains raw host and processor-author contracts, including
//!   [`processor::Processor`], [`processor::Kernel`], blocks, specs, latency,
//!   tails, meters, sources, and variable-rate processing.
//! - [`parameter`] contains parameter identity, metadata, events, normalized
//!   mapping, and smoothing vocabulary. The [`params!`] macro remains at the
//!   crate root.
//! - [`dsp`] contains lower-level deterministic math, smoothing, oversampling,
//!   and sanitization utilities for DSP authors.
//! - [`testing`] provides shared downstream contract tests with `test-support`.
//!
//! Each public concept has one canonical path. There is no prelude and no broad
//! crate-root re-export layer. Realtime and specialized hosts use
//! [`processor::Processor`] directly; `PreparedProcessor` remains optional and
//! owns no graph, queue, transport, file, plugin, or tail-length policy.

mod contract;

pub mod dsp;

#[cfg(feature = "test-support")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-support")))]
pub mod testing;

// Private source modules remain available to sibling implementation modules.
pub(crate) use contract::{block, context, error, param, spec, traits};

/// Low-level lifecycle, block, sample, and processor-authoring contracts.
pub mod processor {
    pub use crate::contract::{
        AudioBlock, AudioBlockMut, DspError, Io, IoMode, Kernel, Measurer, ProcessContext,
        ProcessSpec, Processor, Produced, RingSource, Sample, Source, SubBlock, Tail, VariableRate,
    };
    pub use crate::dsp::driver::KernelProcessor;
}

/// Parameter identity, metadata, automation, smoothing, and normalized mapping.
pub mod parameter {
    pub use crate::contract::{
        NoParams, ParamEvent, ParamId, ParamInfo, ParamSetError, ParamValueError, Params,
        Smoothing, Unit, ValueScale,
    };
}

/// Optional universal lifecycle support for hosting one processor.
pub mod host;

/// RBJ biquad (low-pass, high-pass, shelves, peaking) with response readouts,
/// and a moving-average FIR.
#[cfg(feature = "filters")]
#[cfg_attr(docsrs, doc(cfg(feature = "filters")))]
pub mod filters;

/// Peak-detected hard-knee compressor, expander, gate, and sidechain support.
#[cfg(feature = "dynamics")]
#[cfg_attr(docsrs, doc(cfg(feature = "dynamics")))]
pub mod dynamics;

/// Gain, TPDF dither, quantization, and lookahead true-peak limiting.
#[cfg(feature = "mastering")]
#[cfg_attr(docsrs, doc(cfg(feature = "mastering")))]
pub mod mastering;

/// Peak/true-peak/RMS/windowed-RMS/crest meters, LUFS loudness, and clipping.
#[cfg(feature = "analysis")]
#[cfg_attr(docsrs, doc(cfg(feature = "analysis")))]
pub mod analysis;

/// Sine, seeded white noise, and PolyBLEP saw/square oscillators.
#[cfg(feature = "generators")]
#[cfg_attr(docsrs, doc(cfg(feature = "generators")))]
pub mod generators;

/// Feedback delay and overlap-add time stretching.
#[cfg(feature = "time")]
#[cfg_attr(docsrs, doc(cfg(feature = "time")))]
pub mod time;

/// DC removal.
#[cfg(feature = "repair")]
#[cfg_attr(docsrs, doc(cfg(feature = "repair")))]
pub mod repair;

/// STFT/ISTFT, windows, overlap-add helpers, and streaming spectral processors.
#[cfg(feature = "spectral")]
#[cfg_attr(docsrs, doc(cfg(feature = "spectral")))]
pub mod spectral;
