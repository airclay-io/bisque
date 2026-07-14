// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Level-dependent gain processors.
//!
//! # Public API
//!
//! - [`Compressor`](crate::dynamics::Compressor) and
//!   [`CompressorSettings`](crate::dynamics::CompressorSettings) implement
//!   feed-forward downward compression, with smoothed values in
//!   [`CompressorParams`](crate::dynamics::CompressorParams).
//! - [`Expander`](crate::dynamics::Expander) and
//!   [`ExpanderSettings`](crate::dynamics::ExpanderSettings) implement downward
//!   expansion, with smoothed values in
//!   [`ExpanderParams`](crate::dynamics::ExpanderParams).
//! - [`Gate`](crate::dynamics::Gate) and
//!   [`GateSettings`](crate::dynamics::GateSettings) implement noise gating with
//!   a floor range, with smoothed values in
//!   [`GateParams`](crate::dynamics::GateParams).
//!
//! These processors use linked peak detection and hard-knee static curves.
//! Attack and release smooth the detected level, not the gain reduction.
//! Positive attack and release values are one-pole time constants. Zero applies
//! the detected level immediately.

mod compressor;
mod expander;
mod gate;
mod shared;

pub use compressor::{Compressor, CompressorParams, CompressorSettings};
pub use expander::{Expander, ExpanderParams, ExpanderSettings};
pub use gate::{Gate, GateParams, GateSettings};
