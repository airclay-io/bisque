// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Mastering processors and utilities.
//!
//! # Public API
//!
//! - [`Gain`](crate::mastering::Gain) applies automatable gain in dB, with
//!   smoothed values in [`GainParams`](crate::mastering::GainParams).
//! - [`GainSettings`](crate::mastering::GainSettings) selects the initial gain
//!   before prepare.
//! - [`Scale`](crate::mastering::Scale) applies a fixed, unbounded linear factor
//!   (or dB via `from_db`, or a polarity inversion via `inverted`).
//! - [`Dither`](crate::mastering::Dither) and
//!   [`DitherSettings`](crate::mastering::DitherSettings) apply TPDF dither and
//!   quantization.
//! - [`Limiter`](crate::mastering::Limiter) and
//!   [`LimiterSettings`](crate::mastering::LimiterSettings) implement lookahead
//!   peak limiting, with smoothed values in
//!   [`LimiterParams`](crate::mastering::LimiterParams).
//!   The limiter uses true-peak detection, sample-rate gain application, and a
//!   configurable true-peak safety margin.

mod dither;
mod gain;
mod limiter;
mod scale;

pub use dither::{Dither, DitherSettings};
pub use gain::{Gain, GainParams, GainSettings};
pub use limiter::{Limiter, LimiterParams, LimiterSettings};
pub use scale::Scale;
