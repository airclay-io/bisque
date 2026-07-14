// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! RBJ biquad filters and frequency-response readouts.
//!
//! # Public API
//!
//! - [`Biquad`](crate::filters::Biquad) is the automatable filter kernel, with
//!   smoothed values in [`BiquadParams`](crate::filters::BiquadParams).
//! - [`BiquadSettings`](crate::filters::BiquadSettings) selects the shape and
//!   initial cutoff, Q, and gain before prepare.
//! - [`BiquadKind`](crate::filters::BiquadKind) selects the RBJ response shape.
//! - [`BiquadCoeffs`](crate::filters::BiquadCoeffs) provides coefficient and
//!   response readouts.
//! - [`MovingAverage`](crate::filters::MovingAverage) is a split-I/O FIR moving
//!   average.

mod biquad;
mod moving_average;

pub use biquad::{Biquad, BiquadCoeffs, BiquadKind, BiquadParams, BiquadSettings};
pub use moving_average::MovingAverage;
