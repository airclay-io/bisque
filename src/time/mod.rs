// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Time-domain effects.
//!
//! # Public API
//!
//! - [`Delay`](crate::time::Delay) and
//!   [`DelaySettings`](crate::time::DelaySettings) implement feedback delay,
//!   with smoothed values in [`DelayParams`](crate::time::DelayParams).
//!   `Delay` uses integer-sample delay times and a clean feedback path.
//! - [`TimeStretch`](crate::time::TimeStretch) and
//!   [`TimeStretchSettings`](crate::time::TimeStretchSettings) implement
//!   overlap-add time stretching as a variable-rate processor.
//!   `TimeStretch` is not a phase vocoder and does not perform phase or
//!   transient preservation. Sustained pitches can become unstable away from
//!   unity ratio.

mod delay;
mod time_stretch;

pub use delay::{Delay, DelayParams, DelaySettings};
pub use time_stretch::{TimeStretch, TimeStretchSettings};
