// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Repair processors.
//!
//! # Public API
//!
//! - [`DcBlocker`](crate::repair::DcBlocker) and
//!   [`DcBlockerSettings`](crate::repair::DcBlockerSettings) remove DC offset
//!   with a one-pole, one-zero high-pass filter, with smoothed values in
//!   [`DcBlockerParams`](crate::repair::DcBlockerParams).
//! - [`DcOffset`](crate::repair::DcOffset) applies a fixed uniform or
//!   per-channel additive offset, the exact counterpart to `DcBlocker` for
//!   measured DC removal.

mod dc_blocker;
mod dc_offset;

pub use dc_blocker::{DcBlocker, DcBlockerParams, DcBlockerSettings};
pub use dc_offset::DcOffset;
