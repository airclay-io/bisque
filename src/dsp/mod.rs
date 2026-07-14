// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Lower-level DSP machinery and deterministic math helpers.
//!
//! # Public API
//!
//! - [`SmootherBank`] stores smoothed parameter state (built manually only by
//!   [`VariableRate`](crate::processor::VariableRate) implementations).
//! - [`math`] contains deterministic math wrappers used by byte-exact paths.
//! - [`oversample`] contains oversampling helpers.
//! - [`db_to_linear`] and [`linear_to_db_floor`] convert between decibels and
//!   linear amplitude (amplitude domain, `20 * log10`).
//!
//! [`KernelProcessor`](crate::processor::KernelProcessor) is the host-facing kernel
//! wrapper. It is part of the contract surface and re-exported from
//! [`crate::processor`].

#[doc(hidden)]
pub(crate) mod driver;
pub(crate) mod memory;
#[cfg(any(feature = "generators", feature = "mastering"))]
pub(crate) mod rng;
pub(crate) mod sanitize;

pub mod math;
pub mod oversample;

pub use driver::SmootherBank;

use std::f64::consts::LN_10;

/// Amplitude ratio to decibels, `20 * log10(x)`, with no floor and no domain
/// guard. Callers apply their own `x <= 0` and non-finite policy. Shared by
/// [`linear_to_db_floor`] and the analysis crate's `linear_to_dbfs`, so the
/// crate has a single amplitude-to-dB conversion.
pub(crate) fn amplitude_to_db(x: f64) -> f64 {
    20.0 * math::log10(x)
}

/// Convert a decibel gain to a linear amplitude factor.
///
/// Amplitude domain: `db_to_linear(db) = 10^(db / 20)`, so `0 dB` is unity
/// (`1.0`) and every `+6 dB` roughly doubles amplitude. Uses the pinned
/// deterministic [`math::exp`], so results are byte-exact across supported
/// platforms.
///
/// Boundary and non-finite inputs follow `exp` directly: `db_to_linear(0.0)` is
/// `1.0`, [`f64::NEG_INFINITY`] yields `0.0` (silence), [`f64::INFINITY`] yields
/// `f64::INFINITY`, and [`f64::NAN`] yields `f64::NAN`. A caller that treats the
/// result as a fixed factor should reject a non-finite return itself.
#[must_use]
pub fn db_to_linear(db: f64) -> f64 {
    math::exp(db * (LN_10 / 20.0))
}

/// Convert a linear amplitude to decibels, bounded below by `floor_db`.
///
/// Amplitude domain: `max(20 * log10(x), floor_db)` for finite `x > 0`.
/// Non-positive or non-finite `x` returns `floor_db`, so silence and garbage map
/// to the floor rather than `-inf` or `NaN`. Uses the pinned deterministic
/// [`math::log10`].
///
/// `floor_db` must be finite or [`f64::NEG_INFINITY`] (which means "no floor"); a
/// `+inf` or `NaN` floor is a caller error caught by a debug assertion.
#[must_use]
pub fn linear_to_db_floor(x: f64, floor_db: f64) -> f64 {
    debug_assert!(
        floor_db.is_finite() || floor_db == f64::NEG_INFINITY,
        "floor_db must be finite or negative infinity"
    );
    if x.is_finite() && x > 0.0 {
        amplitude_to_db(x).max(floor_db)
    } else {
        floor_db
    }
}

#[cfg(test)]
mod tests {
    use super::{db_to_linear, linear_to_db_floor};

    #[test]
    fn db_to_linear_reference_points() {
        assert_eq!(db_to_linear(0.0), 1.0);
        // +20 dB is 10x amplitude, -20 dB is 0.1x.
        assert!((db_to_linear(20.0) - 10.0).abs() < 1e-9);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-9);
        assert_eq!(db_to_linear(f64::NEG_INFINITY), 0.0);
        assert!(db_to_linear(f64::INFINITY).is_infinite());
        assert!(db_to_linear(f64::NAN).is_nan());
    }

    #[test]
    fn db_linear_round_trips() {
        for db in [-96.0, -24.0, -6.0, 0.0, 6.0, 24.0] {
            let back = linear_to_db_floor(db_to_linear(db), -200.0);
            assert!((back - db).abs() < 1e-9, "round trip {db} -> {back}");
        }
    }

    #[test]
    fn linear_to_db_floor_bounds_and_non_finite() {
        assert_eq!(linear_to_db_floor(1.0, -120.0), 0.0);
        // A value below the floor clamps up to it.
        assert_eq!(linear_to_db_floor(1e-30, -120.0), -120.0);
        // Non-positive and non-finite x map to the floor.
        assert_eq!(linear_to_db_floor(0.0, -120.0), -120.0);
        assert_eq!(linear_to_db_floor(-1.0, -120.0), -120.0);
        assert_eq!(linear_to_db_floor(f64::NAN, -120.0), -120.0);
        assert_eq!(linear_to_db_floor(f64::INFINITY, -120.0), -120.0);
        // A -inf floor means "no floor".
        assert_eq!(
            linear_to_db_floor(0.0, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert!(linear_to_db_floor(1.0, f64::NEG_INFINITY).abs() < 1e-12);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "floor_db must be finite")]
    fn positive_infinite_floor_panics_in_debug() {
        let _ = linear_to_db_floor(1.0, f64::INFINITY);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "floor_db must be finite")]
    fn nan_floor_panics_in_debug() {
        let _ = linear_to_db_floor(1.0, f64::NAN);
    }
}
