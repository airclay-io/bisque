// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Sample sanitization helpers used at DSP boundaries.

/// Recursive state values below this magnitude are flushed to zero.
///
/// Only processors with recursive state call `flush_denormal`, so builds
/// without any of those domain features compile it without a caller.
#[cfg_attr(
    not(any(
        feature = "analysis",
        feature = "filters",
        feature = "mastering",
        feature = "repair",
        feature = "time"
    )),
    allow(dead_code)
)]
pub(crate) const DENORMAL_FLOOR: f64 = 1e-30;

/// Return `x` if it is finite, otherwise silence.
#[inline]
pub(crate) fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Return a finite recursive-state value, flushing denormals to zero.
#[cfg_attr(
    not(any(
        feature = "analysis",
        feature = "filters",
        feature = "mastering",
        feature = "repair",
        feature = "time"
    )),
    allow(dead_code)
)]
#[inline]
pub(crate) fn flush_denormal(x: f64) -> f64 {
    let x = finite_or_zero(x);
    if x.abs() < DENORMAL_FLOOR {
        0.0
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::{finite_or_zero, flush_denormal, DENORMAL_FLOOR};

    #[test]
    fn finite_or_zero_passes_finite_values_only() {
        assert_eq!(finite_or_zero(0.5), 0.5);
        assert_eq!(finite_or_zero(f64::NAN), 0.0);
        assert_eq!(finite_or_zero(f64::INFINITY), 0.0);
        assert_eq!(finite_or_zero(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn flush_denormal_zeroes_tiny_and_non_finite_values() {
        assert_eq!(flush_denormal(1e-40), 0.0);
        assert_eq!(flush_denormal(-1e-40), 0.0);
        assert_eq!(flush_denormal(f64::NAN), 0.0);
        assert_eq!(flush_denormal(f64::INFINITY), 0.0);
        assert_eq!(flush_denormal(DENORMAL_FLOOR), DENORMAL_FLOOR);
        assert_eq!(flush_denormal(0.5), 0.5);
    }
}
