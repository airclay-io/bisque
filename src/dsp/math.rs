// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Deterministic transcendental functions.
//!
//! These wrappers use the pinned `libm` dependency instead of platform `libm`.
//! Processors with committed cross-platform snapshots use this module for
//! coefficient and audio-path transcendental calculations.

/// Sine of `x` radians.
#[inline]
#[must_use]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine of `x` radians.
#[inline]
#[must_use]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Tangent of `x` radians.
#[inline]
#[must_use]
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}

/// Square root of `x`.
#[inline]
#[must_use]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// `sqrt(x*x + y*y)` without intermediate overflow.
#[inline]
#[must_use]
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

/// Four-quadrant arctangent of `y / x` in radians.
#[inline]
#[must_use]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// `e^x`.
#[inline]
#[must_use]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Natural logarithm of `x`.
#[inline]
#[must_use]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Base-10 logarithm of `x`.
#[inline]
#[must_use]
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// `x` raised to the power `y`.
#[inline]
#[must_use]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
