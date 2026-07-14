// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! The [`Sample`] buffer-element trait.

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// The buffer element type.
///
/// Processors compute internally in concrete `f32` or `f64` types. `T` is buffer
/// storage, so the trait only provides sample conversion. Bisque supports
/// `f32` and `f64`; this trait is sealed so the audio-path conversion contract
/// cannot be replaced by a fallible downstream implementation.
pub trait Sample: private::Sealed + Copy + Send + 'static {
    /// Convert from an `f32` sample.
    fn from_f32(x: f32) -> Self;
    /// Convert to an `f32` sample.
    fn to_f32(self) -> f32;
    /// Convert from an `f64` sample.
    fn from_f64(x: f64) -> Self;
    /// Convert to an `f64` sample.
    fn to_f64(self) -> f64;
}

impl Sample for f32 {
    fn from_f32(x: f32) -> Self {
        x
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Sample for f64 {
    fn from_f32(x: f32) -> Self {
        f64::from(x)
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Sample;

    // Distinct values exercise each conversion path.
    #[test]
    fn f32_conversions_round_trip_exactly() {
        assert_eq!(<f32 as Sample>::from_f32(0.5), 0.5);
        assert_eq!((-0.25f32).to_f32(), -0.25);
        assert_eq!(<f32 as Sample>::from_f64(2.5), 2.5f32);
        assert_eq!((0.75f32).to_f64(), 0.75f64);
    }

    #[test]
    fn f64_conversions_round_trip_exactly() {
        assert_eq!(<f64 as Sample>::from_f32(0.5f32), 0.5f64);
        assert_eq!((-0.25f64).to_f32(), -0.25f32);
        assert_eq!(<f64 as Sample>::from_f64(2.5), 2.5);
        assert_eq!((0.75f64).to_f64(), 0.75);
    }
}
