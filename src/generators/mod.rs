// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Output-only generators driven by parameters.
//!
//! Each generator implements [`Kernel`](crate::processor::Kernel) and declares
//! output-only I/O. Wrap one with
//! [`Kernel::into_processor`](crate::processor::Kernel::into_processor) for framework
//! parameter smoothing and block driving.
//!
//! # Public API
//!
//! - [`SineOsc`](crate::generators::SineOsc) and
//!   [`SineOscSettings`](crate::generators::SineOscSettings) generate a sine
//!   tone, with smoothed values in
//!   [`SineOscParams`](crate::generators::SineOscParams).
//! - [`WhiteNoise`](crate::generators::WhiteNoise) and
//!   [`WhiteNoiseSettings`](crate::generators::WhiteNoiseSettings) generate
//!   seeded white noise, with smoothed values in
//!   [`WhiteNoiseParams`](crate::generators::WhiteNoiseParams).
//! - [`PolyBlepOsc`](crate::generators::PolyBlepOsc),
//!   [`PolyBlepOscSettings`](crate::generators::PolyBlepOscSettings), and
//!   [`Waveform`](crate::generators::Waveform) generate PolyBLEP saw and square
//!   waves with reduced aliasing. Smoothed values are available in
//!   [`PolyBlepOscParams`](crate::generators::PolyBlepOscParams).

/// Keep oscillator phase increments strictly below Nyquist. This matches the
/// guarded cutoff ceiling used by the filter catalog.
const NYQUIST_RATIO: f64 = 0.999;

fn max_oscillator_frequency(sample_rate: f64) -> f64 {
    sample_rate * 0.5 * NYQUIST_RATIO
}

mod noise;
mod poly_blep;
mod sine;

pub use noise::{WhiteNoise, WhiteNoiseParams, WhiteNoiseSettings};
pub use poly_blep::{PolyBlepOsc, PolyBlepOscParams, PolyBlepOscSettings, Waveform};
pub use sine::{SineOsc, SineOscParams, SineOscSettings};

#[cfg(test)]
mod tests {
    use super::noise::{WhiteNoise, WhiteNoiseSettings, DEFAULT_NOISE_SEED};
    use super::poly_blep::poly_blep;
    use super::sine::wrap_phase;
    use crate::dsp::rng::Rng;
    use crate::processor::{Kernel, ProcessSpec};
    use std::f64::consts::TAU;

    /// A stereo 48 kHz spec.
    fn spec() -> ProcessSpec {
        ProcessSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block: 8192,
            max_memory: None,
        }
    }

    #[test]
    fn white_noise_footprint_is_one_rng_per_channel() {
        let mut w = WhiteNoise::new();
        Kernel::<f32>::prepare(&mut w, spec()).expect("prepare");
        let expected = 2 * std::mem::size_of::<Rng>();
        assert_eq!(
            Kernel::<f32>::memory_footprint(&w),
            expected,
            "footprint is exactly two RNGs ({expected} bytes), not a constant or sum"
        );
        // A third channel scales the footprint.
        let mut three = WhiteNoise::new();
        let spec3 = ProcessSpec {
            channels: 3,
            ..spec()
        };
        Kernel::<f32>::prepare(&mut three, spec3).expect("prepare");
        assert_eq!(
            Kernel::<f32>::memory_footprint(&three),
            3 * std::mem::size_of::<Rng>()
        );
    }

    #[test]
    fn white_noise_exposes_its_amplitude_param() {
        // White noise exposes only amplitude.
        let w = WhiteNoise::new();
        let params = Kernel::<f32>::param_info(&w);
        assert_eq!(params.len(), 1, "white noise declares one parameter");
        assert_eq!(
            params[0].id,
            WhiteNoise::AMPLITUDE,
            "and it is the amplitude"
        );
    }

    #[test]
    fn white_noise_reset_reseeds_every_channel() {
        // Reset re-derives each channel seed.
        let mut w = WhiteNoise::with_settings(
            WhiteNoiseSettings::new()
                .amplitude(0.5)
                .seed(DEFAULT_NOISE_SEED),
        );
        Kernel::<f32>::prepare(&mut w, spec()).expect("prepare");
        let fresh: Vec<u64> = w.rngs.iter().map(|r| r.state).collect();
        for r in &mut w.rngs {
            r.next_u64(); // advance both channels' state
            r.next_u64();
        }
        assert_ne!(
            w.rngs.iter().map(|r| r.state).collect::<Vec<_>>(),
            fresh,
            "the dirtying must move the state"
        );
        Kernel::<f32>::reset(&mut w);
        assert_eq!(
            w.rngs.iter().map(|r| r.state).collect::<Vec<_>>(),
            fresh,
            "reset returns every channel to its post-prepare seed"
        );
    }

    #[test]
    fn wrap_phase_folds_only_above_tau() {
        // Values below TAU are unchanged.
        let below = TAU - 0.1;
        assert_eq!(wrap_phase(below), below, "no fold below TAU");
        // Values at or above TAU subtract TAU once.
        assert_eq!(wrap_phase(TAU), 0.0, "TAU folds exactly to 0");
        let above = TAU + 0.25;
        assert!(
            (wrap_phase(above) - 0.25).abs() < 1e-12,
            "TAU + 0.25 folds to 0.25, got {}",
            wrap_phase(above)
        );
    }

    #[test]
    fn poly_blep_residual_matches_the_quadratic() {
        // A non-zero phase increment.
        let dt = 0.02;
        // Just after the wrap, x = 0.5 and the residual is -0.25.
        assert!(
            (poly_blep(dt / 2.0, dt) - (-0.25)).abs() < 1e-12,
            "rising residual at x=0.5 is -0.25, got {}",
            poly_blep(dt / 2.0, dt)
        );
        // Just before the wrap, x = -0.5 and the residual is 0.25.
        assert!(
            (poly_blep(1.0 - dt / 2.0, dt) - 0.25).abs() < 1e-9,
            "falling residual at x=-0.5 is 0.25, got {}",
            poly_blep(1.0 - dt / 2.0, dt)
        );
        // Away from a discontinuity the correction is zero.
        assert_eq!(
            poly_blep(0.5, dt),
            0.0,
            "no residual away from a discontinuity"
        );
    }
}
