// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Spectral processing: windows, FFT/STFT helpers, and streaming STFT processors.
//!
//! The FFT backend is provided by `realfft`. Spectral processors do not currently
//! have committed cross-platform snapshots.
//!
//! # Public API
//!
//! - [`Fft`](crate::spectral::Fft) and [`Complex`](crate::spectral::Complex)
//!   provide real-input FFT analysis and synthesis.
//! - [`Stft`](crate::spectral::Stft) performs offline STFT analysis and
//!   synthesis.
//! - [`SpectralFilter`](crate::spectral::SpectralFilter) and
//!   [`SpectralFilterSettings`](crate::spectral::SpectralFilterSettings) provide
//!   a streaming STFT band filter.
//! - [`Window`](crate::spectral::Window) and
//!   [`cola_sum`](crate::spectral::cola_sum) provide window generation and
//!   overlap checks.

pub mod fft;
pub mod spectral_filter;
pub mod stft;
pub mod window;

pub use fft::{Complex, Fft};
pub use spectral_filter::{SpectralFilter, SpectralFilterSettings};
pub use stft::Stft;
pub use window::{cola_sum, Window};
