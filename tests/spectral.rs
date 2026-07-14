// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Spectral domain suite: declares the FFT, STFT, and filter test modules.

#![cfg(feature = "spectral")]

mod spectral {
    mod fft;
    #[cfg(feature = "test-support")]
    mod spectral_filter;
    mod stft;
    mod window;
}
