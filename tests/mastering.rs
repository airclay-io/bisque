// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Mastering domain suite: declares the processor test modules.

#![cfg(all(feature = "mastering", feature = "test-support"))]

mod mastering {
    mod dither;
    mod gain;
    #[cfg(feature = "analysis")]
    mod limiter;
    mod scale;
}
