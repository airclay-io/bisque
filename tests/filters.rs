// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Filters domain suite: declares the processor test modules.

#![cfg(all(feature = "filters", feature = "test-support"))]

mod filters {
    mod biquad;
    mod moving_average;
}
