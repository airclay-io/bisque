// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Time domain suite: declares the processor test modules.

#![cfg(all(feature = "time", feature = "test-support"))]

mod time {
    mod delay;
    mod time_stretch;
}
