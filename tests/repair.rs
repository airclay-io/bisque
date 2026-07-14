// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Repair domain suite: declares the processor test modules.

#![cfg(all(feature = "repair", feature = "test-support"))]

mod repair {
    mod dc_blocker;
    mod dc_offset;
}
