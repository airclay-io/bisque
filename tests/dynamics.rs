// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Dynamics domain suite: declares the processor test modules.

#![cfg(all(feature = "dynamics", feature = "test-support"))]

mod dynamics {
    mod processors;
}
