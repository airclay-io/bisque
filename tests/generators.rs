// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Generators domain suite: declares the processor test modules.

#![cfg(all(feature = "generators", feature = "test-support"))]

mod generators {
    mod processors;
}
