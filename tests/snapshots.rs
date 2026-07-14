// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Cross-platform snapshot regression tests.
//!
//! Each registered case is driven through the same path used by
//! `cargo xtask gen-snapshots`, and its FNV-1a-128 hash is compared to the
//! committed manifest value.

#![cfg(feature = "snapshot-support")]

mod snapshots {
    use bisque::testing::assert_snapshot;
    use bisque::testing::snapshot_cases::{
        drive_case, drive_vr_case, snapshot_cases, vr_snapshot_cases,
    };

    #[test]
    fn every_snapshot_matches_committed_manifest() {
        for c in snapshot_cases() {
            let out = drive_case(&c);
            assert_snapshot(c.id, &out);
        }
        for c in vr_snapshot_cases() {
            let out = drive_vr_case(&c);
            assert_snapshot(c.id, &out);
        }
    }
}
