<!-- SPDX-License-Identifier: Apache-2.0 -->

# Snapshots

Snapshots are committed-output regression tests. A snapshot means the output is
expected to match byte for byte across every supported CI operating system.

There are no platform snapshots. Output that is not promised cross-platform
exact belongs in behavior and contract tests.

Snapshots detect byte changes. Independent `audio`, `contract`, and
`validation` tests prove the output behavior, and every snapshotted processor
also needs them.

## Current State

Snapshot files live under `testdata/snapshots.manifest` and
`testdata/snapshots/slices/`. Regenerate them with
`cargo xtask gen-snapshots --reason "<why>"`.

The manifest records FNV-1a-128 hashes over a canonical planar byte format.
`f32-le-planar-v1`.

## Snapshot Cases

Concrete snapshot cases live in `src/testing/snapshot_cases.rs` behind the
`snapshot-support` feature. `xtask` and `tests/snapshots.rs` both use that
registry.

Current cases cover:

- Biquad low-pass and high-pass
- DcBlocker
- DcOffset
- Dither
- Gain
- Scale
- Limiter
- TimeStretch

Spectral processors do not currently have committed snapshots because the
`realfft` path has not been declared cross-platform byte-exact.

## Regenerating

Regenerate committed output.

```sh
cargo xtask gen-snapshots --reason "<why>"
```

The `--reason` is required so a regeneration is always deliberate. State it in
the change description; it is not written to the manifest, which carries only the
format meta (version, hash, canon, libm pin) and the case hashes.

Check that generated output matches committed `testdata/`.

```sh
cargo xtask check
```

The freshness check regenerates snapshots and reports any `testdata/` change.
The manifest carries no timestamp or reason, so a clean tree regenerates
identically and stays clean.

## Regeneration Policy

Regenerate snapshots only when output bytes intentionally change. A snapshot
diff means output bytes changed. When the change is intended, regenerate with a
`--reason` that says why and call it out in the change description. When the
byte change is accidental, repair the implementation. CI flags any manifest diff
for explicit sign-off.

## Adding A Case

Add a snapshot only when the output is intended to be byte-exact across supported
platforms.

1. Add the case to `src/testing/snapshot_cases.rs`.
2. Use deterministic input signals and deterministic settings.
3. Verify output on supported CI operating systems.
4. Run `cargo xtask gen-snapshots --reason "<why>"`.
5. Commit the manifest and slice changes under `testdata/`.

Snapshots sit alongside audio, contract, and validation tests.
