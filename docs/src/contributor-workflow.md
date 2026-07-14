<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributor Workflow

The full contributor guide is in `CONTRIBUTING.md`. This page summarizes the
workflow that affects code and docs.

## Before Changing Code

Check the intended public path first.

- Host and authoring contracts belong in their canonical `processor`,
  `parameter`, or `host` module; do not add root aliases or a prelude.
- Lower-level DSP utilities belong under `bisque::dsp`.
- Processors and utilities belong under their domain module.
- Heavy optional code is feature-gated.
- New behavior needs tests in the matching proof categories.

Public API changes follow the Versioning policy in `CONTRIBUTING.md`.

## Contributor License

Meaningful outside contributions require the Airclay LLC Contributor License
Agreement in `CLA.md` before merge. Contributors keep copyright, but the CLA
grants Airclay LLC rights to use, modify, distribute, sublicense, and include the
contribution in open-source, proprietary, and closed-source versions of bisque.

Airclay LLC may waive the CLA requirement for trivial corrections that do not
add copyrightable material.

## Processor Checklist

A processor is ready when it does this work.

- validates unsupported specs and memory budgets in `prepare`
- allocates required state in `prepare`
- keeps the audio path allocation-free after preparation
- reports I/O mode, latency, tail, parameters, sidechain count, and memory
  footprint accurately
- declares a tail when output continues after input ends and tests the drain
- handles reset equivalence
- uses deterministic math in deterministic paths
- has `audio`, `contract`, and `validation` tests where applicable
- has no-allocation coverage for audio-path behavior
- uses snapshots only for cross-platform byte-exact output
- documents user-visible behavior and constraints

## Documentation

Keep docs factual and current.

- Update rustdoc when public types, functions, parameters, errors, panics,
  latency, tail, or allocation behavior changes.
- Update the API surface and processor catalog pages when processors or public
  exports change. `tests/documentation.rs` checks their structured inventories.
  Review explanatory text by hand.
- Update this book when user workflows, architecture, testing, or snapshot policy
  changes.
- Update `README.md` when the public entry point changes.
- Update `CONTRIBUTING.md` when contributor workflow changes.

Rustdoc warnings are denied in CI.

## Pre-Flight

Useful local checks.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo xtask check-docs
cargo xtask check-determinism
```

When snapshot output intentionally changes.

```sh
cargo xtask gen-snapshots --reason "<why>"
cargo xtask check
```

When docs change.

```sh
mdbook build docs
cargo xtask check-docs
cargo test --locked --all-features --test documentation
typos docs
```
