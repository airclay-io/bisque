<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Airclay LLC -->

# Contributing to bisque

bisque is a single Rust crate for contract-tested audio DSP. Contributions
follow these design goals.

- one published library crate named `bisque`
- canonical `processor`, `parameter`, and `host` contract modules
- one public DSP utility namespace at `bisque::dsp`
- feature-gated domain modules for processors and utilities
- tests organized around behavior that users and hosts can rely on
- committed snapshots only for output that is byte-exact across supported
  platforms

## Getting Started

```sh
git clone https://github.com/airclay-io/bisque
cd bisque
cargo test --locked --workspace --all-features
```

The toolchain is stable Rust; the MSRV is 1.85 and CI checks it. Local
verification commands are under [Local Checks](#local-checks).

The mdBook under `docs/src/` is normative for the processing contract. Read
`docs/src/core-contract.md` before changing contract behavior (latency, tails,
events, memory budgets), and update the book in the same change. Render it
locally with `mdbook build docs`.

## Issues And Recommendations

For bugfixes and new feature requests, please open an issue for discussion before attempting a PR.

Users are welcome to open issues, report problems, ask questions, and recommend changes. Issue discussion does not require a CLA. Airclay will review recommendations and may implement accepted changes with the project team.

Please keep issue recommendations as problem reports, use cases, or high-level
design suggestions unless a maintainer asks for implementation detail.
Meaningful outside code, tests, docs, examples, designs, and other
copyrightable contributions still require the CLA described below.

## Licensing

bisque's public source releases are licensed under Apache-2.0.

By contributing, you agree that your contribution is licensed under
Apache-2.0.

Meaningful outside contributions also require the Airclay LLC Contributor
License Agreement in `CLA.md` before merge. Contributors keep copyright in their
contributions, but the CLA grants Airclay LLC broad rights to use, reproduce,
modify, distribute, sublicense, and include contributions in other versions of bisque.

Meaningful contributions include code, tests, docs, examples, designs, and other
copyrightable material. Airclay LLC may waive the CLA requirement for trivial
fixes that do not add copyrightable material, such as spelling or formatting
corrections.

To accept the CLA, post this comment on your pull request from the GitHub
account that authored the commits:

> I have read the Airclay LLC Contributor License Agreement in CLA.md and I
> hereby accept it.

The comment is the recorded acceptance under `CLA.md` section 10, tied to
your GitHub account and timestamped by GitHub. Post it on each pull request:
every contribution then carries its own recorded acceptance, and a CI check
turns green once the comment is present.

Do not submit a contribution unless you have the right to grant the project
license and, when required, the CLA terms. If your employer or another entity may
own your contribution, get authorization before submitting it.

Every `.rs` and `.sh` source file starts with this header.

```rust
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC
```

Shell, TOML, YAML, and other comment styles use the same SPDX identifier where
a header is present. CI checks tracked `.rs` and `.sh` files with
`.github/scripts/check-headers.sh`.

## Repository Layout

The root package is the `bisque` library crate. The only workspace member is
`xtask`, which contains repository maintenance commands.

```text
src/
  lib.rs       # public modules and crate documentation
  contract/   # private implementation of public contract types
  dsp/        # public lower-level DSP utilities
  testing/    # shared test harness behind test-support
  filters/    # filter processors
  dynamics/   # dynamics processors
  mastering/  # mastering processors
  analysis/   # meters and analyzers
  generators/ # generators
  time/       # time-domain processors
  repair/     # repair processors
  spectral/   # optional FFT, STFT, windows, and spectral processors
tests/        # integration tests grouped by domain and proof category
testdata/     # committed snapshot manifest and slices
xtask/        # cargo xtask commands
```

Domain source files live under `src/<domain>/`. Re-export public types from each
domain root so users import `bisque::<domain>::Type`.

## Public API

Keep the public API predictable.

- Import processing contracts from `bisque::processor`, parameter types from
  `bisque::parameter`, and the optional single-processor helper from
  `bisque::host`.
- Keep `params!` at the crate root. Do not add duplicate root aliases or a
  prelude.
- Use `bisque::dsp` for lower-level utilities such as math, oversampling, and
  smoothing.
- Keep processors and utilities under their domain modules:
  `filters`, `dynamics`, `mastering`, `analysis`, `generators`, `time`,
  `repair`, and `spectral`.
- Re-export public domain types at the domain root. Prefer
  `bisque::filters::Biquad` over nested public implementation paths.
- Keep heavy or optional areas behind Cargo features. `spectral` is optional.
  `test-support` and `snapshot-support` are for tests and tooling.
- When adding, removing, or moving public API, update the relevant domain root
  docs, `docs/src/api-surface.md`, and `tests/api.rs` in the same change. Update
  `src/lib.rs` and the README feature map when the domain-level public shape
  changes.
- When adding public numeric helpers, document their domain and units
  (amplitude, power, dBFS, LUFS, and so on), their reference point, and their
  behavior for zero, negative, and non-finite inputs. If an existing public
  helper delegates to the new helper, treat any behavior change as public API
  and add regression tests.

## Versioning

Public API changes require maintainer approval. CI compares the API with the
latest release tag when one exists. Follow Semantic Versioning for released
APIs, describe breaking changes in the pull request, and update the version as
required.

## Feature Flags

Default features enable the current non-spectral domains:

- `filters`
- `dynamics`
- `mastering`
- `analysis`
- `generators`
- `time`
- `repair`

Additional features:

- `spectral` enables FFT, STFT, windows, and spectral processors.
- `test-support` exposes the shared contract-test harness as `bisque::testing`.
- `snapshot-support` enables the concrete snapshot registry used by tests and
  `xtask`.

When adding code behind a feature, gate both the source module and its
integration tests. Domain integration test roots use the matching feature and
`test-support` when they use the shared harness.

## Processor Design

Choose the smallest trait that matches the processor.

- Use `Kernel` for same-rate effects that process samples in place or split I/O.
  Wrap it with `into_processor()` when a `Processor` is needed.
- Sources (oscillators, noise) also implement `Kernel`: they ignore the input
  and overwrite the output.
- Use `Measurer` for meters and analyzers that observe read-only audio.
- Use `VariableRate` for processors whose output frame count can differ from the
  input frame count.
- Implement `Processor` directly for block-level behavior, sidechain routing,
  flushing, or event handling that needs a custom implementation.

A complete processor:

- implement over `T: Sample` unless a backend imposes a documented sample-type
  restriction; built-in same-rate processors are expected to support `f32` and
  `f64`
- validate sample rate, channel count, block size, settings, and memory budgets
  in `prepare`
- compute allocation layouts with checked arithmetic, enforce `max_memory`
  before allocating, and leave the processor unprepared after a failed
  `prepare`
- allocate any required state in `prepare`, not in the audio path
- keep `process`, `render`, `observe`, `flush`, and variable-rate processing
  allocation-free after preparation
- report `io_mode`, `latency`, `tail`, `memory_footprint`, `param_info`, and
  `sidechain_inputs` accurately
- declare a tail when output continues after input ends, and test the drain.
  Flushing should produce the same bytes as processing silence, write at most
  `out.frames()` per call, and start a fresh drain after new input or `reset`
  (any total cap on a drain belongs to the host)
- define parameter ranges and units through `ParamInfo`
- clamp or reject invalid parameter values consistently with the processor's
  documented behavior
- validate fixed construction-time values in `prepare` when those values can
  affect output finiteness. Runtime non-finite audio samples are sanitized at DSP
  boundaries; author-provided constants and settings should be rejected or
  explicitly documented if they intentionally map to a finite behavior
- make `reset` return the processor to its prepared initial state
- make block splits invariant and preserve the documented control-grid
  quantization of parameter events
- flush denormal-sized state where the processor can produce subnormal values
- use `bisque::dsp::math` for all transcendental math in production source.
  Bare standard-library transcendental calls have no production allowlist, FMA
  contraction and fast-math are rejected everywhere under `src/`, and
  `src/dsp/math.rs` is the only file that may call libm directly
- use the crate-internal `dsp::rng` helpers for any production randomness. No
  ambient entropy anywhere; tests use fixed seeds
- in `#[cfg(test)]` code, a bare std transcendental is allowed only for an
  independent reference calculation and must be tagged `// test-oracle:` on the
  same line. CI enforces both rules with `cargo xtask check-determinism`, which
  also rejects a `test-oracle:` tag outside a trailing `#[cfg(test)] mod tests`
  module

The audio path is expected to be infallible. Validation belongs in `prepare` and
constructor or settings APIs.

### Structural Validation Policy

One policy governs how invalid configuration is handled across the public API:

- Processor and settings structural configuration (tap counts, hops, sizes,
  initial parameter values) is stored as constructed and validated in
  `prepare`; invalid values return a `DspError` rather than being silently
  changed.
- Runtime automatable values are clamped to their declared `ParamInfo` ranges
  by the smoother bank.
- Standalone low-level utilities (constructed and used without a `prepare`
  step) either document a clear `# Panics` precondition or provide a fallible
  constructor when invalid input is plausible at runtime.
- A convenience constructor may clamp only when the method name or its rustdoc
  makes that behavior unmistakable, and the processor catalog must disclose
  every intentional clamp.

Host geometry (channel count, block size, I/O shape, sidechain buses) is a
host precondition, not a validation concern: the shared
`KernelProcessor`/driver machinery debug-asserts it before kernel indexing,
and release builds do not pay for the checks. Do not make `process` return a
`Result` for programmer errors.

Meters should keep `Measurer::Reading` cheap to return by value. If a meter also
needs richer data, such as per-channel readouts, expose it through inherent
non-allocating accessors and document channel bounds, unprepared/reset behavior,
and the exact denominator or weighting used by both the pooled trait reading and
the per-channel reading.

### Adding Runtime DSP

This workflow applies to processors, sources, meters, and variable-rate
processors. Read [Authoring Processors](docs/src/authoring-processors.md) for
the trait contracts and implementation patterns.

#### Implement And Enroll

1. Implement the type in its domain module and re-export it from the domain
   root. Same-rate built-ins should support both `f32` and `f64` through
   `T: Sample` unless the algorithm has a documented restriction.
2. Validate settings and calculate checked memory layouts before allocating in
   `prepare`. A rejected preparation must leave the instance unprepared.
3. Add feature-gated entries to `src/testing/registry.rs`. Use
   `ProcessorEntry`, `MeterEntry`, or `VariableRateEntry` as appropriate. Add
   one entry for each materially different runtime path that the shared suites
   must drive. Set each `ProcessorEntry` authoring mode to `Kernel` or `Direct`.
   Examples include main-input and sidechain modes or distinct filter shapes
   with different setup behavior.
4. Add a domain test file under `tests/<domain>/` with the standard `audio`,
   `contract`, and `validation` modules. Add `snapshots` only when exact output
   is portable across supported platforms.

Registry enrollment supplies the following shared checks.

| Entry | Shared coverage |
| --- | --- |
| `ProcessorEntry` | Prepare and metadata contracts, I/O declarations, block-size invariance, reset equivalence, tail and latency invariants, memory-footprint and budget checks, allocation-free processing and flushing, public inventory checks, and a listening smoke render |
| `MeterEntry` | Prepare and footprint contracts, geometry, block-size invariance, reset and reprepare behavior, memory budgets, allocation-free observation, and public inventory checks |
| `VariableRateEntry` | Prepare and footprint contracts, stretch invariance, memory budgets, allocation-free processing, public inventory checks, and a listening smoke render |

The registry uses representative settings. It does not prove the algorithm,
all operating modes, or flush semantics. Domain tests provide that evidence;
the registry automatically checks that declared flush paths do not allocate.

#### Prove The Behavior

Every new runtime type needs focused tests for these behaviors when they
apply.

- Check expected output with an independent calculation or a clearly justified
  signal property.
- Check invalid settings, invalid process specifications, non-finite input,
  exact memory budgets, and one-byte-too-small memory budgets.
- Check that state resets completely and failed preparation does not leave a
  usable partial state.
- Check both `f32` and `f64` for same-rate built-ins.
- Check parameter metadata, immediate writes, timestamped events, clamping,
  smoothing, and unknown parameter identifiers for parameterized processors.
- Check output overwrite behavior and split I/O behavior for the declared I/O
  mode.
- Check output-only geometry and complete output writes for sources.
- Check each sidechain bus independently, including channel routing and the
  absence of signal leakage between buses.
- Check meter readings, per-channel access, window boundaries, reset,
  reprepare, and invalid geometry for meters.
- Check input consumption, output capacity, starvation, duration, terminal
  completion, and reset for variable-rate processors.

Latency-bearing or tail-producing processors also need focused flush tests.
Check the exact latency and tail declarations, equivalence between flush and
processing silence, output-capacity bounds, early completion, terminal
completion, restart after new input, restart after reset, and allocation-free
flush calls. Add a composed flush test when latency or tails interact with
another processor in a chain.

Use seeded property tests in `tests/property.rs` when correctness depends on
many block splits or event schedules. Add a Criterion case to
`benches/processors.rs` when the implementation introduces a new algorithmic
cost, buffering strategy, or processing path. Add a curated case to
`xtask/src/listen.rs` when an audible defect is easier to recognize than a
numeric failure. Add an end-to-end test or example when the type exists to
support a documented composed workflow.

#### Publish The Surface

Complete the public surface in the same change.

- Add the type to the domain rustdoc map and to `docs/src/api-surface.md`.
- Add a one-crate import and basic construction check to `tests/api.rs`.
- Add the runtime inventory row, parameter metadata, and behavioral notes to
  `docs/src/processor-catalog.md`.
- Add a concise entry under `[Unreleased]` in `CHANGELOG.md`.
- Update the README feature map when the domain-level feature surface changes.
- Set `required-features` for examples and benchmarks that depend on optional
  domains.
- Regenerate snapshots and their manifest when snapshot coverage is used.

`tests/documentation.rs` compares the registry with the structured API and
processor catalog inventories. Explanatory notes and examples still require
review for accuracy.

### Adding A Domain

A new DSP domain crosses the crate, tests, documentation, and listening tools.
Complete all of these steps together.

1. Add the Cargo feature and decide explicitly whether it belongs in the
   default feature set.
2. Add the feature-gated module and `doc(cfg(...))` annotation in `src/lib.rs`,
   then create the domain root with its public rustdoc map and re-exports.
3. Add feature-gated registry dispatch and teach the registry documentation
   parsers about the feature.
4. Add the domain integration-test root and its feature gates.
5. Add the domain to the README feature map, API surface, processor catalog,
   and crate-level rustdoc when it changes the top-level map.
6. Forward the feature through `xtask/Cargo.toml` so listening verification can
   see the full catalog.
7. Add `required-features` to domain-specific examples and benchmarks.

### Adding A Dependency

Prefer an existing dependency when it already provides the required behavior.
Keep large or specialized dependencies behind the feature that needs them.
After changing dependencies, update `Cargo.lock`, run `cargo deny check`, run
`cargo machete`, and regenerate `THIRD-PARTY-LICENSES.md` with `cargo-about`
0.9.0.

```sh
cargo about generate --locked --all-features --fail \
  -c about.toml about.hbs \
  -o THIRD-PARTY-LICENSES.md \
  --manifest-path Cargo.toml
```

Review any new license before committing the generated notice. Examples,
benchmarks, and tools must declare or forward the features required by the
dependency.

## Test Organization

Tests are easiest to read by domain first and proof category second.

Root integration-test files under `tests/` are the test binaries. Domain roots
load submodules from same-named directories:

```rust
#![cfg(all(feature = "filters", feature = "test-support"))]

mod filters {
    mod biquad;
    mod moving_average;
}
```

Use the same inner module names across processor test files:

```rust
mod audio {
    // DSP behavior and independent math checks.
}

mod contract {
    // Host and lifecycle guarantees.
}

mod validation {
    // Bad specs, bad settings, bad parameters, and memory budgets.
}

mod snapshots {
    // Optional cross-platform byte-exact output checks.
}
```

Use these categories consistently.

- `audio` covers transfer functions, curves, reconstruction, thresholds,
  quantization, latency behavior heard in the signal, and independent reference
  math.
- `contract` covers block-size invariance, reset equivalence, exact latency and
  tail declarations, flush shape, sidechain routing, split I/O, and
  memory-footprint reporting.
- `validation` covers rejected specs, invalid settings, unsupported modes,
  parameter range behavior, and memory budgets.
- `snapshots` covers committed byte-exact output only.

A snapshot is cross-platform exact by definition.

Tests encode the documented behavior. When an implementation change makes a
ceiling, transparency, or invariance test fail, repair the implementation. Do
not loosen the test to fit the code.

When a new primitive exists to support a documented composed workflow, test the
workflow end to end as well as testing the parts. The end-to-end test should
catch sign conventions, channel ordering, aggregation denominators, and feature
gates that the individual processor or meter tests can miss.

Global test binaries have narrower purposes:

- `tests/api.rs` checks the intended one-crate import paths.
- `tests/registry_contract.rs` drives every `src/testing/registry.rs` entry
  through the shared lifecycle contracts.
- `tests/documentation.rs` keeps the structured API surface and processor
  catalog tables aligned with the registry.
- `tests/property.rs` checks seeded property-based invariance: random block
  splits and event schedules must be bit-identical to the whole-block
  reference.
- `tests/no_alloc.rs` checks allocation behavior for non-spectral audio paths
  by iterating the registry.
- `tests/spectral_no_alloc.rs` checks allocation behavior for spectral audio
  paths by iterating the registry's spectral entries.
- `tests/snapshots.rs` checks the committed snapshot manifest.

No-allocation tests use a custom global allocator with a per-thread counter:
the no-allocation property belongs to the thread driving the audio path, and
libtest's own main thread allocates for its bookkeeping concurrently with the
measured region. Keep every armed region in one test function on one thread,
and keep those tests isolated from normal audio, contract, and validation
tests. The Testing chapter explains the counter and its failure diagnostics.

Randomized tests use seeded randomness only. Fix the RNG seed, disable
persistence files, and keep runs byte-for-byte reproducible (see
`tests/property.rs`).

## Snapshots

Snapshots are committed-output regression tests for processors whose output is
expected to match byte for byte across supported CI operating systems.

Snapshot files live under `testdata/snapshots.manifest` and
`testdata/snapshots/slices/`. Regenerate them with
`cargo xtask gen-snapshots --reason "<why>"`. The reason is required so a
regeneration is always deliberate; state it in the change description. It is not
written to the manifest, which carries only the format meta and the case hashes.
`cargo xtask check` verifies the case rows, so a clean tree stays clean.

When adding a snapshot:

1. Add the case to `src/testing/snapshot_cases.rs`.
2. Use deterministic input signals and deterministic processor settings.
3. Prove that the output is byte-exact on supported CI operating systems before
   committing the case.
4. Run `cargo xtask gen-snapshots --reason "<why>"`.
5. Commit the manifest and slice changes under `testdata/`.
6. Update `docs/src/snapshots.md` if its current-case list changes.

Do not add platform-specific expected output. Do not add spectral snapshots until
the FFT path is proven byte-exact across supported platforms or the backend is
changed to make that guarantee explicit.

Snapshots do not replace behavior tests. Every snapshotted processor also needs
audio, contract, and validation coverage where applicable.

## Shared Test Harness

The shared harness lives in `src/testing` behind the `test-support` feature. It
centralizes contract checks for integration tests.

The most used helpers are `Contract::run` (with `_reusing`, `_split`, and
`_with_sidechain` variants), the block-size-invariance and reset-equivalence
asserts, and the `sine`, `ev`, and `bits_eq` utilities. The Testing chapter of
the mdBook (`docs/src/testing.md`) keeps the full helper list.

The supported processor registry at `bisque::testing::registry` enumerates every
built-in processor, meter, and rate changer for the cross-cutting suites (see
Adding Runtime DSP). Downstream suites may consume that built-in catalog;
downstream processors use `Contract` directly rather than inserting entries into
bisque's registry. The generated snapshot-case registry behind
`snapshot-support` contains repository-only generated cases hidden from rustdoc.

## Documentation

Documentation is concise, factual, and useful in generated rustdoc.

- Document what public types, functions, parameters, and return values mean.
- Document errors, panics, allocation behavior, latency, tail behavior, I/O mode,
  sidechain behavior, and parameter units where they matter.
- README, mdBook, rustdoc, and code comments describe only the code in the tree.
  Put release history in `CHANGELOG.md` and design history in commits.
- Avoid decorative formatting in comments. Use lists and headings only when they
  improve scanning.
- Keep documented imports aligned with the public modules.
- Keep the API surface map at `docs/src/api-surface.md` aligned with rustdoc and
  domain root exports.
- Update `README.md`, `docs/`, examples, and this file when a public API or
  workflow changes.

Rustdoc warnings are denied in CI.

## Local Checks

Run the focused commands for the area you changed, then run the core local
checks before opening or updating a pull request.

Optionally install the Git hooks (managed by [lefthook](https://lefthook.dev))
to run the cheap checks automatically. The pre-commit hook also requires the
same `cargo-machete` version used by CI.

```sh
cargo install cargo-machete --locked --version 0.9.2
lefthook install
```

`pre-commit` runs formatting, the SPDX header check, `cargo machete`, and
opt-in `typos`.
`pre-push` runs clippy, the test suite, `cargo doc`, and
`cargo xtask check-determinism`. The feature powerset, mdBook, and snapshot
checks stay in CI. Mutation sweeps run only when manually triggered. CI remains
the authoritative gate; hooks are bypassable with `--no-verify`, and personal
overrides go in the gitignored `lefthook-local.yml`.

```sh
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps --all-features
cargo xtask check-docs
cargo xtask check-determinism
```

When docs change, also render the book and run the docs typo check:

```sh
mdbook build docs
cargo xtask check-docs
cargo test --locked --all-features --test documentation
typos docs
```

Run proof-category filters when changing tests or processor behavior:

```sh
cargo test --locked --workspace --all-features -- audio::
cargo test --locked --workspace --all-features -- contract::
cargo test --locked --workspace --all-features -- validation::
cargo test --locked --workspace --all-features -- no_alloc::
cargo test --locked --workspace --all-features -- snapshots::
```

Run snapshot tooling when output bytes or snapshot cases change:

```sh
cargo xtask gen-snapshots --reason "<why>"
cargo xtask check
```

Render the listening bench when changing DSP behavior, and listen to the
affected pairs (`target/listen/index.md` says what to listen for in each
case). The bench helps find audible problems. The numeric suites decide
pass/fail, and any audible finding gets a numeric test.

```sh
cargo xtask listen
```

Run the Criterion suite after adding or changing a benchmark. Benchmark results
record trends and do not impose a merge threshold.

```sh
cargo bench --bench processors --all-features
```

Run the feature powerset when changing feature gates or public module paths.
This is the same command CI runs (install with
`cargo install cargo-hack --locked`), and the `-D warnings` matters: dead-code
warnings that only appear in reduced feature sets are errors in CI:

```sh
RUSTFLAGS="-D warnings" cargo hack --locked --workspace --feature-powerset check
```

The authoritative checks are defined in `.github/workflows/ci.yml`. The
aggregate `required` job is the branch-protection gate. CI also checks package
contents, dependency policy, third-party notices, source headers, workflow
files, the MSRV, and Linux coverage. It retains LCOV and listening-bench
artifacts. A manually triggered workflow runs the full mutation sweep.

## Mutation Testing

Mutation testing runs through the manually triggered full sweep. The workflow
fails when a mutant survives the test suite. For a smaller local analysis of
changed code, install `cargo-mutants` with
`cargo install cargo-mutants --locked` and run:

```sh
git diff origin/main...HEAD > mutants.diff
cargo mutants --workspace --no-shuffle --in-diff mutants.diff --all-features
```

`--all-features` is required so spectral and the other feature-gated modules
are mutated. It is passed on the command line, not in `.cargo/mutants.toml`;
supplying it in both places forwards the flag twice and cargo rejects it.

A missed mutant usually means the tests need strengthening. If a mutant is
genuinely equivalent (the mutation cannot change any
observable behavior) or is unobservable through the public contract, document
it as a scoped `exclude_re` entry in `.cargo/mutants.toml` with the reasoning
next to the existing entries. Never weaken a test to make a mutant pass.

## Pull Request Checklist

Before a pull request is ready:

- the CLA requirement is satisfied when applicable
- the public import path is consistent with the one-crate API
- the changed code is behind the correct feature flags
- each materially different runtime path has a registry entry
- new runtime DSP has focused audio, contract, and validation tests where
  applicable
- latency and tail behavior has focused flush and allocation tests
- sources, sidechains, meters, and variable-rate processors have their
  trait-specific tests
- snapshots are added only for cross-platform byte-exact output
- snapshot files are regenerated only when output bytes intentionally change
- rustdoc explains user-visible behavior and constraints
- API surface and processor catalog tables are updated when public exports or
  parameter metadata change
- `[Unreleased]` in `CHANGELOG.md` describes user-visible changes
- benchmarks, property tests, listening cases, and composed tests were added
  where the change calls for them
- dependency policy and third-party notices are current when dependencies
  change
- examples, README, docs, and this guide are updated when workflows or APIs
  change
- local checks relevant to the change have passed
