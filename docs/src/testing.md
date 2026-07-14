<!-- SPDX-License-Identifier: Apache-2.0 -->

# Testing

The repository tests are organized for reading by domain and proof category.

Root files under `tests/` are integration-test binaries. Domain roots load
submodules from same-named directories.

```rust,ignore
#![cfg(all(feature = "filters", feature = "test-support"))]

mod filters {
    mod biquad;
    mod moving_average;
}
```

## Proof Categories

Processor test files use consistent inner module names.

| Module | Purpose |
| --- | --- |
| `audio` | DSP behavior, transfer functions, curves, reconstruction, thresholds, and independent math |
| `contract` | Host and lifecycle guarantees |
| `validation` | Rejected specs, invalid settings, unsupported modes, parameter ranges, and memory budgets |
| `snapshots` | Optional committed byte-exact output checks |

Use `audio`, `contract`, `validation`, `no_alloc`, and `snapshots` in test names
and module paths.

Behavior proof comes before snapshots. A processor gets independent `audio`,
`contract`, and `validation` coverage first. A snapshot records byte stability
after those pass and only when the output is cross-platform byte-exact (see
[Snapshots](snapshots.md)).

## The Processor Registry

`bisque::testing::registry` (implemented in `src/testing/registry.rs`) is the
supported catalog of built-ins for cross-cutting downstream and repository
suites. Within bisque, add an entry for each materially different runtime path.
Use `ProcessorEntry`, `MeterEntry`, or `VariableRateEntry` and feature-gate it
like its domain module. Enrollment provides:

- `tests/registry_contract.rs`: prepare on the standard spec, declared
  metadata consistency, block-size invariance, reset equivalence, and
  memory-footprint stability
- `tests/no_alloc.rs` and `tests/spectral_no_alloc.rs`, which iterate the
  registry for their armed processing and flush loops
- `tests/documentation.rs`, which checks `docs/src/api-surface.md` and the
  structured runtime and parameter tables in
  `docs/src/processor-catalog.md`
- listening smoke renders for processor and variable-rate entries

The documentation tests catch missing registry entries and stale structured
documentation. The registry uses representative settings and checks allocation-free
processing, observation, and declared flush paths. Processor-specific behavior,
operating modes, and flush semantics stay in the domain test files under
`tests/<domain>/`.

## Adding Runtime DSP

Use the narrowest authoring contract. Ordinary same-rate effects implement
`Kernel`; unusual block algorithms implement `Processor` directly. Meters,
pull sources, and rate changers use `Measurer`, `Source`, and `VariableRate`.

A new catalog processor must:

- follow the domain `Foo::new()` / `Foo::with_settings(FooSettings)` convention
  when it is configurable;
- declare typed parameters with `params!` and const-built `ParamInfo` metadata;
- validate settings and preflight its logical memory layout in `prepare`;
- make reset equivalent to the configured post-prepare state;
- report latency and tail, and drain delayed body before append-only tail;
- perform no allocations after prepare;
- add independent audio, contract, validation, exact-budget, and reset tests;
- enroll each materially different runtime path in `testing::registry` and
  update the public domain, API, and processor catalog documentation.

The registry suites verify each entry's lifecycle, block-size invariance,
latency/tail relationship, memory behavior, and allocation-free processing and
flushing.
Separate host-contract tests verify boxed delegation, `PreparedProcessor`
parity, and immediate-write forwarding with representative processors. A
direct `Processor` implementation that customizes those paths needs focused
coverage. Processor-specific DSP always needs an independent reference in its
domain tests.

The generated `testing::snapshot_cases` module is repository-only sharing
between integration tests and `xtask`. It is hidden from rustdoc behind
`snapshot-support`.

## Property-Based Invariance

`tests/property.rs` checks that random block splits and random sorted event
schedules produce output bit-identical to the whole-block reference.
Repository policy is seeded randomness only. Each test builds its own
proptest `TestRunner` from a fixed ChaCha seed with failure persistence
disabled, so runs are byte-for-byte reproducible on every platform and no
`proptest-regressions` files are written.

## Click Tests

Swept-parameter click tests (for example in `tests/mastering/gain.rs` and
`tests/filters/biquad.rs`) drive a full-range parameter sweep and assert that
no sample-to-sample output jump exceeds the analytic worst case of the
declared smoothing. They bound discontinuities and prove byte invariance; they
support click resistance without claiming to prove the absence of every
audible artifact.

## Shared Harness

`bisque::testing` is available behind the `test-support` feature. It provides
these helpers.

- `Contract::run`
- `Contract::run_reusing`
- `Contract::run_split`
- `Contract::run_with_sidechain`
- `Contract::generate`
- `Contract::stretch`
- `Contract::assert_block_size_invariant`
- `Contract::assert_generator_block_size_invariant`
- `Contract::assert_stretch_block_size_invariant`
- `Contract::assert_reset_equivalence`
- `sine`, `ev`, `observe_blocks`, and `bits_eq`
- `InfiniteTailKernel`, a synthetic `Tail::Infinite` test double for
  flush-contract tests

Use the harness for contract checks.

## No-Allocation Tests

No-allocation checks live in separate test binaries because they install a
custom global allocator.

- `tests/no_alloc.rs` covers non-spectral audio paths.
- `tests/spectral_no_alloc.rs` covers spectral audio paths.

Do not mix normal audio, contract, or validation tests into no-allocation test
binaries.

The counter is per-thread. The no-allocation property belongs to the thread
driving the audio path, and libtest's own main thread allocates for its
bookkeeping concurrently with the measured region. When the counter trips, the
report includes the allocation sizes, and the armed allocator prints their
backtraces to the test's captured stderr.

## Useful Commands

Run the full suite.

```sh
cargo test --workspace --all-features
```

Run proof-category filters.

```sh
cargo test --workspace --all-features -- audio::
cargo test --workspace --all-features -- contract::
cargo test --workspace --all-features -- validation::
cargo test --workspace --all-features -- no_alloc::
cargo test --workspace --all-features -- snapshots::
```

Run static checks.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo xtask check-docs
cargo xtask check-determinism
```

Generate the local coverage report that CI stores as an artifact.

```sh
cargo install cargo-llvm-cov --locked
cargo llvm-cov --locked --workspace --exclude xtask --all-features --lcov --output-path lcov.info
```

The determinism xtask rejects FMA contraction and fast-math anywhere under
`src/`, bare standard-library transcendentals in production code (route them
through `bisque::dsp::math`), and any `// test-oracle:` tag outside a trailing
`#[cfg(test)] mod tests` module.

Run snapshot tooling when output bytes or snapshot cases intentionally change.

```sh
cargo xtask gen-snapshots --reason "<why>"
cargo xtask check
```

The `--reason` is required so a regeneration is always deliberate; state it in
the change description. It is not written to the manifest (see
[Snapshots](snapshots.md)).

Run the feature powerset when changing feature gates. This is the command CI
runs, and `-D warnings` matters: dead code that only appears in reduced
feature sets is an error in CI.

```sh
RUSTFLAGS="-D warnings" cargo hack --locked --workspace --feature-powerset check
```

`check-docs` validates the feature and example inventories, then uses an
isolated all-feature build to compile the README and every complete Rust
snippet in the mdBook. Code fragments that are not standalone are marked
`ignore`.

The CI workflow is the authoritative list of repository checks. It retains LCOV
and listening-bench artifacts. Mutation testing runs through a manually
triggered full sweep, which fails on surviving mutants.

## Listening Bench

`cargo xtask listen` renders fixed-input audible comparisons to
`target/listen/`. Its `index.md` links every file and states its duration,
channel layout, encoding, peak level, purpose, and artifacts to listen for.
Cases cover parameter smoothing, filter automation, oscillator aliasing,
dynamics, independent stereo sidechains, EQ shapes, tail decay, dither, time
stretching, and spectral overlap-add.

Ordinary cases use 24-bit PCM so low-level tails and artifacts remain audible.
The dither comparison alone uses 16-bit PCM. Its undithered file and the naive
sawtooth are deliberately degraded references and are labeled in the index.

Every registered `Processor` and `VariableRate` entry gets a short smoke render.
Meters are excluded because they do not produce audio. Neutral defaults may
sound unchanged, so curated cases use non-neutral settings when the processor's
job needs a focused audition. Curated cases live in `xtask/src/listen.rs`.

The renderer validates WAV headers, dimensions, formats, finite output,
clipping, unique filenames, and registry coverage before writing the index.
Fixed inputs and settings make review repeatable, but spectral output is not
promised to have identical bytes on every platform. The numeric and snapshot
suites decide pass/fail. CI uploads the bench so DSP changes can be reviewed by
ear.

## CPU Benchmarks

Run the non-blocking Criterion suite with every processor domain enabled.

```sh
cargo bench --bench processors --all-features
```

The suite covers block sizes 1, 32, 64, and 512 across lightweight wrapped
processing, static and automated filtering, compressor main and sidechain
paths, limiter, delay, dense valid events, true-peak and loudness meters, time
stretch, and spectral filtering. Preparation and input allocation are excluded
from measurement. In-place cases receive fresh input for every measured call
while processor state and the sample timeline continue between calls.

Criterion reports time per processed block and throughput in frame-channel
elements per second. The current cases are stereo, so the 48 kHz real-time
factor is `elements_per_second / 96_000`. Results are trend evidence only;
there is no machine-dependent CI threshold until repeatability is established.
