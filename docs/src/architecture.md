<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture

bisque is organized as one library crate plus repository tooling.

```text
bisque/
  Cargo.toml
  src/
    lib.rs
    contract/
    dsp/
    testing/
    filters/
    dynamics/
    mastering/
    analysis/
    generators/
    time/
    repair/
    spectral/
  tests/
  testdata/
  xtask/
```

The root package is both the published library crate and the workspace root.
`xtask` is the only workspace member because it is tooling, not library API.

## Public Layers

Public types are grouped by role, and each has one documented import path.

```rust
# extern crate bisque;
use bisque::processor::{
    AudioBlock, AudioBlockMut, DspError, Io, IoMode, Kernel, Measurer,
    ProcessContext, ProcessSpec, Processor, Produced, RingSource, Sample,
    Source, SubBlock, Tail, VariableRate,
};
use bisque::parameter::{
    ParamEvent, ParamId, ParamInfo, Smoothing, Unit, ValueScale,
};
```

`bisque::host::PreparedProcessor` is the optional single-processor lifecycle
helper. The crate root exports the `params!` macro but has no prelude or contract
type re-exports. The files in `src/contract` are private implementation modules
behind `processor` and `parameter`.

## Domain Modules

Processors and utilities live under feature-gated domain modules.

```text
bisque::filters
bisque::dynamics
bisque::mastering
bisque::analysis
bisque::generators
bisque::time
bisque::repair
bisque::spectral
```

Public domain types are re-exported from the domain root. Prefer
`bisque::filters::Biquad`, not nested implementation paths.

## DSP Utilities

`bisque::dsp` contains lower-level utilities for processor authors.

- `math` wrappers for deterministic transcendental functions
- `oversample::PolyphaseUpsampler`
- `SmootherBank`
- the hidden block driver used by `KernelProcessor`

`KernelProcessor<K, T = f32>` is exported from `bisque::processor`.
Lower-level utilities are for custom processors, tests, and advanced
integration.

## Three Paths

1. A direct user configures a domain processor through settings and wraps it in
   `PreparedProcessor`.
2. A specialized host drives `Processor` and `ProcessContext` directly.
3. A DSP author normally implements `Kernel`; unusual block algorithms use
   `Processor`, meters use `Measurer`, and rate changers use `VariableRate` with
   a pull-based `Source`.

`PreparedProcessor` owns one prepared processor, its spec, and its next process
position. It does not own a chain, graph, file renderer, transport, plugin
lifecycle, event queue, or presentation metadata.

## Testing Support

`bisque::testing` is available behind the `test-support` feature. It contains
the shared contract-test harness used by the repository and available to
downstream processor authors.

`bisque::testing::registry` enumerates every built-in processor, meter, and
rate changer for the repository's cross-cutting suites (contract, no-alloc,
and API-surface completeness). It is supported test API: downstream suites may
consume the built-in catalog. Downstream processors use `Contract` directly
rather than inserting entries into bisque's registry. The generated concrete
snapshot registry behind `snapshot-support` is repository-only and hidden from
rustdoc.
