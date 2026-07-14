<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Airclay LLC -->

# bisque dsp

[![CI](https://github.com/airclay-io/bisque/actions/workflows/ci.yml/badge.svg)](https://github.com/airclay-io/bisque/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/bisque.svg)](https://crates.io/crates/bisque)
[![docs.rs](https://docs.rs/bisque/badge.svg)](https://docs.rs/bisque)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV: 1.85](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)

bisque is a Rust audio DSP library for building processors that behave
consistently in realtime and offline applications. It includes filters,
dynamics, mastering tools, meters, generators, time-based effects, audio repair,
and optional spectral processing.

Every processor follows the same processing contract. bisque handles parameter
events and smoothing, block splitting, latency and tail reporting, resets,
memory accounting, allocation-free realtime processing, and non-finite samples.
Processor implementations can focus on the DSP while hosts get consistent
behavior across the library.

## Installation

```toml
[dependencies]
bisque = "0.1.0"
```

The default features enable all non-spectral modules. Add `spectral` for FFT
and STFT processing.

```toml
[dependencies]
bisque = { version = "0.1.0", features = ["spectral"] }
```

## Usage

Most applications need four steps: create settings, construct a processor,
prepare it, and process audio. `PreparedProcessor` stores the prepared
configuration and tracks the processing timeline for one processor.

```rust
use bisque::host::PreparedProcessor;
use bisque::mastering::{Gain, GainSettings};
use bisque::processor::{DspError, ProcessSpec};

fn apply_gain(samples: &mut [f32]) -> Result<(), DspError> {
    let spec = ProcessSpec {
        sample_rate: 48_000,
        channels: 1,
        max_block: samples.len().max(1),
        max_memory: None,
    };
    let gain = Gain::with_settings(GainSettings::new().gain_db(-6.0));
    let mut gain = PreparedProcessor::prepare_kernel(gain, spec)?;

    let mut planes = [samples];
    gain.process_in_place(&mut planes, &[]);
    Ok(())
}
```

Use `processor::Processor` directly when your host manages a processing graph,
routing, block scheduling, latency compensation, or tail draining.
`PreparedProcessor` is an optional helper for hosting a single processor; it
does not make those decisions for the host.

For automation, pass `parameter::ParamEvent`s with frame offsets within the
current block. Together with the block's absolute position, each offset gives
the event an exact timestamp. bisque applies the new target at the first
32-frame control boundary at or after that timestamp, then smooths the value
according to its parameter metadata. The target change is quantized to the
control grid rather than applied at the exact sample.

## Public API

| Path | Role |
| --- | --- |
| `bisque::{filters,dynamics,mastering,...}` | Concrete processors grouped by audio domain |
| `bisque::host` | Optional `PreparedProcessor` lifecycle helper |
| `bisque::processor` | Raw host and DSP-author contracts |
| `bisque::parameter` | IDs, metadata, events, mapping, and smoothing types |
| `bisque::dsp` | Lower-level deterministic DSP-author utilities |
| `bisque::testing` | Shared downstream contract tests behind `test-support` |

Each public type has one import path. There is no prelude and contract types are
not re-exported from the crate root. The `params!` macro is available at the
crate root because Rust exports macros there.

Parameter operations are separated by intent:

| Need | API |
| --- | --- |
| Choose initial values | `FooSettings` |
| Set a value immediately between processing calls | `Processor::set_parameter_immediate` |
| Automate a value while processing | `ParamEvent` |
| Inspect ranges, units, normalized mapping, and smoothing | `ParamInfo` |

## Core types

| Type | Role |
| --- | --- |
| `Kernel` | The simpler DSP-author interface. A kernel renders runs with fixed parameter values; the wrapper handles events and smoothing. |
| `PreparedProcessor<P, T = f32>` | An optional helper that prepares and hosts one processor while tracking its processing timeline. |
| `KernelProcessor<K, T = f32>` | Turns a kernel into a `Processor` and handles parameter events and smoothing. |
| `Processor` | The host-facing processing trait. Every built-in processor can be stored as `Box<dyn Processor<f32> + Send>`. |
| `Measurer` | The common interface for peak, RMS, true-peak, and LUFS meters. Hosts send audio with `observe` and retrieve results with `read`. |
| `VariableRate` | The interface for processors such as time-stretch that request audio from a `Source` at their own rate. |

## Modules

| Module | What it includes |
| --- | --- |
| `filters` | RBJ biquads with low-pass, high-pass, shelves, and peaking filters, plus magnitude, phase, group-delay readouts, and moving-average FIR. |
| `dynamics` | Compressor, expander, and gate with channel-linked detection and external sidechain keying. |
| `mastering` | Gain, seeded TPDF dither, and a lookahead limiter with oversampled peak detection, a raised-cosine attack, and a configurable safety margin. |
| `analysis` | Peak, RMS, crest, windowed RMS, oversampled true peak, clip counting, and gated LUFS loudness with allocation-free metering. |
| `generators` | Sine, seeded white noise, and alias-reduced PolyBLEP saw/square oscillators. |
| `time` | Feedback delay with a streamed tail drain and overlap-add time-stretch. |
| `repair` | DC removal. |
| `spectral` | Optional FFT, STFT, windows, and streaming spectral filtering. |

## Processing contract

The same lifecycle and processing rules apply to every built-in processor.
Shared contract tests cover the entire catalog, and processor-specific tests
check the expected audio behavior.

- Given the same input and events, a processor produces bit-identical output
  no matter how the host divides the audio into blocks. Tests cover awkward
  block sizes, generated split patterns, and event schedules.
- Processing does not allocate after `prepare`. A counting allocator checks
  every processor, meter, and rate changer.
- `latency()` is exact and does not change after `prepare`. `tail()` gives a
  real upper bound, and `flush` never writes more than the capacity the host
  supplies.
- Parameter events carry exact sample timestamps. Targets take effect on a
  fixed 32-frame control grid and follow their declared smoothing, and
  built-in frequency parameters ramp logarithmically. The `params!` macro
  gives kernels typed parameter values, so kernel code cannot read the wrong
  parameter.
- Output is deterministic. Transcendental math comes from a pinned pure-Rust
  library, randomness is always seeded, and denormals are flushed in
  software. Processors that promise cross-platform byte-identical output also
  have committed snapshot hashes.
- `memory_footprint()` reports the bytes reserved for processor-owned state
  after `prepare`, and `ProcessSpec::max_memory` caps the same measure.

Tests check behavior directly before relying on snapshots. Filter responses are
compared with their transfer functions, dynamics with their static curves,
oscillators with ideal spectra, and LUFS results with the ebur128 reference.
Mutation testing verifies that meaningful changes to the DSP are caught by the
test suite.

## Examples

```sh
cargo run --example offline_chain
cargo run --example realtime_plugin
cargo run --example prepared_processor
cargo run --example author_kernel --features test-support
```

`offline_chain` creates a short stereo clip and masters it with DC removal, a
rumble filter, compression, measured loudness correction toward -16 LUFS,
true-peak limiting, dither, latency compensation, and tail draining.

`realtime_plugin` demonstrates a broadcast-style ducker. It compresses a music
bus using a mono voice bus as the sidechain, changes the callback block size,
and sends automation without allocating in the processing loop.

Both examples write WAV files to `target/examples_out/` so you can listen to
the results. `prepared_processor` shows the single-processor host helper, and
`author_kernel` shows how to write and test a kernel.

## Feature flags

| Feature | Default | Purpose |
| --- | --- | --- |
| `filters` | Yes | Filters and moving-average processing |
| `dynamics` | Yes | Compression, expansion, and gating |
| `mastering` | Yes | Gain, dither, scaling, and limiting |
| `analysis` | Yes | Level, true-peak, DC, clip, and loudness meters |
| `generators` | Yes | Oscillators and seeded noise |
| `time` | Yes | Delay and time stretching |
| `repair` | Yes | DC removal and correction |
| `spectral` | No | FFT, STFT, windows, and spectral filtering |
| `test-support` | No | Shared downstream contract-test helpers |
| `snapshot-support` | No | Repository snapshot tooling |

## Documentation

The mdBook in `docs/` explains the processing contract, parameters and events,
audio blocks, testing, and snapshots. Build it with `mdbook build docs`.
`cargo xtask check-docs` checks README and feature metadata and compiles the
README and mdBook Rust examples. `tests/documentation.rs` checks the structured
API and processor catalog inventories. Use `cargo doc --all-features` to build
the API documentation.

## Contributing

Meaningful outside contributions require the Airclay LLC Contributor License
Agreement in [CLA.md](CLA.md) before merge. Start with [CONTRIBUTING.md](CONTRIBUTING.md).

## License

bisque is available under the [Apache License, Version 2.0](LICENSE).

## Contact

Contact Airclay at support@airclay.io.
