<!-- SPDX-License-Identifier: Apache-2.0 -->

# Introduction

bisque is a host-agnostic Rust DSP library for audio processors and utilities
that need predictable behavior across block sizes. It is one published crate,
`bisque`, with focused host, processor, and parameter contract modules plus
feature-gated DSP domains.

The main contract covers these areas.

- sample-stamped parameter events, quantized and smoothed on a fixed control grid
- in-place, output-only, and split I/O
- fixed process specifications
- latency and tail reporting
- reset behavior
- memory-footprint reporting
- no allocation in the audio path
- block-size-invariant output where applicable
- finite-sample handling at built-in processor and meter input boundaries

The crate is pre-release but functional. Each trait family has at least one
working implementation, and each planned DSP domain has a tested first processor
or primitive.

## Where It Fits

bisque sits below hosts and applications such as plugin wrappers, realtime audio
engines, and offline renderers. It does not own an audio device, plugin format,
graph scheduler, file decoder, or user interface. Hosts provide buffers and
parameter events. bisque processors transform, generate, analyze, or consume
those buffers through a shared contract.

## Public Shape

Contract types are imported by role.

```rust
# extern crate bisque;
use bisque::processor::{Kernel, ProcessContext, ProcessSpec, Processor};
use bisque::parameter::ParamEvent;
```

Processors are imported from domain modules.

```rust
# extern crate bisque;
use bisque::filters::Biquad;
use bisque::mastering::Limiter;
use bisque::time::Delay;
```

Lower-level DSP utilities live under `bisque::dsp`.

```rust
# extern crate bisque;
use bisque::dsp::math;
use bisque::dsp::oversample::PolyphaseUpsampler;
```

The optional spectral module is available through the same crate when the
`spectral` feature is enabled.

```rust
# extern crate bisque;
use bisque::spectral::{SpectralFilter, SpectralFilterSettings, Stft, Window};
```
