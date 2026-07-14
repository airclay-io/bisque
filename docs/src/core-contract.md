<!-- SPDX-License-Identifier: Apache-2.0 -->

# Core Contract

The contract defines how hosts and processors interact. A processor is prepared
for one fixed spec, then receives blocks, events, sidechains, and stream position
through the contract types.

## ProcessSpec

`ProcessSpec` is fixed after `prepare`.

```rust
pub struct ProcessSpec {
    pub sample_rate: u32,
    pub channels: usize,
    pub max_block: usize,
    pub max_memory: Option<usize>,
}
```

`sample_rate`, `channels`, and `max_block` describe the block geometry a host
will provide. `max_memory` is an optional cap on internal state, measured in
logical reserved payload bytes (see [Memory](#memory)). Processors that
cannot fit the cap return `DspError::OverBudget` from `prepare`.

## Trait Guide

Use the smallest trait that fits the operation.

| Trait | Use for | Output rate |
| --- | --- | --- |
| `Kernel` | Fixed-rate effects and generators with fixed-parameter sub-block rendering | One per process frame |
| `Processor` | Host-facing fixed-rate processors and block-aware implementations | One per process frame |
| `Measurer` | Meters and analyzers that observe read-only audio | Reading only |
| `VariableRate` | Rate changers that pull input and produce a different frame count | Variable |
| `Source` | Pull-based input for `VariableRate` processors | Input only |

Most same-rate processors implement `Kernel` and are wrapped with
`Kernel::into_processor()`. Generators such as oscillators and noise declare
output-only I/O. Implement `Processor` directly for algorithms that need
whole-block buffering or event handling that cannot be expressed as
fixed-parameter sub-blocks.

## Type Erasure And Sample-Type Selection

Hosts can store mixed processors as `Box<dyn Processor<f32> + Send>`. Every
built-in processor is `Send`, so any of them can use that type (wrapped with
`Kernel::into_processor()` where needed) behind that one object type:

Bisque does not promise that processors are `Sync`. Processing mutates owned
state through `&mut self`; move a processor between threads rather than sharing
one instance for concurrent processing.

```rust
# extern crate bisque;
use bisque::processor::KernelProcessor;
use bisque::mastering::Gain;
use bisque::time::Delay;
use bisque::processor::Processor;

let chain: Vec<Box<dyn Processor<f32> + Send>> = vec![
    Box::new(KernelProcessor::new(Gain::new())),
    Box::new(KernelProcessor::new(Delay::new())),
];
```

Processors are generic over the sample type `T`. `KernelProcessor` carries that
sample type in its own type (`KernelProcessor<K, T = f32>`), so the prepared
sample type and the processed sample type cannot drift apart. The default
constructor is the f32 host path:

```rust
# extern crate bisque;
use bisque::processor::KernelProcessor;
use bisque::mastering::Gain;
use bisque::processor::Processor;

# use bisque::processor::{DspError, ProcessSpec};
# fn prepare_f32(spec: ProcessSpec) -> Result<(), DspError> {
let mut p = KernelProcessor::new(Gain::new());
p.prepare(spec)?;
let latency = p.latency();
# assert_eq!(latency, 0);
# Ok(())
# }
```

Use `with_sample_type` or a type-qualified `Kernel::into_processor` call for an
explicit f64 processor:

```rust
# extern crate bisque;
use bisque::processor::KernelProcessor;
use bisque::mastering::Gain;
use bisque::processor::{Kernel, Processor};

# use bisque::processor::{DspError, ProcessSpec};
# fn prepare_f64(spec: ProcessSpec) -> Result<(), DspError> {
let mut p64: KernelProcessor<Gain, f64> =
    KernelProcessor::with_sample_type(Gain::new());
p64.prepare(spec)?;

let mut via_trait = <Gain as Kernel<f64>>::into_processor(Gain::new());
via_trait.prepare(spec)?;
# Ok(())
# }
```

Once the processor is behind `Box<dyn Processor<f32> + Send>`, the sample type is
fixed by the object type.

## Processor Lifecycle

A host-facing `Processor` follows this lifecycle:

1. Construct the processor.
2. Call `prepare(spec)`.
3. Query `io_mode`, `param_info`, `latency`, `tail`, and `memory_footprint` as
   needed.
4. Call `process` for each block.
5. Call `flush` when latency or tail output must be drained.
6. Call `reset` when state should return to the post-prepare state.

`prepare` may allocate and may fail. `process` does not allocate and does not
return a `Result`.

## Host Preconditions

Hosts must call `prepare` before processing and then keep each block within the
prepared `ProcessSpec`:

- `ctx.frames` is no larger than `max_block`
- `ctx.frames` equals the main block frame count, including both the input and
  output frame counts for split I/O
- main I/O has the prepared channel count
- the supplied `Io` shape matches `io_mode()`
- `ctx.sidechain` contains the buses reported by `sidechain_inputs()`, and each
  bus has at least `ctx.frames` frames
- `sample_pos` continues the stream timeline: each block's position is the
  previous block's position plus its frame count. Context constructors take
  the position explicitly; a discontinuity (a seek or a new stream) requires
  `reset` before the timeline restarts
- parameter events are sorted in nondecreasing offset order, in range for the
  block, and finite

The shared drivers debug-assert the event contract. Release processing skips
non-finite and out-of-block events, ignores unknown IDs, and makes no per-event
promise for malformed ordering. `KernelProcessor` (and the crate's direct `Processor`
implementations) also debug-assert the buffer geometry above (block size,
channel count, I/O shape, and sidechain buses) before any kernel indexing, so
a host bug panics with a specific message in debug builds. Release builds do
not pay for those checks, and geometry is a host precondition there.
Process-time event application is best-effort because `process` does not
return a `Result`; direct restoration calls through
`set_parameter_immediate` report unknown
IDs and non-finite values through `ParamSetError`.

Built-in processors and meters treat non-finite audio samples as silence at input
boundaries. This prevents `NaN` and infinity from entering recursive filter,
delay, limiter, spectral, and meter state.

## Latency And Tail

Latency is reported in frames. It is constant after `prepare`.

`Tail` describes output that continues after the input body:

- `Tail::None` means no output remains after the body.
- `Tail::Frames(n)` means at most `n` frames remain.
- `Tail::Infinite` means the drain does not complete on its own and requires a
  host cap.

A recursive processor does not reach an absolute silence floor in a fixed
number of frames for arbitrarily hot (still finite) input, so a declared
finite tail is a decay guarantee, not a silence guarantee. The bound is
computed from the processor's declared range extremes plus documented
headroom for state above full scale, and the drain covers at least the
headroom-to-floor decay ratio relative to the state at end of input. The
built-in IIR filters cover at least 180 dB, and each tail-bearing processor
documents its own floor and headroom. Signals inside the documented
headroom end below the absolute floor before the drain reports `done`. Hotter
input decays by the same ratio, and `done` means the declared tail
has been delivered, not that the state is exactly zero.

`flush(out)` drains latency or tail into `out` and returns `Produced`. A call
writes at most `out.frames()` frames, and `done` reports that the declared
tail has been delivered. Any total cap on a drain is the host's: stop calling
`flush` once enough frames have arrived. New input (`process`) starts fresh
drain accounting, but it does not clear processor state left by an incomplete
drain. Call `reset` before processing input from an independent stream. Every
flush block must use the prepared main-channel count.
Its frame count is per-call capacity and may exceed `ProcessSpec::max_block`;
`max_block` constrains `process` input only.

After `flush` reports `done`, later flush calls write zero frames and remain
done until new input or `reset` starts a fresh drain.

For latency-bearing processors, the first `latency()` drained frames complete
the delayed input body. Any later frames are append-only effect tail.
`Tail::Frames(n)` bounds both and must be at least the latency. A processor
author must preserve that ordering so same-length and full-tail hosts can apply
their different policies without processor-specific knowledge.

A finite-tail host drains to completion:

```rust,ignore
let mut stage = /* host-sized planes */;
loop {
    let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
    let mut out = AudioBlockMut::new(&mut planes);
    let produced = proc.flush(&mut out);
    deliver(&stage, produced.frames);
    if produced.done {
        break;
    }
}
```

A `Tail::Infinite` processor never reports `done`, so its host must bound the
drain itself:

```rust,ignore
let mut remaining = host_cap;
while remaining > 0 {
    let n = stage_frames.min(remaining);
    // Offer exactly `n` frames, as above; an infinite tail fills all of them.
    remaining -= n;
}
```

## Parameters And The Smoother Bank

`KernelProcessor` owns the framework smoother bank for a wrapped `Kernel`.
Events set targets in the bank. The bank advances on a fixed 32-frame control grid
anchored to the absolute sample timeline (`ctx.sample_pos`), and stream start
counts as a grid boundary. Events are sample-stamped, but each target becomes
effective at the first grid boundary at or after its timestamp. Every target
stamped at or before a boundary is applied before that boundary's smoother step,
so output does not depend on how the host splits its blocks. See
[Parameters And Events](parameters-and-events.md).

## Memory

`memory_footprint()` reports internal state after `prepare`, measured in
logical reserved payload bytes. It counts every processor-owned element slot
the processor intentionally keeps available, whether or not that slot currently
contains valid history, times element size. Incidental allocator overcapacity,
inline scalar state (cursors, coefficients, cached values), container metadata,
and allocator bookkeeping are excluded, so the number describes the DSP state
model exactly, not the process's heap usage.
Opaque plan storage owned internally by a third-party backend is also outside
the measure when the backend does not expose its layout; caller-owned scratch
buffers are included.
`ProcessSpec::max_memory` caps this same measure, and the built-in boundary
tests prove exact fit at the cap and failure one byte under it.

For `Kernel`s wrapped by `KernelProcessor`, the reported footprint includes both the
kernel state and the framework-owned smoother bank. When `max_memory` is set,
`KernelProcessor::prepare` reserves the bank's footprint first and hands the kernel
only the remainder, so a cap smaller than the bank alone already fails with
`DspError::OverBudget`. Built-in kernels preflight their known layouts before
allocating; a downstream kernel must honor its sub-budget the same way,
because a kernel that allocates first is only rejected by the wrapper's
post-prepare total check, after the allocation happened.

## Determinism

The deterministic contract is behavioral and test-driven.

- blocks are split at control-rate grid boundaries anchored to the absolute
  sample timeline, and sample-stamped targets take effect at the next boundary
- smoother stepping follows the stream timeline, so any host block splitting
  produces bit-identical output
- block-size invariance is tested where applicable
- snapshot cases are limited to output expected to match byte for byte across
  supported platforms
- all production transcendental math routes through `bisque::dsp::math`, and bare
  standard-library transcendentals are rejected by `cargo xtask
  check-determinism` with no production allowlist (test-only independent
  reference math is tagged `// test-oracle:` inside trailing test modules)
- FMA contraction and fast-math are rejected by CI

Spectral code currently uses `realfft` and has no committed cross-platform
snapshots.
