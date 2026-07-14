<!-- SPDX-License-Identifier: Apache-2.0 -->

# Authoring Processors

bisque provides several contracts for DSP code. Choose the smallest contract
that matches the work.

| Contract | Use |
| --- | --- |
| `Kernel` | Same-rate processing with fixed parameter values during each render call |
| `Processor` | Same-rate block algorithms that cannot use fixed-parameter sub-block rendering |
| `Measurer` | Read-only analysis with a separate reading |
| `Source` | Audio pulled by a variable-rate processor |
| `VariableRate` | Processing whose output length differs from its input consumption |

Most effects should implement `Kernel`. `KernelProcessor` then provides the
host-facing `Processor`, parameter smoothing, event handling, geometry checks,
and the absolute control grid.

## A Small Kernel

This gain kernel has one setting and one automatable parameter.

```rust
# extern crate bisque;
use bisque::parameter::{ParamId, ParamInfo, Unit};
use bisque::processor::{DspError, Kernel, KernelProcessor, ProcessSpec, Sample, SubBlock};

bisque::params! {
    /// Smoothed values supplied to `Trim::render`.
    pub struct TrimParams {
        /// Linear gain.
        pub gain => GAIN,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct TrimSettings {
    pub gain: f64,
}

impl TrimSettings {
    pub const fn new() -> Self {
        Self { gain: 1.0 }
    }

    pub const fn gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }
}

impl Default for TrimSettings {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Trim {
    params: [ParamInfo; 1],
}

impl Trim {
    pub const GAIN: ParamId = TrimParams::GAIN;

    pub fn with_settings(settings: TrimSettings) -> Self {
        Self {
            params: [ParamInfo::new(
                Self::GAIN,
                "gain",
                (0.0, 2.0),
                settings.gain,
                Unit::Linear,
            )],
        }
    }
}

impl<T: Sample> Kernel<T> for Trim {
    type Params = TrimParams;

    fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
        Ok(())
    }

    fn reset(&mut self) {}

    fn param_info(&self) -> &[ParamInfo] {
        &self.params
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &TrimParams) {
        for channel in 0..io.channels() {
            for sample in io.channel_mut(channel) {
                *sample = T::from_f64(sample.to_f64() * params.gain);
            }
        }
    }
}

# fn make_processor() {
let trim = Trim::with_settings(TrimSettings::new().gain(0.5));
let processor = KernelProcessor::new(trim);
# let _ = processor;
# }
```

The generated `TrimParams` type gives `render` named values. Its declaration
order must match `param_info()`. `KernelProcessor::prepare` validates the count,
IDs, ranges, defaults, smoothing, and normalized mapping before rendering.

Use `NoParams` when the processor has no automatable parameters. Fixed choices
that do not change during processing belong in the settings type.

## Preparation And Memory

Use `prepare` for work that can fail or allocate. Validate settings and the
supported `ProcessSpec` before committing new state. A failed prepare must leave
the processor unprepared.

When the kernel reserves buffers, calculate their logical payload size with
checked arithmetic before allocating. Compare that size with `spec.max_memory`
and return `DspError::OverBudget` when it does not fit. `KernelProcessor`
reserves memory for its smoother bank first, so the kernel receives only its
part of the budget.

`memory_footprint()` reports the element slots intentionally kept available
after prepare. It does not include inline scalar fields, container metadata,
allocator bookkeeping, or spare capacity that was not requested.

Do not allocate in `render` or `flush`. Allocate reusable buffers in `prepare`
and clear them in `reset`.

## Rendering And I/O

The default I/O mode is `IoMode::InPlace`. Use `io.channel_mut(channel)` when
the input and output share one buffer. Return `IoMode::OutputOnly` for a
generator and write through `output_mut`. Return `IoMode::Split` when the
algorithm needs disjoint input and output, then use `input`, `output_mut`, or
`split_channel`.

Each render call covers no more than one 32-frame control cell. Parameter values
remain constant for that call. The wrapper handles host block splitting,
event latching, and smoothing.

A kernel that accepts sidechains reports the number of buses through
`sidechain_inputs()`. The kernel defines and documents the channel layout it
accepts for each bus.

Built-in processors treat non-finite input as silence before it enters DSP
state. Downstream processors should choose and document their own finite-input
policy.

## Reset Latency And Tail

`reset` returns the kernel to the same state it had immediately after a
successful prepare. It must not allocate.

Report latency in frames. If output remains after input ends, report a `Tail`
and implement `flush`. A latency-bearing processor drains the delayed input body
first. Any effect tail follows it. `Tail::Frames(n)` is an upper bound and must
be at least the reported latency.

Flush writes no more than the capacity supplied by the host. It returns the
number of frames written and whether the drain is done. New processing starts
fresh drain accounting without resetting DSP state.

## Contract Tests

The optional `test-support` feature provides the same contract harness used by
the built-in catalog. Enable it for downstream tests through a development
dependency.

```toml
[dev-dependencies]
bisque = { version = "0.1.0", features = ["test-support"] }
```

```rust
# extern crate bisque;
# use bisque::parameter::{ParamId, ParamInfo, Unit};
# use bisque::processor::{DspError, Kernel, KernelProcessor, ProcessSpec, Sample, SubBlock};
# bisque::params! {
#     pub struct TestTrimParams {
#         pub gain => GAIN,
#     }
# }
# #[derive(Clone, Debug)]
# struct TestTrim { params: [ParamInfo; 1] }
# impl TestTrim {
#     fn new() -> Self {
#         Self { params: [ParamInfo::new(TestTrimParams::GAIN, "gain", (0.0, 2.0), 1.0, Unit::Linear)] }
#     }
# }
# impl<T: Sample> Kernel<T> for TestTrim {
#     type Params = TestTrimParams;
#     fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> { Ok(()) }
#     fn reset(&mut self) {}
#     fn param_info(&self) -> &[ParamInfo] { &self.params }
#     fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, params: &TestTrimParams) {
#         for channel in 0..io.channels() {
#             for sample in io.channel_mut(channel) {
#                 *sample = T::from_f64(sample.to_f64() * params.gain);
#             }
#         }
#     }
# }
use bisque::testing::Contract;

let input = vec![vec![0.25_f32; 257]; 2];
Contract::default().assert_block_size_invariant(
    || KernelProcessor::new(TestTrim::new()),
    &input,
    &[],
);
```

Run the downstream tests normally.

```sh
cargo test
```

Block-size invariance is only one part of processor testing. Add focused tests
for the DSP result, invalid settings, exact memory budgets, reset behavior,
latency, tail output, and allocation-free rendering where they apply.
