<!-- SPDX-License-Identifier: Apache-2.0 -->

# Parameters And Events

Automatable parameters are declared with the `params!` macro, described by
`ParamInfo`, and changed at runtime with `ParamEvent` (smoothed process-time
targets). Fixed startup values belong in processor settings; checked
`Processor::set_parameter_immediate` writes restore state without a ramp.

## Declaring Parameters

The `params!` macro generates the typed parameter struct a kernel's `render`
receives. It also creates one `ParamId` constant per field and the loader that
reads smoothed values from the framework smoother bank.

```rust
# extern crate bisque;
bisque::params! {
    /// Smoothed parameter values for a gain stage.
    pub struct ExampleParams {
        /// Gain in dB.
        pub gain_db => GAIN_DB,
    }
}
assert_eq!(ExampleParams::GAIN_DB, bisque::parameter::ParamId(0));
```

IDs are declaration indices, sequential from `0` in declaration order.
The kernel's `param_info()` must list the same parameters in the same order.
`prepare` validates both the count (the typed struct's field count must equal
`param_info().len()`) and the ordering, rejecting a mismatch with
`DspError::InvalidParam`. Kernels without automatable parameters use
`NoParams`.

Because IDs are declaration indices, they are stable only while the
declaration order is preserved: adding parameters at the end keeps every
existing generated ID, while reordering or removing existing declarations
changes them and is a compatibility break for hosts that persist IDs (saved
automation, presets).

## ParamInfo

Each parameter declares these fields.

- `id`: stable identity within one processor (the declaration index)
- `name`: display name
- `range`: inclusive minimum and maximum
- `default`: value used after `prepare` and `reset`
- `unit`: physical unit
- `value_scale`: deterministic physical-to-normalized mapping
- `smoothing`: ramp shape
- `smoothing_ms`: smoothing time for the ramp; its exact meaning depends on
  the shape (see Smoothing below). Finite and positive for every shape,
  validated in `prepare`

Runtime target values are clamped to the declared range by the smoother bank.
Downstream authors construct metadata with the const-capable `ParamInfo::new`
and `with_smoothing`, `with_smoothing_ms`, and `with_value_scale` methods.
Construction preserves values; `prepare` validates ranges and behavior.

## Units

`Unit` describes the parameter's physical meaning for hosts, UI, and automation
metadata.

| Unit | Meaning |
| --- | --- |
| `Db` | Decibels |
| `Hz` | Frequency |
| `Ms` | Time in milliseconds |
| `Q` | Quality factor |
| `Linear` | Dimensionless linear value |

`Smoothing` is applied to the raw parameter value. `Smoothing::default_for(unit)`
returns `Exponential` (log-domain) for `Hz` and `Linear` for every other unit.
dB parameters are already logarithmic in their raw form, so a linear dB ramp is
a log-domain amplitude ramp.

## Normalized Values

`ValueScale` is independent of both `Unit` and `Smoothing`. `Linear` maps
uniformly across the physical range; `Logarithmic` maps equal ratios uniformly
and requires a strictly positive range. `Hz` defaults to logarithmic mapping,
but authors may override it independently of exponential smoothing.

`ParamInfo::normalize` maps a physical value to `[0, 1]` and
`ParamInfo::denormalize` maps back. Finite out-of-range inputs clamp. NaN and
infinity return `ParamValueError`. Exact minimum and maximum branches preserve
the declared endpoints bit-for-bit. Interior round trips are tested to a
relative `f64` tolerance of `1e-12`, including ranges whose direct span or
ratio would overflow. Formatting, labels, grouping, gestures, and
plugin-framework flags remain host presentation policy.

## Events

`ParamEvent` is stamped relative to the current block.

```rust
# extern crate bisque;
use bisque::mastering::Gain;
use bisque::parameter::ParamEvent;

let event = ParamEvent {
    offset: 128,
    param: Gain::GAIN_DB,
    value: -6.0,
};
```

`offset` is a frame offset within the block. `sample_pos + offset` is the
absolute sample position. Offsets must be less than the current block's frame
count.

Hosts must provide events sorted by nondecreasing `offset` with finite values.
Debug builds assert ordering, values, and in-block offsets. Release processing
deterministically skips non-finite values and out-of-block offsets, and ignores
unknown IDs. Unsorted input violates the host contract and has no promised
per-event release behavior. Events are process-time input, and
`Processor::process` returns `()`, so application is best-effort by contract.

Events set targets. They are sample-stamped, but a target becomes effective at
the first control-grid boundary at or after its timestamp; the smoothed value
then advances on that grid (see Smoothing). An event after a block's last grid
boundary is carried into the next block's first applicable boundary step.

## Settings, Immediate Writes, And Events

Events are the automation path. Targets are smoothed toward over the ramp
time. Ordinary fixed startup values use `FooSettings` before prepare.
`Processor::set_parameter_immediate` exists for preset/state restoration,
tests, and deliberate discontinuities. It snaps current and target to the
clamped value, is object-safe through `Box<dyn Processor<T>>`, and never
allocates. Unlike process-time events, direct calls return `ParamSetError` for
an unknown ID or non-finite value.

## Parameter Constants

Built-in processors expose associated constants for stable parameter IDs.

```rust
# extern crate bisque;
use bisque::filters::Biquad;
use bisque::mastering::{Gain, Limiter};

let gain = Gain::GAIN_DB;
let cutoff = Biquad::CUTOFF_HZ;
let threshold = Limiter::THRESHOLD_DB;
```

Use `param_info()` when building generic host maps from names, ranges,
configured defaults, units, normalized value scales, and smoothing metadata.

## Smoothing

Events set parameter targets. The smoother bank advances on a fixed 32-frame
control grid anchored to the absolute sample timeline, with stream start
counting as a boundary. `KernelProcessor` splits each host block at those grid
boundaries only. Every target stamped at or before a boundary is applied
before that boundary's smoother step. Targets are therefore quantized to the
next control-grid boundary, and output is bit-identical under any host block
splitting.

Smoothing time is per-parameter (`ParamInfo::smoothing_ms`; every built-in
parameter uses 5 ms), and its exact meaning depends on the shape:

- `Smoothing::Linear` moves from the current value to each new target at a
  constant rate. Without another target change, it reaches the target no later
  than `ceil(steps)` control updates.
- `Smoothing::Exponential` moves from the current value to each new target by
  a constant multiplicative factor. It also reaches the target no later than
  `ceil(steps)` control updates. This is the natural shape for frequency and
  the default for `Hz` parameters. It requires a positive range minimum, which
  is validated in `prepare`.
- `Smoothing::OnePole` approaches the target asymptotically with
  `smoothing_ms` as its time constant; the target is never reached exactly.
- `Smoothing::Step` jumps at the next boundary and ignores `smoothing_ms`
  (the field must be positive for metadata uniformity).

The control grid quantizes effective timing. A later event can also redirect a
ramp before it finishes. A generic host should therefore treat `smoothing_ms`
as a nominal duration rather than an exact wall-clock deadline.

Each `Kernel::render` call receives the smoothed values as its typed
[`params!`](#declaring-parameters) struct, constant for the whole
fixed-parameter run. A processor author implements fixed-parameter DSP. The
wrapper handles grid splitting, latching, and smoothing.

Custom drivers can own a `SmootherBank` directly. Construct it with
`SmootherBank::try_new` so parameter metadata and the bank's memory budget are
checked before allocation.

## Unknown IDs

An event that names an ID the processor did not declare is a deterministic
no-op because `process` cannot report per-event failures. Direct parameter
writes are checked: `Processor::set_parameter_immediate`,
`SmootherBank::set_immediate`, and `SmootherBank::set_target` return
`ParamSetError::UnknownParam` for an undeclared ID. The shorter method name is
used only on `SmootherBank`; the processor method names the parameter
explicitly.

Kernels read only their declared, typed parameters. The generated `params!`
loader indexes the smoother bank after `prepare` validates the declaration
order, so a missing declaration index is treated as an internal invariant
violation rather than normal audio-path control flow.
