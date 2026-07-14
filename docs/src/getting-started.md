<!-- SPDX-License-Identifier: Apache-2.0 -->

# Getting Started

Add the single `bisque` crate to your project.

```toml
[dependencies]
bisque = "0.1.0"
```

The default feature set enables the current non-spectral domains:

- `filters`
- `dynamics`
- `mastering`
- `analysis`
- `generators`
- `time`
- `repair`

Enable spectral processing when FFT, STFT, windows, or streaming spectral
processors are needed.

```toml
[dependencies]
bisque = { version = "0.1.0", features = ["spectral"] }
```

## Minimal Processing Example

The shortest path has four steps. Create settings, construct a processor,
prepare it once, and process as many blocks as needed. `PreparedProcessor` owns
the prepared spec and running input position for one processor.

```rust
# extern crate bisque;
use bisque::host::PreparedProcessor;
use bisque::mastering::{Gain, GainSettings};
use bisque::processor::{DspError, KernelProcessor, ProcessSpec};

type PreparedGain = PreparedProcessor<KernelProcessor<Gain>>;

fn prepare_stereo_gain(max_block: usize) -> Result<PreparedGain, DspError> {
    assert!(max_block > 0, "max_block must be nonzero");
    let spec = ProcessSpec {
        sample_rate: 48_000,
        channels: 2,
        max_block,
        max_memory: None,
    };
    let gain = Gain::with_settings(GainSettings::new().gain_db(-6.0));
    PreparedProcessor::prepare_kernel(gain, spec)
}

fn process_stereo_block(
    processor: &mut PreparedGain,
    left: &mut [f32],
    right: &mut [f32],
) {
    assert_eq!(left.len(), right.len(), "channel lengths must match");
    assert!(
        left.len() <= processor.spec().max_block,
        "block exceeds the prepared maximum"
    );
    let mut planes: [&mut [f32]; 2] = [left, right];
    processor.process_in_place(&mut planes, &[]);
}

# fn main() -> Result<(), DspError> {
# let mut processor = prepare_stereo_gain(256)?;
# let mut left = [0.25_f32; 128];
# let mut right = [0.25_f32; 128];
# process_stereo_block(&mut processor, &mut left, &mut right);
# Ok(())
# }
```

## Raw Realtime Host

A realtime, plugin, graph, or otherwise specialized host uses `Processor`
directly. It owns storage, routing, scheduling, timeline position, latency
compensation, and drain policy:

```rust
# extern crate bisque;
use bisque::mastering::{Gain, GainSettings};
use bisque::processor::KernelProcessor;
use bisque::processor::{DspError, ProcessContext, ProcessSpec, Processor};

fn stream(
    left: &mut [f32],
    right: &mut [f32],
    spec: ProcessSpec,
    block: usize,
) -> Result<(), DspError> {
    assert!(block > 0, "block size must be nonzero");
    assert!(block <= spec.max_block, "block exceeds the prepared maximum");
    assert_eq!(spec.channels, 2, "this example drives stereo audio");
    assert_eq!(left.len(), right.len(), "channel lengths must match");

    let mut processor = KernelProcessor::new(Gain::with_settings(
        GainSettings::new().gain_db(-6.0),
    ));
    processor.prepare(spec)?;

    let total = left.len();
    let mut pos = 0usize;
    while pos < total {
        let n = block.min(total - pos);
        let mut planes: [&mut [f32]; 2] =
            [&mut left[pos..pos + n], &mut right[pos..pos + n]];
        // The cursor keeps every block on one continuous timeline.
        let mut context = ProcessContext::in_place(&mut planes, pos as u64);
        processor.process(&mut context);
        pos += n;
    }
    Ok(())
}
```

A discontinuity on the timeline (a seek, a loop point, or a new stream)
requires `reset` before the timeline restarts. A processor carries smoothing
and DSP state across blocks and cannot infer the jump.

## Settings, Restoration, And Automation

Choose the API by intent.

| Need | API |
| --- | --- |
| Fixed startup value | `FooSettings` before prepare |
| Preset or state restoration without a ramp | `Processor::set_parameter_immediate` between blocks |
| Process-time automation | `ParamEvent` |
| Generic metadata and normalized mapping | `ParamInfo` |

Immediate writes are checked and object-safe so a host can restore state
through `Box<dyn Processor<T>>`. They are not the ordinary configuration path.

```rust
# extern crate bisque;
use bisque::mastering::Gain;
use bisque::parameter::ParamSetError;
use bisque::processor::Processor;

fn restore_gain(
    gain: &mut dyn Processor<f32>,
) -> Result<(), ParamSetError> {
    gain.set_parameter_immediate(Gain::GAIN_DB, -6.0)
}
```

## Runnable Examples

Four examples cover the three supported paths.

- `examples/prepared_processor.rs` shows the short settings plus
  `PreparedProcessor` path. Run it with
  `cargo run --example prepared_processor`.
- `examples/author_kernel.rs` defines a downstream `Kernel`, typed parameters,
  settings, and shared contract check. Run it with
  `cargo run --example author_kernel --features test-support`.
- `examples/offline_chain.rs` shows a batch host. A short stereo clip is
  composed from the library's own sources (bass notes as sample-stamped
  frequency events, filtered-noise hats, a delay-widened pad), then mastered
  end to end with a DC block, high-pass, glue compression, a measured loudness
  correction toward -16 LUFS, true-peak limiting, dither, and a
  latency-compensated `flush` drain. It writes premaster and master WAVs to
  `target/examples_out/`. Run it with `cargo run --example offline_chain`.
- `examples/realtime_plugin.rs` shows a realtime host. It is a broadcast-style
  ducker where a music bus is compressed with a mono voice bus as the
  sidechain key. It uses type-erased `Box<dyn Processor<f32> + Send>` stages,
  per-callback stack plane tables, changing callback block sizes, and
  sample-stamped automation. It writes the ducked mix to
  `target/examples_out/`. Run it with `cargo run --example realtime_plugin`.

## Host Responsibilities

A host is responsible for these steps.

- constructing a `ProcessSpec`
- calling `prepare` before processing
- supplying buffers that match `io_mode`
- supplying events sorted by block offset, in range, and finite
- passing each block's absolute `sample_pos` at context construction so the
  stream timeline stays continuous
- calling `reset` when the stream state should return to the prepared state,
  including before any timeline discontinuity (a seek or a new stream)
- calling `flush` when a processor reports latency or tail that must be
  drained, and capping the total drain itself for `Tail::Infinite`

`prepare` may allocate and may fail. The audio path does not allocate after a
successful `prepare`.

## Parameter IDs

Current processors expose stable associated parameter constants for built-in
controls. Use those constants when building `ParamEvent`s.

```rust
# extern crate bisque;
use bisque::mastering::Limiter;

let threshold = Limiter::THRESHOLD_DB;
```

`param_info()` lets generic hosts read names, physical ranges, configured
defaults, units, normalized value scales, and smoothing metadata.

An event whose ID the processor did not declare is a deterministic no-op because
`process` cannot return per-event errors. Direct restoration calls through
`set_parameter_immediate` return `ParamSetError` for unknown IDs. Kernels read
their own declared parameters as a typed struct, so there is no unknown-ID read
path inside a kernel.
