<!-- SPDX-License-Identifier: Apache-2.0 -->

# Processing Blocks

bisque uses planar audio buffers. Each channel is a slice. A block is a borrowed
view over those slices.

## AudioBlock And AudioBlockMut

`AudioBlock` is read-only and is used for sidechains, meters, and split input.
`AudioBlockMut` is read-write and is used for in-place processing, output-only
generation, split output, and pull targets.

The short borrow of a channel table is separate from the longer borrow of its
sample slices. A host may therefore build one table and reborrow it for
consecutive process calls without allocating or reconstructing the table.

```rust
# extern crate bisque;
use bisque::processor::{AudioBlock, AudioBlockMut};

let left = [0.0_f32; 64];
let right = [0.0_f32; 64];
let input_planes: [&[f32]; 2] = [&left, &right];
let input = AudioBlock::new(&input_planes);

let mut out_left = [0.0_f32; 64];
let mut out_right = [0.0_f32; 64];
let mut output_planes: [&mut [f32]; 2] = [&mut out_left, &mut out_right];
let output = AudioBlockMut::new(&mut output_planes);
```

All channel slices in a block have the same length. The constructors
debug-assert that shape.

Built-in processors and meters treat non-finite audio samples as silence at input
boundaries. Hosts should avoid producing `NaN` or infinity, and those values do
not enter built-in recursive state.

## In-Place I/O

Most processors use `IoMode::InPlace`. The host supplies one mutable main block.

```rust
# extern crate bisque;
use bisque::processor::ProcessContext;

# let mut left = [0.0_f32; 64];
# let mut right = [0.0_f32; 64];
// The second argument is the block's absolute stream position: 0 for a
// one-shot call, the running cursor for a streaming host.
let mut planes: [&mut [f32]; 2] = [&mut left, &mut right];
let mut context = ProcessContext::in_place(&mut planes, 0);
```

In-place processors read and write the same buffer.

## Output-Only I/O

Generators declare `IoMode::OutputOnly`. The host supplies mutable output
planes without a main input signal.

```rust
# extern crate bisque;
use bisque::processor::ProcessContext;

# let mut left = [0.0_f32; 64];
# let mut right = [0.0_f32; 64];
let mut output_planes: [&mut [f32]; 2] = [&mut left, &mut right];
let mut context = ProcessContext::output_only(&mut output_planes, 0);
```

Output-only kernels write through `SubBlock::output_mut`.

## Split I/O

Some processors declare `IoMode::Split`. The host supplies disjoint input and
output blocks.

```rust
# extern crate bisque;
use bisque::processor::ProcessContext;

# let in_left = [0.0_f32; 64];
# let in_right = [0.0_f32; 64];
# let mut out_left = [0.0_f32; 64];
# let mut out_right = [0.0_f32; 64];
let input_planes: [&[f32]; 2] = [&in_left, &in_right];
let mut output_planes: [&mut [f32]; 2] = [&mut out_left, &mut out_right];

let mut context = ProcessContext::split(&input_planes, &mut output_planes, 0);
```

`MovingAverage` and `SpectralFilter` are examples of split-I/O processors.

## Sidechains

Sidechains are read-only `AudioBlock`s supplied through `ProcessContext`.

```rust
# extern crate bisque;
# use bisque::processor::{AudioBlock, ProcessContext};
# let mut left = [0.0_f32; 64];
# let mut right = [0.0_f32; 64];
# let mut planes: [&mut [f32]; 2] = [&mut left, &mut right];
# let key_mono = [0.0_f32; 64];
let sidechain_planes: [&[f32]; 1] = [&key_mono];
let sidechain = [AudioBlock::new(&sidechain_planes)];
let mut context = ProcessContext::in_place(&mut planes, 0).with_sidechains(&sidechain);
```

Processors report how many sidechain buses they need through
`sidechain_inputs()`. Each processor defines the accepted channel layout. The
current dynamics processors accept any nonempty sidechain bus and link their
detector across all channels in that bus. Every sidechain bus must contain at
least as many frames as the main block.

## SubBlock Rendering

`KernelProcessor` splits a host block at control-rate grid boundaries (every 32
frames, anchored to the absolute sample timeline, with stream start counting as
a boundary). Events are sample-stamped, but targets are quantized to the first
grid boundary at or after their timestamp. Every target stamped at or before a
boundary is applied before that boundary's smoother step, so the same values are
produced no matter how the host splits its blocks.

Each `Kernel::render` call receives a `SubBlock` plus the typed parameter
snapshot for that run (`params.field`). Values are constant within a run.
Kernel authors use `channel_mut`, `input`, `output_mut`, or `split_channel`
according to their I/O mode. Calling an input accessor for output-only I/O is a
contract error.

Users of built-in processors normally do not construct `SubBlock` directly.
