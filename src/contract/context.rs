// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Per-block context, fixed-parameter sub-blocks, and status types.

use crate::block::{AudioBlock, AudioBlockMut, Io};
use crate::param::ParamEvent;
use crate::processor::Sample;

/// Output that continues after input ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// No tail.
    None,
    /// At most this many tail frames remain. Stop earlier when `flush` reports
    /// `done`.
    Frames(usize),
    /// Does not complete on its own. The host must cap the drain.
    Infinite,
}

/// Frames produced and whether the producer is finished. Returned by `flush`,
/// [`Source::pull`](crate::processor::Source::pull), and
/// [`VariableRate::process`](crate::processor::VariableRate::process).
#[must_use = "the frame count and completion state must be handled"]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Produced {
    /// Frames written this call.
    pub frames: usize,
    /// `true` once the producer has no more to give.
    pub done: bool,
}

/// Everything a [`Processor::process`](crate::processor::Processor::process) is handed for
/// one block.
#[derive(Debug)]
pub struct ProcessContext<'view, 'samples, T> {
    /// The main signal in its declared I/O shape.
    pub main: Io<'view, 'samples, T>,
    /// Read-only key/aux inputs, one per `sidechain_inputs()`. Disjoint from `main`.
    pub sidechain: &'view [AudioBlock<'view, 'samples, T>],
    /// Sample-stamped parameter changes sorted by nondecreasing offset.
    ///
    /// Offsets must be less than `frames`, and values must be finite. Shared
    /// drivers debug-assert these conditions. Release processing skips
    /// non-finite and out-of-block events; malformed ordering has no promised
    /// per-event behavior.
    pub events: &'view [ParamEvent],
    /// Frame count of this block (`== main.frames()`).
    pub frames: usize,
    /// Absolute frame index of this block's first frame, since stream start.
    pub sample_pos: u64,
}

impl<'view, 'samples, T: Sample> ProcessContext<'view, 'samples, T> {
    /// Build a process context from an already-selected main I/O shape.
    ///
    /// `sample_pos` is the absolute frame index of this block's first frame on
    /// the stream timeline. The parameter-smoothing grid is anchored to it, so
    /// a streaming host must pass its running cursor; a one-shot caller passes
    /// `0`. A discontinuity on the timeline (a seek or a new stream) requires
    /// `reset` before the timeline restarts. Sidechains and parameter events
    /// default to empty slices.
    #[must_use]
    pub fn from_io(main: Io<'view, 'samples, T>, sample_pos: u64) -> Self {
        let frames = match &main {
            Io::InPlace(block) => block.frames(),
            Io::OutputOnly(output) => output.frames(),
            Io::Split { input, output } => {
                let frames = output.frames();
                debug_assert_eq!(
                    input.frames(),
                    frames,
                    "split input and output must have equal frame counts"
                );
                frames
            }
        };
        Self {
            main,
            sidechain: &[],
            events: &[],
            frames,
            sample_pos,
        }
    }

    /// Build an in-place context over caller-provided mutable channel planes.
    ///
    /// All planes must have equal length. The frame count is derived from the
    /// first plane. `sample_pos` is the block's absolute stream position (see
    /// [`from_io`](Self::from_io)); streaming hosts pass their cursor,
    /// one-shot callers pass `0`.
    #[must_use]
    pub fn in_place(planes: &'view mut [&'samples mut [T]], sample_pos: u64) -> Self {
        Self::from_io(Io::InPlace(AudioBlockMut::new(planes)), sample_pos)
    }

    /// Build an output-only context over caller-provided mutable channel planes.
    ///
    /// All planes must have equal length. The frame count is derived from the
    /// first plane. `sample_pos` is the block's absolute stream position (see
    /// [`from_io`](Self::from_io)); streaming hosts pass their cursor,
    /// one-shot callers pass `0`.
    #[must_use]
    pub fn output_only(planes: &'view mut [&'samples mut [T]], sample_pos: u64) -> Self {
        Self::from_io(Io::OutputOnly(AudioBlockMut::new(planes)), sample_pos)
    }

    /// Build a split-I/O context over disjoint input and output channel planes.
    ///
    /// All input planes must have equal length, all output planes must have
    /// equal length, and input/output frame counts must match. `sample_pos` is
    /// the block's absolute stream position (see [`from_io`](Self::from_io));
    /// streaming hosts pass their cursor, one-shot callers pass `0`.
    #[must_use]
    pub fn split(
        input: &'view [&'samples [T]],
        output: &'view mut [&'samples mut [T]],
        sample_pos: u64,
    ) -> Self {
        Self::from_io(
            Io::Split {
                input: AudioBlock::new(input),
                output: AudioBlockMut::new(output),
            },
            sample_pos,
        )
    }

    /// Attach sample-stamped parameter events to this block.
    #[must_use]
    pub fn with_events(mut self, events: &'view [ParamEvent]) -> Self {
        self.events = events;
        self
    }

    /// Attach sidechain buses to this block.
    #[must_use]
    pub fn with_sidechains(mut self, sidechain: &'view [AudioBlock<'view, 'samples, T>]) -> Self {
        self.sidechain = sidechain;
        self
    }
}

/// The contiguous fixed-parameter range passed to `Kernel::render`.
///
/// A `SubBlock` owns no plane array. It reborrows `ctx.main`, sidechains, and
/// the range `[start, start + len)`. Accessors create sub-slices on demand.
/// Parameter values arrive separately as the kernel's typed
/// [`Params`](crate::parameter::Params) snapshot.
#[derive(Debug)]
pub struct SubBlock<'r, 'view, 'samples, T> {
    pub(crate) io: &'r mut Io<'view, 'samples, T>,
    pub(crate) sc: &'r [AudioBlock<'view, 'samples, T>],
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl<T: Sample> SubBlock<'_, '_, '_, T> {
    /// The range frame count.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.len
    }
    /// The channel count of the main signal.
    #[must_use]
    pub fn channels(&self) -> usize {
        match &self.io {
            Io::InPlace(b) => b.channels(),
            Io::OutputOnly(output) | Io::Split { output, .. } => output.channels(),
        }
    }
    /// The in-place main channel `ch`, restricted to this range.
    ///
    /// # Panics
    /// If the processor did not declare `IoMode::InPlace`.
    pub fn channel_mut(&mut self, ch: usize) -> &mut [T] {
        let (start, len) = (self.start, self.len);
        match self.io {
            Io::InPlace(ref mut b) => &mut b.channel_mut(ch)[start..start + len],
            Io::OutputOnly(_) | Io::Split { .. } => {
                panic!("channel_mut requires InPlace I/O; use output_mut()")
            }
        }
    }
    /// The input channel `ch`, restricted to this range.
    ///
    /// For `InPlace` this reads the shared buffer. For `Split` it reads the
    /// disjoint input.
    ///
    /// # Panics
    /// If the processor declared `IoMode::OutputOnly`.
    #[must_use]
    pub fn input(&self, ch: usize) -> &[T] {
        let (start, len) = (self.start, self.len);
        match self.io {
            Io::Split { ref input, .. } => &input.channel(ch)[start..start + len],
            Io::InPlace(ref b) => &b.channel(ch)[start..start + len],
            Io::OutputOnly(_) => panic!("input is unavailable for OutputOnly I/O"),
        }
    }
    /// The output channel `ch`, restricted to this range.
    ///
    /// For `InPlace` this is the shared buffer. For `OutputOnly` and `Split` it
    /// is the output buffer.
    pub fn output_mut(&mut self, ch: usize) -> &mut [T] {
        let (start, len) = (self.start, self.len);
        match self.io {
            Io::InPlace(ref mut b) => &mut b.channel_mut(ch)[start..start + len],
            Io::OutputOnly(ref mut output) | Io::Split { ref mut output, .. } => {
                &mut output.channel_mut(ch)[start..start + len]
            }
        }
    }
    /// The disjoint input and output for channel `ch`. This lets a
    /// [`Split`](crate::processor::Io::Split) processor read original input while writing
    /// output in one pass.
    ///
    /// # Panics
    /// If the processor did not declare `IoMode::Split`.
    pub fn split_channel(&mut self, ch: usize) -> (&[T], &mut [T]) {
        let (start, len) = (self.start, self.len);
        match self.io {
            Io::Split {
                ref input,
                ref mut output,
            } => (
                &input.channel(ch)[start..start + len],
                &mut output.channel_mut(ch)[start..start + len],
            ),
            Io::InPlace(_) | Io::OutputOnly(_) => {
                panic!("split_channel requires Split I/O")
            }
        }
    }
    /// Read sidechain bus `bus`, channel `ch`, restricted to this range.
    #[must_use]
    pub fn sidechain(&self, bus: usize, ch: usize) -> &[T] {
        &self.sc[bus].channel(ch)[self.start..self.start + self.len]
    }
    /// The number of sidechain buses supplied this block.
    #[must_use]
    pub fn sidechain_buses(&self) -> usize {
        self.sc.len()
    }
    /// The channel count of sidechain bus `bus` (mono or main-count).
    #[must_use]
    pub fn sidechain_channels(&self, bus: usize) -> usize {
        self.sc[bus].channels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::AudioBlockMut;

    #[test]
    fn inplace_context_constructor_derives_frames_and_defaults_aux_inputs() {
        let events = [ParamEvent {
            offset: 1,
            param: crate::parameter::ParamId(0),
            value: 2.0,
        }];
        let mut buf = [vec![0.0f32; 4], vec![1.0f32; 4]];
        let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(Vec::as_mut_slice).collect();
        let ctx = ProcessContext::in_place(&mut planes, 128).with_events(&events);

        assert_eq!(ctx.frames, 4);
        assert_eq!(ctx.sample_pos, 128);
        assert_eq!(ctx.events, &events);
        assert!(ctx.sidechain.is_empty());
        let Io::InPlace(block) = &ctx.main else {
            panic!("expected in-place I/O");
        };
        assert_eq!(block.channels(), 2);
        assert_eq!(block.frames(), 4);
    }

    #[test]
    fn output_only_context_derives_frames_and_exposes_only_output() {
        let mut buf = [vec![0.0f32; 4], vec![0.0f32; 4]];
        let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(Vec::as_mut_slice).collect();
        let ctx = ProcessContext::output_only(&mut planes, 64);

        assert_eq!(ctx.frames, 4);
        assert_eq!(ctx.sample_pos, 64);
        let Io::OutputOnly(output) = &ctx.main else {
            panic!("expected output-only I/O");
        };
        assert_eq!(output.channels(), 2);
        assert_eq!(output.frames(), 4);
    }

    #[test]
    fn split_context_constructor_derives_frames_and_accepts_sidechains() {
        let input = [vec![1.0f32; 3], vec![2.0f32; 3]];
        let mut output = [vec![0.0f32; 3], vec![0.0f32; 3]];
        let in_planes: Vec<&[f32]> = input.iter().map(Vec::as_slice).collect();
        let mut out_planes: Vec<&mut [f32]> = output.iter_mut().map(Vec::as_mut_slice).collect();
        let key = [vec![0.5f32; 3]];
        let key_planes: Vec<&[f32]> = key.iter().map(Vec::as_slice).collect();
        let sidechains = [AudioBlock::new(&key_planes)];

        let ctx =
            ProcessContext::split(&in_planes, &mut out_planes, 0).with_sidechains(&sidechains);

        assert_eq!(ctx.frames, 3);
        assert_eq!(ctx.sidechain.len(), 1);
        let Io::Split { input, output } = &ctx.main else {
            panic!("expected split I/O");
        };
        assert_eq!(input.channels(), 2);
        assert_eq!(input.frames(), 3);
        assert_eq!(output.channels(), 2);
        assert_eq!(output.frames(), 3);
    }

    #[test]
    fn split_input_and_output_are_sub_ranged() {
        // A run at [1, 4) reads input[1..4] and writes output[1..4].
        let in_data = [vec![10.0f32, 11.0, 12.0, 13.0, 14.0]];
        let mut out_data = [vec![0.0f32; 5]];
        {
            let in_planes: Vec<&[f32]> = in_data.iter().map(Vec::as_slice).collect();
            let input = AudioBlock::new(&in_planes);
            let mut out_planes: Vec<&mut [f32]> =
                out_data.iter_mut().map(Vec::as_mut_slice).collect();
            let output = AudioBlockMut::new(&mut out_planes);
            let mut io = Io::Split { input, output };
            let mut sub = SubBlock {
                io: &mut io,
                sc: &[],
                start: 1,
                len: 3,
            };
            assert_eq!(sub.input(0), &[11.0f32, 12.0, 13.0][..]);
            sub.output_mut(0).copy_from_slice(&[100.0, 200.0, 300.0]);
        }
        assert_eq!(out_data[0], [0.0, 100.0, 200.0, 300.0, 0.0]);
    }

    #[test]
    fn inplace_output_is_sub_ranged() {
        let mut buf = [vec![0.0f32; 5]];
        {
            let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(Vec::as_mut_slice).collect();
            let block = AudioBlockMut::new(&mut planes);
            let mut io = Io::InPlace(block);
            let mut sub = SubBlock {
                io: &mut io,
                sc: &[],
                start: 1,
                len: 3,
            };
            sub.output_mut(0).copy_from_slice(&[5.0, 6.0, 7.0]);
        }
        assert_eq!(buf[0], [0.0, 5.0, 6.0, 7.0, 0.0]);
    }

    #[test]
    fn output_only_output_is_sub_ranged() {
        let mut buf = [vec![0.0f32; 5]];
        {
            let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(Vec::as_mut_slice).collect();
            let block = AudioBlockMut::new(&mut planes);
            let mut io = Io::OutputOnly(block);
            let mut sub = SubBlock {
                io: &mut io,
                sc: &[],
                start: 1,
                len: 3,
            };
            sub.output_mut(0).copy_from_slice(&[5.0, 6.0, 7.0]);
        }
        assert_eq!(buf[0], [0.0, 5.0, 6.0, 7.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "input is unavailable for OutputOnly I/O")]
    fn output_only_input_accessor_panics() {
        let mut buf = [vec![0.0f32; 1]];
        let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(Vec::as_mut_slice).collect();
        let block = AudioBlockMut::new(&mut planes);
        let mut io = Io::OutputOnly(block);
        let sub = SubBlock {
            io: &mut io,
            sc: &[],
            start: 0,
            len: 1,
        };
        let _ = sub.input(0);
    }

    #[test]
    fn sidechain_buses_channels_and_ranges() {
        let bus0 = [
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        let bus1 = [vec![11.0f32, 12.0, 13.0, 14.0, 15.0]];
        let p0: Vec<&[f32]> = bus0.iter().map(Vec::as_slice).collect();
        let p1: Vec<&[f32]> = bus1.iter().map(Vec::as_slice).collect();
        let sc = [AudioBlock::new(&p0), AudioBlock::new(&p1)];

        let mut buf = [vec![0.0f32; 5]];
        let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(Vec::as_mut_slice).collect();
        let block = AudioBlockMut::new(&mut planes);
        let mut io = Io::InPlace(block);
        let sub = SubBlock {
            io: &mut io,
            sc: &sc,
            start: 1,
            len: 3,
        };
        assert_eq!(sub.sidechain_buses(), 2);
        assert_eq!(sub.sidechain_channels(0), 2);
        assert_eq!(sub.sidechain_channels(1), 1);
        assert_eq!(sub.sidechain(0, 1), &[7.0f32, 8.0, 9.0][..]); // bus0 ch1, [1..4)
    }
}
