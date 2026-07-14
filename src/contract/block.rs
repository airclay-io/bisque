// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Borrowed multichannel block views and [`Io`].
//!
//! Hosts own the audio buffers and build per-channel plane tables for each
//! `process` call. [`AudioBlock`] is read-only, [`AudioBlockMut`] is read-write,
//! and [`Io`] selects in-place, output-only, or split I/O.

use crate::processor::Sample;

/// A read-only multichannel view for sidechains, [`Measurer::observe`], and the
/// input side of [`Io::Split`]. Channels are planar `&[T]` slices.
/// The channel-table borrow is independent of the sample-slice borrow, so a
/// host may reborrow and reuse one table across consecutive calls.
///
/// [`Measurer::observe`]: crate::processor::Measurer::observe
#[derive(Debug)]
pub struct AudioBlock<'view, 'samples, T> {
    planes: &'view [&'samples [T]],
    frames: usize,
}

impl<'view, 'samples, T: Sample> AudioBlock<'view, 'samples, T> {
    /// Build a view over per-channel slices. All channels have equal length.
    pub fn new(planes: &'view [&'samples [T]]) -> Self {
        let frames = planes.first().map_or(0, |p| p.len());
        debug_assert!(
            planes.iter().all(|p| p.len() == frames),
            "all channels must have equal length"
        );
        Self { planes, frames }
    }
    /// The channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.planes.len()
    }
    /// The frame count (equal across channels).
    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }
    /// Read channel `ch`.
    #[must_use]
    pub fn channel(&self, ch: usize) -> &[T] {
        self.planes[ch]
    }
}

/// A read-write multichannel view for the in-place main signal, output buffers,
/// and [`Source::pull`] targets. `&mut [T]` per channel.
/// The channel-table borrow is independent of the sample-slice borrow, so a
/// host may reborrow and reuse one table across consecutive calls.
///
/// [`Source::pull`]: crate::processor::Source::pull
#[derive(Debug)]
pub struct AudioBlockMut<'view, 'samples, T> {
    planes: &'view mut [&'samples mut [T]],
    frames: usize,
}

impl<'view, 'samples, T: Sample> AudioBlockMut<'view, 'samples, T> {
    /// Build a view over per-channel mutable slices. All channels have equal length.
    pub fn new(planes: &'view mut [&'samples mut [T]]) -> Self {
        let frames = planes.first().map_or(0, |p| p.len());
        debug_assert!(
            planes.iter().all(|p| p.len() == frames),
            "all channels must have equal length"
        );
        Self { planes, frames }
    }
    /// The channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.planes.len()
    }
    /// The frame count.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }
    /// Read channel `ch`.
    #[must_use]
    pub fn channel(&self, ch: usize) -> &[T] {
        self.planes[ch]
    }
    /// Read-modify-write channel `ch`.
    pub fn channel_mut(&mut self, ch: usize) -> &mut [T] {
        self.planes[ch]
    }
}

/// How a processor declares the main signal I/O shape. The host reads this with
/// [`Processor::io_mode`](crate::processor::Processor::io_mode) and supplies matching [`Io`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoMode {
    /// Read-modify-write over one buffer.
    InPlace,
    /// Write output without a main input signal.
    OutputOnly,
    /// Disjoint input and output buffers.
    Split,
}

/// The main signal in in-place, output-only, or split form.
#[derive(Debug)]
pub enum Io<'view, 'samples, T> {
    /// Read-modify-write over one buffer. Covers IIR, dynamics, and the host's
    /// `process_replacing`.
    InPlace(AudioBlockMut<'view, 'samples, T>),
    /// Write-only output for a generator or other source processor.
    OutputOnly(AudioBlockMut<'view, 'samples, T>),
    /// Disjoint input and output for work that reads original input while writing.
    Split {
        /// Read-only input.
        input: AudioBlock<'view, 'samples, T>,
        /// Write-only output, disjoint from `input`.
        output: AudioBlockMut<'view, 'samples, T>,
    },
}
