// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! [`RingSource`] implements [`Source`] for a
//! [`VariableRate`](crate::processor::VariableRate).
//!
//! A `VariableRate` pulls its input from a `Source` instead of receiving a block,
//! because its output timeline is independent of its input timeline. `RingSource`
//! streams a finite planar buffer in `pull` chunks and reports end of input with
//! `done`. The optional chunk cap exercises partial-pull and underrun behavior.

use crate::block::AudioBlockMut;
use crate::context::Produced;
use crate::processor::Sample;
use crate::traits::Source;

/// A [`Source`] backed by a finite planar buffer.
#[derive(Clone, Debug)]
pub struct RingSource<T> {
    /// `[ch][frame]`, all channels equal length.
    data: Vec<Vec<T>>,
    channels: usize,
    /// Frames per channel.
    len: usize,
    /// Next frame to emit.
    read: usize,
    /// Maximum frames returned per `pull`. `usize::MAX` means unlimited.
    chunk_cap: usize,
}

impl<T: Sample> RingSource<T> {
    /// Build from owned planar buffers.
    ///
    /// All channels must have equal length. This matches the block-view caller
    /// precondition, debug-asserted here. In release a violation is safe but
    /// unspecified. It may hit a bounds-check panic in `pull`, never UB.
    #[must_use]
    pub fn new(data: Vec<Vec<T>>) -> Self {
        let channels = data.len();
        let len = data.first().map_or(0, Vec::len);
        debug_assert!(
            data.iter().all(|c| c.len() == len),
            "all channels must have equal length"
        );
        Self {
            data,
            channels,
            len,
            read: 0,
            chunk_cap: usize::MAX,
        }
    }

    /// Build with a maximum number of frames per `pull`.
    ///
    /// `cap` is clamped to at least one frame.
    #[must_use]
    pub fn with_chunk_cap(data: Vec<Vec<T>>, cap: usize) -> Self {
        Self {
            chunk_cap: cap.max(1),
            ..Self::new(data)
        }
    }

    /// Frames not yet pulled.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.len - self.read
    }

    /// Rewind to the start.
    pub fn rewind(&mut self) {
        self.read = 0;
    }
}

impl<T: Sample> Source<T> for RingSource<T> {
    fn channels(&self) -> usize {
        self.channels
    }

    fn pull(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let avail = self.len - self.read;
        let n = out.frames().min(avail).min(self.chunk_cap);
        for ch in 0..self.channels {
            let src = &self.data[ch][self.read..self.read + n];
            out.channel_mut(ch)[..n].copy_from_slice(src);
        }
        self.read += n;
        // End of input is reported when the buffer is exhausted.
        Produced {
            frames: n,
            done: self.read >= self.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RingSource;
    use crate::block::AudioBlockMut;
    use crate::context::Produced;
    use crate::traits::Source;

    fn data() -> Vec<Vec<f32>> {
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
    }

    /// Pull up to `frames` into a fresh block and return what was produced and the
    /// buffer it landed in.
    fn pull_into(src: &mut RingSource<f32>, frames: usize) -> (Produced, Vec<Vec<f32>>) {
        let nch = Source::channels(src);
        let mut chans: Vec<Vec<f32>> = vec![vec![0.0f32; frames]; nch];
        let produced = {
            let mut planes: Vec<&mut [f32]> = chans.iter_mut().map(Vec::as_mut_slice).collect();
            let mut block = AudioBlockMut::new(&mut planes);
            src.pull(&mut block)
        };
        (produced, chans)
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "all channels must have equal length")]
    fn ragged_channels_are_a_debug_assert() {
        // The same caller precondition as the block views.
        let _ = RingSource::new(vec![vec![1.0f32, 2.0], vec![3.0f32]]);
    }

    #[test]
    fn reports_channels_and_remaining_then_advances() {
        let mut src = RingSource::new(data());
        assert_eq!(Source::channels(&src), 2);
        assert_eq!(src.remaining(), 4);
        let (p, out) = pull_into(&mut src, 2);
        assert_eq!(p.frames, 2);
        assert!(!p.done, "two of four left some remaining");
        assert_eq!(src.remaining(), 2, "pull advanced the read cursor by two");
        assert_eq!(out[0][0], 1.0);
        assert_eq!(out[0][1], 2.0);
        assert_eq!(out[1][0], 5.0);
    }

    #[test]
    fn over_request_pulls_all_and_reports_eof() {
        let mut src = RingSource::new(data());
        let (p, _) = pull_into(&mut src, 10);
        assert_eq!(p.frames, 4, "only four frames exist");
        assert!(p.done, "buffer exhausted => done");
        assert_eq!(src.remaining(), 0);
    }

    #[test]
    fn rewind_restores_the_cursor() {
        let mut src = RingSource::new(data());
        let _ = pull_into(&mut src, 4);
        assert_eq!(src.remaining(), 0);
        src.rewind();
        assert_eq!(
            src.remaining(),
            4,
            "rewind makes every frame available again"
        );
        let (p, out) = pull_into(&mut src, 1);
        assert_eq!(p.frames, 1);
        assert_eq!(out[0][0], 1.0, "the rewound read starts at the front");
    }

    #[test]
    fn chunk_cap_limits_each_pull_while_data_remains() {
        let mut src = RingSource::with_chunk_cap(data(), 2);
        let (p, _) = pull_into(&mut src, 10);
        assert_eq!(p.frames, 2, "capped to chunk_cap, not the four available");
        assert!(!p.done, "data remains, so not done");
        assert_eq!(src.remaining(), 2);
    }
}
