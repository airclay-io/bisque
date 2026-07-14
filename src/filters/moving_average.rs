// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Moving-average FIR filtering.

use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{
    AudioBlockMut, DspError, IoMode, Kernel, ProcessSpec, Produced, Sample, SubBlock, Tail,
};

// ---------------------------------------------------------------------------
// FIR moving average (Split I/O)
// ---------------------------------------------------------------------------

/// A length-`taps` moving-average FIR lowpass.
///
/// It uses [`IoMode::Split`] to read original input while writing disjoint
/// output. [`Kernel::latency`] reports the whole-frame floor of the exact
/// `(taps - 1) / 2` group delay. Use [`group_delay_frames`](Self::group_delay_frames)
/// when fractional delay matters.
///
/// The tail is exactly `taps - 1` frames: the FIR response of the final
/// inputs. `flush` continues the convolution with silent input, so
/// process-plus-flush reconstructs the complete causal response. Odd tap counts
/// align to the reported integer latency; even tap counts retain half a frame
/// of fractional delay. New input starts a new drain.
///
/// A compensated running sum gives amortized constant work per sample. The sum
/// is recomputed from the ring at every wrap to bound numerical drift on a
/// deterministic, block-independent schedule.
#[derive(Debug, Clone)]
pub struct MovingAverage {
    taps: usize,
    rings: Vec<Vec<f64>>, // per channel, the last `taps` inputs
    sums: Vec<f64>,
    sum_corrections: Vec<f64>,
    ring_pos: usize,
    flushed: usize, // tail frames drained since the last process/reset
}

impl MovingAverage {
    /// A moving average over `taps` samples (at least 1).
    ///
    /// The tap count is stored as given: `prepare` rejects `taps == 0` with
    /// [`DspError::InvalidParam`] instead of silently clamping it.
    #[must_use]
    pub fn new(taps: usize) -> Self {
        Self {
            taps,
            rings: Vec::new(),
            sums: Vec::new(),
            sum_corrections: Vec::new(),
            ring_pos: 0,
            flushed: 0,
        }
    }

    /// Exact linear-phase group delay in frames.
    #[must_use]
    pub fn group_delay_frames(&self) -> f64 {
        self.taps.saturating_sub(1) as f64 * 0.5
    }
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    if sum.abs() >= value.abs() {
        *correction += (*sum - next) + value;
    } else {
        *correction += (value - next) + *sum;
    }
    *sum = next;
}

impl<T: Sample> Kernel<T> for MovingAverage {
    type Params = crate::parameter::NoParams;

    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        if self.taps == 0 {
            return Err(DspError::InvalidParam(
                "moving average taps must be at least 1",
            ));
        }
        MemoryLayout::new()
            .repeated_array::<f64>(spec.channels, self.taps)
            .array::<f64>(spec.channels)
            .array::<f64>(spec.channels)
            .preflight(spec.max_memory)?;
        self.rings = vec![vec![0.0; self.taps]; spec.channels];
        self.sums = vec![0.0; spec.channels];
        self.sum_corrections = vec![0.0; spec.channels];
        self.ring_pos = 0;
        self.flushed = 0;
        Ok(())
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            ring.fill(0.0);
        }
        self.sums.fill(0.0);
        self.sum_corrections.fill(0.0);
        self.ring_pos = 0;
        self.flushed = 0;
    }

    fn latency(&self) -> usize {
        // `saturating_sub` keeps the (invalid, prepare-rejected) zero-tap
        // configuration from underflowing if queried before prepare.
        self.group_delay_frames().floor() as usize
    }

    fn tail(&self) -> Tail {
        // The FIR response of the final `taps - 1` inputs outlives the body.
        Tail::Frames(self.taps.saturating_sub(1))
    }

    fn io_mode(&self) -> IoMode {
        IoMode::Split
    }

    fn memory_footprint(&self) -> usize {
        (self.rings.iter().map(Vec::len).sum::<usize>()
            + self.sums.len()
            + self.sum_corrections.len())
            * std::mem::size_of::<f64>()
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, _params: &crate::parameter::NoParams) {
        // New input starts a new drain: the drained-frame counter resets.
        self.flushed = 0;
        let n = self.taps;
        let scale = 1.0 / n as f64;
        let start = self.ring_pos;
        let mut end = start;
        for (ch, ((ring, sum), correction)) in self
            .rings
            .iter_mut()
            .zip(&mut self.sums)
            .zip(&mut self.sum_corrections)
            .enumerate()
        {
            // Read original input and write disjoint output together.
            let (inp, out) = io.split_channel(ch);
            let mut p = start;
            for (slot, &x) in out.iter_mut().zip(inp) {
                let x = finite_or_zero(x.to_f64());
                let old = ring[p];
                ring[p] = x;
                compensated_add(sum, correction, x - old);
                p = if p + 1 == n { 0 } else { p + 1 };
                if p == 0 {
                    *sum = ring.iter().sum();
                    *correction = 0.0;
                }
                *slot = T::from_f64((*sum + *correction) * scale);
            }
            end = p;
        }
        self.ring_pos = end;
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let n = self.taps;
        let bound = n - 1;
        let want = out.frames().min(bound.saturating_sub(self.flushed));
        let scale = 1.0 / n as f64;
        let start = self.ring_pos;
        let mut end = start;
        for (ch, ((ring, sum), correction)) in self
            .rings
            .iter_mut()
            .zip(&mut self.sums)
            .zip(&mut self.sum_corrections)
            .enumerate()
        {
            let buf = out.channel_mut(ch);
            let mut p = start;
            for slot in buf.iter_mut().take(want) {
                // The convolution continues with silent input.
                let old = ring[p];
                ring[p] = 0.0;
                compensated_add(sum, correction, -old);
                p = if p + 1 == n { 0 } else { p + 1 };
                if p == 0 {
                    *sum = ring.iter().sum();
                    *correction = 0.0;
                }
                *slot = T::from_f64((*sum + *correction) * scale);
            }
            end = p;
        }
        self.ring_pos = end;
        self.flushed += want;
        Produced {
            frames: want,
            done: self.flushed >= bound,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{compensated_add, MovingAverage};
    use crate::parameter::NoParams;
    use crate::processor::{AudioBlock, AudioBlockMut, Io, Kernel, ProcessSpec, SubBlock};

    const LARGE: f64 = 1.0e16;

    fn prepared(taps: usize) -> MovingAverage {
        let mut filter = MovingAverage::new(taps);
        Kernel::<f64>::prepare(
            &mut filter,
            ProcessSpec {
                sample_rate: 48_000,
                channels: 1,
                max_block: 8,
                max_memory: None,
            },
        )
        .expect("prepare");
        filter
    }

    fn render_one(filter: &mut MovingAverage, sample: f64) -> f64 {
        let input_samples = [sample];
        let input_planes = [input_samples.as_slice()];
        let input = AudioBlock::new(&input_planes);
        let mut output_samples = [0.0];
        {
            let mut output_planes = [output_samples.as_mut_slice()];
            let output = AudioBlockMut::new(&mut output_planes);
            let mut io = Io::Split { input, output };
            let mut sub_block = SubBlock {
                io: &mut io,
                sc: &[],
                start: 0,
                len: 1,
            };
            Kernel::<f64>::render(filter, &mut sub_block, &NoParams);
        }
        output_samples[0]
    }

    fn flush_one(filter: &mut MovingAverage) -> (f64, crate::processor::Produced) {
        let mut output_samples = [0.0];
        let produced;
        {
            let mut output_planes = [output_samples.as_mut_slice()];
            let mut output = AudioBlockMut::new(&mut output_planes);
            produced = Kernel::<f64>::flush(filter, &mut output);
        }
        (output_samples[0], produced)
    }

    fn filter_with_compensated_history() -> (MovingAverage, f64) {
        let mut filter = prepared(16);
        let _ = render_one(&mut filter, LARGE);
        let mut output = 0.0;
        for _ in 0..12 {
            output = render_one(&mut filter, 1.0);
        }
        (filter, output)
    }

    #[test]
    fn compensated_add_preserves_low_order_terms_in_both_branches() {
        let mut sum = LARGE;
        let mut correction = 2.0;
        compensated_add(&mut sum, &mut correction, 1.0);
        assert_eq!(sum, LARGE);
        assert_eq!(correction, 3.0);

        let mut sum = 1.0;
        let mut correction = 2.0;
        compensated_add(&mut sum, &mut correction, LARGE);
        assert_eq!(sum, LARGE);
        assert_eq!(correction, 3.0);
    }

    #[test]
    fn render_uses_compensation_between_ring_recomputations() {
        let (filter, output) = filter_with_compensated_history();

        assert_eq!(output, (LARGE + 12.0) / 16.0);
        assert_eq!(filter.ring_pos, 13);
        assert_eq!(filter.sum_corrections[0], 12.0);
    }

    #[test]
    fn flush_uses_compensation_between_ring_recomputations() {
        let (mut filter, _) = filter_with_compensated_history();

        let (sample, produced) = flush_one(&mut filter);
        assert_eq!(sample, (LARGE + 12.0) / 16.0);
        assert_eq!(produced.frames, 1);
        assert!(!produced.done);
        assert_eq!(filter.ring_pos, 14);
        assert_eq!(filter.sum_corrections[0], 12.0);
    }

    #[test]
    fn memory_footprint_counts_rings_sums_and_corrections() {
        let filter = prepared(5);
        assert_eq!(
            Kernel::<f64>::memory_footprint(&filter),
            7 * std::mem::size_of::<f64>()
        );
    }
}
