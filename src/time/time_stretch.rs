// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Overlap-add time stretching.

use std::f64::consts::TAU;

use crate::dsp::math;
use crate::dsp::memory::MemoryLayout;
use crate::dsp::sanitize::finite_or_zero;
use crate::processor::{
    AudioBlockMut, DspError, ProcessSpec, Produced, Sample, Source, VariableRate,
};

// ---------------------------------------------------------------------------
// TimeStretch overlap-add implementation.
// ---------------------------------------------------------------------------

/// Hann window length.
const W: usize = 1024;
/// Synthesis hop.
///
/// At 50% overlap, the periodic Hann is COLA and its overlapped copies sum to
/// 1, so a constant signal reconstructs exactly.
const HS: usize = W / 2;
/// Supported channel count for the allocation-free stack plane table.
const MAX_CH: usize = 16;
/// Stretch range. `Ha = round(HS/stretch)` stays in `[HS/2, W]`.
const RATIO_MIN: f64 = 0.5;
const RATIO_MAX: f64 = 2.0;
// HS divides W and the overlap is one hop.
const _: () = assert!(W % HS == 0 && W == 2 * HS);

/// Construction settings for [`TimeStretch`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct TimeStretchSettings {
    /// Requested output duration divided by input duration.
    pub stretch: f64,
}

impl Default for TimeStretchSettings {
    fn default() -> Self {
        Self { stretch: 1.0 }
    }
}

impl TimeStretchSettings {
    /// Default time-stretch settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the requested output-duration divided by input-duration ratio.
    #[must_use]
    pub fn stretch(mut self, stretch: f64) -> Self {
        self.stretch = stretch;
        self
    }
}

/// A deterministic overlap-add time-stretcher.
///
/// It pulls from a [`Source`] and writes a different number of output frames, so
/// it implements [`VariableRate`], not [`crate::processor::Processor`]. It analyzes
/// overlapping `W`-frame windows at hop `Ha = round(HS/stretch)`, applies a
/// periodic Hann window, and overlap-adds them at synthesis hop `HS = W / 2`.
///
/// This is plain overlap-add, not a phase vocoder. It does not preserve spectral
/// phase coherence or detect transients. Non-unity ratios can smear attacks and
/// move sustained tones among grain-rate sidebands, which can sound phasey or
/// pitch-unstable.
///
/// The requested stretch ratio is fixed at construction in `[0.5, 2.0]` and is
/// not automatable ([`VariableRate`] is a pull/produce contract with no
/// parameter events).
///
/// Latency is `0` on the output timeline. The pull model absorbs the analysis
/// window internally. Boundary overlap normalization and constant edge
/// extension avoid a startup or ending taper. At unity stretch, every finite
/// input is reproduced at the same length with no leading shift. At other
/// ratios, output length is the input length multiplied by the effective ratio
/// and rounded to the nearest frame.
///
/// One to 16 channels are supported. This fixed limit keeps the source plane
/// table on the stack, so processing does not allocate.
///
/// `T` is the storage type used by pull scratch buffers. Audio state is `f64`.
#[derive(Debug, Clone)]
pub struct TimeStretch<T = f32> {
    // Configuration.
    stretch: f64,
    channels: usize,
    ha: usize,        // analysis hop = round(HS/stretch).clamp(1, W)
    window: Vec<f64>, // [W] periodic Hann
    prepared: Option<PreparedVariableRateContract>,

    // Input staging.
    in_buf: Vec<Vec<f64>>, // [ch][W]
    in_valid: Vec<u8>,     // [W], one for source or edge-extended input
    in_fill: usize,        // staged frames including boundary padding (0..=W)
    input_pulled: u64,     // total real input frames pulled
    analysis_start: i128,  // current analysis-frame start on the input timeline
    last_input: Vec<f64>,  // [ch], finite sample used for ending extension

    // OLA accumulator.
    acc: Vec<Vec<f64>>, // [ch][W]
    norm: Vec<f64>,     // [W], accumulated valid-window weights

    // Output FIFO.
    out_fifo: Vec<Vec<f64>>, // [ch][2W] ring
    fifo_head: usize,
    fifo_len: usize, // bounded by HS (we drain before producing)
    output_emitted: u64,
    startup_drop: usize,
    target_output: Option<u64>,

    // Pull scratch.
    scratch: Vec<Vec<T>>, // [ch][W]

    // End-of-input state.
    input_eof: bool,
    tail_flushed: bool,
}

#[derive(Clone, Copy, Debug)]
struct PreparedVariableRateContract {
    channels: usize,
}

impl<T: Sample> TimeStretch<T> {
    /// Maximum supported channel count.
    pub const MAX_CHANNELS: usize = MAX_CH;

    /// A time-stretcher configured from `settings`. `prepare` validates
    /// `stretch` in `[0.5, 2.0]`.
    #[must_use]
    pub fn with_settings(settings: TimeStretchSettings) -> Self {
        Self {
            stretch: settings.stretch,
            channels: 0,
            ha: HS,
            window: Vec::new(),
            prepared: None,
            in_buf: Vec::new(),
            in_valid: Vec::new(),
            in_fill: 0,
            input_pulled: 0,
            analysis_start: -(HS as i128),
            last_input: Vec::new(),
            acc: Vec::new(),
            norm: Vec::new(),
            out_fifo: Vec::new(),
            fifo_head: 0,
            fifo_len: 0,
            output_emitted: 0,
            startup_drop: HS,
            target_output: None,
            scratch: Vec::new(),
            input_eof: false,
            tail_flushed: false,
        }
    }

    /// A unity 1.0x stretch.
    #[must_use]
    pub fn new() -> Self {
        Self::with_settings(TimeStretchSettings::default())
    }

    /// Requested output-duration divided by input-duration ratio.
    #[must_use]
    pub fn stretch(&self) -> f64 {
        self.stretch
    }

    /// Ratio produced by the rounded analysis hop after successful preparation.
    #[must_use]
    pub fn effective_stretch(&self) -> Option<f64> {
        self.prepared.map(|_| HS as f64 / self.ha as f64)
    }
}

impl Default for TimeStretch<f32> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sample> TimeStretch<T> {
    fn rounded_output_frames(input_frames: u64, analysis_hop: usize) -> u64 {
        let scaled = u128::from(input_frames) * HS as u128;
        let rounded = (scaled + (analysis_hop / 2) as u128) / analysis_hop as u128;
        rounded.min(u128::from(u64::MAX)) as u64
    }

    fn finish_input(&mut self) {
        self.input_eof = true;
        let target = Self::rounded_output_frames(self.input_pulled, self.ha);
        debug_assert!(
            self.output_emitted <= target,
            "TimeStretch emitted more frames than its final duration"
        );
        let queued_limit =
            usize::try_from(target.saturating_sub(self.output_emitted)).unwrap_or(usize::MAX);
        self.fifo_len = self.fifo_len.min(queued_limit);
        self.target_output = Some(target);
    }

    /// Pull up to one window into `in_buf` and return appended frame count.
    fn refill(&mut self, input: &mut dyn Source<T>) -> usize {
        let want = W - self.in_fill;
        if want == 0 {
            return 0;
        }
        let mut planes: [&mut [T]; MAX_CH] = std::array::from_fn(|_| &mut [] as &mut [T]);
        let mut rest: &mut [Vec<T>] = &mut self.scratch[..self.channels];
        let mut i = 0;
        while let Some((head, tail)) = rest.split_first_mut() {
            planes[i] = &mut head[..want];
            rest = tail;
            i += 1;
        }
        let produced = {
            let mut block = AudioBlockMut::new(&mut planes[..self.channels]);
            input.pull(&mut block)
        };
        debug_assert!(
            produced.frames <= want,
            "Source returned more frames than requested"
        );
        let got = produced.frames.min(want);
        let input_was_empty = self.input_pulled == 0;
        for ch in 0..self.channels {
            if input_was_empty {
                if let Some(first) = self.scratch[ch][..got].first() {
                    let first = finite_or_zero(first.to_f64());
                    self.in_buf[ch][..self.in_fill].fill(first);
                }
            }
            for j in 0..got {
                self.in_buf[ch][self.in_fill + j] = finite_or_zero(self.scratch[ch][j].to_f64());
            }
            if let Some(last) = self.in_buf[ch][self.in_fill..self.in_fill + got].last() {
                self.last_input[ch] = *last;
            }
        }
        if input_was_empty && !self.scratch[0][..got].is_empty() {
            self.in_valid[..self.in_fill].fill(1);
        }
        self.in_valid[self.in_fill..self.in_fill + got].fill(1);
        self.in_fill += got;
        self.input_pulled = self.input_pulled.saturating_add(got as u64);
        if produced.done {
            self.finish_input();
        }
        got
    }

    /// Move finalized accumulator frames into the output FIFO. Initial padding
    /// and samples beyond the rounded output duration are discarded here.
    fn push_finalized(&mut self, frames: usize) {
        for k in 0..frames {
            if self.startup_drop > 0 {
                self.startup_drop -= 1;
                continue;
            }
            if self.target_output.is_some_and(|target| {
                self.output_emitted.saturating_add(self.fifo_len as u64) >= target
            }) {
                continue;
            }
            let w = (self.fifo_head + self.fifo_len) % (2 * W);
            let weight = self.norm[k];
            for ch in 0..self.channels {
                let sample = if weight > f64::EPSILON {
                    self.acc[ch][k] / weight
                } else {
                    0.0
                };
                self.out_fifo[ch][w] = finite_or_zero(sample);
            }
            self.fifo_len += 1;
        }
    }

    /// Window the current `W`-frame analysis frame, push the `HS` finalized output
    /// frames to the FIFO, shift the accumulator by `HS`, and slide the analysis
    /// window forward by `Ha`.
    fn produce_hop(&mut self) {
        for n in 0..W {
            if self.in_valid[n] != 0 {
                self.norm[n] += self.window[n];
            }
        }
        for ch in 0..self.channels {
            let win = &self.window;
            let x = &self.in_buf[ch];
            let acc = &mut self.acc[ch];
            for n in 0..W {
                acc[n] += win[n] * finite_or_zero(x[n]);
            }
        }
        self.push_finalized(HS);
        for ch in 0..self.channels {
            let acc = &mut self.acc[ch];
            acc.copy_within(HS..W, 0);
            for v in &mut acc[W - HS..W] {
                *v = 0.0;
            }
        }
        self.norm.copy_within(HS..W, 0);
        self.norm[W - HS..W].fill(0.0);
        for ch in 0..self.channels {
            self.in_buf[ch].copy_within(self.ha..W, 0);
            self.in_buf[ch][W - self.ha..W].fill(0.0);
        }
        self.in_valid.copy_within(self.ha..W, 0);
        self.in_valid[W - self.ha..W].fill(0);
        self.analysis_start += self.ha as i128;
        self.in_fill = self.in_fill.saturating_sub(self.ha);
    }

    /// Emit residual accumulator frames at end of input.
    fn flush_tail(&mut self) {
        if self.input_pulled == 0 {
            self.tail_flushed = true;
            return;
        }
        self.push_finalized(W - HS);
        self.tail_flushed = true;
    }

    /// Advance synthesis by one hop or final tail flush.
    fn make_more(&mut self, input: &mut dyn Source<T>) -> bool {
        while self.in_fill < W && !self.input_eof {
            if self.refill(input) == 0 {
                break;
            }
        }
        if self.in_fill == W {
            self.produce_hop();
            return true;
        }
        if self.input_eof {
            if self.input_pulled == 0 {
                self.analysis_start = 0;
                self.tail_flushed = true;
                return false;
            }
            if self.analysis_start < i128::from(self.input_pulled) {
                for ch in 0..self.channels {
                    self.in_buf[ch][self.in_fill..W].fill(self.last_input[ch]);
                }
                self.in_valid[self.in_fill..W].fill(1);
                self.in_fill = W;
                self.produce_hop();
                return true;
            }
            if !self.tail_flushed {
                self.flush_tail();
                return true;
            }
            return false; // fully drained
        }
        false // underrun, not EOF
    }

    /// True once input is exhausted, every hop is produced, the tail is flushed,
    /// and the FIFO is empty.
    fn fully_done(&self) -> bool {
        self.input_eof
            && self.analysis_start >= i128::from(self.input_pulled)
            && self.tail_flushed
            && self.fifo_len == 0
            && self
                .target_output
                .is_some_and(|target| self.output_emitted >= target)
    }
}

impl<T: Sample> VariableRate<T> for TimeStretch<T> {
    fn prepare(&mut self, spec: ProcessSpec) -> Result<(), DspError> {
        self.prepared = None;
        if spec.sample_rate == 0 {
            return Err(DspError::UnsupportedSpec("sample rate must be non-zero"));
        }
        let ch = spec.channels;
        if ch == 0 || ch > MAX_CH {
            return Err(DspError::UnsupportedSpec(
                "TimeStretch supports 1..=16 channels",
            ));
        }
        if !(RATIO_MIN..=RATIO_MAX).contains(&self.stretch) {
            return Err(DspError::InvalidParam("stretch must be in [0.5, 2.0]"));
        }
        let ha = ((HS as f64 / self.stretch).round() as usize).clamp(1, W);
        // Check the budget before allocating the Hann window, validity and
        // normalization state, plus the per-channel staging, accumulator, FIFO,
        // and pull scratch.
        MemoryLayout::new()
            .array::<f64>(2 * W) // Hann window and normalization accumulator
            .array::<u8>(W) // valid-input mask
            .array::<f64>(ch) // ending extension sample per channel
            .repeated_array::<f64>(ch, W) // input staging
            .repeated_array::<f64>(ch, W) // OLA accumulator
            .repeated_array::<f64>(ch, 2 * W) // output FIFO
            .repeated_array::<T>(ch, W) // pull scratch
            .preflight(spec.max_memory)?;
        let window = (0..W)
            .map(|n| 0.5 - 0.5 * math::cos(TAU * n as f64 / W as f64))
            .collect::<Vec<_>>();
        let in_valid = vec![0; W];
        let in_buf = vec![vec![0.0; W]; ch];
        let last_input = vec![0.0; ch];
        let acc = vec![vec![0.0; W]; ch];
        let norm = vec![0.0; W];
        let out_fifo = vec![vec![0.0; 2 * W]; ch];
        let scratch = vec![vec![T::from_f64(0.0); W]; ch];

        self.channels = ch;
        self.ha = ha;
        self.window = window;
        self.in_valid = in_valid;
        self.in_buf = in_buf;
        self.last_input = last_input;
        self.acc = acc;
        self.norm = norm;
        self.out_fifo = out_fifo;
        self.scratch = scratch;
        self.prepared = Some(PreparedVariableRateContract { channels: ch });
        self.reset();
        Ok(())
    }

    fn reset(&mut self) {
        for v in &mut self.in_buf {
            v.fill(0.0);
        }
        for v in &mut self.acc {
            v.fill(0.0);
        }
        self.norm.fill(0.0);
        for v in &mut self.out_fifo {
            v.fill(0.0);
        }
        self.in_valid.fill(0);
        self.last_input.fill(0.0);
        // Scratch is written before read.
        self.in_fill = if self.prepared.is_some() { HS } else { 0 };
        self.input_pulled = 0;
        self.analysis_start = -(HS as i128);
        self.fifo_head = 0;
        self.fifo_len = 0;
        self.output_emitted = 0;
        self.startup_drop = HS;
        self.target_output = None;
        self.input_eof = false;
        self.tail_flushed = false;
    }

    fn memory_footprint(&self) -> usize {
        // The Hann window, normalization state, input staging, OLA accumulator,
        // and output FIFO are `f64`; validity is `u8` and pull scratch is `T`.
        let f = std::mem::size_of::<f64>();
        self.window.len() * f
            + self.norm.len() * f
            + self.in_valid.len() * std::mem::size_of::<u8>()
            + self.last_input.len() * f
            + self.in_buf.iter().map(Vec::len).sum::<usize>() * f
            + self.acc.iter().map(Vec::len).sum::<usize>() * f
            + self.out_fifo.iter().map(Vec::len).sum::<usize>() * f
            + self.scratch.iter().map(Vec::len).sum::<usize>() * std::mem::size_of::<T>()
    }

    fn process(
        &mut self,
        input: &mut dyn Source<T>,
        out: &mut AudioBlockMut<'_, '_, T>,
    ) -> Produced {
        debug_assert!(
            self.prepared.is_some(),
            "TimeStretch::process requires a successful prepare"
        );
        if let Some(prepared) = self.prepared {
            debug_assert_eq!(
                input.channels(),
                prepared.channels,
                "TimeStretch source channels must match the prepared spec"
            );
            debug_assert_eq!(
                out.channels(),
                prepared.channels,
                "TimeStretch output channels must match the prepared spec"
            );
        }
        if self.ha == HS {
            let produced = input.pull(out);
            debug_assert!(
                produced.frames <= out.frames(),
                "Source returned more frames than requested"
            );
            let frames = produced.frames.min(out.frames());
            for ch in 0..out.channels() {
                for sample in &mut out.channel_mut(ch)[..frames] {
                    *sample = T::from_f64(finite_or_zero(sample.to_f64()));
                }
            }
            return Produced {
                frames,
                done: produced.done,
            };
        }
        let need = out.frames();
        let mut written = 0;
        // Deliver finalized frames before synthesizing another hop.
        while written < need {
            if self.fifo_len > 0 {
                let n = (need - written).min(self.fifo_len);
                for ch in 0..self.channels {
                    let dst = out.channel_mut(ch);
                    for f in 0..n {
                        let r = (self.fifo_head + f) % (2 * W);
                        dst[written + f] = T::from_f64(finite_or_zero(self.out_fifo[ch][r]));
                    }
                }
                self.fifo_head = (self.fifo_head + n) % (2 * W);
                self.fifo_len -= n;
                self.output_emitted = self.output_emitted.saturating_add(n as u64);
                written += n;
                continue;
            }
            if !self.make_more(input) {
                break;
            }
        }
        Produced {
            frames: written,
            done: self.fully_done(),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the `TimeStretch` end-of-input state machine.
    use super::*;
    use crate::processor::RingSource;

    /// A 48 kHz stereo spec with the channel count overridden.
    fn spec(channels: usize) -> ProcessSpec {
        ProcessSpec {
            sample_rate: 48_000,
            channels,
            max_block: 8192,
            max_memory: None,
        }
    }

    /// A freshly prepared stereo stretcher at unity ratio.
    fn prepared() -> TimeStretch<f32> {
        let mut v = TimeStretch::with_settings(TimeStretchSettings::default());
        VariableRate::<f32>::prepare(&mut v, spec(2)).expect("prepare");
        v
    }

    /// A `Source` that reports `frames = 0, done = false`.
    struct StarvingSource;
    impl Source<f32> for StarvingSource {
        fn channels(&self) -> usize {
            2
        }
        fn pull(&mut self, _out: &mut AudioBlockMut<'_, '_, f32>) -> Produced {
            Produced {
                frames: 0,
                done: false,
            }
        }
    }

    struct UnexpectedPull;
    impl Source<f32> for UnexpectedPull {
        fn channels(&self) -> usize {
            2
        }

        fn pull(&mut self, _out: &mut AudioBlockMut<'_, '_, f32>) -> Produced {
            panic!("source was pulled after it reported completion")
        }
    }

    /// `memory_footprint` equals the byte count derived from the allocation
    /// layout.
    #[test]
    fn footprint_is_the_exact_layout_byte_count() {
        let f = std::mem::size_of::<f64>();
        for ch in [1usize, 2, 3] {
            let mut v: TimeStretch<f32> = TimeStretch::new();
            VariableRate::<f32>::prepare(&mut v, spec(ch)).expect("prepare");
            // Window and normalization [W each], validity [W], plus per-channel
            // ending sample [1], in_buf [W], acc [W], out_fifo [2W], and
            // scratch [W].
            let expected = 2 * W * f
                + W * std::mem::size_of::<u8>()
                + ch * (1 + W + W + 2 * W) * f
                + ch * W * std::mem::size_of::<f32>();
            assert_eq!(
                VariableRate::<f32>::memory_footprint(&v),
                expected,
                "{ch}-channel layout byte count"
            );
        }
    }

    /// Latency is genuinely zero on the output timeline: at unity stretch the
    /// complete output reproduces the input at the same index, which the
    /// integration suite verifies sample by sample.
    #[test]
    fn latency_is_zero_on_the_output_timeline() {
        let v = prepared();
        assert_eq!(VariableRate::<f32>::latency(&v), 0);
    }

    #[test]
    fn flush_tail_on_empty_input_emits_nothing() {
        // Empty input emits no tail frames.
        let mut v = prepared();
        v.flush_tail();
        assert_eq!(v.fifo_len, 0, "empty input must flush no tail frames");
        assert!(v.tail_flushed, "the tail is marked flushed either way");
    }

    #[test]
    fn flush_tail_after_real_input_emits_one_overlap() {
        // Non-empty input emits the residual accumulator frames.
        let mut v = prepared();
        v.input_pulled = 4096;
        v.startup_drop = 0;
        v.flush_tail();
        assert_eq!(
            v.fifo_len,
            W - HS,
            "a non-empty tail flush emits one overlap of frames"
        );
    }

    #[test]
    fn make_more_breaks_on_a_starving_source() {
        // A zero-frame, non-EOF pull produces no progress.
        let mut v = prepared();
        let mut src = StarvingSource;
        assert!(
            !v.make_more(&mut src),
            "a starving non-EOF source makes no progress this call"
        );
        assert_eq!(v.in_fill, HS, "only the hidden boundary padding remains");
        assert_eq!(v.fifo_len, 0, "and nothing was produced");
    }

    #[test]
    fn make_more_produces_a_hop_once_a_window_is_full() {
        // A full analysis window produces one synthesis hop.
        let mut v = prepared();
        let input: Vec<Vec<f32>> = vec![vec![0.25f32; 2 * W]; 2];
        let mut src = RingSource::new(input);
        assert!(v.make_more(&mut src), "a full window must produce a hop");
        assert_eq!(v.fifo_len, 0, "the hidden startup hop is discarded");
        assert_eq!(v.startup_drop, 0, "startup padding is now fully consumed");
        assert!(
            v.make_more(&mut src),
            "the next full window produces output"
        );
        assert_eq!(
            v.fifo_len, HS,
            "one visible hop finalizes exactly HS frames"
        );
        assert!(!v.input_eof, "input still remains, so not at EOF");
    }

    #[test]
    fn make_more_flushes_the_tail_at_the_drain_boundary() {
        // At the drain boundary, make_more flushes the residual tail.
        let mut v = prepared();
        v.input_eof = true;
        v.input_pulled = 4096;
        v.analysis_start = 4096;
        v.in_fill = 0;
        v.startup_drop = 0;
        v.target_output = Some(4096);
        v.tail_flushed = false;
        let mut src = RingSource::new(vec![Vec::<f32>::new(); 2]);
        assert!(v.make_more(&mut src), "the tail flush still makes progress");
        assert!(
            v.tail_flushed,
            "at the drain boundary make_more flushes the tail (no extra hop)"
        );
        assert_eq!(
            v.fifo_len,
            W - HS,
            "the flush enqueues exactly one residual overlap"
        );
    }

    #[test]
    fn end_of_input_never_pulls_the_source_again() {
        let mut v = prepared();
        v.input_eof = true;
        v.input_pulled = 1;
        v.analysis_start = 0;
        let mut source = UnexpectedPull;
        assert!(v.make_more(&mut source));
    }

    #[test]
    fn finalized_weight_at_epsilon_is_treated_as_uncovered() {
        let mut v = prepared();
        v.startup_drop = 0;
        v.norm[0] = f64::EPSILON;
        v.acc[0][0] = f64::EPSILON;
        v.push_finalized(1);
        assert_eq!(v.fifo_len, 1);
        assert_eq!(v.out_fifo[0][v.fifo_head], 0.0);
    }

    #[test]
    fn completion_requires_every_drain_condition() {
        let mut v = prepared();
        v.input_eof = false;
        v.input_pulled = 0;
        v.analysis_start = 0;
        v.tail_flushed = true;
        v.fifo_len = 0;
        v.target_output = Some(0);
        v.output_emitted = 0;
        assert!(!v.fully_done());

        v.input_eof = true;
        v.analysis_start = -1;
        assert!(!v.fully_done());
        v.analysis_start = 0;
        v.tail_flushed = false;
        assert!(!v.fully_done());
        v.tail_flushed = true;
        v.fifo_len = 1;
        assert!(!v.fully_done());
        v.fifo_len = 0;
        v.target_output = Some(1);
        assert!(!v.fully_done());
        v.target_output = Some(0);
        assert!(v.fully_done());
    }

    #[test]
    fn failed_prepare_does_not_commit_new_geometry() {
        let mut v = prepared();
        let old_footprint = VariableRate::<f32>::memory_footprint(&v);
        let result = VariableRate::<f32>::prepare(
            &mut v,
            ProcessSpec {
                channels: 3,
                max_memory: Some(0),
                ..spec(2)
            },
        );
        assert!(matches!(result, Err(DspError::OverBudget { .. })));
        assert!(
            v.prepared.is_none(),
            "a failed prepare leaves no active contract"
        );
        assert_eq!(v.channels, 2, "failed geometry was not committed");
        assert_eq!(VariableRate::<f32>::memory_footprint(&v), old_footprint);
        assert_eq!(v.effective_stretch(), None);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TimeStretch::process requires a successful prepare")]
    fn process_before_prepare_panics() {
        let mut v = TimeStretch::<f32>::new();
        let mut src = RingSource::new(vec![vec![0.0; 1]; 2]);
        let mut storage = vec![vec![0.0f32; 1]; 2];
        let mut planes: Vec<&mut [f32]> = storage.iter_mut().map(Vec::as_mut_slice).collect();
        let mut out = AudioBlockMut::new(&mut planes);
        let _ = v.process(&mut src, &mut out);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TimeStretch source channels must match the prepared spec")]
    fn source_channel_mismatch_panics() {
        let mut v = prepared();
        let mut src = RingSource::new(vec![vec![0.0; 1]]);
        let mut storage = vec![vec![0.0f32; 1]; 2];
        let mut planes: Vec<&mut [f32]> = storage.iter_mut().map(Vec::as_mut_slice).collect();
        let mut out = AudioBlockMut::new(&mut planes);
        let _ = v.process(&mut src, &mut out);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TimeStretch output channels must match the prepared spec")]
    fn output_channel_mismatch_panics() {
        let mut v = prepared();
        let mut src = RingSource::new(vec![vec![0.0; 1]; 2]);
        let mut storage = [vec![0.0f32; 1]];
        let mut planes: Vec<&mut [f32]> = storage.iter_mut().map(Vec::as_mut_slice).collect();
        let mut out = AudioBlockMut::new(&mut planes);
        let _ = v.process(&mut src, &mut out);
    }
}
