// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Airclay LLC

//! Shared contract-test harness.
//!
//! - [`Contract::run`] drives a fresh processor over a signal at a chosen host
//!   block size and splits absolute-stamped events per block.
//! - [`Contract::assert_block_size_invariant`] checks that the same input and
//!   events produce byte-identical output under each configured block split.
//! - [`Contract::assert_reset_equivalence`] checks that `reset` returns a
//!   processor to its prepared state.
//! - [`sine`], [`bits_eq`], and [`ev`] provide shared test utilities.
//!
//! These helpers use `f32` storage. Allocation checks live in integration tests
//! because they install a process-wide global allocator.
//!
//! # Public API
//!
//! - [`Buffers`] is the common planar `f32` buffer type.
//! - [`Contract`] drives processors, generators, and variable-rate processors
//!   through shared contract checks.
//! - [`ev`], [`sine`], [`observe_blocks`], and [`bits_eq`] are general test
//!   helpers.
//! - [`InfiniteTailKernel`] is a synthetic [`Tail::Infinite`] test double for
//!   flush-contract tests.
//! - [`registry`] is the supported built-in processor, meter, and variable-rate
//!   catalog for downstream and repository-wide contract suites.
//! - [`snapshot_hex`], [`manifest_hash`], [`assert_snapshot`], [`tone_stereo`],
//!   [`sweep_stereo`], and [`loud_stereo`] support byte-exact snapshot tests.

use crate::block::{AudioBlock, AudioBlockMut};
use crate::context::{ProcessContext, Produced, SubBlock, Tail};
use crate::error::DspError;
use crate::param::{ParamEvent, ParamId};
use crate::processor::{RingSource, Sample};
use crate::spec::ProcessSpec;
use crate::traits::{Kernel, Measurer, Processor, VariableRate};

pub mod registry;

// Repository-only generated cases are the narrow exception to the rule that
// supported public test infrastructure appears in rustdoc.
#[cfg(feature = "snapshot-support")]
#[doc(hidden)]
pub mod snapshot_cases;

/// A planar `f32` buffer: one `Vec` per channel, all equal length.
pub type Buffers = Vec<Vec<f32>>;

/// Construct an absolute-stamped parameter event.
///
/// `at` is the absolute sample index on the harness timeline. [`Contract::run`]
/// converts it to a per-block offset.
#[must_use]
pub fn ev(at: u32, param: ParamId, value: f64) -> ParamEvent {
    ParamEvent {
        offset: at,
        param,
        value,
    }
}

/// Generate a deterministic, channel-distinct multi-tone test signal.
///
/// Each channel uses a different frequency and amplitude. The tone is computed
/// through the vendored [`crate::dsp::math`], so it is identical on every
/// platform.
#[must_use]
pub fn sine(channels: usize, frames: usize) -> Buffers {
    (0..channels)
        .map(|ch| {
            let w = 0.011 + 0.006 * ch as f64;
            let amp = 0.7 - 0.08 * ch as f32;
            (0..frames)
                .map(|i| crate::dsp::math::sin(i as f64 * w) as f32 * amp)
                .collect()
        })
        .collect()
}

/// Feed `signal` to a [`Measurer`] in `block`-frame chunks.
///
/// Each chunk is a read-only [`AudioBlock`] over all channels. This is the meter
/// counterpart to [`Contract::run`].
pub fn observe_blocks<M: Measurer<f32>>(meter: &mut M, signal: &[Vec<f32>], block: usize) {
    assert!(block >= 1, "block size must be >= 1");
    let frames = signal.first().map_or(0, Vec::len);
    let mut pos = 0;
    while pos < frames {
        let blk = block.min(frames - pos);
        let planes: Vec<&[f32]> = signal.iter().map(|c| &c[pos..pos + blk]).collect();
        meter.observe(AudioBlock::new(&planes));
        pos += blk;
    }
}

/// Compare two planar buffers by raw bit pattern.
///
/// This distinguishes `-0.0`, `+0.0`, and every `NaN` payload.
#[must_use]
pub fn bits_eq(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// Test configuration used by contract helpers.
///
/// It contains the [`ProcessSpec`] and host block sizes used by invariance
/// checks. [`Contract::default`] uses 48 kHz stereo and block sizes around common
/// scheduler boundaries.
#[derive(Clone, Debug)]
pub struct Contract {
    /// The spec every processor in this contract is prepared for.
    pub spec: ProcessSpec,
    /// Host block sizes the invariance check replays the stream under.
    pub block_sizes: Vec<usize>,
}

impl Default for Contract {
    fn default() -> Self {
        Self {
            spec: ProcessSpec {
                sample_rate: 48_000,
                channels: 2,
                max_block: 8192,
                max_memory: None,
            },
            // Includes the control-rate grid and neighboring sizes.
            block_sizes: vec![1, 7, 32, 33, 64, 65, 128, 257, 999],
        }
    }
}

impl Contract {
    /// Run `input` through a freshly built and prepared processor.
    ///
    /// `events` are stamped on the whole-stream timeline. Each block receives
    /// only the events that fall in it, converted to block-relative offsets.
    /// Returns the in-place output.
    ///
    /// # Panics
    /// If `prepare` fails, or `block` is zero.
    pub fn run<P: Processor<f32>>(
        &self,
        make: impl Fn() -> P,
        input: &[Vec<f32>],
        events: &[ParamEvent],
        block: usize,
    ) -> Buffers {
        assert!(block >= 1, "block size must be >= 1");
        let mut proc = make();
        proc.prepare(self.spec).expect("prepare");
        self.run_reusing(&mut proc, input, events, block)
    }

    /// Drive an already-prepared processor without preparing it again.
    ///
    /// This is used for tests that need a reused instance.
    ///
    /// # Panics
    /// If `block` is zero.
    pub fn run_reusing<P: Processor<f32> + ?Sized>(
        &self,
        proc: &mut P,
        input: &[Vec<f32>],
        events: &[ParamEvent],
        block: usize,
    ) -> Buffers {
        assert!(block >= 1, "block size must be >= 1");
        let frames = input.first().map_or(0, Vec::len);
        let mut buf: Buffers = input.to_vec();
        let mut pos = 0usize;
        while pos < frames {
            let blk = block.min(frames - pos);
            let (lo, hi) = (pos, pos + blk);
            let evs: Vec<ParamEvent> = events
                .iter()
                .filter(|e| {
                    let o = e.offset as usize;
                    o >= lo && o < hi
                })
                .map(|e| ParamEvent {
                    offset: e.offset - lo as u32,
                    param: e.param,
                    value: e.value,
                })
                .collect();
            let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(|c| &mut c[lo..hi]).collect();
            let mut ctx = ProcessContext::in_place(&mut planes, pos as u64).with_events(&evs);
            proc.process(&mut ctx);
            pos = hi;
        }
        buf
    }

    /// Run a freshly built processor with one read-only sidechain bus.
    ///
    /// `sidechain` must have the same frame count as `input`. Returns the
    /// in-place main output.
    ///
    /// # Panics
    /// If `prepare` fails, or `block` is zero.
    pub fn run_with_sidechain<P: Processor<f32>>(
        &self,
        make: impl Fn() -> P,
        input: &[Vec<f32>],
        sidechain: &[Vec<f32>],
        events: &[ParamEvent],
        block: usize,
    ) -> Buffers {
        let mut proc = make();
        proc.prepare(self.spec).expect("prepare");
        self.run_with_sidechain_reusing(&mut proc, input, sidechain, events, block)
    }

    /// Drive an already-prepared sidechain processor without preparing it again.
    ///
    /// # Panics
    /// If `block` is zero.
    pub fn run_with_sidechain_reusing<P: Processor<f32> + ?Sized>(
        &self,
        proc: &mut P,
        input: &[Vec<f32>],
        sidechain: &[Vec<f32>],
        events: &[ParamEvent],
        block: usize,
    ) -> Buffers {
        assert!(block >= 1, "block size must be >= 1");
        let frames = input.first().map_or(0, Vec::len);
        let mut buf: Buffers = input.to_vec();
        let mut pos = 0usize;
        while pos < frames {
            let blk = block.min(frames - pos);
            let (lo, hi) = (pos, pos + blk);
            let evs: Vec<ParamEvent> = events
                .iter()
                .filter(|e| {
                    let o = e.offset as usize;
                    o >= lo && o < hi
                })
                .map(|e| ParamEvent {
                    offset: e.offset - lo as u32,
                    param: e.param,
                    value: e.value,
                })
                .collect();
            let mut planes: Vec<&mut [f32]> = buf.iter_mut().map(|c| &mut c[lo..hi]).collect();
            let sc_planes: Vec<&[f32]> = sidechain.iter().map(|c| &c[lo..hi]).collect();
            let sc = [AudioBlock::new(&sc_planes)];
            let mut ctx = ProcessContext::in_place(&mut planes, pos as u64)
                .with_sidechains(&sc)
                .with_events(&evs);
            proc.process(&mut ctx);
            pos = hi;
        }
        buf
    }

    /// Run a freshly built processor with split input and output buffers.
    ///
    /// This is for processors that declare [`IoMode::Split`](crate::processor::IoMode::Split).
    /// Returns the output.
    ///
    /// # Panics
    /// If `prepare` fails, or `block` is zero.
    pub fn run_split<P: Processor<f32>>(
        &self,
        make: impl Fn() -> P,
        input: &[Vec<f32>],
        block: usize,
    ) -> Buffers {
        let mut proc = make();
        proc.prepare(self.spec).expect("prepare");
        self.run_split_reusing(&mut proc, input, block)
    }

    /// Drive an already-prepared split-I/O processor without preparing it again.
    ///
    /// # Panics
    /// If `block` is zero.
    pub fn run_split_reusing<P: Processor<f32> + ?Sized>(
        &self,
        proc: &mut P,
        input: &[Vec<f32>],
        block: usize,
    ) -> Buffers {
        assert!(block >= 1, "block size must be >= 1");
        let frames = input.first().map_or(0, Vec::len);
        let mut out_buf: Buffers = vec![vec![0.0f32; frames]; self.spec.channels];
        let mut pos = 0usize;
        while pos < frames {
            let blk = block.min(frames - pos);
            let (lo, hi) = (pos, pos + blk);
            let in_planes: Vec<&[f32]> = input.iter().map(|c| &c[lo..hi]).collect();
            let mut out_planes: Vec<&mut [f32]> =
                out_buf.iter_mut().map(|c| &mut c[lo..hi]).collect();
            let mut ctx = ProcessContext::split(&in_planes, &mut out_planes, pos as u64);
            proc.process(&mut ctx);
            pos = hi;
        }
        out_buf
    }

    /// Drive a freshly built and prepared source.
    ///
    /// A source produces output from parameters without reading main input;
    /// build it with [`Kernel::into_processor`]. Returns `frames` frames for the
    /// configured channel count.
    ///
    /// # Panics
    /// If `prepare` fails, or `block` is zero.
    pub fn generate<P: Processor<f32>>(
        &self,
        make: impl Fn() -> P,
        frames: usize,
        events: &[ParamEvent],
        block: usize,
    ) -> Buffers {
        let mut gen = make();
        gen.prepare(self.spec).expect("prepare");
        self.generate_reusing(&mut gen, frames, events, block)
    }

    /// Drive an already-prepared source without preparing it again.
    ///
    /// # Panics
    /// If `block` is zero.
    pub fn generate_reusing<P: Processor<f32> + ?Sized>(
        &self,
        gen: &mut P,
        frames: usize,
        events: &[ParamEvent],
        block: usize,
    ) -> Buffers {
        assert!(block >= 1, "block size must be >= 1");
        let mut output: Buffers = vec![vec![0.0f32; frames]; self.spec.channels];
        let mut pos = 0usize;
        while pos < frames {
            let blk = block.min(frames - pos);
            let (lo, hi) = (pos, pos + blk);
            let evs: Vec<ParamEvent> = events
                .iter()
                .filter(|e| {
                    let offset = e.offset as usize;
                    offset >= lo && offset < hi
                })
                .map(|e| ParamEvent {
                    offset: e.offset - lo as u32,
                    param: e.param,
                    value: e.value,
                })
                .collect();
            let mut planes: Vec<&mut [f32]> = output
                .iter_mut()
                .map(|channel| &mut channel[lo..hi])
                .collect();
            let mut ctx = ProcessContext::output_only(&mut planes, pos as u64).with_events(&evs);
            gen.process(&mut ctx);
            pos = hi;
        }
        output
    }

    /// Check source block-size invariance.
    ///
    /// The same `frames` and `events` must yield byte-identical output under each
    /// configured block split.
    ///
    /// # Panics
    /// If any split diverges, or `frames` exceeds `spec.max_block`.
    pub fn assert_generator_block_size_invariant<P: Processor<f32>>(
        &self,
        make: impl Fn() -> P,
        frames: usize,
        events: &[ParamEvent],
    ) {
        assert!(
            frames <= self.spec.max_block,
            "invariance reference needs frames ({frames}) <= max_block ({})",
            self.spec.max_block
        );
        let reference = self.generate(&make, frames, events, frames.max(1));
        for &block in &self.block_sizes {
            let out = self.generate(&make, frames, events, block);
            assert!(
                bits_eq(&out, &reference),
                "generator block size {block} diverged from the whole-block reference"
            );
        }
    }

    /// Check processor block-size invariance.
    ///
    /// The same input and events must yield byte-identical output under each
    /// configured block split. The reference is the whole stream processed as one
    /// block.
    ///
    /// # Panics
    /// If any block split diverges from the whole-block reference, or the signal
    /// is longer than `spec.max_block` (so the whole-block reference is itself a
    /// legal call).
    pub fn assert_block_size_invariant<P: Processor<f32>>(
        &self,
        make: impl Fn() -> P,
        input: &[Vec<f32>],
        events: &[ParamEvent],
    ) {
        let frames = input.first().map_or(0, Vec::len);
        assert!(
            frames <= self.spec.max_block,
            "invariance reference needs frames ({frames}) <= max_block ({})",
            self.spec.max_block
        );
        let reference = self.run(&make, input, events, frames.max(1));
        for &block in &self.block_sizes {
            let out = self.run(&make, input, events, block);
            assert!(
                bits_eq(&out, &reference),
                "block size {block} diverged from the whole-block reference"
            );
        }
    }

    /// Check that `reset` returns a processor to its prepared state.
    ///
    /// A fresh instance and a reused-then-reset instance must produce
    /// byte-identical output for the same input and events. The reused instance
    /// is first driven once to advance internal state.
    ///
    /// # Panics
    /// If the post-reset output differs from a fresh instance's.
    pub fn assert_reset_equivalence<P: Processor<f32>>(
        &self,
        make: impl Fn() -> P,
        input: &[Vec<f32>],
        events: &[ParamEvent],
    ) {
        let fresh = self.run(&make, input, events, 64);
        let mut proc = make();
        proc.prepare(self.spec).expect("prepare");
        // Advance internal state before reset.
        let _ = self.run_reusing(&mut proc, input, events, 50);
        proc.reset();
        let after = self.run_reusing(&mut proc, input, events, 64);
        assert!(
            bits_eq(&after, &fresh),
            "reset must reproduce a fresh instance bit for bit"
        );
    }

    /// Drive an already-prepared [`VariableRate`] to completion.
    ///
    /// Input is pulled from a fresh [`RingSource`]. `out_block` is the number of
    /// output frames offered per call. `cap` limits source frames per pull, with
    /// `0` or `usize::MAX` meaning unlimited. Returns the concatenated output.
    ///
    /// # Panics
    /// If `out_block` is zero, or processing yields no frames without `done` while
    /// input remains.
    pub fn stretch_reusing<V: VariableRate<f32> + ?Sized>(
        &self,
        v: &mut V,
        input: &[Vec<f32>],
        out_block: usize,
        cap: usize,
    ) -> Buffers {
        assert!(out_block >= 1, "out_block must be >= 1");
        let mut src = if cap == 0 || cap == usize::MAX {
            RingSource::new(input.to_vec())
        } else {
            RingSource::with_chunk_cap(input.to_vec(), cap)
        };
        let ch = self.spec.channels;
        let mut out: Buffers = vec![Vec::new(); ch];
        let mut stage: Buffers = vec![vec![0.0f32; out_block]; ch];
        loop {
            let produced = {
                let mut planes: Vec<&mut [f32]> = stage.iter_mut().map(Vec::as_mut_slice).collect();
                let mut blk = AudioBlockMut::new(&mut planes);
                v.process(&mut src, &mut blk)
            };
            for (chan, st) in out.iter_mut().zip(&stage) {
                chan.extend_from_slice(&st[..produced.frames]);
            }
            if produced.done {
                break;
            }
            assert!(
                produced.frames > 0 || src.remaining() > 0,
                "VariableRate yielded 0 frames without done and with no input left"
            );
        }
        out
    }

    /// Build, prepare, and drive a fresh [`VariableRate`] over `input`.
    ///
    /// # Panics
    /// If `prepare` fails (see [`stretch_reusing`](Self::stretch_reusing) for the
    /// rest).
    pub fn stretch<V: VariableRate<f32>>(
        &self,
        make: impl Fn() -> V,
        input: &[Vec<f32>],
        out_block: usize,
        cap: usize,
    ) -> Buffers {
        let mut v = make();
        v.prepare(self.spec).expect("prepare");
        self.stretch_reusing(&mut v, input, out_block, cap)
    }

    /// Check output-timeline block-size invariance for a [`VariableRate`].
    ///
    /// The same stretch must produce byte-identical output under each configured
    /// output block size. The reference uses a block large enough to deliver the
    /// full output.
    ///
    /// # Panics
    /// If any out-block's output differs from the reference.
    pub fn assert_stretch_block_size_invariant<V: VariableRate<f32>>(
        &self,
        make: impl Fn() -> V,
        input: &[Vec<f32>],
    ) {
        let n = input.first().map_or(0, Vec::len);
        // Use one block large enough to contain the full output.
        let reference = self.stretch(&make, input, 2 * n + 4 * W_REF, usize::MAX);
        for &block in &self.block_sizes {
            let out = self.stretch(&make, input, block, usize::MAX);
            assert!(
                bits_eq(&out, &reference),
                "stretch out-block {block} diverged from the whole-output reference"
            );
        }
    }
}

/// Slack for the whole-output reference block used by
/// [`Contract::assert_stretch_block_size_invariant`].
const W_REF: usize = 1024;

// ---------------------------------------------------------------------------
// Flush-contract test doubles.
// ---------------------------------------------------------------------------

/// A synthetic [`Tail::Infinite`] kernel for flush-contract tests.
///
/// `render` passes audio through untouched and copies the newest input sample
/// of channel 0 into a one-sample recursive state. `flush` fills every
/// requested frame with `state *= 0.999` and always reports `done == false`:
/// an infinite tail never completes, so its host must cap the drain by
/// bounding the frames it requests.
#[derive(Debug, Clone, Default)]
pub struct InfiniteTailKernel {
    state: f64,
}

impl InfiniteTailKernel {
    /// A kernel with silent recursive state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Sample> Kernel<T> for InfiniteTailKernel {
    type Params = crate::parameter::NoParams;

    fn prepare(&mut self, _spec: ProcessSpec) -> Result<(), DspError> {
        self.state = 0.0;
        Ok(())
    }

    fn reset(&mut self) {
        self.state = 0.0;
    }

    fn tail(&self) -> Tail {
        Tail::Infinite
    }

    fn render(&mut self, io: &mut SubBlock<'_, '_, '_, T>, _params: &crate::parameter::NoParams) {
        // Copy the newest input sample into the one-sample recursive state.
        if io.frames() > 0 && io.channels() > 0 {
            self.state = io.input(0)[io.frames() - 1].to_f64();
        }
    }

    fn flush(&mut self, out: &mut AudioBlockMut<'_, '_, T>) -> Produced {
        let want = out.frames();
        for i in 0..want {
            self.state *= 0.999;
            for ch in 0..out.channels() {
                out.channel_mut(ch)[i] = T::from_f64(self.state);
            }
        }
        // An infinite tail keeps running; only the host's cap ends the drain.
        Produced {
            frames: want,
            done: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot hashing.
// ---------------------------------------------------------------------------

/// FNV-1a-128 offset basis.
const FNV128_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
/// FNV-1a-128 prime.
const FNV128_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

/// Streaming FNV-1a-128 hasher used for snapshot regression detection.
#[derive(Debug)]
struct Fnv1a128 {
    state: u128,
}

impl Fnv1a128 {
    fn new() -> Self {
        Self {
            state: FNV128_OFFSET,
        }
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= u128::from(b);
            self.state = self.state.wrapping_mul(FNV128_PRIME);
        }
    }
    fn finish_hex(&self) -> String {
        format!("{:032x}", self.state)
    }
}

/// Hash a planar `f32` buffer as a lowercase 32-character hex string.
///
/// The hash excludes the `fnv1a128:` tag. The byte layout is
/// `f32-le-planar-v1`: a domain-separated, length-framed header
/// (`b"f32-le-planar-v1\0"`, channel count, and frame count as little-endian
/// `u32`), followed by each sample's `to_bits()` in little-endian
/// `[channel][frame]` order.
#[must_use]
pub fn snapshot_hex(out: &[Vec<f32>]) -> String {
    let channels = out.len() as u32;
    let frames = out.first().map_or(0, Vec::len) as u32;
    let mut h = Fnv1a128::new();
    h.write(b"f32-le-planar-v1\0");
    h.write(&channels.to_le_bytes());
    h.write(&frames.to_le_bytes());
    for ch in out {
        for &s in ch {
            h.write(&s.to_bits().to_le_bytes());
        }
    }
    h.finish_hex()
}

/// The committed snapshot manifest, baked in at compile time from the repo root.
const MANIFEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/snapshots.manifest"
));

/// Return the committed `<algo>:<hex>` hash for snapshot case `id`.
///
/// Returns `None` if the case is absent. Parses tab-separated `case` rows and
/// skips `#` header and `meta` rows.
#[must_use]
pub fn manifest_hash(id: &str) -> Option<String> {
    for line in MANIFEST.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut f = line.split('\t');
        if f.next() != Some("case") || f.next() != Some(id) {
            continue;
        }
        // Remove a trailing carriage return from the last field if present.
        return f.next_back().map(|h| h.trim_end().to_owned());
    }
    None
}

/// Assert that `out` matches the committed snapshot for case `id`.
///
/// # Panics
/// If the case is missing from the manifest, or the recomputed hash differs.
pub fn assert_snapshot(id: &str, out: &[Vec<f32>]) {
    let got = format!("fnv1a128:{}", snapshot_hex(out));
    let want = manifest_hash(id).unwrap_or_else(|| {
        panic!(
            "no committed snapshot for case `{id}`. run `cargo xtask gen-snapshots --reason \
             \"<why>\"`"
        )
    });
    assert_eq!(
        got, want,
        "snapshot mismatch for case `{id}` (output changed)"
    );
}

// ---------------------------------------------------------------------------
// Snapshot reference signals.
// ---------------------------------------------------------------------------
// Snapshot inputs use the vendored libm through `crate::dsp::math`.

/// Generate a deterministic two-channel tone below full scale.
#[must_use]
pub fn tone_stereo(frames: usize) -> Buffers {
    (0..2)
        .map(|ch| {
            let w = 0.011 + 0.006 * ch as f64;
            let amp = 0.7 - 0.08 * ch as f64;
            (0..frames)
                .map(|i| (crate::dsp::math::sin(i as f64 * w) * amp) as f32)
                .collect()
        })
        .collect()
}

/// Generate a deterministic two-channel linear chirp below full scale.
#[must_use]
pub fn sweep_stereo(frames: usize) -> Buffers {
    (0..2)
        .map(|ch| {
            let w0 = 0.010 + 0.003 * ch as f64;
            let k = 2.0e-5;
            (0..frames)
                .map(|i| {
                    let t = i as f64;
                    (crate::dsp::math::sin(w0 * t + 0.5 * k * t * t) * 0.6) as f32
                })
                .collect()
        })
        .collect()
}

/// Generate a deterministic two-channel tone driven above full scale.
#[must_use]
pub fn loud_stereo(frames: usize) -> Buffers {
    (0..2)
        .map(|ch| {
            let w = 0.020 + 0.005 * ch as f64;
            (0..frames)
                .map(|i| (crate::dsp::math::sin(i as f64 * w) * 1.8) as f32)
                .collect()
        })
        .collect()
}
